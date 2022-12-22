# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F
import math
import warnings
from addict import Dict
from paddle.jit import to_static


from .seq_postprocess import DetPoseProcess
from .classwise_criterion import ClasswiseCriterion
from .attention_modules import DeformableDecoderLayer
from ..initializer import linear_init_, constant_, xavier_uniform_, normal_

# @to_static
def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1/x2)

def unflatten(x, old_axis, new):
    old_shape = list(x.shape)
    old_shape = old_shape[:old_axis] + new + old_shape[old_axis+1:]
    return x.reshape(old_shape)



class AbstractClassifier(nn.Layer):
    def __init__(self, kwargs):
        args = Dict(kwargs)
        super(AbstractClassifier, self).__init__()
        self.num_layers = args.num_layers
        if args.num_layers > 0:
            self.feature_linear = nn.LayerList([nn.Linear(args.hidden_dim, args.hidden_dim) for i in range(args.num_layers)])
            self.skip_and_init = args.skip_and_init
            if args.skip_and_init:
                # nn.init.constant_(self.feature_linear[-1].weight, 0.)
                # nn.init.constant_(self.feature_linear[-1].bias, 0.)
                nn.initializer.Constant(0)(self.feature_linear[-1].weight)
                nn.initializer.Constant(0)(self.feature_linear[-1].bias)
        else:
            self.feature_linear = None
        if True:
            self.bias = True
            # self.b = nn.Parameter(paddle.to_tensor(1))
            self.b = self.create_parameter(shape=(1,))
            # nn.initializer.Constant(1)( self.b )
        self.reset_parameters(args.init_prob)

    def reset_parameters(self, init_prob):
        if True:
            prior_prob = init_prob
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            # nn.init.constant_(self.b.data, bias_value)
            nn.initializer.Constant(bias_value)(self.b)

    def forward(self, x, class_vector=None, cls_idx=None):
        # x: bs,cs,(nobj,)d
        # class_vector: bs,cs,d
        if self.feature_linear is not None:
            skip = x
            for i in range(self.num_layers): # 中间 relu
                x = F.relu(self.feature_linear[i](x)) if i < self.num_layers - 1 else self.feature_linear[i](x)
            if self.skip_and_init:
                x = skip + x
        new_feat = x
        assert x.dim() == 3
        W = self.getClassifierWeight(class_vector, cls_idx) # W: csall * d # class_vector 缩放了

        sim = (x * W).sum(-1) # bs*cs*nobj
        if True:
            sim = sim + self.b
        return sim



class DictClassifier(AbstractClassifier):
    # could be changed to: 
    # output = paddle.einsum('ijk,zjk->ij', x, self.W)
    # or output = paddle.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, args):
        super(DictClassifier, self).__init__(args)
        self.scale = args.hidden_dim ** -0.5

    def getClassifierWeight(self, class_vector=None, cls_idx=None):
        # class_vector: bs, cs, d
        W = class_vector * self.scale # 为啥缩放呢, 实际上就是除以  `/ sqrt(hidden_dim)`
        return W




class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, dropout=0., proj=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim) if proj else nn.Identity()

    def forward(self, x, pre_kv=None, attn_mask=None):
        N, B, C = x.shape
        qkv = self.qkv(x).reshape([N, B, 3, self.num_heads, C // self.num_heads]).transpose([2, 1, 3, 0, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if pre_kv is not None:
            k = paddle.concat([pre_kv[0], k], axis=2)
            v = paddle.concat([pre_kv[1], v], axis=2)
        pre_kv = paddle.stack([k, v], axis=0)

        # k_T = k.transpose(-2, -1)
        k_T = k.transpose([0, 1, 3, 2])
        attn = (q @ k_T) * self.scale

        if attn_mask is not None:
            # attn.masked_fill_(attn_mask, float('-inf'))
            attn = masked_fill(attn, attn_mask, float('-inf'))

        # attn = attn.softmax(axis=-1)
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([2, 0, 1, 3]).reshape([N, B, C])
        x = self.proj(x)
        return x, pre_kv


def update_reference_points_xy(output_signals, reference_points, id_step):
    # reference_points: cs_all, nobj, 2
    # output_signals List( Tensor(cs_all, nobj) )
    if id_step < 2:
        new_reference_points = inverse_sigmoid(reference_points)
        new_reference_points[:, :, id_step] += output_signals[-1]
        # new_reference_points = new_reference_points.sigmoid()
        new_reference_points = F.sigmoid(new_reference_points)
        return new_reference_points
    else:
        return reference_points


class UnifiedSeqHead(DeformableDecoderLayer):
    def __init__(self, args):
        # required keyts:
        #   num_steps (int), pos_emb (bool), sg_previous_logits (bool), combine_method (str)
        #   task_category (str: filename), args.num_classes (int)
        #   LOSS, CLASSIFIER
        #   other args as for decoder layer
        super(UnifiedSeqHead, self).__init__(args)
        
        self.d_model = args.hidden_dim
        self.n_heads = args.nheads
        self.normalize_before = args.pre_norm
        
        if args.no_ffn:
            del self.ffn
            self.ffn = nn.Identity()
        if self.self_attn:
            del self.self_attn
            self.self_attn = Attention(self.d_model, self.n_heads, dropout=args.dropout, proj=args.self_attn_proj)

        # TODO: Number of classes
        # self.classifier = build_label_classifier(args.CLASSIFIER)
        self.classifier = DictClassifier(args.CLASSIFIER)
        self.num_steps = args.num_steps
        self.output_embeds = nn.LayerList([
            MLP(self.d_model, self.d_model, c_out, 1) for c_out in [1] * self.num_steps
        ])
        self.reset_parameters_as_first_head()

        # TODO: more general functions
        self.adjust_reference_points = update_reference_points_xy
        if args.pos_emb:
            self.pos_emb = nn.Embedding(self.num_steps, self.d_model)
            # trunc_normal_(self.pos_emb.weight, std=.02)
            nn.initializer.TruncatedNormal(std=.02)(self.pos_emb.weight)
        else:
            self.pos_emb = None

        # for post logits
        # self.post_process = build_sequence_postprocess(args)
        self.post_process = DetPoseProcess(args)
        self.sg_previous_logits = args.sg_previous_logits
        self.combine_method = args.combine_method

        # for loss
        self.criterion = ClasswiseCriterion(args.LOSS)
        
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

    def reset_parameters_as_first_head(self):
        for i in range(self.num_steps):
            # nn.initializer.Constant(0)(self.output_embeds[i].layers[-1].weight)
            # nn.initializer.Constant(0. if (i < 2 or i >= 4) else -2.0)(self.output_embeds[i].layers[-1].bias)
            constant_(self.output_embeds[i].layers[-1].weight)
            constant_(self.output_embeds[i].layers[-1].bias, 
                      0. if (i < 2 or i >= 4) else -2.0)

    def reset_parameters_as_refine_head(self):
        for i in range(self.num_steps):
            # nn.initializer.Constant(0)(self.output_embeds[i].layers[-1].weight)
            # nn.initializer.Constant(0)(self.output_embeds[i].layers[-1].bias)
            constant_(self.output_embeds[i].layers[-1].weight)
            constant_(self.output_embeds[i].layers[-1].bias)

    def self_attn_forward(self, tgt, query_pos, **kwargs):
        # q = k = self.with_pos_embed(tgt, query_pos_self)
        bs, l, c = tgt.shape
        tgt2, self.pre_kv = self.self_attn(tgt.reshape([1, bs*l, c]), pre_kv=self.pre_kv)
        return tgt2.reshape([bs, l, c])
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    # def cross_attn_forward(self, tgt, query_pos, reference_points, srcs, src_padding_masks, **kwargs):
    #     # tgt: bs_all, seq, c
    #     # src: bs, seq_src, c
    #     # reference_points: bs / bs_all, seq, lvl, 2 or 4 (len_pt)
    #     bs_all, seq, c = tgt.shape
    #     num_levels = reference_points.shape[-2]
    #     bs = srcs.shape[0]
    #     cs_batch = kwargs.pop("cs_batch", None)
    #     src_spatial_shapes = kwargs.pop("src_spatial_shapes")
    #     level_start_index = kwargs.pop("src_level_start_index")

    #     tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
    #                            reference_points,
    #                            srcs, src_spatial_shapes, level_start_index, src_padding_masks, cs_batch=cs_batch)
    #     return tgt2


    # def forward_post(self, tgt, query_pos, **kwargs):
    #     # self attention
    #     if self.self_attn:
    #         tgt2 = self.self_attn_forward(tgt, query_pos, **kwargs)
    #         tgt = tgt + self.dropout2(tgt2)
    #         tgt = self.norm2(tgt)

    #     tgt2 = self.cross_attn_forward(tgt, query_pos, **kwargs)
    #     tgt = tgt + self.dropout1(tgt2)
    #     tgt = self.norm1(tgt)

    #     # ffn
    #     tgt = self.ffn(tgt)

    #     return tgt
    

    # def forward_pre(self, tgt, query_pos, **kwargs):
    #     # self attention
    #     if self.self_attn:
    #         tgt2 = self.norm2(tgt)
    #         tgt2 = self.self_attn_forward(tgt2, query_pos, **kwargs)
    #         tgt = tgt + self.dropout2(tgt2)

    #     tgt2 = self.norm1(tgt)
    #     tgt2 = self.cross_attn_forward(tgt2, query_pos, **kwargs)
    #     tgt = tgt + self.dropout1(tgt2)

    #     # ffn
    #     tgt = self.ffn(tgt)

    #     return tgt
    
    
    # def super_super_forward(self, *args, **kwargs):
    #     if self.normalize_before:
    #         return self.forward_pre(*args, **kwargs)
    #     return self.forward_post(*args, **kwargs)
    
    # def super_forward(self, tgt, query_pos, reference_points, srcs, src_padding_masks, **kwargs):
    #     # reference_points: bs / bs_all, seq, 2 or 4
    #     src_valid_ratios = kwargs.pop("src_valid_ratios") # bs, level, 2
    #     if reference_points.shape[-1] == 4:
    #         src_valid_ratios = paddle.concat([src_valid_ratios, src_valid_ratios], axis=-1)
    #     # if the number of reference_points and number of src_valid_ratios not match.
    #     # Expand and repeat for them
    #     if src_valid_ratios.shape[0] != reference_points.shape[0]:
    #         repeat_times = (reference_points.shape[0] // src_valid_ratios.shape[0])
    #         src_valid_ratios = src_valid_ratios.repeat_interleave(repeat_times, axis=0)
    #     src_valid_ratios = src_valid_ratios[:, None] if reference_points.dim() == 3 else src_valid_ratios[:, None, None]
    #     reference_points_input = reference_points[..., None, :] * src_valid_ratios
    #     return self.super_super_forward(tgt, query_pos, 
    #                                     reference_points=reference_points_input, 
    #                                     srcs=srcs, src_padding_masks=src_padding_masks, **kwargs)

    def forward(self, feat, query_pos, reference_points, srcs, src_padding_masks, **kwargs):
        # feat: cs_all, nobj, c
        # srcs: bs, l, c
        # reference_points: cs_all, nobj, 2
        cs_all, nobj, c = feat.shape

        # output for scores
        previous_logits = kwargs.pop("previous_logits", None)
        class_vector = kwargs.pop("class_vector", None)
        bs_idx, cls_idx = kwargs.pop("bs_idx", None), kwargs.pop("cls_idx", None)
        if kwargs.pop("rearrange", False):
            num_steps, \
            cls_idx, \
            feat, \
            class_vector, \
            bs_idx, \
            kwargs["src_valid_ratios"] = self.post_process.taskCategory.arrangeBySteps(cls_idx, 
                                                                                       feat, 
                                                                                       class_vector, 
                                                                                       bs_idx, 
                                                                                       kwargs["src_valid_ratios"])
        else:
            num_steps = self.post_process.taskCategory.getNumSteps(cls_idx)
        output_classes = self.classifier(feat, 
                                         class_vector=class_vector.unsqueeze(1) if class_vector is not None else None)
        output_classes = self.postprocess_logits(output_classes, previous_logits, bs_idx, cls_idx)

        # prepare for sequence
        input_feat = feat
        output_signals = [] # a list of Tensor(cs_all, nobj)
        original_reference_points = reference_points
        self.pre_kv = None
        self.cross_attn.preprocess_value(srcs, src_padding_masks, bs_idx=bs_idx) # 操作 self.cross_attn.value
        for id_step, output_embed in enumerate(self.output_embeds):
            # forward the features, get output_features
            if self.pos_emb is not None:
                feat = feat + self.pos_emb.weight[id_step]
            forward_reference_points = reference_points.detach()
            output_feat = super().forward(feat, 
                                          query_pos,  # None
                                          forward_reference_points, 
                                          srcs, 
                                          src_padding_masks, 
                                          **kwargs)
            # output_feat = self.super_forward(feat, query_pos, forward_reference_points, srcs, src_padding_masks, **kwargs)
            output_signal = output_embed(output_feat).squeeze(-1)
            output_signals.append(output_signal)

            feat = self.generate_feat_for_next_step(output_feat, output_signal, reference_points, None,id_step)
            reference_points = self.adjust_reference_points(output_signals, reference_points, id_step)
            # TODO: make this more suitable for other tasks
            if (num_steps == id_step + 1).sum() > 0 and id_step < self.num_steps:
                count_needs = (num_steps > id_step + 1).sum()
                old_cs = feat.shape[0]
                feat = feat[:count_needs]
                reference_points = reference_points[:count_needs]
                # self.pre_kv = self.pre_kv.unflatten(1, (old_cs, nobj))[:, :count_needs].flatten(1,2)
                self.pre_kv = unflatten(self.pre_kv, 1, [old_cs, nobj])[:, :count_needs].flatten(1,2)
                self.cross_attn.value = self.cross_attn.value[:count_needs]
                kwargs["src_valid_ratios"] = kwargs["src_valid_ratios"][:count_needs]

        outputs = self.post_process(output_signals, output_classes, original_reference_points, bs_idx, cls_idx)
        # prepare targets
        targets = kwargs.pop("targets", None)
        if targets is not None and self.training:
            loss_dict = self.criterion(outputs, targets)
        else:
            assert not self.training, "Targets are required for training mode (unified_seq_head.py)"
            loss_dict = {}
        return outputs, loss_dict

    def generate_feat_for_next_step(self, output_feat, output_signal, reference_logits, boxes, id_step):
        # prepare inputs for the next input
        # output_feat:   bs*cs*nobj, 1, c
        # output_signal: bs*cs*nobj, 1, 1
        # reference_points: bs*cs*nobj, 1, 2
        # boxes: bs*cs*nobj, 1, 4
        feat = output_feat.clone().detach()
        return feat

    def postprocess_logits(self, outputs_logits, previous_logits, bs_idx, cls_idx):
        if previous_logits is not None:
            # 把 previous_logits 的信息拿出来
            previous_logits = previous_logits[bs_idx, cls_idx]
            previous_logits = previous_logits.unsqueeze(-1)
            if self.sg_previous_logits:
                previous_logits = previous_logits.detach()
        if self.combine_method =="none":
            return outputs_logits
        elif self.combine_method == "add":
            return F.sigmoid(outputs_logits) + F.sigmoid(previous_logits)
        elif self.combine_method == "multiple":
            return inverse_sigmoid(F.sigmoid(outputs_logits) * F.sigmoid(previous_logits))
        else:
            raise KeyError


class MLP(nn.Layer):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x) # 中间 relu, 最后输出
        return x
