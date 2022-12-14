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

# from util.misc import inverse_sigmoid
# from timm.models.layers import trunc_normal_
# from .classifiers import build_label_classifier
# from .seq_postprocess import build_sequence_postprocess
from .seq_postprocess import DetPoseProcess
# from ..transformer.attention_modules import DeformableDecoderLayer
# from models.ops.functions import MSDeformAttnFunction
from .classwise_criterion import ClasswiseCriterion
from .deformable_transformer import deformable_attention_core_func
from .attention_modules import DeformableDecoderLayer

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
        super().__init__()
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
            for i in range(self.num_layers):
                x = F.relu(self.feature_linear[i](x)) if i < self.num_layers - 1 else self.feature_linear[i](x)
            if self.skip_and_init:
                x = skip + x
        new_feat = x
        assert x.dim() == 3
        W = self.getClassifierWeight(class_vector, cls_idx) # W: csall*d

        sim = (x * W).sum(-1) # bs*cs*nobj
        if True:
            sim = sim + self.b
        return sim



class DictClassifier(AbstractClassifier):
    # could be changed to: 
    # output = paddle.einsum('ijk,zjk->ij', x, self.W)
    # or output = paddle.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, args):
        super().__init__(args)
        self.scale = args.hidden_dim ** -0.5

    def getClassifierWeight(self, class_vector=None, cls_idx=None):
        # class_vector: bs,cs,d
        W = class_vector * self.scale
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


# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     if activation == "prelu":
#         return F.prelu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



# class FFN(nn.Layer):

#     def __init__(self, d_model=256, d_ffn=1024, dropout=0., activation='relu', normalize_before=False):
#         super(FFN, self).__init__()
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation)
#         self.dropout2 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout3 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.normalize_before = normalize_before

#     def forward_post(self, src):
#         src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
#         src = src + self.dropout3(src2)
#         src = self.norm2(src)
#         return src

#     def forward_pre(self, src):
#         src2 = self.norm2(src)
#         src2 = self.linear2(self.dropout2(self.activation(self.linear1(src2))))
#         src = src + self.dropout3(src2)
#         return src

#     def forward(self, src):
#         if self.normalize_before:
#             return self.forward_pre(src)
#         return self.forward_post(src)






# def _is_power_of_2(n):
#     if (not isinstance(n, int)) or (n < 0):
#         raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
#     return (n & (n-1) == 0) and n != 0


# class MSDeformAttn(nn.Layer):
#     def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, no_value_proj=False):
#         """
#         Multi-Scale Deformable Attention Module
#         :param d_model      hidden dimension
#         :param n_levels     number of feature levels
#         :param n_heads      number of attention heads
#         :param n_points     number of sampling points per attention head per feature level
#         """
        
#         super(MSDeformAttn, self).__init__()
        
#         if d_model % n_heads != 0:
#             raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
#         _d_per_head = d_model // n_heads
#         # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
#         if not _is_power_of_2(_d_per_head):
#             warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
#                           "which is more efficient in our CUDA implementation.")

#         self.im2col_step = 64

#         self.d_model = d_model
#         self.n_levels = n_levels
#         self.n_heads = n_heads
#         self.n_points = n_points

#         self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
#         self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
#         self.no_value_proj = no_value_proj
#         self.value_proj = nn.Identity() if no_value_proj else nn.Linear(d_model, d_model)
#         self.output_proj = nn.Linear(d_model, d_model)
#         self.value = None


#         self.constant_ = nn.initializer.Constant(0)
#         self.xavier_uniform_ = nn.initializer.XavierNormal()
#         self.param_assign = lambda x,y : nn.initializer.Assign(x)(y)
#         self._reset_parameters()

#     def _reset_parameters(self):
#         self.constant_(self.sampling_offsets.weight)
#         thetas = paddle.arange(self.n_heads, dtype="float32") * (2.0 * math.pi / self.n_heads)
#         grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).reshape([self.n_heads, 1, 1, 2]).tile([1, self.n_levels, self.n_points, 1])
#         for i in range(self.n_points):
#             grid_init[:, :, i, :] *= i + 1
#         with paddle.no_grad():
#             # self.sampling_offsets.bias = nn.Parameter(grid_init.reshape([-1]))
#             self.param_assign(grid_init.reshape([-1]), self.sampling_offsets.bias)
#         self.constant_(self.attention_weights.weight)
#         self.constant_(self.attention_weights.bias)
#         if not self.no_value_proj:
#             self.xavier_uniform_(self.value_proj.weight)
#             self.constant_(self.value_proj.bias)
        
#         self.xavier_uniform_(self.output_proj.weight)
#         self.constant_(self.output_proj.bias)

#     def preprocess_value(self, input_flatten, input_padding_mask=None, cs_batch=None, bs_idx=None):
#         N, Len_in, _ = input_flatten.shape
#         value = self.value_proj(input_flatten)
#         if input_padding_mask is not None:
#             # value = value.masked_fill(input_padding_mask[..., None], float(0))
#             value = masked_fill(value, input_padding_mask[..., None], float(0))
#         self.value = value.reshape([N, Len_in, self.n_heads, self.d_model // self.n_heads])
#         if bs_idx is not None:
#             self.value = self.value[bs_idx]
#         elif cs_batch is not None:
#             self.value = paddle.concat([
#                 v.expand(cs ,-1, -1, -1) for cs, v in zip(cs_batch, self.value)
#             ]) # cs_all, *, *, *

#     def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, cs_batch=None):
#         """
#         :param query                       (N, Length_{query}, C)
#         :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
#                                         or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
#         :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
#         :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
#         :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
#         :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

#         :return output                     (N, Length_{query}, C)
#         """
#         N, Len_q, _ = query.shape
#         if self.value is None:
#             N, Len_in, _ = input_flatten.shape
#             assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

#             value = self.value_proj(input_flatten)
#             if input_padding_mask is not None:
#                 value = value.masked_fill(input_padding_mask[..., None], float(0))
#             value = value.reshape([N, Len_in, self.n_heads, self.d_model // self.n_heads])

#             if cs_batch is not None:
#                 value = paddle.concat([
#                     v.expand(cs ,-1, -1, -1) for cs, v in zip(cs_batch, value)
#                 ]) # cs_all, *, *, *
#                 N = value.shape[0]
#         else:
#             value = self.value
#             assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == value.shape[1]

#         sampling_offsets = self.sampling_offsets(query).reshape([N, Len_q, self.n_heads, self.n_levels, self.n_points, 2])
#         attention_weights = self.attention_weights(query).reshape([N, Len_q, self.n_heads, self.n_levels * self.n_points])
#         attention_weights = F.softmax(attention_weights, -1).reshape([N, Len_q, self.n_heads, self.n_levels, self.n_points])
        
#         # N, Len_q, n_heads, n_levels, n_points, 2
#         if reference_points.shape[-1] == 2:
#             offset_normalizer = paddle.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
#             sampling_locations = reference_points[:, :, None, :, None, :] \
#                                  + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
#         elif reference_points.shape[-1] == 4:
#             sampling_locations = reference_points[:, :, None, :, None, :2] \
#                                  + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
#         else:
#             raise ValueError(
#                 'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
#         # output = MSDeformAttnFunction.apply(
#         #     value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
#         output = deformable_attention_core_func(
#             value, input_spatial_shapes, sampling_locations, attention_weights)
#         output = self.output_proj(output)
#         return output






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
        
        
        # self.cross_attn = MSDeformAttn(self.d_model, 
        #                                args.n_levels, 
        #                                args.nheads, 
        #                                args.n_points, 
        #                                no_value_proj=args.cross_attn_no_value_proj)
        # # self.cross_attn = nn.MultiheadAttention(self.d_model, args.nheads, dropout=args.dropout)

        
        # # ffn
        # self.ffn = FFN(self.d_model, 
        #                args.dim_feedforward, 
        #                args.dropout, 
        #                args.activation, 
        #                normalize_before=self.normalize_before)

        if args.no_ffn:
            del self.ffn
            self.ffn = nn.Identity()
        # self.self_attn = True
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
            nn.initializer.Constant(0)(self.output_embeds[i].layers[-1].weight)
            nn.initializer.Constant(0. if (i < 2 or i >= 4) else -2.0)(self.output_embeds[i].layers[-1].bias)

    def reset_parameters_as_refine_head(self):
        for i in range(self.num_steps):
            nn.initializer.Constant(0)(self.output_embeds[i].layers[-1].weight)
            nn.initializer.Constant(0)(self.output_embeds[i].layers[-1].bias)

    def self_attn_forward(self, tgt, query_pos, **kwargs):
        # q = k = self.with_pos_embed(tgt, query_pos_self)
        bs, l, c = tgt.shape
        tgt2, self.pre_kv = self.self_attn(tgt.reshape([1, bs*l, c]), pre_kv=self.pre_kv)
        return tgt2.reshape([bs, l, c])
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def cross_attn_forward(self, tgt, query_pos, reference_points, srcs, src_padding_masks, **kwargs):
        # tgt: bs_all, seq, c
        # src: bs, seq_src, c
        # reference_points: bs / bs_all, seq, lvl, 2 or 4 (len_pt)
        bs_all, seq, c = tgt.shape
        num_levels = reference_points.shape[-2]
        bs = srcs.shape[0]
        cs_batch = kwargs.pop("cs_batch", None)
        src_spatial_shapes = kwargs.pop("src_spatial_shapes")
        level_start_index = kwargs.pop("src_level_start_index")

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               srcs, src_spatial_shapes, level_start_index, src_padding_masks, cs_batch=cs_batch)
        return tgt2


    def forward_post(self, tgt, query_pos, **kwargs):
        # self attention
        if self.self_attn:
            tgt2 = self.self_attn_forward(tgt, query_pos, **kwargs)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        tgt2 = self.cross_attn_forward(tgt, query_pos, **kwargs)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        return tgt
    

    def forward_pre(self, tgt, query_pos, **kwargs):
        # self attention
        if self.self_attn:
            tgt2 = self.norm2(tgt)
            tgt2 = self.self_attn_forward(tgt2, query_pos, **kwargs)
            tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm1(tgt)
        tgt2 = self.cross_attn_forward(tgt2, query_pos, **kwargs)
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt = self.ffn(tgt)

        return tgt
    
    
    def super_super_forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)
    
    def super_forward(self, tgt, query_pos, reference_points, srcs, src_padding_masks, **kwargs):
        # reference_points: bs / bs_all, seq, 2 or 4
        src_valid_ratios = kwargs.pop("src_valid_ratios") # bs, level, 2
        if reference_points.shape[-1] == 4:
            src_valid_ratios = paddle.concat([src_valid_ratios, src_valid_ratios], axis=-1)
        # if the number of reference_points and number of src_valid_ratios not match.
        # Expand and repeat for them
        if src_valid_ratios.shape[0] != reference_points.shape[0]:
            repeat_times = (reference_points.shape[0] // src_valid_ratios.shape[0])
            src_valid_ratios = src_valid_ratios.repeat_interleave(repeat_times, axis=0)
        src_valid_ratios = src_valid_ratios[:, None] if reference_points.dim() == 3 else src_valid_ratios[:, None, None]
        reference_points_input = reference_points[..., None, :] * src_valid_ratios
        return self.super_super_forward(tgt, query_pos, 
                                        reference_points=reference_points_input, 
                                        srcs=srcs, src_padding_masks=src_padding_masks, **kwargs)

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
            num_steps, cls_idx, feat, class_vector, bs_idx, kwargs["src_valid_ratios"] = self.post_process.taskCategory.arrangeBySteps(cls_idx, feat, class_vector, bs_idx, kwargs["src_valid_ratios"])
        else:
            num_steps = self.post_process.taskCategory.getNumSteps(cls_idx)
        output_classes = self.classifier(feat, class_vector=class_vector.unsqueeze(1) if class_vector is not None else None)
        output_classes = self.postprocess_logits(output_classes, previous_logits, bs_idx, cls_idx)

        # prepare for sequence
        input_feat = feat
        output_signals = [] # a list of Tensor(cs_all, nobj)
        original_reference_points = reference_points
        self.pre_kv = None
        self.cross_attn.preprocess_value(srcs, src_padding_masks, bs_idx=bs_idx)
        for id_step, output_embed in enumerate(self.output_embeds):
            # forward the features, get output_features
            if self.pos_emb is not None:
                feat = feat + self.pos_emb.weight[id_step]
            forward_reference_points = reference_points.detach()
            # output_feat = super().forward(feat, query_pos, forward_reference_points, srcs, src_padding_masks, **kwargs)
            output_feat = self.super_forward(feat, query_pos, forward_reference_points, srcs, src_padding_masks, **kwargs)
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
        if targets is not None:
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
            previous_logits = previous_logits[bs_idx, cls_idx]
            previous_logits = previous_logits.unsqueeze(-1)
            if self.sg_previous_logits:
                previous_logits = previous_logits.detach()
        if self.combine_method =="none":
            return outputs_logits
        elif self.combine_method == "add":
            return outputs_logits.sigmoid() + previous_logits.sigmoid()
        elif self.combine_method == "multiple":
            return inverse_sigmoid(outputs_logits.sigmoid() * previous_logits.sigmoid())
        else:
            raise KeyError


class MLP(nn.Layer):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
