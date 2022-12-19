# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np
from addict import Dict
import math
import copy

from ppdet.core.workspace import register, serializable
# from .attention_modules import MultiHeadDecoderLayer as TransformerDecoderLayer, _get_clones
# from ..predictors.classifiers import build_label_classifier
# from .class_criterion import ClassDecoderCriterion
from .deformable_transformer import DeformableTransformerDecoderLayer, DeformableTransformerDecoder
from .asl_losses import AsymmetricLoss, AsymmetricLossOptimized
from .attention_modules import MultiHeadDecoderLayer as TransformerDecoderLayer

def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


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


# class LinearClassifier(AbstractClassifier):
#     # could be changed to: 
#     # output = paddle.einsum('ijk,zjk->ij', x, self.W)
#     # or output = paddle.einsum('ijk,jk->ij', x, self.W[0])
#     def __init__(self, args):
#         super().__init__(args)

#         self.hidden_dim = args.hidden_dim
#         self.W = nn.Parameter(paddle.Tensor(self.hidden_dim))
#         stdv = 1. / math.sqrt(self.W.size(0))
#         self.W.data.uniform_(-stdv, stdv)

#     def getClassifierWeight(self, class_vector=None, cls_idx=None):
#         return self.W


class DictClassifier(AbstractClassifier):
    # could be changed to: 
    # output = paddle.einsum('ijk,zjk->ij', x, self.W)
    # or output = paddle.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, kwargs):
        super(DictClassifier, self).__init__(kwargs)
        self.scale = kwargs["hidden_dim"] ** -0.5

    def getClassifierWeight(self, class_vector=None, cls_idx=None):
        # class_vector: bs,cs,d
        W = class_vector * self.scale
        return W

# ==============================================================================================











def build_asymmetricloss(args):
    lossClass = AsymmetricLossOptimized if args.asl_optimized else AsymmetricLoss
    return lossClass(gamma_neg=args.asl_gamma_neg,
                     gamma_pos=args.asl_gamma_pos,
                     clip=args.asl_clip,
                     disable_paddle_grad_focal_loss=True)



class ClassDecoderCriterion(nn.Layer):
    def __init__(self, args):
        super().__init__()
        self.losses = args.losses
        self.loss_weights = {
            "asl": args.asl_loss_weight,
            "bce": args.asl_loss_weight,
        }
        self.asl_loss = build_asymmetricloss(args)
        self.loss_funcs = {
            "asl": lambda outputs, targets: self.asl_loss(outputs['cls_label_logits'], targets["multi_label_onehot"], targets["multi_label_weights"]),
            "bce": lambda outputs, targets: F.binary_cross_entropy_with_logits(outputs['cls_label_logits'], 
                                                                               targets["multi_label_onehot"], 
                                                                               targets["multi_label_weights"], 
                                                                               reduction="sum") / targets["multi_label_weights"].sum(),
        }

    def prepare_targets(self, outputs, targets):
        return {
            # "multi_label_onehot": paddle.stack([t["multi_label_onehot"] for t in targets], axis=0),
            # "multi_label_weights": paddle.stack([t["multi_label_weights"] for t in targets], axis=0),
            "multi_label_onehot": targets["multi_label_onehot"],
            "multi_label_weights": targets["multi_label_weights"],
        }

    def forward(self, outputs, aux_outputs, targets):
        targets = self.prepare_targets(outputs, targets)
        loss_dict = {}
        for loss in self.losses:
            loss_dict[f"cls_{loss}"] = self.loss_weights[loss] * self.loss_funcs[loss](outputs, targets)
            for i, aux_label_output in enumerate(aux_outputs):
                loss_dict[f"cls_{loss}_{i}"] = self.loss_weights[loss] * self.loss_funcs[loss](aux_label_output, targets)
        return loss_dict
    
    


class RetentionPolicy(nn.Layer):
    def __init__(self, args):
        super(RetentionPolicy, self).__init__()
        """ args: MODEL.PROMPT_INDICATOR.RETENTION_POLICY"""
        # select some class
        self.train_min_classes=args.train_min_classes
        self.train_max_classes=args.train_max_classes
        self.train_class_thr=args.train_class_thr
        # self.select_class_method=args.select_class_method
        # for eval
        self.eval_min_classes = args.eval_min_classes
        self.eval_max_classes = args.eval_max_classes
        self.eval_class_thr = args.eval_class_thr
    
    @paddle.no_grad()
    def forward(self, label_logits, force_sample_probs=None, num_classes=None):
        """ label_logits: bs * K  """
        """ Return:       bs * K' """
        # label_prob = label_logits.sigmoid() # bs, K
        label_prob = paddle.nn.functional.sigmoid(label_logits)
        if self.training:
            if force_sample_probs is not None:
                label_prob = paddle.where(force_sample_probs >= 0., x=force_sample_probs.astype("float32"), y=label_prob)
            min_classes = num_classes.clip(max=self.train_min_classes) if num_classes is not None else self.train_min_classes
            max_classes = num_classes.clip(max=self.train_max_classes) if num_classes is not None else self.train_max_classes
            class_thr = self.train_class_thr
        else:
            min_classes = num_classes.clip(max=self.eval_min_classes) if num_classes is not None else self.eval_min_classes
            max_classes = num_classes.clip(max=self.eval_max_classes) if num_classes is not None else self.eval_max_classes
            class_thr = self.eval_class_thr
        num_above_thr = (label_prob >= class_thr).sum(axis=1) # bs
        if isinstance(min_classes, paddle.Tensor):
            # num_train = num_above_thr.where(num_above_thr > min_classes, min_classes).where(num_above_thr < max_classes, max_classes)
            num_train = paddle.where(num_above_thr > min_classes, num_above_thr, min_classes)
            num_train = paddle.where(num_above_thr < max_classes, num_train, max_classes)
        else:
            num_train = num_above_thr.clip(min=min_classes, max=max_classes) # bs
        sorted_idxs = label_prob.argsort(axis=1, descending=True) # bs, nc
        bs_idx, cls_idx = [], []
        for id_b, (sorted_idx) in enumerate(sorted_idxs):
            n_train = num_train[id_b]
            cls_idx.append(sorted_idx[:n_train].sort())
            bs_idx.append(paddle.full_like(cls_idx[-1], id_b))
        return paddle.concat(bs_idx), paddle.concat(cls_idx)



# @serializable
# @register
class PromptIndicator(nn.Layer):
    def __init__(self, d_model, args): # MODEL.PROMPT_INDICATOR
        super(PromptIndicator, self).__init__()
        
        # __inject__ = [
        #     'bbox_post_process',
        #     'mask_post_process',
        # ]
        self.d_model = d_model
        
        # if CLASS_PROMPTS is None:
        #     CLASS_PROMPTS = dict(
        #         num_classes = 80,
        #         init_vectors = "",     # .npy or .pth file, empty means random initialized
        #         fix_class_prompts = False)
        
        self._init_class_prompts(args.CLASS_PROMPTS)
        # prompt blocks
        self.num_blocks = args.num_blocks
        self.level_preserve = args.level_preserve # only work for DeformableDETR

        prompt_block = TransformerDecoderLayer(args.BLOCK)
        self.prompt_blocks = _get_clones(prompt_block, self.num_blocks)
        
        hidden_dim = args.BLOCK.hidden_dim
        nhead = args.BLOCK.nheads
        dim_feedforward = args.BLOCK.dim_feedforward
        dropout = args.BLOCK.dropout
        activation = args.BLOCK.activation
        num_feature_levels = args.BLOCK.n_levels
        num_decoder_points = args.BLOCK.n_points
        
        weight_attr = None
        bias_attr = None
        

        # For classification
        # self.classifier_label = build_label_classifier(args.CLASSIFIER)
        self.classifier_label = DictClassifier(args.CLASSIFIER)
        self.classifier_label = nn.LayerList([self.classifier_label for _ in range(self.num_blocks)])
        self.criterion = ClassDecoderCriterion(args.LOSS)

        # For filter
        if args.retain_categories:
            self.retention_policy = RetentionPolicy(args.RETENTION_POLICY)
        else:
            self.retention_policy = None

    def _init_class_prompts(self, args): # MODEL.PROMPT_INDICATOR.CLASS_PROMPTS
        # load given vectors
        if args.init_vectors:
            if args.init_vectors[-3:] == "pth":
                class_prompts = paddle.load(args.init_vectors)
            elif args.init_vectors[-3:] == "npy":
                class_prompts = paddle.to_tensor(np.load(args.init_vectors), dtype="float32")
            else:
                raise KeyError
            if args.fix_class_prompts:
                self.register_buffer("class_prompts", class_prompts)
            else:
                # self.register_parameter("class_prompts", nn.Parameter(class_prompts))
                self.class_prompts = self.create_tensor(name="class_prompts")
                paddle.assign(class_prompts, self.class_prompts)
        # rand init
        else:
            num_classes = args.num_classes
            class_prompts = paddle.zeros([num_classes, self.d_model])
            assert args.fix_class_prompts == False
            # self.register_parameter("class_prompts", nn.Parameter(class_prompts))
            # nn.init.normal_(self.class_prompts.data)
            self.class_prompts = self.create_tensor(name="class_prompts")
            paddle.assign(class_prompts, self.class_prompts)
            nn.initializer.Normal()(self.class_prompts)
        
        # if the dimensiton does not match.
        if class_prompts.shape[1] != self.d_model:
            self.convert_vector = nn.Linear(class_prompts.shape[1], self.d_model)
            self.vector_ln = nn.LayerNorm(self.d_model)
        else:
            self.convert_vector = None

    def forward(self, srcs, mask, targets=None, kwargs={}):
        """
        srcs: bs, l, c
        mask:
        """
        bs = srcs.shape[0]
        # srcs process: only for deformable
        
        if len(self.level_preserve) > 0 and 'src_level_start_index' in kwargs: # False
            
            src_level_start_index = kwargs.pop('src_level_start_index')
            num_level = src_level_start_index.shape[0]
            new_srcs, new_mask = [], []
            for lvl in self.level_preserve:
                if lvl < num_level - 1:
                    new_srcs.append(srcs[:, src_level_start_index[lvl]: src_level_start_index[lvl+1], :])
                    new_mask.append(mask[:, src_level_start_index[lvl]: src_level_start_index[lvl+1]])
                else:
                    new_srcs.append(srcs[:, src_level_start_index[lvl]:, :])
                    new_mask.append(mask[:, src_level_start_index[lvl]:])
            src_level_start_index = paddle.to_tensor([0] + [m.shape[1] for m in new_mask[:-1]], device=src_level_start_index.device, dtype=src_level_start_index.dtype)
            src_level_start_index = src_level_start_index.cumsum(axis=0)
            kwargs['src_level_start_index'] = src_level_start_index
            srcs, mask = paddle.concat(new_srcs, axis=1), paddle.concat(new_mask, axis=1)

        # get class prompts
        if self.convert_vector is not None:
            class_prompts = self.vector_ln(self.convert_vector(self.class_prompts))
        else:
            class_prompts = self.class_prompts
        tgt_class = class_prompts.unsqueeze(0).tile([bs, 1, 1]) # [bs, 80, d_model]
        origin_class_vector = tgt_class
        # tgt_class: bs, K, d

        output_label_logits = []
        output_feats = []
        for lid, layer in enumerate(self.prompt_blocks):
            tgt_class = layer(tgt_class, None, None, srcs=srcs, src_padding_masks=mask, **kwargs) # bs, 91, c
            label_logits = self.classifier_label[lid](tgt_class, class_vector=origin_class_vector)
            label_logits = label_logits.reshape([bs, -1])
            output_label_logits.append(label_logits)
            output_feats.append(tgt_class)
            
        # organize outputs
        outputs = {
            'tgt_class': tgt_class,               # bs, cls_k, d
            'cls_label_logits': label_logits,     # bs, cls_k
            'cls_output_feats': tgt_class,        # bs, cls_k, d
        }

        # select some classes
        if self.retention_policy is not None:
            # force_sample_probs = paddle.stack([t["force_sample_probs"] for t in targets]) if self.training else None
            force_sample_probs = targets["force_sample_probs"] if self.training else None
            # num_classes = paddle.concat([t["num_classes"] for t in targets])
            # num_classes = targets["num_classes"]
            num_classes = paddle.to_tensor([80])
            bs_idxs, cls_idxs = self.retention_policy(label_logits, force_sample_probs, num_classes)    # bs, k'
            return_tgts = tgt_class[bs_idxs, cls_idxs]
            outputs.update({
                'bs_idx': bs_idxs,# cs_all
                'cls_idx': cls_idxs,  # cs_all
                'tgt_class': return_tgts, # cs_all, c
            })

        if len(output_label_logits) > 1:
            aux_outputs = [{'cls_label_logits': a}
                           for a in output_label_logits[:-1]]
        else:
            aux_outputs = []

        # organize losses
        assert targets is not None
        # targets["cls_class_prompts"] = class_prompts
        loss_dict = self.criterion(outputs, aux_outputs, targets)

        return outputs, loss_dict
