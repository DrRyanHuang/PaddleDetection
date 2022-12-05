# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import numpy as np
import math
from addict import Dict
import paddle
from paddle import nn
from paddle.nn import functional as F

# from util.misc import inverse_sigmoid
# from .classifiers import build_label_classifier
from .classwise_criterion import ClasswiseCriterion
from .set_criterion import SetCriterion


def inverse_sigmoid(x, eps=1e-5):
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1/x2)



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


class SeparateDetectHead(nn.Layer):
    def __init__(self, args):
        super().__init__()
        # output prediction
        d_model = args.CLASSIFIER.hidden_dim
        self.points_per_query = args.CLASSIFIER.num_points
        assert self.points_per_query == 1
        # self.class_embed = build_label_classifier(args.CLASSIFIER)
        self.class_embed = DictClassifier(args.CLASSIFIER)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.reset_parameters_as_first_head()

        # for loss
        self.criterion = SetCriterion(args.LOSS) if "multi" in args.CLASSIFIER.type else ClasswiseCriterion(args.LOSS)

    def reset_parameters_as_first_head(self):
        # nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        nn.initializer.Constant(0)(self.bbox_embed.layers[-1].weight)
        nn.initializer.Constant(0)(self.bbox_embed.layers[-1].bias)
        # nn.initializer.Constant(-2.0)(self.bbox_embed.layers[-1].bias[2:])
        nn.initializer.Assign(np.array([0, 0, -2, -2]))(self.bbox_embed.layers[-1].bias)

    def reset_parameters_as_refine_head(self):
        # nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.initializer.Constant(0)(self.bbox_embed.layers[-1].weight)
        nn.initializer.Constant(0)(self.bbox_embed.layers[-1].bias)

    def forward(self, feat, query_pos, reference_points, srcs, src_padding_masks, **kwargs):
        # feat:
        # reference_points
        # kwargs:
        ## class_vector:
        ## cls_idx:
        class_vector, cls_idx = kwargs.pop("class_vector", None), kwargs.pop("cls_idx", None)
        reference = inverse_sigmoid(reference_points)
        outputs_class = self.class_embed(feat, class_vector=class_vector, cls_idx=cls_idx) # bs(, cs), obj(, num_classes)
        # TODO: Implement for poins_per_query > 1
        tmp = self.bbox_embed(feat) # bs, cs, obj, 4
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
        outputs_coord = tmp.sigmoid()
        outputs = {
            "pred_logits": outputs_class.unsqueeze(-1) if  feat.dim() == 4 else outputs_class,
            "pred_boxes": outputs_coord,
            "class_index": cls_idx if cls_idx is not None else paddle.zeros((bs, 1), dtype=paddle.int64, device=outputs_class.device)
        }

        targets = kwargs.pop("targets", None)
        if targets is not None:
            detection_loss_dict = self.criterion(outputs, targets)
        else:
            assert not self.training, "Targets are required for training mode (separate_detect_head.py)"
            detection_loss_dict = {}
        return outputs, detection_loss_dict


class MLP(nn.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        # x: bs, cs, obj, c
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x