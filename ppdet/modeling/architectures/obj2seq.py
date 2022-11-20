# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['DETR']


@register
class Obj2Seq(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self,
                 backbone,
                 transformer,
                 detr_head,
                 post_process='DETRBBoxPostProcess',
                 num_feature_levels=4,
                 hidden_dim=256):
        super(Obj2Seq, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.num_feature_levels = num_feature_levels
        self.post_process = post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # transformer
        kwargs = {'input_shape': backbone.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            "detr_head": detr_head,
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Transformer
        out_transformer = self.transformer(body_feats, self.inputs['pad_mask'])

        # DETR Head
        if self.training:
            return self.detr_head(out_transformer, body_feats, self.inputs)
        else:
            preds = self.detr_head(out_transformer, body_feats)
            bbox, bbox_num = self.post_process(preds, self.inputs['im_shape'],
                                               self.inputs['scale_factor'])
            return bbox, bbox_num

    def get_loss(self, ):
        losses = self._forward()
        losses.update({
            'loss':
            paddle.add_n([v for k, v in losses.items() if 'log' not in k])
        })
        return losses

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {
            "bbox": bbox_pred,
            "bbox_num": bbox_num,
        }
        return output






#---------------------------------------------------------------------
class Obj2Seq(nn.Layer):
    """ This is Obj2Seq, our main framework """

    def __init__(self, args):
        """ Initializes the model.
        Parameters:
            args: refers _C.MODEL in config.yaml.
        """
        super().__init__()
        self.backbone = build_backbone(args.BACKBONE)
        self.transformer = build_transformer(args)
        num_feature_levels = args.BACKBONE.num_feature_levels
        hidden_dim = self.transformer.d_model

        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2D(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2D(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.LayerList(input_proj_list)
        else:
            self.input_proj = nn.LayerList([
                nn.Sequential(
                    nn.Conv2D(self.backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        for proj in self.input_proj:
            # nn.init.xavier_uniform_(proj[0].weight, gain=1)
            # nn.init.constant_(proj[0].bias, 0)
            nn.initializer.XavierNormal()(proj[0].weight)
            nn.initializer.Constant(0)(proj[0].bias)

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].cast("float32"), size=src.shape[-2:]).cast("bool")[0]
                srcs.append(src)
                masks.append(mask)

        outputs, loss_dict = self.transformer(srcs, masks, targets=targets)
        return outputs, loss_dict


def build_model(args):
    model = Obj2Seq(args.MODEL)

    # load-pretrained
    if args.MODEL.pretrained:
        pretrained_dict = paddle.load(args.MODEL.pretrained)
        if "model" in pretrained_dict:
            pretrained_dict = pretrained_dict["model"]
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Pretrain Model Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Pretrain Model Unexpected Keys: {}'.format(unexpected_keys))

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    if len(args.MODEL.fixed_params) > 0:
        for n, p in model.named_parameters():
            if match_name_keywords(n, args.MODEL.fixed_params):
                p.requires_grad_(False)

    return model