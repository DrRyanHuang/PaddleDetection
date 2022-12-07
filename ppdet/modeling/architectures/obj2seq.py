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
import paddle.nn as nn
import paddle.nn.functional as F
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['Obj2Seq']


@register
class Obj2Seq(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self,
                 backbone,
                 transformer,
                 detr_head=None,
                 post_process=None,
                 num_feature_levels=4):
        super(Obj2Seq, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.num_feature_levels = num_feature_levels
        # self.post_process = post_process
        
        # -------------- 在 DETR Transformer 里 --------------
        # hidden_dim = self.transformer.hidden_dim
        
        # if num_feature_levels > 1:
        #     num_backbone_outs = len(self.backbone.out_shape)
        #     self.backbone.num_channels = [self.backbone.out_shape[_].channels for _ in range(num_backbone_outs)]
        #     input_proj_list = []
        #     for _ in range(num_backbone_outs):
        #         in_channels = self.backbone.num_channels[_]
        #         input_proj_list.append(nn.Sequential(
        #             nn.Conv2D(in_channels, hidden_dim, kernel_size=1),
        #             nn.GroupNorm(32, hidden_dim),
        #         ))
        #     for _ in range(num_feature_levels - num_backbone_outs):
        #         input_proj_list.append(nn.Sequential(
        #             nn.Conv2D(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
        #             nn.GroupNorm(32, hidden_dim),
        #         ))
        #         in_channels = hidden_dim
        #     self.input_proj = nn.LayerList(input_proj_list)
        # else:
        #     self.input_proj = nn.LayerList([
        #         nn.Sequential(
        #             nn.Conv2D(self.backbone.num_channels[0], hidden_dim, kernel_size=1),
        #             nn.GroupNorm(32, hidden_dim),
        #         )])

        # for proj in self.input_proj:
        #     nn.initializer.XavierNormal()(proj[0].weight)
        #     nn.initializer.Constant(0)(proj[0].bias)

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
        # detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            # "detr_head": detr_head,
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)
        
        # feats = []
        # for l, feat in enumerate(body_feats):
        #     feats.append(self.input_proj[l](feat))
        # if self.num_feature_levels > len(feats):
        #     _len_srcs = len(feats)
        #     for l in range(_len_srcs, self.num_feature_levels):
        #         if l == _len_srcs:
        #             src = self.input_proj[l](body_feats[-1])
        #         else:
        #             src = self.input_proj[l](feats[-1])
        #         feats.append(src)

        # Transformer
        out_transformer = self.transformer(body_feats, self.inputs['pad_mask'], self.inputs)

        
        outputs, loss_dict = out_transformer
        # losses = sum(loss_dict[k] for k in loss_dict.keys())

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced = {k: v
        #                      for k, v in loss_dict_reduced.items()}
        # losses_reduced = sum(loss_dict_reduced.values())

        # det_loss  = sum(loss_dict_reduced[k] for k in loss_dict_reduced.keys() if 'kps' not in k).item()
        # loss_value = losses_reduced.item()
        
        # # DETR Head
        # if self.training:
        #     return self.detr_head(out_transformer, body_feats, self.inputs)
        # else:
        #     preds = self.detr_head(out_transformer, body_feats)
        #     bbox, bbox_num = self.post_process(preds, self.inputs['im_shape'],
        #                                        self.inputs['scale_factor'])
        # return bbox, bbox_num
        
        if self.training:
            return loss_dict


    def get_loss(self, ):
        losses = self._forward()
        losses.update({
            'loss':
            # paddle.add_n([v for k, v in losses.items() if 'log' not in k])
            sum(losses[k] for k in losses.keys())
        })
        return losses

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {
            "bbox": bbox_pred,
            "bbox_num": bbox_num,
        }
        return output

