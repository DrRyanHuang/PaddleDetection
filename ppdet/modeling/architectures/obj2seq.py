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

import time

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create
from paddle.jit import to_static
from tqdm import tqdm

__all__ = ['Obj2Seq']


@register
class Obj2Seq(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self,
                 backbone,
                 transformer,
                 post_process=None,
                 num_feature_levels=4):
        super(Obj2Seq, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
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

        return {
            'backbone': backbone,
            'transformer': transformer,
        }

    # @to_static
    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)
        # print(self.inputs['im_id'])
        
        # 把最初的 mask 插值为小的, 为了和 torch NestedTensor 对齐, -1
        body_feats_mask = [1-F.interpolate(self.inputs['pad_mask'][None], 
                                         size=x.shape[-2:]).squeeze()
                               for x in body_feats]
        
        
        # ------------------ transformer 测试时间不到 0.5s ------------------
        out_transformer = self.transformer(body_feats, body_feats_mask, self.inputs)
        outputs, loss_dict = out_transformer
        
        if self.training:
            return loss_dict
        else:
            orig_target_sizes = self.inputs["im_shape"]
            results = self.post_process(outputs, orig_target_sizes)
            # res = {tgt.item(): output for tgt, output in zip(self.inputs["im_id"], results)}
            
            # num_id, score, xmin, ymin, xmax, ymax
            bbox = [
                paddle.concat([
                    results[i]['labels'][:, None].cast("float32"),
                    results[i]["scores"][:, None],
                    results[i]['boxes']
                ], axis=1)
                for i in range(len(results))
            ]
            bbox = paddle.concat(bbox, axis=0)
            
            bbox_num = [results[i]["scores"].shape[0] for i in range(len(results))]
            bbox_num = paddle.to_tensor(bbox_num, dtype="int32")
            
            return bbox, bbox_num


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

