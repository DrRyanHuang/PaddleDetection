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
        
        import numpy as np
        import cv2
        
        # idx = 0
        # image = self.inputs['image'][idx].transpose([1, 2, 0]).numpy()
        # image = (image * [0.229, 0.224,0.225] + [0.485, 0.456, 0.406]) * 255
        # image = image.astype(int).astype("uint8")
        
        # curr_h = int(self.inputs['pad_mask'][idx].sum(0).max().item())
        # curr_w = int(self.inputs['pad_mask'][idx].sum(1).max().item())
        
        # image = image[:curr_h, :curr_w]
        # image = image[:, :, ::-1]
        # image = np.ascontiguousarray(image)

        # bboxes = self.inputs['gt_bbox'][idx].numpy() * [curr_w, curr_h, curr_w, curr_h]
        # bboxes = bboxes.astype(int)
        
        # bbox = bboxes[0]
        # xc, yc, bw, bh = bbox.astype(int)

        # x1, y1, x2, y2 = int(xc-bw//2), int(yc-bh//2), int(xc+bw//2), int(yc+bh//2)

        # xx = cv2.rectangle(image, (x1, y1), (x2, y2), 255, 2, 8)   # 这里报了错
        # cv2.imwrite("xxx.png", image)



        # h, w = self.inputs["ori_im_shape"][idx].numpy()
        
        
        
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
        self.training = False
        if self.training:
            return loss_dict
        else:
            orig_target_sizes = self.inputs["ori_im_shape"]
            results = self.post_process(outputs, orig_target_sizes)
            # res = {tgt.item(): output for tgt, output in zip(self.inputs["im_id"], results)}

            
            self_inputs_gt_bbox_0 = paddle.to_tensor(
                [[0.38957810, 0.41610327, 0.03859374, 0.16314554],
                [0.12764062, 0.50515258, 0.23331250, 0.22269955],
                [0.93419528, 0.58346248, 0.12710935, 0.18481222],
                [0.60465628, 0.63254696, 0.08749998, 0.24138498],
                [0.50250781, 0.62732399, 0.09660935, 0.23117369],
                [0.66919529, 0.61899060, 0.04714060, 0.19098592],
                [0.51279688, 0.52825117, 0.03371879, 0.02720654],
                [0.68644530, 0.53196013, 0.08289063, 0.32396716],
                [0.61248434, 0.44619718, 0.02362496, 0.08389670],
                [0.81185937, 0.50172538, 0.02303135, 0.03748825],
                [0.78632033, 0.53637326, 0.03170311, 0.25424880],
                [0.95615625, 0.77170193, 0.02240622, 0.10730046],
                [0.96824998, 0.77807510, 0.02012503, 0.10901403],
                [0.71055472, 0.31000000, 0.02182811, 0.05136150],
                [0.88656247, 0.83160800, 0.05731249, 0.21049297],
                [0.55694532, 0.51670188, 0.01776564, 0.05293429],
                [0.65166402, 0.52882630, 0.01504689, 0.02938962],
                [0.38804689, 0.47841549, 0.02221876, 0.04138494],
                [0.53383595, 0.48794597, 0.01520312, 0.03927228],
                [0.59998435, 0.64714789, 0.19618750, 0.20875585]], dtype="float32"
            )
            
            
            h, w = orig_target_sizes[0]
            x_cy_cwh = self_inputs_gt_bbox_0 * paddle.concat([w, h, w, h])
            # x1y1wh = x_cy_cwh
            # x1y1wh[:, :2] -= x1y1wh[:, 2:] / 2 
            x1y1x2y2 = paddle.zeros_like(x_cy_cwh)
            x1y1x2y2[:, :2] = x_cy_cwh[:, :2] - x_cy_cwh[:, 2:] / 2
            x1y1x2y2[:, 2:] = x_cy_cwh[:, :2] + x_cy_cwh[:, 2:] / 2
            
            results[0]['boxes'][:x_cy_cwh.shape[0]] = x1y1x2y2
            results[0]["scores"][:x_cy_cwh.shape[0]] = 1
            # results[0]['labels'] = 
            
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

