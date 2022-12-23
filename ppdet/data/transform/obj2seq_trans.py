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
import copy


from .operators import register_op
import numpy as np

__all__ = [
    "GenerateClassificationResults",
    "RearrangeByCls",
    "Resume"
]

@register_op
class GenerateClassificationResults(object):
    def __init__(self, num_cats, infer=False):
        self.num_cats = num_cats
        self.infer = infer

    def __call__(self, image):
        
        if "gt_class" in image:  # 什么意思意思 Obj2Seq 在推理时也传入了信息?
            multi_labels = np.unique(image['gt_class'])
        else:
            multi_labels = []
        # multi_labels = target["labels"].unique()
        multi_label_onehot = np.zeros((self.num_cats,))
        if len(multi_labels):
            multi_label_onehot[multi_labels] = 1
        multi_label_weights = np.ones_like(multi_label_onehot)

        # filter crowd items
        if not self.infer:
            keep = (image["is_crowd"] == 0).flatten()
            fields = ["is_crowd"]
            if 'gt_class' in image:
                fields.append("gt_class")
            if "gt_bbox" in image:
                fields.append("gt_bbox")
            if "masks" in image: # 分割的 masks, 不是 NestedTensor 的 masks
                fields.append("masks")
            if "keypoints" in image:
                fields.append("keypoints")
            for field in fields:
                image[field] = image[field][keep]

        if 'neg_category_ids' in image: # False
            # TODO:LVIS
            not_exhaustive_category_ids = [self.json_category_id_to_contiguous_id[idx] for idx in img_info['not_exhaustive_category_ids'] if idx in self.json_category_id_to_contiguous_id]
            neg_category_ids = [self.json_category_id_to_contiguous_id[idx] for idx in img_info['neg_category_ids'] if idx in self.json_category_id_to_contiguous_id]
            multi_label_onehot[not_exhaustive_category_ids] = 1
            multi_label_weights = multi_label_onehot.clone()
            multi_label_weights[neg_category_ids] = 1
        else:
            sample_prob = np.zeros_like(multi_label_onehot) - 1
            # sample_prob[target["labels"].unique()] = 1
            if len(multi_labels):
                sample_prob[multi_labels] = 1
        image["multi_label_onehot"] = multi_label_onehot   # {0, 1}
        image["multi_label_weights"] = multi_label_weights # {1}
        image["force_sample_probs"] = sample_prob          # {+1, -1}

        return image


@register_op
class RearrangeByCls(object):
    def __init__(self, keep_keys=["size", "orig_size", "im_id", "multi_label_onehot", "multi_label_weights", "force_sample_probs"], 
                 min_keypoints_train=0,
                 infer=False):
        # TODO: min_keypoints_train is deperacated
        self.min_keypoints_train = min_keypoints_train
        self.keep_keys = keep_keys
        self.infer=infer

    def __call__(self, image):

        if self.infer:
            return image
        
        # target["class_label"] = target["labels"].unique()
        image["class_label"] = np.unique(image['gt_class'])

        # new_target = {}
        # for icls in image["class_label"]:
        #     # icls = icls.item()
        #     new_target[icls] = {}
        #     where = (image["gt_class"] == icls).flatten()

        #     new_target[icls]["gt_bbox"] = image["gt_bbox"][where]
        #     if icls == 0 and "keypoints" in image:
        #         new_target[icls]["keypoints"] = image["keypoints"][where]

        # for key in self.keep_keys:
        #     if key not in ['size', 'orig_size']:
        #         new_target[key] = image[key]
        
        # w, h = image["w"], image["h"]
        # new_target['size'] = np.array([w, h])
        # new_target['orig_size'] = np.array([w, h])
        
        # # image["new_target"] = new_target
        
        return image
   
 
@register_op
class Resume:
    
    def __init__(self, kv):
        self.kv = dict(kv)
    
    def __call__(self, image):
        
        for key_o, key_n in self.kv.items():
            image[key_n] = copy.deepcopy( image[key_o] )
        
        return image