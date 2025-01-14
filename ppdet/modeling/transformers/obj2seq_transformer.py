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
#
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from addict import Dict
from yacs.config import CfgNode as CN
import copy
from paddle.jit import to_static


from ppdet.core.workspace import register
# from ..layers import MultiHeadAttention
from .position_encoding import PositionEmbedding
from .utils import _get_clones, deformable_attention_core_func
from ..initializer import linear_init_, constant_, xavier_uniform_, normal_
from .prompt_indicator import PromptIndicator
from .object_decoder import ObjectDecoder
from .attention_modules import DeformableEncoderLayer


__all__ = ['Obj2SeqDeformableTransformer']



# basic layer config
BASIC_LAYER_CFG = CN()
BASIC_LAYER_CFG.hidden_dim = 256
BASIC_LAYER_CFG.nheads = 8
BASIC_LAYER_CFG.dim_feedforward = 1024
BASIC_LAYER_CFG.dropout = 0.
BASIC_LAYER_CFG.self_attn_dropout = 0.
BASIC_LAYER_CFG.activation = "relu"
BASIC_LAYER_CFG.pre_norm = False
# some removal
BASIC_LAYER_CFG.no_self_attn = False
BASIC_LAYER_CFG.cross_attn_no_value_proj = False
# for Deformable-DETR like
BASIC_LAYER_CFG.n_levels = 4
BASIC_LAYER_CFG.n_points = 4


ENCODER_LAYER = copy.deepcopy(BASIC_LAYER_CFG)

OBJECT_DECODER = CN()
OBJECT_DECODER.LAYER = copy.deepcopy(BASIC_LAYER_CFG)
OBJECT_DECODER.num_layers = 4
OBJECT_DECODER.num_query_position = 100
OBJECT_DECODER.spatial_prior = 'sigmoid'
OBJECT_DECODER.refine_reference_points = False  # add 
OBJECT_DECODER.with_query_pos_embed = False
# OUTPUT Layers
OBJECT_DECODER.HEAD = CN()
OBJECT_DECODER.HEAD.type = "SeqHead"            # add
OBJECT_DECODER.HEAD.sg_previous_logits = False
OBJECT_DECODER.HEAD.combine_method = "multiple" # add
# for sequence head
OBJECT_DECODER.HEAD.pos_emb = True
OBJECT_DECODER.HEAD.num_steps = 4
OBJECT_DECODER.HEAD.num_classes = 80
OBJECT_DECODER.HEAD.task_category = "configs/obj2seq/tasks/coco_detection.json"
## for change structure in attention
OBJECT_DECODER.HEAD.self_attn_proj = True
OBJECT_DECODER.HEAD.cross_attn_no_value_proj = True
OBJECT_DECODER.HEAD.no_ffn = True
## to deperacate
OBJECT_DECODER.HEAD.keypoint_output = "nd_box_relative"

# -------------- 有更新 --------------
OBJECT_DECODER.HEAD.hidden_dim = 256
OBJECT_DECODER.HEAD.nheads = 8
OBJECT_DECODER.HEAD.dim_feedforward = 1024
OBJECT_DECODER.HEAD.dropout = 0.
OBJECT_DECODER.HEAD.self_attn_dropout = 0.
OBJECT_DECODER.HEAD.activation = "relu"
OBJECT_DECODER.HEAD.pre_norm = False
# some removal
OBJECT_DECODER.HEAD.no_self_attn = False
OBJECT_DECODER.HEAD.cross_attn_no_value_proj = False
# for Deformable-DETR like
OBJECT_DECODER.HEAD.n_levels = 4
OBJECT_DECODER.HEAD.n_points = 4


# if single classifier
OBJECT_DECODER.HEAD.CLASSIFIER = CN()
OBJECT_DECODER.HEAD.CLASSIFIER.type = 'dict'
OBJECT_DECODER.HEAD.CLASSIFIER.hidden_dim = 256
OBJECT_DECODER.HEAD.CLASSIFIER.num_layers = 2
OBJECT_DECODER.HEAD.CLASSIFIER.init_prob = 0.01
OBJECT_DECODER.HEAD.CLASSIFIER.num_points = 1
OBJECT_DECODER.HEAD.CLASSIFIER.skip_and_init = False
OBJECT_DECODER.HEAD.CLASSIFIER.normalize_before = False

OBJECT_DECODER.HEAD.LOSS = CN()
OBJECT_DECODER.HEAD.LOSS.num_classes = 80
OBJECT_DECODER.HEAD.LOSS.losses = ['labels', 'boxes']
OBJECT_DECODER.HEAD.LOSS.aux_loss = True
OBJECT_DECODER.HEAD.LOSS.focal_alpha = 0.25
OBJECT_DECODER.HEAD.LOSS.cls_loss_coef = 2.0
OBJECT_DECODER.HEAD.LOSS.bbox_loss_coef = 5.0
OBJECT_DECODER.HEAD.LOSS.giou_loss_coef = 2.0
OBJECT_DECODER.HEAD.LOSS.mse_loss_coef = 0.0
OBJECT_DECODER.HEAD.LOSS.keypoint_l1_loss_coef = 1.0
OBJECT_DECODER.HEAD.LOSS.keypoint_oks_loss_coef = 1.0
# more options for class loss
OBJECT_DECODER.HEAD.LOSS.bce_negative_weight = 1.0
OBJECT_DECODER.HEAD.LOSS.class_normalization = "num_box"  # ["num_box", "num_pts", "mean", "none"]

# --------- 更新版本 ---------
OBJECT_DECODER.HEAD.LOSS.task_category = OBJECT_DECODER.HEAD.task_category
OBJECT_DECODER.HEAD.LOSS.num_classes   = OBJECT_DECODER.HEAD.num_classes

# for keypoints
OBJECT_DECODER.HEAD.LOSS.keypoint_criterion = "L1"
OBJECT_DECODER.HEAD.LOSS.keypoint_normalization = "num_box"  # ["num_box", "num_pts", "mean", "none"]
OBJECT_DECODER.HEAD.LOSS.oks_normalization = "num_box"
OBJECT_DECODER.HEAD.LOSS.keypoint_reference = "absolute" # ["absolute" or "relative"]
OBJECT_DECODER.HEAD.LOSS.keypoint_relative_ratio = 1.0

OBJECT_DECODER.HEAD.LOSS.MATCHER = CN()
OBJECT_DECODER.HEAD.LOSS.MATCHER.fix_match_train = ""
OBJECT_DECODER.HEAD.LOSS.MATCHER.fix_match_val = ""
OBJECT_DECODER.HEAD.LOSS.MATCHER.set_class_type = "focal" # ["focal", "bce", "logits", "probs"]
OBJECT_DECODER.HEAD.LOSS.MATCHER.set_cost_class = 2.0
OBJECT_DECODER.HEAD.LOSS.MATCHER.set_cost_bbox = 5.0
OBJECT_DECODER.HEAD.LOSS.MATCHER.set_cost_giou = 2.0
OBJECT_DECODER.HEAD.LOSS.MATCHER.set_cost_keypoints_oks = 0.0
OBJECT_DECODER.HEAD.LOSS.MATCHER.set_cost_keypoints_l1 = 0.0
OBJECT_DECODER.HEAD.LOSS.MATCHER.set_class_normalization = "none"
OBJECT_DECODER.HEAD.LOSS.MATCHER.set_box_normalization = "none" # ["num_box", "num_pts", "mean", "none"]
OBJECT_DECODER.HEAD.LOSS.MATCHER.set_keypoint_normalization = "none"
OBJECT_DECODER.HEAD.LOSS.MATCHER.set_oks_normalization = "none"
#### maybe this is deprecated ?
OBJECT_DECODER.HEAD.LOSS.MATCHER.set_keypoint_reference = "absolute" # ["absolute" or "relative"]





# prompt_indicator
PROMPT_INDICATOR = CN()
PROMPT_INDICATOR.num_blocks = 2
PROMPT_INDICATOR.return_intermediate = True
PROMPT_INDICATOR.level_preserve = [] # only for deformable, empty means all feature levels are used
# cfg for attention layer
PROMPT_INDICATOR.BLOCK = copy.deepcopy(BASIC_LAYER_CFG)
PROMPT_INDICATOR.BLOCK.no_self_attn = True
# cfg for prompt vectors
PROMPT_INDICATOR.CLASS_PROMPTS = CN()
PROMPT_INDICATOR.CLASS_PROMPTS.num_classes = 80
PROMPT_INDICATOR.CLASS_PROMPTS.init_vectors = "configs/obj2seq/word_arrays/coco_clip_v2.npy" # .npy or .pth file, empty means random initialized
PROMPT_INDICATOR.CLASS_PROMPTS.fix_class_prompts = False
# cfg for classifier
PROMPT_INDICATOR.CLASSIFIER = CN()
PROMPT_INDICATOR.CLASSIFIER.type = 'dict'
PROMPT_INDICATOR.CLASSIFIER.hidden_dim = 256
PROMPT_INDICATOR.CLASSIFIER.num_layers = 2
PROMPT_INDICATOR.CLASSIFIER.init_prob = 0.1
PROMPT_INDICATOR.CLASSIFIER.num_points = 1
PROMPT_INDICATOR.CLASSIFIER.skip_and_init = False
PROMPT_INDICATOR.CLASSIFIER.normalize_before = False
# asl loss
PROMPT_INDICATOR.LOSS = CN()
PROMPT_INDICATOR.LOSS.losses = ['asl']
PROMPT_INDICATOR.LOSS.asl_optimized = True
PROMPT_INDICATOR.LOSS.asl_loss_weight = 0.25
PROMPT_INDICATOR.LOSS.asl_gamma_pos = 0.0
PROMPT_INDICATOR.LOSS.asl_gamma_neg = 2.0
PROMPT_INDICATOR.LOSS.asl_clip = 0.0
# cfg for retention_policy
PROMPT_INDICATOR.retain_categories = True
PROMPT_INDICATOR.RETENTION_POLICY = CN()
PROMPT_INDICATOR.RETENTION_POLICY.train_max_classes = 20
PROMPT_INDICATOR.RETENTION_POLICY.train_min_classes = 20
PROMPT_INDICATOR.RETENTION_POLICY.train_class_thr = 0.0
PROMPT_INDICATOR.RETENTION_POLICY.eval_min_classes = 20
PROMPT_INDICATOR.RETENTION_POLICY.eval_max_classes = 20
PROMPT_INDICATOR.RETENTION_POLICY.eval_class_thr = 0.0

class TransformerEncoder(nn.Layer):
    def __init__(self, args_ENCODER_LAYER, enc_layers=6):
        super(TransformerEncoder, self).__init__()
        self.enc_layers = enc_layers
        encoder_layer = DeformableEncoderLayer(args_ENCODER_LAYER)
        self.encoder_layers =  _get_clones(encoder_layer, self.enc_layers)
    
    def forward(self, tgt, *args, **kwargs):
        # tgt: bs, h, w, c || bs, l, c
        
        for layer in self.encoder_layers: # 6 个 Deformable Encoder
            tgt = layer(tgt, *args, **kwargs)
        return tgt


@register
class Obj2SeqDeformableTransformer(nn.Layer):
    __shared__ = ['hidden_dim']

    def __init__(self,
                 prompt_indicator=None,
                 num_queries=80, # TODO
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 backbone_num_channels=[512, 1024, 2048],
                 num_feature_levels=4,
                 num_encoder_points=4,
                 num_decoder_points=4,
                 hidden_dim=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 lr_mult=0.1,
                 weight_attr=None,
                 bias_attr=None):
        super(Obj2SeqDeformableTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_num_channels) <= num_feature_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        
        self.encoder = TransformerEncoder(ENCODER_LAYER)


        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)

        self.reference_points = nn.Linear(
            hidden_dim,
            2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))

        # ========= 此处用来给 ResNet 加层的 =========
        self.input_proj = nn.LayerList()
        for in_channels in backbone_num_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        hidden_dim,
                        kernel_size=1,
                        weight_attr=weight_attr,
                        bias_attr=bias_attr),
                    nn.GroupNorm(32, hidden_dim)
                )
            )
        in_channels = backbone_num_channels[-1]
        for _ in range(num_feature_levels - len(backbone_num_channels)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        weight_attr=weight_attr,
                        bias_attr=bias_attr),
                    nn.GroupNorm(32, hidden_dim)
                    )
                )
            in_channels = hidden_dim

        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type,
            offset=-0.5)

        self._reset_parameters()        
        
        d_model = 256
        if isinstance(prompt_indicator, str):
            self.prompt_indicator = PromptIndicator(
                d_model,
                PROMPT_INDICATOR
            )

        with_object_decoder = True
                
        
        # object decoder
        if with_object_decoder:
            self.object_decoder = ObjectDecoder(d_model, args=OBJECT_DECODER)
        else:
            self.object_decoder = None
        

    def _reset_parameters(self):
        normal_(self.level_embed.weight)
        normal_(self.tgt_embed.weight)
        normal_(self.query_pos_embed.weight)
        xavier_uniform_(self.reference_points.weight)
        constant_(self.reference_points.bias)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)
            constant_(l[0].bias)


    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_num_channels': [i.channels for i in input_shape], }

    # @to_static
    def _get_valid_ratio(self, mask): # mask 的比例 [bs, ]
        mask = 1 - mask.astype(paddle.float32)
        _, H, W = mask.shape
        valid_ratio_h = paddle.sum(mask[:, :, 0], 1) / H
        valid_ratio_w = paddle.sum(mask[:, 0, :], 1) / W
        valid_ratio = paddle.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    
    # @staticmethod
    # @to_static
    def get_reference_points(self, spatial_shapes, valid_ratios):
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []
        for i, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = paddle.meshgrid(
                paddle.linspace(0.5, H - 0.5, H),
                paddle.linspace(0.5, W - 0.5, W))
            ref_y = ref_y.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 1] * H)
            ref_x = ref_x.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 0] * W)
            reference_points.append(
                paddle.stack((ref_x, ref_y), axis=-1)
            )
        reference_points = paddle.concat(reference_points, 1).unsqueeze(2)
        reference_points = reference_points * valid_ratios
        # https://github.com/fundamentalvision/Deformable-DETR/issues/36
        return reference_points # [bs, 17197, 4, 2] # [x, x, 1, x] 重复了4次
    

    def forward(self, src_feats, src_mask, targets=None):
        
        with paddle.no_grad():
        # if True:
            
            # src_feats: a list of tensors [(bs, c, h_i, w_i), ...]
            # src_mask : a list of tensors [(bs, h_i, w_i), ...]
            
            if src_mask[0].ndim == 2:
                src_mask = [
                    mask[None] for mask in src_mask
                ]
            
            
            # ========= 加工变量 srcs 和 masks =========
            srcs =  []
            masks = []
            for i in range(len(src_feats)):
                srcs.append(self.input_proj[i](src_feats[i]))
                masks.append(src_mask[i])
                
            if self.num_feature_levels > len(srcs):
                len_srcs = len(srcs)
                for i in range(len_srcs, self.num_feature_levels):
                    if i == len_srcs:
                        srcs.append(self.input_proj[i](src_feats[-1]))
                    else:
                        srcs.append(self.input_proj[i](srcs[-1]))
                        
                    mask = F.interpolate(masks[-1][None], size=srcs[-1].shape[-2:])[0]
                    masks.append(mask)
            # ========= 加工变量 srcs 和 masks =========
            
            
            # ---------- 测试时间 1.7s 100次 ----------
            srcs, mask, enc_kwargs, cls_kwargs, obj_kwargs = self.prepare_for_deformable(srcs, masks)

            # # encoder ----- 0.5s -----
            srcs = self.encoder(srcs, padding_mask=mask, **enc_kwargs) if self.encoder is not None else srcs
            
            
            # prompt_indicator ----- 0.03s -----
            outputs, loss_dict = {}, {}
            if self.prompt_indicator is not None:
                cls_outputs, cls_loss_dict = self.prompt_indicator(srcs, mask, targets=targets, kwargs=cls_kwargs)
                outputs.update(cls_outputs)
                loss_dict.update(cls_loss_dict)
                additional_object_inputs = dict(
                    bs_idx = outputs["bs_idx"] if "bs_idx" in outputs else None,
                    cls_idx = outputs["cls_idx"] if "cls_idx" in outputs else None,
                    class_vector = outputs["tgt_class"],           # cs_all, d
                    previous_logits = outputs["cls_label_logits"], # bs, 80
                )
            else:
                additional_object_inputs = {}
            
            # print(cls_outputs['cls_label_logits'].argsort(1, True)[:, :20].numpy())
            # print([l.numpy().tolist() for l in targets['class_label']])
            
            # per = ((F.sigmoid(cls_outputs['cls_label_logits']) > 0.2).cast("int32") * targets['multi_label_onehot']).sum() / targets['multi_label_onehot'].sum()
            # print(per)


        loss_dict = {}
        if self.object_decoder is not None:
        # if False: # 先去只判断类别的有无，之后再看结果
            
            # ------ object_decoder ------ 约 0.8s 
            obj_outputs, obj_loss_dict = self.object_decoder(srcs, 
                                                             mask, 
                                                             targets=targets, 
                                                             additional_info=additional_object_inputs, 
                                                             kwargs=obj_kwargs)
            outputs.update(obj_outputs)
            loss_dict.update(obj_loss_dict)

        
        return outputs, loss_dict


    # @to_static
    def prepare_for_deformable(self, srcs, masks):
        
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for level, (src, mask) in enumerate(zip(srcs, masks)):
            bs, c, h, w = src.shape
            spatial_shapes.append([h, w])
            
            if self.encoder is not None:
                pos_embed = self.position_embedding(1-mask).flatten(2).transpose([0, 2, 1])    # 相对位置 embed
                lvl_pos_embed = pos_embed + self.level_embed.weight[level].reshape([1, 1, -1]) # 不同尺度的 embed
                lvl_pos_embed_flatten.append(lvl_pos_embed)
            
            valid_ratios.append(self._get_valid_ratio(mask)) # [bs, h_, w_]
            
            # =============== src 和 mask 部分 ===============
            src = src.flatten(2).transpose([0, 2, 1])
            src_flatten.append(src)
            mask = mask.flatten(1)
            mask_flatten.append(mask)
            # =============== src 和 mask 部分 ===============

        src_flatten  = paddle.concat(src_flatten, 1)  # [bs, 11400, 256]
        mask_flatten = paddle.concat(mask_flatten, 1) # [bs, 11400]
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1) if self.encoder is not None else None  # [4, 11400, 256]
        spatial_shapes = paddle.to_tensor(spatial_shapes, dtype='int32') # [4, 2]
        level_start_index = paddle.concat(
            [ paddle.zeros((1, ), dtype="int32"), spatial_shapes.prod(1).cumsum(0)[:-1] ]
            ) # [4]
        valid_ratios = paddle.stack(valid_ratios, 1) # [bs, 4, 2]
        
        reference_points_enc = self.get_reference_points(spatial_shapes, valid_ratios)
        
        enc_kwargs = dict(spatial_shapes = spatial_shapes,
                          level_start_index = level_start_index,
                          reference_points = reference_points_enc,
                          pos = lvl_pos_embed_flatten)
        cls_kwargs = dict(src_level_start_index=level_start_index)
        obj_kwargs = dict(src_spatial_shapes=spatial_shapes,
                          src_level_start_index=level_start_index,
                          src_valid_ratios=valid_ratios)
        
        return src_flatten, mask_flatten, enc_kwargs, cls_kwargs, obj_kwargs