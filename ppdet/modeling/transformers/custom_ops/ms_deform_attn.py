# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.jit import to_static
from ...initializer import linear_init_, constant_, xavier_uniform_, normal_

# from ..functions import MSDeformAttnFunction
from ppdet.modeling.transformers.custom_ops.model import deformable_attention_core_func


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

# @to_static
def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x) # 


class MSDeformAttn(nn.Layer):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, no_value_proj=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super(MSDeformAttn, self).__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        # self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.no_value_proj = no_value_proj
        self.value_proj = nn.Identity() if no_value_proj else nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.value = None


        # self.constant_ = nn.initializer.Constant(0)
        # self.xavier_uniform_ = nn.initializer.XavierNormal()
        # self.param_assign = lambda x,y : nn.initializer.Assign(x)(y)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight, 0.)
        thetas = paddle.arange(self.n_heads, dtype="float32") * (2.0 * math.pi / self.n_heads)
        
        # grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).reshape([self.n_heads, 1, 1, 2]).tile([1, self.n_levels, self.n_points, 1])
        # for i in range(self.n_points):
        #     grid_init[:, :, i, :] *= i + 1
        # with paddle.no_grad():
        #     # self.sampling_offsets.bias = nn.Parameter(grid_init.reshape([-1]))
        #     self.param_assign(grid_init.reshape([-1]), self.sampling_offsets.bias)
        
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)
        grid_init = grid_init.reshape([self.n_heads, 1, 1, 2]).tile(
            [1, self.n_levels, self.n_points, 1])
        scaling = paddle.arange(
            1, self.n_points + 1,
            dtype=paddle.float32).reshape([1, 1, -1, 1])
        grid_init *= scaling
        self.sampling_offsets.bias.set_value(grid_init.flatten())
        
        constant_(self.attention_weights.weight, 0.)
        constant_(self.attention_weights.bias, 0.)
        if not self.no_value_proj:
            xavier_uniform_(self.value_proj.weight)
            constant_(self.value_proj.bias, 0.)
        
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias, 0.)

    def preprocess_value(self, input_flatten, input_padding_mask=None, cs_batch=None, bs_idx=None):
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten) # 线性投射层
        if input_padding_mask is not None:
            value = masked_fill(value, input_padding_mask[..., None], float(0)) # 反了吗? # TODO
        self.value = value.reshape([N, Len_in, self.n_heads, self.d_model // self.n_heads])
        if bs_idx is not None:
            self.value = self.value[bs_idx]
        elif cs_batch is not None:
            self.value = paddle.concat([
                v.expand([cs ,-1, -1, -1]) for cs, v in zip(cs_batch, self.value)
            ]) # cs_all, *, *, *

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, cs_batch=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
            # 注意 paddle mask要取反?
        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        if self.value is None:
            N, Len_in, _ = input_flatten.shape
            assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

            value = self.value_proj(input_flatten)
            if input_padding_mask is not None:
                # value = value.masked_fill(input_padding_mask[..., None], float(0))
                value = masked_fill(value, input_padding_mask[..., None], 0.0)
            value = value.reshape([N, Len_in, self.n_heads, self.d_model // self.n_heads])

            if cs_batch is not None:
                value = paddle.concat([
                    v.expand([cs ,-1, -1, -1]) for cs, v in zip(cs_batch, value)
                ]) # cs_all, *, *, *
                N = value.shape[0]
        else:
            value = self.value
            assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == value.shape[1]

        sampling_offsets = self.sampling_offsets(query).reshape([N, Len_q, self.n_heads, self.n_levels, self.n_points, 2])
        attention_weights = self.attention_weights(query).reshape([N, Len_q, self.n_heads, self.n_levels * self.n_points])
        attention_weights = F.softmax(attention_weights, -1).reshape([N, Len_q, self.n_heads, self.n_levels, self.n_points])
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        # output = MSDeformAttnFunction.apply(
        #     value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = deformable_attention_core_func(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output
