import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def deformable_attention_core_func(value, value_spatial_shapes,
                                   value_level_start_index, sampling_locations,
                                   attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor): [n_levels, 2]
        value_level_start_index (Tensor): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]
    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    value_list = value.split(
        value_spatial_shapes.prod(1).split(n_levels), axis=1)
    sampling_grids = 2 * sampling_locations - 1 # ?
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(
            [0, 2, 1]).reshape([bs * n_head, c, h, w])
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(
            [0, 2, 1, 3, 4]).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose([0, 2, 1, 3, 4]).reshape(
        [bs * n_head, 1, Len_q, n_levels * n_points])
    output = (paddle.stack(
        sampling_value_list, axis=-2).flatten(-2) *
              attention_weights).sum(-1).reshape([bs, n_head * c, Len_q])

    return output.transpose([0, 2, 1])