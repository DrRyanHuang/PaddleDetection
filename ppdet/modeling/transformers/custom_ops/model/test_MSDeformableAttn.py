# 引入自定义op
import paddle
from deformable_detr_ops import ms_deformable_attn # 编译后才可

# 构造fake input tensor
bs, n_heads, c = 2, 8, 8
query_length, n_levels, n_points = 2, 2, 2
spatial_shapes = paddle.to_tensor([(6, 4), (3, 2)], dtype=paddle.int64)
level_start_index = paddle.concat((paddle.to_tensor(
    [0], dtype=paddle.int64), spatial_shapes.prod(1).cumsum(0)[:-1]))
value_length = sum([(H * W).item() for H, W in spatial_shapes])

def get_test_tensors(channels):
    value = paddle.rand(
        [bs, value_length, n_heads, channels], dtype=paddle.float32) * 0.01
    sampling_locations = paddle.rand(
        [bs, query_length, n_heads, n_levels, n_points, 2],
        dtype=paddle.float32)
    attention_weights = paddle.rand(
        [bs, query_length, n_heads, n_levels, n_points],
        dtype=paddle.float32) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
        -2, keepdim=True)

    return [value, sampling_locations, attention_weights]

value, sampling_locations, attention_weights = get_test_tensors(c)

output = ms_deformable_attn(value,
                            spatial_shapes,
                            level_start_index,
                            sampling_locations,
                            attention_weights)

print(output.shape)