"""Default-off exact BF16 head replay candidate for Qwen Flash-Next.

One SIMDgroup computes four vocabulary outputs for one/two input vectors.
Each vector retains MLX 0.32.4 GEMV's sequential four-column lane work and
shuffle-down reduction. The two-vector tile shares weight loads, without
changing the accumulation tree. No model route is replaced by this module.
"""

import mlx.core as mx

BF16_HEAD_ENABLED = False
BF16_HEAD_CALLS = {1: 0, 2: 0}
BF16_HEAD_DISABLED = False
_VALIDATED = set()
_KERNEL = None

_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const size_t first = (threadgroup_position_in_grid.y * 8 + sg) * 4;
    float result[M][4] = {0};
    for (int block = 0; block < 20; block++) {
        const int col = int(lane) * 4 + block * 128;
        float vectors[M][4];
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < 4; j++) {
                vectors[m][j] = float(x[m * x_strides[0] + (col + j) * x_strides[1]]);
            }
        }
        for (int out = 0; out < 4; out++) {
            T values[4];
            for (int j = 0; j < 4; j++) {
                values[j] = w[(first + out) * w_strides[0] + (col + j) * w_strides[1]];
            }
            for (int m = 0; m < M; m++) {
                for (int j = 0; j < 4; j++) {
                    result[m][out] += float(values[j]) * vectors[m][j];
                }
            }
        }
    }
    for (int m = 0; m < M; m++) {
        for (int out = 0; out < 4; out++) {
            for (ushort delta = 16; delta >= 1; delta >>= 1) {
                result[m][out] += simd_shuffle_down(result[m][out], delta);
            }
        }
    }
    if (lane == 0) {
        for (int m = 0; m < M; m++) {
            for (int out = 0; out < 4; out++) {
                y[m * O + first + out] = T(result[m][out]);
            }
        }
    }
"""


def _project(x, weight):
    """Pure numerical seam; K=2560, O divisible by 32, one/two rows."""
    global _KERNEL
    rows, width = x.size // 2560, weight.shape[0]
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen4_bf16_head_pair",
            input_names=["x", "w"], output_names=["y"],
            source=_SOURCE, ensure_row_contiguous=False,
        )
    return _KERNEL(
        inputs=[x.reshape(rows, 2560), weight],
        template=[("T", mx.bfloat16), ("M", rows), ("O", width)],
        grid=(32, width // 4, 1), threadgroup=(32, 8, 1),
        output_shapes=[(*x.shape[:-1], width)], output_dtypes=[mx.bfloat16],
    )[0]


def try_bf16_head(x, weight):
    """Return an opt-in exact head projection, or None for the stock route."""
    global BF16_HEAD_DISABLED
    if (
        not BF16_HEAD_ENABLED or BF16_HEAD_DISABLED
        or x.ndim != 3 or x.shape[0] != 1 or x.shape[1] not in (1, 2)
        or x.shape[2] != 2560 or weight.shape != (248320, 2560)
        or x.dtype != mx.bfloat16 or weight.dtype != mx.bfloat16
        or mx.default_device() != mx.gpu or not mx.metal.is_available()
    ):
        return None
    rows = x.shape[1]
    try:
        result = _project(x, weight)
        if rows not in _VALIDATED:
            mx.eval(result)
            _VALIDATED.add(rows)
        BF16_HEAD_CALLS[rows] += 1
        return result
    except Exception:
        BF16_HEAD_DISABLED = True
        return None
