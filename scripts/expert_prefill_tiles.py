"""Two experimental M4 sorted-expert tiles. No serving integration.

Reuse pinned MLX0.32.4's actual scalar dequantisation, BlockMMA, K loop,
offset search and row-tile scheduling. Change only the output tile extent:
BM32/BN32 or BM16/BN64, retaining BK32/WM1/WN2. Bit equality to the sealed
primitive is a mandatory measured gate, not an assumption of this module.
"""
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from omlx.patches.m5_gather_qmm_nax import _read_mlx_headers

VARIANTS = {"m32": (32, 32), "n64": (16, 64)}


@lru_cache(maxsize=1)
def source_parts():
    if mx.__version__ != "0.32.4":
        raise RuntimeError("This experiment requires the pinned MLX0.32.4 source")
    path = Path(mx.__file__).parent / "include/mlx/backend/metal/kernels/quantized.h"
    quantized = path.read_text()
    header = _read_mlx_headers((
        "mlx/backend/metal/kernels/steel/gemm/gemm.h",
        "mlx/backend/metal/kernels/quantized_utils.h",
        "mlx/backend/metal/kernels/quantized.h",
    ))
    if header is None:
        raise RuntimeError("Pinned Metal kernel headers unavailable")
    # Only aligned N/K are eligible, so these function constants are literal.
    for name, number in (("align_M", 200), ("align_N", 201), ("align_K", 202)):
        old = f"constant bool {name} [[function_constant({number})]];"
        if header.count(old) != 1:
            raise RuntimeError(f"Unexpected pinned-source declaration: {name}")
        header = header.replace(old, f"constant bool {name} = true;")
    start = quantized.index("[[kernel]] void affine_gather_qmm_rhs(")
    start = quantized.index("  constexpr int pack_factor", start)
    stop = quantized.index("\ntemplate ", start)
    body = quantized[start:stop].rsplit("}", 1)[0]
    return header, body


@lru_cache(maxsize=1)
def offsets_kernel():
    # Exact binary-search prepass from MLX0.32.4 gather_mm_offsets.metal.
    return mx.fast.metal_kernel(name="expert_tile_offsets", input_names=["indices", "params"],
        output_names=["offsets"], compile_options={"math_mode": "safe"}, source="""
        const uint g = thread_position_in_grid.x;
        if (g > uint(params[3])) return;
        int lo = 0;
        int hi = params[0];
        while (lo < hi) {
          const int mid = (lo + hi) / 2;
          if (indices[mid] < g) lo = mid + 1;
          else hi = mid;
        }
        offsets[g] = lo;
        """)


@lru_cache(maxsize=2)
def matmul_kernel(variant):
    bm, bn = VARIANTS[variant]
    header, body = source_parts()
    return mx.fast.metal_kernel(name=f"expert_tile_bm{bm}_bn{bn}",
        input_names=["x", "w", "scales", "biases", "offsets", "params"],
        output_names=["y"], header=header, compile_options={"math_mode": "safe"}, source="""
        const uint3 tid = threadgroup_position_in_grid;
        const uint simd_group_id = simdgroup_index_in_threadgroup;
        const uint simd_lane_id = thread_index_in_simdgroup;
        const int M = params[0], N = params[1], K = params[2], num_groups = params[3];
        """ + body)


def apply(x, weight, scales, biases, indices, *, variant, bits, group_size, mode,
          sorted_indices):
    bm, bn = VARIANTS[variant]
    if not (mode == "affine" and sorted_indices and x.ndim == 3 and x.shape[1] == 1
            and x.dtype in (mx.bfloat16, mx.float16) and weight.ndim == 3
            and bits in (4, 8) and group_size == 64 and indices.ndim == 1
            and indices.dtype == mx.uint32 and biases is not None):
        raise ValueError("Unsupported experiment layout")
    m, _, k = x.shape
    e, n, _ = weight.shape
    if not (m == indices.size and n % 64 == 0 and k % 32 == 0):
        raise ValueError("Unaligned experiment layout")
    params = mx.array([m, n, k, e], dtype=mx.int32)
    offsets = offsets_kernel()(inputs=[indices, params], grid=(e + 1, 1, 1),
        threadgroup=(128, 1, 1), output_shapes=[(e + 1,)], output_dtypes=[mx.int32])[0]
    # MLX's conservative upper bound, unchanged except for the selected BM.
    tiles = min(m, (m + bm - 1) // bm + e - 1)
    return matmul_kernel(variant)(inputs=[x, weight, scales, biases, offsets, params],
        template=[("T", x.dtype), ("group_size", group_size), ("bits", bits),
                  ("BM", bm), ("BN", bn), ("BK", 32), ("WM", 1), ("WN", 2),
                  ("transpose", True)],
        grid=(n // bn * 32, tiles * 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(m, 1, n)], output_dtypes=[x.dtype])[0]
