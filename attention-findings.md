# M4 attention trials

Base installed source: `0713dc8ea7d8e4a5b269a10cdbea4706a0e68240`.
Pinned MLX 0.32.4 and mlx-lm 0.32.3.dev8+g0d4d3d04f; Apple M4 Max,
architecture `applegpu_g16s`. Main owns live full-model runs and GPU allocation.
Both source toggles default off; no installed runtime is changed.

## Independent contracts

| Candidate | Request-boundary toggle | Actual dispatch count | Failure state |
| --- | --- | --- | --- |
| masked-s | `qsa_fast.MASKED_S_ENABLED` | `qsa_fast.MASKED_S_CALLS` | `qsa_fast._DECODE_SDPA_DISABLED` |
| prep-fusion | `attn_fused.PREP_FUSION_ENABLED` | `attn_fused.PREP_FUSION_CALLS` | `attn_fused.PREP_FUSION_DISABLED` |

Module prefix: `mlx_vlm.models.qwen4_exp`. A disables both; a B enables only
its candidate. Counts represent successful actual candidate operator returns,
not enable flags or readiness probes. Main should record per-request counter
differences and assert failure states stay false. Off branches retain the
original d-only complete attention plan and existing text-position policy.

## Masked-s

Source reference: saved upstream `773e2d6.patch`. The saved official MLX
0.32.4 `scaled_dot_product_attention.cpp` confirms the s-class two-pass plan
for this 12:1 GQA singleton geometry: 256 partitions for 8193..32768 keys,
512 for 32769..65536, and 1024 above 65536. Below or at 8192 keys the trial
keeps MLX. Existing d-class rules are unchanged. Kernel arithmetic, mask
selection order, reductions, tail handling and stride-aware loads are reused.

New focused coverage includes exact partition boundaries, actual s-class
bit comparisons against MLX for QSA/all/sparse/empty-selected masks, independent
cache and mask strides, explicit off/build-failure controls, and two-row
verification followed by accepted-prefix trim and continuation. GPU execution
is pending the exclusive slot. Only static/AST checks have run so far.
