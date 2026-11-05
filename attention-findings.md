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

## Qualified runtime activation - 6 November 2026

The masked winner alone is retained on pinned idle-cache source
`0a33621d7fc180057e1934ed7ede7a07fb4c3d51`. The numerical implementation is
unchanged from `e2a79c9028ce819bc2a33dcb70160c0437688385`; no preparation,
index-bank or selection-barrier experiment is included.

Set `OMLX_QWEN4_MASKED_S=1` before process startup to enable the qualified
s-class path. Unset, empty and every other value leave it disabled. The
existing GPU, shape, context and fallback checks remain unchanged. The managed
Qwen runtime owns the opt-in; the package remains default off.

The first successful s-class dispatch logs `Qwen M4 masked attention active`
at INFO once per process. A kernel failure disabling the enabled s-class path
logs `Qwen M4 masked attention disabled after kernel failure` at WARNING.

Focused real-M4 verification passed: 41 checks in 5.53 seconds, including
environment activation/scope, partition boundaries, bit-exact outputs,
independent strides, failure fallback, verification-cache rollback and
continuation and one-time success/failure logging. Tests used Python 3.12.15
with `-BIS`, the sealed r2 dependency
closure and isolated pytest dependencies; no runtime installation or full
model load was performed here.
