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

## Prep-fusion

Source reference: saved upstream `7ffa94f.patch`, using the existing exact
`attn_fused.prep_qk` kernel without changing its arithmetic. Independent
`prep_ready` validates only preparation on a tiny input, with its own ready
set and failure state. The normal q/k/v preparation and gathered-QSA paths
share this preparation seam. With the flag off they keep the composed MLX
norm/rotary operations; enabling it only replaces q/k norms and MRoPE.

The opt-in path requires batch one, positive length, integer cache offset,
head dimension 256, canonical rank-two/rank-three position shape, supported
floating dtype, matching ordinary q/k RMS norms and half-split rotary layout.
External position embeddings, other batches/layouts and failed readiness keep
the base preparation. Values, masks, cache append order and attention remain
unchanged. The gate projection still follows its original sigmoid path.

The original `ready`, `rows_available`, `row_plan`, `_fused_rows`,
`_fused_projections_ready` and `_fused_prologue` functions are AST-identical
to installed source. `omlx/models/vlm.py`, including the text-position policy,
is byte-identical. These static assertions passed without importing MLX.
The new independent preparation flag cannot enable complete s-class rows or
alter which attention reduction plan runs.

New numerical coverage is ready for BF16/FP32 raw q/k preparation at rows
1/2/16/300, both position ranks, ordinary RoPE and YaRN; ordinary/gathered
prefill, two-row verification, accepted-prefix rollback and continuation;
explicit off, batch-two, external-position-embedding and readiness-failure
controls. Every active case asserts actual preparation count increments and
no fallback. These GPU checks have not yet run. CPU AST and diff checks pass.

## Pending verification

Run both focused files with the pinned Python `-BIS` bootstrap, setting
`sys.dont_write_bytecode=True` before source/r2 paths, appending supplied pure
pytest paths, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and `-p no:cacheprovider`.
Main grants the only GPU slot before execution, then runs full-model ABBA and
quality qualification only for useful, exact candidates. No performance or
readiness claim is made from static checks alone.
