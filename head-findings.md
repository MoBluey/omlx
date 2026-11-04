# Qwen BF16 head trial, 4 November 2026

Decision: reject this exact paired-GEMV design. It produced no useful pure-head
improvement. Main accepted this bounded screening outcome; no full-model head
trial, model-route replacement, installation or precision change is needed.

## Identity and scope

Base: installed `0713dc8ea7d8e4a5b269a10cdbea4706a0e68240`, MLX 0.32.4,
mlx-lm 0.32.3.dev8+g0d4d3d04f; actual M4 Max `applegpu_g16s`.
The worktree was clean and both log directions against the intended installed
base were empty before edits. Main fetched the remotes; this isolated branch
has no tracking branch until main pushes it.

Checkpoint revision: `7d92e74f3e1dda4504e86a58d47716e44c36521a`.
The actual safetensors header identifies `language_model.lm_head.weight` as
BF16 `[248320,2560]`, 1,271,398,400 bytes (1.18408203125 GiB), with untied
embeddings. Only that tensor was mapped/read. No other checkpoint tensor or
full model was loaded, and no weight value, dtype or quantisation changed.

## Native dispatch and one bounded design

Before implementation, source review checked official version-matched
[MLX matmul dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/matmul.cpp)
and [GEMV reduction](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/gemv.h),
plus the installed `mlx_vlm/models/exact_speculative_verify.py` and
`mlx_vlm/speculative/ops/linear.py`.

Source-derived one-row dispatch is ordinary GEMV: four outputs per SIMDgroup,
32 reduction lanes, four consecutive columns per lane per 128-column step,
with eight SIMDgroups per threadgroup. The installed two-row bias-free dense
verification route instead uses its exact custom GEMV, padding the input-row
grid to eight rows. The numerical replay invokes those installed reference
routes directly; it is not a whole-model shader trace. Native wide GEMV is
implemented in the installed binary; its dot/chunk accumulation is
not assumed equivalent to the serial verification reference.

The single candidate shares each weight load across two independent input
accumulators, packs eight output SIMDgroups per threadgroup, and retains each
row's original column order, FP32 accumulation, shuffle-down tree and BF16
output rounding. It supports the actual head geometry and one/two BF16 rows;
other shapes/dtypes/devices and build/initial-launch failures return to the
stock caller. Independent input/weight strides are read directly with
`ensure_row_contiguous=False`. No weight transpose/copy cache is added.

Private archive seam: `omlx.patches.qwen4_bf16_head.try_bf16_head(x, weight)`.
`BF16_HEAD_ENABLED` defaults false; successful-return counters are
`BF16_HEAD_CALLS[1]`/`[2]`, and failure state is `BF16_HEAD_DISABLED`.
This screening module is deliberately not registered into the serving model:
its measured lack of benefit stopped the design before runtime integration.

## Exactness and measurements

13 focused small-tensor cases passed in 6.63 seconds. They cover both row
widths, independent input/weight strides, zero/random/wide-magnitude inputs,
exact baseline and serial bits, explicit off/unsupported controls, and a
forced failure that disables the candidate without counting successful work.
The first small fixture used too few output rows for the installed reference
helper's eligibility; correcting the fixture to 256 outputs made it exercise
the intended two-row reference. No kernel redesign or parameter sweep followed.

`head-replay.json` contains six actual-weight cases: zero, random and
wide-magnitude input at one/two rows. Every output bit matches both its
installed reference and serial decode. All six assert engagement. Including
warm-up and timing, each row width records 38 successful candidate calls;
no candidate failure occurred.

Same-input warmed ABBAABBA used eight evaluated operations per arm occurrence,
32 samples per arm and row width. Times include dispatch/materialisation and
synchronisation; they are not GPU-counter or whole-server measurements.

| Input rows | Installed reference median, ms | Candidate median, ms |
| --- | ---: | ---: |
| 1 | 2.774583 | 2.792125 |
| 2 | 2.754500 | 2.780896 |

The sample distributions overlap. These results show no useful improvement;
the small median differences are not claimed as precise causal slowdowns.
The native verification baseline already performs well, so fewer source-level
weight loads did not establish a measured gain. No server-speed or bandwidth
saturation claim follows from these operation timings. Stop this design.

## Memory, safety and reproduction

The replay used a read-only NumPy memmap of the head's exact byte range, then
uint16 storage viewed as unchanged BF16 bits. Its analytical tensor footprint
is one 1.184 GiB MLX weight plus up to one head-sized set of mapped file pages,
small inputs and sub-MiB outputs per one/two-row result. This is a bound on
the known tensor/mapping payload, not a measured process peak or allocator
cache bound. No full-shard materialisation or derived persistent weight bank
was created. The head's source file stayed read-only.

The private 1 GiB new host swap-write abort remained active around load,
numerical cases and timing groups. The completed replay recorded zero new
swap writes. The exclusive GPU slot was released immediately after these
checks. No further GPU work is required for this rejected candidate.

Use the pinned Python `-BIS` bootstrap, setting
`sys.dont_write_bytecode=True` before adding this worktree then sealed-r2
site to `sys.path`. Append the supplied pure pytest paths and set
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; run
`pytest.main(['-q','-p','no:cacheprovider', '<worktree>/tests/test_qwen4_bf16_head.py'])`.
For actual weights run `scripts/benchmark_qwen4_bf16_head.py` through that
bootstrap with absolute checkpoint and output-JSON paths. The committed
`head-replay.json` contains every numerical result and timing sample.

Before this archive commit, Hindsight knowledge search and the current
BF16-head/long-context initiative page were read as requested. They supplied
no additional constraint beyond the current brief; source, header and actual
GPU evidence above decide this result. Main owns final report, independent
review and authorised push. No merge or installation was performed.
