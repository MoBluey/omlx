# Remaining Qwen GPU operations, 4 November 2026

Status: current Qwen short/28K capture and fused-gather numerical checks
completed. Down2 tiling is rejected. One default-off fused K/V gather candidate
is undergoing main's same-resident ABBA. Its narrow steady-state coverage is
described below. No change is accepted or installed.

## Scope and provenance

Exact source baseline `0713dc8ea7d8e4a5b269a10cdbea4706a0e68240`, MLX
0.32.4 and mlx-lm 0.32.3.dev8+g0d4d3d04f. The worktree was clean and both
log directions against the intended base were empty before edits. Main fetched
the remotes before creating the deliberately pinned installed-r2 worktree.

The old partial, swapping Qwen Instruments trace cannot establish current
decode bottlenecks. Ornith timings and inclusive forward wall time are not
substitutes. Current source already uses fused routed/shared experts,
row-exact verification, grouped projections and per-module HC optimisation.
The once-only HC incompatibility warning identifies an unsupported instance;
it does not establish whole-model fused-HC inactivity. No existing compiler,
precision, fusion, model weights or recurrent/cache semantics are changed.

## Actual-input capture

`gpu-capture-c/probe.json` contains completed short and 28K captures after
warming the private full model. Main reported zero new swap writes. Earlier
capture-a failed an already-applied-registration assertion; corrected in
`5acd21ad`. Capture-b exceeded the private load-time swap guard before arming.
Those failed loads are not candidate performance evidence.

`scripts/qwen_remaining_gpu_probe.py::install()` stays inert until the absolute
`QWEN_GPU_PROBE_ARM` file exists. Its text labels the capture;
`QWEN_GPU_PROBE_OUTPUT` names the private JSON. Pure operators receive their
actual live inputs/weights, materialise dependencies, warm and replay eight
times. At most four samples per operator/row width/label are taken. No tensors
are retained or exported, no full KV copies are made, and no mutable model or
cache method is replayed. Reference plus replay final outputs are capped at
64 MiB, excluding normal operator intermediates and MLX allocator caching.

The 64 sampled baseline operations and all repeated outputs matched exact
uint8 views. Table values are medians of four per-sample medians, microseconds:

| Pure operation | Rows | Short | 28K |
| --- | ---: | ---: | ---: |
| Fused expert gate/up/top-k | 1 | 153.646 | 182.615 |
| Fused expert down/combine | 1 | 120.448 | 152.062 |
| Fused expert gate/up/top-k | 2 | 224.479 | 240.989 |
| Fused expert down/combine | 2 | 164.386 | 183.333 |
| QSA indexer head scores | 1 | 139.146 | 154.000 |
| QSA block/token selection | 1 | 118.364 | 161.021 |
| One K or V gather | 1 | 141.885 | 154.062 |
| Gathered QSA SDPA | 1 | 154.844 | 163.906 |

These synchronised micro-operation latencies include dispatch/evaluation cost.
They are not shader-only timings, a model-time decomposition, measured memory
bandwidth or predicted end-to-end gains. They must not be summed into shares.
Both captures are already past the 2048-token selection budget. The measured
QSA operators are the gathered sparse arm; the dense hook was not exercised.
These samples do not establish coverage of the backbone's attention path.

### Steady-state coverage correction

Source-path review after the ABBA engagement check identified a narrower
reach than the initial capture suggested:

* `omlx/models/vlm.py::_step_text_positions_min_context` returns infinity
  while `attn_fused.rows_available()` is true. Its `_position_ids_from_starts`
  therefore retains rank-three positions for backbone decode/verify, whose
  `language.py::_row_exact_masked_verify` uses masked QSA and does not call
  the gathered-copy helper. Even without fused rows, the normal threshold is
  32768 cached tokens, above both measured workloads.
* `language.py::Qwen4ExpMTPModule.__call__` explicitly constructs rank-two
  positions for its single draft layer. Singleton draft-head folds can take
  the gathered arm. `batch_generator.py::_chain_next_drafts` folds all committed
  tokens at once; accepted drafts usually produce two-row folds, which do not
  use this singleton helper. The ordinary gathered-prefill gate requires at
  least 16 query rows (`omlx/memory_monitor.py`), excluding those two-row folds.
* `gather-abba-b/trial-arms.jsonl` shows zero factory-call increments in A and
  repeated increments of exactly 113 in warmed short B requests. The native
  logs report 404 cycles and 110 rejected drafts in each matching request.
  This supports effective request-boundary toggling with sparse draft-head
  reach; 113 is consistent with those rejected cycles plus setup. Module
  ownership and each setup call were not separately counted, so that final
  attribution is a source-based inference rather than a per-instance trace.

The helper and its gathered-QSA callers are ordinary Python functions, not
compiled wrappers. The live capture confirmed matching BF16 cache and int32
index eligibility. No evidence supports a captured false flag or silent
fallback as the explanation for the low count. The kernel factory is called
on each eligible helper invocation even though the underlying kernel factory
is cached, making main's counter meaningful.

The bounded capture proves actual execution and copy cost in this narrow
arm. It does not profile the dominant masked backbone attention, all phases,
or complete intermediate memory traffic. Expected full-model headroom is
correspondingly limited. No third design or runtime-path broadening is added.

## Two bounded designs

1. Down/combine two-row tiling changed output rows per SIMDgroup from four to
   two only in pure replay. All eight candidate layer/input samples and all
   restored-baseline samples matched bits. Median-of-medians latency worsened
   from 164.386 to 174.469 us short and 183.333 to 191.073 us at 28K. This
   candidate is rejected; it does not proceed to full-model ABBA. RPS is an
   explicit per-invocation template argument and no cached plan was mutated.
2. Fused K/V gather targets the observed two independent copies. Actual shapes
   were BF16 `[1,2,2443/2444/28287/28288,256]` with selected widths 2048/2051.
   `qsa_fast.py::_gather_decode_kv_pair` uses one two-output copy launch, keeps
   chronological/duplicate/negative valid index semantics, and reads independent
   K/V and index strides. `ensure_row_contiguous=False` avoids full cache copies.
   Only batch-one BF16, two K/V heads and dimension 256 engage. Empty inputs,
   foreign shapes/dtypes/devices and build/initial-launch failures preserve
   the old independent gathers. No arithmetic, attention or cache update changes.

The candidate's scalar request-boundary toggle is
`mlx_vlm.models.qwen4_exp.qsa_fast._FUSED_DECODE_KV_GATHER`, default `False`.
The flag only changes the singleton gathered-QSA copy consumer. Synthetic
rank-two row-exact verification also reaches it, but the normal served
backbone uses the masked path described above. Main should assert engagement
through `_FUSED_DECODE_KV_GATHER_PROVEN` and that
`_FUSED_DECODE_KV_GATHER_DISABLED` remains false. First engagement evaluates
outputs eagerly to surface a build/launch failure safely; warm both ABBA arms.

The initial fused-copy grid did 64-bit division per element and was slower.
A single correction maps dimension/token/head directly onto the three grid
axes, removing that unnecessary integer work. This is the same gather design;
no additional algorithm, precision change or parameter sweep was introduced.

## Numerical/state checks and microcheck

24 focused checks passed on the final grid kernel in 6.03 seconds:

* 16 actual-width/tail cases span short and long contexts, 0..3 tails,
  reserved capacity, different K/V head/token/dimension strides and non-unit
  index strides. Raw BF16 encodings include NaN payloads, signed zero and
  infinities. Every output matches raw uint8 bits and asserts fused engagement.
* Toggle-on/off/restored output matches, including negative valid indices,
  duplicates and nonchronological order.
* Five empty/unsupported cases use the old path, and a deliberate build
  failure verifies fallback and candidate disablement.
* Actual-geometry quantised attention runs two-row verification, trims a
  rejected row and continues decode. Outputs and full visible K/V/indexer
  state remain bit-exact between off/on; the candidate is proven engaged.

Commands use pinned Python `-BIS`, set `sys.dont_write_bytecode=True` before
adding this source and sealed-r2 dependencies, append the supplied pure pytest
paths, set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, then invoke
`pytest.main(['-q','-p','no:cacheprovider', '<worktree>/tests/test_qwen4_decode_kv_pair.py'])`.
Evidence: `gpu-pair-numeric-grid.log`. Earlier initial-grid checks are preserved
in `gpu-pair-numeric.log`.

The final bounded same-input pair microcheck uses synthetic raw BF16 bits and
exact captured shapes, reserved-capacity views and two ABBA blocks. Every arm
is warmed; each block has 16 evaluated pair operations per arm occurrence.
All four shapes assert fused engagement, exact on/restored output and no
fallback. Median latency, microseconds:

| Keys / selected rows | Existing pair | Fused pair |
| --- | ---: | ---: |
| 2443 / 2051 | 168.771 | 146.000 |
| 2444 / 2048 | 164.500 | 146.042 |
| 28287 / 2051 | 149.917 | 132.562 |
| 28288 / 2048 | 148.312 | 124.438 |

Evidence: `gpu-pair-microcheck-grid.json`; the slower initial grid is retained
as `gpu-pair-microcheck.json`. This motivates full-model measurement and does
not establish an end-to-end gain. Main owns warmed short/28K 700-token ABBA,
file edits, any winner's production quality checks and fresh review. No push,
merge, installation or production configuration change is authorised here.
