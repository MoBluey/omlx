# Remaining Qwen GPU operations, 4 November 2026

Status: CPU discovery and bounded capture hook ready. GPU measurements pending.
No performance change has been accepted or installed.

## Scope and evidence

Exact source baseline `0713dc8ea7d8e4a5b269a10cdbea4706a0e68240`, MLX
0.32.4 and mlx-lm 0.32.3.dev8+g0d4d3d04f. Worktree status was clean and
both log directions against that intended base were empty before edits.
Main fetched both remotes before creating the intentionally pinned worktree.
The source is deliberately older than fork main because it matches installed r2.

The 3 November Qwen Instruments capture only covered 8.67 seconds of a cold
prefill and exceeded the private swap-write guard. It cannot identify current
decode bottlenecks. Ornith's GPU gaps and Qwen's inclusive forward timers are
not substituted for Qwen kernel evidence.

CPU source inspection found these operations already active:

* `omlx/patches/qwen35_moe_routed_decode.py`: fused router, expert gate/up
  with activation, and down/combine. Row-exact verification uses window
  variants of the same kernels, not the composed ~15-launch MoE path.
* `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/`:
  `language.py` compiles hyper-connections and uses existing fused HC,
  grouped attention projections, q/k preparation and attention kernels.
* `qsa_fast.py`: sparse QSA gathers chronological selected K/V rows; row-exact
  verify in `language.py::_row_exact_gathered_verify` invokes each serial
  row's gathered-QSA computation separately, preserving exact arithmetic.

No proposal disables existing compiler/fusion support or changes precision,
reduction order, weights, cache updates, rollback or recurrent state.

## Bounded actual-input capture

New `scripts/qwen_remaining_gpu_probe.py` exports `install()`. Main owns the
private harness call after the normal compatibility patches are applied.
The probe is inactive unless `QWEN_GPU_PROBE_ARM` names an existing file.
Its contents label a capture, for example `short` or `long`.
`QWEN_GPU_PROBE_OUTPUT` names the private derived JSON output.

The hook intercepts pure operator functions, materialises their already-live
input dependencies, evaluates their normal result, then warms and replays the
same pure operation eight times. It records shapes, dtypes and synchronised
latency. Each output is compared through an exact uint8 bit view. It keeps
at most four samples for each operator, row width and capture label. It never
serialises or retains tensors after the call, copies full KV caches, invokes
mutable attention/PLE methods, or changes any cache state.

Reference plus replay final outputs are capped at 64 MiB. Normal kernel
intermediates still allocate; this cap is not a bound on MLX's allocator cache.
Only one evaluated replay is live at a time. The sampled actual pure operators
are MoE gate/up, down/combine, dense SDPA/gate, sparse SDPA, KV gathers,
indexer scores and block selection. Sparse verify legitimately records
one-row operations while processing its two-row window.

These are micro-operation latency observations under forced synchronisation,
not a model-time decomposition, native GPU shader durations, achieved bandwidth
or end-to-end speedups. Profiling runs must not enter throughput comparisons.
Short and 28K captures are required because sparse QSA only activates past
the configured selection budget.

## At most two designs

1. Expert down/combine grid tiling: on actual two-row verification inputs,
   compare existing four output rows per SIMDgroup with two. The existing
   `_down_rows` contract states each output row retains the same arithmetic
   under regrouping. The probe temporarily changes `_DOWN_ROWS` only for a
   pure replay and restores it in `finally`. Candidate compilation/warm-up
   is excluded from reported timings. Actual output bits must match before
   any opt-in full-model ABBA run is considered.
2. If measured sparse K/V gather cost is material, combine their two pure
   copies into one launch with two outputs. Preserve selected indices, token
   order, cache strides, BF16 values and the existing SDPA consumer. This
   remains a conditional hypothesis, not an implementation or claimed gain.

Only the first design is present as an isolated numerical/microtiming probe.
The second will be stopped without implementation if live evidence does not
support it. No wide parameter sweep or unrelated kernel is in scope.

## Verification so far

Standard-library AST parsing passed under pinned Python with `-BIS`.
The first capture startup exposed an incorrect assertion: compatibility
registration returns false when already applied. The corrected hook accepts
that case and is itself idempotent. Actual isolated `-BIS` preflight applied
compatibility twice, installed the probe twice without double wrapping, and
ran an eight-replay BF16 gather with exact output bits. Derived evidence:
`gpu-probe-preflight.json` in the current trial directory.

No full-model capture has completed yet. Main owns the GPU slot and full-model
controls. Remaining acceptance requires actual capture, bit-exact candidate
evidence and, for a supported candidate, same-resident warmed short/28K ABBA
with 700 output tokens. A winner would additionally need the existing
production quality and independent review gates.

The tile candidate does not mutate a cached plan. `_window_down` constructs
its Metal template on each call, including explicit `RPS` and `M` entries;
the cached kernel factory only captures weight format and shader source.
After restoring `_DOWN_ROWS`, a further baseline replay checks output bits.
Direct imports of the gathered-QSA entry function do not bypass the probes:
its internal `_gather_kv_rows`/`_decode_qsa_sdpa`/indexer references resolve
against the wrapped `qsa_fast` module globals. Dense attention uses the
`attn_fused` module directly.
