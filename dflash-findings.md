# Cheaper exact DFlash verification trial

Scope: exact installed source `0713dc8ea7d8e4a5b269a10cdbea4706a0e68240`, native adapter ported from `c3353b3d527b7212e6504db414980217928c379c` as `6dcd6ff0`. Default fixed-depth-1 MTP is unchanged. No installed runtime/settings changes. Full-model timings belong to main.

## Candidates and feasibility

Only three settings are proposed: fixed block 2, fixed block 3, adaptive block 3. A block includes its confirmed anchor, hence these have at most one/two draft tokens.

```text
OMLX_QWEN4_DFLASH_TRIAL=1
mtp_enabled=false
dflash_enabled=true
dflash_draft_model=/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/qwen-trials-20261104/dflash-drafter
dflash_draft_quant_enabled=false
dflash_block_size=2 | 3
dflash_verify_mode=dflash | adaptive
```

Keep temperature 1, top_p .95, top_k 20, medium thinking, target weights/quantisation, context and admission/abort limits unchanged. The pinned drafter is BF16, trained block 8, captures layers 5/15/25/35/45. Native target cap remains four rows. Loader `resolve_block_size` accepts 2 and 3 without source changes. The generic mlx-vlm generator's `DFlash2DraftModel.dflash_min_block_size=3` marker does not govern this batched drafter; its actual `draft_block` accepts positive proposal lengths. Its kernel-size-2 dynamic convolutions require testing the two-row block, covered below.

`adaptive` uses the existing controller and clips both drafts and their sparse q distributions before verification. It still computes the selected block's full drafter forward, so savings concern target rows; it may also select a depth-0 one-row serial step. Native one-row capture avoids speculative transactions and observes committed hidden states.

The shared `_call_backbone_impl` arms row-exact QMM for this target at batch=1 for every verify length. Native Qwen4 projections, HC capture, recurrent transactions, PLE rollback and QSA index rollback are retained. At production temperature the chain route calls `_stochastic_verify_tokens` with the target sampler and conditional selector q. Top-k 20 selects its sparse Leviathan route, which uses filtered p/q acceptance and max(p-q,0) correction. The native adapter is single-request only. EOS and output limits clip the accepted prefix before cache rollback and discard an incompatible predraft.

## Verification

Completed CPU-only adapter check: 9 passed (NumPy execution of production capture/transaction/gate methods). GPU numerical checks completed under the exclusive slot: 27 passed in 7.94 seconds. Slot released immediately afterwards. Log: `/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/qwen-three-trials-20261104/dflash-numerical-tests.log`.

Passing coverage:

- Native BF16 target: verify lengths 2/3/4, every accepted prefix; bitwise target logits, all cache state arrays, QSA index arrays and next-token continuation. Captured hidden shape checked for each length.
- Smaller blocks 2/3 against the checkpoint's `draft_block` oracle for short and full draft windows.
- Predraft adoption/discard versus drafting only committed context for block 2/3/4, across ring wrap.
- Fused selector proposal lengths 1/2/3 at production sampler settings, matching conditional q densities against the ordinary candidate sampler.
- Sparse production acceptance: every rejection/full-accept prefix at depths 1/2, clipped q alignment, residual and bonus densities, proposal support outside target support, and the target marginal identity.
- Actual sampled generator route for fixed block2, fixed block3, adaptive block3: EOS and remaining-token truncation with cache assertions.

Exact isolated command used after the GPU grant:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/homebrew/Cellar/python@3.12/3.12.15/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -BIS -c 'import sys; sys.dont_write_bytecode=True; sys.path[:0]=["/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/qwen-three-trials-20261104/worktrees/dflash", "/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/runtime/omlx-0.7.3-m4-max-r2-sealed-site", "/Users/mohameddarwiche/Library/Python/3.14/lib/python/site-packages", "/opt/homebrew/lib/python3.14/site-packages"]; import pytest; raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "tests/test_qwen4_dflash_native.py", "tests/test_qwen4_dflash_smaller_blocks.py", "tests/test_dflash_batched.py::test_predraft_adopt_matches_drafting_committed_rows", "tests/test_dflash_batched.py::test_fused_selector_matches_candidate_sampling"]))'
```

Working directory: this worktree. Capture command output at `/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/qwen-three-trials-20261104/dflash-numerical-tests.log`. Do not use these tiny tests as full-model production qualification.

## Benchmark and instrumentation contract

Main's `run_variant.py qwen TAG --source /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/qwen-three-trials-20261104/worktrees/dflash --dflash --dflash-block 2` selects fixed block2; block3 is analogous. Adaptive requires main's `--dflash-verify-mode adaptive` option. Export the explicit trial environment variable in the launcher environment. No target precision or sampler override is required.

The existing INFO `MTP[...]` summary already reports `cycles`, `accept`, per-depth drafted/accepted counts, depth-0 cycles, emitted sources and `backbone/mtp/sample/cache` milliseconds. Accepted drafts/cycle is `accepts/cycles`; committed generation tokens/cycle is `(accepts+cycles)/cycles` before EOS/length truncation. Existing displayed `tok/cycle` includes activation emissions and should not be mistaken for the latter.

Timing limitation: the drafter queues asynchronous work and predrafts may overlap the verify/acceptance sync. `mtp_head_ms` is host dispatch time, while some draft GPU time lands in `backbone_ms`. These fields are useful cost attribution but are not independent GPU execution times. End-to-end warmed timings remain decisive; isolate draft/verification GPU cost only in a separate synchronised profile run, not by adding synchronisation to timed candidates.
