# Qwen prompt-prefill PLE row-deduplication trial

Disposition: archive-only, default off. No useful full-model TTFT gain established.

## Evidence and scope

The installed-source baseline already announces the first prefill chunk and
gathers the next chunk on one bounded assembly worker. The mmap reader already
deduplicates pages and uses at most 48 page-read workers. This trial changes
only prompt row assembly, not generation read-ahead or index reuse.

Main's `../../qwen-profile-a/instrumentation.json` supplied the screening evidence:

- First 2.4K prompt: TTFT 3.858 s, exposed PLE call 669.98 ms, synchronous
  assembly 663.65 ms; 39,088 rows / 22,248 unique rows.
- Different warm-page 2.4K prompt: TTFT 3.077 s, PLE call 45.90 ms, assembly
  37.52 ms. This is not a controlled cold/warm timing comparison.
- First 28K prompt: TTFT 44.256 s, background assembly 4.002 s across 8 calls,
  exposed call 504 ms, assembly-future wait 435 ms; 7/8 prefetch hits.
- Repeated 28K: TTFT 41.356 s, background assembly 496 ms, exposed call 136 ms,
  future wait 70 ms; 7/8 prefetch hits. Existing overlap hides most read work.

Nested phase measurements must not be added together. Short-prompt future
counts include nested page-worker waits, not only the assembly future.

## Candidate

`OMLX_QWEN4_PLE_PREFILL_DEDUP_TRIAL=1` enables one bounded candidate. For
prompt shapes with more than two token positions, `numpy.unique` selects each
global row once; the unchanged assembly code reads the uniform dense/affine
families; inverse indices restore the original raw row order before the
unchanged upload/dequantisation/scaling path. Decode and two-row verification
keep the original assembler. The switch is read at module import and defaults
off. Mixed-layout fallback, pending-key matching, bounded queues/workers,
ownership guards and cleanup are unchanged.

Each embedding exposes `prefill_dedup_stats`: calls, rows, unique_rows and
failures. The candidate adds only chunk-sized unique/inverse/buffer storage;
it never materialises the table. No precision, token history or cache mutation
changes were made. Context7 was unavailable; existing pinned runtime source
and the installed NumPy API were used.

## Checks

Pinned Python 3.12.15, `-BIS`, `sys.dont_write_bytecode=True` before source/seal
paths, MLX 0.32.4, sealed oMLX dependencies. Pytest plugin autoload disabled;
`-p no:cacheprovider`.

- CPU: 19 passed, including existing real-fork/reader/load cleanup tests and
  three new raw BF16/packed-affine row, failure and prompt-only dispatch checks.
- GPU: 34 passed in 6.85 s using the focused selection below. Dense BF16/F16
  and affine 4/5-bit output tests run both candidate and baseline. Existing
  page-read failures, displaced-prefetch cancellation/draining, n-gram hashing,
  QSA cache restore and PLE rollback checks pass with the trial enabled.
- GPU: one additional actual disk-backed tiny-model test passed in 5.37 s.
  Baseline/candidate prefill logits, restored-cache verification logits,
  accepted-prefix rollback, continuation logits and every exported cache array
  match exactly. Candidate engagement is asserted and failures remain zero.
- Syntax and `git diff --check` pass.

GPU selector: `disk_backed or disk_ple or ngram_prefetch or ple_rollback or
qsa_cache_round_trip` in `tests/test_mlx_vlm_qwen4_exp_compat.py`. The extra
test is `test_disk_ple_dedup_prefill_restore_rollback_and_continuation`.

## Bounded real-shard screen

Command (no model/server load, headers/mmap readers and small row buffers only):

```sh
/opt/homebrew/Cellar/python@3.12/3.12.15/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -BIS /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/worktrees/ple/scripts/benchmark_qwen4_ple_prefill_dedup.py /Users/mohameddarwiche/work/p2pit/development/QwenLocal/models/candidates/omlx/jedisct1/Qwen3.8-Flash-Next-Uncensored-oQ4e-100K-MTP /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/worktrees/ple/logs/ple-real-shard-screen.json
```

Actual packed PLE shards, synthetic indices matching the measured short-prompt
duplication ratio: 39,088 rows / 22,248 unique rows, all 128 shards, 3,908,800
output host bytes. Exact raw-family buffers and exact GPU-dequantised rows pass.
Warm-page ABBAABBA median A 14.3562 ms / B 13.5382 ms, a saving of 0.8180 ms
(5.70% of host assembly). Both paired blocks favour B, but the absolute warm
component saving is small and the existing pipeline usually hides it.

The first observed A took 499.282 ms before the warm comparison. This is not
a cold A/B result and cannot establish a cold-page improvement. No cache purge
or model-residency change was used. Peak process RSS 1,508,917,248 bytes.
System Swapouts stayed 13,000,271 pages (16,384-byte pages) throughout both GPU
test processes, the component process and its shutdown: zero new swap writes.

Raw timing/parity/engagement evidence is in `logs/ple-real-shard-screen.json`;
GPU checks are in `logs/ple-gpu-tests.log` and
`logs/ple-continuation-check.log`. No production/Hermes or full-model candidate
ABBA is claimed. No full-model winner, installation, push or merge is justified
by this screen. No second implementation was attempted.

## Final full-model screen

Main subsequently ran an isolated full-model comparison with the unchanged
reviewed candidate source (`language.py` SHA-256
`7d71c7e969ec90662a85200a567861c4d89877979fa910735fad114be328e737`).
Evidence: `../../qwen-fixed-a/analysis.json`, `../../qwen-ple-a/analysis.json`,
`../../matched-comparison.json` and `../../qwen-ple-a/instrumentation.json`.
All eight candidate requests completed normally (`finish_reason=stop`). Every
paired prompt hash, prompt/cache token count, answer hash and reasoning hash
matches the baseline.

Median TTFT, two requests per group:

| Prompt/cache case | Baseline (s) | Dedup (s) |
| --- | ---: | ---: |
| Short cold | 3.2105 | 3.2060 |
| Short reused | 0.9095 | 0.8900 |
| Long cold | 41.0530 | 41.5405 |
| Long reused | 1.4370 | 1.4460 |

The two cold short requests saved only 6/3 ms; the cold long requests regressed
by 203/772 ms. Final cumulative live counters, including warm-up, confirm
engagement: 50 assembly calls, 1,240,800 rows, 656,630 unique rows and zero
failures. These count assembly, rather than only consumed-prefetch hits.

Whole-run host swap writes were 543.125 MiB for the candidate run and zero for
baseline, below the unchanged cumulative 1 GiB guard. This is global host
activity; the comparison does not establish that the candidate caused it.
This small isolated screen establishes no useful speed win and does not
establish a causal long-context slowdown. Reject promotion; retain the
candidate default-off and archive-only. No source change followed the screen.
