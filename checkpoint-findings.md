# Prefill checkpoint overhead screen

Disposition: archive-only. Reject evaluation batching from measured upper
bounds before implementation. Runtime source is unchanged.

Base: `0713dc8ea7d8e4a5b269a10cdbea4706a0e68240`, the installed sealed runtime.
Branch: `experiment/ttft-four-checkpoint-20261105`.
Evidence base `B`:
`/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105`.

## Candidate and existing safeguards

`_serialize_extracted` batch-evaluates payload tensors before per-tensor
`_extract_tensor_bytes` repeats evaluation. BF16 also evaluates each uint16
buffer view. One bounded candidate was identified: retain initial payload
evaluation, batch BF16 views, then copy exact bytes without redundant evals.

The store already writes bytes asynchronously. Moving lazy MLX work to the
writer violates owner-stream safety. Immutable containers, exact byte copies,
checkpoint token positions, sliceable-KV filtering, pending-byte reservations,
duplicate coalescing, fallbacks and every reusable checkpoint must remain.

## Actual baseline results

Read-only phase snapshots: `B/{qwen,ornith}-profile-a/instrumentation.json`.
Paired probes: `B/MODEL-profile-a/MODEL-ttft-LABEL.json`.
Long prompts have 28,280 uncached tokens. Nested durations overlap.

| Model / cold prompt | TTFT s | Captures ms | All save ms | All eval ms | Eval ceiling % TTFT |
|---|---:|---:|---:|---:|---:|
| Qwen short 1 | 3.858 | 13.078 | 13.046 | 0.867 | 0.02246 |
| Qwen short 2 | 3.077 | 16.482 | 16.415 | 1.101 | 0.03579 |
| Qwen long 1 | 44.256 | 39.221 | 49.277 | 8.255 | 0.01865 |
| Qwen long 2 | 41.356 | 41.185 | 49.530 | 8.101 | 0.01959 |
| Ornith short 1 | 1.306 | 6.643 | 6.603 | 0.030 | 0.00231 |
| Ornith short 2 | 1.233 | 4.922 | 4.895 | 0.020 | 0.00160 |
| Ornith long 1 | 23.239 | 25.938 | 33.776 | 0.764 | 0.00329 |
| Ornith long 2 | 24.329 | 27.914 | 32.138 | 0.850 | 0.00349 |

Qwen long-1: 7 capture calls; 8 saves/serialisations; 592 exports taking
44.726 ms; 896 evaluations = 592 payloads + 296 BF16 views + 8 initial batches.
Ornith long-1: 7 captures; 8 saves/serialisations; 480 exports taking 29.728 ms;
488 evaluations = 480 payloads + 8 batches, with no BF16-view evaluation.

Even impossible deletion of every evaluation saves under 0.036% of measured
TTFT. This includes required initial evaluation and captures after first token,
so the proposed batching saves less. Deleting the entire long-request save path
would bound savings at 0.12% for Qwen and 0.15% for Ornith. Required immutable
byte copies dominate export. No useful implementation screen is justified.

The first long reasoning/answer times were 44.255963/53.523917 s for Qwen and
23.239536/31.429682 s for Ornith. Qwen long-1 diagnostic GDN capture added
25.366 ms; excluding that cost does not change the decision. Reused Qwen long
snapshot totals were 8.72/11.03 ms of 1.270/1.708 s TTFT; batching addresses only
part of those totals. Prefix reconstruction belongs to the separate RAM trial.

Selected probes report zero new swap writes and no monitor error. Qwen staging
after request 8 had pending bytes 0, peak 115,642,384 bytes, backpressure
0.023668 ms and zero SSD errors. These are baseline observations, not candidate
parity or memory claims. Larger contexts remain unmeasured.

## Verification and reproduction

`checkpoint-screen.json` retains eight exact phase/probe records and arithmetic.
Checked using Python 3.12 `-BIS` with standard-library-only JSON parsing. No GPU
imports, model loads, candidate exactness tests or full-model A/B were needed:
no runtime code changed. No install, push or merge was performed.

Context7 was unavailable; matching MLX 0.32.4 declarations document `view` and
`eval` in installed `mlx/core/__init__.pyi`. Hindsight was searched before
commit and its initiative read. Current source and actual profiles govern.

Inspected scheduler lines 8670-9030, store save lines 221-355, serialisation
lines 1212-1355 and paged SSD byte export lines 949-982, using absolute-path
`rg -n` and `sed -n` commands. Arithmetic verification command:

```sh
/opt/homebrew/Cellar/python@3.12/3.12.15/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -BIS - <<'CHECK'
import json
from pathlib import Path
p = Path('/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/worktrees/checkpoint/checkpoint-screen.json')
records = json.loads(p.read_text())['records']
assert len(records) == 8
for r in records:
    for phase, field in [('save', 'optimistic_all_save_pct_ttft'),
                         ('snapshot_mx_eval', 'optimistic_all_eval_pct_ttft')]:
        seconds = r['metrics']['background:' + phase]['seconds']
        assert abs(r[field] - 100 * seconds / r['result']['ttft_s']) < 1e-12
    assert r['vm_delta_bytes']['Swapouts'] == 0
    assert r['monitor_error'] is None
print('8 baseline records: arithmetic and probe checks pass')
CHECK
```
