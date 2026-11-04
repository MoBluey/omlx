# M4 Max tuning, 3 November 2026

The useful change from this investigation is `MLX_MAX_MB_PER_BUFFER=512`
for the tested Ornith oQ5e bundle. Qwen did not improve against a warmed
stock control. No numerical kernel patch is retained on this branch.

This branch starts at oMLX v0.7.3, commit
`9de79bc39a01c1f3fec1e849903aaf0dd5478d7a`, to match the installed runtime.
All 577 Python source files matched the installed wheel. The comparisons
used its native extensions and sealed dependencies, including MLX 0.32.4.

## Results

Host: Apple M4 Max, 128 GiB unified memory. One model resident at a time,
with background applications left open. MTP remained at fixed depth 1.

| Model and workload | Stock tokens/s | 512 MB tokens/s | Difference |
| --- | ---: | ---: | ---: |
| Ornith, short prompts and prefix reuse | 120.3 | 129.0 | +7.2% |
| Ornith, 28K context and prefix reuse | 77.9 | 84.4 | +8.3% |
| Qwen, short prompts and prefix reuse | 62.9 | 62.4 | -0.9% |
| Qwen, 28K context and prefix reuse | 53.4 | 51.7 | -3.0% |

These are median decode rates for matching prompts and 700-token greedy
outputs with thinking disabled for timing. Each short run has three
cold/reuse pairs. Ornith's 28K candidate pools two runs of three pairs;
Qwen's 28K result has one pair per run. Controls are the faster stock
Ornith run and the warmed stock Qwen run. These samples do not establish
a universal speedup or a prefill improvement.

The tested bundles were:

- `funnygeeker/Huihui-Ornith-1.5-35B-A3B-abliterated-oQ5e-fp16-mtp`
- `jedisct1/Qwen3.8-Flash-Next-Uncensored-oQ4e-100K-MTP`

All 18 matched Ornith candidate comparisons and eight Qwen candidate
comparisons preserved prompt and output hashes. The 68 timed
stock/scheduling requests produced no new host swap writes. A 79K Ornith
cold/reuse pair also completed, but has no matched stock control and is
compatibility evidence only.

Installed Hermes coding qualification passed with production thinking and
sampling: 176.253 seconds, 13 tool calls, no malformed or repeated calls,
verified request controls and an independent passing code oracle. That
run recorded 1.25 GiB of host-wide swap writes; attribution to the setting
is unproven without a matched coding control. The functional pass does
not establish zero-swap behaviour beyond the controlled timing samples.

Qwen's initial apparent improvement was confounded by file-cache warming.
Stock short-context performance rose from 54.0 to 62.9 tokens/s after
warming; the candidate achieved 62.4. The initial stock request read
1,442.9 MiB from storage versus 157.0 MiB for the candidate.

## Applying the result

Set `MLX_MAX_MB_PER_BUFFER=512` in the process environment before MLX
initialises, only for the tested Ornith bundle and runtime. The associated
QwenLocal change applies it in that model's managed child environment.
Its original sealed launcher, model weights, memory guards, context,
cache budgets, production sampling and thinking settings remain intact.

This is a command-buffer scheduling bound, not a model RAM limit. MLX
0.32.4 defaults it to 50 MB for the Max GPU architecture. See the
[environment-variable documentation](https://ml-explore.github.io/mlx/build/html/usage/environment_variables.html)
and the [version-matched device defaults](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/device.cpp).
Other models, quantisations, versions and Macs need their own controls.

## Rejected kernel experiment

Extending the existing row-exact fused MoE path to ordinary two-row Ornith
MTP verification changed 10 of 12 numerical outputs. A narrower variant
kept the shared expert's original two-row arithmetic and fused only the
routed experts. That version passed exact parity, 61 kernel checks,
full-model comparisons and a real Hermes coding fixture.

It did not add a reliable gain beyond scheduling alone. Interleaved block
timings were 0.30685 ms stock and 0.31048 ms patched. The experimental patch
was archived privately and removed. The evidence does not justify adding
that maintenance cost to this fork.

Upstream FP16 fast-path broadening was also inspected. It did not address
the relevant execution paths of these installed bundles. Existing
quantisation eligibility checks were preserved.

Detailed managed-runtime acceptance and rollback notes are recorded in
[QwenLocal's tuning report](https://github.com/MoBluey/QwenLocal/blob/319770a/docs/m4-max-tuning-2026-11-03.md).

## Qwen follow-up: lazy grammar setup

This branch also carries upstream [ae659d39](https://github.com/jundot/omlx/commit/ae659d3979e72b800a4d2a0ad469d315c71a9aa6):
plain requests return before accessing the engine's lazy grammar compiler.
Structured requests still build and use it. The first structured request
pays that setup cost if no earlier structured request has done so.

The actual Qwen tokenizer, with 248,320 vocabulary entries, was measured in
four fresh processes in stock/candidate/candidate/stock order. Plain-request
compiler setup took 444.223 and 446.159 ms on stock, and 0.0055 and 0.0043 ms
on the candidate. Each process subsequently compiled a real JSON schema.
These are component timings, not full-model latency or decode throughput.

All 114 grammar tests passed. An independent review found no blockers.
Stock and candidate Ornith API runs preserved the same greedy plain-text
output and passed strict JSON-schema generation plus six tool-call checks.
The first plain request avoided 464.879 ms of compiler work in the stock
Ornith API comparison. Its total request times are confounded by model and
file-cache loading and are not reported as a speedup.

Full Qwen validation completed on 4 November after the active Claude browser
workload ended. Ten matched generations, including 28K-context reuse,
preserved exact output hashes against both stock controls. Strict JSON,
six tool cases and a real Hermes coding fixture passed. The coding run took
120.350 seconds with nine tool calls, no malformed or repeated calls and no
new swap writes.

QwenLocal installed a separate sealed build containing the grammar patch.
Its normal-router coding fixture passed in 113.583 seconds with eight tool
calls and no malformed/repeated calls or new swap writes. Both installed
model routes passed strict JSON. The original runtime remains available
for rollback. The installed grammar suite passed 114 checks; integration,
integrity and retained-Ornith-tuning checks passed another 35.

Qwen retains its command-buffer defaults. A 25-operation setting was 12.7%
slower on short prompts and 9.0% slower at 28K than warmed stock. A
100-operation load hit the private swap guard and yielded no performance
result; host-state effects were not isolated. Live PLE profiling placed
raw row reads at about 3.8% of the observed decoding interval, so no new
cache or prefetch patch is retained.

Of 40 completed timing requests, 39 recorded no new swap writes. The
patched-source cold load recorded 317.563 MiB, below the abort threshold;
its subsequent requests recorded zero. Earlier stock loads also exceeded
the guard under a busier host. These host-wide events do not establish a
patch-induced regression, and the evidence does not justify a zero-swap
guarantee for model loading.

The retained benefit is first-request setup, not sustained decode speed.
Installed identity, full measurements, limitations and rollback are in
[QwenLocal's follow-up report](https://github.com/MoBluey/QwenLocal/blob/5df664544506e010db413a1aeb3b0b1c1eeb79d9/docs/qwen-m4-max-follow-up-2026-11-03.md).
