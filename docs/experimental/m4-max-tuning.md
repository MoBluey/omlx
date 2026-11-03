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
