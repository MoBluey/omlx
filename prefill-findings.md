# Exact GPU prefill staging screens, 2026-11-05

Disposition: reject both staging designs for both models. No runtime source, switch,
installed files, model weights, precision, quantisation or decode route changed.
The GPU lease is released; the four component processes exited normally.

## Scope and route evidence

Worktree branch `experiment/ttft-four-prefill-20261105`, baseline
`0713dc8ea7d8e4a5b269a10cdbea4706a0e68240`; MLX 0.32.4, installed r2 source.
CPU source inspection covered the Qwen4 gathered QSA prefill, head-dim256 dense
steel attention, hyperconnection prefill, GDN convolution/normalisation, and
quantised/MoE projection routing. The measured device is Apple M4 Max,
`applegpu_g16s`, 137438953472 bytes unified memory. The installed NAX predicate
requires generation17 for the `s` suffix, so NAX-specific kernels are ineligible.
No M5-specific performance claim is applied to M4.

Both models' normal scalar128 GDN prefill reaches the installed pipelined
TB12/DB16 kernel via the cache-aware update's recursive call. Decode and
speculative-history routes have their own gates. Qwen uses BF16, Hk16/Hv48;
Ornith uses FP16, Hk16/Hv32. Qwen has fused prework/norm-gate, whereas Ornith's
prework remains the ordinary path. Actual per-model input captures confirmed
both layouts; this is not an architecture-only inference.

The two screened designs were exactly TB24/DB16 and TB12/DB8. They reuse the
installed `_pipelined_source` generator and change only token staging or rows
per threadgroup, preserving its FP32 operations, state fragments and reductions.
They are explicit CLI-only experiments, with no production engagement switch.
No bare prompt-chunk sweep, approximation or lower precision was used.

## Real4096-token replay timing

Each timing is a median of6 evaluated invocations per arm in ABBA order repeated3
(times include Python/evaluation overhead). Compilation/warmup precedes timing.
The exact raw samples and all checks are in the adjacent four JSON files.

| Model | Design | Baseline ms | Candidate ms | Slower | Peak GPU bytes | Host swap writes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen | tb24 | 4.592521 | 6.042959 | 31.58% | 474,314,408 | 0 |
| ornith | tb24 | 3.793021 | 4.060125 | 7.04% | 377,040,876 | 0 |
| qwen | db8 | 4.646937 | 7.424625 | 59.77% | 474,330,792 | 0 |
| ornith | db8 | 4.022625 | 5.299395 | 31.74% | 377,040,876 | 0 |

TB24 is slower on the real Qwen capture by1.450438ms and on Ornith by0.267104ms;
DB8 is slower by2.777688ms and1.276770ms respectively. The synthetic4096 cases
agree in direction. Small257-row samples have more noise and do not establish a
useful win. The measured real4096 baselines span approximately4.59–4.65ms for
Qwen and3.79–4.02ms for Ornith. Calls captured per4096 chunk were36 and30,
respectively. Multiplying these component times by the call count would suggest
about0.165–0.167s and0.114–0.121s per chunk, but that is an extrapolation from a
single layer replay, not measured whole-model GDN attribution.

## Correctness and limits

All20 cases passed140 bit-exact assertions across output, FP32 recurrent state,
restored-prefix output/state, unchanged saved snapshot, and subsequent baseline
continuation output/state. Each model/design covered64,79,257,4096 synthetic
lengths with a nonzero input state, plus its real4096 production capture.
The snapshot restore uses a prefix of65 tokens, exercising staging tails; the
continuation uses17 tokens. No candidate numerical mismatch, runtime exception
or swap write occurred. Each process applies a cumulative1GiB host swap-write
abort starting before the MLX import. All processes exited and retained no GPU
resident service.

This is a decisive component rejection, not end-to-end qualification. No candidate
was added to the normal launch path, so decode has no changed code path; no new
full-model decode-throughput measurement is claimed. Full Hermes, long-context,
tool/JSON and production-thinking gates are not applicable to promotion because
neither design is proposed for retention. Their component slowdown already
fails the keeper condition. No further designs or broad sweeps were run.

The native Metal System Trace has generic command-buffer labels, not individual
GDN/QMM/SDPA timings. Installed `mlx.core.metal` declarations expose capture
start/stop but no operation profiler. Searching the installed MLX package found
no separate debug/profiler API; inspected libmlx environment strings did not
expose a METAL_DEBUG labelling option. No unsupported profiling capability is
claimed. Main's dedicated Ornith trace sampled9.21s of prefill with8.42s target
GPU execution (91.38%); those are process-level totals. A separate long28K
Ornith diagnostic prefill was23.489s. Neither figure identifies the dominant
operation. Existing unrelated Laya-router GPU work was left running and is a
source of background noise. These results must not be represented as measured
GDN share or proof that all exact GPU prompt optimisation avenues are exhausted.

## Reproduction

The two safetensors captures are outside this branch in the parent evidence
folder; they contain exact evaluated `q,k,v,g,beta,state`, saved after the timed
chunk. The Qwen capture is88605163 bytes. Each command below was run separately
with the exclusive GPU lease, with the displayed output file path:

```sh
/opt/homebrew/Cellar/python@3.12/3.12.15/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -BIS /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/worktrees/prefill/scripts/benchmark_gdn_prefill_staging.py --candidate tb24 --model qwen --capture /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/qwen-profile-a/gdn-inputs.safetensors --output /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/worktrees/prefill/qwen-tb24.json

/opt/homebrew/Cellar/python@3.12/3.12.15/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -BIS /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/worktrees/prefill/scripts/benchmark_gdn_prefill_staging.py --candidate tb24 --model ornith --capture /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/ornith-prefill-capture/gdn-inputs.safetensors --output /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/worktrees/prefill/ornith-tb24.json

/opt/homebrew/Cellar/python@3.12/3.12.15/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -BIS /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/worktrees/prefill/scripts/benchmark_gdn_prefill_staging.py --candidate db8 --model qwen --capture /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/qwen-profile-a/gdn-inputs.safetensors --output /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/worktrees/prefill/qwen-db8.json

/opt/homebrew/Cellar/python@3.12/3.12.15/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -BIS /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/worktrees/prefill/scripts/benchmark_gdn_prefill_staging.py --candidate db8 --model ornith --capture /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/ornith-prefill-capture/gdn-inputs.safetensors --output /Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/ttft-four-20261105/worktrees/prefill/ornith-db8.json
```

Owner checks: CPU AST/source-generation invariants;20 actual-M4 GPU cases and140
exactness checks; all four JSON outcomes independently parsed/asserted; final AST
and diff-whitespace checks. The only edit after GPU runs adds a nonzero process
exit on numerical failure; the evaluated kernel and timing paths are unchanged.
Hindsight was searched before GPU checks and commit. Retrieved older initiative
pages were not used as current performance or installed-state evidence.
No push, merge or installation was performed by this owner.
