Diagnostic invocation (main owns the full-model process)

Use the normal private source_cli and existing harness/settings, adding:

```sh
MLX_MAX_OPS_PER_BUFFER=0
QWEN_METAL_PROFILE_LIBRARY=/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/qwen-idle-expert-20261105/diagnostic/metal_profile.dylib
QWEN_METAL_TIMESTAMPS=<QWEN_PROBE_DIR>/native-kernels.jsonl
```

At the end of private `trial_hooks.install()` (after its existing early-return
branches, or unconditionally before those branches), add only:

```python
import sys
sys.path.insert(0, "/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/qwen-idle-expert-20261105/worktrees/expert/scripts")
if os.environ.get("QWEN_METAL_PROFILE_LIBRARY"):
    import expert_prefill_profile
    expert_prefill_profile.install()
```

Run a fresh request with at least 8193 uncached prompt tokens and normal 4096
chunk size. The hook waits for the second actual VLMModelAdapter 4096-token
model call, synchronises, enables native timestamps, and disables them after
the corresponding `_record_chunk_transient`. Short boundary chunks do not
count. It leaves all values and every model/cache/guard call intact. It records
the operation shapes and native-clock timing window. This is a diagnostic only:
one primitive per buffer and boundary synchronisation change scheduling. Never
use its wall time or GPU share as an unchanged-runtime performance baseline.
The hook uses `ctypes.CDLL` to load the shim in the server only. Do not set
DYLD_INSERT_LIBRARIES on the runner or clients. Install before model load.

Set `QWEN_CAPTURE_EXPERT_INPUTS=1` only for a separate tensor-capture diagnostic.
It retains the first two routed QuantizedSwitchLinear inputs in the selected
chunk (normally already-fused gate/up and down) and saves each linear's tensors
once after the chunk. Weights are references until export, no concatenation or
dequantisation. Export time and this run must be excluded from timing claims.
Optionally set `QWEN_CAPTURE_EXPERT_BITS=4` to capture the first real Q4 pair;
the model's first pair is Q8. Both mlx-lm and mlx-vlm switch classes are wrapped.

No MLX debug library was used: the build failed because the optional Xcode Metal
Toolchain is absent. The shim links ordinary host Metal/Foundation APIs and uses
the sealed MLX0.32.4 frontend, libmlx and metallib unchanged. Tiny smoke proof:
`diagnostic/smoke-native.jsonl` has real pipeline names and native GPU duration;
`diagnostic/smoke.gputrace` is a valid capture, and `smoke.json` proves exact
arithmetic on M4 Max g16s. Mixed kernel lists are one indivisible duration group.
For another tiny smoke run only, add `QWEN_METAL_PROFILE_INITIAL=1` to enable
timestamps immediately. Full-model runs omit it and use the hook's selected
chunk. Analyse with `scripts/analyse_metal_profile.py INPUT.jsonl OUTPUT.json`.
