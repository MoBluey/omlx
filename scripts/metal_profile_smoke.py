"""Small real-Metal feasibility proof; invoke with the sealed Python -BIS."""
import json
import ctypes
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
native = ctypes.CDLL(os.environ["QWEN_METAL_PROFILE_LIBRARY"])
sys.path.insert(0, "/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/runtime/omlx-0.7.3-m4-max-r2-sealed-site")
import mlx.core as mx

output = Path(sys.argv[1]).resolve()
a = mx.ones((512, 512), dtype=mx.bfloat16)
b = mx.ones((512, 512), dtype=mx.bfloat16)
mx.eval(a, b)
mx.metal.start_capture(str(output / "smoke.gputrace"))
started = time.perf_counter()
for _ in range(8):
    c = mx.matmul(a, b)
    mx.eval(c)
mx.synchronize()
duration = time.perf_counter() - started
mx.metal.stop_capture()
assert bool(mx.all(c == 512).item())
(output / "smoke.json").write_text(json.dumps({
    "version": mx.__version__, "device": mx.metal.device_info(),
    "wall_seconds": duration, "exact": True,
    "diagnostic_only": True, "matmul_repeats": 8,
}, indent=2))
print("capture and arithmetic passed")
