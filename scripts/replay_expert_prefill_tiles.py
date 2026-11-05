"""Real captured expert replay; launch only with main's explicit GPU lease."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def swapout_bytes():
    raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=10)
    page = int(re.search(r"page size of (\d+) bytes", raw).group(1))
    return int(re.search(r"^Swapouts:\s+(\d+)", raw, re.M).group(1)) * page


SWAP_BASELINE = swapout_bytes()


def check_swap():
    latest = swapout_bytes()
    if latest - SWAP_BASELINE > 1024**3:
        raise RuntimeError("Component replay exceeded 1 GiB cumulative host swap writes")
    return latest


SOURCE = Path(__file__).resolve().parents[1]
SITE = Path("/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/runtime/omlx-0.7.3-m4-max-r2-sealed-site")
sys.path[:0] = [str(SOURCE), str(SITE), str(SOURCE / "scripts")]
import mlx.core as mx
import expert_prefill_tiles as tiles


def replay(tensors, metadata):
    args = [tensors[key] for key in ("x", "weight", "scales", "biases", "indices")]
    x, weight, scales, biases, indices = args

    def baseline():
        return mx.gather_qmm(x, weight, scales, biases, rhs_indices=indices,
            transpose=True, **metadata)

    mx.eval(*args)
    reference = baseline()
    mx.eval(reference)
    cases = {"baseline": baseline}
    checks = {}
    for variant in tiles.VARIANTS:
        invoke = lambda variant=variant: tiles.apply(*args, variant=variant, **metadata)
        try:
            result = invoke()
            mx.eval(result)
            exact = bool(mx.array_equal(reference.view(mx.uint16), result.view(mx.uint16)).item())
            checks[variant] = {"bits_equal": exact,
                "max_abs_error": float(mx.max(mx.abs(reference.astype(mx.float32) - result.astype(mx.float32))).item())}
            if exact:
                cases[variant] = invoke
        except Exception as error:
            checks[variant] = {"bits_equal": False, "error": str(error)}
        check_swap()
    samples = {name: [] for name in cases}
    for invoke in cases.values():
        for _ in range(3):
            mx.eval(invoke())
            check_swap()
    names = list(cases)
    for repeat in range(12):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        if repeat % 2:
            order.reverse()
        for name in order:
            mx.synchronize()
            started = time.perf_counter()
            mx.eval(cases[name]())
            mx.synchronize()
            samples[name].append(time.perf_counter() - started)
            check_swap()
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {"x_shape": x.shape, "weight_shape": weight.shape, "dtype": str(x.dtype),
        "metadata": metadata, "exact_checks": checks, "samples_seconds": samples,
        "median_seconds": medians,
        "speedup": {name: medians["baseline"] / value for name, value in medians.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if "MLX_MAX_OPS_PER_BUFFER" in os.environ or "MLX_MAX_MB_PER_BUFFER" in os.environ:
        raise RuntimeError("Candidate timing requires the unchanged default buffer policy")
    report = {"mlx_version": mx.__version__, "source": str(SOURCE),
        "candidate_count": 2, "serving_integration": False, "cases": []}
    for path in sorted(args.capture.glob("expert-*.safetensors")):
        check_swap()
        metadata = json.loads(path.with_suffix(".json").read_text())
        result = replay(mx.load(str(path)), metadata)
        result["capture_file"] = str(path.resolve())
        report["cases"].append(result)
        report["swap_guard"] = {"baseline_bytes": SWAP_BASELINE,
            "latest_bytes": check_swap(), "limit_bytes": 1024**3}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        mx.clear_cache()
    if not report["cases"]:
        raise RuntimeError("No captured expert fixture files")
    print(json.dumps({"cases": len(report["cases"]), "output": str(args.output)}))
