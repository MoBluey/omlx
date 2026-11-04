"""Exact staging-only replay screens. Run only while holding the GPU lease.

Bootstrap with the pinned Python's -BIS flags; no model weights are loaded.
The optional capture is q/k/v/g/beta/state from the installed production call.
"""

import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
SITE = Path("/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/runtime/omlx-0.7.3-m4-max-r2-sealed-site")


def swap_bytes():
    raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+)", raw)[1])
    return page * int(re.search(r"Swapouts:\s*(\d+)", raw)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("tb24", "db8"), required=True)
    parser.add_argument("--model", choices=("qwen", "ornith"), required=True)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert args.output.is_absolute()
    first_swap = swap_bytes()

    def guard():
        delta = swap_bytes() - first_swap
        if delta > 1024**3:
            raise RuntimeError("Stopped at cumulative 1 GiB host swap-write guard")
        return delta

    sys.path[:0] = [str(SOURCE), str(SITE)]
    import mlx.core as mx
    from omlx.custom_kernels.attention import gated_delta as gd

    dtype, hv = (mx.bfloat16, 48) if args.model == "qwen" else (mx.float16, 32)
    tb, db = (24, 16) if args.candidate == "tb24" else (12, 8)
    # The source generator uses these constants solely for staging geometry.
    # Restore immediately; the baseline retains its unchanged TB12/DB16.
    original_db, original_threads = gd._PIPE_DB, gd._PIPE_THREADS
    try:
        gd._PIPE_DB, gd._PIPE_THREADS = db, db * 8
        source = gd._pipelined_source(tb)
    finally:
        gd._PIPE_DB, gd._PIPE_THREADS = original_db, original_threads
    kernel = mx.fast.metal_kernel(
        name=f"omlx_gdn_staging_tb{tb}_db{db}",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"], source=source, header=gd._HEADER,
    )

    def candidate(q, k, v, g, beta, state):
        b, length, hk, dk = q.shape
        vh, dv = v.shape[2:]
        return kernel(
            inputs=[q, k, v, g, beta, state, length],
            template=[("InT", q.dtype), ("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", vh)],
            grid=(db * 8 * (dv // db), vh, b), threadgroup=(db * 8, 1, 1),
            output_shapes=[(b, length, vh, dv), (b, vh, dv, dk)],
            output_dtypes=[q.dtype, mx.float32],
        )

    def inputs(length, seed):
        def normal(shape, offset):
            return mx.random.normal(shape, key=mx.random.key(seed + offset))
        q, k = (normal((1, length, 16, 128), i) for i in (0, 1))
        q = q * mx.rsqrt(mx.sum(q * q, -1, keepdims=True) + 1e-6) * 128**-0.5
        k = k * mx.rsqrt(mx.sum(k * k, -1, keepdims=True) + 1e-6)
        return (q.astype(dtype), k.astype(dtype), normal((1, length, hv, 128), 2).astype(dtype),
                mx.exp(-mx.random.uniform(0.01, 3.0, (1, length, hv), key=mx.random.key(seed + 3))),
                mx.sigmoid(normal((1, length, hv), 4)), normal((1, hv, 128, 128), 5) * 0.1)

    def exact(left, right):
        integer = mx.uint32 if left.dtype == mx.float32 else mx.uint16
        return bool(mx.array_equal(left.view(integer), right.view(integer)).item())

    baseline = gd.gated_delta_pipelined
    result = {"model": args.model, "candidate": args.candidate, "tb": tb, "db": db,
              "device": mx.device_info(), "correctness": [], "timings": []}
    cases = [(f"synthetic-{length}", inputs(length, 710 + length)) for length in (64, 79, 257, 4096)]
    if args.capture:
        assert args.capture.is_absolute()
        saved = mx.load(str(args.capture))
        captured = tuple(saved[name] for name in ("q", "k", "v", "g", "beta", "state"))
        assert captured[0].dtype == dtype and captured[0].shape[2:] == (16, 128)
        assert captured[2].shape[2:] == (hv, 128)
        assert captured[5].dtype == mx.float32
        cases.append(("production-capture", captured))
    for label, values in cases:
        guard()
        mx.eval(*values)
        expected = baseline(*values)
        actual = candidate(*values)
        mx.eval(expected, actual)
        record = {"case": label, "shape": list(values[0].shape),
                  "output_exact": exact(expected[0], actual[0]),
                  "state_exact": exact(expected[1], actual[1])}
        # Restore the immutable prefix state, then continue both routes from it.
        split = min(65, values[0].shape[1] - 1)
        prefix = tuple(x[:, :split] for x in values[:5]) + (values[5],)
        saved_state = baseline(*prefix)[1]
        mx.eval(saved_state)
        saved_bits = saved_state.view(mx.uint32) + mx.zeros_like(saved_state.view(mx.uint32))
        mx.eval(saved_bits)
        suffix = tuple(x[:, split:] for x in values[:5]) + (saved_state,)
        resumed_expected, resumed_actual = baseline(*suffix), candidate(*suffix)
        mx.eval(resumed_expected, resumed_actual, saved_bits)
        record.update(restore_output_exact=exact(resumed_expected[0], resumed_actual[0]),
                      restore_state_exact=exact(resumed_expected[1], resumed_actual[1]),
                      snapshot_unchanged=bool(mx.array_equal(saved_bits, saved_state.view(mx.uint32)).item()))
        continuation = inputs(17, 191)
        after_expected = baseline(*continuation[:5], expected[1])
        after_actual = baseline(*continuation[:5], actual[1])
        mx.eval(after_expected, after_actual)
        record.update(continuation_output_exact=exact(after_expected[0], after_actual[0]),
                      continuation_state_exact=exact(after_expected[1], after_actual[1]))
        result["correctness"].append(record)
        print(json.dumps(record), flush=True)
        if not all(value for key, value in record.items() if key.endswith("exact") or key == "snapshot_unchanged"):
            break
        if values[0].shape[1] >= 257:
            times = {"baseline": [], "candidate": []}
            for run in (baseline, candidate) * 2:
                mx.eval(run(*values))
            for arm in ("baseline", "candidate", "candidate", "baseline") * 3:
                guard()
                run = baseline if arm == "baseline" else candidate
                mx.synchronize()
                start = time.perf_counter_ns()
                mx.eval(run(*values))
                times[arm].append((time.perf_counter_ns() - start) / 1e6)
            timing = {"case": label, "samples_ms": times,
                      "baseline_median_ms": statistics.median(times["baseline"]),
                      "candidate_median_ms": statistics.median(times["candidate"])}
            result["timings"].append(timing)
            print(json.dumps(timing), flush=True)
    result.update(peak_gpu_bytes=mx.get_peak_memory(), active_gpu_bytes=mx.get_active_memory(),
                  swap_write_bytes=guard())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if any(not value for record in result["correctness"] for key, value in record.items()
           if key.endswith("exact") or key == "snapshot_unchanged"):
        raise SystemExit("Exactness check failed; timing screen rejected")


if __name__ == "__main__":
    main()
