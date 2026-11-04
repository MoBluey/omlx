"""Opt-in, bounded pure-operator replay on actual warmed Qwen inputs.

Install after the normal compatibility patches. The arm file's text labels a
capture (for example short/long); removing it disables replay immediately.
This deliberately synchronises operators and is not an end-to-end profiler.
No arrays or model parameters are retained or written to disk.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import statistics
import time

_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import mlx.core as mx
    from omlx.patches import qwen35_moe_routed_decode as moe
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    # False also means the normal harness already registered this package.
    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import attn_fused, qsa_fast

    output = Path(os.environ["QWEN_GPU_PROBE_OUTPUT"])
    arm = Path(os.environ["QWEN_GPU_PROBE_ARM"])
    assert output.is_absolute() and arm.is_absolute()
    records, counts = [], {}
    active = False
    # At most four actual layers per operator/row width/capture label. Every
    # replay is evaluated and released before the next, never a graph batch.
    repeats, max_samples = 8, 4
    max_output_bytes = 32 * 1024**2

    def arrays(value):
        if isinstance(value, mx.array):
            return [value]
        if isinstance(value, (tuple, list)):
            return [a for child in value for a in arrays(child)]
        if isinstance(value, dict):
            return [a for child in value.values() for a in arrays(child)]
        return []

    def describe(value):
        if isinstance(value, mx.array):
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "nbytes": value.nbytes}
        if isinstance(value, (tuple, list)):
            return [describe(child) for child in value]
        if isinstance(value, dict):
            return {key: describe(child) for key, child in value.items()}
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return type(value).__name__

    def bits_equal(left, right):
        aa, bb = arrays(left), arrays(right)
        return len(aa) == len(bb) and all(
            a.shape == b.shape and a.dtype == b.dtype
            and bool(mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item())
            for a, b in zip(aa, bb)
        )

    def save():
        output.write_text(json.dumps({
            "method": "actual-input pure-op replay, synchronised latency; not model shares",
            "repeats": repeats, "max_samples_per_op_rows_label": max_samples,
            "max_output_bytes": max_output_bytes, "records": records,
        }, indent=2) + "\n")

    def wrap(module, name, rows_of, candidate=False):
        original = getattr(module, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            nonlocal active
            if active or not arm.is_file():
                return original(*args, **kwargs)
            rows = rows_of(args)
            if rows not in (1, 2):
                return original(*args, **kwargs)
            label = arm.read_text().strip()[:64]
            key = (label, name, rows)
            if counts.get(key, 0) >= max_samples:
                return original(*args, **kwargs)
            counts[key] = counts.get(key, 0) + 1
            active = True
            try:
                # These are already-live model/cache arrays. Materialisation
                # does not copy the cache or move it to host memory.
                inputs = arrays((args, kwargs))
                mx.eval(*inputs)
                mx.synchronize()
                reference = original(*args, **kwargs)
                mx.eval(*arrays(reference))
                output_bytes = sum(a.nbytes for a in arrays(reference))
                record = {"label": label, "operator": name, "rows": rows,
                          "sample": counts[key], "args": describe(args),
                          "kwargs": describe(kwargs), "output": describe(reference)}
                if output_bytes > max_output_bytes:
                    record["skipped"] = "output exceeds 32 MiB replay bound"
                    records.append(record)
                    save()
                    return reference
                # One reference + one replay <= 64 MiB final outputs. Existing
                # kernels may allocate their normal bounded intermediate work;
                # no new cache/weight copies or persistent tensors are added.
                for variant in (("baseline", "down2") if candidate and rows == 2 else ("baseline",)):
                    previous = moe._DOWN_ROWS
                    try:
                        if variant == "down2":
                            moe._DOWN_ROWS = 2
                        warm = original(*args, **kwargs)
                        mx.eval(*arrays(warm))
                        exact = bits_equal(reference, warm)
                        del warm
                        elapsed = []
                        for _ in range(repeats):
                            mx.synchronize()
                            start = time.perf_counter_ns()
                            result = original(*args, **kwargs)
                            mx.eval(*arrays(result))
                            elapsed.append((time.perf_counter_ns() - start) / 1000)
                            exact = exact and bits_equal(reference, result)
                            del result
                        record[variant] = {"microseconds": elapsed,
                                           "median_us": statistics.median(elapsed),
                                           "bit_exact": exact}
                    finally:
                        moe._DOWN_ROWS = previous
                if candidate and rows == 2:
                    restored = original(*args, **kwargs)
                    mx.eval(*arrays(restored))
                    record["restored_baseline_bit_exact"] = bits_equal(reference, restored)
                    del restored
                records.append(record)
                save()
                return reference
            finally:
                active = False

        setattr(module, name, measured)

    wrap(moe, "_gate_up_topk", lambda a: a[1].size // a[0].hidden)
    wrap(moe, "_down", lambda a: 1)
    wrap(moe, "_window_down", lambda a: a[2].shape[0], candidate=True)
    wrap(attn_fused, "dense_sdpa_gate", lambda a: a[0].shape[2])
    wrap(qsa_fast, "_decode_qsa_sdpa", lambda a: a[0].shape[2])
    wrap(qsa_fast, "_gather_kv_rows", lambda a: a[1].shape[1] if a[1].ndim == 3 else 1)
    wrap(qsa_fast, "_portable_indexer_head_scores", lambda a: a[0].shape[1])
    wrap(qsa_fast, "decode_block_selection_tokens", lambda a: a[0].shape[1])
    save()
    _INSTALLED = True
