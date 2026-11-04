"""Bounded actual-head replay. Bootstrap pinned source/site before runpy."""

import json
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np

from omlx.patches import qwen4_bf16_head as head
from mlx_vlm.models.exact_speculative_verify import exact_speculative_verify_weight


def swap_bytes():
    raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+)", raw)[1])
    return page * int(re.search(r"Swapouts:\s*(\d+)", raw)[1])


def main():
    model, output = map(Path, sys.argv[1:])
    assert model.is_absolute() and output.is_absolute()
    first_swap = swap_bytes()

    def guard():
        delta = swap_bytes() - first_swap
        if delta > 1024**3:
            raise RuntimeError("Head replay stopped at private 1 GiB swap-write guard")
        return delta

    name = "language_model.lm_head.weight"
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard = model / index["weight_map"][name]
    with shard.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        entry = json.loads(stream.read(length))[name]
    assert entry["dtype"] == "BF16" and entry["shape"] == [248320, 2560]
    start, end = entry["data_offsets"]
    assert end - start == 248320 * 2560 * 2
    # Map only this tensor's byte range. Construct uint16 storage then view
    # its unchanged bits as BF16; never materialise other shard tensors.
    mapped = np.memmap(shard, dtype=np.uint16, mode="r", offset=8 + length + start,
                       shape=(248320, 2560))
    weight = mx.array(mapped).view(mx.bfloat16)
    mx.eval(weight)
    guard()
    head.BF16_HEAD_ENABLED = True
    result = {"architecture": mx.device_info()["architecture"], "tensor": name,
              "shape": entry["shape"], "weight_bytes": end - start,
              "numerical": [], "timings": []}

    def reference(x):
        return x @ weight.T if x.shape[1] == 1 else exact_speculative_verify_weight(weight, x)

    for rows in (1, 2):
        for kind in ("zero", "random", "wide"):
            x = mx.random.normal((1, rows, 2560), key=mx.random.key(31 + rows)).astype(mx.bfloat16)
            if kind == "zero":
                x = mx.zeros_like(x)
            elif kind == "wide":
                x = (x * 128).astype(mx.bfloat16)
            before = head.BF16_HEAD_CALLS[rows]
            actual = head.try_bf16_head(x, weight)
            expected = reference(x)
            serial = mx.concatenate([x[:, i:i+1] @ weight.T for i in range(rows)], axis=1)
            assert actual is not None and expected is not None
            mx.eval(actual, expected, serial)
            exact = bool(mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item())
            serial_exact = bool(mx.array_equal(actual.view(mx.uint16), serial.view(mx.uint16)).item())
            record = {"rows": rows, "input": kind, "exact_baseline": exact,
                      "exact_serial": serial_exact, "engaged": head.BF16_HEAD_CALLS[rows] == before + 1}
            result["numerical"].append(record)
            print(json.dumps(record), flush=True)
            guard()

    if all(r["exact_baseline"] and r["exact_serial"] and r["engaged"] for r in result["numerical"]):
        for rows in (1, 2):
            x = mx.random.normal((1, rows, 2560), key=mx.random.key(41)).astype(mx.bfloat16)
            for candidate in (False, True) * 3:
                mx.eval(head.try_bf16_head(x, weight) if candidate else reference(x))
            times = {False: [], True: []}
            for candidate in (False, True, True, False) * 2:
                guard()
                for _ in range(8):
                    mx.synchronize()
                    start = time.perf_counter_ns()
                    y = head.try_bf16_head(x, weight) if candidate else reference(x)
                    mx.eval(y)
                    times[candidate].append((time.perf_counter_ns() - start) / 1e6)
            record = {"rows": rows, "baseline_median_ms": statistics.median(times[False]),
                      "candidate_median_ms": statistics.median(times[True]),
                      "samples_ms": {str(k): v for k, v in times.items()}}
            result["timings"].append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "samples_ms"}), flush=True)
    result["calls"] = head.BF16_HEAD_CALLS
    result["disabled"] = head.BF16_HEAD_DISABLED
    result["swap_write_bytes"] = guard()
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
