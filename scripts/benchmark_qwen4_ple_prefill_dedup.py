"""Bounded real-shard host-assembly screen; synthetic indices, no model load."""

import json
import os
from pathlib import Path
import re
import resource
import statistics
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
SITE = Path("/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/runtime/omlx-0.7.3-m4-max-r2-sealed-site")
sys.path[:0] = [str(SOURCE), str(SITE)]

import mlx.core as mx
import numpy as np
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch

assert apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp.language import DiskBackedShardedEmbedding, _SafeTensorMMap


def swap_written():
    raw = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+)", raw).group(1))
    return int(re.search(r"^Swapouts:\s+(\d+)", raw, re.M).group(1)) * page


def main():
    model = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    swap_start = swap_written()
    config = json.loads((model / "config.json").read_text())["text_config"]
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    weights = sorted(key for key in index if "ngram_embedding.shards." in key and key.endswith(".weight"))
    headers = {}
    rows = 0
    for key in weights:
        filename = index[key]
        if filename not in headers:
            with (model / filename).open("rb", buffering=0) as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                headers[filename] = json.loads(handle.read(size))
        rows += headers[filename][key]["shape"][0]
    prefix = weights[0].split(".shards.")[0]
    dims = config["ple_embed_dim"] // ((config["ngram_size"] - 1) * config["heads_per_ngram"])
    embedding = DiskBackedShardedEmbedding(model, prefix, rows, dims, len(weights))
    rng = np.random.default_rng(51)
    unique = rng.choice(rows, size=22248, replace=False).astype(np.int64)
    host = np.concatenate([unique, rng.choice(unique, size=39088 - len(unique))])
    rng.shuffle(host)
    plan = embedding._plan(host)
    records = []
    try:
        started = time.perf_counter()
        baseline = embedding._assemble(host, plan)
        first_seconds = time.perf_counter() - started
        candidate = embedding._assemble_prefill_deduplicated(host, plan)
        raw_equal = all(np.array_equal(baseline[family], candidate[family]) for family in baseline)
        _, _, touched, _, families, dtypes, bits, group_size = plan

        def upload(buffers):
            arrays = [_SafeTensorMMap.to_mx(buffers[family], dtypes[family]) for family in families]
            return mx.dequantize(*arrays, bits=bits, group_size=group_size, mode="affine")

        a, b = upload(baseline), upload(candidate)
        mx.eval(a, b)
        gpu_equal = bool(mx.array_equal(a, b).item())
        assert raw_equal and gpu_equal
        del a, b, baseline, candidate
        for label in "ABBAABBA":
            before_swap = swap_written()
            started = time.perf_counter()
            buffers = (embedding._assemble if label == "A" else embedding._assemble_prefill_deduplicated)(host, plan)
            elapsed = time.perf_counter() - started
            records.append({"label": label, "seconds": elapsed, "buffer_bytes": sum(array.nbytes for array in buffers.values())})
            del buffers
            assert swap_written() - swap_start <= 1024**3, "new host swap-write limit exceeded"
            assert before_swap >= swap_start
        medians = {label: statistics.median(row["seconds"] for row in records if row["label"] == label) for label in "AB"}
        result = dict(source=str(SOURCE), model=str(model), pid=os.getpid(), rows=int(host.size),
                      unique_rows=int(np.unique(host).size), touched_shards=len(touched),
                      first_observed_baseline_seconds=first_seconds, records=records,
                      medians=medians, raw_equal=raw_equal, gpu_equal=gpu_equal,
                      prefill_dedup_stats=embedding.prefill_dedup_stats,
                      max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      swap_write_bytes=swap_written() - swap_start,
                      limitation="Synthetic indices at the measured short-prompt duplication ratio; warmed page-cache component only, no TTFT or model load.")
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        embedding.close()


if __name__ == "__main__":
    main()
