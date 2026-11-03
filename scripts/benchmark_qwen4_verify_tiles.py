"""Bounded two-row Qwen4 GDN tile sweep using installed affine weights.

Run with the isolated checkout and installed dependency closure on sys.path.
Only GDN projection tensors are read; the model is never instantiated.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import struct
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from omlx.patches import row_exact_qmv as qmv

TILES = [(rps, rows) for rps in (1, 2, 4, 8) for rows in (1, 2)]
PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def projection_tensors(model: Path):
    """Extract the small selected tensors without mapping entire model shards."""
    selected, chunks, offset = {}, [], 0
    for shard in sorted(model.glob("*.safetensors")):
        with shard.open("rb") as source:
            header_bytes = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(header_bytes))
            for key, meta in header.items():
                if ".linear_attn." not in key or not any(
                    f".{name}." in key for name in (*PROJECTIONS, "out_proj")
                ):
                    continue
                start, end = meta["data_offsets"]
                source.seek(8 + header_bytes + start)
                data = source.read(end - start)
                assert len(data) == end - start
                selected[key] = {**meta, "data_offsets": [offset, offset + len(data)]}
                chunks.append(data)
                offset += len(data)
    encoded = json.dumps(selected).encode()
    encoded += b" " * (-len(encoded) % 8)
    with tempfile.NamedTemporaryFile(suffix=".safetensors") as subset:
        subset.write(struct.pack("<Q", len(encoded)))
        subset.write(encoded)
        for data in chunks:
            subset.write(data)
        subset.flush()
        tensors = mx.load(subset.name)
        mx.eval(tensors)
    return tensors, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()
    tensors, weight_bytes = projection_tensors(args.model)
    prefixes = sorted({key.rsplit(".in_proj_qkv.", 1)[0] for key in tensors if ".in_proj_qkv." in key}, key=lambda p: int(p.split(".layers.")[1].split(".")[0]))
    assert len(prefixes) == 36, len(prefixes)
    weights = {"in": [], "out": []}
    for prefix in prefixes:
        weights["in"].append(tuple(mx.concatenate([tensors[f"{prefix}.{name}.{kind}"] for name in PROJECTIONS]) for kind in ("weight", "scales", "biases")))
        weights["out"].append(tuple(tensors[f"{prefix}.out_proj.{kind}"] for kind in ("weight", "scales", "biases")))
    mx.eval(weights)
    report = {"device": mx.device_info(), "weight_bytes": weight_bytes, "layers": len(prefixes), "rows": 2, "results": {}}
    launchers = {}
    for kind, k, n, bits, group, baseline in (("in", 2560, 16480, 4, 64, (1, 2)), ("out", 6144, 2560, 5, 128, (4, 1))):
        x = (mx.random.normal((1, 2, k), key=mx.random.key(17)) * 0.5).astype(mx.bfloat16)
        mx.eval(x)
        references = [mx.concatenate([mx.quantized_matmul(x[:, r:r + 1], w, scales=s, biases=b, transpose=True, group_size=group, bits=bits) for r in range(2)], axis=1) for w, s, b in weights[kind]]
        mx.eval(references)
        entries = {}
        for tile in TILES:
            runs = [qmv.rows_qmv(w, s, b, bits, group, "affine", mx.bfloat16, lambda _, t=tile: t, unrolled=True) for w, s, b in weights[kind]]
            assert all(run is not None for run in runs)
            ys = [run(x) for run in runs]
            mx.eval(ys)
            assert all(mx.array_equal(y.view(mx.uint16), ref.view(mx.uint16)).item() for y, ref in zip(ys, references)), (kind, tile)
            # Extra inputs on every real layer catch value-dependent differences.
            for seed in (31, 73):
                probe = (mx.random.normal(x.shape, key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)
                observed = [run(probe) for run in runs]
                expected = [mx.concatenate([mx.quantized_matmul(probe[:, r:r + 1], w, scales=s, biases=b, transpose=True, group_size=group, bits=bits) for r in range(2)], axis=1) for w, s, b in weights[kind]]
                mx.eval(observed, expected)
                assert all(mx.array_equal(y.view(mx.uint16), ref.view(mx.uint16)).item() for y, ref in zip(observed, expected)), (kind, tile, seed)
            verdict_key = ("rows", bits, group, k, n, 2, tile[1], tile[0], mx.bfloat16)
            entries[str(tile)] = {"parity": True, "unrolled": qmv._UNROLLED_VERDICTS[verdict_key], "samples_ms": []}
            launchers[(kind, tile)] = (runs, x)
        # Shuffle candidates each round to spread thermal and timing drift.
        rng = random.Random(97)
        for _ in range(2 + args.samples):
            order = list(TILES)
            rng.shuffle(order)
            for tile in order:
                runs, value = launchers[kind, tile]
                mx.synchronize()
                start = time.perf_counter()
                ys = [run(value) for run in runs]
                mx.async_eval(ys)
                mx.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                entries[str(tile)]["samples_ms"].append(elapsed)
        for entry in entries.values():
            entry["samples_ms"] = entry["samples_ms"][2:]
            entry["median_ms"] = statistics.median(entry["samples_ms"])
        winner = min(TILES, key=lambda t: entries[str(t)]["median_ms"])
        confirmations = {str(t): [] for t in {baseline, winner}}
        for _ in range(12):
            order = list({baseline, winner})
            rng.shuffle(order)
            for tile in order:
                runs, value = launchers[kind, tile]
                mx.synchronize()
                start = time.perf_counter()
                ys = [run(value) for run in runs]
                mx.async_eval(ys)
                mx.synchronize()
                confirmations[str(tile)].append((time.perf_counter() - start) * 1000)
        report["results"][kind] = {"shape": [2, k, n], "bits": bits, "group_size": group, "baseline": baseline, "winner": winner, "tiles": entries, "confirmation_ms": confirmations}
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({"kind": kind, **report["results"][kind]}), flush=True)


if __name__ == "__main__":
    main()
