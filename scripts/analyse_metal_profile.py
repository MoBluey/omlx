"""Summarise native Metal buffer groups without double-counting kernels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def union_seconds(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start >= end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = max(end if end is not None else stop, stop)
    return total


def analyse(records):
    groups = defaultdict(list)
    invalid = empty = 0
    for row in records:
        start, end = row["gpu_start"], row["gpu_end"]
        if row["status"] != 4 or start <= 0 or end <= start:
            invalid += 1
            continue
        names = tuple(row["kernels"])
        if not names:
            empty += 1
        groups[names].append((start, end))
    all_intervals = [item for group in groups.values() for item in group]
    summed = sum(end - start for start, end in all_intervals)
    union = union_seconds(all_intervals)
    entries = []
    for names, intervals in groups.items():
        seconds = sum(end - start for start, end in intervals)
        entries.append({"kernels": names, "buffers": len(intervals),
            "summed_buffer_seconds": seconds, "interval_union_seconds": union_seconds(intervals),
            "share_of_summed_buffer_seconds": seconds / summed if summed else None})
    return {"diagnostic_only": True, "scheduling": "MLX_MAX_OPS_PER_BUFFER=0",
        "valid_buffers": len(all_intervals), "invalid_or_zero_buffers": invalid,
        "empty_kernel_buffers": empty, "summed_buffer_seconds": summed,
        "gpu_interval_union_seconds": union, "overlap_seconds": max(0.0, summed - union),
        "span_seconds": max((b for _, b in all_intervals), default=0) - min((a for a, _ in all_intervals), default=0),
        "groups": sorted(entries, key=lambda row: row["summed_buffer_seconds"], reverse=True)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    args.output.write_text(json.dumps(analyse(records), indent=2) + "\n")
