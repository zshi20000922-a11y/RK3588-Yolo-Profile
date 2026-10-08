#!/usr/bin/env python3
"""Summarize RK3588 YOLOv5/v5u board timings and resource monitor logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


STAGES = ("copy_ms", "queue_ms", "rga_ms", "inference_ms", "output_sync_ms",
          "postprocess_ms", "e2e_ms")


def percentile(values: list[float], q: float) -> float:
    values = sorted(values); pos = (len(values) - 1) * q
    lo = int(pos); hi = min(lo + 1, len(values) - 1); part = pos - lo
    return values[lo] * (1 - part) + values[hi] * part


def resource(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    loads = [[float(x) for x in match] for match in re.findall(
        r"Core0:\s*(\d+)%.*Core1:\s*(\d+)%.*Core2:\s*(\d+)%", text)]
    temps = [float(x) / 1000 for x in re.findall(r"thermal_zone\d+/temp=(\d+)", text)]
    freqs = [float(x) for x in re.findall(r"npu/cur_freq=(\d+)", text)]
    return {
        "npu_core0_mean_pct": statistics.fmean(x[0] for x in loads) if loads else math.nan,
        "npu_core1_mean_pct": statistics.fmean(x[1] for x in loads) if loads else math.nan,
        "npu_core2_mean_pct": statistics.fmean(x[2] for x in loads) if loads else math.nan,
        "npu_total_mean_pct": statistics.fmean(sum(x) for x in loads) if loads else math.nan,
        "temperature_mean_c": statistics.fmean(temps) if temps else math.nan,
        "temperature_max_c": max(temps) if temps else math.nan,
        "npu_frequency_min_hz": min(freqs) if freqs else math.nan,
        "npu_frequency_max_hz": max(freqs) if freqs else math.nan,
    }


def write(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path); args = p.parse_args()
    run_rows = []
    for metadata_path in sorted(args.input.rglob("run.json")):
        meta = json.loads(metadata_path.read_text()); run_dir = metadata_path.parent
        samples = list(csv.DictReader((run_dir / "raw_timings.csv").open()))
        row = dict(meta); row.update(resource(run_dir / "monitor.log"))
        for stage in STAGES:
            values = [float(x[stage]) for x in samples]
            for name, value in (("mean", statistics.fmean(values)), ("std", statistics.pstdev(values)),
                                ("p50", percentile(values, .5)), ("p90", percentile(values, .9)),
                                ("p95", percentile(values, .95)), ("p99", percentile(values, .99)),
                                ("min", min(values)), ("max", max(values))):
                row[f"{stage}_{name}"] = value
        run_rows.append(row)
    if not run_rows: raise RuntimeError("no completed runs")
    args.output.mkdir(parents=True, exist_ok=True); write(args.output / "run_summary.csv", run_rows)
    groups = defaultdict(list)
    for row in run_rows: groups[(row["model"], row["mode"], row["input_mode"])].append(row)
    aggregate = []
    for (model, mode, input_mode), rows in sorted(groups.items()):
        fps = [float(x["measured_fps"]) for x in rows]
        e2e = [float(x["e2e_ms_mean"]) for x in rows]
        aggregate.append({
            "model": model, "mode": mode, "input_mode": input_mode, "rounds": len(rows),
            "fps_mean": statistics.fmean(fps), "fps_std": statistics.pstdev(fps),
            "fps_cv": statistics.pstdev(fps) / statistics.fmean(fps),
            "e2e_ms_mean": statistics.fmean(e2e),
            "inference_ms_mean": statistics.fmean(float(x["inference_ms_mean"]) for x in rows),
            "postprocess_ms_mean": statistics.fmean(float(x["postprocess_ms_mean"]) for x in rows),
            "copy_ms_mean": statistics.fmean(float(x["copy_ms_mean"]) for x in rows),
            "process_cpu_pct_mean": statistics.fmean(float(x["process_cpu_pct"]) for x in rows),
            "npu_total_mean_pct": statistics.fmean(float(x["npu_total_mean_pct"]) for x in rows),
            "temperature_max_c": max(float(x["temperature_max_c"]) for x in rows),
            "npu_frequency_min_hz": min(float(x["npu_frequency_min_hz"]) for x in rows),
            "npu_frequency_max_hz": max(float(x["npu_frequency_max_hz"]) for x in rows),
        })
    write(args.output / "aggregate_summary.csv", aggregate)


if __name__ == "__main__": main()
