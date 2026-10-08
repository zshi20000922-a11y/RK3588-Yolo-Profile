#!/usr/bin/env python3
"""Summarize controlled RK3588 2560x1440 NV12 motion profiling runs."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/rk3588_yolo_int8/results/motion_algorithms_2k_60"
BOARD = OUT / "board/results2k"
GROUPS = {
    "three_frame_2k_direct": [f"frame_diff_2k_direct_r{i}" for i in (1, 2, 3)],
    "three_frame_4k_to_2k": [f"frame_diff_4k_to_2k_r{i}" for i in (1, 2, 3)],
    "mog2_2k_direct": [f"mog2_2k_direct_r{i}" for i in (1, 2, 3)],
    "mog2_4k_to_2k": [f"mog2_4k_to_2k_r{i}" for i in (1, 2, 3)],
}


def load(group, name):
    path = BOARD / name
    start = json.loads((path / "status_start.json").read_text())
    end = json.loads((path / "status_end.json").read_text())
    samples = [json.loads(x) for x in (path / "resources.jsonl").read_text().splitlines()]
    duration = samples[-1]["elapsed_s"]
    sm, em = start["motion_pipeline_ms"]["cam0"], end["motion_pipeline_ms"]["cam0"]
    ss, es = start["sources"][0], end["sources"][0]
    cpu = statistics.mean(x["cpu_pct"] for x in samples)
    return {
        "group": group, "run": name, "duration_s": duration,
        "motion_frames": em["samples"] - sm["samples"],
        "motion_fps": (em["samples"] - sm["samples"]) / duration,
        "capture_fps": (es["frames"] - ss["frames"]) / duration,
        "transfer_ms_avg": em["rga"]["avg"],
        "transfer_ms_p50": em["rga"]["p50"],
        "transfer_ms_p95": em["rga"]["p95"],
        "algorithm_ms_avg": em["postprocess"]["avg"],
        "algorithm_ms_p50": em["postprocess"]["p50"],
        "algorithm_ms_p95": em["postprocess"]["p95"],
        "process_cpu_pct": cpu, "cpu_cores": cpu / 100,
        "soc_cpu_pct": cpu / 8,
        "rss_mib": max(x["rss_kb"] for x in samples) / 1024,
        "npu_total_pct": statistics.mean(sum(x["npu_core_pct"]) for x in samples),
        "max_temp_c": max(x["temp_mC"] for x in samples) / 1000,
        "detections": es["detection_completed"] - ss["detection_completed"],
    }


rows = [load(group, run) for group, runs in GROUPS.items() for run in runs]
with (OUT / "per_run_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
aggregates = []
for group in GROUPS:
    chosen = [r for r in rows if r["group"] == group]
    item = {"group": group}
    for key in rows[0]:
        if key in ("group", "run"): continue
        values = [float(r[key]) for r in chosen]
        item[key + "_median"] = statistics.median(values)
        item[key + "_mean"] = statistics.mean(values)
        item[key + "_cv_pct"] = statistics.stdev(values) / statistics.mean(values) * 100 if statistics.mean(values) else 0
    aggregates.append(item)
with (OUT / "aggregate_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(aggregates[0])); w.writeheader(); w.writerows(aggregates)
(OUT / "summary.json").write_text(json.dumps({"runs": rows, "aggregate": aggregates}, indent=2))
