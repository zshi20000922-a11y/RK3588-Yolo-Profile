#!/usr/bin/env python3
"""Summarize the controlled RK3588 4K motion profiling runs."""
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/rk3588_yolo_int8/results/motion_algorithms_4k_60"
BOARD = OUT / "board"
GROUPS = {
    "three_frame_copy": [f"frame_diff_4k_copy_r{i}" for i in (3, 4, 5)],
    "three_frame_direct": [f"frame_diff_4k_direct_fixed_r{i}" for i in (1, 2, 3)],
    "mog2_copy": [f"mog2_4k_copy_r{i}" for i in (1, 2, 3)],
    "mog2_direct": [f"mog2_4k_direct_fixed_r{i}" for i in (1, 2, 3)],
}


def load_run(group, name):
    path = BOARD / name
    start = json.loads((path / "status_start.json").read_text())
    end = json.loads((path / "status_end.json").read_text())
    resources = [json.loads(line) for line in (path / "resources.jsonl").read_text().splitlines()]
    duration = resources[-1]["elapsed_s"]
    sm = start["motion_pipeline_ms"]["cam0"]
    em = end["motion_pipeline_ms"]["cam0"]
    ss, es = start["sources"][0], end["sources"][0]
    cpu = statistics.mean(x["cpu_pct"] for x in resources)
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
        "process_cpu_pct": cpu, "cpu_cores": cpu / 100.0,
        "soc_cpu_pct": cpu / 8.0,
        "rss_mib": max(x["rss_kb"] for x in resources) / 1024.0,
        "npu_total_pct": statistics.mean(sum(x["npu_core_pct"]) for x in resources),
        "max_temp_c": max(x["temp_mC"] for x in resources) / 1000.0,
        "detections": es["detection_completed"] - ss["detection_completed"],
    }


rows = [load_run(group, run) for group, runs in GROUPS.items() for run in runs]
fields = list(rows[0])
with (OUT / "per_run_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)

aggregate = []
metrics = [x for x in fields if x not in ("group", "run")]
for group in GROUPS:
    selected = [x for x in rows if x["group"] == group]
    item = {"group": group}
    for metric in metrics:
        values = [float(x[metric]) for x in selected]
        item[metric + "_median"] = statistics.median(values)
        item[metric + "_mean"] = statistics.mean(values)
        item[metric + "_cv_pct"] = (statistics.stdev(values) / statistics.mean(values) * 100
                                      if len(values) > 1 and statistics.mean(values) else 0)
    aggregate.append(item)
with (OUT / "aggregate_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
    writer.writeheader(); writer.writerows(aggregate)
(OUT / "summary.json").write_text(json.dumps({"runs": rows, "aggregate": aggregate}, indent=2))
