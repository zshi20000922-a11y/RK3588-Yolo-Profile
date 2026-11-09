#!/usr/bin/env python3
"""Aggregate TensorRT workstation benchmark outputs into CSV and Markdown."""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    quality = read_csv(out / "quality_metrics.csv")
    timings = read_csv(out / "raw_timings.csv")
    rounds = read_csv(out / "round_throughput.csv")
    models = sorted({r["model"] for r in quality})
    summary = []

    for model in models:
        q = {r["backend"]: r for r in quality if r["model"] == model}
        pt, trt = q["PyTorch FP32"], q["TensorRT FP16"]
        raw = [r for r in timings if r["model"] == model]
        round_fps = [f(r, "disk_to_results_fps") for r in rounds if r["model"] == model]
        pre = [f(r, "preprocess_ms") for r in raw]
        inf = [f(r, "inference_ms") for r in raw]
        post = [f(r, "postprocess_ms") for r in raw]
        total = [a + b + c for a, b, c in zip(pre, inf, post)]
        fps_mean = statistics.mean(round_fps)
        fps_std = statistics.stdev(round_fps) if len(round_fps) > 1 else 0.0
        summary.append({
            "model": model,
            "pt_fp32_map50_95": f(pt, "map50_95"),
            "trt_fp16_map50_95": f(trt, "map50_95"),
            "delta_map50_95": f(trt, "map50_95") - f(pt, "map50_95"),
            "pt_fp32_map50": f(pt, "map50"),
            "trt_fp16_map50": f(trt, "map50"),
            "delta_map50": f(trt, "map50") - f(pt, "map50"),
            "trt_inference_mean_ms": statistics.mean(inf),
            "trt_inference_p50_ms": percentile(inf, 0.50),
            "trt_inference_p95_ms": percentile(inf, 0.95),
            "trt_inference_p99_ms": percentile(inf, 0.99),
            "trt_preprocess_mean_ms": statistics.mean(pre),
            "trt_postprocess_mean_ms": statistics.mean(post),
            "trt_stages_mean_fps": 1000.0 / statistics.mean(total),
            "trt_stages_p95_ms": percentile(total, 0.95),
            "trt_stages_p99_ms": percentile(total, 0.99),
            "disk_to_results_fps_mean": fps_mean,
            "disk_to_results_fps_std": fps_std,
            "disk_to_results_fps_cv_pct": 100.0 * fps_std / fps_mean if fps_mean else float("nan"),
            "disk_to_results_fps_min": min(round_fps),
            "disk_to_results_fps_max": max(round_fps),
            "engine_bytes": (out / "engines" / model / f"{model}.engine").stat().st_size,
            "frame_samples": len(raw),
            "rounds": len(round_fps),
        })

    csv_path = out / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# NVIDIA TensorRT workstation benchmark — 2026-11-05",
        "",
        "## Scope and method",
        "",
        "- Hardware: NVIDIA GeForce RTX 5090, CUDA 12.8 PyTorch runtime, TensorRT 10.16.1.11.",
        "- Ten checkpoints: YOLOv5/v5u/v8/11/26, n and s variants; static 640×640, batch 1.",
        "- Accuracy: the fixed, non-overlapping COCO val1000 manifest used in the RKNN work (1000 images; 7476 labeled instances). PT FP32 is the per-checkpoint reference; TensorRT FP16 is the exported deployment engine.",
        "- Validation thresholds: confidence 0.001, NMS IoU 0.7, max detections 300. YOLO26 uses its native end-to-end output path (no traditional NMS); its postprocess timing is the native end-to-end filter/Results path.",
        "- Throughput: three repeated 1000-frame file-list runs, streaming `Results`, batch 1. `disk→Results FPS` includes image decode/load and Python/Ultralytics overhead; stage FPS is computed from per-frame pre/infer/post timings and excludes file I/O. Inference FPS is `1000 / inference_ms`.",
        "- The TRT engine files are hardware/runtime-specific and are deliberately ignored by Git. The benchmark script rebuilds them locally.",
        "",
        "## Results",
        "",
        "| Model | PT32 mAP50-95 | TRT FP16 mAP50-95 | Δ AP | TRT infer ms (mean/P50/P95) | Infer-only FPS | Stage FPS | Disk→Results FPS (mean, range) | Post ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        inf_fps = 1000.0 / r["trt_inference_mean_ms"]
        lines.append(
            f"| {r['model']} | {r['pt_fp32_map50_95']:.5f} | {r['trt_fp16_map50_95']:.5f} | "
            f"{r['delta_map50_95']:+.6f} | {r['trt_inference_mean_ms']:.3f} / {r['trt_inference_p50_ms']:.3f} / {r['trt_inference_p95_ms']:.3f} | "
            f"{inf_fps:.0f} | {r['trt_stages_mean_fps']:.0f} | "
            f"{r['disk_to_results_fps_mean']:.1f} ({r['disk_to_results_fps_min']:.1f}–{r['disk_to_results_fps_max']:.1f}) | "
            f"{r['trt_postprocess_mean_ms']:.3f} |"
        )

    lines += [
        "",
        "## Interpretation and caveats",
        "",
        "- Disk→Results throughput is the application-like FPS. It is lower than summed-stage FPS because the former includes disk decode/load, Python scheduling and the Results iterator; do not call it pure TensorRT inference throughput.",
        "- YOLO26 exports with output shape `(1, 300, 6)` and stays on the end-to-end, NMS-free detection path. Its postprocess is correspondingly shorter than the traditional NMS models.",
        "- This run measures FP16 TensorRT, not INT8. PT FP32 is the accuracy baseline; no INT8 calibration dataset was substituted.",
        "- The printed legacy YOLOv5 validator rounds AP to three decimal places. Its TensorRT-vs-PT delta is therefore only trustworthy at about 0.001 resolution.",
        "- The measured PT baselines do not numerically match every older value in `reports/YOLO-performance-summary.md` (for example, the current YOLO11n is 0.40464 vs the old 0.39437). The 1000-image ID set is the same; prior validation implementation/environment differed. Use the paired PT FP32 vs TRT FP16 values in this report for TensorRT deltas, and re-run RKNN/PT together before making cross-backend accuracy claims.",
        "- RTX 5090 FPS and RK3588 NPU FPS are measurements on different hardware and must not be compared as an apples-to-apples device ranking.",
        "",
        "## Files",
        "",
        "- `quality_metrics.csv`: per-model PT FP32 and TensorRT FP16 metrics and validator timing.",
        "- `round_throughput.csv`: three external disk-to-Results throughput rounds per engine.",
        "- `raw_timings.csv`: 3000 per-frame stage timing rows per model.",
        "- `environment.json`: versions, GPU state, test manifest and model hashes.",
        "- `logs/`: export and validation logs.",
        "- `engines/`: local TensorRT engines/ONNX (ignored by Git).",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path} and {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
