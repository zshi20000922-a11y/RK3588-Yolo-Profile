#!/usr/bin/env python3
"""Revalidate PT/TensorRT on the RKNN COCO-val1000 set and thresholds."""
from __future__ import annotations

import csv
import os
import subprocess
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO_PROFILE_ROOT", Path(__file__).resolve().parents[1])).resolve()
RK = Path(os.environ.get("RK3588_ROOT", ROOT.parent)).resolve()
BENCH = Path(os.environ.get(
    "RK3588_YOLO_WORKSPACE", RK / "rknn-cpp-Multithreading-main"
)).resolve() / "benchmarks/rk3588_yolo_int8"
WEIGHTS = BENCH / "artifacts"
ENGINES = ROOT / "reports/tensorrt-5090-2026-11-05/engines"
COCO_ROOT = Path(os.environ.get("COCO_DATASET_ROOT", RK.parent / "ultralytics-main/coco")).resolve()
V5_REPO = Path(os.environ.get("YOLOV5_LEGACY_ROOT", RK.parent / "yolov5-master")).resolve()
OUT = ROOT / "reports/common-quality-2026-11-09"
MANIFEST = BENCH / "yolov5_additions/data/test_1000_relocated.txt"
LEGACY = {"yolov5n", "yolov5s"}
MODELS = ["yolov5n", "yolov5s", "yolov5nu", "yolov5su", "yolov8n", "yolov8s",
          "yolo11n", "yolo11s", "yolo26n", "yolo26s"]


def resolve_pt(name: str) -> Path:
    if name in {"yolov5n", "yolov5s", "yolov5nu", "yolov5su"}:
        return BENCH / f"yolov5_additions/artifacts/{name}.pt"
    return WEIGHTS / name / f"{name}.pt"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing a CPU validation")
    images = MANIFEST.read_text().splitlines()
    if len(images) != 1000 or len(set(images)) != 1000 or not all(Path(p).is_file() for p in images):
        raise RuntimeError("Expected the fixed 1000-image COCO manifest")
    OUT.mkdir(parents=True, exist_ok=True)
    yaml = OUT / "coco_val1000.yaml"
    source_yaml = ROOT / "reports/tensorrt-5090-2026-11-05/coco_val1000.yaml"
    yaml.write_text(source_yaml.read_text())
    rows = []
    for name in MODELS:
        for backend, path in (("PyTorch FP32", resolve_pt(name)),
                              ("TensorRT FP16", ENGINES / name / f"{name}.engine")):
            if not path.is_file():
                raise FileNotFoundError(path)
            run_name = f"{name}_{'pt_fp32' if backend == 'PyTorch FP32' else 'trt_fp16'}"
            print(f"VALIDATE {run_name}: {path}", flush=True)
            start = time.perf_counter()
            if name in LEGACY:
                cmd = [
                    "python", str(V5_REPO / "val.py"), "--weights", str(path), "--data", str(yaml),
                    "--imgsz", "640", "--batch-size", "1", "--device", "0", "--workers", "4",
                    "--conf-thres", "0.001", "--iou-thres", "0.45", "--max-det", "100",
                    "--project", str(OUT / "validation"), "--name", run_name, "--exist-ok",
                ]
                if backend == "TensorRT FP16":
                    cmd.append("--half")
                proc = subprocess.run(cmd, cwd=V5_REPO, text=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
                log = proc.stdout
                (OUT / f"{run_name}.log").write_text(log)
                if proc.returncode:
                    raise RuntimeError(f"validation failed for {run_name}; see {OUT / (run_name + '.log')}\n{log[-4000:]}")
                import re
                matches = re.findall(r"^\s*all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", log, re.M)
                if not matches:
                    raise RuntimeError(f"could not parse legacy metrics for {run_name}")
                p, recall, ap50, ap95 = map(float, matches[-1])
                rows.append({"model": name, "backend": backend, "map50_95": ap95, "map50": ap50,
                             "precision": p, "recall": recall, "wall_s": time.perf_counter() - start,
                             "conf": 0.001, "iou": 0.45, "max_det": 100,
                             "metric_source": "YOLOv5 internal validator; AP rounded to 0.001", "run": run_name})
            else:
                model = YOLO(str(path), task="detect")
                metrics = model.val(
                    data=str(yaml), imgsz=640, batch=1, device=0,
                    half=(backend == "TensorRT FP16"), workers=4, plots=False, verbose=False,
                    conf=0.001, iou=0.45, max_det=100, rect=False, save_json=True,
                    project=str(OUT / "validation"), name=run_name, exist_ok=True,
                )
                results = metrics.results_dict
                rows.append({
                    "model": name, "backend": backend,
                    "map50_95": float(results["metrics/mAP50-95(B)"]),
                    "map50": float(results["metrics/mAP50(B)"]),
                    "precision": float(results["metrics/precision(B)"]),
                    "recall": float(results["metrics/recall(B)"]),
                    "wall_s": time.perf_counter() - start, "conf": 0.001, "iou": 0.45,
                    "max_det": 100, "metric_source": "Ultralytics validator; save_json=True", "run": run_name,
                })
            with (OUT / "quality_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"DONE {run_name} in {time.perf_counter() - start:.1f}s", flush=True)
    print(f"RESULT {OUT / 'quality_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
