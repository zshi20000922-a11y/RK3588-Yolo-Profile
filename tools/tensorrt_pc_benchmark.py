#!/usr/bin/env python3
"""Reproducible TensorRT FP16 benchmark for the RK3588 YOLO model set.

Runs on the NVIDIA workstation (not on the RK3588). The model set and COCO
val1000 manifest match the prior RKNN study. YOLOv5 anchor-based checkpoints
are exported/validated through their original YOLOv5 repository; all other
checkpoints use the matching Ultralytics API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import torch
import tensorrt as trt
import ultralytics


RK_ROOT = Path(os.environ.get("RK3588_ROOT", Path(__file__).resolve().parents[2])).resolve()
BENCH_ROOT = Path(os.environ.get(
    "RK3588_YOLO_WORKSPACE", RK_ROOT / "rknn-cpp-Multithreading-main"
)).resolve() / "benchmarks/rk3588_yolo_int8"
ARTIFACTS = BENCH_ROOT / "artifacts"
V5_ARTIFACTS = BENCH_ROOT / "yolov5_additions/artifacts"
V5_REPO = Path(os.environ.get("YOLOV5_LEGACY_ROOT", RK_ROOT.parent / "yolov5-master")).resolve()
COCO_ROOT = Path(os.environ.get("COCO_DATASET_ROOT", RK_ROOT.parent / "ultralytics-main/coco")).resolve()
TEST_LIST = ARTIFACTS / "yolo26_document_method/test_1000_relocated.txt"
V5_TEST_LIST = BENCH_ROOT / "yolov5_additions/data/test_1000_relocated.txt"
MODEL_WEIGHTS = {
    "yolov5n": V5_ARTIFACTS / "yolov5n.pt",
    "yolov5s": V5_ARTIFACTS / "yolov5s.pt",
    "yolov5nu": V5_ARTIFACTS / "yolov5nu.pt",
    "yolov5su": V5_ARTIFACTS / "yolov5su.pt",
    "yolov8n": ARTIFACTS / "yolov8n/yolov8n.pt",
    "yolov8s": ARTIFACTS / "yolov8s/yolov8s.pt",
    "yolo11n": ARTIFACTS / "yolo11n/yolo11n.pt",
    "yolo11s": ARTIFACTS / "yolo11s/yolo11s.pt",
    "yolo26n": ARTIFACTS / "yolo26n/yolo26n.pt",
    "yolo26s": ARTIFACTS / "yolo26s/yolo26s.pt",
}
LEGACY_V5 = {"yolov5n", "yolov5s"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_logged(argv: list[str], log_path: Path, cwd: Path | None = None) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(argv)}; see {log_path}")
    return proc.stdout


def symlink_weight(name: str, out: Path) -> Path:
    src = MODEL_WEIGHTS[name].resolve(strict=True)
    dst = out / "engines" / name / f"{name}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and dst.resolve() == src:
        return dst
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    return dst


def export_engine(name: str, out: Path) -> Path:
    pt_link = symlink_weight(name, out)
    engine = pt_link.with_suffix(".engine")
    if engine.exists():
        return engine
    if name in LEGACY_V5:
        cmd = [
            sys.executable, str(V5_REPO / "export.py"), "--weights", str(pt_link),
            "--include", "engine", "--imgsz", "640", "--batch-size", "1",
            "--device", "0", "--half", "--workspace", "8",
        ]
        run_logged(cmd, out / "logs" / f"{name}_export.log", cwd=V5_REPO)
    else:
        from ultralytics import YOLO

        model = YOLO(str(pt_link), task="detect")
        result = model.export(
            format="engine", imgsz=640, batch=1, half=True, dynamic=False,
            device=0, workspace=8, verbose=False,
        )
        if Path(result).resolve() != engine.resolve():
            shutil.copy2(result, engine)
    if not engine.is_file() or engine.stat().st_size == 0:
        raise RuntimeError(f"{name}: TensorRT export did not create a valid engine")
    return engine


def write_dataset_yaml(path: Path, test_list: Path) -> None:
    names = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
        6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
        11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
        16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
        22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
        27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
        32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
        36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
        40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
        46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
        51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
        57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
        62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
        67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
        72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
        77: "teddy bear", 78: "hair drier", 79: "toothbrush",
    }
    lines = [f"path: {COCO_ROOT}", "train: images/val2017", f"val: {test_list.resolve()}", "names:"]
    lines.extend(f"  {i}: {label}" for i, label in names.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_modern(name: str, model_path: Path, data_yaml: Path, out: Path, half: bool) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(data_yaml), imgsz=640, batch=1, device=0, half=half,
        workers=4, plots=False, verbose=False, conf=0.001, iou=0.7,
        max_det=300, rect=False, project=str(out / "validation"),
        name=f"{name}_{'trt_fp16' if model_path.suffix == '.engine' else 'pt_fp32'}",
        exist_ok=True,
    )
    wall = time.perf_counter() - t0
    results = metrics.results_dict
    speed = metrics.speed
    return {
        "model": name,
        "backend": "TensorRT FP16" if model_path.suffix == ".engine" else "PyTorch FP32",
        "map50_95": float(results.get("metrics/mAP50-95(B)", float("nan"))),
        "map50": float(results.get("metrics/mAP50(B)", float("nan"))),
        "precision": float(results.get("metrics/precision(B)", float("nan"))),
        "recall": float(results.get("metrics/recall(B)", float("nan"))),
        "preprocess_ms": float(speed.get("preprocess", float("nan"))),
        "inference_ms": float(speed.get("inference", float("nan"))),
        "postprocess_ms": float(speed.get("postprocess", float("nan"))),
        "val_wall_s": wall,
        "val_loop_fps": 1000.0 / sum(float(speed.get(k, 0.0)) for k in ("preprocess", "inference", "postprocess")),
    }


def parse_v5_metrics(text: str, name: str, backend: str) -> dict:
    # YOLOv5 val prints the aggregate row as: all <n> <targets> P R mAP50 mAP50-95
    candidates = re.findall(r"^\s*all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", text, re.M)
    if not candidates:
        raise RuntimeError(f"Could not parse YOLOv5 validation metrics for {name}")
    p, r, map50, map95 = map(float, candidates[-1])
    speed = re.search(r"Speed:\s*([\d.]+)ms pre-process,\s*([\d.]+)ms inference,\s*([\d.]+)ms NMS", text)
    pre, inf, post = map(float, speed.groups()) if speed else (float("nan"),) * 3
    return {
        "model": name, "backend": backend, "map50_95": map95, "map50": map50,
        "precision": p, "recall": r, "preprocess_ms": pre, "inference_ms": inf,
        "postprocess_ms": post, "val_wall_s": float("nan"),
        "val_loop_fps": 1000.0 / (pre + inf + post) if pre + inf + post > 0 else float("nan"),
    }


def validate_legacy(name: str, model_path: Path, data_yaml: Path, out: Path, engine: bool) -> dict:
    t0 = time.perf_counter()
    cmd = [
        sys.executable, str(V5_REPO / "val.py"), "--weights", str(model_path),
        "--data", str(data_yaml), "--imgsz", "640", "--batch-size", "1",
        "--device", "0", "--conf-thres", "0.001", "--iou-thres", "0.7",
        "--max-det", "300", "--workers", "4", "--project", str(out / "validation"),
        "--name", f"{name}_{'trt_fp16' if engine else 'pt_fp32'}", "--exist-ok",
    ]
    if engine:
        cmd += ["--half"]
    text = run_logged(cmd, out / "logs" / f"{name}_{'trt_fp16' if engine else 'pt_fp32'}_val.log", cwd=V5_REPO)
    row = parse_v5_metrics(text, name, "TensorRT FP16" if engine else "PyTorch FP32")
    row["val_wall_s"] = time.perf_counter() - t0
    return row


def profile_engine_stream(
    name: str, engine: Path, image_paths: list[str], manifest: Path, rounds: int = 3
) -> tuple[list[dict], list[dict]]:
    """Measure per-frame stages and disk-to-Results throughput through Python TensorRT."""
    from ultralytics import YOLO

    model = YOLO(str(engine), task="detect")
    predict_args = {
        "imgsz": 640, "batch": 1, "device": 0, "half": True,
        "stream": True, "save": False, "verbose": False,
    }
    for _ in model.predict(source=image_paths[0], **predict_args):
        pass
    raw_rows, round_rows = [], []
    for round_id in range(1, rounds + 1):
        start = time.perf_counter()
        count = 0
        for result in model.predict(source=str(manifest), **predict_args):
            count += 1
            timing = result.speed
            raw_rows.append({
                "model": name, "backend": "TensorRT FP16", "round": round_id,
                "image_index": count - 1,
                "preprocess_ms": float(timing.get("preprocess", float("nan"))),
                "inference_ms": float(timing.get("inference", float("nan"))),
                "postprocess_ms": float(timing.get("postprocess", float("nan"))),
            })
        elapsed = time.perf_counter() - start
        if count != len(image_paths):
            raise RuntimeError(f"{name}: expected {len(image_paths)} predictions, got {count}")
        round_rows.append({
            "model": name, "backend": "TensorRT FP16", "round": round_id,
            "frames": count, "disk_to_results_wall_s": elapsed,
            "disk_to_results_fps": count / elapsed,
        })
    return raw_rows, round_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=list(MODEL_WEIGHTS))
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; refusing to benchmark on CPU")
    for name in args.models:
        if name not in MODEL_WEIGHTS:
            raise ValueError(f"Unknown model {name}")
        if not MODEL_WEIGHTS[name].is_file():
            raise FileNotFoundError(MODEL_WEIGHTS[name])
    test_list = V5_TEST_LIST if args.models and args.models[0] in LEGACY_V5 else TEST_LIST
    paths = test_list.read_text(encoding="utf-8").splitlines()
    if len(paths) != 1000 or len(set(paths)) != 1000 or not all(Path(p).is_file() for p in paths):
        raise RuntimeError(f"Expected exactly 1000 unique existing test images in {test_list}")
    data_yaml = out / "coco_val1000.yaml"
    write_dataset_yaml(data_yaml, test_list)

    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.gr,clocks.mem", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
        "trt": trt.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__, "imgsz": 640, "batch": 1,
        "test_images": len(paths), "test_manifest": str(test_list),
        "test_manifest_sha256": sha256(test_list), "test_image_order_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
        "test_set_root": str(COCO_ROOT), "model_weight_sha256": {k: sha256(MODEL_WEIGHTS[k]) for k in args.models},
        "precision": "TensorRT FP16; PyTorch FP32 reference",
        "metric_scope": "COCO val1000, conf=0.001, IoU=0.7, max_det=300; same 1000-image manifest as RKNN evaluation",
    }
    (out / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    quality_rows: list[dict] = []
    raw_timing_rows: list[dict] = []
    round_rows: list[dict] = []
    engines = {}
    for i, name in enumerate(args.models, 1):
        print(f"[{i}/{len(args.models)}] {name}: export TensorRT FP16", flush=True)
        engine = out / "engines" / name / f"{name}.engine"
        if not args.skip_export:
            engine = export_engine(name, out)
        if not engine.is_file():
            raise FileNotFoundError(engine)
        engines[name] = engine
        if not args.skip_validation:
            pt = symlink_weight(name, out)
            print(f"[{i}/{len(args.models)}] {name}: PT FP32 COCO1000 validation", flush=True)
            if name in LEGACY_V5:
                quality_rows.append(validate_legacy(name, pt, data_yaml, out, engine=False))
            else:
                quality_rows.append(validate_modern(name, pt, data_yaml, out, half=False))
            print(f"[{i}/{len(args.models)}] {name}: TensorRT FP16 COCO1000 validation", flush=True)
            if name in LEGACY_V5:
                quality_rows.append(validate_legacy(name, engine, data_yaml, out, engine=True))
            else:
                quality_rows.append(validate_modern(name, engine, data_yaml, out, half=True))
            print(f"[{i}/{len(args.models)}] {name}: TensorRT three-round stream profile", flush=True)
            raw_rows, throughput_rows = profile_engine_stream(name, engine, paths, test_list)
            raw_timing_rows.extend(raw_rows)
            round_rows.extend(throughput_rows)
            write_csv(out / "raw_timings.csv", raw_timing_rows)
            write_csv(out / "round_throughput.csv", round_rows)
            write_csv(out / "quality_metrics.csv", quality_rows)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    write_csv(out / "quality_metrics.csv", quality_rows)
    write_csv(out / "raw_timings.csv", raw_timing_rows)
    write_csv(out / "round_throughput.csv", round_rows)
    print(f"DONE output={out}", flush=True)


if __name__ == "__main__":
    main()
