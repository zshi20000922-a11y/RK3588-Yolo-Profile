#!/usr/bin/env python3
"""Run 1000-image stage profiles for the optimized RKNN postprocessors."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("RK3588_YOLO_WORKSPACE", Path(__file__).resolve().parents[1])).resolve()
BENCH = ROOT / "benchmarks/rk3588_yolo_int8"
SERIAL = os.environ.get("ANDROID_SERIAL")
REMOTE = "/data/local/tmp/optimized_profile_20261109/natural"
BOARD_BENCH = "/data/local/tmp/rk_yolo_bench"
BOARD_V5 = "/data/local/tmp/yolov5_additions"
BOARD_LIST = "/data/local/tmp/rk_yolo_quality/input_list.txt"
BOARD_LABELS = f"{BOARD_BENCH}/labels.txt"
MODELS = {
    "yolov5n": (f"{BOARD_V5}/yolov5n_rk3588_int8.rknn", f"{BOARD_V5}/yolov5n.yaml", "yolov5"),
    "yolov5s": (f"{BOARD_V5}/yolov5s_rk3588_int8.rknn", f"{BOARD_V5}/yolov5s.yaml", "yolov5"),
    "yolov5nu": (f"{BOARD_V5}/yolov5nu_rk3588_int8.rknn", f"{BOARD_V5}/yolov5nu.yaml", "yolov8"),
    "yolov5su": (f"{BOARD_V5}/yolov5su_rk3588_int8.rknn", f"{BOARD_V5}/yolov5su.yaml", "yolov8"),
    "yolo26n": (f"{BOARD_BENCH}/yolo26n.rknn", f"{BOARD_BENCH}/yolo26n.yaml", "yolo26"),
    "yolo26s": (f"{BOARD_BENCH}/yolo26s.rknn", f"{BOARD_BENCH}/yolo26s.yaml", "yolo26"),
}


def adb(*args: str) -> str:
    command = ["adb"] + (["-s", SERIAL] if SERIAL else []) + list(args)
    return subprocess.run(command, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not BENCH.is_dir():
        raise FileNotFoundError(
            f"benchmark workspace not found at {BENCH}; set RK3588_YOLO_WORKSPACE "
            "to the rknn-cpp-Multithreading-main checkout"
        )
    output = BENCH / "results/optimized_full_20261109/natural"
    output.mkdir(parents=True, exist_ok=True)
    local_list = BENCH / "quality_input/input_list.txt"
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "serial": SERIAL,
        "confidence": 0.25,
        "warmup": 20,
        "input_list_local_sha256": sha256(local_list),
        "input_list_board_sha256": adb("shell", f"sha256sum {BOARD_LIST}").split()[0],
        "device": adb("shell", "uname -a; cat /sys/kernel/debug/rknpu/load 2>/dev/null; cat /proc/loadavg"),
        "model_paths": {name: list(spec) for name, spec in MODELS.items()},
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    adb("shell", f"mkdir -p {REMOTE}")
    binary = ROOT / "build-yolov5-cross/rk3588_yolo_quality"
    remote_binary = f"{REMOTE}/rk3588_yolo_quality"
    adb("push", str(binary), remote_binary)
    adb("shell", f"chmod 755 {remote_binary}")
    for name, (model, manifest, family) in MODELS.items():
        remote_det = f"{REMOTE}/{name}.jsonl"
        remote_tim = f"{REMOTE}/{name}_timings.csv"
        command = (
            f"LD_LIBRARY_PATH={BOARD_BENCH} {remote_binary} --model {model} "
            f"--manifest {manifest} --labels {BOARD_LABELS} --family {family} "
            f"--input-list {BOARD_LIST} --detections {remote_det} --timings {remote_tim} "
            "--warmup 20 --confidence 0.25"
        )
        started = datetime.now(timezone.utc).isoformat()
        result = adb("shell", command)
        stem = output / name
        stem.with_suffix(".log").write_text(result)
        stem.with_name(stem.name + "_command.txt").write_text(command + "\n")
        adb("pull", remote_det, str(stem.with_suffix(".jsonl")))
        adb("pull", remote_tim, str(stem.with_name(stem.name + "_timings.csv")))
        with (output / "run_index.jsonl").open("a") as f:
            f.write(json.dumps({"model": name, "started_utc": started, "stdout": result.strip()}, ensure_ascii=False) + "\n")
        print(f"COMPLETED {name}: {result.strip()}", flush=True)


if __name__ == "__main__":
    main()
