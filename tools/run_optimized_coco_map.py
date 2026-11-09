#!/usr/bin/env python3
"""Generate low-confidence RKNN predictions for COCOeval on the fixed 1000-image set."""
from __future__ import annotations

import subprocess
import os
from pathlib import Path

SERIAL = os.environ.get("ANDROID_SERIAL")
REMOTE = "/data/local/tmp/optimized_profile_20261109/natural"
BOARD_BENCH = "/data/local/tmp/rk_yolo_bench"
BOARD_V5 = "/data/local/tmp/yolov5_additions"
BOARD_LIST = "/data/local/tmp/rk_yolo_quality/input_list.txt"
QUALITY_BIN = f"{REMOTE}/rk3588_yolo_quality"
MODELS = {
    "yolov5n": (f"{BOARD_V5}/yolov5n_rk3588_int8.rknn", f"{BOARD_V5}/yolov5n.yaml", "yolov5"),
    "yolov5s": (f"{BOARD_V5}/yolov5s_rk3588_int8.rknn", f"{BOARD_V5}/yolov5s.yaml", "yolov5"),
    "yolov5nu": (f"{BOARD_V5}/yolov5nu_rk3588_int8.rknn", f"{BOARD_V5}/yolov5nu.yaml", "yolov8"),
    "yolov5su": (f"{BOARD_V5}/yolov5su_rk3588_int8.rknn", f"{BOARD_V5}/yolov5su.yaml", "yolov8"),
    "yolo26n": (f"{BOARD_BENCH}/yolo26n.rknn", f"{BOARD_BENCH}/yolo26n.yaml", "yolo26"),
    "yolo26s": (f"{BOARD_BENCH}/yolo26s.rknn", f"{BOARD_BENCH}/yolo26s.yaml", "yolo26"),
}
ROOT = Path(os.environ.get("RK3588_YOLO_WORKSPACE", Path(__file__).resolve().parents[1])).resolve()
OUTPUT = ROOT / "benchmarks/rk3588_yolo_int8/results/optimized_full_20261109/natural"


def adb(*args: str) -> str:
    command = ["adb"] + (["-s", SERIAL] if SERIAL else []) + list(args)
    return subprocess.run(command, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def main() -> None:
    if not OUTPUT.is_dir():
        raise FileNotFoundError(
            f"result workspace not found at {OUTPUT}; set RK3588_YOLO_WORKSPACE "
            "to the rknn-cpp-Multithreading-main checkout"
        )
    for name, (model, manifest, family) in MODELS.items():
        detections = f"{REMOTE}/{name}_map.jsonl"
        timings = f"{REMOTE}/{name}_map_timings.csv"
        command = (
            f"LD_LIBRARY_PATH={BOARD_BENCH} {QUALITY_BIN} --model {model} "
            f"--manifest {manifest} --labels {BOARD_BENCH}/labels.txt "
            f"--family {family} --input-list {BOARD_LIST} --detections {detections} "
            f"--timings {timings} --warmup 20 --confidence 0.001"
        )
        print(f"RUN {name}: {adb('shell', command).strip()}", flush=True)
        local_prefix = OUTPUT / name
        local_prefix.with_name(f"{name}_map_command.txt").write_text(command + "\n")
        adb("pull", detections, str(OUTPUT / f"{name}_map.jsonl"))
        adb("pull", timings, str(OUTPUT / f"{name}_map_timings.csv"))


if __name__ == "__main__":
    main()
