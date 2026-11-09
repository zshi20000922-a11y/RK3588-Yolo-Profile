#!/usr/bin/env python3
"""Run repeated optimized RKNN latency/throughput profiles on an attached RK3588."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("RK3588_YOLO_WORKSPACE", Path(__file__).resolve().parents[1])).resolve()
BENCH = ROOT / "benchmarks/rk3588_yolo_int8"
SERIAL = os.environ.get("ANDROID_SERIAL")
REMOTE_ROOT = "/data/local/tmp/optimized_profile_20261109"
BOARD_BENCH = "/data/local/tmp/rk_yolo_bench"
BOARD_V5 = "/data/local/tmp/yolov5_additions"
BOARD_Y26_DOC = "/data/local/tmp/yolo26_document_method"
BOARD_Y26_MAX = "/data/local/tmp/yolo26_scoremax_coco128"
BOARD_QUALITY = "/data/local/tmp/rk_yolo_quality"
BOARD_BIN = f"{REMOTE_ROOT}/rk3588_yolo_benchmark"
NV12 = f"{BOARD_BENCH}/frame.nv12"
LABELS = f"{BOARD_BENCH}/labels.txt"

MODELS = {
    "yolov5n": (f"{BOARD_V5}/yolov5n_rk3588_int8.rknn", f"{BOARD_V5}/yolov5n.yaml", "yolov5"),
    "yolov5s": (f"{BOARD_V5}/yolov5s_rk3588_int8.rknn", f"{BOARD_V5}/yolov5s.yaml", "yolov5"),
    # Ultralytics YOLOv5u uses the decoupled DFL head and the shared v8 adapter.
    "yolov5nu": (f"{BOARD_V5}/yolov5nu_rk3588_int8.rknn", f"{BOARD_V5}/yolov5nu.yaml", "yolov8"),
    "yolov5su": (f"{BOARD_V5}/yolov5su_rk3588_int8.rknn", f"{BOARD_V5}/yolov5su.yaml", "yolov8"),
    "yolo26n-score-sum": (f"{BOARD_BENCH}/yolo26n.rknn", f"{BOARD_BENCH}/yolo26n.yaml", "yolo26"),
    "yolo26s-score-sum": (f"{BOARD_BENCH}/yolo26s.rknn", f"{BOARD_BENCH}/yolo26s.yaml", "yolo26"),
    "yolo26n-score-max-coco128": (f"{BOARD_Y26_MAX}/yolo26n.rknn", f"{BOARD_Y26_MAX}/manifest.yaml", "yolo26"),
    "yolo26s-score-max-coco128": (f"{BOARD_Y26_MAX}/yolo26s.rknn", f"{BOARD_Y26_MAX}/manifest.yaml", "yolo26"),
}


def adb(*args: str, check: bool = True, capture: bool = True):
    command = ["adb"] + (["-s", SERIAL] if SERIAL else []) + list(args)
    return subprocess.run(command, check=check, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


def collect_environment(out: Path) -> None:
    commands = {
        "uname": "uname -a",
        "build": "cat /etc/os-release 2>/dev/null | head -5",
        "rknn": "dmesg 2>/dev/null | grep -i rknpu | tail -12",
        "governors": "for p in /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/class/devfreq/dmc/governor /sys/class/devfreq/fdab0000.npu/governor; do printf '%s=' \"$p\"; cat \"$p\" 2>/dev/null; done",
        "temperatures": "for p in /sys/class/thermal/thermal_zone*/temp; do printf '%s=' \"$p\"; cat \"$p\" 2>/dev/null; done",
        "frequencies": "for p in /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq /sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq /sys/class/devfreq/dmc/cur_freq /sys/class/devfreq/fdab0000.npu/cur_freq; do printf '%s=' \"$p\"; cat \"$p\" 2>/dev/null; done",
        "npu_load": "cat /sys/kernel/debug/rknpu/load 2>/dev/null",
        "processes": "ps -A 2>/dev/null | head -40",
    }
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "serial": SERIAL,
              "remote_root": REMOTE_ROOT, "device": {}}
    for key, command in commands.items():
        result["device"][key] = adb("shell", command).stdout.strip()
    (out / "environment.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def telemetry() -> dict[str, str]:
    cmd = ";".join([
        "for p in /sys/class/thermal/thermal_zone0/temp /sys/class/thermal/thermal_zone1/temp; do cat $p 2>/dev/null; done",
        "for p in /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq /sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq /sys/class/devfreq/dmc/cur_freq /sys/class/devfreq/fdab0000.npu/cur_freq; do cat $p 2>/dev/null; done",
        "cat /sys/kernel/debug/rknpu/load 2>/dev/null",
        "cat /proc/loadavg",
    ])
    lines = adb("shell", cmd).stdout.splitlines()
    padded = lines + [""] * 9
    return dict(zip(["temp0_mC", "temp1_mC", "cpu0_khz", "cpu4_khz", "cpu6_khz", "ddr_hz", "npu_hz", "npu_load", "loadavg"], padded))


def board_command(name: str, mode: str, round_id: int) -> str:
    model, manifest, family = MODELS[name]
    mode_dir = f"{REMOTE_ROOT}/{mode}/{name}"
    file_prefix = f"{name}_r{round_id}"
    return (
        f"LD_LIBRARY_PATH={BOARD_BENCH} \"{BOARD_BIN}\" "
        f"--model \"{model}\" --manifest \"{manifest}\" --labels \"{LABELS}\" "
        f"--family {family} --nv12 \"{NV12}\" --width 1920 --height 1080 "
        f"--mode {mode} --input-mode dma-replay --warmup 100 --iterations 3000 --seconds 60 "
        f"--output \"{mode_dir}/{file_prefix}.csv\" "
    )


def run_one(name: str, mode: str, round_id: int, out: Path, resume: bool = False) -> None:
    mode_dir = out / mode / name
    mode_dir.mkdir(parents=True, exist_ok=True)
    adb("shell", f"mkdir -p {REMOTE_ROOT}/{mode}/{name}")
    remote_csv = f"{REMOTE_ROOT}/{mode}/{name}/{name}_r{round_id}.csv"
    command = board_command(name, mode, round_id)
    started = time.monotonic()
    adb_command = ["adb"] + (["-s", SERIAL] if SERIAL else []) + ["shell", command]
    proc = subprocess.Popen(adb_command, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    samples: list[dict[str, str]] = []
    while proc.poll() is None:
        row = {"host_time_utc": datetime.now(timezone.utc).isoformat(), **telemetry()}
        samples.append(row)
        time.sleep(2.0)
    output = proc.communicate()[0]
    elapsed = time.monotonic() - started
    stem = f"{name}_r{round_id}"
    final_csv = mode_dir / f"{stem}.csv"
    if resume and final_csv.exists() and (mode_dir / f"{stem}.log").exists():
        print(f"SKIP existing {name} {mode} r{round_id}", flush=True)
        return
    (mode_dir / f"{stem}.log").write_text(output)
    (mode_dir / f"{stem}_command.txt").write_text(command + "\n")
    if samples:
        with (mode_dir / f"{stem}_monitor.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(samples[0]))
            w.writeheader()
            w.writerows(samples)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} {mode} round {round_id} failed ({proc.returncode}): {output[-4000:]}")
    adb("pull", remote_csv, str(mode_dir / f"{stem}.csv"))
    summary = {"model": name, "mode": mode, "round": round_id,
               "wall_s_including_adb": elapsed, "stdout": output.strip()}
    with (out / "run_index.jsonl").open("a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"COMPLETED {name} {mode} r{round_id}: {output.strip()}", flush=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--output", type=Path, default=BENCH / "results/optimized_full_20261109")
    parser.add_argument("--resume", action="store_true", help="reuse completed runs and continue missing matrix cells")
    args = parser.parse_args()
    if not BENCH.is_dir():
        raise FileNotFoundError(
            f"benchmark workspace not found at {BENCH}; set RK3588_YOLO_WORKSPACE "
            "to the rknn-cpp-Multithreading-main checkout"
        )
    if args.seconds != 60 or args.iterations != 3000:
        raise ValueError("Formal matrix requires >=60 seconds and >=3000 frames; use defaults")
    if any(x not in MODELS for x in args.models):
        raise ValueError("unknown model name")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    adb("get-state")
    adb("shell", f"mkdir -p {REMOTE_ROOT}")
    collect_environment(out)
    adb("push", str(ROOT / "build-yolov5-cross/rk3588_yolo_benchmark"), BOARD_BIN)
    adb("shell", f"chmod 755 {BOARD_BIN}")
    # Interleave latency and throughput order; rotate each round to reduce order/thermal bias.
    for round_id in range(1, args.rounds + 1):
        order = args.models[round_id - 1:] + args.models[:round_id - 1]
        for mode, names in (("latency", order), ("throughput", list(reversed(order)))):
            for name in names:
                run_one(name, mode, round_id, out, args.resume)
    print(f"ALL RUNS COMPLETE: {out}", flush=True)


if __name__ == "__main__":
    main()
