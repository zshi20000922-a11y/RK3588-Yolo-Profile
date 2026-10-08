#!/usr/bin/env python3
"""Reproducible RK3588 INT8 conversion, deployment, profiling and reporting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

MODELS = {
    "yolov8n": "yolov8", "yolov8s": "yolov8",
    "yolo11n": "yolo11", "yolo11s": "yolo11",
    "yolo26n": "yolo26", "yolo26s": "yolo26",
}
STAGES = ("copy_ms", "queue_ms", "rga_ms", "inference_ms", "output_sync_ms", "postprocess_ms", "e2e_ms")


def run(cmd, *, check=True, capture=False, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=check, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None, env=env)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adb(serial, *args, capture=False, check=True):
    return run(["adb", "-s", serial, *args], capture=capture, check=check)


def require_device(serial):
    state = adb(serial, "get-state", capture=True, check=False)
    if state.returncode or state.stdout.strip() != "device":
        raise RuntimeError(f"ADB {serial} is not online (state={state.stdout.strip()!r})")


def cmd_preflight(a):
    require_device(a.serial)
    probes = {
        "uname": "uname -a", "os_release": "cat /etc/os-release 2>/dev/null",
        "rknpu_driver": "cat /sys/kernel/debug/rknpu/version 2>/dev/null || dmesg | grep -i rknpu | tail -20",
        "runtime_libs": "ldconfig -p 2>/dev/null | grep -E 'rknn|rga' || find /usr/lib /lib -name 'librknnrt.so*' -o -name 'librga.so*' 2>/dev/null",
        "npu_freq": "for f in /sys/class/devfreq/*npu*/{governor,cur_freq,available_frequencies}; do echo $f; cat $f 2>/dev/null; done",
        "temperatures": "for f in /sys/class/thermal/thermal_zone*/temp; do echo -n \"$f \"; cat $f; done",
        "dma_heaps": "ls -l /dev/dma_heap 2>/dev/null", "rga": "cat /proc/rga/version 2>/dev/null",
    }
    result = {"serial": a.serial, "captured_at": time.strftime("%FT%T%z")}
    for name, shell in probes.items():
        result[name] = adb(a.serial, "shell", shell, capture=True, check=False).stdout
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_prepare_data(a):
    """Create deterministic, hashed and non-overlapping calibration/test lists."""
    train = sorted(p.resolve() for p in Path(a.train).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    test = sorted(p.resolve() for p in Path(a.test).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(train) < 500 or len(test) < 1000: raise ValueError("need >=500 train and >=1000 test images")
    # Ordering by the hash of the stable relative filename avoids filesystem-order bias.
    ranked = lambda paths: sorted(paths, key=lambda p: hashlib.sha256(p.name.encode()).digest())
    calibration, evaluation = ranked(train)[:500], ranked(test)[:1000]
    overlap = {p.name for p in calibration} & {p.name for p in evaluation}
    if overlap: raise ValueError(f"calibration/test overlap: {len(overlap)} filenames")
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    for name, paths in (("calibration_500.txt", calibration), ("test_1000.txt", evaluation)):
        target = output / name
        target.write_text("".join(f"{p}\n" for p in paths), encoding="utf-8")
        (output / f"{name}.sha256").write_text(f"{sha256(target)}  {name}\n")
    (output / "dataset_manifest.json").write_text(json.dumps({
        "train_root": str(Path(a.train).resolve()), "test_root": str(Path(a.test).resolve()),
        "selection": "sha256(filename), ascending", "calibration_count": 500, "test_count": 1000,
        "filename_overlap": 0}, indent=2), encoding="utf-8")


def export_optimized_onnx(name, weight, repo, python, destination):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    code = ("from ultralytics import YOLO; "
            f"YOLO({str(weight)!r}).export(format='rknn', imgsz=640, batch=1, dynamic=False, opset=14)")
    run([python, "-c", code], env=env)
    produced = weight.with_suffix(".onnx")
    if not produced.exists():
        candidates = sorted(weight.parent.glob(f"{weight.stem}*.onnx"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise RuntimeError(f"optimized ONNX export produced no file for {name}")
        produced = candidates[-1]
    shutil.copy2(produced, destination)


def onnx_schema(path):
    import onnx
    model = onnx.load(str(path))
    def value(v):
        shape = []
        for d in v.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param or "?")
        return {"name": v.name, "dtype": v.type.tensor_type.elem_type, "shape": shape}
    return {"inputs": [value(v) for v in model.graph.input], "outputs": [value(v) for v in model.graph.output]}


def cmd_convert(a):
    from rknn.api import RKNN
    output = Path(a.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    calibration = Path(a.calibration).resolve()
    lines = [Path(x.strip()) for x in calibration.read_text().splitlines() if x.strip()]
    if len(lines) != 500 or len(set(map(str, lines))) != 500:
        raise ValueError("calibration manifest must contain exactly 500 unique images")
    if any(not p.is_file() for p in lines): raise FileNotFoundError("calibration manifest contains missing images")
    repos = {"yolov8": Path(a.v8_repo), "yolo11": Path(a.v11_repo), "yolo26": Path(a.v26_repo)}
    records = []
    for name, family in MODELS.items():
        weight = Path(a.weights) / f"{name}.pt"
        if not weight.is_file(): raise FileNotFoundError(weight)
        model_dir = output / name; model_dir.mkdir(exist_ok=True)
        onnx_path = model_dir / f"{name}_rkopt.onnx"
        export_weight = model_dir / weight.name
        shutil.copy2(weight, export_weight)
        export_optimized_onnx(name, export_weight, repos[family], a.python, onnx_path)
        schema = onnx_schema(onnx_path)
        if len(schema["outputs"]) not in (6, 9):
            raise RuntimeError(f"{name}: expected Rockchip 6/9-head ONNX, got {len(schema['outputs'])}")
        rknn_path = model_dir / f"{name}_rk3588_int8.rknn"
        log = model_dir / "conversion.log"
        toolkit = RKNN(verbose=True)
        with log.open("w") as stream:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = stream
            try:
                rc = toolkit.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                                    target_platform="rk3588", quantized_dtype="asymmetric_quantized-8")
                if rc: raise RuntimeError(f"rknn.config={rc}")
                rc = toolkit.load_onnx(model=str(onnx_path));
                if rc: raise RuntimeError(f"rknn.load_onnx={rc}")
                rc = toolkit.build(do_quantization=True, dataset=str(calibration));
                if rc: raise RuntimeError(f"rknn.build={rc}")
                rc = toolkit.export_rknn(str(rknn_path));
                if rc: raise RuntimeError(f"rknn.export_rknn={rc}")
            finally:
                toolkit.release(); sys.stdout, sys.stderr = old_out, old_err
        labels = Path(a.labels).resolve()
        manifest = model_dir / "manifest.yaml"
        manifest.write_text(
            f"family: {family}\ntask: detect\ninput:\n  width: 640\n  height: 640\n  format: rgb\n  layout: nhwc\n"
            f"quantization: int8\nlabels: {labels}\npostprocess:\n  confidence: {a.confidence}\n"
            f"  nms_iou: {a.iou}\n  max_detections: 100\n  class_agnostic_nms: false\n", encoding="utf-8")
        records.append({"model": name, "family": family, "nms_free": family == "yolo26",
                        "weight": str(weight), "weight_sha256": sha256(weight),
                        "onnx": str(onnx_path), "onnx_sha256": sha256(onnx_path), "onnx_schema": schema,
                        "rknn": str(rknn_path), "rknn_sha256": sha256(rknn_path),
                        "rknn_bytes": rknn_path.stat().st_size, "calibration": str(calibration),
                        "calibration_sha256": sha256(calibration), "target": "rk3588", "int8": True})
    (output / "conversion_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def percentile(values, q):
    if not values: return math.nan
    ordered = sorted(values); pos = (len(ordered) - 1) * q
    lo = int(pos); hi = min(lo + 1, len(ordered)); frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def cmd_summarize(a):
    root = Path(a.input); rows = []
    for file in sorted(root.rglob("raw_timings.csv")):
        metadata_path = file.parent / "run.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        data = list(csv.DictReader(file.open()))
        row = dict(metadata); row["frames"] = len(data)
        for stage in STAGES:
            values = [float(x[stage]) for x in data]
            row.update({f"{stage}_{k}": v for k, v in {
                "mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0,
                "p50": percentile(values, .5), "p90": percentile(values, .9), "p95": percentile(values, .95),
                "p99": percentile(values, .99), "min": min(values), "max": max(values)}.items()})
        duration = sum(float(x["e2e_ms"]) for x in data) / 1000
        row["fps_serial_equivalent"] = len(data) / duration if duration else math.nan
        rows.append(row)
    if not rows: raise RuntimeError(f"no raw_timings.csv under {root}")
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def monitor(serial, remote_log):
    script = ("while true; do date +%s.%N; "
              "for f in /sys/class/thermal/thermal_zone*/temp /sys/class/devfreq/*npu*/cur_freq; do echo -n \"$f=\"; cat $f 2>/dev/null; done; "
              "cat /proc/loadavg; sleep 1; done")
    return subprocess.Popen(["adb", "-s", serial, "shell", script], stdout=open(remote_log, "w"), text=True)


def cmd_run(a):
    require_device(a.serial)
    artifacts = Path(a.artifacts).resolve(); output = Path(a.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    remote = a.remote.rstrip("/"); adb(a.serial, "shell", "mkdir", "-p", remote)
    adb(a.serial, "push", a.binary, f"{remote}/benchmark")
    adb(a.serial, "push", a.labels, f"{remote}/labels.txt")
    adb(a.serial, "push", a.nv12, f"{remote}/frame.nv12")
    adb(a.serial, "shell", "chmod", "+x", f"{remote}/benchmark")
    order = list(MODELS); order[1::2] = reversed(order[1::2])
    for round_index in range(1, a.rounds + 1):
        models = order if round_index % 2 else list(reversed(order))
        for name in models:
            family = MODELS[name]; local = artifacts / name
            adb(a.serial, "push", local / f"{name}_rk3588_int8.rknn", f"{remote}/{name}.rknn")
            adb(a.serial, "push", local / "manifest.yaml", f"{remote}/{name}.yaml")
            for mode in ("latency", "throughput"):
                for input_mode in ("dma-replay", "cpu-copy"):
                    run_dir = output / name / mode / input_mode / f"round_{round_index}"; run_dir.mkdir(parents=True)
                    mon = monitor(a.serial, run_dir / "monitor.log")
                    remote_csv = f"{remote}/raw.csv"
                    cmd = [f"cd {remote} && LD_LIBRARY_PATH=. ./benchmark", "--model", f"{name}.rknn",
                           "--manifest", f"{name}.yaml", "--labels", "labels.txt", "--family", family,
                           "--nv12", "frame.nv12", "--width", str(a.width), "--height", str(a.height),
                           "--mode", mode, "--input-mode", input_mode, "--warmup", str(a.warmup),
                           "--iterations", str(a.iterations), "--seconds", str(a.seconds), "--output", remote_csv]
                    result = adb(a.serial, "shell", " ".join(cmd), capture=True, check=False)
                    mon.terminate(); mon.wait(timeout=5)
                    if result.returncode: raise RuntimeError(result.stdout)
                    match = re.search(r"frames=(\d+) seconds=([0-9.]+) fps=([0-9.]+)", result.stdout)
                    adb(a.serial, "pull", remote_csv, run_dir / "raw_timings.csv")
                    (run_dir / "stdout.log").write_text(result.stdout)
                    (run_dir / "run.json").write_text(json.dumps({"model": name, "family": family,
                        "mode": mode, "input_mode": input_mode, "round": round_index,
                        "nms_free": family == "yolo26", "serial": a.serial,
                        "measured_frames": int(match.group(1)) if match else None,
                        "wall_seconds": float(match.group(2)) if match else None,
                        "measured_fps": float(match.group(3)) if match else None}, indent=2))


def parser():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("preflight"); q.add_argument("--serial", default="10.153.18.29:5555"); q.add_argument("--output", required=True); q.set_defaults(func=cmd_preflight)
    q = sub.add_parser("prepare-data"); q.add_argument("--train", required=True); q.add_argument("--test", required=True); q.add_argument("--output", required=True); q.set_defaults(func=cmd_prepare_data)
    q = sub.add_parser("convert"); q.add_argument("--weights", required=True); q.add_argument("--calibration", required=True); q.add_argument("--labels", required=True); q.add_argument("--output", required=True)
    q.add_argument("--v8-repo", required=True); q.add_argument("--v11-repo", required=True); q.add_argument("--v26-repo", required=True); q.add_argument("--python", default=sys.executable); q.add_argument("--confidence", type=float, default=.25); q.add_argument("--iou", type=float, default=.45); q.set_defaults(func=cmd_convert)
    q = sub.add_parser("run"); q.add_argument("--serial", default="10.153.18.29:5555"); q.add_argument("--artifacts", required=True); q.add_argument("--binary", required=True); q.add_argument("--labels", required=True); q.add_argument("--nv12", required=True); q.add_argument("--output", required=True); q.add_argument("--remote", default="/data/local/tmp/rk_yolo_bench"); q.add_argument("--width", type=int, required=True); q.add_argument("--height", type=int, required=True); q.add_argument("--warmup", type=int, default=100); q.add_argument("--iterations", type=int, default=3000); q.add_argument("--seconds", type=float, default=60); q.add_argument("--rounds", type=int, default=3); q.set_defaults(func=cmd_run)
    q = sub.add_parser("summarize"); q.add_argument("--input", required=True); q.add_argument("--output", required=True); q.set_defaults(func=cmd_summarize)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    try: args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
