#!/usr/bin/env python3
"""Export and INT8-quantize a raw-head YOLO26 model for RK3588.

This reproduces the validated one-to-many/raw-head path used by this project.
It is intentionally not the official one-to-one [1, 300, 6] NMS-free export.
The selected Ultralytics checkout must contain the RKNN/RKOptimized exporter
used for the reported model; see docs/board-inference.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> None:
    weight = args.weights.resolve()
    export_repo = args.ultralytics_repo.resolve()
    calibration = args.calibration.resolve()
    output = args.output.resolve()
    if not weight.is_file():
        raise FileNotFoundError(weight)
    if not export_repo.is_dir():
        raise NotADirectoryError(export_repo)
    if not calibration.is_file():
        raise FileNotFoundError(calibration)

    images = [Path(line.strip()) for line in calibration.read_text().splitlines() if line.strip()]
    if len(images) != 500 or len({str(p) for p in images}) != 500:
        raise ValueError("calibration manifest must contain exactly 500 unique images")
    missing = [str(p) for p in images if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing calibration images (first): {missing[0]}")

    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / f"{weight.stem}_rkopt.onnx"
    rknn_path = output / f"{weight.stem}_rk3588_int8.rknn"
    copied_weight = output / weight.name
    shutil.copy2(weight, copied_weight)

    # The project used the RKOptimized export implemented by this checkout.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(export_repo) + os.pathsep + env.get("PYTHONPATH", "")
    export_code = (
        "from ultralytics import YOLO; "
        f"model=YOLO({str(copied_weight)!r}); "
        "model.export(format='rknn', imgsz=640, batch=1, dynamic=False, opset=14)"
    )
    subprocess.run([args.python, "-c", export_code], cwd=output, env=env, check=True)
    produced = copied_weight.with_suffix(".onnx")
    if not produced.is_file():
        candidates = sorted(output.glob(f"{weight.stem}*.onnx"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise RuntimeError("RKOptimized export completed without producing an ONNX file")
        produced = candidates[-1]
    if produced.resolve() != onnx_path.resolve():
        shutil.copy2(produced, onnx_path)

    import onnx
    from rknn.api import RKNN
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    schema = {
        "inputs": [v.name for v in model.graph.input],
        "outputs": [
            {"name": v.name,
             "shape": [d.dim_value if d.dim_value else d.dim_param or "?"
                       for d in v.type.tensor_type.shape.dim],
             "dtype": v.type.tensor_type.elem_type}
            for v in model.graph.output
        ],
    }
    if len(schema["outputs"]) != 9:
        raise RuntimeError(
            f"expected the validated 9-output YOLO26 raw-head schema, got {len(schema['outputs'])}; "
            "do not run board inference with an unreviewed output schema"
        )

    conversion_log = output / "conversion.log"
    toolkit = RKNN(verbose=True)
    try:
        with conversion_log.open("w", encoding="utf-8") as log:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = log
            try:
                operations = (
                    ("config", lambda: toolkit.config(
                        mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                        target_platform="rk3588", quantized_dtype="asymmetric_quantized-8")),
                    ("load_onnx", lambda: toolkit.load_onnx(model=str(onnx_path))),
                    ("build", lambda: toolkit.build(do_quantization=True, dataset=str(calibration))),
                    ("export_rknn", lambda: toolkit.export_rknn(str(rknn_path))),
                )
                for operation, execute in operations:
                    rc = execute()
                    if rc:
                        raise RuntimeError(f"RKNN {operation} failed with code {rc}")
            finally:
                sys.stdout, sys.stderr = old_out, old_err
    finally:
        toolkit.release()

    manifest = {
        "model": weight.stem,
        "target": "rk3588",
        "input": {"width": 640, "height": 640, "batch": 1, "format": "RGB", "layout": "NHWC"},
        "quantization": "INT8 asymmetric",
        "export_path": "RKOptimized raw-head one-to-many",
        "output_semantics": "raw_head_one_to_many",
        "nms_free": False,
        "postprocess": "score-sum confidence filter + raw-head decode + class-wise NMS",
        "outputs": schema["outputs"],
        "files": {
            "weights": {"path": copied_weight.name, "sha256": sha256(copied_weight)},
            "onnx": {"path": onnx_path.name, "sha256": sha256(onnx_path)},
            "rknn": {"path": rknn_path.name, "sha256": sha256(rknn_path), "bytes": rknn_path.stat().st_size},
            "calibration_list": {"path": str(calibration), "sha256": sha256(calibration), "images": len(images)},
        },
        "exporter_repo": str(export_repo),
        "exporter_revision": subprocess.run(
            ["git", "-C", str(export_repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip() or None,
        "postprocess_note": "This is not the official one-to-one [1,300,6] NMS-free model.",
    }
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"onnx": str(onnx_path), "rknn": str(rknn_path),
                      "manifest": str(output / 'conversion_manifest.json')}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="YOLO26n/s .pt checkpoint")
    parser.add_argument("--ultralytics-repo", type=Path, required=True,
                        help="RKOptimized Ultralytics checkout used for the validated export")
    parser.add_argument("--calibration", type=Path, required=True,
                        help="500-image COCO train calibration list; paths must exist")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable,
                        help="Python with Ultralytics, ONNX, and RKNN Toolkit 2 installed")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
