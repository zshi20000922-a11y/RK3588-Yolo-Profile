#!/usr/bin/env python3
"""Build RK3588 INT8 RKNN artifacts for YOLOv5 and YOLOv5u additions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from rknn.api import RKNN


MODELS = {
    "yolov5n": {"family": "yolov5", "outputs": 3, "head": "anchor_based"},
    "yolov5s": {"family": "yolov5", "outputs": 3, "head": "anchor_based"},
    # YOLOv5u uses the Ultralytics anchor-free DFL head and therefore shares
    # the board decoder with YOLOv8, not the legacy YOLOv5 anchor decoder.
    "yolov5nu": {"family": "yolov8", "outputs": 9, "head": "anchor_free_dfl"},
    "yolov5su": {"family": "yolov8", "outputs": 9, "head": "anchor_free_dfl"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema(path: Path) -> dict:
    model = onnx.load(str(path), load_external_data=False)

    def value(item):
        return {
            "name": item.name,
            "shape": [d.dim_value or d.dim_param or "?" for d in item.type.tensor_type.shape.dim],
        }

    return {
        "opset": max(x.version for x in model.opset_import if not x.domain),
        "inputs": [value(x) for x in model.graph.input],
        "outputs": [value(x) for x in model.graph.output],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    records = []
    calibration = args.calibration.resolve()
    images = [Path(x) for x in calibration.read_text().splitlines() if x]
    if len(images) != 500 or len(set(images)) != 500 or not all(x.is_file() for x in images):
        raise RuntimeError("calibration list must contain 500 unique existing images")

    for name, spec in MODELS.items():
        onnx_path = (args.artifacts / f"{name}.onnx").resolve()
        pt_path = (args.artifacts / f"{name}.pt").resolve()
        model_schema = schema(onnx_path)
        if len(model_schema["outputs"]) != spec["outputs"]:
            raise RuntimeError(f"{name}: expected {spec['outputs']} outputs, got {len(model_schema['outputs'])}")
        rknn_path = args.artifacts / f"{name}_rk3588_int8.rknn"
        log_path = args.artifacts / f"{name}_conversion.log"
        toolkit = RKNN(verbose=True)
        try:
            with log_path.open("w") as log:
                import contextlib
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    rc = toolkit.config(
                        mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                        target_platform="rk3588", quantized_dtype="asymmetric_quantized-8",
                        optimization_level=3,
                    )
                    if rc: raise RuntimeError(f"{name}: rknn.config={rc}")
                    rc = toolkit.load_onnx(model=str(onnx_path))
                    if rc: raise RuntimeError(f"{name}: rknn.load_onnx={rc}")
                    rc = toolkit.build(do_quantization=True, dataset=str(calibration))
                    if rc: raise RuntimeError(f"{name}: rknn.build={rc}")
                    rc = toolkit.export_rknn(str(rknn_path))
                    if rc: raise RuntimeError(f"{name}: rknn.export_rknn={rc}")
        finally:
            toolkit.release()

        manifest_path = args.artifacts / f"{name}.yaml"
        manifest_path.write_text(
            f"family: {spec['family']}\ntask: detect\n"
            "input:\n  width: 640\n  height: 640\n  format: rgb\n  layout: nhwc\n"
            "quantization: int8\n"
            f"labels: {args.labels.resolve()}\n"
            "postprocess:\n  confidence: 0.25\n  nms_iou: 0.45\n"
            "  max_detections: 100\n  class_agnostic_nms: false\n",
            encoding="utf-8",
        )
        records.append({
            "model": name, **spec, "pt": str(pt_path), "pt_sha256": sha256(pt_path),
            "onnx": str(onnx_path), "onnx_sha256": sha256(onnx_path), "onnx_schema": model_schema,
            "rknn": str(rknn_path.resolve()), "rknn_sha256": sha256(rknn_path),
            "rknn_bytes": rknn_path.stat().st_size, "manifest": str(manifest_path.resolve()),
            "calibration": str(calibration), "calibration_sha256": sha256(calibration),
            "target": "rk3588", "precision": "int8",
        })
    (args.artifacts / "conversion_manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
