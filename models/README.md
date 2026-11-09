# Model and export artifacts

This directory includes the ten PT checkpoints used by the host benchmark and two ONNX export sets for YOLOv5/v5u/v8/11/26 n/s:

- `pt/`: the exact PT files used for the paired PT FP32 / TensorRT FP16 desktop evaluation.
- `onnx/desktop-export/`: ONNX exports produced by the desktop TensorRT benchmark workflow. They are ONNX files only; TensorRT `.engine` binaries are intentionally not included.
- `onnx/rknn-input/`: ONNX graphs supplied to the RKNN INT8 conversion path. The YOLO26 files are the validated nine-output raw-head one-to-many exports, not the failed one-to-one export. External-data sidecars (`*.onnx.data`) are required parts of their corresponding ONNX models and must remain beside them.
- `rk3588-int8/`: board-validated RKNN deployments, documented separately.

The desktop ONNX exports and RKNN-input ONNX graphs are different artifacts where export paths differ; do not substitute one for the other without checking the expected output schema. The relevant export/quantization tools are in [`../tools/`](../tools/), and benchmark conditions/results are described in [`../reports/`](../reports/).

Verify the PT and ONNX artifact contents from the repository root:

```bash
sha256sum -c models/SHA256SUMS
```

The TensorRT engine is not distributed because it is specific to the target GPU, driver, and TensorRT runtime; rebuild it with `tools/tensorrt_pc_benchmark.py`. COCO images are not mirrored here. Download COCO from its official source and use the fixed manifests and hashes in the reports to reconstruct the evaluation subset. Dataset images and model artifacts remain subject to their upstream licenses and terms; this project's source license does not replace those terms.
