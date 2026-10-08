# RK3588 INT8 deployment models

This directory contains the ten reviewed RKNN binaries used for the repository's RK3588 YOLO comparisons. All are INT8, static 640×640, batch 1. Total size is about 70.6 MiB. SHA-256 verification:

```bash
cd models/rk3588-int8
sha256sum -c SHA256SUMS
```

| Model | File | Result / caveat |
|---|---|---|
| YOLOv5n/s | `yolov5n.rknn`, `yolov5s.rknn` | Anchor-based head, CPU decode/NMS |
| YOLOv5nu/su | `yolov5nu.rknn`, `yolov5su.rknn` | Ultralytics anchor-free models; FP32 baseline is not considered reliable, see reports |
| YOLOv8n/s | `yolov8n.rknn`, `yolov8s.rknn` | Rockchip optimized raw-head outputs, CPU decode/NMS |
| YOLO11n/s | `yolo11n.rknn`, `yolo11s.rknn` | Rockchip optimized raw-head outputs, CPU decode/NMS |
| YOLO26n/s | `yolo26n.rknn`, `yolo26s.rknn` | Nine-output score-sum, one-to-many raw-head; CPU decode and class-wise NMS |

For accuracy, latency, and FPS associated with each artifact, see [the results overview](../../README.md), [YOLO26 details](../../reports/yolo26/REPORT.md), and [v8/v11 profile](../../reports/v8-v11/REPORT.md). Use with a compatible RKNN Runtime/RKNPU driver and the matching family adapter in `src/rkvs/detector_adapter.cc`; a `.rknn` model alone does not define application preprocessing or postprocessing.

## Scope and provenance

These are the board-validated INT8 deployment artifacts for the listed benchmark paths—not the original `.pt` checkpoints or ONNX intermediates. The rejected YOLO26 one-to-one and one-to-many/no-NMS test artifacts are intentionally excluded. Checksum verification establishes file identity; it does not replace device/runtime compatibility checks.

The binaries are derivatives of upstream YOLO model weights. Their distribution is separate from this repository's Apache-2.0 source-code license. They are included only under the project's confirmed authorization to redistribute these artifacts. Respect the applicable upstream/model licenses and RKNN tool/runtime terms.
