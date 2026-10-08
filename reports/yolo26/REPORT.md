# YOLO26n/s on RK3588: INT8 accuracy and board performance

## Tested path and interpretation

These results are for the validated RKNN INT8 **raw-head, one-to-many** export at model input 640×640, batch 1. The RKNN graph returns nine raw head tensors; CPU code performs score-sum candidate filtering, box decoding, and class-wise NMS. This is not the official YOLO26 one-to-one `[1, 300, 6]` NMS-free output. The latter was not accepted as a valid deployment path, so its speed/accuracy is intentionally absent.

All board FPS below were measured on RK3588 with a preloaded 1920×1080 NV12 DMA-BUF replay. RGA converts/resizes/letterboxes into the RKNN input, followed by NPU execution, output synchronization and CPU postprocessing to detections. File transfer and model initialization are excluded. This is a **board replay end-to-end (E2E) detection throughput**, not camera capture FPS. No V4L2 camera-to-results YOLO26 FPS is claimed.

## Board performance

| Model | Mode | NPU inference (ms) | Postprocess (ms) | E2E (ms) | E2E FPS |
|---|---|---:|---:|---:|---:|
| YOLO26n | 1 context, Core 0 | 24.33 | 7.89 | 36.02 | 27.70 |
| YOLO26n | 3 contexts, Core 0/1/2 | 26.78 | 5.95 | 35.83 | 83.46 |
| YOLO26s | 1 context, Core 0 | 44.25 | 8.62 | 56.82 | 17.57 |
| YOLO26s | 3 contexts, Core 0/1/2 | 47.66 | 7.97 | 59.12 | 50.66 |

The NPU-only rate can be estimated as `1000 / inference_ms`: n ≈41.1/37.3 FPS and s ≈22.6/21.0 FPS for one/three-context measurements. These are **inference-stage reciprocal estimates**, not end-to-end rates. E2E ms is per-frame latency; three-context FPS is aggregate throughput across the three concurrent workers, so these two columns are not mathematical inverses. Three contexts process three different frames concurrently; one frame is not split across three NPU cores.

The E2E replay values are the postprocess-optimized fixed-frame results: one 1000-frame measurement per model/mode. They are useful engineering measurements but are not the same repeatability tier as the three-round, ≥60-second v8/v11 runs. Per-frame measurements are preserved in [`raw/`](raw/). A separate optimization report records the postprocessing A/B method and exact detection equivalence at confidence 0.25 in [`reports/yolo26-postprocess`](../yolo26-postprocess/REPORT.md).

Postprocessing is candidate-dependent. On the fixed replay, it contributes about 6–9 ms after optimization. On a separate 1000-image natural COCO input, postprocess means were about 17.4 ms (n) and 27.0 ms (s) before the later NEON implementation; do not splice those natural-image timings into the fixed-replay FPS table. The latest NEON A/B on RK3588 reduced single-context postprocess from 8.216 to 3.064 ms (n) and 8.671 to 3.320 ms (s), while per-image detections were byte-identical. That A/B is a postprocessing-only measurement; a new full E2E throughput run with the NEON binary has not been reported here. The implementation and explanation of both rejected one-to-one export paths are in the [postprocessing/export report](../yolo26-postprocess/REPORT.md).

## COCO subset accuracy

The paired COCO evaluation used 1000 unique COCO val images, with a separate 500-image COCO train calibration set. Each test image is evaluated once per model/checkpoint; benchmark replay and warmup are performance tests, not additional accuracy samples. AP values are on the 0–1 scale and are subset metrics, not full COCO val2017 results.

| Model | PyTorch FP32 AP50-95 | RKNN INT8 AP50-95 | Change | AP50 FP32 → INT8 |
|---|---:|---:|---:|---:|
| YOLO26n | 0.41273 | 0.36373 | -0.04900 | 0.56910 → 0.52215 |
| YOLO26s | 0.48989 | 0.43238 | -0.05751 | 0.65290 → 0.60047 |

These are the corrected raw-head score-sum results. The repository deliberately excludes the earlier invalid decode results and rejected no-NMS/one-to-one experiments from its published precision and FPS tables. INT8 loss is material, particularly for s; evaluate recall and AP on the intended small-target application set before deployment.

## Conversion and reproduction

[`tools/convert_yolo26_rk3588.py`](../../tools/convert_yolo26_rk3588.py) exports a fixed-shape 640×640 RKOptimized ONNX from YOLO26n/s weights and builds RK3588 asymmetric INT8 RKNN with the supplied 500-image calibration manifest. It verifies that the resulting ONNX has the expected nine outputs and writes SHA-256/model-schema metadata. The exporter checkout must be the same patched Ultralytics/RKOptimized source used for the tested export; Toolkit version used for the reported run was 2.3.2.

The measurement implementation is [`tools/rk3588_yolo_benchmark.cc`](../../tools/rk3588_yolo_benchmark.cc), with the Y26 board-run wrapper [`tools/run_yolo26_document_board.py`](../../tools/run_yolo26_document_board.py). Build and ADB invocation details are in [`docs/board-inference.md`](../../docs/board-inference.md). Model weights, ONNX/RKNN binaries, COCO images, and image-level detections are not included in this public repository.
