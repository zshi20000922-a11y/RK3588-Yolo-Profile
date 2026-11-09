# NVIDIA TensorRT workstation benchmark — 2026-11-05

## Scope and method

- Hardware: NVIDIA GeForce RTX 5090, CUDA 12.8 PyTorch runtime, TensorRT 10.16.1.11.
- Ten checkpoints: YOLOv5/v5u/v8/11/26, n and s variants; static 640×640, batch 1.
- Accuracy: the fixed, non-overlapping COCO val1000 manifest used in the RKNN work (1000 images; 7476 labeled instances). PT FP32 is the per-checkpoint reference; TensorRT FP16 is the exported deployment engine.
- Validation thresholds: confidence 0.001, NMS IoU 0.7, max detections 300. YOLO26 uses its native end-to-end output path (no traditional NMS); its postprocess timing is the native end-to-end filter/Results path.
- Throughput: three repeated 1000-frame file-list runs, streaming `Results`, batch 1. `disk→Results FPS` includes image decode/load and Python/Ultralytics overhead; stage FPS is computed from per-frame pre/infer/post timings and excludes file I/O. Inference FPS is `1000 / inference_ms`.
- The TRT engine files are hardware/runtime-specific and are deliberately ignored by Git. The benchmark script rebuilds them locally.

## Results

| Model | PT32 mAP50-95 | TRT FP16 mAP50-95 | Δ AP | TRT infer ms (mean/P50/P95) | Infer-only FPS | Stage FPS | Disk→Results FPS (mean, range) | Post ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo11n | 0.40464 | 0.40463 | -0.000003 | 0.384 / 0.360 / 0.509 | 2605 | 814 | 407.3 (400.7–410.8) | 0.366 |
| yolo11s | 0.47865 | 0.47733 | -0.001321 | 0.462 / 0.451 / 0.557 | 2162 | 766 | 396.1 (390.9–401.7) | 0.366 |
| yolo26n | 0.41369 | 0.41334 | -0.000348 | 0.480 / 0.468 / 0.589 | 2084 | 938 | 442.2 (435.2–447.6) | 0.127 |
| yolo26s | 0.49153 | 0.49146 | -0.000070 | 0.585 / 0.574 / 0.678 | 1710 | 851 | 421.8 (419.0–427.0) | 0.130 |
| yolov5n | 0.28500 | 0.28700 | +0.002000 | 0.264 / 0.257 / 0.309 | 3784 | 956 | 452.2 (447.4–456.1) | 0.343 |
| yolov5nu | 0.35241 | 0.35194 | -0.000473 | 0.347 / 0.336 / 0.454 | 2884 | 862 | 422.8 (420.1–427.0) | 0.348 |
| yolov5s | 0.37800 | 0.37600 | -0.002000 | 0.346 / 0.337 / 0.435 | 2889 | 876 | 433.9 (431.8–438.1) | 0.354 |
| yolov5su | 0.43364 | 0.43290 | -0.000737 | 0.428 / 0.420 / 0.517 | 2337 | 802 | 406.5 (403.7–411.4) | 0.357 |
| yolov8n | 0.37844 | 0.37800 | -0.000440 | 0.330 / 0.322 / 0.416 | 3027 | 865 | 422.0 (420.0–424.7) | 0.359 |
| yolov8s | 0.46225 | 0.46168 | -0.000572 | 0.443 / 0.433 / 0.539 | 2260 | 776 | 398.9 (395.7–401.9) | 0.369 |

## Interpretation and caveats

- Disk→Results throughput is the application-like FPS. It is lower than summed-stage FPS because the former includes disk decode/load, Python scheduling and the Results iterator; do not call it pure TensorRT inference throughput.
- YOLO26 exports with output shape `(1, 300, 6)` and stays on the end-to-end, NMS-free detection path. Its postprocess is correspondingly shorter than the traditional NMS models.
- This run measures FP16 TensorRT, not INT8. PT FP32 is the accuracy baseline; no INT8 calibration dataset was substituted.
- The printed legacy YOLOv5 validator rounds AP to three decimal places. Its TensorRT-vs-PT delta is therefore only trustworthy at about 0.001 resolution.
- The measured PT baselines do not numerically match every older value in `reports/YOLO-performance-summary.md` (for example, the current YOLO11n is 0.40464 vs the old 0.39437). The 1000-image ID set is the same; prior validation implementation/environment differed. Use the paired PT FP32 vs TRT FP16 values in this report for TensorRT deltas, and re-run RKNN/PT together before making cross-backend accuracy claims.
- RTX 5090 FPS and RK3588 NPU FPS are measurements on different hardware and must not be compared as an apples-to-apples device ranking.

## Files

- `quality_metrics.csv`: per-model PT FP32 and TensorRT FP16 metrics and validator timing.
- `round_throughput.csv`: three external disk-to-Results throughput rounds per engine.
- `raw_timings.csv`: 3000 per-frame stage timing rows per model.
- `environment.json`: versions, GPU state, test manifest and model hashes.
- `logs/`: export and validation logs.
- `engines/`: local TensorRT engines/ONNX (ignored by Git).
