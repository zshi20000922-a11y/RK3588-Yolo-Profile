# PT / TensorRT / RKNN YOLO 性能与精度对比

## 结论先看

- **精度保持：**本次 RTX 5090 TensorRT FP16 相对其 PT FP32 基线的 mAP50-95 变化基本在 ±0.14 个 AP 点内；而 RK3588 INT8 对可用的 v5、v8、11、26 模型分别有不同程度的量化损失。v5 anchor-based 损失最大（约 8.2–9.3 AP 点）；v8/11 约 2.5–2.9 点；当前有效的 YOLO26 raw-head/score-sum 约 4.9–5.8 点。
- **PC 推理：**TensorRT FP16 在 RTX 5090 上的模型执行均值为 0.264–0.585 ms。YOLOv5n 最快（0.264 ms，约 3784 纯推理 FPS）；YOLO26n 虽然模型执行约 0.480 ms，但 NMS-free TRT 导出使后处理约 0.127 ms。
- **板端部署：**RKNN INT8 在 RK3588 上，当前长测中 YOLOv8n 的 DMA 回放到检测结果吞吐最高：单 context 56.90 FPS、三 context 聚合 156.84 FPS。三 context 是三个独立 context 并行处理不同帧，不是单帧跨三核。
- **不可直接做绝对速度排名：**PC PT/TRT 跑在 RTX 5090；RKNN 跑在 RK3588。输入和端到端边界也不同：TRT 是磁盘文件读图到 Results，RKNN 是预加载 NV12 DMA-BUF/RGA 到检测结果。下面并列呈现指标，但不能把它解释成纯后端的公平 A/B。

## 精度：分别看各后端相对自己的 PT FP32 基线

测试集按 COCO val2017 `image_id` 核对为相同的 1000 张、顺序也相同。TensorRT 清单与 RKNN 的 relocated 清单逐图文件字节校验全部一致；较早的 RKNN v8/v11 清单仍指向已不存在的旧数据根目录，当前能确认其 1000 个有序 image ID 一致，但无法重新读取旧路径下的历史文件做字节级复核。若需要对历史 v8/v11 实测输入做字节级审计，应使用当前 relocated 清单重跑 RKNN 精度验证。

TensorRT/PT 与 RKNN/PT 的评估器参数不同：TensorRT 使用 Ultralytics validator、`max_det=300`；RKNN质量报告使用 COCOeval、`max_det=100`。因此**绝对 AP 不宜跨这两张表直接比较**，更应看每张表内部的量化/导出差值。

### RTX 5090：TensorRT FP16 对 PT FP32

| 模型 | PT FP32 mAP50-95 | TensorRT FP16 mAP50-95 | Δ AP（FP16−PT） |
|---|---:|---:|---:|
| YOLOv5n | 0.285（旧版验证器仅三位小数） | 0.287 | +0.002（精度受旧验证器舍入限制） |
| YOLOv5s | 0.378（旧版验证器仅三位小数） | 0.376 | -0.002（精度受旧验证器舍入限制） |
| YOLOv5nu | 0.35241 | 0.35194 | -0.00047 |
| YOLOv5su | 0.43364 | 0.43290 | -0.00074 |
| YOLOv8n | 0.37844 | 0.37800 | -0.00044 |
| YOLOv8s | 0.46225 | 0.46168 | -0.00057 |
| YOLO11n | 0.40464 | 0.40463 | -0.000003 |
| YOLO11s | 0.47865 | 0.47733 | -0.00132 |
| YOLO26n | 0.41369 | 0.41334 | -0.00035 |
| YOLO26s | 0.49153 | 0.49146 | -0.00007 |

### RK3588：RKNN INT8 对其 PT FP32 基线

| 模型 | PT FP32 mAP50-95 | RKNN INT8 mAP50-95 | Δ AP（INT8−PT） | 说明 |
|---|---:|---:|---:|---|
| YOLOv5n | 0.34551 | 0.25212 | -0.09339 | 量化损失大 |
| YOLOv5s | 0.42114 | 0.33900 | -0.08214 | 量化损失大 |
| YOLOv5nu | 0.34551† | 0.32097 | 不作结论 | PT 预测曾与 v5n 逐框完全相同，基线不可信 |
| YOLOv5su | 0.42114† | 0.39334 | 不作结论 | PT 预测曾与 v5s 逐框完全相同，基线不可信 |
| YOLOv8n | 0.36766 | 0.34308 | -0.02458 |  |
| YOLOv8s | 0.44461 | 0.41650 | -0.02811 |  |
| YOLO11n | 0.39437 | 0.36517 | -0.02920 |  |
| YOLO11s | 0.46473 | 0.43766 | -0.02707 |  |
| YOLO26n | 0.41273 | 0.36373 | -0.04900 | 已修正 raw-head/score-sum 模型；非失败的 one-to-one 导出 |
| YOLO26s | 0.48989 | 0.43238 | -0.05751 | 已修正 raw-head/score-sum 模型；非失败的 one-to-one 导出 |

† v5u 的 PT 基线不能作为有效量化损失比较；RKNN INT8 AP 仅保留为该转换产物的观测值。

## 速度：分别说明纯模型执行与端到端吞吐

| 模型 | PT FP32 GPU推理 ms* | TensorRT FP16 GPU推理 ms† | TRT纯推理 FPS† | TRT磁盘读图→Results FPS‡ | RKNN单context NPU ms | RKNN DMA回放E2E FPS：单 / 三context |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv5n | ~1.400 | 0.264 | 3784 | 452.2 | 16.58 | 47.58 / 137.49 |
| YOLOv5s | ~1.400 | 0.346 | 2889 | 433.9 | 26.07 | 31.52 / 94.03 |
| YOLOv5nu | 1.445 | 0.347 | 2884 | 422.8 | 20.90 | 39.95 / 122.70 |
| YOLOv5su | 1.540 | 0.428 | 2337 | 406.5 | 32.61 | 26.44 / 79.96 |
| YOLOv8n | 1.373 | 0.330 | 3027 | 422.0 | 15.21 | 56.90 / 156.84 |
| YOLOv8s | 1.564 | 0.443 | 2260 | 398.9 | 28.80 | 32.09 / 87.80 |
| YOLO11n | 1.798 | 0.384 | 2605 | 407.3 | 18.66 | 47.27 / 130.45 |
| YOLO11s | 1.834 | 0.462 | 2162 | 396.1 | 33.12 | 28.14 / 75.10 |
| YOLO26n | 2.215 | 0.480 | 2084 | 442.2 | 24.33 | 27.70 / 83.46 |
| YOLO26s | 2.370 | 0.585 | 1710 | 421.8 | 44.25 | 17.57 / 50.66 |

* PT FP32 推理毫秒来自本次桌面验证器的 speed profile，是单次验证统计，不是单独的长时间流式 FPS；YOLOv5 原版只记录到 0.1 ms 精度。没有可比的 PT 磁盘读图/流式端到端基准，故不补造 PT E2E FPS。

† TRT 推理时延为 3 轮、每轮 1000 帧逐帧计时的均值；纯推理 FPS 为 `1000 / 均值毫秒`。其 GPU 为 RTX 5090。纯推理不含读图或 Results 处理。

‡ TRT 磁盘读图→Results 是应用路径端到端吞吐，三轮均值；RKNN 回放是预加载 NV12 DMA-BUF 经 RGA、NPU、同步和 CPU 后处理到检测结果。两者测试设备、输入源及运行时不同，只适合分别描述部署表现，不构成同设备 A/B。RKNN v5/v5u 为三轮短回放，v8/v11 为三轮 Profile；YOLO26 为历史单次 1000 帧回放，稳定性证据弱于其他组。

### YOLO26 后处理特别说明

TensorRT 使用官方 end-to-end 输出，后处理约 0.127 ms（n）/0.130 ms（s），不走传统 NMS。RKNN 上当前有效模型是九输出 raw-head/score-sum one-to-many，仍需要 CPU decode/NMS；旧版完整 E2E 回放记录的后处理为 7.89/8.62 ms。后处理优化 A/B 已将固定 COCO NV12 1000 图的单 context 均值降到 3.064/3.320 ms，且逐图检测结果与标量版完全一致，但尚未使用这版完整重跑 RKNN E2E / 三 context。因此当前 RKNN FPS 列没有把 3 ms A/B 代入，也没有宣称新 FPS。

## 复现与来源

- TensorRT：`reports/tensorrt-5090-2026-11-05/REPORT.md`、`summary.csv`、`quality_metrics.csv`、`raw_timings.csv`、`round_throughput.csv` 和 `environment.json`。硬件 RTX 5090，TensorRT 10.16.1.11，静态 640×640、batch 1。
- RKNN 汇总：见仓库中的 [RKNN 家族摘要](rk3588/optimized_full_20261109/REPORT.md) 和 [统一测试集报告](rk3588/unified_same_dataset_rknn/REPORT.md)。
- RKNN 精度：同一 benchmark 下 `results/quality_1000/`、`results/yolov5_quality_1000/`；YOLO26 有效精度使用 `reports/yolo26/REPORT.md` 中的 score-sum raw-head 结果，不使用失败的 one-to-one 或早期错误精度。
- 当前 RKNN 校准清单均为同一批 500 张图片、顺序一致；不同文件路径前缀导致清单文件 SHA 不同。TensorRT 与 relocated RKNN 测试清单为同一批 1000 张 COCO 图片、顺序一致且可访问的逐图文件字节一致；旧路径版 v8/v11 历史清单仅完成 image ID/顺序核对，旧图像目录已不可用，需重跑才能补做字节级审计。
- 所有模型输入为 640×640、batch 1。权重哈希在 TRT 环境记录和 RKNN 转换清单中一致。
