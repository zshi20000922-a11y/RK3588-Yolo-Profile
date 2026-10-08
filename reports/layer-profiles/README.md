# RK3588 RKNN 逐层耗时报告

本目录提供仓库中 10 个 YOLO INT8 模型的板端 RKNN `PERF_DETAIL` 逐算子明细。每份原始日志均包含层/算子名称、数据类型、执行目标、输入输出 shape、周期、单层耗时、MAC 利用率、NPU core WorkLoad 和每层读写量。

## 测量口径

- 模型输入为静态 `640×640`、batch 1、RKNN INT8；均为单 context 固定 NPU Core 0 的独立插桩 profile。
- RKNN Runtime 2.3.2、RKNPU driver 0.9.8、NPU 频率 1 GHz。测试来自同一块 RK3588 开发板，但采集日期分批，详见各家族报告。
- `逐层总时间` 是 RKNN perf-detail 中所有算子时间之和，包含输入/输出算子的 CPU I/O 时间；“NPU算子时间”是 NPU target 算子耗时之和。该插桩结果用于分析网络图，不等于无插桩的 NPU-only 延迟或整条 pipeline FPS，也不应直接替代主性能表。
- v5u 是 Ultralytics YOLOv5u（anchor-free/DFL），不是 YOLOv5 anchor-based。YOLO26 采用 2026-10-08 修正输出解释后的 RKNN raw-head one-to-many 模型；不是失败的 one-to-one/NMS-free 导出版本。

## 逐层总览

| 模型 | 逐层总时间 (ms) | NPU算子 (ms) | CPU输入/输出 (ms) | 单帧读写 (KB) | 原始逐算子日志 |
|---|---:|---:|---:|---:|---|
| YOLOv5n | 14.478 | 14.370 | 0.108 | 30,114.03 | [日志](raw/yolov5n_layer_perf.txt) |
| YOLOv5s | 32.070 | 31.622 | 0.448 | 58,729.78 | [日志](raw/yolov5s_layer_perf.txt) |
| YOLOv5nu | 18.001 | 17.880 | 0.121 | 32,463.15 | [日志](raw/yolov5nu_layer_perf.txt) |
| YOLOv5su | 30.652 | 30.247 | 0.405 | 63,532.15 | [日志](raw/yolov5su_layer_perf.txt) |
| YOLOv8n | 15.162 | 15.116 | 0.046 | 36,296.27 | [日志](raw/yolov8n_layer_perf.txt) |
| YOLOv8s | 29.802 | 29.679 | 0.123 | 72,183.77 | [日志](raw/yolov8s_layer_perf.txt) |
| YOLO11n | 19.869 | 19.742 | 0.127 | 40,049.45 | [日志](raw/yolo11n_layer_perf.txt) |
| YOLO11s | 34.698 | 34.558 | 0.140 | 78,395.41 | [日志](raw/yolo11s_layer_perf.txt) |
| YOLO26n | 20.344 | 20.296 | 0.048 | 40,427.02 | [日志](raw/yolo26n_layer_perf.txt) |
| YOLO26s | 40.674 | 40.525 | 0.149 | 81,970.04 | [日志](raw/yolo26s_layer_perf.txt) |

## 报告解释

- YOLOv5n/s、YOLOv5nu/su 的分解来自各自 `Total Operator Elapsed Per Frame` 与 operator ranking footer；v8/v11 的总计还收录于 [v8/v11 家族报告](../v8-v11/REPORT.md)。
- YOLO26 行来自 2026-10-08 文档方法重新导出后的逐层日志，图为修正逻辑 4 通道解析的 one-to-many raw-head。它们是单独插桩采集；不得与 [YOLO26 报告](../yolo26/REPORT.md) 中较早版本的 E2E 回放分项拼接，也不代表失败的官方 one-to-one 导出。
- perf-detail 测量在每个 RKNN op 上启用采集标志，插桩会增加耗时。尤其不同家族日志采集批次和图结构不同，跨模型比较应优先看无插桩、同口径的正式性能报告；这里的层时间适合定位算子和数据搬运热点。

## 各模型主要耗时算子

- YOLOv5n/s：`ConvExSwish` 占 NPU 计算的大头（约 71%–74%）；其次为检测头 `ConvSigmoid`、Concat。anchor-based 解码/NMS 在 CPU 后处理，不在这些 NPU layer timings 中。
- YOLOv5nu/su：主耗时同样是 `ConvExSwish`；为 DFL/anchor-free 图，CPU DFL 解码和 NMS 不包含在 NPU 逐层时间内。
- YOLOv8n/s、YOLO11n/s：`ConvExSwish` 为主要耗时；YOLO11 的 Attention 与 Concat 也有可见占比。详细逐算子行可在各自 raw 日志中查看。
- YOLO26n/s：主要算子为 `ConvExSwish`（n/s 分别约 67.52%/66.19%），其次 Attention（约 13.14%/13.07%）与 Concat（约 8.09%/8.17%）。这些网络侧数字不含 CPU 侧 raw-head 解码与分类 NMS。
