# RK3588 YOLO 性能评测与板端推理

本仓库汇总 YOLOv5、YOLOv5u、YOLOv8、YOLO11、YOLO26 在 RK3588 上的 INT8 推理研究，包含 DMA-BUF/RGA/RKNN C++ 推理代码、连板测量工具、运动算法测试及经过筛选的性能/精度报告。

> **结果口径先看这里：**下表的板端 FPS 是“预加载 NV12 DMA-BUF 回放 → RGA 预处理 → RKNN NPU → 输出同步 → CPU 后处理”的实测吞吐，不是摄像头采集 FPS，也不是单独的 NPU FPS。NPU-only FPS 是由实测 NPU 阶段均值取倒数得到的估算值。三 context 结果是三个独立任务各绑定一个 NPU 核同时处理不同帧，不代表一帧被三个核拆分。

## RK3588 性能总览

分辨率 640×640、batch 1、RKNN INT8。单 context 固定 Core 0；三 context 分别绑定 Core 0/1/2。

| 模型 | NPU 推理均值 (ms) | NPU-only 估算 (FPS)¹ | 连板回放 E2E，1 context (FPS) | 连板回放并发吞吐，3 contexts (FPS) | 后处理 (ms) |
|---|---:|---:|---:|---:|---:|
| YOLOv5n | 16.58 | 60.31 | 47.58 | 137.49 | 2.49 |
| YOLOv5s | 26.07 | 38.36 | 31.52 | 94.03 | 2.76 |
| YOLOv5nu | 20.90 | 47.85 | 39.95 | 122.70 | 0.96 |
| YOLOv5su | 32.61 | 30.67 | 26.44 | 79.96 | 1.61 |
| YOLOv8n | 15.210 | 65.75 | 56.90 | 156.84 | 0.261 |
| YOLOv8s | 28.796 | 34.73 | 32.09 | 87.80 | 0.257 |
| YOLO11n | 18.659 | 53.59 | 47.27 | 130.45 | 0.266 |
| YOLO11s | 33.124 | 30.19 | 28.14 | 75.10 | 0.211 |
| YOLO26n | 24.33 / 26.78 | 41.1 / 37.3 | 27.70 | 83.46 | 7.89 / 5.95 |
| YOLO26s | 44.25 / 47.66 | 22.6 / 21.0 | 17.57 | 50.66 | 8.62 / 7.97 |

¹ `1000 / NPU 推理均值(ms)`，仅表示模型执行阶段的倒数，不含 RGA、同步、后处理、排队或应用开销；不是端到端 FPS。YOLO26 的 E2E 性能是优化后的 RKNN raw-head one-to-many 模型在 RK3588 上的固定 NV12 DMA 回放结果；它是单次 1000 帧工程测试，不具备 v8/v11 三轮长测的重复性等级。其后处理仍包括 CPU NMS。逐帧数据与质量结果见 [YOLO26 完整报告](reports/yolo26/REPORT.md)。

- **连板回放 E2E**：基准程序运行在 RK3588 上。计时覆盖 DMA-BUF 输入回放、RGA 色彩转换/resize/letterbox、RKNN 推理、输出同步、解码/筛选/NMS（或模型相应的检测过滤）到检测结果。模型和 DMA 输入预先加载；ADB 部署、文件传输不在逐帧计时中。它不是 V4L2 摄像头实时链路的采集到显示 FPS。
- **1 context** 是单帧串行延迟/吞吐口径；**3 contexts** 是并行不同帧的设备总吞吐。两者用途不同，不能用三 context FPS 代表单帧时延。
- YOLOv5/v5u 数据来自已有三轮回放汇总；YOLOv8/11 结果来自三轮独立 Profile。v8/v11 的完整分项和资源记录见 [v8/v11 报告](reports/v8-v11/REPORT.md)。跨批次 FPS 作为方向性比较；严谨复测请用仓库 benchmark 在同一轮次、同一输入下重跑。
- 当前发布结果不含相机 V4L2 → 检测 → 显示/编码完整链路 FPS。运动算法报告中的 camera FPS 是相机/运动处理吞吐，不应当误读为 YOLO 检测 FPS。

## COCO 精度对比

AP50-95 使用一组固定、互不重复的 1000 张 COCO val2017 图片评估；RKNN INT8 量化校准使用独立的 500 张 COCO train 图片。表中数值为 AP（0–1），`Δ` = RKNN INT8 − PyTorch FP32。数据代表本次固定子集，不等同于完整 COCO val2017 官方指标。

| 模型 | PyTorch FP32 AP50-95 | RKNN INT8 AP50-95 | Δ | 结论/质量状态 |
|---|---:|---:|---:|---|
| YOLOv5n | 0.34551 | 0.25212 | -0.09339 | 量化损失较大，部署前应重新校准并复测 |
| YOLOv5s | 0.42114 | 0.33900 | -0.08214 | 量化损失较大，部署前应重新校准并复测 |
| YOLOv5nu | — | — | — | PT 基线预测曾与 YOLOv5 对应尺寸逐框完全相同；精度比较判为无效，不发布数值 |
| YOLOv5su | — | — | — | PT 基线预测曾与 YOLOv5 对应尺寸逐框完全相同；精度比较判为无效，不发布数值 |
| YOLOv8n | 0.36766 | 0.34308 | -0.02458 | 本次有效比较中量化损失较小 |
| YOLOv8s | 0.44461 | 0.41650 | -0.02811 | 本次有效比较中量化损失较小 |
| YOLO11n | 0.39437 | 0.36517 | -0.02920 | 本次有效比较中量化损失较小 |
| YOLO11s | 0.46473 | 0.43766 | -0.02707 | 本次有效比较中量化损失较小 |
| YOLO26n | 0.41273 | 0.36373 | -0.04900 | RKNN INT8 raw-head score-sum 结果；存在明显量化损失 |
| YOLO26s | 0.48989 | 0.43238 | -0.05751 | RKNN INT8 raw-head score-sum 结果；存在明显量化损失 |

YOLO26 的发布结果来自修正后的 raw-head one-to-many 路径，仍执行传统 NMS；并非官方 one-to-one/NMS-free 端到端模型。此前无效输出协议和 no-NMS 实验的精度数据不纳入本仓库公开精度表。FP32/RKNN INT8 精度、固定回放 FPS、后处理优化与测试边界见 [YOLO26 完整报告](reports/yolo26/REPORT.md) 和 [后处理 A/B 报告](reports/yolo26-postprocess/REPORT.md)。

## 板端推理文件与复现

主要连板推理、评估和 YOLO26 转换代码已纳入仓库，可从以下文件进入：

- [`tools/rk3588_yolo_benchmark.cc`](tools/rk3588_yolo_benchmark.cc)：板端 C++ DMA-BUF/NV12 回放基准；输出 copy、queue、RGA、NPU、输出同步、后处理和 E2E 逐帧时间。支持 CPU-copy 对照及单 context/三 context 模式。
- [`tools/rk3588_yolo_quality.cc`](tools/rk3588_yolo_quality.cc)：固定图片列表逐张上板推理，输出 COCO 检测 JSONL 与分阶段 timing。
- [`tools/rk3588_int8_pipeline.py`](tools/rk3588_int8_pipeline.py)：ADB 预检、校准/测试集清单、模型转换、部署运行、原始数据回收和统计汇总的编排脚本。
- [`tools/convert_yolo26_rk3588.py`](tools/convert_yolo26_rk3588.py)：从项目所用 RKOptimized Ultralytics 分支导出九输出 raw-head ONNX，并构建 RK3588 INT8 RKNN；导出 manifest 会明确标记 one-to-many/NMS 语义。
- [`tools/evaluate_coco_jsonl.py`](tools/evaluate_coco_jsonl.py)、[`tools/evaluate_coco_pair_jsonl.py`](tools/evaluate_coco_pair_jsonl.py)：检测结果 COCO AP 计算与成对比较。
- `src/rkvs/rknn_detector.cc`、`src/rkvs/detector_adapter.cc` 及 `include/rkvs/`：DMA-BUF 导入、RGA 预处理、RKNN context/核绑定、输出解码和各模型后处理适配。

详细依赖、构建、ADB 运行示例和口径说明见 [RK3588 连板推理指南](docs/board-inference.md)。注意：当前通用 C++ adapter 覆盖 YOLOv5 anchor-based、YOLOv8、YOLO11 和 YOLO26；YOLOv5u 虽有历史 FPS 摘要，但尚无与此通用 runner 对应的独立 adapter/有效精度基线，不要把它的摘要数字当成这份 runner 已复现的结果。

## 其他实验

- [640×640 运动算法](reports/motion-640/REPORT.md)、[2K](reports/motion-2k/REPORT.md)、[4K](reports/motion-4k/REPORT.md)：帧差/MOG2 的分辨率、帧率和资源数据。它们测的是运动处理而非 YOLO FPS。
- [YOLO 完整摘要](reports/YOLO-performance-summary.md)：早期结果及数据边界。

## 构建依赖与数据

复现需自行准备 RK3588、与系统匹配的 RKNN Runtime、授权的 RKNN API headers、RGA/MPP/OpenCV 运行依赖、模型文件和数据集。仓库不包含权重、`.rknn`、数据集、NV12 帧或逐图预测；Rockchip 带专有标识的 RKNN headers/runtime 不随仓库发布。SDK 路径通过 `RKNN_API_PATH`、`RKNN_RT_LIB`、`RGA_PATH`、`RGA_LIB`、`MPP_PATH`、`MPP_LIB` 配置。校准图片与评估图片应保持不重合，模型结果不要只看帧率，需同时看业务集召回率和精度。
