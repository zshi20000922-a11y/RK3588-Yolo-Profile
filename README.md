# RK3588 YOLO Profile

RK3588 上的 YOLO 推理、后处理与视频预处理基准项目。仓库包含 C++/RKNN 推理代码、DMA-BUF/RGA 输入路径、板端 profile 工具，以及经过口径说明的性能和 COCO 子集精度结果。

> **快速结论**：这里的 FPS 分为模型推理阶段估算、板端回放 E2E、三 context 并发吞吐和相机链路。它们不是同一个指标。表中 E2E 回放 FPS 覆盖预加载 NV12 DMA-BUF → RGA → RKNN NPU → CPU 后处理至检测结果；不含摄像头采集、显示或编码。

## 目录

- [结果摘要](#结果摘要)
- [精度对比](#精度对比)
- [YOLO26 后处理与导出结论](#yolo26-后处理与导出结论)
- [RKNN 逐层 Profile](#rknn-逐层-profile)
- [复现与工具](#复现与工具)
- [模型文件](#模型文件)
- [仓库结构](#仓库结构)
- [许可与第三方组件](#许可与第三方组件)

## 结果摘要

当前总览以 [PT / TensorRT / RKNN 综合报告](reports/YOLO-backend-comparison-PT-TRT-RKNN.md) 和 [优化后板端完整复测](reports/rk3588/optimized_full_20261109/REPORT.md) 为准。下表为 COCO val1000 固定图像清单上的 RKNN INT8 自然图单 context Profile：预载 NV12 → RGA → NPU → 输出同步 → CPU 后处理；不含相机采集、JPEG 解码、显示或存盘。FPS 是 `1000 / E2E mean` 的样本均值折算，不是相机长时帧率。

| 模型 | NPU 均值 ms | CPU 后处理 mean / P95 ms | E2E mean / P95 ms | 折算 FPS | 三 context NV12 回放总吞吐 FPS |
|---|---:|---:|---:|---:|---:|
| YOLOv5n | 15.620 | 1.111 / 2.151 | 18.024 / 21.966 | 55.48 | 145.30 |
| YOLOv5s | 25.071 | 1.751 / 2.805 | 28.636 / 30.876 | 34.92 | 98.77 |
| YOLOv5nu | 19.047 | 1.069 / 2.500 | 22.062 / 25.975 | 45.33 | 125.37 |
| YOLOv5su | 31.977 | 1.691 / 3.466 | 36.127 / 39.258 | 27.68 | 79.64 |
| YOLOv8n | 15.160 | 0.460 / 1.040 | 16.370 / 18.150 | 61.08 | 156.84 |
| YOLOv8s | 28.620 | 0.380 / 1.030 | 29.650 / 32.440 | 33.72 | 87.80 |
| YOLO11n | 18.680 | 0.500 / 1.190 | 20.070 / 22.140 | 49.82 | 130.45 |
| YOLO11s | 33.830 | 0.710 / 1.570 | 35.560 / 38.050 | 28.12 | 75.10 |
| YOLO26n score-sum | 24.369 | 2.380 / 2.727 | 29.305 / 33.201 | 34.12 | 92.81 |
| YOLO26s score-sum | 42.274 | 0.638 / 0.722 | 45.657 / 50.398 | 21.90 | 57.62 |

三 context 列是三个 context 分别绑定 NPU Core 0/1/2、并行处理不同帧的总吞吐，不是单帧三核协同延迟。该列来自固定 NV12 回放三轮长测，和上面的自然图逐帧 Profile 不是同一测试边界。TensorRT 数据来自桌面 RTX 5090，不是 RK3588 性能。相机/运动算法结果见 [640p](reports/motion-640/REPORT.md)、[2K](reports/motion-2k/REPORT.md)、[4K](reports/motion-4k/REPORT.md)。

详细分项、稳定性、处理定义和原始结果见 [PT / TensorRT / RKNN 综合报告](reports/YOLO-backend-comparison-PT-TRT-RKNN.md)、[优化后板端完整复测](reports/rk3588/optimized_full_20261109/REPORT.md)、[同测试集 RKNN 报告](reports/rk3588/unified_same_dataset_rknn/REPORT.md) 和 [TensorRT RTX 5090 原始报告](reports/tensorrt-5090-2026-11-05/REPORT.md)。

各模型 RKNN 单 context/Core 0 的逐算子性能明细（包括每层耗时、WorkLoad、shape 和读写量）见 [RKNN 逐层 Profile 报告](reports/layer-profiles/README.md)。

## RKNN 逐层 Profile

已将 YOLOv5/v5u/v8/v11/v26 的 n/s 共 10 份逐算子原始日志加入仓库：[逐层报告与汇总](reports/layer-profiles/README.md)。它们是启用 RKNN perf-detail 的独立插桩结果，不等同于主表的无插桩推理 FPS。

## 精度对比

AP50-95 使用同一组 1000 张 COCO val 图片评估；INT8 校准采用另外 500 张 COCO train 图片。表中是子集 AP（0–1），不是完整 COCO val2017 官方指标。`变化 = RKNN INT8 - PyTorch FP32`。

| 模型 | PyTorch FP32 AP50-95 | RKNN INT8 AP50-95 | 变化 | 说明 |
|---|---:|---:|---:|---|
| YOLOv5n | 0.28800* | 0.25212 | -0.03588* | YOLOv5 validator，日志精度为 0.001 |
| YOLOv5s | 0.37800* | 0.33900 | -0.03900* | YOLOv5 validator，日志精度为 0.001 |
| YOLOv5nu | 0.34407 | 0.32097 | -0.02310† | PT/RKNN 基线协议疑点，差值谨慎解读 |
| YOLOv5su | 0.42152 | 0.39334 | -0.02818† | PT/RKNN 基线协议疑点，差值谨慎解读 |
| YOLOv8n | 0.36997 | 0.34308 | -0.02689 | 同一验证图清单 |
| YOLOv8s | 0.44897 | 0.41650 | -0.03247 | 同一验证图清单 |
| YOLO11n | 0.39580 | 0.36517 | -0.03063 | 同一验证图清单 |
| YOLO11s | 0.46706 | 0.43766 | -0.02940 | 同一验证图清单 |
| YOLO26n | 0.40683 | 0.36250 | -0.04433 | 正式九输出 raw-head score-sum |
| YOLO26s | 0.48459 | 0.43819 | -0.04640 | 正式九输出 raw-head score-sum |

评估参数为 `imgsz=640, conf=0.001, IoU=0.45, max_det=100`。十个模型使用相同 COCO val2017 1000 张清单，但不同模型族的 validator 实现并非完全相同；完整限制见综合报告。`*` YOLOv5 legacy validator 仅报告三位小数；`†` v5u 曾有 PT/RKNN 对应基线异常，差值仅供参考。AP 变化用 0–1 表示，部署前应使用业务数据复核小目标召回率。

## YOLO26 后处理与导出结论

正式板端模型为九输出 raw-head score-sum one-to-many，CPU 仍负责解码和 NMS。最新正式模型 COCO val1000 Profile 的后处理均值为 YOLO26n/s **2.380 / 0.638 ms**（conf=0.25）；分位数、精度及三轮吞吐见[优化后完整复测报告](reports/rk3588/optimized_full_20261109/REPORT.md)。独立校准集导出的 score-max 实验不替代正式结果；旧 NEON A/B 数字也不与当前正式模型数据混用。one-to-one 导出问题见 [YOLO26 后处理/导出报告](reports/yolo26-postprocess/REPORT.md)。

已验证、用于性能和精度表的 YOLO26 RKNN 是**九输出 raw-head one-to-many**，CPU 仍执行 NMS；不是官方 one-to-one/NMS-free 模型。one-to-one 尝试的两个问题如下：

1. 完整 `[1,300,6]` 端到端 ONNX 可以生成 RKNN，但板端 `GatherElements` 出现越界索引；单个 INT8 输出还混合坐标、分数和类别 ID，量化精度不足，未通过正确性验收。
2. 保留 one-to-one head、去掉末端 TopK/Gather、拆成 boxes `[1,8400,4]` 与 scores `[1,8400,80]` 的导出，通过了 PyTorch 导出等价检查且 RKNN 构建成功；但当时板端 runner 不支持该双输出协议，1000 帧测试均未得到检测结果，故不发布其 AP/FPS。

不要将上述 one-to-one 路径与“把 one-to-many raw-head 当成 one-to-one、用 TopK 替代 NMS”的失败实验混淆。完整实验边界见 [YOLO26 后处理/导出报告](reports/yolo26-postprocess/REPORT.md)。

## 复现与工具

### 依赖

- RK3588 设备、匹配的 RKNPU driver / RKNN Runtime、RGA；本次转换 Toolkit 版本为 2.3.2。
- AArch64 GNU 工具链或板端原生 C++ 编译环境；COCO 图片、COCO annotations、模型权重由使用者自行准备。
- YOLO26 转换需要项目使用的 RKOptimized Ultralytics checkout；普通上游 Ultralytics 不保证支持仓库导出参数。

### 构建

以下命令示例使用本地 SDK 路径，实际路径按你的环境修改：

```bash
cmake --preset rkvs-minimal \
  -DRKNN_API_PATH=/opt/rockchip/librknn_api \
  -DRKNN_RT_LIB=/opt/rockchip/lib/librknnrt.so \
  -DRGA_PATH=/opt/rockchip/rga \
  -DRGA_LIB=/opt/rockchip/rga/lib/Linux/aarch64/librga.so
cmake --build build-rkvs-minimal --target rk3588_yolo_benchmark rk3588_yolo_quality -j4
```

### 常用入口

- [连板推理指南](docs/board-inference.md)：编译、ADB 部署、DMA-BUF 回放、计时字段和质量评估。
- [`tools/rk3588_yolo_benchmark.cc`](tools/rk3588_yolo_benchmark.cc)：单 context 延迟、三 context 并发吞吐与 CPU-copy 对照。
- [`tools/rk3588_yolo_quality.cc`](tools/rk3588_yolo_quality.cc)：固定测试清单逐图推理与 timing 输出。
- [`tools/rk3588_int8_pipeline.py`](tools/rk3588_int8_pipeline.py)：环境预检、固定数据清单、转换编排、上板运行和汇总。
- [`tools/convert_yolo26_rk3588.py`](tools/convert_yolo26_rk3588.py)：YOLO26 raw-head RKOptimized ONNX → RK3588 asymmetric INT8 RKNN。
- [`tools/evaluate_coco_pair_jsonl.py`](tools/evaluate_coco_pair_jsonl.py)：相同 image IDs 的 FP32 / INT8 成对 AP 评估。
- [`tools/run_optimized_rk3588_matrix.py`](tools/run_optimized_rk3588_matrix.py)、[`tools/run_optimized_natural_profile.py`](tools/run_optimized_natural_profile.py)、[`tools/run_optimized_coco_map.py`](tools/run_optimized_coco_map.py)：2026-11-09 优化后 RK3588 测试矩阵、自然图分项和 COCOeval 预测采集脚本。
- [`tools/tensorrt_pc_benchmark.py`](tools/tensorrt_pc_benchmark.py) 与 [`tools/validate_common_accuracy.py`](tools/validate_common_accuracy.py)：上位机 TensorRT 性能及 PT/TRT 同清单精度复核。

完整重跑需准备外部 COCO 数据、交叉编译产物和原始实验工作区；这些大体积数据不随仓库发布。可通过 `RK3588_ROOT`、`RK3588_YOLO_WORKSPACE`、`COCO_DATASET_ROOT`、`YOLOV5_LEGACY_ROOT` 指定本机路径；ADB 设备可用 `ANDROID_SERIAL` 选择。报告附有输入清单/内容哈希和模型哈希用于复核。

## 模型文件

已提供 10 个板端部署用 INT8 RKNN 文件（v5/v5u/v8/v11/v26 各 n/s），总计约 70.6 MiB，统一收录于 [`models/rk3588-int8/`](models/rk3588-int8/README.md)。该目录含文件 SHA-256 清单和模型语义说明，可用 `sha256sum -c models/rk3588-int8/SHA256SUMS` 校验。只发布有公开再分发授权的、通过对应输出协议和板端验证的 RKNN；原始 `.pt`、ONNX 中间文件、校准图片和无效 YOLO26 one-to-one 实验二进制不在本仓库。

## 仓库结构

```text
config/       RK3588 服务配置样例
docs/         板端部署和复现说明
include/      C++ API、RGA/MPP/RKNN 头文件
src/          推理、解码、DMA/RGA、视频流水线实现
tests/        核心和双摄服务测试
tools/        转换、板端 benchmark、精度评估脚本
reports/      性能、精度、运动算法报告和精简原始 CSV
models/       已验证的 RK3588 INT8 RKNN 部署文件
```

## 许可与第三方组件

项目源代码许可见 [`LICENSE`](LICENSE)，第三方组件说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。RKNN Toolkit/Runtime、Rockchip SDK、Ultralytics 代码及预训练权重各有自己的分发条款；项目 Apache-2.0 许可不自动覆盖这些组件或权重。发布模型文件前请核实相应授权。
