# RK3588 YOLO Profile

RK3588 上的 YOLO 推理、后处理与视频预处理基准项目。仓库包含 C++/RKNN 推理代码、DMA-BUF/RGA 输入路径、板端 profile 工具，以及经过口径说明的性能和 COCO 子集精度结果。

> **快速结论**：这里的 FPS 分为模型推理阶段估算、板端回放 E2E、三 context 并发吞吐和相机链路。它们不是同一个指标。表中 E2E 回放 FPS 覆盖预加载 NV12 DMA-BUF → RGA → RKNN NPU → CPU 后处理至检测结果；不含摄像头采集、显示或编码。

## 目录

- [结果摘要](#结果摘要)
- [精度对比](#精度对比)
- [YOLO26 后处理与导出结论](#yolo26-后处理与导出结论)
- [复现与工具](#复现与工具)
- [模型文件](#模型文件)
- [仓库结构](#仓库结构)
- [许可与第三方组件](#许可与第三方组件)

## 结果摘要

测试统一为 RK3588、RKNN INT8、模型输入 640×640、batch=1。单 context 固定 Core 0；三 context 是分别绑定 Core 0/1/2、同时处理不同帧的**整机吞吐**，不是一帧拆给三个 NPU 核。

| 模型 | NPU 推理均值 ms（单/三 context） | NPU-only 倒数估算 FPS（单/三 context） | 板端 E2E 回放 FPS（单/三 context） | 后处理 ms（单 context 最新优化 / 三 context） |
|---|---:|---:|---:|---:|
| YOLOv5n | 16.58 / 16.82 | 60.31 / 59.45 | 47.58 / 137.49 | 2.49 / 2.55 |
| YOLOv5s | 26.07 / 26.65 | 38.36 / 37.52 | 31.52 / 94.03 | 2.76 / 2.52 |
| YOLOv5nu | 20.90 / 20.61 | 47.85 / 48.52 | 39.95 / 122.70 | 0.96 / 0.90 |
| YOLOv5su | 32.61 / 33.27 | 30.67 / 30.06 | 26.44 / 79.96 | 1.61 / 1.26 |
| YOLOv8n | 15.21 / 16.56 | 65.75 / 60.39 | 56.90 / 156.84 | 0.26 / 0.34 |
| YOLOv8s | 28.80 / 31.56 | 34.73 / 31.69 | 32.09 / 87.80 | 0.26 / 0.36 |
| YOLO11n | 18.66 / 20.43 | 53.59 / 48.95 | 47.27 / 130.45 | 0.27 / 0.30 |
| YOLO11s | 33.12 / 37.33 | 30.19 / 26.79 | 28.14 / 75.10 | 0.21 / 0.28 |
| YOLO26n | 24.33 / 26.78 | 41.11 / 37.34 | 27.70 / 83.46 | **3.064 / 未复测** |
| YOLO26s | 44.25 / 47.66 | 22.60 / 20.98 | 17.57 / 50.66 | **3.320 / 未复测** |

- `NPU-only FPS` 是 `1000 / rknn_run 均值(ms)`，仅用于表示模型执行阶段速度；不含 RGA、同步和 CPU 后处理，**不是整机帧率**。
- YOLO26 的 E2E 回放 FPS 仍是较早版本的固定 NV12 输入、每配置单次 1000 帧结果；**最新 NEON 后处理尚未重新跑完整 E2E**。表内后处理列已更新为最新单 context A/B 均值；三 context 优化后数值尚未测量，故不沿用旧版数字。详见 [YOLO26 报告](reports/yolo26/REPORT.md)。
- 这些是预加载输入回放，不是 V4L2 相机到检测结果的实测 FPS。摄像头/运动算法 FPS 见 [640p](reports/motion-640/REPORT.md)、[2K](reports/motion-2k/REPORT.md)、[4K](reports/motion-4k/REPORT.md)。

详细分项、稳定性、后处理口径和每轮数据见 [v8/v11 报告](reports/v8-v11/REPORT.md)、[YOLO26 报告](reports/yolo26/REPORT.md) 与 [完整家族摘要](reports/YOLO-performance-summary.md)。

## 精度对比

AP50-95 使用同一组 1000 张 COCO val 图片评估；INT8 校准采用另外 500 张 COCO train 图片。表中是子集 AP（0–1），不是完整 COCO val2017 官方指标。`变化 = RKNN INT8 - PyTorch FP32`。

| 模型 | PyTorch FP32 AP50-95 | RKNN INT8 AP50-95 | 变化 | 说明 |
|---|---:|---:|---:|---|
| YOLOv5n | 0.34551 | 0.25212 | -0.09339 | 量化损失较大 |
| YOLOv5s | 0.42114 | 0.33900 | -0.08214 | 量化损失较大 |
| YOLOv5nu | — | 0.32097 | — | PT 基线复核前不报告量化损失 |
| YOLOv5su | — | 0.39334 | — | PT 基线复核前不报告量化损失 |
| YOLOv8n | 0.36766 | 0.34308 | -0.02458 | 本组子集比较 |
| YOLOv8s | 0.44461 | 0.41650 | -0.02811 | 本组子集比较 |
| YOLO11n | 0.39437 | 0.36517 | -0.02920 | 本组子集比较 |
| YOLO11s | 0.46473 | 0.43766 | -0.02707 | 本组子集比较 |
| YOLO26n | 0.41273 | 0.36373 | -0.04900 | raw-head score-sum 导出 |
| YOLO26s | 0.48989 | 0.43238 | -0.05751 | raw-head score-sum 导出 |

YOLOv5u 的历史 PT 预测与对应 YOLOv5 结果曾逐框完全相同，因此不将其 PT 基线作为有效比较。最终部署前应使用目标场景数据复测小目标召回率、误检和漏检。

## YOLO26 后处理与导出结论

当前最优后处理实现在 [`src/rkvs/detector_adapter.cc`](src/rkvs/detector_adapter.cc)：INT8/UINT8 score-sum 量化域早筛、AArch64 NEON 类别 argmax、候选向量预留和 NMS 框面积缓存。在 RK3588、1000 张相同 COCO NV12 输入、置信度 0.25 的后处理 A/B 中，YOLO26n/s 分别为 **3.064 / 3.320 ms**；新旧逐图检测输出完全一致。该值只代表后处理，不应和完整 E2E 帧率互换。

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
