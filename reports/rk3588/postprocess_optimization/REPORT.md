# RK3588 YOLO 后处理优化实测

> **版本边界（2026-11-09）：**此报告保存的是较早一批模型/优化过程记录。YOLOv5/v5u 的 conf=0.25 优化输出已用当前板端优化版重新复跑 1000 张，预测 JSONL SHA-256 与本报告存档一致，分项数字更新见 [`optimized_full_20261109/REPORT.md`](../optimized_full_20261109/REPORT.md)。本报告中的 YOLO26 score-sum 性能和逐框一致性来自不同 SHA 的早期实验 RKNN，不能用作当前正式模型结论。当前正式 Y26 模型的自然图 profile、三轮单/三 context 吞吐与重新计算的 COCOeval 见上述新报告；score-max 仍是独立实验模型。

## 结论

在同一块 RK3588、同一组按序排列的 1000 张 COCO 测试图片、对应实验批次 INT8 RKNN 模型、`640×640`、batch 1、单 context/Core 0、`conf=0.25` 条件下，针对 YOLOv5/v5u/YOLO26 的 CPU 后处理实现进行了优化。该实验批次中六个模型的逐图检测 JSONL 与其基线一致；当前 v5/v5u 输出再次复核一致。YOLO26 当前正式模型已重新导出/校验，需以 2026-11-09 报告为准。

对现有 RKNN 权重只改 CPU 后处理实现时，后处理均值下降约 18%–50%，六个模型均未在完整 1000 张集上低于 1 ms；YOLOv5n/u-n 分别为 1.115/1.044 ms。报告后段另列了需要重新导出 INT8 模型的 YOLO26 score-max 实验：在 `conf=0.25` 部署阈值下，YOLO26n 达到 0.776 ms、YOLO26s 为 1.175 ms。

## 方法改动

- 缓存 RKNN 输出 tensor 的通道平面地址与 stride，避免在热点循环里反复计算 layout、通道和 NC1HWC2 的 C2 分组偏移。
- YOLOv5 在读取类别前先用量化域 objectness 做早期阈值筛选；候选数组预留空间，避免动态扩容。
- NC1HWC2 类别最大值搜索采用 NEON 向量化路径；分数门控后仅对通过的格点做完整类别扫描。
- YOLO26/DFL 输出缓存 box 与 score 平面，减少布局寻址开销。
- NMS 使用交叉乘法比较 IoU 阈值，避免热点路径中逐对除法；保留原有严格阈值语义。
- 记录 decode/NMS 分项、候选数量、分数门控通过数、类别门控通过数及 NMS 比较次数。

这些改动只优化 host CPU 的后处理，不改变 RKNN 模型、量化参数、置信度阈值、NMS 阈值或检测输出语义。YOLO26 当前仍使用 raw-head one-to-many 输出与 class-wise NMS，并非 NMS-free one-to-one。

## 1000 张测试集结果

所有数值为 1000 张的算术平均，单位 ms。`E2E` 是板端采集帧进入处理流程至 Results 返回的测量值（含 RGA、NPU、输出同步及后处理；不含测试机磁盘读取）。不同模型 E2E 受 NPU 推理耗时影响，后处理优化不会显著改变其主导项。

| 模型 | 基线后处理 | 优化后后处理 | 优化后 decode | 优化后 NMS | 优化后 E2E | 后处理加速比 | JSONL 完全一致 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| YOLOv5n | 2.163 | **1.115** | 1.024 | 0.083 | 17.835 | 1.94× | 是 |
| YOLOv5s | 3.576 | **1.782** | 1.586 | 0.180 | 27.927 | 2.01× | 是 |
| YOLOv5u-n | 1.265 | **1.044** | 0.979 | 0.056 | 21.576 | 1.21× | 是 |
| YOLOv5u-s | 2.049 | **1.656** | 1.541 | 0.101 | 34.777 | 1.24× | 是 |
| YOLO26n | 3.020 | **2.552** | 2.460 | 0.079 | 29.159 | 1.18× | 是 |
| YOLO26s | 3.395 | **2.795** | 2.649 | 0.128 | 48.047 | 1.21× | 是 |

NMS 本身仅占优化后总后处理约 0.056–0.180 ms，继续只改 NMS 无法把 YOLO26 的后处理压到 1 ms。后续重点应放在候选解码和量化输出扫描。YOLO26n/s 当前每图约有 6.4k 个格点通过 score-sum 门控，但最终只有约 41/60 个候选进入 NMS，说明分数求和/扫描阶段有很大冗余。

## 质量与可比性说明

- 测试集：同一个固定、有序的 1000 图 NV12 manifest；基线和优化两轮各 1000 行。
- 输出：每个模型基线与优化的逐检测 JSONL 字节级相同，因此本次纯软件后处理实现变更没有改变预测结果；这不是重新训练/重新量化后的精度结论。
- 同一置信度阈值 `0.25`；RKNN 权重、输入图、NPU Core、预热设置均保持不变。
- 本报告验证的是优化前后推理结果一致性；不替代 FP32 对 INT8 的 mAP 量化精度评估。
- 原定 500 张 YOLO26 校准图在本机路径下已缺失。没有使用另一个校准集重建模型，以免混淆本次同模型、同精度的软件优化对照。

## YOLO26 score-max 门控实验

为减少 score-sum 门控后仍需读取的 80 类分数，导出额外的每格点最大类别分数输出，并在 CPU 侧使用带量化误差余量的保守筛选。YOLO26n/s 的 score-sum 与 score-max 对照均使用相同的 128 张 COCO128 train2017 图像校准；校准图与固定测试 1000 图 image ID 交集为 0。两个输出版本都在同一块 RK3588、同一输入顺序和相同参数上运行。

| 模型 | score-sum 后处理 | score-max 后处理 | decode：sum→max | score gate：sum→max | conf=0.25 逐图预测一致 |
|---|---:|---:|---:|---:|:---:|
| YOLO26n | 3.197 ms | **0.776 ms** | 3.102→0.700 ms | 8019→43 格点/图 | 是 |
| YOLO26s | 2.799 ms | **1.175 ms** | 2.649→1.025 ms | 6455→62 格点/图 | 是 |

同一 128 张校准集下，conf=0.001 的 COCOeval 指标也完全相同：YOLO26n AP50-95=0.363838、AP50=0.520770；YOLO26s AP50-95=0.432827、AP50=0.600851。有效预测数量亦相同。低阈值下 score-max 量化余量会让门控保守地放行所有 8400 个格点，因而这条优化针对部署阈值 `conf=0.25`；低阈值精度验证仍正确，但不会带来同等后处理加速。

这是使用 128 张校准图的独立实验模型，不替代原 500 张校准集正式 RKNN 结果，也不把它们的 AP 混作同量化设置排名。导出器及模型、ONNX、RKNN 哈希记录在 `artifacts/yolo26_scoremax_coco128/` 和 `artifacts/yolo26_scoresum_coco128/`。

## 上位机与 RKNN 后处理阈值审计

YOLOv5 PT FP32 对 RKNN INT8 的统一 COCO1000 精度口径配置相同：PT 预测脚本 `predict_coco_pt_jsonl.py` 默认 `conf=0.001, iou=0.45, max_det=100`；RKNN 精度运行传入 `conf=0.001`，`ModelConfig` 默认 `nms_iou=0.45, max_detections=100`，且两侧都是按类别 NMS。因此统一报告中 v5n/s 的 AP 损失（-0.09339/-0.08214）不是由这些阈值不一致造成的。

另一份 PC TensorRT/PT 基准 `tensorrt_pc_benchmark.py::validate_legacy` 对 YOLOv5 使用 `conf=0.001, iou=0.7, max_det=300`，和 RKNN 精度评估不同；若拿该表的绝对 AP 与 RKNN 表直接比较，会同时混入 NMS IoU 与最大检测数差异。另有 RKNN部署性能测试使用 `conf=0.25`，它不是 COCO AP 评估阈值。

## 文件

- 基线逐帧数据：`baseline_conf025/postprocess_base1000/<model>.csv` 与 `.jsonl`
- 优化逐帧数据：`optimized_conf025/<model>.csv` 与 `.jsonl`
- 优化 CSV 增加 decode/NMS/候选与门控计数列；基线 CSV 为旧 profiler 字段。
- 对应实现：`src/rkvs/detector_adapter.cc`、`src/rkvs/rknn_detector.cc`、`include/rkvs/detector_adapter.hpp`、`include/rkvs/types.hpp`、`tools/rk3588_yolo_quality.cc`

## 运行与后续优化建议

交叉编译验证命令：

```bash
cmake --build build-yolov5-cross --target rk3588_yolo_quality -j2
```

1 ms 是后处理 CPU 预算目标，不包含前处理、NPU 推理或端到端总时延。v5n/v5u-n 可视作接近目标，但不能宣称达标；要让 v5s 与 YOLO26 达到目标，建议评估模型图端输出每格最大类别分数（而不是 80 类 score-sum）并在 NPU 上做保守门控，同时以相同的 1000 张验证集确认没有漏检。该路径需要按明确记录的校准集重新构建 RKNN，并重新做 INT8 精度对照后才能作为正式结果。
