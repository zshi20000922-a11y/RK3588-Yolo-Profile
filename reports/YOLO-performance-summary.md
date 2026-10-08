# RK3588 YOLO 性能与精度摘要

来源为本机已完成的板端报告；不同批次的性能数据按各报告注明的测试方式解释，不把单轮数据包装成三轮稳定结果。所有延迟均为 RKNN INT8、640×640、batch 1。

## 端到端吞吐

FPS 为 DMA 输入回放到检测结果的吞吐。单核为一个 context 固定 Core 0；三核为三个独立 context 分别固定 Core 0/1/2。

| 模型 | 单核 FPS | 三 context FPS | NPU ms（单核） | 后处理 ms（单核） | 说明 |
|---|---:|---:|---:|---:|---|
| YOLOv5n | 47.58 | 137.49 | 16.58 | 2.49 | Anchor decode + NMS |
| YOLOv5s | 31.52 | 94.03 | 26.07 | 2.76 | Anchor decode + NMS |
| YOLOv5nu | 39.95 | 122.70 | 20.90 | 0.96 | DFL decode + NMS；精度基线待复核 |
| YOLOv5su | 26.44 | 79.96 | 32.61 | 1.61 | DFL decode + NMS；精度基线待复核 |
| YOLOv8n | 56.90 | 156.84 | 15.21 | 0.26 | 三轮 Profile |
| YOLOv8s | 32.09 | 87.80 | 28.80 | 0.26 | 三轮 Profile |
| YOLO11n | 47.27 | 130.45 | 18.66 | 0.27 | 三轮 Profile |
| YOLO11s | 28.14 | 75.10 | 33.12 | 0.21 | 三轮 Profile |

YOLOv5/v5u 数据来自记录的三轮回放；v8/v11 来自三轮独立 Profile。FPS 包含 RGA 与 CPU 后处理，不等同于 NPU-only 速度。YOLO26 的完整吞吐数字未在本表发布，以避免将历史导出/精度问题与有效模型结果混杂；其独立后处理 A/B 见下文。

## COCO1000 精度（仅列可用比较）

测试为 1000 张互不重复 COCO val2017 图片，量化校准集为不重合的 500 张 COCO train 图片。AP 为 COCO AP50-95。

| 模型 | PT FP32 AP | RKNN INT8 AP | INT8−FP32 |
|---|---:|---:|---:|
| YOLOv5n | 0.34551 | 0.25212 | -0.09339 |
| YOLOv5s | 0.42114 | 0.33900 | -0.08214 |
| YOLOv8n | 0.36766 | 0.34308 | -0.02458 |
| YOLOv8s | 0.44461 | 0.41650 | -0.02811 |
| YOLO11n | 0.39437 | 0.36517 | -0.02920 |
| YOLO11s | 0.46473 | 0.43766 | -0.02707 |

YOLOv5u 的 PT 预测 JSONL 曾与对应 YOLOv5 n/s 结果逐框完全相同，故只保留速度，不提供不可信的精度变化结论。YOLO26 历史错误精度和旧导出结果未复制到本仓库；发布的 YOLO26 文档只涉及经相同输出校验的后处理耗时 A/B，不报告 mAP。

## YOLO26 后处理 A/B

当前被验证的 score-sum raw-head 路径仍是 one-to-many，需要分类别 NMS。量化域早筛、NEON 类别 argmax 与小容量候选向量将原标量后处理均值从 8.216/8.671 ms 降至 3.064/3.320 ms（n/s）；1000 张输入的检测 JSONL 新旧逐条一致。完整方法、分位数和原始逐帧 CSV 见 [`yolo26-postprocess/REPORT.md`](yolo26-postprocess/REPORT.md)。

## 可解释的选择

- 吞吐优先：本批次 YOLOv8n 单核与三 context 吞吐最高。
- v8/v11 的量化 AP 损失较小；n/s 的最终选择仍应在业务小目标测试集上看召回率与漏检，而非只看 COCO 总 AP。
- YOLOv5n/s 的 INT8 AP 损失明显较大；v5u 精度比较因 FP32 基线异常不做推荐。
