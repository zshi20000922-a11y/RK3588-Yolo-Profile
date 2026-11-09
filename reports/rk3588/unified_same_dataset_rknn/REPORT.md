# RKNN INT8 同测试集精度与端到端性能报告

日期：2026-11-05。本文统一 v5/v5u/v8/v11/v26 共 10 个 n/s RKNN INT8 模型的自然图像测试结果，并与 PT FP32、TensorRT 测试集做数据集一致性审计。

## 数据集一致性审计

- 测试集是 COCO val2017 的固定 1000 张图像；共 1000 个唯一 image ID，所有模型的逐图 timing 覆盖这相同的 ID 和顺序。
- TensorRT 测试清单 SHA-256：`11e1c1510ac323443c1e6bcff982c4154e90582756375904555e3207376c1fc8`。RKNN 当前重定位清单和 NV12 质量集 manifest 的 ID/顺序与之相同。
- ID/顺序 SHA-256：`be6a4f1d5e3c593917fbcaaab91053625b192f5a4aad2c1278bc953df50582e0`；按逐图 JPEG 内容计算的序列 SHA-256：`41e46a49623e50071cc4a2644841d71dd7e982e93bf6db7edadfcdabdc96ce2d`。
- 审计读取了当前 TensorRT 清单中的全部 JPEG，按 RKNN 预处理规则（BGR→I420/NV12、右侧补齐至 16 像素 stride、底部补齐至偶数、填充值 114）重新生成 NV12，与 1000 个已保存的板端 NV12 文件逐字节比较，全部一致。
- 精度均按 COCO val2017 标注和同一 1000 个 image ID 用 COCOeval、`max_det=100` 计算。RKNN 精度推理使用低置信度阈值以覆盖 AP 曲线；性能 profile 使用部署阈值 `conf=0.25`。两种阈值的用途不同，不混用。
- 校准集为固定的 COCO train2017 500 张，与该测试集不重叠。

## RKNN INT8 精度

AP 越高越好；变化为 `RKNN INT8 - PT FP32`。YOLOv5u 的 PT 预测文件此前审计发现与 v5 对应尺寸模型预测逐框完全相同，尽管 checkpoint 不同；因此它的 INT8 AP 可报告，但 PT 基线和量化损失不作有效结论。

| 模型 | PT FP32 AP50-95 | RKNN INT8 AP50-95 | 变化 | RKNN AP50 | 量化精度判断 |
|---|---:|---:|---:|---:|---|
| YOLOv5n | 0.28800* | 0.25212 | -0.03588* | 0.42073 | YOLOv5 validator 精度仅 0.001 |
| YOLOv5s | 0.37800* | 0.33900 | -0.03900* | 0.52443 | YOLOv5 validator 精度仅 0.001 |
| YOLOv5nu | 0.34407† | 0.32097 | †不解读 | 0.47466 | PT/RKNN 基线对应关系疑点 |
| YOLOv5su | 0.42152† | 0.39334 | †不解读 | 0.56126 | PT/RKNN 基线对应关系疑点 |
| YOLOv8n | 0.36997 | 0.34308 | -0.02689 | 0.49276 | 同一验证图清单 |
| YOLOv8s | 0.44897 | 0.41650 | -0.03247 | 0.58698 | 同一验证图清单 |
| YOLO11n | 0.39580 | 0.36517 | -0.03063 | 0.52041 | 同一验证图清单 |
| YOLO11s | 0.46706 | 0.43766 | -0.02940 | 0.60288 | 同一验证图清单 |
| YOLO26n | 0.40683 | 0.36250 | -0.04433 | 0.52564 | 正式 score-sum RKNN |
| YOLO26s | 0.48459 | 0.43819 | -0.04640 | 0.61321 | 正式 score-sum RKNN |

`*` v5 legacy validator 的 PT 结果只保留三位小数。v5u 的历史预测曾与 v5 对应模型逐框相同；该对应问题在本轮通过原始 v5u checkpoint 重新验证后仍无法作可靠误差归因，因此不解释其 PT−RKNN 差值。不同模型族的验证器实现仍不完全一致；如需审计级统一精度比较，应保存全部逐图预测并以同一版本 COCOeval 重新评估。

YOLO26 表格使用修正后的九输出 score-sum RKNN，而非早期错误解码的模型。该图输出 one-to-many raw heads，CPU 仍执行解码/NMS；不能称为 NMS-free one-to-one。

## RKNN 单 context 自然图 Profile

同一批 1000 个不同图像，每个图只纳入一次；输入是已经准备好的 NV12 DMA buffer。单 context 固定 NPU Core 0，640×640 batch=1。阶段时间单位 ms，格式为 `均值 / P50 / P95 / P99`。帧率为 `1000 / E2E 均值` 的单 context 样本均值折算值，不是三核并行吞吐，也不是三轮 60 秒稳态 FPS。

| 模型 | RGA预处理 均值 | NPU推理 均值 | 输出同步 均值 | CPU后处理 均值 / P95 | E2E 均值 / P50 / P95 / P99 | 折算帧率 |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv5n | 0.608 | 15.620 | 0.281 | **1.111 / 2.151** | 18.024 / 18.280 / 21.966 / 23.391 | 55.48 FPS |
| YOLOv5s | 0.781 | 25.071 | 0.429 | **1.751 / 2.805** | 28.636 / 28.777 / 30.876 / 33.398 | 34.92 FPS |
| YOLOv5nu | 0.686 | 19.047 | 0.283 | **1.069 / 2.500** | 22.062 / 22.006 / 25.975 / 28.519 | 45.33 FPS |
| YOLOv5su | 0.806 | 31.977 | 0.438 | **1.691 / 3.466** | 36.127 / 35.940 / 39.258 / 41.188 | 27.68 FPS |
| YOLOv8n | 0.43 | 15.16 | 0.07 | 0.46 / 1.04 | 16.37 / 16.48 / 18.15 / 18.73 | 61.08 FPS |
| YOLOv8s | 0.41 | 28.62 | 0.05 | 0.38 / 1.03 | 29.65 / 29.03 / 32.44 / 33.09 | 33.72 FPS |
| YOLO11n | 0.46 | 18.68 | 0.08 | 0.50 / 1.19 | 20.07 / 20.13 / 22.14 / 22.79 | 49.82 FPS |
| YOLO11s | 0.49 | 33.83 | 0.09 | 0.71 / 1.57 | 35.56 / 35.16 / 38.05 / 38.94 | 28.12 FPS |
| YOLO26n | 0.781 | 24.369 | 0.297 | **2.380 / 2.727** | 29.305 / 29.375 / 33.201 / 36.896 | 34.12 FPS |
| YOLO26s | 0.807 | 42.274 | 0.368 | **0.638 / 0.722** | 45.657 / 45.358 / 50.398 / 53.256 | 21.90 FPS |

E2E 定义为 NV12 buffer 就绪后，经 RGA 色彩/尺寸处理、RKNN 推理、输出同步及 CPU 后处理到检测结果；不含摄像头曝光/采集、JPEG 解码、显示和存盘。均值分项相加与 E2E 的差额来自计时边界/线程调度。折算 FPS 由逐图同步计时得到；固定帧 DMA 回放的单 context/三 context 三轮长测另见 [优化后完整复测](../optimized_full_20261109/REPORT.md)，不可与本表折算 FPS 混用。

## 性能解释

- v8/v11 的 CPU 后处理均值约 0.38–0.71 ms，P95 不超过 1.57 ms；它们是本次自然图集合中后处理成本最低的系列。
- YOLOv5/v5u 仍有 anchor/DFL 解码与 CPU NMS，后处理均值约 7–15 ms，且 P95 会随候选数量升高。
- 当前可用 YOLO26 score-sum 输出不是端到端 NMS-free。优化后正式模型自然图 n/s CPU 后处理均值为 2.380/0.638 ms；n 的 score_sum gate 平均放行 6,404 格点，而 s 约 7 格点，提示 n 的门控效果是后处理耗时差异来源。正式模型本轮 AP 和分项见 [优化后完整复测](../optimized_full_20261109/REPORT.md)。
- 其他报告中的约 47.58/137.49 FPS 等来自固定 NV12 单帧回放；三 context 项表示三条 context 并发处理不同帧的吞吐。它们不是本报告 1000 张自然图上的测量，不能与本表的折算 FPS 混成一列。相机采集上限也不包含在内。

## 结果文件与复现依据

- 测试图像/顺序：`benchmarks/rk3588_yolo_int8/yolov5_additions/data/test_1000_relocated.txt`
- NV12 输入及来源元数据：`benchmarks/rk3588_yolo_int8/quality_input/manifest.json`、`quality_input/input_list.txt`
- v5/v5u 精度和逐图时延：`results/yolov5_quality_1000/`
- v8/v11 精度和逐图时延：`results/quality_1000/`
- 修正后 YOLO26 精度和逐图时延：`results/yolo_postprocess_optimized/yolo26_scoresum_quality.csv`、`results/yolo_postprocess_optimized/quality_scoresum/`
- 优化后完整六模型复测（当前正式 v5/v5u/Y26模型 SHA-256、统一 COCOeval、三轮单/三 context、RGA/NPU/输出同步/后处理/E2E）：[`../optimized_full_20261109/REPORT.md`](../optimized_full_20261109/REPORT.md)
- 同测试集 TensorRT 环境和清单哈希：[TensorRT 环境记录](../../tensorrt-5090-2026-11-05/environment.json)
