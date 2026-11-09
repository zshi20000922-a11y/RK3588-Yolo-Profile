# YOLOv5 / v5u / v8 / v11 / v26：PT、TensorRT、RKNN 对比

更新：2026-11-09。覆盖 10 个 n/s 模型。本文把桌面 RTX 5090 与 RK3588 的测量分开，避免将不同硬件和不同输入边界拼成一个“速度排名”。“tentort”按 TensorRT 理解。

## 结论摘要

- **RK3588 上实际测到的是 RKNN INT8。** TensorRT FP16 和 PT FP32 数据来自桌面 RTX 5090；本次没有在 RK3588 上运行 TensorRT，也没有将 PT CPU 推理冒充板端 NPU 结果。TensorRT 不能作为 RK3588 NPU 后端。
- **精度已按同一 COCO val1000 清单重测 PT/TensorRT**，统一 `imgsz=640, batch=1, conf=0.001, IoU=0.45, max_det=100`。RKNN 采用同一组 1000 张验证图，PT/RKNN 与 TRT 清单的 image ID 和顺序一致；已审计的当前清单图像内容与 NV12 测试输入一致。
- **速度不是同一个端到端定义。** 桌面 TRT 的应用吞吐为磁盘读图→Ultralytics Results；RKNN 的 E2E 为预载 NV12 DMA-BUF→RGA→NPU→同步→CPU 后处理。PT 没有同口径、长测的磁盘到 Results 基准。因此跨设备结果用于描述部署现状，不是公平硬件 A/B。
- **RKNN 单 context 的 1000 张自然图 E2E 最快仍为 YOLOv8n：16.37 ms/张，折算 61.08 FPS。** 本轮新增 v5/v5u/Y26 当前正式模型 optimized profile；固定 NV12 回放三 context 的总吞吐为 v5n 145.30、v5s 98.77、v5nu 125.37、v5su 79.64、Y26n 92.81、Y26s 57.62 FPS。三路 context 并行处理不同帧，不是单帧同时跨三核。
- **优化版后处理已用当前正式权重复测：**v5/v5u 为 1.07–1.75 ms；当前正式 score-sum YOLO26n/s 为 2.380/0.638 ms（自然图1000均值）。Y26n score-sum 门控约通过 6,404 格点/图，是它仍慢于 Y26s 的主要关联因素。score-max 的 0.776/1.175 ms 来自不同校准设置下的独立实验模型，不能替代正式结果。
- **INT8 量化损失仍需按模型族关注：**v5n/s 约 -3.6/-3.9 AP 点（PT 与 RKNN validator 不完全相同）；v5u 的 PT 基线异常，不解读 delta。统一 COCOeval 下当前正式 Y26n/s AP50-95 为 0.36250/0.43819，相对同报告 FP32 基线约 -0.050/-0.052。各后端验证器实现存在差异，细小 delta 只作近似对比。

## 统一精度评测

固定输入为 COCO val2017 的 1000 张图、7476 个标注实例，`640×640, batch=1`。本轮 PT/TensorRT 使用相同数据 yaml、图像清单和后处理参数；v5 原始 anchor-based 使用其 YOLOv5 validator，v5u/v8/v11/v26 使用 Ultralytics validator；RKNN 指标从同一测试清单的板端结果经 COCOeval 计算。数据集与阈值已统一，但 validator 实现并非同一套代码，且 v5 原始版只输出三位小数；因此细小 delta 应视为近似。若需审计级完全同一评估器，应将所有后端逐图预测统一映射为 COCO category ID 后用同一个 COCOeval 重算。

### mAP50-95 与相对 PT FP32 的变化

| 模型 | PT FP32 | TRT FP16 | TRT−PT | RKNN INT8 | RKNN−PT |
|---|---:|---:|---:|---:|---:|
| YOLOv5n | 0.28800* | 0.28900* | +0.00100* | 0.25212 | -0.03588* |
| YOLOv5s | 0.37800* | 0.37700* | -0.00100* | 0.33900 | -0.03900* |
| YOLOv5nu | 0.34407 | 0.34367 | -0.00040 | 0.32097 | -0.02310 |
| YOLOv5su | 0.42152 | 0.42095 | -0.00057 | 0.39334 | -0.02818 |
| YOLOv8n | 0.36997 | 0.36938 | -0.00058 | 0.34308 | -0.02689 |
| YOLOv8s | 0.44897 | 0.44831 | -0.00066 | 0.41650 | -0.03247 |
| YOLO11n | 0.39580 | 0.39576 | -0.00004 | 0.36517 | -0.03063 |
| YOLO11s | 0.46706 | 0.46567 | -0.00138 | 0.43766 | -0.02940 |
| YOLO26n | 0.40683 | 0.40643 | -0.00040 | 0.36250 | -0.04433 |
| YOLO26s | 0.48459 | 0.48473 | +0.00014 | 0.43819 | -0.04640 |

AP 是 0–1 比例；例如 `-0.03588` 相当于约 **-3.588 AP 点**。TensorRT 相对 PT 的变化均小于约 0.14 AP 点。RKNN 与 PT 的差值显示 INT8 量化退化，但不能只看精度决定部署，还应结合目标场景召回率和小目标表现。

**后处理可比性边界：**所有传统 NMS 模型使用相同 conf/IoU/max_det。YOLO26 的 PT/TensorRT 是原生 end-to-end/NMS-free 输出，`IoU=0.45` 不参与其常规后处理；当前 RKNN YOLO26 是 raw-head score-sum 输出，需要 CPU 解码及 NMS。因而 YOLO26 的阈值/max_det 对齐，但处理图结构并非完全相同，mAP 只作为统一数据与评估指标下的参考，不能宣称其后处理逐项等价。RKNN YOLO26 不是错误的 one-to-one 导出模型。

`*` YOLOv5 原始版验证器仅在控制台输出 3 位小数，表中用该分辨率计算 delta；其差值精度也只有约 ±0.001。其余 PT/TRT 数字来自 validator 的 COCO 风格检测指标。旧报告曾把 YOLOv5 PT 基线记为 0.34551/0.42114；本次相同 1000 图重新验证得到 0.288/0.378，因此本报告以重新验证值为准，旧值不再用于量化损失判断。YOLOv5u 旧 PT 预测曾与 v5 对应尺寸逐框相同的问题，本轮已从原始 v5u checkpoint 重新验证，采用下表新值。

### 后端对 PT 精度的影响

- **TensorRT FP16 vs PT FP32：**十个模型 mAP50-95 变化在 -0.00138 至 +0.00100 范围，精度基本保持。
- **RKNN INT8 vs PT FP32：**v5n/s、v5u/s、v8、11、26 变化如上表。v5n/s 和 v26s 是当前相对损失较大的组。
- 本报告不使用 YOLO26 失败的 one-to-one/错误输出解码精度；板端有效模型为修正后的九输出 raw-head/score-sum 版本。

## 桌面 RTX 5090 速度（PT / TensorRT）

PT/TensorRT 纯推理时延来自同一模型集的桌面 profile；TensorRT FP16 的每模型磁盘读图→Results 吞吐为 3 轮、每轮 1000 张。TensorRT 测试环境为 RTX 5090、TensorRT 10.16.1.11、640×640、batch 1。速度 profile 使用预测默认 `conf=0.25`（TensorRT NMS IoU 取引擎/运行时默认），与上面的精度阈值分开；该桌面数据不能当成 RK3588 性能。

| 模型 | PT FP32 推理 ms | TRT FP16 推理 ms | TRT 纯推理折算 FPS | TRT 磁盘→Results FPS |
|---|---:|---:|---:|---:|
| YOLOv5n | 1.400 | 0.264 | 3784 | 452.2 |
| YOLOv5s | 1.400 | 0.346 | 2889 | 433.9 |
| YOLOv5nu | 1.445 | 0.347 | 2884 | 422.8 |
| YOLOv5su | 1.540 | 0.428 | 2337 | 406.5 |
| YOLOv8n | 1.373 | 0.330 | 3027 | 422.0 |
| YOLOv8s | 1.564 | 0.443 | 2260 | 398.9 |
| YOLO11n | 1.798 | 0.384 | 2605 | 407.3 |
| YOLO11s | 1.834 | 0.462 | 2162 | 396.1 |
| YOLO26n | 2.215 | 0.480 | 2084 | 442.2 |
| YOLO26s | 2.370 | 0.585 | 1710 | 421.8 |

PT ms 来自桌面验证 profile；TensorRT infer ms 来自 GPU execution 计时；纯推理 FPS 为 `1000/ms`。TRT 磁盘→Results 包含磁盘读图、解码、预处理、推理、后处理和 Python/Results 开销，**不是纯推理 FPS**。PT 没有同口径磁盘→Results 长测，不补造该数字。

## RK3588 RKNN INT8 速度与前后处理

### 1000 张自然图：单 context 端到端

单 context 固定 NPU Core 0，输入是固定 COCO val1000 清单中的 NV12 DMA buffer，每模型每张图一次。E2E = RGA 预处理+NPU+输出同步+优化后 CPU 后处理，直到检测列表；排除相机采集、JPEG 解码、显示、存盘。折算 FPS=`1000/E2E mean`，是 1000 图样本均值折算，不是三轮长测 FPS。v5/v5u/正式 score-sum v26 使用本轮优化后完整分项，v8/v11 保留原 profile。

| 模型 | RGA ms | NPU ms | 输出同步 ms | CPU 后处理 mean / P95 ms（优化版） | E2E mean / P95 ms | 折算 FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv5n | 0.608 | 15.620 | 0.281 | **1.111 / 2.151** | 18.024 / 21.966 | 55.48 |
| YOLOv5s | 0.781 | 25.071 | 0.429 | **1.751 / 2.805** | 28.636 / 30.876 | 34.92 |
| YOLOv5nu | 0.686 | 19.047 | 0.283 | **1.069 / 2.500** | 22.062 / 25.975 | 45.33 |
| YOLOv5su | 0.806 | 31.977 | 0.438 | **1.691 / 3.466** | 36.127 / 39.258 | 27.68 |
| YOLOv8n | 0.43 | 15.16 | 0.07 | 0.46 / 1.04 | 16.37 / 18.15 | 61.08 |
| YOLOv8s | 0.41 | 28.62 | 0.05 | 0.38 / 1.03 | 29.65 / 32.44 | 33.72 |
| YOLO11n | 0.46 | 18.68 | 0.08 | 0.50 / 1.19 | 20.07 / 22.14 | 49.82 |
| YOLO11s | 0.49 | 33.83 | 0.09 | 0.71 / 1.57 | 35.56 / 38.05 | 28.12 |
| YOLO26n（score-sum） | 0.781 | 24.369 | 0.297 | **2.380 / 2.727** | 29.305 / 33.201 | 34.12 |
| YOLO26s（score-sum） | 0.807 | 42.274 | 0.368 | **0.638 / 0.722** | 45.657 / 50.398 | 21.90 |

**主要瓶颈：**v8/v11 的后处理约 0.38–0.71 ms。当前正式模型本轮实测 v5/v5u 后处理约 1.07–1.75 ms；YOLO26n 为 2.38 ms、Y26s 为 0.64 ms。Y26n score-sum 门控均值放行约 6,404 格点而 Y26s 约 7 格点，显示 n 分支分数扫描/门控是当前分项差异来源；Y26 NMS 均值仅 0.017 ms。v5/v5u 本轮预测 JSONL 与优化后存档逐字节一致。正式 Y26 模型 hash 和本轮重新 COCOeval 已记录在完整报告；旧优化实验目录中的 Y26 模型 hash 不同，不能用其逐框输出声明代替正式模型复核。E2E与阶段计时边界不同，合计差约 0.4–1.57 ms，原始逐帧结果可审计。

另有 YOLO26 score-max 输出门控实验，使用独立的 128 张 COCO128 校准集导出，不替代正式 500 张校准模型：conf=0.25 时 Y26n 后处理 0.776 ms、Y26s 1.175 ms，且与对应 score-sum 模型逐图预测一致；它不是上表正式量化精度/正式 E2E 排名的模型版本。

### 固定 NV12 DMA-BUF 回放：单 context / 三 context 优化版三轮长测

此处是当前优化版三轮长测：预载 NV12 DMA buffer，单 context 固定 Core 0；三个 context 分别绑定 Core 0/1/2 并发处理不同帧。每轮至少 3000 帧且至少 60 秒，运行顺序轮换/反向交错；NPU debugfs 采样确认三核均参与。**三核列是三路总吞吐，不是单帧延迟除以三，也不含相机采集帧率上限。**完整逐帧 CSV 和温度/频率/负载日志见 `rknn-cpp-Multithreading-main/benchmarks/rk3588_yolo_int8/results/optimized_full_20261109/`。

| 模型 | 单 context FPS 三轮；mean ± std (CV) | 三 context 总吞吐 FPS 三轮；mean ± std (CV) | 三路/单路 |
|---|---:|---:|---:|
| YOLOv5n | 52.03/50.94/51.08; **51.35±0.59 (1.16%)** | 146.10/144.30/145.50; **145.30±0.92 (0.63%)** | 2.83× |
| YOLOv5s | 34.06/33.66/33.28; **33.67±0.39 (1.17%)** | 98.83/98.49/98.98; **98.77±0.25 (0.26%)** | 2.93× |
| YOLOv5nu | 43.04/42.88/42.88; **42.93±0.10 (0.22%)** | 125.52/124.34/126.25; **125.37±0.96 (0.77%)** | 2.92× |
| YOLOv5su | 27.58/27.30/27.38; **27.42±0.14 (0.51%)** | 80.20/79.60/79.12; **79.64±0.54 (0.68%)** | 2.90× |
| YOLOv8n | 56.90 | 156.84 | 2.76× |
| YOLOv8s | 32.09 | 87.80 | 2.74× |
| YOLO11n | 47.27 | 130.45 | 2.76× |
| YOLO11s | 28.14 | 75.10 | 2.67× |
| YOLO26n score-sum | 32.81/32.55/32.25; **32.54±0.28 (0.86%)** | 94.33/93.68/90.40; **92.81±2.11 (2.27%)** | 2.85× |
| YOLO26s score-sum | 21.23/21.17/21.27; **21.22±0.05 (0.24%)** | 57.78/57.40/57.69; **57.62±0.20 (0.34%)** | 2.72× |

上表六个模型三轮吞吐 CV 均低于 2.3%；Y26n 三核 CV 最高、CPU 占用也偏高。该回放口径与上节 1000 张自然图逐帧同步统计的数据集/采集方式不同，**两表不可相互替换或混合计算**。三 context 也不是单帧 NPU 三核融合推理。

## 结果怎么用

- 若看 RK3588 单 context 实时响应，参考 1000 张图端到端延迟；如果关注流水线吞吐，参考 NV12 回放三 context 列。
- 若目标是 60 FPS，当前自然图单 context 端到端只有 v8n 的折算均值超过 60 FPS，但固定 NV12 回放单路为 56.90 FPS；受负载/温度和图像复杂度影响，不能据此保证相机实时 60 FPS。三 context 聚合吞吐普遍超过 60 FPS，但这代表多路异步帧吞吐，必须额外验证时序和延迟要求。
- YOLOv8/11 后处理开销最低；v5n/nu 接近 1 ms，v5s/su 约 1.7 ms；正式 YOLO26n 仍约 2.38 ms、s 约 0.64 ms。下一步应核查 Y26n score-sum 门控放行格点异常偏多的原因，并在当前正式模型上做逐图输出与 AP 回归。
- 若需要真正严格的一次性三后端精度 A/B，下一步应统一导出三种后端逐图 COCO JSON，并使用同一个 COCOeval 程序重算；当前 v5 原版精度保留 YOLOv5 validator 的 0.001 舍入精度，YOLO26 的端到端/NMS-free 与 raw-head/NMS 语义也需保留其架构差别。

## 复现与数据来源

- 本轮重测脚本：[validate_common_accuracy.py](../tools/validate_common_accuracy.py)
- PT/TensorRT 相同参数验证原始结果：[quality_metrics.csv](common-quality-2026-11-09/quality_metrics.csv)；YOLOv5 legacy 的逐模型日志位于该目录，其他模型的 validator 结果位于 `validation/<model_backend>/`。保留的原始预测 JSON 尚未统一映射 COCO category ID，不能直接用作 COCOeval 结果；报告中的 PT/TRT mAP 使用各自 validator 的指标。
- 精度验证参数：`conf=0.001, IoU=0.45, max_det=100, imgsz=640, batch=1`；TensorRT 为 FP16、PT 为 FP32。
- COCO 清单及内容审计、RKNN 单 context 分项：[统一测试集 RKNN 报告](rk3588/unified_same_dataset_rknn/REPORT.md)
- v5/v5u/正式 score-sum YOLO26 的优化后分项、同测试集精度复核、三轮单核/三核吞吐和原始监控数据：[优化后完整复测报告](../rknn-cpp-Multithreading-main/benchmarks/rk3588_yolo_int8/results/optimized_full_20261109/REPORT.md)
- RKNN 三 context 固定 NV12 回放与 PT/TRT 历史对照：`PT-TensorRT-RKNN-comparison.md`；TRT 原始速度文件位于 `tensorrt-5090-2026-11-05/summary.csv`、`raw_timings.csv`、`round_throughput.csv`、`environment.json`。
- 精度测试集为 COCO val2017 固定 1000 张（7476 个实例）；校准集是独立的 COCO train2017 500 张。数据集清单 ID/顺序 SHA-256：`be6a4f1d5e3c593917fbcaaab91053625b192f5a4aad2c1278bc953df50582e0`；图片内容序列 SHA-256：`41e46a49623e50071cc4a2644841d71dd7e982e93bf6db7edadfcdabdc96ce2d`。
- 当前 TensorRT 精度重测运行在桌面 RTX 5090；板端 RKNN 固件/runtime 和速度 profile 信息见 RKNN 原报告。不同设备/运行时的速度只按其自身边界解释。
