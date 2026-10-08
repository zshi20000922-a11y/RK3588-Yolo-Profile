# RK3588 连板推理：源码、运行与计时口径

本文描述仓库内可复现的 RK3588 推理工具，以及它们与已发布性能表之间的关系。README 中的 FPS 是板端运行数据；PC 上运行 PyTorch 的结果不能替代这里的 RKNN 数据。

## 已发布文件

| 文件 | 用途 |
|---|---|
| [`tools/rk3588_yolo_benchmark.cc`](../tools/rk3588_yolo_benchmark.cc) | 板端连续帧性能测试，写出逐帧 CSV 和总吞吐；支持 DMA-BUF 回放/CPU-copy 基线、单 context/三 context。 |
| [`tools/rk3588_yolo_quality.cc`](../tools/rk3588_yolo_quality.cc) | 按固定 NV12 清单逐图推理，生成 COCO JSONL 预测与各阶段 timing。 |
| [`tools/rk3588_int8_pipeline.py`](../tools/rk3588_int8_pipeline.py) | ADB 预检、数据清单、量化转换、上板运行、采回原始数据和汇总。脚本当前转换矩阵是 YOLOv8/11/26 n/s；不要据此声称它自动完成 YOLOv5/v5u 转换。 |
| [`tools/convert_yolo26_rk3588.py`](../tools/convert_yolo26_rk3588.py) | 单独转换 YOLO26n/s：RKOptimized raw-head ONNX → RK3588 asymmetric INT8 RKNN，校验九输出 schema 并保存哈希 manifest。 |
| [`tools/prepare_coco_nv12.py`](../tools/prepare_coco_nv12.py) | 将固定图片清单转成板端 NV12 输入并生成输入列表。 |
| [`tools/evaluate_coco_jsonl.py`](../tools/evaluate_coco_jsonl.py)、[`tools/evaluate_coco_pair_jsonl.py`](../tools/evaluate_coco_pair_jsonl.py) | 从逐图 JSONL 计算 COCO AP，或在完全相同测试图片上成对比较 FP32 与 INT8。 |
| [`src/rkvs/rknn_detector.cc`](../src/rkvs/rknn_detector.cc)、[`src/rkvs/detector_adapter.cc`](../src/rkvs/detector_adapter.cc) | RKNN context、NPU 核绑定、RGA 预处理、输出同步和 YOLO 解码/NMS。 |

仓库有 C++ YOLOv5 anchor-based、YOLOv8、YOLO11、YOLO26 adapter。YOLOv5u 的历史 FPS 被保留在性能摘要中，但当前通用 adapter 没有独立的 YOLOv5u 输出协议；其精度基线也因预测重复而判为无效，因此不能声称已由这里的 runner 独立复现。

本仓库公开并评测的 YOLO26 是九输出 raw-head one-to-many 图，CPU 执行 score-sum 过滤、解码和分类 NMS；不是官方 one-to-one `[1,300,6]` NMS-free 模型。专用转换脚本只接受该已验证的九输出协议，遇到不同输出 schema 会失败退出，避免误把不同导出路径的精度/性能混在一起。通用 `rk3588_int8_pipeline.py convert` 仍适用于 v8/11/26 raw-head 模型，但生成的 manifest 也将明确记为 raw-head + NMS。

### YOLO26n/s 导出与 INT8 转换

转换需要模型权重、固定 500 张 COCO train 校准图片清单、RKNN Toolkit 2 和项目测试时使用的 RKOptimized Ultralytics checkout。转换器并不下载权重或图片。校准清单每行是一张图片路径；脚本要求恰好 500 个唯一且存在的文件。`--ultralytics-repo` 指向含对应 RKNN exporter 的源码 checkout，普通上游 Ultralytics 不保证支持 `format='rknn'`。

```bash
python3 tools/convert_yolo26_rk3588.py \
  --weights /path/to/yolo26n.pt \
  --ultralytics-repo /path/to/rkoptimized-ultralytics \
  --calibration /path/to/calibration_500.txt \
  --output artifacts/yolo26n
```

输出包含固定 640×640、batch 1 的 `_rkopt.onnx`、RK3588 INT8 `.rknn`、转换日志及带权重/ONNX/RKNN/校准列表 SHA-256 和 output schema 的 `conversion_manifest.json`。导出参数为 `dynamic=False, opset=14`。性能/精度表只对应九输出 raw-head 路径；不要用该转换结果声称实现了官方 NMS-free one-to-one 语义。

## 构建基准程序

在 RK3588 原生 AArch64 环境或已配置好的 AArch64 交叉编译环境中构建。需要自行提供与目标板版本匹配、且获授权的 RKNN API/runtime；仓库不包含专有 RKNN 头文件或运行库。

```bash
cmake --preset rkvs-minimal \
  -DRKNN_API_PATH=/opt/rockchip/librknn_api \
  -DRKNN_RT_LIB=/opt/rockchip/lib/librknnrt.so \
  -DRGA_PATH=/opt/rockchip/rga \
  -DRGA_LIB=/opt/rockchip/rga/lib/Linux/aarch64/librga.so
cmake --build build-rkvs-minimal --target rk3588_yolo_benchmark rk3588_yolo_quality -j4
```

如果是交叉编译，需要额外使用本机工具链/sysroot 参数，并确保生成二进制、RGA 和 RKNN runtime 的 ABI 都是 AArch64 且与板端兼容。构建成功不代表模型输出 schema 已适配；程序首次初始化会验证实际 tensor schema。

## 板端 DMA-BUF 回放测试

先确认 ADB 状态为 `device`，并将 benchmark、RKNN 文件、对应 manifest、COCO 标签和一帧完整 NV12 放到板端同一目录。以下例子在板端测试 640×640 的 YOLOv8n；输入帧应与 `--width/--height` 一致。

```bash
adb -s 10.153.18.29:5555 shell 'mkdir -p /data/local/tmp/rk_yolo_bench'
adb -s 10.153.18.29:5555 push build-rkvs-minimal/rk3588_yolo_benchmark /data/local/tmp/rk_yolo_bench/
adb -s 10.153.18.29:5555 push yolov8n.rknn yolov8n.yaml coco80.txt frame_640.nv12 /data/local/tmp/rk_yolo_bench/
adb -s 10.153.18.29:5555 shell 'cd /data/local/tmp/rk_yolo_bench && chmod +x rk3588_yolo_benchmark && ./rk3588_yolo_benchmark --model yolov8n.rknn --manifest yolov8n.yaml --labels coco80.txt --family yolov8 --nv12 frame_640.nv12 --width 640 --height 640 --mode latency --input-mode dma-replay --warmup 100 --iterations 3000 --seconds 60 --output latency.csv'
```

把 `--mode latency` 改为 `--mode throughput` 可测三个独立 context 的并行吞吐。程序按模式强制选择 1 或 3 个 context，并分别绑定 Core 0 或 Core 0/1/2；需结合板端 RKNN/NPU 监控日志确认实际负载。将 `--input-mode dma-replay` 改为 `cpu-copy` 可测每帧额外 CPU memcpy 的对照。建议模型顺序交错、三轮独立运行并同时保存板端温度、NPU 频率/利用率和 `raw_timings.csv`。

### 哪些时间计入 FPS

基准通过一次性装入的 NV12 帧反复构造 DMA-BUF frame reference。逐帧计时记录：

- `copy_ms`：仅 `cpu-copy` 模式每帧的显式复制；DMA 回放模式为 0。
- `queue_ms`：进入检测阶段前的队列时间。
- `rga_ms`：NV12→模型输入格式、resize/letterbox。
- `inference_ms`：RKNN 模型执行时间；这是 **NPU inference-only latency**，不是端到端延迟。
- `output_sync_ms`：输出同步/必要的输出访问准备。
- `postprocess_ms`：输出解码、分数筛选、NMS/最终检测整理。
- `e2e_ms`：DMA 回放帧到检测结果；含 RGA、NPU、输出同步和后处理。此程序不测真实相机 dequeue、JPEG 解码、网络传输、画框或落盘。

单 context 端到端 FPS 近似为串行结果速率；三 context FPS 是三个 worker 对不同帧的总吞吐。不要用 `1000 / inference_ms` 冒充 E2E FPS，也不要把 DMA replay FPS 写成摄像头采集 FPS。

## 连板精度评测

精度评估需固定且不重叠的校准/测试集：本次报告使用 500 张 COCO train 校准图、1000 张 COCO val 测试图；同一测试图片在每个模型/格式下只推理一次，FP32 与 INT8 在相同 image ID 上配对评估。

`rk3588_yolo_quality` 读取格式为 `nv12_path width height image_id` 的逐图输入清单，输出 `--detections` JSONL 与 `--timings` CSV。使用 `evaluate_coco_pair_jsonl.py` 计算 FP32 与 RKNN 的 AP50-95/AP50 变化。不同测试集、类别映射、输入尺寸或预处理不一致时，AP 差值不能作为量化损失直接比较。

模型权重、校准/测试图片、`.rknn`、逐图 JSONL 和大体积原始帧不提交到公开仓库；请在本机实验目录保存 manifest、SHA-256、Runtime/Toolkit 版本与原始计时结果。

## 结果范围与限制

- 可用的 YOLOv8/11 完整分阶段数据和三轮轮间稳定性见 [`reports/v8-v11/REPORT.md`](../reports/v8-v11/REPORT.md)。
- YOLOv5/v5u 历史吞吐与 v5 FP32/INT8 精度见 [`reports/YOLO-performance-summary.md`](../reports/YOLO-performance-summary.md)；v5u 的精度值不发布，v5u 对应 runner 适配仍缺失。
- YOLO26 的有效 COCO 子集精度与 RK3588 连板 DMA 回放吞吐见 [`reports/yolo26/REPORT.md`](../reports/yolo26/REPORT.md)。它明确区分 NPU inference-only、单 context E2E FPS 和 3-context 总吞吐；该固定回放只有一次 1000 帧工程测量，不等同三轮长测，也不是 V4L2 相机 FPS。
- 表中 `640×640` 是模型输入尺寸。性能基准的输入为已预加载内存图像；运动算法测试包含真实 V4L2 摄像头，但不代表 YOLO 检测链路。
