# RK3588 YOLO Profile

RK3588 上的 YOLO/RKNN 性能评测、DMA-BUF/RGA 零拷贝执行、后处理优化和摄像头运动算法验证代码与结果摘要。

本仓库发布可复现的源码、脚本和经过筛选的板端报告，不包含模型权重、COCO 图片、NV12 帧、逐图检测结果或本机 build 目录。复测需要用户自行准备 RK3588、匹配版本的 RKNN Runtime/Toolkit、RGA/MPP/OpenCV 运行库及 COCO 数据。

## 已整理的内容

- `src/rkvs/`：C++ DMA-BUF 摄像头/帧流、RGA 预处理、RKNN 多 context 检测服务、后处理与跟踪组件。
- `tools/`：RK3588 性能/质量测试、COCO 评估、运动算法资源统计与汇总脚本。
- `reports/`：YOLOv5/v5u/v8/11/26 性能摘要、MOG2/帧差在 640、2K、4K 的板端报告，以及 YOLO26 后处理 A/B 结果。
- `config/`：硬件链路示例配置；请按本机摄像头节点、RKNN 模型路径和网络环境修改。

## 测试口径摘要

- YOLO 基准：RK3588、RKNN INT8、640×640、batch 1。单核指单 context 绑定 NPU Core 0；三核吞吐为三个独立 context 分别绑定 Core 0/1/2，并行处理不同帧，不是单帧跨核拆分。
- DMA 回放 FPS 包括 NV12→RGB/resize/letterbox、NPU、输出同步和 CPU 后处理；不是裸 NPU FPS，也不代表相机采集上限。
- YOLOv8/YOLO11 的 COCO1000 FP32→INT8 AP50-95 变化约为 -0.0246 至 -0.0292。YOLOv5u 的 FP32 基线检测与 YOLOv5 检测完全重复，故不发布其精度差值。
- YOLO26 后处理 A/B 只发布相同输入、预测逐条完全一致的耗时对比，不在这里发布历史 YOLO26 精度数值。此前精度异常的模型导出/量化结果不作为有效数据。
- 摄像头运动测试表明：优化三帧差可在 640×640 与原生 2K NV12 接近/达到 60 FPS；4K 全分辨率帧差和 MOG2 均达不到 60 FPS。

## 关键结果

详见 [`reports/YOLO-performance-summary.md`](reports/YOLO-performance-summary.md)、[`reports/v8-v11/REPORT.md`](reports/v8-v11/REPORT.md)、[`reports/yolo26-postprocess/REPORT.md`](reports/yolo26-postprocess/REPORT.md) 以及三个 `reports/motion-*` 报告。

## 构建与运行

项目通过 CMake 构建。目标板需要匹配的 RKNN Runtime、RGA、DMA heap 与相机驱动；交叉编译时通过工具链文件配置 AArch64 编译器。Rockchip SDK 中带有专有标识的 RKNN API headers/runtime libraries 未随仓库发布，请通过 `RKNN_API_PATH` 与 `RKNN_RT_LIB` 指向你本机获授权的 SDK 文件；RGA/MPP 头文件可按 Apache-2.0 条款使用，运行库应从匹配版本 SDK 提供。CMake 参数也支持用 `RGA_PATH`、`RGA_LIB`、`MPP_PATH`、`MPP_LIB` 指定本机依赖。

COCO 精度/校准实验须使用独立的校准集与测试集，清单与权重哈希应保存在本机实验目录；不要把数据集、权重或逐图结果提交到 Git。

更多 pipeline 参数见 [`tools/rk3588_int8_pipeline.py`](tools/rk3588_int8_pipeline.py) 和 [`reports/v8-v11/REPORT.md`](reports/v8-v11/REPORT.md)。
