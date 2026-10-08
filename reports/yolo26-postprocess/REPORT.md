# YOLO26 RK3588 后处理 A/B（精度数值不发布）

这是 score-sum raw-head RKNN INT8 one-to-many 路径的 CPU 后处理优化记录。输出协议、置信度阈值、坐标解码、NMS 和测试输入保持不变；该报告只对比运行时间与检测输出一致性，不发布任何 YOLO26 mAP/AP 或历史量化精度数值。

## 优化与结果

测试平台为 RK3588；COCO1000 固定 NV12 输入、单 context、conf=0.25。新旧两个 ARM64 程序在相同逐图输入上运行，检测 JSONL 逐条完全相同。

| 模型 | 后处理版本 | Mean ms | P50 ms | P95 ms | P99 ms | Max ms | 检测数量 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26n | 原标量 | 8.216 | 8.853 | 9.258 | 9.583 | 17.094 | 10,239 |
| YOLO26n | NEON + 量化早筛 | 3.064 | 3.281 | 3.644 | 4.058 | 4.450 | 10,239 |
| YOLO26s | 原标量 | 8.671 | 8.943 | 9.501 | 9.885 | 18.364 | 17,161 |
| YOLO26s | NEON + 量化早筛 | 3.320 | 3.345 | 3.823 | 4.177 | 4.634 | 17,161 |

最优实现已纳入 [`src/rkvs/detector_adapter.cc`](../../src/rkvs/detector_adapter.cc)。优化包括：INT8/UINT8 score-sum 门限直接在量化域筛选；AArch64 NEON 加速 NC1HWC2 的类别 argmax；对候选向量做小容量预留，避免每帧先为全部 8,400 个 grid cell 分配对象。NMS 会缓存候选框面积以减少重复计算。所有优化前后检测内容逐条相同。

这是 raw-head one-to-many 模型的后处理 A/B，不是官方 one-to-one/NMS-free 推理。模型导出精度争议不在本报告解决；历史错误精度已从此次 GitHub 发布材料中排除。

## One-to-one 导出尝试及未采用原因

尝试过两种真正基于 YOLO26 one-to-one head 的导出形态，但都没有作为正式部署结果：

1. **完整端到端 `[1,300,6]` ONNX → RKNN。** RKNN Toolkit 2.3.2 能生成 engine，但板端运行在 `GatherElements` 出现越界 index（s 的复现日志中单帧可报约 1,200 次），输出无法通过正确性验收。此外，单一 INT8 输出把坐标、score 和类别 id 放在同一量化张量中，动态范围差别很大，分数精度不足。因此不报告它的有效 FPS 或精度。
2. **保留 one-to-one head、移除末端 TopK/Gather，拆成 boxes `[1,8400,4]` 与 scores `[1,8400,80]`。** ONNX 导出时 PyTorch TopK 等价性检查通过，RKNN engine 也成功生成；但板端 runner 当时没有这套双输出的完整解码/全局 TopK adapter，保存的 1000 帧记录均为 0 个检测，后处理均值约 54 ms。因此不能当作可用模型，也没有可发布的 AP/FPS。即使补齐 adapter，它仍需 CPU 阈值筛选和全局 TopK，并不是零后处理。

另有一次把 Rockchip **九输出 one-to-many** raw-head 误当成 one-to-one、直接用 TopK 代替 NMS 的试验。根本问题是该导出路径实际关闭了 `end2end`，图仍然是 one-to-many；跳过 NMS 会保留重复框并损伤检测质量。这个试验不是上述第二种“one-to-one head”导出，相关结果不纳入正式对比。当前已验收方案仍使用九输出 raw-head + 分类 NMS，后处理 NEON/早筛实现已经随仓库源码上传。

原始逐帧 timing CSV：[`raw/`](raw/)。
