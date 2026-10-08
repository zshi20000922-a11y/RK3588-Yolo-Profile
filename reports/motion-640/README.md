# 640×640 NV12 运动检测

板端测试达到 60 FPS：优化三帧差约 60.47 FPS，OpenCV MOG2 约 60.33 FPS。三帧差约 2.01 CPU 核、RSS 48.45 MB；MOG2 约 2.99 核、RSS 67.16 MB。测试由 4K 摄像头经 RGA 缩至 640×640，详细参数与限制见 [`REPORT.md`](REPORT.md)。
