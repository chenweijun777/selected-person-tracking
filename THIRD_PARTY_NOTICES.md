# 第三方来源与许可

- 检测、基础跟踪器与 ReID 接口：[Ultralytics](https://github.com/ultralytics/ultralytics)，版本 8.4.158，[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)。
- CrowdHuman AP／MR 评价工具：[CrowdDetection](https://github.com/megvii-model/CrowdDetection/tree/master/evaluate/APMRToolkits)，原文件与 [Apache-2.0 许可证](tools/vendor/crowdhuman_eval/LICENSE)位于 `tools/vendor/crowdhuman_eval/`。
- 多目标跟踪评价：[TrackEval](https://github.com/JonathonLuiten/TrackEval)，版本 1.3.0。PyTorch、ONNX Runtime、OpenCV 等依赖遵循各自许可。
- 自有源码：[AGPL-3.0](LICENSE)。
- 数据：[MOT17](https://motchallenge.net/data/MOT17/)、[CrowdHuman](https://www.crowdhuman.org/)，按官方条款获取。本仓库提供目标与事件清单，原始图像、视频和 GT 由使用者准备。
- README 预览来自 MOT17-05 的跟踪运行。
- 模型来源与校验值：[models.json](manifests/models.json)。微调权重由训练流程生成。
- GPU 推理运行时：[ONNX Runtime](https://github.com/microsoft/onnxruntime)，部署实测使用 GPU 版 1.30.0；[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)，实测版本 10.14.1.48.post1，按 NVIDIA 提供的软件条款安装使用。
