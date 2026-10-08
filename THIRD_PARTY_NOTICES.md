# 第三方来源与许可

- 检测器、基础 ByteTrack/BoT-SORT/其他 Tracker 和 ReID 接口来自 [Ultralytics](https://github.com/ultralytics/ultralytics)，固定安装版本 8.4.158；未将其基础算法署为本项目原创。来源许可见其 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)。
- CrowdHuman AP/MR 工具来自 [Megvii CrowdDetection](https://github.com/megvii-model/CrowdDetection/tree/master/evaluate/APMRToolkits)，原文件与 Apache-2.0 许可保留在 `tools/vendor/crowdhuman_eval/`。
- 多目标评价使用 [TrackEval](https://github.com/JonathonLuiten/TrackEval)，固定版本 1.3.0；ONNX Runtime、OpenCV、NumPy、PyTorch 等通过依赖安装，遵循各自许可。
- 自有身份恢复、整合、评估适配、GUI 及测试源码采用根目录 [AGPL-3.0](LICENSE)。本说明不改变第三方代码、模型和数据本身的权利。
- [MOT17](https://motchallenge.net/data/MOT17/) 和 [CrowdHuman](https://www.crowdhuman.org/) 原始图像、视频和 GT 不提交 Git。README 跟踪预览来自 MOT17-05 的本地运行，是功能展示候选；源码许可证不授予原图像权利。
- 微调权重基于官方 YOLO26s，100 轮训练记录可从本地主权重核验；权重不提交 Git，目前没有已发布下载地址。官方参考权重的来源链接和校验值见 `manifests/models.json`。
