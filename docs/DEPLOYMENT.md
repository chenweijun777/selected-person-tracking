# 推理部署

将微调 YOLO26s O2M 检测器部署到 ONNX Runtime CUDA 和 TensorRT，并接入指定行人跟踪应用。支持 PyTorch、ORT FP32、TensorRT FP32 和 TensorRT FP16 四种后端。

## 部署流程

`best.pt → 动态 ONNX → TensorRT 引擎 → 检测 → BoT-SORT → ShortLink → A2`

- 导出 O2M 原始预测，在模型外执行 NMS 和坐标还原。
- 动态高宽保留矩形 letterbox。例如 1920×1080 图像对应 544×960 网络输入。
- ORT 用 GPU IO binding 绑定输入和动态输出，通过 DLPack 交给 PyTorch；预处理、推理和 NMS 使用同一 CUDA stream。
- TensorRT 使用 FP32 输入、输出；FP16 引擎在内部选择 FP16／FP32 计算。
- 两路 ReID 在 CPU 上独立运行，输入均为 448 RGB。

实现：[部署适配器](../src/a1_tracker/detectors/deployed.py)、[导出与构建](../tools/deploy_o2m.py)、[运行入口](../tools/run_deployed_joint.py)、[测速入口](../tools/benchmark_deployment.py)。

## 安装

实测环境为 Windows 11、Python 3.12、RTX 3090。实测版本见[部署环境](../provenance/environment_deployment.txt)。在仓库根目录新建部署环境：

```powershell
python -m venv .venv-deploy
.\.venv-deploy\Scripts\python.exe -m pip install "torch==2.14.0+cu130" "torchvision==0.29.0+cu130" --index-url https://download.pytorch.org/whl/cu130
.\.venv-deploy\Scripts\python.exe -m pip install -e ".[deploy,evaluation,dev]"
.\.venv-deploy\Scripts\python.exe -m pip install --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple "tensorrt-cu13==10.14.1.48.post1"
```

`deploy` 安装 GPU 版 ONNX Runtime；普通应用的 `runtime` 安装 CPU 版，分别用于各自环境。模型准备见[模型说明](EVALUATION.md#模型准备)。

## 导出与构建

```powershell
.\.venv-deploy\Scripts\python.exe tools/deploy_o2m.py --stage all --model models/best.pt --output outputs/deployment
```

输出包括 `best_o2m.onnx`、`best_o2m_fp32.engine`、`best_o2m_fp16.engine`，以及导出和构建记录。记录包含模型哈希、GPU、版本、输入尺寸范围、IO 类型和构建耗时。

分阶段运行：

```powershell
.\.venv-deploy\Scripts\python.exe tools/deploy_o2m.py --stage export --model models/best.pt --output outputs/deployment
.\.venv-deploy\Scripts\python.exe tools/deploy_o2m.py --stage build --onnx outputs/deployment/best_o2m.onnx --output outputs/deployment --precision both
```

引擎按目标机器的 GPU 和 TensorRT 构建。默认尺寸范围为 min `1×3×32×32`、opt `1×3×544×960`、max `1×3×960×960`，batch 为 1。重复导出或构建时使用新的输出目录。

## 运行跟踪应用

```powershell
.\.venv-deploy\Scripts\python.exe tools/run_deployed_joint.py --backend trt-fp16 --source data/demo.mp4 --output outputs/deployed_demo --display
```

`--backend` 可选 `pytorch`、`onnx`、`trt-fp32`、`trt-fp16`；`--model` 可指定模型文件。点击人物后按空格播放，空格暂停／继续，`r` 清除后重选，`q` 退出。输出包含标注视频、轨迹、目标状态和运行配置。

部署对照使用**微调 best.pt＋2026-10-06 固定跟踪参数**，对应仓库现有的 `configs/pretrained/tracker.yaml` 和 `configs/pretrained/target.json`。这里的目录名表示已有参数组；检测权重仍为微调模型。运行入口默认使用这组配置，也可通过 `--tracker`、`--target-config` 指定配置。

桌面 GUI 使用微调参数与 PyTorch 后端；部署后端由上述命令行入口选择。

## 实测结果

2026-10-09，RTX 3090，MOT17-09/10 验证集，共 1179 帧。四组使用相同权重、预处理、外部 NMS 和跟踪参数。

| 后端 | 独立检测 FPS | 整链计算 FPS | HOTA | IDF1 | 恢复事件 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch | 87.08 | 6.13 | 53.677 | 67.755 | 14/14 |
| ORT FP32 | 91.64 | 6.18 | 53.678 | 68.075 | 14/14 |
| TensorRT FP32 | 101.99 | 6.26 | 53.679 | 68.075 | 14/14 |
| TensorRT FP16 | 116.69 | 6.75 | 53.496 | 67.582 | 14/14 |

独立检测测试覆盖 1179 帧，预热 20 次并同步 GPU 计时，包含预处理、前向、NMS、坐标恢复和返回 CPU 检测框。整链测速采用两段序列、正逆序各一轮，每个后端共 320 个样本，包含检测、跟踪、两路 ReID 和 A2，排除解码、加载、绘制与视频编码。首次独立 TRT 检测采集期间另有 CPU 精度作业；整链测速单独运行。

FP16 独立检测速度提高约 34%，整链计算速度提高约 10%。项目恢复评价包含 47 个目标、9640 个目标评价人帧和 14 个共同恢复事件。完整数据见[检测表](../results/deployment_detector.csv)、[整链表](../results/deployment_pipeline.csv)、[质量表](../results/deployment_quality.csv)。

## 重新测速

检测测速：

```powershell
.\.venv-deploy\Scripts\python.exe tools/benchmark_deployment.py --backend all --mode detector --source data/MOT17/train/MOT17-09-SDP --max-frames 525 --skip-frames 0 --output outputs/benchmark_detector_09
```

指定目标整链测速，下面的点击坐标对应 MOT17-09 第 1 帧：

```powershell
.\.venv-deploy\Scripts\python.exe tools/benchmark_deployment.py --backend all --mode pipeline --source data/MOT17/train/MOT17-09-SDP --select-frame 1 --select-point 1788.8345 558.621 --max-frames 90 --skip-frames 10 --output outputs/benchmark_pipeline_09
```

默认预热 20 次，重复两轮，第二轮反转后端顺序。每轮新建跟踪和目标状态。更换视频时同步指定选人帧与点击坐标；选择失败会终止测速。输出逐帧耗时及平均值、P50、P95、FPS、有效目标样本数。

结果表保存原部署实验；这些命令生成当前机器上的新测量。新测速入口按点击坐标选人，原实验按固定目标框初始化，初始化记录随各次实验保存。

## 数值与运行验证

四后端抽样预处理张量逐位相同，原始输出均为有限值。检查范围涵盖原始框和分数、NMS 后检测、全局 MOT 指标以及指定目标恢复。

ORT 算子分析记录 1029 次 CUDA 节点执行；CPU 辅助节点中没有卷积或矩阵乘法。TensorRT 引擎 IO 均为 FP32。原 PyTorch 基准开启 cuDNN TF32，ORT／TRT FP32 关闭 TF32；数值对照同时记录了这一设置。

原应用验证中，ORT 运行 30 帧、TRT FP16 运行 60 帧，均成功选人并保持 TRACKING。数值摘要、设备、配置和结果来源见[部署验证记录](../provenance/deployment_validation.json)。
