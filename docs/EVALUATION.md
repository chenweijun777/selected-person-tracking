# 评价与复现

## 数据划分

使用 MOT17 官方训练集中的 SDP 序列，按指定目标跟踪任务划分：

| 集合 | 序列 | 指定目标数 | 目标评价帧数 |
| --- | --- | ---: | ---: |
| 开发集 | MOT17-02/04/05 | 116 | 28,109 |
| 验证集 | MOT17-09/10 | 47 | 9,640 |
| 测试集 | MOT17-11 | 31 | 5,153 |

各方法使用相同目标范围和选人起点，初始化失败也计入评价。

| 清单 | 开发／验证集 | 测试集 |
| --- | --- | --- |
| GT ID 与选人起点 | [目标清单](../protocols/current/devval/selection_manifest.json) | [目标清单](../protocols/current/test/selection_manifest.json) |
| 恢复事件 | [事件表](../protocols/current/devval/gt_events.csv) | [事件表](../protocols/current/test/gt_events.csv) |
| 计分协议 | [开发](../protocols/current/development.json)、[验证](../protocols/current/validation.json) | [测试](../protocols/current/test.json) |
| 事件排除记录 | [开发／验证](../manifests/recovery_event_exclusions_20261005.json) | [测试](../manifests/test_recovery_event_exclusions_20261007.json) |

计分事件见协议的 `common_event_ids`；无合格选人起点的目标记录在 `no_qualifying_start`。

## 评价指标

| 层级 | 指标 | 评价内容 |
| --- | --- | --- |
| 检测 | Precision、Recall、AP50、AP75、mAP50–95 | 检出、误检与定位 |
| 多目标跟踪 | HOTA、IDF1、IDSW、Frag、MOTA | 轨迹与身份关联 |
| 指定目标 | 正确覆盖率、误跟帧数 | 是否持续跟对选定的人 |
| 遮挡恢复 | 稳定恢复事件数、恢复延迟 | 是否找回目标及所需时间 |

检测使用 `mark=1、class=1` 的行人标注，其余区域按 ignore 处理；推理阈值 0.001，Precision／Recall 工作点 0.10，匹配 IoU 0.5，AP 采用 101 点插值。多目标指标使用 TrackEval 的 MOTChallenge 实现。

指定目标统计选人后可见度 ≥0.3 的人帧，IoU ≥0.5 判定身份。C 为跟对，W 为跟错人，R 为无有效输出，U 为未匹配行人，A 为多行人重叠歧义。

**正确覆盖率 = C / (C + W + R + U + A) × 100%。** 缺标注帧记为 unknown。

目标重现后，在可见度 >0 的帧中连续跟对 `ceil(0.2 × fps)` 帧计为稳定恢复。延迟从重现帧算到稳定正确区间首帧。共同事件清单用于同口径诊断比较，开发／验证／测试为 32／14／18 个可评分事件。

`formal/` 统计已复核且事件前跟对的事件，分母随方法变化。最终方案测试为共同清单 17/18、条件统计 17/17；验证集条件统计无可评分事件。

## 模型准备

安装见 [README](../README.md#安装)。命令在仓库根目录执行，输出目录须为新目录。

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
python tools/prepare_models.py --download --profile pretrained
python tools/run_video_joint.py --source data/demo.mp4 --output outputs/pretrained_demo --device 0 --display --model models/yolo26s.pt --tracker configs/pretrained/tracker.yaml --target-config configs/pretrained/target.json
```

默认界面使用 `models/best.pt` 和 `models/yolo26n-reid.onnx`。微调权重由下面的训练流程生成，复制为 `models/best.pt` 后启动界面；命令行也可用 `--model` 指定模型。配套原权重的校验值见 [models.json](../manifests/models.json)。

## 检测微调

CrowdHuman 放入 `data/CrowdHuman/`，包含 `annotation_train.odgt`、`annotation_val.odgt`、`CrowdHuman_train*/Images` 和 `CrowdHuman_val/Images`。

```powershell
python tools/run_crowdhuman_yolo26s.py prepare --source data/CrowdHuman --output outputs/crowdhuman_run
python tools/run_crowdhuman_yolo26s.py initialize --output outputs/crowdhuman_run
python tools/run_crowdhuman_yolo26s.py smoke --output outputs/crowdhuman_run --device 0
python tools/run_crowdhuman_yolo26s.py train --output outputs/crowdhuman_run --device 0
```

训练 100 轮，模型输出至 `outputs/crowdhuman_run/train/weights/`。训练设计见 [EXPERIMENTS.md](EXPERIMENTS.md#crowdhuman-微调与下游传递)。

## 结果复现

MOT 数据目录为 `data/MOT17/train/<序列>/`，包含 `img1/`、`gt/gt.txt` 和 `seqinfo.ini`。以下以测试集为例，使用微调模型生成输入、绑定固定选人起点并评估：

```powershell
python tools/build_sequence_inputs.py --source data/MOT17/train/MOT17-11-SDP --output reproduction_inputs/MOT17-11-SDP --device 0
python tools/prepare_benchmark_protocol.py --split test --output outputs/test_protocol.json
python tools/replay_benchmark.py --split test --protocol outputs/test_protocol.json --output outputs/reproduce_test
```

开发、验证集先逐序列生成输入，再将 `--split` 换为 `development`、`validation`，并使用各自的协议与输出文件名。

已有配套输入时，按固定清单直接回放：

```powershell
python tools/prepare_reproduction.py --source-dir resources/benchmark_inputs
python tools/replay_benchmark.py --split test --output outputs/replay_test
```

输出包括逐帧评价、恢复事件明细和 `summary.csv`。公开结果对应固定微调权重与输入；重新训练或更换输入可能产生不同结果。参考[策略与消融结果](../results/policy_reproduction.csv)。

```powershell
python tools/verify_results.py
python tools/check_release.py
```
