# 评价口径与复现工作流

## 数据划分和证据边界

MOT17-02/04/05 为开发集，09/10 为验证集，11 为自定义测试序列；均使用 SDP 对应原视频/GT，不把同视频 DPM/FRCNN 当成独立样本。测试序列来自官方训练集，不宣称 MOTChallenge 官方隐藏测试成绩。初期/当前检测开发样本为 2,487 帧，其中 05 使用 837 帧；指定目标会话范围由冻结协议单独规定。

当前指定目标开发集 116 个会话、28,109 主可见人帧，验证 47 个会话、9,640 帧，测试 31 个会话、5,153 帧。冻结的文件可能保留 legacy `historical` 标签，入口转换为 `validation`，原字节及哈希不改。

## 指标为什么分层

| 层级 | 指标 | 回答的问题 |
| --- | --- | --- |
| 检测 | Precision、Recall、AP50、AP75、mAP50–95 | 人有没有被检测，误检与框定位如何 |
| 全行人 MOT | TrackEval HOTA、IDF1、IDSW、Frag、MOTA | 多目标轨迹和身份关联的整体质量 |
| 指定目标帧 | C、W、R、U、A 与正确覆盖率 | 用户选定的那个人有没有持续跟对 |
| 恢复事件 | 稳定恢复数量、误跟事件、恢复延迟 | 遮挡或低可见性后能否重新稳定跟对 |
| 工程 | 推理阶段耗时、整段耗时、缓存一致性 | PC 应用是否可用、缓存是否改变行为 |

检测为 marked=1、class=1 的行人 GT，其他区域作 ignore；IoU≥0.5 匹配，低分推理 0.001、统计工作点 0.10，AP 为本项目 101 点插值口径，不能冒充 COCO 官方 AP。原始 GT 不裁剪。TrackEval 采用官方 MOTChallenge 预处理和指标实现，与指定目标评分独立。

指定目标采用冻结清晰起点及结束帧。主分母为选人后 GT 可见度≥0.3 的目标人帧，未成功初始化仍保留。预测框对当帧有效行人 GT 做 best-IoU 判定，多人 IoU≥0.5 的重叠记为 A，不强行判定身份：

- C：正确指定目标；W：跟到其他人；R：拒绝/无有效输出；U：有效输出但不匹配有效行人；A：多 GT 重叠歧义。
- 正确覆盖率 = C / (C+W+R+U+A) ×100%。W 同时展示帧数和比例，不能仅凭更高覆盖率认定策略更好。
- GT 缺失为 unknown，不能视为目标离场。没有确认离场分母时，离场误输出率为 null。

## 事件恢复的两种分母

重现后可见度>0 的正确帧持续至少 `ceil(0.2×fps)` 帧，认定稳定恢复；时间线缺标注会截断观察窗。延迟从重现起点到稳定正确区间起点计算，只对恢复成功事件统计。

共同队列固定事件清单，用于同口径诊断，`strict=False` 明确标记 provisional。开发/验证/测试为 32/14/18 个可评分事件；测试原保留 20 个事件，2 个位于初始化之前。正式条件 `strict=True` 还要求人工已复核及事件前跟对，各方法合格分母可能不同。

例如测试完整方案共同队列 17/18，正式条件 17/17；基础 T2+A0 为 10/18 与 10/11。不能用正式条件的百分比替代共同队列比较。验证事件的冻结 review 标签使当前 strict 口径分母为 0，正式恢复率 null；共同队列仍可做诊断。

## 从安装到演示

先按 [README 的 CUDA 安装步骤](../README.md#运行与复现)准备 Python 3.12、NVIDIA 驱动、CUDA 版 PyTorch 和项目依赖。所有命令直接使用当前 Python，在仓库根目录执行。GUI 需要 Tkinter，Linux 可额外安装 python3-tk；模型路径/哈希见 `manifests/models.json`。

```powershell
python -m pip install -e ".[runtime,evaluation,dev]"
python tools/prepare_models.py --source-dir "D:\tracking-models"
python tools/verify_models.py
python tools/run_tracking_gui.py
```

原始预训练参考（不等于正式微调结果）：

```powershell
python tools/prepare_models.py --download --profile pretrained
python tools/run_video_joint.py --source "D:\videos\demo.mp4" --output outputs/pretrained_demo --device 0 --display --model models/yolo26s.pt --tracker configs/pretrained/tracker.yaml --target-config configs/pretrained/target.json
```

## 结果复现：固定输入回放

当前本地整理版已经准备好忽略于 Git 的 `reproduction_inputs/` 和 GT。冻结协议在 `protocols/frozen/`；路径映射把旧路径映射到仓库内的有效输入副本。评估代码已经收进可安装的 `a1_tracker.evaluation`，不再动态导入草稿工具。

```powershell
python tools/replay_benchmark.py --split development --output outputs/reproduce_development
python tools/replay_benchmark.py --split validation --output outputs/reproduce_validation
python tools/replay_benchmark.py --split test --output outputs/reproduce_test
```

默认同时重跑 A0/A1/A2、NoBank、NoEMA，输出预测、逐帧评价、共同队列与正式事件结果、汇总及代码/输入哈希。输出目录必须是新目录。缓存只含当前观察轨迹、图像派生外观、因果 GMC 和同帧事件；每个策略、目标会话使用独立状态，不读取 GT 来做恢复。

仓库公开文件仅含协议、结果及清单，图像、GT、完整轨迹和特征缓存不提交。已有本地实验资料可按 `tools/prepare_reproduction.py --source-dir <实验资料根目录>` 校验迁移；该命令明确用于已有资料，不宣称从 Git clone 自动获得未发布数据缓存。

无旧缓存时，可以从本地 MOT 图像与权重重新构建因果输入：

```powershell
python tools/build_sequence_inputs.py --source "D:\datasets\MOT17\train\MOT17-11-SDP" --output reproduction_inputs/MOT17-11-SDP --device 0
```

对其他序列逐条执行。使用对应官方 GT/seqinfo，以及本仓库冻结选人和事件协议；复制 GT 到 `data/MOT17/MOT17/train/<序列>/gt/gt.txt`，seqinfo 放在同序列目录。更换设备/依赖可能引入检测数值差异，先核对本地锁定模型和输入哈希、初始化绑定，再判断能否复现相同数字。冻结的祖先记录只保留溯源，不加载旧 GT/旧事件实验结果。

## 训练工作流

外部 CrowdHuman 数据目录应包含 annotation_train.odgt、annotation_val.odgt 和 CrowdHuman_train*/Images、CrowdHuman_val/Images。

```powershell
python tools/run_crowdhuman_yolo26s.py prepare --source "D:\datasets\CrowdHuman" --output outputs/crowdhuman_run
python tools/run_crowdhuman_yolo26s.py initialize --output outputs/crowdhuman_run
python tools/run_crowdhuman_yolo26s.py smoke --output outputs/crowdhuman_run --device 0
python tools/run_crowdhuman_yolo26s.py train --output outputs/crowdhuman_run --device 0
```

快速 CPU 安装检查可在独立输出目录 prepare 时加 `--smoke-only`，随后 `smoke --device cpu --batch 2`。仅准备 64/32 张图、训练 1 轮；该产物不替代正式模型，也不能拿它复现最终成绩。完整训练为 15,000/4,370 张图、100 轮。训练脚本保留分类初始化、ignore 监督和 checkpoint 选择规则。

## 整理版已经验证的范围

- 全新 Windows/Python 3.12 虚拟环境安装与依赖检查，以及正式算法、缓存和研究隔离测试。
- 独立 A2 与旧实现：194 个当前会话、49,126 会话帧，输出和完整策略状态一致，随机会话 token 除外，见[等价记录](../provenance/a2_equivalence.json)。
- 开发/验证/测试 A0/A1/A2、模板库和 EMA 回放重新评分，见[复现表](../results/policy_reproduction.csv)。
- 真实模型 GUI 路径：36 帧、14fps MP4，脚本化选择/清除/重选，持久缓存重开后输出一致；同进程新对象检查，不冒充物理鼠标或跨进程性能测量，见[GUI 验证](../provenance/gui_validation.json)。
- CrowdHuman 64/32 图的一轮 CPU 训练与权重重加载。主权重内含 1–100 轮训练记录；完整 100 轮未在整理时重训，见[训练验证](../provenance/training_validation.json)。
- 初期选型及 108/2,916 网格保留经过筛选的现有记录，没有全部重跑；未验证机器人/板端部署。

另外从公开文件独立导出了一个不含 Git 历史的目录，运行全部测试、真实模型视频，并重新评分三个集合的 15 行策略结果；结果全部一致，源代码和评估输入均来自该目录，见[独立目录验证](../provenance/portable_validation.json)及[预测一致性](../provenance/prediction_equivalence.json)。

运行 `python tools/verify_results.py` 校验公开数字与哈希，`python tools/check_release.py` 检查代码依赖、文档链接和误纳入的大文件。验证记录放 `provenance/`，本地训练、视频和日志放 `outputs/`。GitHub Actions 配置已准备，但尚未在远端运行；本地验证平台为 Windows/Python 3.12。
