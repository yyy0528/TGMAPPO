# 第 7 步：实验工具与三种子短预实验

日期：2026-11-03。用户选择本轮“先完成工具和 3 个训练种子的短预实验”。
**工具与预实验已完成；正式长训练未启动，不能作为论文性能验收。**
主线仍为原 TG-MAPPO/原浮标环境；WM/PM 沿用前两步最小版与局限。

本页原结果属于 legacy_v1。后续[task_v2 奖励实验](TG_MAPPO_TASK_REWARD_V2.md)已另目录统一重跑六组；两版奖励回报不可混合，物理指标与结果分别保存。

公共学习器的[PopArt修复与六组复跑](TG_MAPPO_POPART_FIX.md)随后完成；该实现修复不算辅助模型收益，20回合基线诊断也单独保存。

随后[学习率开关修复](TG_MAPPO_LR_SWITCH_FIX.md)完成：后续套件使用v3显式fixed/linear，旧schema按其历史实际linear计划兼容；115项与两套72任务通过，旧线性18组参数及36份物理日志保留一致。

## 1. 六个对照组

| 组 | 原 TG 上的变化 | Actor 可训练参数 | 冻结辅助参数 |
|---|---|---:|---|
| A | 无 WM/PM | 21,956 | 0 |
| B | 学习得到的 WM ACK/队列预测 | 24,165 | WM 66,789 |
| C | 删失投递时间 PM 塑形 | 21,956 | PM 24,496 |
| D | WM+PM | 24,165 | WM 66,789 + PM 24,496 |
| E | Actor 增加 64→17→64 残差，无 WM/PM | 24,213 | 0 |
| F | 同 B 的候选融合层，输入改为四帧历史统计 | 24,165 | 载入但不调用的 WM 66,789 |

所有组 Critic 可训练参数均为 18,616；上述不包含历史包装中未用于 TG PPO 的 DQN 网络。
E 新增 2,257 参数，与 B 融合层新增 2,209 相差 2.17%，匹配的是**新增可训练 Actor 容量**，不是 WM+Actor 的总容量。
E 保持 Critic 与基础 Actor 的结构/初始化；新增残差初始为零，初始化不推进原策略 RNG。
F 使用与 B 相同的四帧观察/本节点过去动作和 ACK；用匹配接收者的历史 ACK/Beta 统计、发送/接收队列历史平均变化取代学习预测。
F 不调用 WM.forward，已用真实 PPO+禁用 forward 的测试验证。为保持接口/窗口/版本校验，仍载入 WM，内存中的闲置参数与运行成本明确记录。
F 是压缩历史统计对照，不是所有可能历史编码器的等价对照；总容量匹配、历史 Critic 等仍待正式实验补充。

## 2. 统一配置与统计单位

预实验配置：[tg_ablation_pilot.json](../configs/tg_ablation_pilot.json)。

- Python 3.6.8 / Torch 1.8.1+cpu，25 节点，移动节点配置 5，半径 1.5，随机信道。
- 训练种子 **11、22、33**；每个种子启动独立进程、初始化独立 Actor/Critic，场景种子同该训练种子。
- 每组每种子 **2 回合×80 联合时隙=160 训练步**；PPO epoch=1/minibatch=2。
- 每份最终权重另在场景/解码种子 **7001** 做 2×80 greedy 冻结评估，不更新任何网络。
- 6 组×3 种子=18 个训练任务，加 18 个冻结评估任务；共训练 2,880 步、评估 2,880 步。
- WM/PM 权重 SHA、原环境、奖励/PPO 参数固定；种子固定相同初始条件，不保证不同动作消耗 RNG 后的所有随机事件逐帧相同。
- 不以评估挑选中间 checkpoint，统一使用预定预算完成后的最终权重；不反复查看旧 test=202。
- 先按同一训练策略汇总其评估结果，再在三个独立训练策略之间求均值与样本标准差。两次评估回合不冒充两个独立训练种子。

固定观察范围仍是原全图输入，原 dense/历史路径奖励保持不变；本次不是稀疏任务实验。
7001 现已用于预实验开发，后续不能再标作最终盲测场景。

## 3. 新增物理指标，保留旧 CSV

新增旁路 `physical_metrics.py::PacketMetrics`，由真实尝试/成功转发事件驱动，每回合写 `logs/TG_physical_metrics.jsonl`。
它不改变队列、动作、奖励或随机数；独立 seed=0 的 2×80 对比已确认 Actor/Critic 全部权重与第 6 步基线相同，旧 CSV 字节相同。

| 新字段 | 定义/分母 |
|---|---|
| generated/delivered/unresolved_packets | 初始实际包数 / 实际到 Sink 的包数 / 截止仍未结束包数 |
| delivery_rate、unresolved_fraction | 上述计数分别除 generated |
| success_hops_mean | 成功包真实转发次数 len(route)-1 的平均值 |
| success_delay_mean | 从生成时隙 0 到真实投递时隙 t+1 的联合时隙数，分母仅成功包 |
| success_delay_p95 | 成功包原始时延的 nearest-rank 第 95 百分位，按训练策略汇总原始样本后计算 |
| ack_successes/ack_failures | 真实发送尝试的成功/失败次数；尝试失败不等于最终丢包 |
| retry_attempts | 同一包上一次尝试失败、尚未成功前，再进行的发送尝试次数 |
| loop_transfers | 成功转发到该包之前访问过节点的次数 |
| true_drop_count/rate | null；当前环境没有真实 TTL/丢弃机制，不能把 unresolved 当 drop |

无成功包时成功跳数/时延为 null，不能填 0；汇总记录 n_defined。旧 average_hops/average_delay 文件继续原定义，不与新字段混用。
Collector 检查 ACK/转发对应、包守恒和每包每时隙最多一跳。

## 4. 工具、快照与任务接续

入口 `scripts/run_tg_ablation.py`，模块 `src/buoy_network/ablation_suite.py`。

- plan：严格校验预算、种子/场景分离、模型 SHA、配置与小批量整除，不启动训练。
- run：创建独立时间戳 suite，复制真实源码/脚本/配置，逐文件 SHA256；**每个子任务实际从 source_snapshot 执行**。
- 保存依赖版本、父 Git HEAD 和 130 个快照文件摘要。HEAD 只作父版本，不冒充未提交工作区源码。
- 冻结的 manifest 包含每条完整命令、种子、依赖关系、退出码、用时、尝试次数、产物目录与摘要。
- 每个训练任务后立即执行它的冻结评估，任一异常则停止；失败重试用 attempt_2 等新目录，保留原日志。
- 单写入者文件锁防止两个调度器同时写同一 suite；resume 校验源码和已完成产物摘要，跳过已完成任务。
- 接续是任务级：重跑未完成训练任务，不支持恢复其半途的优化器/RNG/rollout。已完成 suite 的接续实测新调度任务数为 0。
- 汇总失败/不完整 suite 会明确报错，不生成“成功”性能表。

`logs/TG_policy_report.json` 另存 Actor/Critic/冻结辅助参数数目、设备及 CPU collect/PPO/诊断墙钟计时。
collect 包含候选特征和策略前向，PPO 为优化更新，诊断另记；不将其冒充纯环境耗时。任务整体时间另外包含启动/环境/保存/画图等开销。
本轮有并行测试干扰，计时只作开发记录，不能直接当正式速度基准；CUDA 峰值为 null。
suite 另复制最终 WM/PM 预训练和数据清单，辅助采样/预训练成本单列；没有声称完整汇总历次开发成本。
大模型/原始数据/完整 suite 仍在本地 results，Git 保存源码、配置、文档和小型凭据。

## 5. 真实结果

运行目录：`results/tg_ablation_pilot/20261103_151015_964923`。
**36/36 子任务成功，所有任务均一次完成；18 份独立训练权重，冻结评估权重不变。**

| 组 | 训练种子 11 的评估投递 | 种子 22 | 种子 33 | 按三个策略的均值 ± 样本 SD |
|---|---:|---:|---:|---:|
| A | 47/48 | 0/48 | 0/48 | 0.326389 ± 0.565322 |
| B | 47/48 | 0/48 | 0/48 | 0.326389 ± 0.565322 |
| C | 47/48 | 0/48 | 0/48 | 0.326389 ± 0.565322 |
| D | 47/48 | 0/48 | 0/48 | 0.326389 ± 0.565322 |
| E | 47/48 | 0/48 | 0/48 | 0.326389 ± 0.565322 |
| F | 47/48 | 0/48 | 0/48 | 0.326389 ± 0.565322 |

本次没有组间提升，种子差异很大；2 回合预算仅验证工具链，不能证明创新无效、策略收敛或显著差异。
成功包指标只有一个训练种子可定义，汇总 n_defined=1、SD=null；不能把另两个种子无投递写成零时延。
预定义“连续 20 回合窗口平均投递率≥0.8”的训练阈值，当前只有 2 回合，因此达到阈值步数未定义。

产物包括 summary.json、group_summary.csv、全部逐回合物理指标，以及 figures/evaluation_delivery.png/.svg、training_delivery.png。
图展示每个独立策略的点和样本 SD，不是置信区间；已实际查看生成图。标准差条可以越出 [0,1]，不表示负投递概率。
小型[验收凭据](evidence/TG_ABLATION_PILOT_20261103.json)保存汇总、测试、基线一致性、快照/产物 SHA 和接续结果。

## 6. 可复制命令

仓库根目录执行，每个代码块为一条完整 PowerShell 命令。

仅查看短预实验预算：

```powershell
& "D:\python3.6.8\python.exe" scripts/run_tg_ablation.py --config configs/tg_ablation_pilot.json --mode plan
```

重跑整个短预实验，生成新 suite：

```powershell
& "D:\python3.6.8\python.exe" scripts/run_tg_ablation.py --config configs/tg_ablation_pilot.json --mode run --output-root results/tg_ablation_pilot
```

接续本次 suite（已完成时仅校验/汇总，不重复训练）：

```powershell
& "D:\python3.6.8\python.exe" scripts/run_tg_ablation.py --resume results/tg_ablation_pilot/20261103_151015_964923 --mode run
```

导出描述性图/CSV：

```powershell
& "D:\python3.6.8\python.exe" scripts/plot_tg_ablation.py results/tg_ablation_pilot/20261103_151015_964923
```

新工具分层测试：

```powershell
& "D:\python3.6.8\python.exe" scripts/test_tg_ablation.py
```

正式预算提案仅查看，不启动：

```powershell
& "D:\python3.6.8\python.exe" scripts/run_tg_ablation.py --config configs/tg_ablation_formal_proposal.json --mode plan
```

正式提案为 6 组×3 训练种子（1011/2022/3033），每组每种子 1000×80，合计 1,440,000 训练步；三个预留盲测场景 91001/91002/91003×20 回合，86,400 评估步。
本轮没有运行该提案，也没有查看其场景结果。正式前仍需复核预算、总容量对照、PM 缺项和计算成本。

## 7. 测试与下一步

最终 **86 项回归通过、0 失败/错误/跳过**：原 TG/数据 51、WM 接入 8、PM 接入 6、辅助模型 11、新工具 10。
新工具按函数 6→模块 4 执行；随后完成 36 个真实训练/评估子任务，并独立重新汇总、检查所有摘要与基线权重。
失败路径测试主动退出码 7，证明失败任务新尝试保留旧日志；这不是实际预实验失败。
核验器首次因 Python 3.6/3.10 标准差相差约 1.1e-16 而报错，已先说明原因后对派生浮点统计采用 1e-12 容差；原始文件/源码/权重 SHA 仍严格一致。固定解释器接续的 summary 字节不变。

未完成：正式训练/多盲测场景、总模型容量匹配对照、历史 Critic 等扩展、PM 真失败分类/可靠性、GPU 与训练半途恢复。
本轮按用户选择完成并停止在短预实验。下一轮先诊断两个零投递种子的动作/环路与初始策略，确定中等预算，再考虑正式实验；不能直接将当前结果当论文结论。

## 8. 后续只读诊断（2026-11-03）

零投递诊断已完成，见[动作/环路/奖励说明](TG_MAPPO_ZERO_DELIVERY_DIAGNOSIS.md)。确认物理可达而 greedy 困在环路/等待，六组动作序列相同；奖励存在等待零成本的冷启动风险。
24 个冻结诊断复现和 2 个规则奖励探针不是新的训练或正式消融结果，不改变本文件的原三种子统计。下一次先评审奖励契约并小预算验证，尚未实施修复。
