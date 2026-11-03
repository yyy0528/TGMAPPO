# 第 5 步：冻结 WM 接入原 TG-MAPPO

日期：2026-11-03。**接入、短训练、权重保存与冻结评估已通过；尚未证明性能提升。**
仍使用 `run_algorithm.py --algorithm tg-mappo`、原浮标环境和原 TG Actor/Critic。PM 未启用。

## 1. 实际接入位置

```text
当前观测/图/合法掩码 + 本节点过去动作/ACK
                ↓
冻结 WM，对合法候选动作做预测（Actor 采样前）
                ↓
原 Actor 图编码/MLP → 原动作 logits + 可训练 WM 融合残差 → 采样动作
                ↓
原 Runner/环境执行 → 原奖励、下一时隙观测、实际 ACK
                ↓
buffer：原有轨迹 + 采样时 WM 特征 → PPO 更新 Actor/Critic
```

WM 独立于 Actor 主干，使用 `eval`、`no_grad`、`requires_grad=False`；不进入 PPO 优化器。
Actor 内增加 `Linear(hidden+3,32) → Tanh → Linear(32,1)`，逐候选动作产生分数残差。
hidden=64 时新增 **2,209 个可训练参数**，WM 本体 **66,789 个冻结参数**。
残差最后一层初始化为零；初始化不消耗原策略 RNG，原 Actor/Critic 初始权重与关闭 WM 一致。
没有采样后替换动作；环境执行的就是策略动作。

输入沿用现有全图观测，不宣称严格局部执行。WM 不读取未来状态、当前其它节点动作、最终包结果或私有仿真真值。
当前时隙真实 ACK 只在下一时隙进入历史；每回合清空历史，窗口最多 4 步。

## 2. 特征、缓存与奖励

每个候选动作的 3 个数，按固定顺序：

| 字段 | 含义 |
|---|---|
| ack_probability | 本次尝试成功概率，来自验证过的 ACK 头 |
| sender_joint_queue_fraction_delta | 到下一联合时隙发送节点队列占比变化 |
| receiver_joint_queue_fraction_delta | 到下一联合时隙接收节点队列占比变化 |

队列预测包含其它节点同期行为的影响，是条件预测，不是固定其它动作的因果反事实。
队列变化裁剪至 [-1,1]；非法候选全部为 0，概率由当前掩码置 0。
WAIT 的 ACK 和接收方变化置 0；仅能 WAIT 的节点不预测，三个数均为 0。
位移、独立投递/丢弃、可靠性/不确定性未接入，报告明确列为 not_used。
ACK 在 20–39 阶段仍弱于链路统计基线，不能把总体验证通过理解为所有阶段可靠。

- 在线特征形状 `[1,25,26,3]`，buffer 缓存 `[T,1,25,26,3]`，float32；T=80 约 0.60 MiB。
- 采样和 PPO 使用同一缓存，与观测/动作/掩码使用同一 minibatch 索引；更新时不重新运行 WM。
- 每次 PPO 前重放整段行为概率，要求 `max(abs(exp(logp_replay-logp_old)-1)) <= 1e-4`。
- 更新前后核对 WM 全部权重和归一化 SHA256、缓存 SHA256；记录 Actor/Critic/融合层是否更新。
- 缓存留在本次进程内；报告保存摘要，checkpoint 不支持恢复未完成的 rollout。
- Critic 维持原输入；训练回报仍来自原环境 dense/历史路径奖励，尚无 PM 奖励。

## 3. 模块职责和边界

| 文件 | 本步职责 |
|---|---|
| src/buoy_network/world_model.py | 拆分纯输入 encode_inputs 与监督标签 encode_sample，复用相同编码 |
| src/buoy_network/wm_runtime.py | 校验/冻结 WM、维护历史、候选推理、冻结摘要 |
| src/buoy_network/agent.py | collect 缓存、train 概率重放和冻结核验 |
| src/onpolicy/algorithms/graph_actor_critic.py | Actor 候选残差；Critic 不改输入 |
| src/onpolicy/algorithms/graph_MAPPOPolicy.py | 透传可选 wm_features |
| src/onpolicy/utils/graph_buffer.py | 特征缓存、采样对齐、回合清空 |
| src/onpolicy/algorithms/graph_mappo.py | PPO 读取缓存特征 |
| src/buoy_network/runner.py | 记录真实尝试 ACK；冻结评估跳过 compute/train |
| src/buoy_network/policy_artifacts.py | 原 TG 权重保存、配置校验加载、冻结评估报告 |

当前支持 25 节点、半径 1.5、单环境、前馈 TG。WM 历史 GRU 不等于 Actor 使用循环策略。
WM 与规则采集/原始数据导出不同时启用；循环 Actor、其它节点数/通信半径需要单独适配和验证。
原研究版 `--use_world_model` 仍被原 TG 入口拒绝；本步开关是 `--use_wm_actor`。
扩展预测字段时，应同时升级特征 schema、融合输入、buffer 和兼容校验，再重跑概率回放测试。

## 4. 已实际执行的训练与评估

解释器 `D:\python3.6.8\python.exe`，Torch 1.8.1+cpu；25 节点、5 移动节点配置、随机信道。
WM 使用上一阶段 v4、epoch 2，SHA256 `2919dff60d3128321170410e632e7bddd476a4f09c766b92108bfe9cb73daca9`。
功能评估采用独立场景 303 和策略种子 303、greedy 解码、2×80 时隙；没有重新查看 WM test=202。

| 运行 | 产物目录（results 下） | 两回合投递率 |
|---|---|---|
| 关闭 WM 短训练，种子 0 | tg_step5_final_baseline/runs/20261103_134541_832767/moves_5 | 0.375、0.250 |
| 开启 WM 短训练，种子 0 | tg_step5_final_wm/runs/20261103_134541_780345/moves_5 | 0.375、0.250 |
| 关闭 WM 冻结评估，场景 303 | tg_step5_final_baseline_eval/runs/20261103_134829_709841/moves_5 | 0.166667、0.166667 |
| 开启 WM 冻结评估，场景 303 | tg_step5_final_wm_eval/runs/20261103_134829_660838/moves_5 | 0.166667、0.166667 |
| 开启 WM 重复冻结评估 | tg_step5_final_wm_eval_repeat/runs/20261103_134829_537564/moves_5 | 与上行逐字段相同 |

关闭 WM 的 CSV 与第 1 步历史基线完全一致。WM 两次 PPO 前重放误差均为 **0.0**，Actor/Critic/融合层都更新，WM 保持不变。
三次冻结评估均为 **0 次优化更新**，Actor/Critic/WM 权重不变。
两组短训和本次评估未显示 WM 提升；不能据此判断长期收益或收敛。场景 303 已用于功能调试，正式盲测应另选未查看的场景。
average_hops/average_delay 继续使用旧口径，不当作成功包真实跳数/时延。本步未改变环境、任务或奖励规则。

训练和评估各自保存完整配置、CSV、日志、图和 `logs/TG_policy_report.json`。
训练另存 `checkpoints/tg_policy_latest.pt`，包含 Actor/Critic、PopArt/可选 ValueNorm、配置与 WM 摘要。
加载校验环境、网络结构/激活配置、WM 开关和文件 SHA；支持重新评估，不包含优化器/RNG 训练续跑。
模型文件和大体积运行结果仍在本地 results；Git 保存代码、文档及[小型验收凭据](evidence/TG_WM_ACTOR_STEP5_20261103.json)。

## 5. 下次可直接使用的命令

仓库根目录执行；每个代码块都是一条完整 PowerShell 命令。

关闭 WM 的短训练：

```powershell
& "D:\python3.6.8\python.exe" run_algorithm.py --algorithm tg-mappo --num_episodes 2 --episode_length 80 --num_env_steps 160 --ppo_epoch 1 --num_mini_batch 2 --num_moving_buoys 5 --output_root results/tg_step5_final_baseline
```

开启 WM 的短训练：

```powershell
& "D:\python3.6.8\python.exe" run_algorithm.py --algorithm tg-mappo --num_episodes 2 --episode_length 80 --num_env_steps 160 --ppo_epoch 1 --num_mini_batch 2 --num_moving_buoys 5 --use_wm_actor true --wm_actor_checkpoint results/tg_wm_multiscene_gate_selection/20261103_000951_565881/wm_best.pt --output_root results/tg_step5_final_wm
```

加载本次 TG+WM 权重冻结评估：

```powershell
& "D:\python3.6.8\python.exe" run_algorithm.py --algorithm tg-mappo --num_episodes 2 --episode_length 80 --num_env_steps 160 --ppo_epoch 1 --num_mini_batch 2 --num_moving_buoys 5 --tg_evaluate true --seed 303 --scenario_seed 303 --use_wm_actor true --wm_actor_checkpoint results/tg_wm_multiscene_gate_selection/20261103_000951_565881/wm_best.pt --tg_policy_checkpoint results/tg_step5_final_wm/runs/20261103_134541_780345/moves_5/checkpoints/tg_policy_latest.pt --output_root results/tg_step5_final_wm_eval
```

关闭 WM 的评估去掉两个 WM 参数，将策略路径换成 baseline 行目录下的 checkpoint。
每次训练会生成新时间戳；评估新权重时要使用新打印目录，不能继续加载上例旧路径。

## 6. 测试与未完成项

按函数 → 模块 → 联调执行；总计 **64 项通过、0 失败、0 错误、0 跳过**。

```powershell
& "D:\python3.6.8\python.exe" scripts/test_tg_wm_actor.py
```

本步 8 项：函数 3、模块 3、联调 2。覆盖纯输入编码、历史反馈/复位、损坏权重拒绝、RNG 保持、初始中性、非法动作/全 WAIT、shuffle 对齐、训练后预测影响、缓存复用、WM 不进优化器和冻结评估。

```powershell
& "D:\python3.6.8\python.exe" scripts/test_legacy_fixes.py --skip-smoke
```

原 TG/数据回归 50 项：20/24/6 分阶段通过；真实 2×80 另外实际运行，见上表。

```powershell
& "D:\python3.10.9\python.exe" -m unittest tests.test_tg_world_model -v
```

WM 训练模块 6 项通过。3.10 仅作独立 WM 测试，原 TG 训练与评估都使用 3.6，不要求迁移环境。
逐阶段原始报告与运行配置、指标、文件和源码 SHA256 已汇总进上述验收凭据；核验入口 `scripts/verify_tg_wm_actor_runs.py --help`。

曾失败：新测试入口未加入仓库根目录，导致旧 Runner 的 `src.experiment_io` 导入失败；修正测试路径后通过。
复查补齐了配置校验，防止相同权重形状却使用不同激活函数；最终训练/评估均在补齐后重跑。
CPU 的 CUDA AMP warning 表示自动关闭 AMP，不是测试失败；没有安装依赖或更改环境。

待后续：PM 训练与奖励接入、多独立训练种子/正式预算/盲测、参数量与历史信息匹配的 Actor 对照、完整预测头和可靠性、GPU 实测、训练断点恢复。
正式资源比较需分开统计可训练融合参数、冻结 WM 参数、预训练成本和本步整段概率重放的诊断开销。
第 6 步首先明确 PM 的成功/失败/删失标签与训练数据；原数据的未结束包不能直接标作失败。
