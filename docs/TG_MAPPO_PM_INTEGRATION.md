# 第 6 步：带删失数据的 PM 最小版

日期：2026-11-03。**独立预训练、冻结接入、分项奖励、短训练与冻结评估已通过。**
当前没有真实 TTL/丢弃失败记录，实现的是投递时间 PM；原案的完整成败分类、可靠性门控及性能收益尚未验收。
继续使用原 TG-MAPPO 和原浮标环境，没有新增研究环境，没有把未结束包改标失败。

## 1. 标签与模型定义

原数据的 `outcome=1` 表示真实投递；`outcome=null,censored=true` 表示观察截止仍未结束。
当前 `life=80` 是静态字段，不能当作剩余 TTL。本步没有修改 TTL、包删除、原奖励或任务规则。

因此最小版采用离散投递时间的右删失似然，而非伪造最终失败标签：

- 已投递包：前缀后每个“尚未投递”时隙的生存项，加上实际投递时隙的事件项。
- 未结束包：只使用已经观察到的“尚未投递”时段，不对截止之后的结果作标签。
- 按包均匀取样，包内所有前缀取平均，避免长轨迹支配损失；不人为改变成功/删失比例。
- 截止之前已知的有限窗口可以验证是否发生投递，但该标签不等于最终成败。不能确认的窗口从 Brier 中屏蔽。

损失为每个前缀 `sum(softplus(z_k), 已观察时隙) - z_投递时隙（仅真实投递）`，先包内平均，再按包平均。
方法依据：[Kvamme 与 Borgan，离散时间生存预测](https://arxiv.org/abs/1910.06724)。这是已有统计方法在当前数据条件下的适配，本身不作为创新收益。

模型为 `14 维数值特征 → train-only 归一化 → Linear(14,32)/Tanh → 单向 GRU(64) → 80 维投递风险 logits`，共 **24,496 参数**。
输入包括当前 Sink 距离、队列、出口距离/链路统计、已用/剩余观察窗口、历史跳数、回访、等待及邻居变化。
节点编号仅用于组织已知路径，不作为模型特征。剩余窗口指 `80-t`，不冒充 TTL。
最终结果、未来状态、真实丢包率和诊断字段不进入输入；单向 GRU 不读取未来前缀。WM 不参与 PM 预训练。

时隙 t 的评分是下一 `80-t` 时隙内发生投递的预测值：`score = 1 - product(1-sigmoid(z_k))`。
它受混合行为数据与前缀抽样分布影响，未完成面向当前策略的独立概率校准，不是最终投递概率保证，也不代表因果贡献。

## 2. 势函数与真实接入点

对本回合初始 P 个包固定归一化：`Phi_t = sum(score_p(t), 尚未投递的包)/max(P,1)`。
实际投递回调立即结束该包的势函数与历史。完整任务结束时 `Phi_T=0`，未结束包仍记删失。

```text
原 Actor → 原环境执行 → 原团队奖励 r_env + 下一观测/包路径
                              ↓
冻结 PM：Phi_t、Phi_tp1 → r_progress = alpha*(gamma*Phi_tp1-Phi_t)
                              ↓
r_train = r_env+r_progress → 广播一次给各节点 → 原 buffer → GAE → PPO
```

- gamma 取 `GraphReplayBuffer.gamma`，本次为 0.99，不使用历史 DQN 包装的 `reward_decay=0.9`。
- alpha 默认 1；PM 权重、归一化、alpha、gamma 在任务内固定，PM 无梯度且不进入 PPO 优化器。
- 塑形有正有负，不裁负项；团队项只算一次，不乘节点数。
- `logs/TG_PM_rewards.csv` 逐时隙保存 r_env、Phi_t、Phi_tp1、r_progress、r_train 和边界标志。
- 控制台回报和原指标继续按 r_env/实际包结果记录，Critic/GAE 学习 r_train。
- 当前完整 rollout 就是完整有限任务。通用函数已测试非终止截断保留势函数，但原 runner 尚不支持跨 rollout 延续任务。
- 每回合验证 `sum(gamma**t*r_progress_t)=-alpha*Phi_0`。原环境历史奖励跨回合保留，不能据此进一步宣称跨回合全局策略排序或 PPO 性能保证。
- 可靠性 gate 目前固定为 1，没有不确定性模型或自适应门控。

## 3. 文件与扩展接口

| 文件 | 职责 |
|---|---|
| src/buoy_network/progress_model.py | 当前/历史数值特征、真实 trace 读取、删失损失、模型与评估 |
| scripts/train_tg_pm.py | 审计原数据、独立预训练、验证选模、保存权重；不评估 test |
| src/buoy_network/progress_runtime.py | 冻结加载、按包 GRU 历史、实际投递回调、团队势函数和边界核验 |
| src/buoy_network/runner.py | 环境之后/GAE 之前组合奖励，保存独立 CSV |
| src/buoy_network/agent.py | 核对 shaped buffer，PPO 前后检查 PM 冻结 |
| src/buoy_network/policy_artifacts.py | 保存/加载 PM SHA、alpha/gamma；兼容未使用 PM 的旧 TG checkpoint |
| scripts/test_tg_pm.py | 按函数→模块→原 TG 联调执行 |
| scripts/verify_tg_pm_runs.py | 独立审计最终真实运行、权重、奖励代数及测试凭据 |

原 TG 开关是 `--use_pm_reward true --pm_reward_checkpoint PATH --pm_reward_alpha 1`。
原研究版 `--use_progress_reward` 仍被原 TG 入口拒绝，避免走错实现。
当前只支持已验证的 25 节点、半径 1.5、对应 checkpoint 时长；不与原始数据导出/规则采集同时启用。
增加真实失败事件需版本化标签/任务语义，并对所有对照组统一重跑；增加特征需同步升级 schema、归一化和加载校验。

## 4. 本次真实预训练

- 数据：`results/tg_wm_multiscene/20261102_235951_684782`，复用原环境已审计数据。
- train：场景 0/1/2/3，576 个包，286 投递/290 删失；validation：场景 101，144 个包，70 投递/74 删失。
- 全部归一化与风险统计基线仅拟合 train，test=202 本轮未做模型推理或选模。
- Python 3.10.9 / Torch 2.11 CPU，seed=12，40 epoch，batch=24，lr=0.001；按验证门槛选第 29 epoch。
- 验证 survival NLL **1.375765**，train 风险统计基线 **2.093271**；窗口 Brier **0.073161**，基线 **0.213990**。
- 此处统计是按包/前缀的有限窗口指标，不是最终失败分类指标或路由性能结果。
- 最终目录：`results/tg_pm_final_pretrain/20261103_142745_276042`，包含 pm_best.pt、learning_curve.csv、report.json。
- 权重 SHA256：`f62f05b520cbaf665823ce38981c6c88f5e4f816dc01f85d3149aca7d3f81240`。
- 报告包含审计、参数、数据划分、实际墙钟时间；该时间有并行开发运行影响，不作正式速度基准。
- 原 TG 的 Torch 1.8 另行加载最终权重并重算全部 validation，NLL/Brier 与 Torch 2.11 上述数值一致。

## 5. 已运行的原 TG 验收

训练/评估均使用 `D:\python3.6.8\python.exe`，Torch 1.8.1+cpu，25 节点、5 移动配置、随机信道。
每组训练 seed=0、2×80 时隙、PPO epoch=1/minibatch=2；各组统一使用下述 Huber 修复。

| 组别 | 最终训练目录（results 下，均取 moves_5） | 两回合投递率 |
|---|---|---|
| TG | tg_step6_final_baseline/runs/20261103_142752_782391 | 0.375 / 0.250 |
| TG+WM | tg_step6_final_wm/runs/20261103_142752_052297 | 0.375 / 0.250 |
| TG+PM，alpha=0 | tg_step6_final_alpha0/runs/20261103_142940_292196 | 0.375 / 0.250 |
| TG+PM | tg_step6_final_pm/runs/20261103_142940_497301 | 0.375 / 0.250 |
| TG+WM+PM | tg_step6_final_wm_pm/runs/20261103_142939_927101 | 0.375 / 0.250 |

alpha=0 的 Actor/Critic 全部权重与同版本 TG 完全一致，指标 CSV 一致。其余组 Actor/Critic 正常更新，WM/PM 权重与归一化不变。
PM 与组合组各有 160 行真实奖励，塑形范围约 [-0.14114,0.01904]，两回合折扣边界误差均为 0。
四组另在独立场景/策略种子 404 各做 2×80 greedy 冻结评估，投递率均为 0.166667/0.166667，优化器更新均为 0，权重不变。
组合组重复评估逐字段一致。原平均跳数/时延仍为旧口径，不称成功包真实指标。
完整十个最终运行目录见[清单](evidence/TG_PM_STEP6_RUNS_20261103.json)，配置/指标/模型 SHA 与源码字节摘要见[验收凭据](evidence/TG_PM_STEP6_20261103.json)。
另加载第 5 步旧 baseline checkpoint 完成 1×80 冻结兼容检查，退出码 0：results/tg_step6_old_checkpoint_eval/runs/20261103_144210_275541/moves_5。
当前没有性能提升或收敛证据；303/404 已用于功能调试，后续正式盲测需另定场景。上述五个短训练组不等于多独立训练种子实验。
权重和大体积运行结果保留本地 results；Git 仅保存代码、文档和小型凭据，尚无优化器/RNG 训练续跑。

## 6. 命令

仓库根目录执行；每个代码块是一条完整命令。

重新预训练（生成新时间戳）：

```powershell
& "D:\python3.10.9\python.exe" scripts/train_tg_pm.py results/tg_wm_multiscene/20261102_235951_684782 --epochs 40 --output-root results/tg_pm_final_pretrain
```

PM 单独短训练：

```powershell
& "D:\python3.6.8\python.exe" run_algorithm.py --algorithm tg-mappo --num_episodes 2 --episode_length 80 --num_env_steps 160 --ppo_epoch 1 --num_mini_batch 2 --num_moving_buoys 5 --use_pm_reward true --pm_reward_checkpoint results/tg_pm_final_pretrain/20261103_142745_276042/pm_best.pt --pm_reward_alpha 1 --output_root results/tg_step6_final_pm
```

组合短训练在上行追加 `--use_wm_actor true --wm_actor_checkpoint results/tg_wm_multiscene_gate_selection/20261103_000951_565881/wm_best.pt`，输出目录改为 `results/tg_step6_final_wm_pm`。
零系数退化检查改为 `--pm_reward_alpha 0`，输出目录改为 `results/tg_step6_final_alpha0`。

加载本次组合权重做冻结评估：

```powershell
& "D:\python3.6.8\python.exe" run_algorithm.py --algorithm tg-mappo --num_episodes 2 --episode_length 80 --num_env_steps 160 --ppo_epoch 1 --num_mini_batch 2 --num_moving_buoys 5 --tg_evaluate true --seed 404 --scenario_seed 404 --use_pm_reward true --pm_reward_checkpoint results/tg_pm_final_pretrain/20261103_142745_276042/pm_best.pt --use_wm_actor true --wm_actor_checkpoint results/tg_wm_multiscene_gate_selection/20261103_000951_565881/wm_best.pt --tg_policy_checkpoint results/tg_step6_final_wm_pm/runs/20261103_142939_927101/moves_5/checkpoints/tg_policy_latest.pt --output_root results/tg_step6_final_eval_wm_pm
```

加载新训练权重时换成新时间戳，模型、PM/WM 开关和 alpha/gamma 必须匹配保存配置。

## 7. 测试、失败原因及未完成项

最终 **76 项通过，0 失败、0 错误、0 跳过**：原 TG/数据 51、WM 接入 8、PM 接入 6、PM 基础 5、原 WM 基础 6。
分层命令：`D:\python3.6.8\python.exe scripts/test_tg_pm.py`，函数 2/模块 2/联调 2，顺序执行。
原回归和 WM 接入分别运行 scripts/test_legacy_fixes.py --skip-smoke、scripts/test_tg_wm_actor.py；3.10 运行 tests.test_tg_progress_model、tests.test_tg_world_model。
PM 基础测试验证删失似然、未来隔离、GRU 在线一致性、真实梯度与归一化保持、未知窗口屏蔽。
PM 联调验证 alpha=0 精确退化、真实导出和在线特征逐前缀一致、手工 GAE 对照、冻结优化器边界及独立评估。

发现并修复：原 `onpolicy/utils/util.py::huber_loss` 用 `e>delta` 判大误差，遗漏 `e<-delta`，导致未归一化大负回报下 Critic 零梯度。
修复为 `abs(e)>delta`，增加对称损失/梯度回归；之前失败的 Critic 联调现已通过。最终五个训练组和四组独立评估已统一重跑，修复不归因于 PM/WM。
同时修复未确定窗口不能计作零 Brier 误差的边界。早期调试目录和失败测试日志仍保留，正式凭据使用最终通过结果。

未完成：真实 TTL/丢弃/任务失败的结果分类、成功/失败分层训练、可靠性门控、正式多种子预算与匹配历史/容量的消融、标准物理指标、跨 rollout 任务、GPU 与训练续跑。
第 6 步最小版工程路径完成，不能标成原研究方案全部 PM 功能完成。下一步先准备第 7 步正式对照预算/配置，并保留以上缺项。
