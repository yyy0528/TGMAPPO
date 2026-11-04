# 原TG-MAPPO：固定学习率20/100回合交互预算对照

日期：2026-11-04。状态：**100回合实验进行中，尚无最终性能结论。**

接续[Critic时间信息验证](TG_MAPPO_CRITIC_TIME.md)，保持时钟disabled、Actor/Critic结构、原环境和task_v2。这里增加新的环境采样回合，每回合PPO仍1×2；不复用同一buffer增加ppo_epoch。

## 1. 对照条件

| 项目 | 两种预算共同条件 |
|---|---|
| 环境 | 原25节点、5移动浮标、随机信道、80联合时隙/回合 |
| 初始训练种子 | 11、22、33 |
| 奖励 | task_v2，投递/时间/窗口权重均1 |
| 模型 | 原TG-MAPPO，WM/PM关闭，Critic时钟disabled |
| 学习率 | fixed；Actor0.0005、Critic0.0003，全程不衰减 |
| PPO | 每rollout 1遍、2小批次 |
| 冻结评估 | 开发场景7001，各2×80，greedy、更新0 |

20回合=1600环境交互，100回合=8000；三种子分别4800/24000。新100回合每网络应实际200步Adam，20回合前缀应40步。
新数据量与实际优化次数同时增加5倍，这是保持每回合PPO配置时的自然结果，不能把它说成等计算预算。
训练回合存在历史观测反馈延续，不是统计独立样本；统计单位仍为独立训练种子n=3。

## 2. 保留旧fixed20基线

直接使用results/tg_lr_budget_fixed，不重训、不改写旧权重/日志。
配置、奖励分项/真实投递时间、模型SHA和冻结评估已只读核验：

| 种子 | 旧fixed20投递/48 |
|---|---:|
| 11 | 12 |
| 22 | 10 |
| 33 | 14 |

旧均值25.00%±4.17%样本SD。旧记录有Critic头40次Adam，但当时没有Actor累计计数字段；保持缺失信息，不编造历史实测值。
旧模型训练源码位置及原命令保存在该目录commands.json，新套件baseline20.json另保存原模型/物理日志哈希。

## 3. 工具和严格前缀检查

本轮未修改项目训练实现，只新增外部入口适配器和预算配置：

- scripts/interaction_budget_worker.py：沿用原run_algorithm默认参数及buoy_network.main；在第20次PPO更新后只读保存Actor/Critic状态，立即与旧fixed20逐位比较，不一致就停止该任务。
- scripts/run_tg_interaction_budget.py：核验旧参考、冻结源码、生成固定100回合套件，沿用原任务锁、超时处理、独立尝试和接续。
- configs/tg_interaction_fixed100.json：A基线、三训练种子、fixed100、PPO1×2、时钟disabled。

前缀保存不执行前向/梯度更新，不消耗Torch RNG；它是参数检查快照，不提供优化器中途恢复。
函数3→模块2→真实联调2共7项通过，包含拒绝错误参考/覆盖、RNG与参数不变、plan模式不启动训练、原公开入口与适配器的最终权重/物理轨迹一致。
前一轮139项未全重跑，因为训练实现未改；最终再核对真实100回合任务与20回合前缀。

## 4. 当前运行位置

套件：results/tg_interaction_fixed100/20261104_132048_759710。
manifest.json记录6个训练/冻结评估任务；job_logs保留每次实际命令输出；各训练run的prefix目录保留prefix_at_20.pt/.json。
种子11已通过实际第20回合参数检查，后续100回合训练继续；种子22/33及最终统计待完成。
旧20回合三个冻结动作重放已与原评估完全一致，另记录WAIT、Sink选择、概率和包环路，位于套件diagnostics/fixed20_seed11/22/33。

## 5. 命令与接续

每块一条完整PowerShell命令，在仓库根目录执行。已有训练正在运行时不要再次启动同一套件。

```powershell
& "D:\python3.6.8\python.exe" -m unittest tests.test_interaction_budget.BudgetFunctions tests.test_interaction_budget.BudgetModules tests.test_interaction_budget.BudgetIntegration -v
```

只看预算和旧参考：

```powershell
& "D:\python3.6.8\python.exe" scripts/run_tg_interaction_budget.py --config configs/tg_interaction_fixed100.json --mode plan
```

任务进程已结束后接续未完成任务（任务锁会防止重复运行）：

```powershell
& "D:\python3.6.8\python.exe" scripts/run_tg_interaction_budget.py --resume results/tg_interaction_fixed100/20261104_132048_759710 --mode run
```

单个训练未完成时仍需从该次任务开头重新训练；已完成任务不会重跑。不要把新100fixed与上一轮20linear混为只增加交互量的对照。

后续待完成：三个100回合模型、三次原greedy评估、全部20回合前缀/奖励/固定LR/真实Adam计数验收，以及同口径的动作/环路比较。当前不宣称收敛、显著性或WM/PM收益，未运行正式1000回合或预留盲测。
