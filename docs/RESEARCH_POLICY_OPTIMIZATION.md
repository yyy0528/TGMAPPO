# 按评分表持续优化：研究 Graph-MAPPO 基线

日期：2026-10-21。对应 [PROJECT_SCORECARD.md](PROJECT_SCORECARD.md) 的 C2。
**本次从 45 分更新为 53 分，仅增加 C2 的 8 分；完整方案仍未通过验收。**

## 逐步实现及验证

| 步骤 | 改动/原因 | 继续下一步前的测试 |
|---|---|---|
| 1 | 旧 TG 无条件导入 tensorboardX/imageio，但 requirements 未声明；补齐这两项 | 原 Python 3.6 实际导入二者和 buoy_network.agent 成功 |
| 2 | 研究侧只有规则策略；新增纯 Torch 两层消息传递 Actor 及独立集中 Critic | 6 项图策略测试，以及真实参考环境 smoke 通过 |
| 3 | 研究侧没有梯度训练；新增 clipped PPO 和真实网络权重输出 | 最初 5 项 PPO 测试、25 节点 32 时隙/16 次优化器更新通过 |
| 4 | 已保存权重无法独立评估；新增 checkpoint 校验、加载与冻结评估 | 概率/价值一致、版本/维度/NaN/损坏文件拒绝、评估不改权重通过 |
| 5 | 独立复核发现 KL/裁剪统计受非活动节点稀释，二进制截断错误穿透，评估默认值覆盖 JSON 显式设置 | 每项修复后单独复测；最终 PPO 6 项、checkpoint/评估 7 项全部通过 |
| 6 | 有限负数屏蔽会在极低合法 logits 下失效；大共同偏置还使 float32 的 log probability 失精 | 改为负无穷屏蔽并在构造分布前减每行最大值；图策略 7 项通过，直接核对 log probability 与实际概率，再跑完整回归 |
| 7 | 环境清单仍沿用早期“rule policies”限制，标签与学习策略不符 | 去掉过期标签；分别实际运行 geographic 与 graph_mappo smoke，核对两种新产物，再跑完整回归 |

证据目录：[本轮汇总](../results/optimization/20261021_140749_977167/optimization_summary.json)。
每步保留了命令、退出码和原始输出；初始编码检查脚本改为显式 UTF-8 后验证通过。

## 架构与信息范围

- `research/policies/graph.py`：纯 PyTorch，无 PyG/Gym 运行依赖。共享 Actor 对每个节点自己的局部图独立计算 N+1 个动作。
- Actor 编码不可见节点时，输入与带偏置消息传递后均清零；其它节点的私有观测不会进入本节点 logits。
- Critic 使用独立图编码器，聚合所有节点在当前时刻的局部观测；它是联合观测 Critic，不是隐藏全状态 oracle。
- 当前策略为前馈图策略，不包含策略 GRU，也不等同于旧 TG 注意力网络的逐层复现。
- `research/training/ppo.py`：GAE 使用原 rollout 的行为价值和 bootstrap；按时间步取完整 N 节点批次，保留 Critic 上下文。
- Actor loss、entropy、KL/裁剪指标只对实际可决策节点统计；全 WAIT 批次不建立 Actor 反向图，Adam 动量也不会移动 Actor 参数。Critic 仍学习团队回报。
- rollout 内版本固定，更新仅发生在边界；未终止任务的环境和包队列在边界保留，终止 bootstrap 为 0。
- 非法动作使用负无穷严格屏蔽；每行 logits 先减最大值，极端负偏置下采样概率、log probability 与熵仍一致。

## 权重与评估

训练生成 `policy_latest.pt`，包含真实网络 state_dict、环境/观测/动作 metadata、架构版本、图层/隐维度及更新版本。
加载在改动参数前检查键、维度、dtype、有限性和版本；损坏或截断数据转换为明确错误，失败不创建评估输出目录。

`evaluate` 默认使用贪心点质量分布，log probability 为 0，与执行分布一致；配置中的显式 `false` 保留，CLI 可再覆盖。
未指定模式的 JSON 值为 `null`：训练/采集时随机采样，评估时贪心。
评估不创建训练数据归档，不更新网络；本次对 checkpoint 文件做了前后 SHA256 核对。

**这不是完整断点恢复。** 当前未保存/恢复优化器、训练 RNG、归一化器和训练阶段；`--resume` 及训练 warm-start 仍不支持，D2 仍为 0 分。

## 实际运行与效果边界

在 Python 3.10.9、Torch 2.11.0+cpu 上运行：25 节点，5 个移动节点，移动标准差 0.05，独立丢包概率 0.2，稀疏任务奖励。

| 项目 | 结果 |
|---|---|
| 训练任务 | 3 回合 × 80 时隙；rollout=16，epochs=2，minibatch_steps=8 |
| 真实更新 | 240 环境时隙、15 次 rollout 更新、60 次优化器更新 |
| 参数量 | 55,362 |
| Actor 参数变化 L2 | 约 0.40553 |
| Critic 参数变化 L2 | 约 1.15404 |
| 训练内部计时 / 完整命令计时 | 约 10.72 / 14.86 秒；本机单次运行，不作为标准性能基准 |
| 独立评估 | 场景种子 101/102/103，加载同一训练策略，贪心解码 |
| 投递率 | 三个场景均约 4.17% |
| 冻结检查 | 评估前后 checkpoint 哈希一致 |

短训练效果很低，尚不能称为收敛或性能合格。本轮验证的是训练与评估闭环，不能用这组结果声称超过规则、旧 TG 或 EARP。
三种子是评估场景，不是三个独立训练种子，D4 不加分；GPU 显存与完整 CUDA 路径未验证。

实际运行证据：[learned_policy_acceptance.json](../results/optimization/20261021_140749_977167/final_stability_acceptance/learned_policy_acceptance.json)。
该记录由动作屏蔽和分布稳定性修复后的代码重新训练、评估生成。
其中环境清单生成时仍带旧的 `rule policies` 文案；配置、模型版本和真实权重均为图策略。
步骤 7 已更正后续清单并复测两种策略；历史运行证据保留原貌。

## 完整回归

- 研究回归：函数 51、模块 34、进程联调 11，共 **96 项通过**。
- 旧路径/预算/绘图回归：**35 项通过**，另完成旧 TG 2×80 时隙训练。
- 合计 **131 项通过，失败/错误/跳过均为 0**。
- 研究包具名实现函数 **97/97** 被函数层实际调用；不是全仓库行或分支覆盖率。

证据：[96 项研究测试](../results/optimization/20261021_140749_977167/final_metadata_regression/20261021_185243_505770/test_report.json)、
[35 项旧路径测试](../results/optimization/20261021_140749_977167/final_legacy_regression/20261021_183251_601533/test_report.json)。

## 复测和使用

在项目根目录执行：

```powershell
& D:\python3.10.9\python.exe run_research.py train-policy --config configs/research_graph_mappo.json
& D:\python3.10.9\python.exe run_research.py evaluate --config <训练输出目录>/config.json --policy-checkpoint <训练输出目录>/checkpoints/policy_latest.pt --seed 101 --episodes 3 --output-root results/my_graph_eval
& D:\python3.10.9\python.exe scripts/test_research.py
& D:\python3.6.8\python.exe scripts/test_legacy_fixes.py
```

示例环境参数与本次实际运行一致。正式 WM、PM、冻结辅助模型接入、EARP、完整恢复和 A～E 实验仍缺失。
旧 2026-10-20 交付包保留为历史快照；本轮源码和权重在当前工作区，未覆盖旧包。
