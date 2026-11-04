# PM价值估计与GAE：一致性检查和安全对齐模式

日期：2026-11-04。主线仍为原TG-MAPPO与浮标环境，Python3.6.8 / Torch1.8.1 CPU。
**已实现并验收base_residual模式：消除固定PM势差对基线PPO更新的干扰；这是安全/一致性对照，不是PM创新收益。旧learned_shaped默认保留用于兼容，PM仍默认关闭。**

## 1. 检查结论：不能把旧GAE直接说成写错

旧奖励为r'=r+alpha*(gamma*Phi_next-Phi)，旧Critic直接学习对应的塑形价值V'，GAE/价值损失/PopArt目标也使用塑形回报。这种实现本身合法；没有发现漏更新、GAE公式反号或已证实的归一化混用bug。
问题在于有限预算的估计负担：同样初始化的神经网络，在PM开启后要额外拟合势函数造成的偏移；Phi来自包GRU历史，原非循环Critic不能直接看到全部PM历史状态。它可能形成不同价值误差与局部优势，尚不能认定这是100回合退化的唯一原因。
上一轮完成信用候选的效果失败保留，未启用它来掩盖本轮问题。

## 2. 正确的坐标关系

固定势函数、相同gamma、有限任务末端Phi_H=0时：

```text
V'_PM = U_base - alpha * Phi
delta_PM = r' + gamma * V'_next * mask_next - V'
         = r  + gamma * U_next * mask_next - U
```

因此在**同一条缓存轨迹、同一组基础价值U**上，正确平移后的每步TD、任意lambda的GAE均相同；不只整段Monte Carlo回报相差常数。
旧模式的网络数字不能直接断言是U：它原本被训练成V'。本诊断把同一组数值作为反事实对照，不把两种含义混为同一个真值。

## 3. 新模式如何实现

显式参数：`--pm_value_mode base_residual`。旧模式为`learned_shaped`。

- Critic输出U_base，采样value_preds、PPO旧值裁剪、训练return、PopArt/ValueNorm统计和价值损失都保持基础回报坐标。
- 有效塑形价值按U_base-alpha*Phi解释；Phi只在原尺度处理，绝不直接从归一化的网络输出里减去概率。
- 为避免float32势差相加再相消造成参数漂移，实际实现直接用缓存r_env计算基础坐标GAE。这与r'配合U_base-alpha*Phi解析计算等价。
- buffer.rewards继续保存原始采集的r_train，PM奖励/势函数CSV不改写。GraphReplayBuffer.compute_returns新增只读rewards_override接口，不把已采集奖励重标成基础奖励。
- 报告明确记录`gae_uses_shaped_rewards=false`、value_mode和原尺度TD/GAE恒等式误差，避免声称PM提供了额外学习收益。
- PM仍冻结计算和审计；Actor输入/结构、动作掩码、基础奖励、gamma、lambda、PPO预算均不变。WM可组合。
- 新旧价值模式checkpoint互相拒绝，旧缺字段模型视为learned_shaped。不能把旧塑形Critic直接按基础价值加载。

当前只接受feedforward、整段有限任务GAE、连续势缓存、末端归零、无中途终止/截断、bad_masks全1；不支持的情况拒绝，不自动套用。

**训练更新效果等同原TG是这个模式的预期行为。它可规避PM引入的退化，不能被称为“PM提高了TG”。** 如果需要证明PM能加速学习，须另行检验与真实任务回报匹配的价值先验/可靠性设计。
如果只需要当前可用策略，直接使用关闭PM的TG及已保存A权重即可；无需为了相同更新付出额外PM前向成本。base_residual主要用于诊断、回归和验证坐标一致性。

## 4. 同轨迹真实检查

三种子各重建baseline、旧PM、新模式的两回合训练；旧A/旧C参数、物理JSONL和奖励CSV严格复现原fixed2。回放观测器只在compute完成后读取缓存，不额外前向或触碰训练分布缓存。
三组首回合r_env、r_train、神经网络原尺度值、Phi、masks逐元素完全一致；这是同一缓存轨迹上的对照。

| 种子 | 旧与新实际GAE最大差 | 旧有效值−塑形MC均值 | 对齐后有效值−同一MC均值 |
|---|---:|---:|---:|
| 11 | 0.585019 | +0.505318 | +0.301276 |
| 22 | 0.547507 | +0.981038 | +0.767365 |
| 33 | 0.536075 | +1.080099 | +0.852833 |

偏差取活跃决策，使用同一首回合采样MC，不是精确条件V，不构成显著性/最终校准结论。
正确平移后TD/GAE恒等式误差小于5e-16；实际float32 buffer相对双精度独立重算的GAE最大误差约2.51e-7，无数值发散。

## 5. 实际PPO和回归结果

新增函数3→模块2→真实PPO联调2，共7项通过；覆盖lambda=0/.95/1、alpha=0/.1/1、缓存错位拒绝、原尺度PopArt、缓存不变、普通价值/PopArt/独立ValueNorm、WM组合和checkpoint保护。
原test_legacy*.py共105项通过，上一轮完成信用5项也通过，共117项唯一测试，最终失败/错误/跳过0；其余研究/实验编排测试未全重跑。
首次测试启动因新测试sys.path漏项目根目录而失败，测试运行0项；已先解释原因、保留源码/记录、补路径后重跑，不把启动失败写成通过或跳过。记录：results/tg_pm_value_tests/startup_failure。

原环境短验证12个任务全部成功，各训练2×80、每网络4个实际Adam步，场景7001冻结评估2×80。

| 种子 | TG基线投递/48 | 新价值对齐模式投递/48 | 校验 |
|---|---:|---:|---|
| 11 | 8 | 8 | Actor/Critic及冻结物理轨迹完全一致 |
| 22 | 0 | 0 | Actor/Critic及冻结物理轨迹完全一致 |
| 33 | 4 | 4 | Actor/Critic及冻结物理轨迹完全一致 |

这里仍有零投递，不能写成任务效果全部修好。旧PM相同短预算为12/2/4；本模式不是按旧PM短成绩挑选的更优模型，而是恢复基线的确定性控制。
原100回合A=32/23/18，C=36/14/16。本轮**没有重跑100回合，也不能声称实际恢复了50.69%的100回合投递率**。
实际新增1440训练/480正式冻结评估时隙；单元/集成测试的微型运行不充当独立训练种子，CPU并行测试时长不作速度基准。

## 6. 交付与可复制命令

实现：src/buoy_network/pm_value.py、agent.py、progress_runtime.py、policy_artifacts.py及graph_buffer.py；CLI main.py/legacy_cli.py。
测试目录：results/tg_pm_value_tests/20261104_214741_986527。
原环境/缓存目录：results/tg_pm_value_probe/20261104_213758_403702；value_audit保留每回合NPZ和summary.jsonl。
[机器凭据](evidence/TG_PM_VALUE_ALIGNMENT_20261104.json)含源快照、12条实际命令/退出码、测试原输出、相同轨迹/参数/物理证据、首轮启动失败和边界说明。

重跑分层测试：

```powershell
& "D:\python3.6.8\python.exe" scripts/test_tg_pm_value.py
```

已有短验证已完成，只有确需新复现时才执行以下命令，会创建新目录：

```powershell
& "D:\python3.6.8\python.exe" scripts/check_tg_pm_value.py
```

使用安全对齐模式的新训练示例（新增实验，不加载旧PM策略权重）：

```powershell
& "D:\python3.6.8\python.exe" run_algorithm.py --algorithm tg-mappo --use_pm_reward true --pm_reward_checkpoint results/tg_pm_final_pretrain/20261103_142745_276042/pm_best.pt --pm_value_mode base_residual --pm_potential_mode remaining --tg_reward_version task_v2 --tg_critic_time_mode disabled --seed 22 --scenario_seed 22 --num_episodes 2 --episode_length 80 --num_env_steps 160 --ppo_epoch 1 --num_mini_batch 2 --num_moving_buoys 5 --output_root results/tg_pm_value_safe
```

冻结评估必须带同样的PM/value_mode参数，否则checkpoint保护会拒绝。不要直接接续旧fixed100套件，它按旧源码冻结；原结果和大模型均保留。

## 7. 下一步

价值坐标与GAE一致性修复/对照已通过；PM增加额外收益仍未通过。
如果下一步验长期恢复，应在新目录预定100回合、三种子并严格比较原A，不根据7001反复调参。
如果研究PM的正收益，优先评审“送达概率”与task_v2折扣回报（投递、等待、未完成成本）的尺度/目标关系；新势函数或先验须单独做对照，不能把本模式的基线恢复当创新收益。历史53分保持。
