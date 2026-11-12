# 论文实验执行清单与接续记录

日期：2026-11-12。用户授权按实验清单顺序执行；采用原TG环境，PPO/WM/PM/新增学习基线在GPU，仿真在CPU。

## 当前执行

2026-11-12更新：主队列已全部完成。按用户要求，Greedy已从后续默认队列和主展示图表移除。当前展示目录：`results/thesis_experiments/20261112_011930_763352/reports/without_greedy_20261112_104018`。9种方法的数据、种子和均值保持原样，3项报告测试通过，原完整结果与模型保留。该展示仅代表所列算法，排除Greedy不构成对它的超越证据。以下执行过程保留为历史记录。

当前目录：`results/thesis_experiments/20261112_011930_763352`。首次目录`results/thesis_experiments/20261112_011122_994968`因检查点观察器漏建logs目录而在TG e0停止，保留失败日志；已通过的5种基线短测按代码/结果指纹复用，未重复训练。此前单独MAPPO短测在`results/thesis_preflight/20261112_010853_585258`。

第二个目录`results/thesis_experiments/20261112_011640_791945`完成TG短训后，控制器将历史结果路径拼到source_snapshot/results，验收读取失败。新增独立于cwd的路径回归测试；在第三个目录准备时，直接核对该TG权重和物理日志与旧参考精确一致后复用，没有重训TG。当前后台控制器PID73840。10组短验证现已全部通过，TG/Immediate/PM均精确复现旧参考，主队列已经启动，任一失败立即停止。工程工具提交90240aa5已推送。

按函数、模块、真实短训练/保存/冻结评估顺序验收，再放行300回合。`manifest.json`是任务状态事实来源，`jobs/*/execution.log`与`progress.json`记录具体任务。失败立即停止，不静默重训；完成的历史300回合模型通过哈希和兼容性检查后复用。

当前选用WM仍为shared融合、residual_scaled输入、原训练预测标准差校准、温度1.0；不更改用户选定版本。此前raw WM与raw PM+WM结果保留为历史，不将其冒充选定WM的四组消融。

## 先修正基线可比性

旧`baselines/main.py`运行`src.environment`和`src.runner`，与TG主线的`src.buoy_network.environment/runner`不同。旧Runner会强制选择可达Sink，并使用旧奖励与另一种动作映射，不能直接与当前TG曲线比较。

旧`baselines/agents/agentQmix.py`每个节点各自建立`n_agents=1`的mixing network，且hypernetwork权重未非负约束，不能作为标准合作式QMIX证据。旧文件和历史结果完整保留。

新增`src/buoy_network/thesis_baselines.py`仅实现决策/学习适配器，由`scripts/thesis_experiment_worker.py`在独立子进程接入原Runner，不另建环境、不修改原TG代码。

- MAPPO：参数共享MLP Actor，joint observations集中式团队Critic，fresh rollout、terminal GAE、clipped PPO。
- IPPO：参数共享MLP Actor，每个节点按自身观测编码估值，不拼接其他节点的观测给Critic。任务原本提供global graph，故这不是局部传感设定的IPPO。
- QMIX：25节点联合回放、共享个体Q、集中状态条件mixer、非负混合权重、同步target networks、合法动作bootstrap。
- IQL：共享参数的独立Q学习，无联合mixer；是独立DQN适配，不另把相同实现重复标为第二种DQN算法。
- Greedy：合法邻居中选距Sink最近者，无训练、无强制额外动作特权；遇空队列或无邻居按同一mask执行WAIT。

MLP输入为原可观测的本节点ACK历史、所有节点位置/队列/类型、到各节点的距离及节点ID。位置/距离只作固定物理缩放，没有读取真实丢包率、未来状态或包的剩余真实路程。信息范围与原global TG匹配，表示结构和参数量不同，需在论文记录。

新增基线是本项目的feedforward实现，还没有独立超参数搜索；第一轮300回合属于统一交互预算的开发对比，不据此宣称所有基线已达到最优性能。基线参数量、PPO实际更新数、Q更新数、推理/更新时间均记录。

代价字段限制：当前baseline报告的trainable_parameters是适配器分配的网络总参数，Q算法含未用的占位Critic，Greedy也保留了适配器网络。后续开销分析必须分列实际参与学习的参数与适配器开销，不能把这个原始字段直接当作算法净参数量。Greedy没有训练种子意义，主队列三个规则任务是在相同场景上的重复核验，不应声称它们是三次独立训练；论文图表需明确此例外。

## 顺序与预算

| 顺序 | 内容 | 本轮安排 |
|---|---|---|
| 0 | 函数→模块→联调 | 当前11项单测通过，含目录回归、统计与出图；10组各2回合短训及e0/e2冻结评估；TG/Immediate/PM短前缀需精确复现旧结果 |
| 1 | 算法对比 | Greedy、IPPO、QMIX、MAPPO、IQL与TG/最终组合比较 |
| 2 | 消融、即时奖励对照 | 复用TG/Immediate/PM/选定WM各三种子300模型；补选定WM+PM三模型 |
| 3 | 收敛速度 | 学习方法300×80联合时隙，每20回合，三个开发场景各2回合冻结评估；含e0，共16点 |
| 4 | 未见场景、丢包/移动/负载 | 主实验之后，先验证统一干预接口再执行；尚未启动 |
| 5 | WM/PM模块能力 | 主实验之后审计选定模型的数据划分、预测误差和删失标签；尚未完成本轮验证 |
| 6 | 参数敏感性 | 主实验之后预先固定取值和预算；尚未启动，不按测试集挑最佳参数 |
| 7 | 开销与失败案例 | 所有任务持续收集原物理事件与计时；主实验后汇总，完整分析尚未完成 |
| 8 | 扩展文献基线/更大规模 | 可选，主线之后；尚未启动 |

训练种子11/22/33，开发评估场景8001/8002/8003。它们已用于前期调试，不能称为未见测试。3041–3045等此前封存训练种子本轮不启用。主学习预算为每模型24000环境交互步；PPO为1 epoch×2 minibatch/回合，Q算法每回合2个32联合样本更新，buffer=3000时隙、64时隙warmup、100更新同步target，epsilon从1降到0.05。不同算法保留各自更新机制。

新增正式学习任务：IPPO/QMIX/MAPPO/IQL/选定PM+WM各3模型，共15×300=4500回合、360000训练时隙。另有3个Greedy规则任务，仅做规则评估，不把其内部1回合采样记作学习训练。已有12个300回合模型复用，不重复训练。短验证、冻结评估和历史WM/PM预训练成本单列。

## 输出与判读

主队列结束生成`per_seed.csv`、`learning_curves.csv`、`comparison/convergence.png/svg/pdf`、`report.json`。报告的`all_thesis_experiments_completed`在后续阶段未完成时保持false；`main_completed`只表示主对比/消融队列完成。

比较最终固定300回合检查点，不按评估结果挑最好一轮。主指标投递率，同时保留成功包跳数/时延、未完成比例、ACK失败、环路；没有成功包时的时延记null，不记0。总奖励因PM塑形而不同，不混作任务收益。图为训练种子均值与样本SD，保留各seed散点，开发均值较高不等于未见场景泛化已通过。

短测已发现Greedy在一个场景达到100%投递；这只是初步结果，必须保留并在多场景比较后解释。如果最终学习方法低于规则基线，仍如实报告，不通过删除规则、更换测试口径或降低基线预算制造优势。

## 命令与接续

```powershell
# 只读查看状态，不启动任务
Get-Content results/thesis_experiments/20261112_011930_763352/manifest.json -Raw

# 验证历史模型（只读历史结果，审计报告写入本轮目录）
& "D:\anaconda3\envs\buoy_gpu\python.exe" -s scripts/audit_thesis_references.py results/thesis_experiments/20261112_011930_763352

# 仅当manifest为preflight_passed且没有控制器/训练进程时，接着执行主队列
& "D:\anaconda3\envs\buoy_gpu\python.exe" -s results/thesis_experiments/20261112_011930_763352/source_snapshot/scripts/run_thesis_experiments.py --run results/thesis_experiments/20261112_011930_763352
```

若任务failed，先读对应日志、解释原因、定点修复并重测；不自行覆盖原输出目录。当前工具没有优化器/RNG恢复，不能把训练到一半的policy权重当作可精确续训检查点。


## 2026-11-12：补充经典VDN、COMA

新增目录：`results/thesis_extra_baselines/20261112_104503_699371`，后台控制器PID10652。先VDN/COMA各seed22两回合、e0/e2场景8001冻结评估，全部通过再顺序跑VDN三种子300、COMA三种子300；每20回合在8001/8002/8003各冻结2回合。新增1800训练回合、144000训练时隙、46080正式冻结评估时隙；预检单列。GPU模型计算、CPU原仿真不变。

15项函数/模块/报告测试通过；历史27个学习模型、432个检查点哈希审计通过，全部复用。不是重跑原9组，也不重新启用Greedy。结束自动合并为11组CSV/对比图/收敛图。

- VDN依据https://arxiv.org/abs/1706.05296 ：所有活跃个体Q相加学习团队TD误差，无可学习mixer；与QMIX使用相同联合回放、target同步、epsilon与预算。
- COMA依据https://arxiv.org/abs/1705.08926 ：Critic输入集中观测、本节点编码和其他节点动作，显式屏蔽自身已选动作，输出所有自身候选动作Q。Actor优势为已选动作Q减策略概率加权Q基线，baseline与Q均detach；fresh rollout，TD(lambda=.95)有限任务终止目标，target Critic每20回合同步。每回合2个Critic minibatch、1次Actor更新，没有冒充PPO。

二者为feedforward/参数共享/原global观测适配，非原论文任务复现或独立调参最优版本。新测试覆盖VDN求和及各节点梯度、COMA自身动作不泄漏/其他动作依赖、反事实基线手算及零期望、TD(lambda)终端与bootstrap手算；真实联调和长期效果以manifest为准。

源代码入口：`scripts/run_thesis_extra_baselines.py`。接续只读manifest/progress，running时不得重复启动；任一失败队列立即停止，保留具体日志。

最新状态：VDN与COMA两组真实短训练/保存/冻结评估均已通过，后台已进入main_VDN_11的300回合任务；当前尚无新基线长期结果。机器凭据：`docs/evidence/TG_THESIS_EXTRA_BASELINES_20261112.json`。
