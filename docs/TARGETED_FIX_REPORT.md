# TG-MAPPO 定点修复记录

本页为 2026-10-20 历史修复记录。后续旧 TG 预算与绘图修复见
[稳定性记录](STABILITY_OPTIMIZATION_LOG.md)，研究侧 PPO 与策略权重评估实现见
[C2 优化记录](RESEARCH_POLICY_OPTIMIZATION.md)。下文“仍未实现”描述保留当日状态。

日期：2026-10-20。依据：`WORLD_MODEL_PROGRESS_REWARD_PLAN.md` 及
`results/acceptance/20261020_194430_744262/ACCEPTANCE_REPORT.md`。

**结论：本轮针对旧 TG-MAPPO 路径的可复现运行和语义问题完成修复，回归及真实 CPU 短训练通过。方案整体仍未通过 P0～P4 验收。**

## 修复项与原因

| 问题 | 原因 | 修复位置及行为 |
|---|---|---|
| 图距离错误，移动后图不变 | 距离向量存放节点编号 `[ia, ib]` | `environment.py:graph_observation` 使用真实坐标差；节点特征为 x、y、队列比例、实体类别；GNN 最后一列保持合法类别索引 |
| 通信半径不一致 | 邻接、动作和 GNN 使用不同常量或严格小于边界 | `main.py` 将图半径与 `comm_range` 对齐，默认 1.5；`buoy.py`、`environment.py`、`gnn.py` 统一物理半径并包含边界 |
| 丢包配置实际失效 | 初始化、回合重置和信道轮换后又把丢包率写成 0 | `buoy.py` 删除覆盖；`--random_channel true/false` 独立控制随机信道；按实际调度表长度循环 |
| 链路成功率索引错误、拓扑更新丢失统计 | 将邻居列表局部索引用于全局数组；刷新时重新初始化 | `environment.py:buoy_go` 按接收节点全局 ID 更新；`buoy.py:update_topology` 按发送/接收节点保留已有 ACK 统计，新链路先验为 0.5 |
| 采样动作与执行动作不一致 | Sink 动作采样后覆盖、邻居取模映射，采样不传掩码 | `environment.py`、`agent.py`、`runner.py` 使用 N+1 全局动作及 WAIT；采样和 PPO 重算使用对应时刻的合法掩码；执行前检查，执行时不再改写动作 |
| 新回合使用旧观测和隐状态 | warmup 没调用，buffer 没按任务重置 | `runner.py` 每回合生成包后 warmup；`agent.py`、`graph_buffer.py` 写真实初始观测并清空隐状态、计数和历史标签 |
| 一个包在一个时隙连续转发 | 后处理节点可以立即消费刚到达的包 | `runner.py` 在联合时隙开始冻结每个发送节点的 FIFO 服务包 ID，每节点最多发送一个原有包 |
| 新到达包插到老包前面 | 按全局包编号列表选包，忽略当前节点的实际入队时间 | `buoy.py:data_packet` 增加入队顺序；`runner.py` 成功到达后排入队尾，下一时隙按该顺序选包 |
| t+1 图观测混杂多个执行时刻 | 逐节点执行时立即拼接观测 | `runner.py` 在全部动作及移动完成后统一生成观测和下一时刻掩码 |
| T+1 次转移覆盖 buffer | 使用大于 T 才结束，buffer 索引循环复用 | `runner.py` 精确执行 T 次；`graph_buffer.py` 增加容量与掩码形状检查；本旧任务以 T 为真实截止，最后一项终止 mask 为 0 |
| 时钟按节点数倍增 | `step_one_agent` 每处理一个节点就加一步 | `environment.py` 的单节点观测不推进时间；`runner.py` 每个联合时隙结束推进一次 |
| WAIT 分支运行报错 | 未给 `reward` 赋值就累计 | `runner.py` 初始化 WAIT 奖励为 0 |
| 全 WAIT 时 PPO 产生 NaN | 对空活跃集合取均值，损失/熵除以 0 | `graph_mappo.py` 对空活跃集合使用零优势；策略损失、价值损失和 `utils/act.py` 熵分母设置下界 1 |
| 研究参数被旧入口静默忽略 | 多层 `parse_known_args` 后没有严格解析 | `src/legacy_cli.py` 在旧入口拒绝研究参数，提示 `run_research.py`；`main.py` 最终严格解析未知参数，不再静默启动旧训练 |
| Windows 临时输出无法清理 | Runner 的文件日志句柄未关闭 | `runner.py:run` 用 finally 关闭当前实例的指标文件与日志处理器 |

运行目录新增 `logs/TG_MAPPO_environment.json`，记录实际配置、
`implementation=legacy_fixed_v1`、节点特征名及 `action_schema=global_node_wait_v1`。
节点输入和动作输出维度已变化，旧权重不能直接复用，需要重新训练。

## 实际验证

定点测试使用 `D:\python3.6.8\python.exe`，其已安装旧 TG-MAPPO 所需依赖。
按函数、模块、联调顺序执行：6 + 7 + 3 = **16 项全部通过，无跳过**。

- 函数：真实距离、物理特征、半径边界、丢包调度、关闭随机信道、拓扑统计保留。
- 模块：全局动作掩码、warmup、buffer 溢出与旧形状拒绝、WAIT、精确 T 次转移、联合计时、单包单时隙最多一跳、按入队顺序服务。
- 联调：研究参数提前拒绝、真实两回合 PPO、全网隔离的 WAIT 训练。
- 真实 PPO 联调逐时隙核对采样动作、buffer 动作和 old log probability；训练后 Actor、Critic 参数均发生变化，损失与参数均为有限数。
- 最终代码实际启动旧训练入口，25 个移动节点配置、随机丢包开启、2 回合 × 80 时隙、每回合 1 次 PPO epoch / 2 个 minibatch，退出码 0，生成两行指标和曲线；耗时见原始报告。
- 研究框架原有 **74 项回归全部通过**：函数 31、模块 34、联调 9。

证据：

- [16 项及真实训练报告](../results/targeted_fixes/20261020_202147_622246/test_report.json)
- [函数日志](../results/targeted_fixes/20261020_202147_622246/functions.log)
- [模块日志](../results/targeted_fixes/20261020_202147_622246/modules.log)
- [真实 PPO 联调日志](../results/targeted_fixes/20261020_202147_622246/integration.log)
- [80 时隙训练输出](../results/targeted_fixes/20261020_202147_622246/smoke.stdout.log)
- [74 项研究回归](../results/targeted_fixes/research_regression/20261020_202242_476221/test_report.json)

复测命令，均在项目根目录执行：

```powershell
& D:\python3.6.8\python.exe scripts/test_legacy_fixes.py
& D:\python3.10.9\python.exe scripts/test_research.py
```

第一条会按三阶段运行定点测试，随后启动真实 2×80 时隙训练，保存 JSON 报告和原始输出。
可加 `--skip-smoke` 仅运行 16 项测试。缺旧依赖时会明确失败，不会以跳过冒充通过。

## 验收边界及未完成事项

- 本轮修复旧训练路径，没有将 TG-MAPPO 接入 `research_v1`；其全局观测、旧奖励和旧指标口径仍需与研究任务统一。
- 旧 `average_hops` 仍包含未投递包的时限惩罚和路线节点数，不能当作方案要求的“成功包真实跳数”；旧投递统计也仍依赖原有回溯结算。参考研究环境的独立指标实现不受本轮影响。
- `num_moving_buoys=25` 表示选中 25 个节点调用旧移动函数；旧函数仍固定 Sink 及 Sink 邻居，因此不表示 25 个节点都发生位移。
- 旧 `num_env_steps` 仍不是循环预算，实际训练预算由 `num_episodes × episode_length` 决定；本次两者显式匹配。T 是旧任务截止，不能据此声称实现了长任务跨 rollout 的 PPO 学习。
- 正式 WM、PM、研究环境 Graph-MAPPO trainer、完整学习权重保存/恢复及 A～E 实验仍未实现，原验收报告的对应结论保持有效。
- 本次验证为 CPU。当前两个解释器的 PyTorch 均不支持 CUDA，未验证 GPU 显存与性能，也未将短训练投递率解释为学习效果。

后续应先统一研究环境中的基础 Graph-MAPPO 训练，再分别实现 WM、PM 及其冻结接入，继续按原方案验收。
