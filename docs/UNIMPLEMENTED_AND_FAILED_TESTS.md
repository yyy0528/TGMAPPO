# 未实现与失败测试清单

## 当前状态：2026-10-21

已完成基础研究 Graph-MAPPO/PPO、真实策略权重保存及独立冻结评估，F03/F04/F10 的原缺项已解决。
训练需显式选择 `--policy graph_mappo` 或使用 `configs/research_graph_mappo.json`；评估必须提供真实兼容 checkpoint。
原先的缺失文件占位命令仍应报错，不能用其退出码判断新功能失败。
实现、逐步复测和效果限制见 [C2 优化记录](RESEARCH_POLICY_OPTIMIZATION.md)。

剩余能力失败 **8 项**，已重新实际执行：F01/F02/F05/F06（WM/PM 未注册）、F07（EARP）、
F08（CPU PyTorch 不支持 CUDA）、F09（无完整 resume）、F11（当前 Python 3.10 缺旧路径依赖）。
命令与错误证据见 [remaining_failures.json](../results/optimization/20261021_140749_977167/final_acceptance/remaining_failures.json)。
基础策略虽可训练，短训评估投递率约 4.17%，尚未通过效果验收。

此前已修复单回合绘图、CSV 校验及旧 TG 预算，见 [小步优化记录](STABILITY_OPTIMIZATION_LOG.md)。
`requirements.txt` 已补 tensorboardX/imageio 声明；完整依赖锁和新机器安装验证仍缺失。
以下第 1、2、4 节保留 **2026-10-20 历史证据**，不将已解决项计入当前失败数。

历史验收日期：2026-10-20。**当日只验收和记录，不修改代码。**
原始命令、退出码、stdout、stderr 均位于
[最终执行索引](../results/final_acceptance/20261020_210112_027890/execution_index.json)对应的同目录 JSON。

## 1. 历史失败的能力验收：11 项

以下命令省略公共前缀 `D:\python3.10.9\python.exe run_research.py`。
WM/PM 使用本次真实采集且具有三组拆分的数据目录，失败发生在 trainer 解析阶段。
返回 2 是明确报错；负向单元测试可以因此通过，但“功能可运行”的验收仍失败。

| 编号/证据文件 | 实测调用 | 退出码 | 原因及位置 | 建议修改；本次未执行 |
|---|---|---:|---|---|
| F01 `train_wm.json` | `train-wm --dataset <本次data>` | 2 | `trainer:supervised_wm` 未注册；`research/bootstrap.py`、`training/` | 实现解耦 GNN/GRU WM、目标编码/损失、监督 trainer、验证与校准 |
| F02 `train_pm.json` | `train-pm --dataset <本次data>` | 2 | `trainer:supervised_pm` 未注册 | 实现前缀 GRU、成功失败分层、删失处理和监督 trainer |
| F03 `train_policy.json` | `train-policy` | 2 | `trainer:graph_mappo` 未注册 | 在研究环境接入真实 PPO、价值回归和缓存特征 |
| F04 `graph_mappo_backend.json` | `smoke --policy graph_mappo` | 2 | 只有 geographic/random 策略 | 实现 RoutingPolicy adapter 及网络采样/评估接口 |
| F05 `world_model_backend.json` | `smoke --use-world-model true --wm-checkpoint not_loaded.pt` | 2 | `world_model:local_graph_gru` 未注册，先于文件读取报错 | 实现并注册正式 WM，再验证真实权重加载 |
| F06 `progress_model_backend.json` | `smoke --use-progress-reward true --pm-checkpoint not_loaded.pt` | 2 | `progress_model:prefix_gru` 未注册 | 实现并注册正式 PM，再验证真实权重加载 |
| F07 `earp_backend.json` | `smoke --use-earp true` | 2 | reference_grid 不支持 EARP | 实现研究环境可开关的 EARP，并独立记录回报 |
| F08 `cuda_backend.json` | `check-config --device cuda` | 2 | 本解释器是 CPU PyTorch；`runtime/device.py` 正确拒绝 | 建立独立兼容 CUDA/PyG 环境，再测三模块前后向和峰值显存 |
| F09 `resume_cli.json` | `train-policy --resume not_loaded.pt` | 2 | CLI 不识别 `--resume`，也无完整恢复 payload | 保存并恢复网络、优化器、归一化器、RNG、阶段和版本 |
| F10 `policy_checkpoint_cli.json` | `evaluate --policy-checkpoint not_loaded.pt` | 2 | CLI 不识别学习策略 checkpoint 参数 | 增加学习策略加载、冻结评估及兼容检查 |
| F11 `legacy_current_python.json` | `run_algorithm.py --algorithm tg-mappo ...`，同一 3.10 解释器 | 1 | `ModuleNotFoundError: gym`；PyG、wandb、tensorboardX 也缺失 | 使用已验证旧解释器复现；另行建设研究依赖环境 |

F05/F06 的路径仅用于确认组件缺失的报错顺序，不是交付的权重文件。
F09/F10 在当日首先受参数及实现缺失阻断；F10 已有本轮真实加载证据，F09 完整恢复仍未解决。

## 2. 历史通过与失败的区分

- 74 项研究回归 + 16 项定点回归 = **90/90 通过**，失败 0、错误 0、跳过 0。
- `legacy_research_guard.json` 虽退出 2，但预期就是拒绝研究参数，属于通过的负向检查，未计入上述 11 项失败。
- 本次八算法短启动、两回合 CSV 和三张图通过；短基线运行未证明每种基线都执行了反向更新或已收敛。
- 上一轮修复前的距离、丢包、时钟、FIFO、WAIT、NaN 等失败已通过定点复测，列在 [修复记录](TARGETED_FIX_REPORT.md)，不混入当前未解决失败数。

## 3. 仍未实现或未验收

| 优先级 | 缺项 | 当前已有 | 仍需完成 |
|---|---|---|---|
| 效果待验收 | 研究基础 Graph-MAPPO | 局部图策略、真实 PPO、权重冻结评估已实现 | 长期训练/收敛、效果基准及同条件 A 组正式实验 |
| 阻断 | 一步 WM | 抽象协议、候选特征契约、原始数据 | 动作条件/拓扑业务解耦、WAIT 掩码、归一化、训练、预测基线、Brier/NLL 校准 |
| 阻断 | PM | 抽象协议、包轨迹、势函数数学 | 结果监督网络、前缀采样、成功失败分层、删失、冷启动及验证 |
| 阻断 | 冻结模型接入 PPO | 特征缓存和版本边界基础设施 | 正式网络的 collect/buffer/minibatch/evaluate_actions 贯通及可靠性门控 |
| 阻断 | 完整保存恢复 | JSON 元信息、真实策略 `.pt` 权重和独立冻结评估 | 优化器、归一化器、RNG、训练阶段的保存恢复及恢复一致性验证 |
| 实验前 | 统一环境与 EARP | reference_grid 与旧 TG 分开运行 | 统一任务/观测/信道/奖励，独立 EARP 开关；不可跨环境直接比较 |
| 实验前 | 数据预算 | 完整回合导出、按回合/种子拆分、replay 类 | WM/PM 分离采样、全局状态去重、容量接入采集、磁盘和包轨迹上限 |
| 实验前 | 标准实验指标 | 参考环境成功包跳数/时延/P95及分项回报 | 重传/环路、样本效率、模型指标、学习曲线及预训练成本 |
| 实验前 | GPU 实测 | 已识别 4060 8188 MiB | 完整前后向、实际参数量、allocated/reserved、各阶段耗时 |
| 实验前 | A～E 与多种子 | 三种子规则运行和通用配置 | 同环境同预算的学习实验、至少三开发种子及离散度 |
| 交付限制 | 可重建依赖 | requirements 已补两项直接依赖，本机版本快照和旧环境导入验证 | 完整锁定兼容版本并在新环境安装验证 |
| 可后置 | P5 多步想象 | 设计描述 | 多步误差、行为分布、可信推演验证；不计首版完成要求 |

## 4. 历史发现但当日未修复的边界问题

1. 单回合 CSV 绘图只有图例、无可见数据点：`plot_comparison.py` 的 `plt.plot` 没有 marker。建议为单点设置 marker，或对不足两个点给出说明。本交付示例使用真实两回合数据生成，未改绘图代码。
2. 旧 `average_hops` 是含未投递惩罚的历史量，投递统计依赖回溯结算；不能改名解释成真实成功跳数。建议研究实验使用独立标准字段。
3. 旧 `num_env_steps` 不控制实际循环；仅改它不能改变旧训练预算。建议统一预算或明确拒绝冲突配置。
4. 旧移动函数固定 Sink 和 Sink 邻居，且保留原幅度规则；所选移动数量不等于实际发生位移数量。建议记录实际移动统计或统一研究移动规则。
5. `requirements.txt` 未锁版本且漏 `tensorboardX`、`imageio`；当前 3.10 无法运行旧 TG，新机器安装未验证。

以上为当日建议；第 1、3 项及第 5 项中的直接依赖声明已在 2026-10-21 修复。
完整研究验收仍须按方案逐项补测。
