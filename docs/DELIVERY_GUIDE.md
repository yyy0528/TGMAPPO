# 运行与交付说明

## 1. 目标与可用范围

项目目标是为浮标网络的下一跳路由增加一步图世界模型（WM）和结果监督进度模型（PM），
冻结辅助模型后用真实仿真轨迹训练 Actor/Critic，并在 RTX 4060 8GB 上验证资源与效果。

当前有三条独立路径：

| 路径 | 入口 | 当前能力 |
|---|---|---|
| 修正后的旧 TG-MAPPO | `run_algorithm.py --algorithm tg-mappo` | 真实 Actor/Critic 更新，`legacy_fixed_v1`；全局观测与旧奖励 |
| 七种旧基线 | `run_algorithm.py --algorithm <name>` | DQN、IQL、MAPPO、QMIX、QMIX-ATTN、VDN、RBMADQ 各自原环境 |
| 研究框架 | `run_research.py` | reference_grid、规则/局部图策略、真实 PPO、策略权重冻结评估、采集归档、GAE 与势函数基础设施 |

研究 Graph-MAPPO 基础训练/评估闭环已实际运行，见 [C2 优化记录](RESEARCH_POLICY_OPTIMIZATION.md)。
正式 WM/PM 网络、冻结组合、EARP、完整训练恢复及效果验收仍未完成。

## 2. 环境与依赖

实际版本来自 [runtime_inventory.json](../results/final_acceptance/20261020_210112_027890/runtime_inventory.json)。

| 项目 | 旧训练环境 | 研究框架验证环境 |
|---|---|---|
| Python | `D:\python3.6.8\python.exe` | `D:\python3.10.9\python.exe` |
| PyTorch | 1.8.1+cpu | 2.11.0+cpu |
| NumPy | 1.19.4 | 2.2.6 |
| Gym / PyG | 0.10.5 / 2.0.3 | 未安装 |
| wandb / tensorboardX | 0.10.31 / 2.1 | 未安装 |
| CUDA 可用 | 否 | 否 |

研究框架要求 Python 3.10+；CPU 规则仿真本体使用标准库，图策略/PPO 和完整研究测试需要 PyTorch，绘图使用 matplotlib。
研究图策略使用纯 Torch，无 PyG/Gym 依赖；当前已验证 Torch 2.11.0+cpu。
旧训练使用已有 Python 3.6 依赖。不要将两套环境当作同一个环境。

README 中 `pip install -r requirements.txt` 是安装入口。2026-10-21 已补齐旧代码直接导入的
`tensorboardX`、`imageio` 声明，并验证原旧解释器导入；当前文件仍没有版本锁定，新机器完整安装未通过验收。
交付的版本清单是本机观测结果，不是已验证的依赖锁文件。

## 3. 启动及复测

所有命令均在项目根目录执行，输出目录采用新的时间戳，不覆盖旧实验。

```powershell
# 列出统一入口支持的算法
& D:\python3.6.8\python.exe run_algorithm.py --list-algorithms

# 修正后的旧 TG-MAPPO 真实短训练
& D:\python3.6.8\python.exe run_algorithm.py --algorithm tg-mappo --num_episodes 2 --episode_length 80 --num_env_steps 160 --ppo_epoch 1 --num_mini_batch 2 --num_moving_buoys 25 --output_root results/my_tg_smoke

# README 中八种算法的启动与输出冒烟；仅使用各算法共同支持的参数
& D:\python3.6.8\python.exe run_algorithm.py --algorithm all --seed 12 --grid_size 5 --num_moving_buoys 5 --num_episodes 2 --episode_length 8 --output_root results/my_all_smoke

# 三阶段测试；第一条还会运行 2×80 时隙真实训练
& D:\python3.6.8\python.exe scripts/test_legacy_fixes.py
& D:\python3.10.9\python.exe scripts/test_research.py

# 研究组件与配置检查
& D:\python3.10.9\python.exe run_research.py describe
& D:\python3.10.9\python.exe run_research.py check-config --config configs/research_v1.json --device cpu

# 三个独立回合的规则数据；可生成 train/validation/test 拆分
& D:\python3.10.9\python.exe run_research.py collect --config configs/research_v1.json --episodes 3 --seed 41 --output-root results/my_research_data
& D:\python3.10.9\python.exe run_research.py evaluate --episodes 3 --seed 41 --output-root results/my_rule_eval

# 研究图策略短训练及独立权重评估
& D:\python3.10.9\python.exe run_research.py train-policy --config configs/research_graph_mappo.json
& D:\python3.10.9\python.exe run_research.py evaluate --config <训练输出目录>/config.json --policy-checkpoint <训练输出目录>/checkpoints/policy_latest.pt --seed 101 --episodes 3 --output-root results/my_graph_eval
```

`run_training.cmd` 仍是旧训练快捷入口，默认运行预算较大；上面的显式短命令适合复验。
`all` 顺序执行八算法，遇非零退出码停止。不要把 `--ppo_epoch` 等仅 TG 支持的参数传给 `all`，基线会严格拒绝。

## 4. 输出及查看

旧统一入口输出：

```text
results/<自选目录>/runs/<run>/moves_<k>/logs/<ALGORITHM>_metrics.csv
episode,delivery_rate,average_hops,average_delay
```

把终端打印的 `Run output:` 绝对路径传给：

```powershell
& D:\python3.10.9\python.exe scripts/plot_comparison.py --output-root <Run-output路径> --num-moving-buoys 5
```

生成投递率、跳数、时延三张 PNG。单回合使用点标记，多回合保留连线。
空 CSV、缺失/非法数值、NaN/无穷值会在生成图片前报错，非法行会标明文件和行号。
短回合图仅验证产物链路，两组环境未统一，不能作为算法优劣或论文结果。

研究路径每次运行包含 `config.json`、`environment_manifest.json`、`research_metrics.csv`、
`run_report.json`；采集还包含 `data/data_manifest.json` 和完整回合转移/包轨迹 JSONL。
评估不归档训练数据。图策略训练另有 `model_metrics.csv`、`checkpoints/policy_latest.pt` 和 metadata；
可加载真实网络权重独立评估，完整训练恢复尚不支持。

## 5. 与旧 README 的重要差异

- 修正后的旧 TG-MAPPO 按联合时隙更新选中节点和拓扑，旧 README 的“每轮结束后移动”不再适用于该路径。
- 旧移动函数固定 Sink 及 Sink 邻居；选择 25 个节点不表示 25 个节点都产生位移。
- 旧 TG 不给 `num_env_steps` 时按 `num_episodes × episode_length` 运行；只给步数时推导回合数。
  同时给步数和回合数必须一致，步数必须为完整回合的整数倍，否则明确报错。
  预算单位为单环境联合时隙；启动输出和 `TG_MAPPO_environment.json` 记录实际预算与来源。
  `all` 仍应使用公共的 `num_episodes`，不能传只适用于 TG 的 `num_env_steps`。
- 旧 CSV 的 `average_hops` 包含未投递惩罚，并非研究方案要求的真实成功包跳数。
- 旧 TG 输入及动作维度已变，旧权重需重新训练；本项目尚无正式完整 checkpoint/恢复入口。
- 研究参数必须走 `run_research.py`；旧入口会显式拒绝，研究入口也会明确报告尚未实现的组件。

## 6. 扩展及后续交接

目录职责、形状和 `register_components(registry)` 接口详见 [架构文档](RESEARCH_ARCHITECTURE.md)。
下一步独立实现并训练 WM 和 PM，冻结后接入已有 Graph-MAPPO 的 collect、buffer、evaluate_actions，
再完成完整保存恢复、长期效果验证和 A～E 同条件实验。
完整缺项和优先级见 [未实现与失败清单](UNIMPLEMENTED_AND_FAILED_TESTS.md)。

2026-10-21 的逐项优化和复测证据见 [稳定性优化记录](STABILITY_OPTIMIZATION_LOG.md)与
[Graph-MAPPO 优化记录](RESEARCH_POLICY_OPTIMIZATION.md)。
