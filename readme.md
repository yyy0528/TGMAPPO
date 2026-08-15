# Buoy Network 多智能体强化学习实验

本项目目前包含两组彼此隔离的实现：

- **TG-MAPPO**：项目的主要算法，代码位于 `src/buoy_network/` 和 `src/onpolicy/`。
- **基线算法**：DQN、IQL、MAPPO、QMIX、QMIX-ATTN、VDN 和 RBMADQ，入口位于 `baselines/main.py`，算法代码位于 `baselines/agents/`。

当前整合已统一启动方式、公共实验参数、CSV 指标格式和对比绘图，同时保留两组算法各自的环境与训练逻辑。

## 安装

建议使用项目原先配置的 Python 环境，并安装依赖：

```powershell
pip install -r requirements.txt
```

`run_training.cmd` 当前使用 `D:\python3.6.8\python.exe`。如果解释器位置不同，请修改脚本顶部的 `PYTHON_EXE`。

## 统一启动

查看支持的算法：

```powershell
python run_algorithm.py --list-algorithms
```

运行 TG-MAPPO：

```powershell
.\run_training.cmd
```

也可以直接使用统一入口，并向 TG-MAPPO 传递原有参数：

```powershell
python run_algorithm.py --algorithm tg-mappo --num_env_steps 10000
```

运行一个基线算法：

```powershell
python run_algorithm.py --algorithm dqn
python run_algorithm.py --algorithm iql
python run_algorithm.py --algorithm mappo
python run_algorithm.py --algorithm qmix
python run_algorithm.py --algorithm qmix-attn
python run_algorithm.py --algorithm vdn
python run_algorithm.py --algorithm rbmadq
```

使用相同的公共参数依次运行全部算法：

```powershell
python run_algorithm.py --algorithm all --seed 12 --grid_size 5 --num_moving_buoys 5 --num_episodes 1000 --episode_length 80 --output_root results
```

`all` 按 TG-MAPPO、DQN、IQL、MAPPO、QMIX、QMIX-ATTN、VDN、RBMADQ 的顺序运行；任一算法失败时停止，避免继续生成不完整的对比结果。

统一入口会自动为 TG-MAPPO 补充原 `run_training.cmd` 使用的 25 个节点、单 rollout 线程等必要配置；命令行显式传入的参数优先级更高。

这里的 `mappo` 表示原基线 MAPPO；`tg-mappo` 表示本项目的 TG-MAPPO。

为了兼容旧用法，下面的命令仍然可以运行基线：

```powershell
python main.py --algorithm VDN
```

## 当前结构

```text
buoy_network/
├─ run_algorithm.py          统一算法分发入口
├─ run_training.cmd          TG-MAPPO 的 Windows 快捷入口
├─ main.py                   旧基线入口（兼容保留）
├─ src/
│  ├─ buoy_network/          TG-MAPPO 浮标环境和执行器
│  ├─ onpolicy/              MAPPO / Graph-MAPPO 框架
│  ├─ environment.py         基线原环境（第一阶段保留）
│  ├─ runner.py              基线原执行器（第一阶段保留）
│  └─ replay_buffer.py       基线原经验回放池
├─ baselines/
│  ├─ main.py                基线统一训练入口
│  ├─ agents/                各基线算法
│  ├─ scripts/               原日志分析和绘图脚本
│  └─ archive/               历史代码，不参与正式启动
├─ outputs/                  TG-MAPPO 运行输出
└─ results/                  所有算法的统一 CSV 指标与对比图
```

## 统一实验结果

公共参数包括：

- `--seed`：随机种子
- `--grid_size`：浮标网格边长
- `--num_moving_buoys`：每轮结束后移动的浮标数量；支持 0 到浮标总数
- `--num_episodes`：训练轮数
- `--episode_length`：每轮最大步数
- `--output_root`：统一结果根目录

每种算法完成一轮后都会更新：

```text
results/runs/<运行时间>/moves_<移动浮标数>/logs/<ALGORITHM>_metrics.csv
```

统一列为：

```text
episode,delivery_rate,average_hops,average_delay
```

生成当前网格规模下的全部算法对比图：

```powershell
python scripts/plot_comparison.py --output-root results/runs/<运行时间> --num-moving-buoys 5
```

图片保存在 `results/runs/<运行时间>/moves_<移动浮标数>/comparison_plots/`。

在默认 5×5、共 25 个浮标的实验中，移动节点按编号均匀选择：`0` 表示全部静止，`1` 移动节点 0，`5` 移动节点 0/5/10/15/20，`25` 表示全部移动。

每次执行统一入口都会创建新的时间戳目录，并在终端输出其绝对路径，因此不会覆盖之前的 `moves_*` 实验结果。也可以用 `--run-name experiment_01` 指定便于识别的运行名称；如果同名目录已经存在，程序会停止而不是覆盖。

## 当前边界

- TG-MAPPO 和基线仍使用各自原有训练流程。
- 基线环境仍保留在 `src/` 根部，以避免第一阶段搬迁改变实验行为。
- `baselines/archive/` 仅用于保留历史文件，不属于运行依赖。
- 两组算法现在使用相同参数名和指标列，但其环境实现仍不同；正式论文对比前仍需逐项核验环境语义是否一致。
