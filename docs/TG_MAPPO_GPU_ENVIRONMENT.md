# 独立 Conda GPU 环境与验证记录

日期：2026-11-07。目标：保留正在运行的 CPU 300 回合实验，新建独立环境，让原 TG-MAPPO 能显式使用 GPU。

## 环境与使用

- 环境名：`buoy_gpu`。
- Python：`D:\anaconda3\envs\buoy_gpu\python.exe`，3.10.22。
- PyTorch：2.6.0+cu124；CUDA runtime：12.4；PyG：2.6.1；NumPy：1.26.4；Gym：0.23.1。
- 显卡：RTX 4060 Laptop，8GB；驱动 576.80；实际 CUDA 检查为 True。
- 原 CPU 环境 `D:\python3.6.8\python.exe` 保留。未修改现有 autotrain 和 demo venv 的依赖。
- 新环境设置 `PYTHONNOUSERSITE=1`，激活时生效。直接指定 Python 时加 `-s`，避免 Windows 共享用户包混入。
- 顶层版本配置：[environment-gpu.yml](../environment-gpu.yml)、[requirements-gpu.txt](../requirements-gpu.txt)；完整实际包版本：[锁定清单](evidence/TG_GPU_ENV_PACKAGES_20261107.txt)。

在新 PowerShell 中可以使用：

```powershell
conda activate buoy_gpu
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

若 Conda 命令受本机策略限制，直接指定解释器即可，不需要执行 Activate.ps1：

```powershell
& "D:\anaconda3\envs\buoy_gpu\python.exe" -s -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

原 TG 的 GPU 短训练命令如下，每次会生成独立结果目录：

```powershell
& "D:\anaconda3\envs\buoy_gpu\python.exe" -s run_algorithm.py --algorithm tg-mappo --tg_device cuda --num_episodes 2 --episode_length 80 --num_env_steps 160 --ppo_epoch 1 --num_mini_batch 2 --num_moving_buoys 5 --tg_reward_version task_v2 --seed 22 --scenario_seed 22 --output_root results/gpu_manual_smoke
```

默认仍是 CPU；原 TG 通过 `--tg_device cuda` 选择显卡，不使用另一个研究框架的 `--device`。参数省略或 `--tg_device cpu` 均保持 CPU。请求 CUDA 而不可用时明确报错。

TG+PM 在同一命令上追加以下参数，固定 PM 仍在 CPU 推理，Actor/Critic 与 PPO 使用 GPU：

```text
--use_pm_reward true --pm_reward_checkpoint results/tg_pm_final_pretrain/20261103_142745_276042/pm_best.pt --pm_reward_alpha 0.1 --pm_potential_mode task_return --pm_value_mode base_residual --pm_action_credit learned_set
```

复建时在项目根目录执行 `conda env create -f environment-gpu.yml`（环境名已存在时不要重复创建）。本次实际执行的是 Conda create Python，再使用该环境的 `python -s -m pip install -r requirements-gpu.txt` 安装，并验证了导入路径与依赖；没有另建第二个环境重复验证 YAML 全流程。

版本选择依据：[PyTorch 官方历史版本及 CUDA wheel](https://docs.pytorch.org/get-started/previous-versions/)、[PyG 官方安装说明](https://pytorch-geometric.readthedocs.io/en/stable/notes/installation.html)。本项目使用的图算子已通过原生 PyTorch 路径验证，没有额外编译可选的 torch-scatter 等扩展。

## 发现的问题与修正

1. **固定 CPU**：原 agent.py 的全局 device 固定为 cpu，安装 CUDA 版 Torch 不会自动切换。增加独立的 compute_device 解析与 `--tg_device`，Policy/Trainer/旧辅助网络统一使用实例设备；NumPy 输出显式回到 CPU。
2. **GPU 参数丢失注册**：PopArt 使用 `nn.Parameter(...).to(cuda)`，返回普通 Tensor，weight/bias 不在参数列表与优化器内；ValueNorm 统计量有相同注册问题。改为先注册/初始化，再整体移动模块；CPU 初始化顺序保持一致。测试验证 GPU 参数入优化器、更新、统计量保存与还原。
3. **自动 FP16 不稳定**：第一次 GPU 两回合运行，Critic 梯度范数出现 inf，严格 JSON 报告保存失败。保留失败目录 `results/gpu_env_validation_20261107/tg_gpu_1` 和日志。移除 PPO 的自动 FP16 装饰器，禁用 GradScaler，当前 CPU/GPU 都使用 FP32；没有通过写入 null 或忽略 inf 掩盖问题。FP16 尚未作为可用功能验收。
4. **共享用户包污染**：初次 pip 可见 AppData/Roaming 下的 Python3.10 包；改用 `-s` 补齐新环境本地依赖，并为该 Conda 环境设置 PYTHONNOUSERSITE。检查核心包路径均位于新环境。
5. 资源报告按 GPU 设备记录实际峰值显存，计时字段改为主机观测的 wall time。

## 验证结果

原始结果目录：`results/gpu_env_validation_20261107`。摘要与源码哈希：[验收证据](evidence/TG_GPU_ENV_VALIDATION_20261107.json)。

- 新环境：pip check 通过；CUDA 矩阵乘及反向、TransformerConv 图网络前向/反向通过。
- 新环境设备/注册测试：6/6 通过，包含真实 CUDA 优化器更新。
- 旧环境相关回归：137 项，136 通过、1 项因旧 Torch 无 CUDA 正常跳过；无失败。
- 最新默认 CPU 原生 2×80 训练再次完成，与修改前探针参考的 Actor/Critic 张量、PPO 更新记录及物理日志逐项完全一致。
- 新环境 TG、TG+PM：各在 CPU/GPU 完成 2×80 时隙；每个模型均完成 2 次 PPO 更新、Actor/Critic 各 4 次 Adam 步，参数有限且实际改变。PM 权重保持冻结。
- 两个 GPU 模型各完成 1×80 冻结评估：检查点加载成功，0 次优化器更新，Actor/Critic 权重均不变。
- 修正后没有遗留失败的上述测试。首次 FP16 失败是已记录并规避的失败尝试，不计作通过。
- 这里验收的是设备与运行链路，尚未验证新环境 GPU 的长训练收敛、跨种子收益或 WM+PM 全组合。

## GPU 是否更快

以下仅为同一新环境、1 个 Torch CPU 线程、同种子/同预算的各一次 2 回合粗测。GPU 为 FP32；后台 CPU 长实验持续运行，且 CPU/GPU 随机采样轨迹可不同。不能据此作正式性能或策略效果结论。

| 路径 | CPU PPO | GPU PPO | CPU 训练循环 | GPU 训练循环 |
|---|---:|---:|---:|---:|
| TG | 3.120s | 0.653s | 8.811s | 8.309s |
| TG+PM | 3.712s | 0.990s | 17.198s | 15.744s |

PPO 约加快 4.8/3.8 倍，整个循环仅缩短约 6%/9%。循环包含环境推进及原有日志/绘图，排除模块导入及模型初始化。峰值 Torch 显存分配约 722/729MiB，不等于 nvidia-smi 的整个进程/显示系统占用。

原因：网络较小、每个时隙仍需同步 CPU 环境；PM 及动作教师推理仍在 CPU，GPU 不能消除这些开销。新环境不是把当前 CPU 实验“搬到 GPU”；Python/Torch/PyG 与随机数实现也有变化，后续 GPU 对比应各组统一环境重新开始，不能拼接当前 CPU 曲线。

## 原 CPU 300 回合实验

继续使用目录 `results/tg_pm_convergence300/20261107_143512_724348`，控制器 PID 66616，冻结源码 74f4d630。此次安装和代码修改不改变其 source_snapshot、模型、预算或协议。

本次只读检查：整体 running，TG_11 已完成 300 回合，Immediate_11 正在训练；其余任务按原顺序继续。确切回合数以该目录 manifest 和 progress 为准。这里只完成独立 GPU 环境准备与短验证，不代表原 9 模型长实验完成。
