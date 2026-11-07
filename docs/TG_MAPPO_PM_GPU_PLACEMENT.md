# PM 与 PPO 在 GPU、仿真在 CPU

日期：2026-11-07。原 TG-MAPPO 入口使用 `--tg_device cuda` 时，冻结 PM 在线推理现在跟随 Actor/Critic 使用 GPU。原 CPU 长实验继续运行。

## 设备划分

| 部分 | 设备/存储 | 具体内容 |
|---|---|---|
| 仿真 | CPU | 拓扑、移动、信道、队列、数据包转发及环境奖励 |
| 观测准备与缓存 | CPU | NumPy 观测、14 维前缀特征组装、动作掩码、回放缓存、奖励/GAE及小规模目标分布后处理 |
| PM 在线推理 | GPU | 冻结 GRU/hazard 网络、归一化缓冲、包隐藏状态、hazard 到期望任务收益的张量计算 |
| PPO 神经网络训练 | GPU | Actor/Critic 前向、反向、梯度裁剪与 Adam 更新；当前使用 FP32 |

PM 不参与反向训练。GPU 输出的势能标量和候选预测值返回 CPU，形成原有奖励与目标缓存；抽取的 PPO minibatch 再传到 GPU。没有将整个仿真或整个回放缓存搬入显存。

## 实现与接口

- `FrozenPMRuntime(..., device="cpu")` 保持默认兼容；原 TG 的 agent 显式传入自身策略设备。因此 `--tg_device cpu` 对应 PM/PPO 均在 CPU，`--tg_device cuda` 对应 PM/PPO 均在 GPU。
- PM 模型在 CPU 完成检查点校验和不消耗外部随机数的初始化，再整体移到指定设备。隐藏状态、输入、剩余窗口张量都使用同一设备，固定权重与设备检查继续执行。
- `metadata().inference_device` 记录 PM 实际设备；策略报告中的 `resources.device` 记录 Actor/Critic 设备。
- 同一时隙所有有效节点的假设候选前缀合成一次 GPU 批处理，随后一次返回候选预测。独立包的隐藏历史不被假设动作推进，候选特征、成功/失败概率和奖励公式保持原含义。
- CPU 路径保留逐包 batch 形状与计算顺序，防止设备优化改变旧 CPU 基线。诊断工具仍可单独调用 `packet_action_scores`。
- 最后一个动作没有未来收益，zero_set 不查询未来 PM；无活跃节点时不调用候选模型，WAIT 掩码逻辑保留。

## 使用命令

使用新 Conda 环境的解释器，以下是独立的两回合短验证命令，不会接管原 300 回合实验：

```powershell
& "D:\anaconda3\envs\buoy_gpu\python.exe" -s run_algorithm.py --algorithm tg-mappo --tg_device cuda --use_pm_reward true --pm_reward_checkpoint results/tg_pm_final_pretrain/20261103_142745_276042/pm_best.pt --pm_reward_alpha 0.1 --pm_potential_mode task_return --pm_value_mode base_residual --pm_action_credit learned_set --tg_reward_version task_v2 --seed 22 --scenario_seed 22 --num_episodes 2 --episode_length 80 --num_env_steps 160 --ppo_epoch 1 --num_mini_batch 2 --num_moving_buoys 5 --output_root results/pm_gpu_manual_smoke
```

运行结束检查 `moves_5/logs/TG_policy_report.json`：

- `resources.device` 为 `cuda`。
- `pm.inference_device` 为 `cuda:0`。
- `pm.frozen` 为 true；`training_updates` 显示实际 Actor/Critic Adam 步数。
- `pm.checkpoint_sha256` 应为 `f62f05b520cbaf665823ce38981c6c88f5e4f816dc01f85d3149aca7d3f81240`。

环境依赖及激活方式见 [GPU 环境说明](TG_MAPPO_GPU_ENVIRONMENT.md)。

## 本轮验证

结果：`results/tg_pm_gpu_split/20261107_163119_026665`。摘要：[验收证据](evidence/TG_PM_GPU_PLACEMENT_20261107.json)。

1. 最小设备迁移后：25 项原 PM CPU 测试通过；3 项真实 CUDA 测试通过。
2. 合批后：4 项 PM CUDA 测试通过，包含同快照 CPU/GPU 势能和目标一致性、逐包/合批一致性、CPU/CUDA/Python/NumPy 随机数与历史不变、终止/zero_set/无活跃节点边界、原生 GPU PPO。
3. 最终：6 项设备测试 + 4 项 PM CUDA 测试，10/10 通过；旧 CPU 相关测试 141 项，136 通过、5 个 CUDA 用例因旧 Torch 无 CUDA 正常跳过，无失败。
4. 默认 CPU 的原生 TG+PM 2×80 训练完成。与修改前参考逐项比较：Actor/Critic 张量、PPO 更新记录、物理日志完全一致。
5. 实际 PM H80 检查点的 GPU 2×80 训练完成，Actor/Critic 各更新 4 次，PM 保持冻结。包装器逐次检查 PM 前向的特征、隐藏、logits 和后继隐藏都在 GPU，仿真观测仍为 CPU 数据。
6. 保存后的 GPU 检查点通过 1×80 冻结评估：Actor/Critic 权重不变、0 次优化器更新，在线 PM 报告设备为 cuda:0。
7. 旧 CPU 长实验仍 running；其冻结源码快照哈希、PM 检查点哈希未改变。没有切换该实验的设备或预算，也没有启动新的 300 回合任务。

## 耗时与局限

仅作一次同预算短测：PPO 均在 GPU、同一新环境、同种子22、2×80 时隙、Torch CPU线程1；一个测试进程临时强制 PM 留在 CPU 作为对照，未增加公开设备组合选项。后台 CPU 长实验与部分回归检查也在运行，计时不能作为稳定性能结论。

| PM 位置 | PM 前向总次数 | 最大 batch | collect | PPO | 训练循环 |
|---|---:|---:|---:|---:|---:|
| CPU 对照 | 2026 | 24 | 6.263s | 0.855s | 15.049s |
| GPU，候选合批 | 318 | 303 | 5.686s | 0.951s | 15.376s |

GPU 前向次数包括真实进度势能和候选查询。collect 有所缩短，但总循环没有显示可靠加速，不能将调用次数减少换算成提速倍数。循环计时包含环境推进及原有日志/绘图，不含导入与模型初始化。

该对照的两份物理轨迹日志完全一致，Actor/Critic 参数最大绝对差异分别约 1.58e-7 / 1.79e-7，属于本次实测的微小设备数值差异。GPU 峰值 Torch 显存分配约 729MiB；这不等于 nvidia-smi 的进程总显存。

本轮只验证设备放置与运行链路，没有新增 PM 算法收益、长训练收敛或跨种子稳定性结论。当前 CPU 300 回合曲线仍以其独立 manifest/progress 为准；后续 GPU 研究需要统一环境重新比较，不能拼接旧 CPU 轨迹。
