# 五组 TG-MAPPO 300 回合对比实验

日期：2026-11-08。用户要求：原 TG-MAPPO、即时奖励对照、PM、WM、PM+WM 五组，每组运行 300 回合，完成后生成对比图。

## 固定协议

| 组 | WM Actor 特征 | PM 势能与价值坐标 | 动作辅助目标 | Actor / Critic 可训练参数 |
|---|---|---|---|---:|
| TG | 关闭 | 关闭 | 关闭 | 21956 / 18615 |
| Immediate | 关闭 | task_return / base_residual，alpha=.1 | zero_set，即时收益对照 | 21956 / 18615 |
| PM | 关闭 | 同上 | learned_set，冻结 PM 预测未来收益 | 21956 / 18615 |
| WM | 冻结 WM 的 ACK/联合队列变化预测 | 关闭 | 关闭 | 24165 / 18615 |
| PM_WM | 同 WM | 同 PM | 同 PM | 24165 / 18615 |

原环境：25 节点、5 个移动浮标、随机信道、80 联合时隙/回合；task_v2 三项权重均为 1。PPO 每回合 1 epoch、2 minibatch，Actor/Critic 学习率固定为 .0005/.0003。PM 辅助损失系数 .1、温度 .2，保持已经验证的版本。

本轮五组统一使用 `D:\anaconda3\envs\buoy_gpu\python.exe -s`，PyTorch 2.6.0+cu124 / PyG2.6.1。Actor、Critic、冻结 PM 和冻结 WM 模型计算均在 GPU；仿真、观测特征组装、NumPy 回放/GAE 在 CPU。Torch CPU 线程固定为 1，FP32，关闭 TF32 和 AMP，启用确定性算法及 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

三个训练种子为 11/22/33；每个模型从初始化连续训练 300 回合。每 20 回合（含第 0 和 300 回合）在场景8001/8002/8003上各做2回合独立冻结评估。

- 15 个模型，共 4500 训练回合、360000 训练时隙。
- 240 个策略检查点，720 个场景评估进程、1440 评估回合、115200 评估时隙。
- 顺序运行一个训练任务；检查点评估用隔离子进程，暂停父训练并校验参数/优化器、CPU/CUDA RNG、WM/PM 历史均不改变。
- 三个已知训练种子的初始共享 Actor 主干、Critic、优化器设置及 RNG 一致。WM 有额外的 2209 参数零残差融合头，报告不将它隐去；未增加额外容量匹配第六组。
- 仅是用户指定的五组已知种子对比。3041–3045 保留，不据此声称未见种子泛化；完整原始研究方案的可靠性/失败分类等未实现部分也不因本次实验而完成。

## 冻结辅助模型与旧实验

- WM：`results/tg_wm_multiscene_gate_selection/20261103_000951_565881/wm_best.pt`，SHA256 `2919dff60d3128321170410e632e7bddd476a4f09c766b92108bfe9cb73daca9`。
- PM：`results/tg_pm_final_pretrain/20261103_142745_276042/pm_best.pt`，SHA256 `f62f05b520cbaf665823ce38981c6c88f5e4f816dc01f85d3149aca7d3f81240`。
- 准备正式目录时复制到 frozen_models 并固定哈希；不重新训练或调选 WM/PM。
- 旧 CPU 三组实验 `results/tg_pm_convergence300/20261107_143512_724348` 已完成，工程验收通过，最终投递均值 TG64.81% / Immediate47.22% / PM97.22%。它作为历史参考独立保留。
- 新五组全部在统一 GPU 环境从头开始，不把旧 CPU 曲线拼接进新五组。

## 入口与输出

配置：`configs/tg_five_gpu_300.json`；短验证配置：`configs/tg_five_gpu_probe.json`。

```powershell
& "D:\anaconda3\envs\buoy_gpu\python.exe" -s scripts/run_tg_five.py --prepare configs/tg_five_gpu_300.json
& "D:\anaconda3\envs\buoy_gpu\python.exe" -s scripts/run_tg_five.py --run <prepare输出的目录>
```

不要对已运行目录重复启动。runner.lock 防止同目录并发；中断或失败模型保留现场，不静默从头重复训练。现有策略检查点用于冻结评估，不支持完整优化器/RNG训练恢复。

状态 prepared / running / postprocessing / completed / failed 明确区分。只有所有15个300回合模型、全部冻结评估、独立审计及出图成功，才写 completed。算法效果不好也照实生成结果，不将工程验收通过写成所有创新均有效。

结果目录：

- `manifest.json`：15个模型任务状态、预算、源码哈希。
- `execution.json`：启动记录、后台PID；最终状态以 manifest/report 为准。
- `protocol.json`、`dependencies.txt`、`source_snapshot/`、`frozen_models/`：配置、依赖、代码和辅助模型。
- `jobs/<组_种子>/convergence/progress.json`：当前回合和已完成检查点；对应 worker.log 为详细日志。
- `live/convergence.png` 及 CSV：逐检查点更新的部分结果，不作为全部完成的结论。
- `analysis/convergence.png/.svg`：五组冻结投递率均值±训练种子SD、投递率随训练耗时、训练探索曲线、原任务回报。
- `analysis/final_comparison.png/.svg/.pdf`：第300回合最终投递率与AUC汇总图。
- `analysis/final_comparison.csv`、`evaluation_curves.csv`、`training_curves.csv`、`optimization_diagnostics.csv`：原始数值。
- `REPORT.md`、`report.json`：最终审计、按种子达标时间、逐种子组间效果差异与范围说明。

PM+WM 分别与 PM、WM 比较；报告同时保留即时对照、PM和WM相对TG的配对差异。最终值固定取第300回合检查点，不事后挑最好的回合冒充最终结果。

50%/80%/90% 达标时间以连续两个检查点确认，分辨率20回合；未达到保留未确认状态，不当成0回合。最后5个点均值≥80%、极差≤5个百分点只是观察到的平台，不是数学收敛证明。误差带是3个训练种子的样本SD；场景在每个种子内求平均。

## 启动前验证与记录

- 旧 CPU 相关回归156项：148通过、8个CUDA用例按环境正常跳过。
- 新环境设备、PM、WM、协议和收敛测量共25项通过，含真实WM及PM+WM GPU短训练。
- 初次 WM CPU/GPU 对齐检查失败：GPU默认TF32带来差异；固定同权重实测开启时最大约4.1e-5，关闭后约3.2e-8。测试采用正式实验的FP32设置后通过，没有放宽容差或修改辅助模型权重。
- 原型五组短训及自动审计已完成：`results/tg_five_gpu_probe/20261108_134544_901175`。补齐精度记录、短图回合/种子标注及子进程隔离后，最终版本探针为 `results/tg_five_gpu_probe/20261108_135245_396245`。
- 正式目录及最终状态见下节。短验证仅检验链路，不用其两回合性能筛选算法。


## 正式运行记录（已完成）

- 实际结果目录：`results/tg_five_gpu300/20261108_140256_956180`。
- 启动时间：2026-11-08 14:03（Asia/Shanghai）；后台控制器PID：60524。
- 源码冻结版本：`be36e860`，源文件哈希记录在 manifest.json；WM/PM 已复制并固定版本。
- 当前状态：completed。15个模型均完成300回合，全部720次场景冻结评估、工程验收及出图通过；2026-11-09 01:07（北京时间）结束。
- 最终探针5/5模型、0/1/2全部检查点及冻结评估已完成；独立验收和图形检查通过，完整记录见 [启动前证据](evidence/TG_FIVE_GPU300_PREFLIGHT_20261108.json)。
- [启动证据](evidence/TG_FIVE_GPU300_START_20261108.json) 保存预算、协议、依赖环境、PID和结果入口。execution.json 是启动记录；实时状态看 manifest/progress。
- 自动最终图：`analysis/convergence.png`、`analysis/final_comparison.png`；同时输出SVG/PDF和CSV。只有全部训练、评估、验收和绘图结束才标记completed。
- 实际从2026-11-08 14:03运行至11月9日01:07，约11小时4分钟（含评估与收尾）。任务已结束，不重复启动。


## 最终结果（2026-11-09）

三种子最终冻结评估的均值±样本SD：

|组|投递率|标准差|投递AUC|
|---|---:|---:|---:|
|TG|59.95%|5.02个百分点|0.5081|
|Immediate|68.29%|21.02个百分点|0.5385|
|PM|95.60%|4.19个百分点|0.9439|
|WM|59.95%|7.29个百分点|0.4584|
|PM+WM|100.00%|0.00个百分点|0.9630|

PM相对TG三种子最终投递率均提升。WM单独最终均值相同，且三个种子的AUC均低于TG，不能写成WM单独已有效。PM+WM相对PM最终投递率两个种子提升、一个持平，平均提升4.40个百分点；原任务回报和AUC在三个种子均提升。本轮不能推出统计显著性或未见种子泛化，WM融合头额外2209参数的容量因素仍需对照。

结果和图已核验哈希，PNG已人工查看。原始结果在上述目录的REPORT.md、report.json、analysis/convergence.png和analysis/final_comparison.png，另有SVG/PDF/CSV。[完成证据](evidence/TG_FIVE_GPU300_COMPLETED_20261109.json)。没有为修饰结果而改模型、重训或挑选回合。
