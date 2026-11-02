# 第 4 步：原 TG 数据的一步 WM 监督预训练

历史说明：本页保留 2026-11-02 首轮结果。2026-11-03 最小三头版本已通过原定简单基线门槛，最新对照、选模修复和限制见 [续轮优化记录](TG_MAPPO_WM_OPTIMIZATION.md)。

日期：2026-11-02。当前结论：**网络/监督训练/保存/验证已跑通；预测质量门槛未全部通过，第 4 步仍在进行，不进入 Actor 集成。**

## 架构与数据

- 独立模块：[world_model.py](../src/buoy_network/world_model.py)，纯 Torch，无 Gym/PyG 依赖；Python 3.10 + Torch 2.11.0+cpu 实测。
- 数据仍来自第 3 步原 TG 环境，没有更换环境或原 TG Actor/Critic。
- 两层图消息传递编码原全局图，拼接本节点图表示、全图聚合和历史上下文，经独立 GRU 编码真实历史。
- 拓扑分支预测本节点下一时隙位移，不接收当前候选动作；业务分支结合候选接收节点图表示、相对位置、队列比例、链路历史和 WAIT 标记，预测 ACK 及发送/接收节点的联合时隙队列变化。
- 历史上下文包含本节点过去动作的方向、WAIT、已经发生的 ACK/尝试标记；当前真实 ACK、其它节点当前真实动作、未来状态不进入输入。
- 当前是最小三头版本；还没有完整链路概率、投递/丢弃预测头、不确定性/可靠性门控和 Actor 特征适配。
- 数据基于 25 节点、半径 1.5 的当前实验；固定坐标特征单位 /4、时间特征 /80，不声称已覆盖任意环境配置。

输入历史不足时右侧 padding，GRU 使用真实序列长度；节点均值/标准差和连续目标尺度只拟合 train。
WAIT 不参与 ACK 分类损失；接收端队列目标在 WAIT 时屏蔽。队列预测是整个联合时隙的变化，包含其它节点行为影响，不是 ACK 直接决定的即时队列变化。
队列误差单位为“队列占初始包数比例”的平方；位移误差单位为仿真坐标平方，基线采用相同单位。

训练入口：[train_tg_wm.py](../scripts/train_tg_wm.py)。模型参数 64,517，隐维 64，历史 4 步，Adam，梯度裁剪 5，CPU 单线程。
训练损失为 ACK BCE + 标准化队列 Huber + 标准化位移 Huber。checkpoint 按 validation 的 Brier 与标准化连续误差组合选取，训练后冻结权重评估。
测试集不参与归一化、梯度或选 checkpoint；第一次开发运行评估过 test，第二次修订加 `--skip-test`，避免进一步反复查看。正式结论仍需更多独立保留场景。

## 实际执行

数据：`results/tg_wm_mixed_data/20261102_231446_057743`。
训练 3,799 条有效包服务样本，验证 3,823 条，其中验证实际发送 3,301 条。多个节点对应同一联合转移，因此样本数大于时隙数。

先用 32 条训练样本检查过拟合：

```powershell
& "D:\python3.10.9\python.exe" scripts/train_tg_wm.py results/tg_wm_mixed_data/20261102_231446_057743 --overfit-samples 32 --epochs 200 --batch-size 32 --lr 0.003 --output-root results/tg_wm_overfit_v2
```

当前 ACK-history 版本：ACK Brier 由约 0.25 降到接近 0，队列 MSE 约 1.92e-8，位移 MSE 约 3.43e-8；说明小数据输入/标签、反向和保存链路可工作，不能用来说明泛化。
证据：[过拟合报告](../results/tg_wm_overfit_v2/20261102_233357_056183/report.json)。

完整预训练：

```powershell
& "D:\python3.10.9\python.exe" scripts/train_tg_wm.py results/tg_wm_mixed_data/20261102_231446_057743 --epochs 40 --skip-test --output-root results/tg_wm_pretrain_v2
```

本次运行选中第 11 epoch，训练内部耗时约 44.22 秒（本机一次运行，不作性能基准）。
权重：`results/tg_wm_pretrain_v2/20261102_233448_145133/wm_best.pt`，保存 state_dict、归一化、架构/维度、历史长度、数据路径及选中 epoch。
这是监督模型权重，不是 PPO checkpoint 或完整训练断点；暂未提供 Actor 候选特征转换和恢复训练接口。

| 验证指标（越低越好） | WM | 简单基线 | 结论 |
|---|---:|---:|---|
| ACK Brier | 0.24978426 | 训练 ACK 均值：0.24974817；历史链路：0.24904819 | 未优于两项基线 |
| ACK NLL | 0.69291925 | 本轮报告 Brier 基线 | 保留校准指标，未据此加分 |
| 联合队列变化 MSE | 0.00090870 | 不变：0.00121752 | 降低约 25.4% |
| 本节点位移 MSE | 0.00076813 | 零位移：0.00076772 | 未优于基线 |

证据：[训练与验证报告](../results/tg_wm_pretrain_v2/20261102_233448_145133/report.json)、[学习曲线](../results/tg_wm_pretrain_v2/20261102_233448_145133/learning_curve.csv)。
ACK/queue/motion 接入门槛为 false/true/false。checkpoint 文件在评估前后 SHA256 一致。

## 测试与本轮修订

WM 函数/模型测试：

```powershell
& "D:\python3.10.9\python.exe" -m unittest discover -s tests -p test_tg_world_model.py -v
```

3 项通过：未来标签变化不影响当前输入、拓扑头不依赖当前候选动作、真实梯度及权重往返一致且评估不改归一化。
加上真实 32 样本过拟合、完整监督预训练与独立验证作为训练联调证据。
原 TG/数据回归 50 项全部通过，失败/错误/跳过均为 0，见 [报告](../results/tg_step4_final_data_regression/20261102_233436_288707/test_report.json)。

第一版历史只含延迟更新的成功率摘要，验证 ACK 未过门槛；修订后加入严格上一时隙的 ACK/尝试标记，架构标记升级为 `legacy_tg_graph_gru_wm_v2_ack_history`。
原始采集 JSONL 无需重采；reader 从已存事件构建过去反馈。回归中一度错误查找“发完唯一包的空节点”下一条服务样本，已改用发包同时收到包的节点验证，未修改 reader 跳过无服务包节点的语义。
第一版记录保留在 [原报告](../results/tg_wm_pretrain/20261102_233002_679924/report.json)，不将其权重当作当前 v2 架构加载。

## Git 与下一步

第 3 步已本地提交 `babad62a`，分支 experiment/tg-wm-pm；首次推送 GitHub HTTPS 443 连接失败。
本次阶段结束会另行提交第 4 步代码及小型记录，原始数据与权重仍留在 results 本地，不自动上传。

下一步继续第 4 步：增加训练场景覆盖，按行为/阶段分别检查 ACK 校准；独立分析随机位移与可预测部分，不用加深网络替代误差诊断。
当前训练优于验证的差距提示存在泛化/场景覆盖问题，但这只是待验证解释，不是已证明的唯一原因。
只有可靠预测头通过独立验证后，才决定冻结接入第 5 步；本轮不声称创新收益或路由性能提升。
