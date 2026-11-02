# 第 3 步：原 TG-MAPPO 数据采集与核验

日期：2026-11-02。接续 [接口说明](TG_MAPPO_INTERFACE.md)与 [实验进度](TG_MAPPO_EXPERIMENT_PROGRESS.md)。
这里使用原 TG-MAPPO 的 `buoy_environment`，没有替换为 reference_grid，也没有训练 WM/PM。

## 1. 本次实现

| 文件 | 职责 |
|---|---|
| [transition_data.py](../src/buoy_network/transition_data.py) | 不可变 JSON 快照、实际传输事件、包轨迹与预算限制；数据专用规则采样器 |
| [runner.py](../src/buoy_network/runner.py) | 可选采集钩子；成功包实际转移后、删除前归档；规则采集回合跳过 PPO |
| [main.py](../src/buoy_network/main.py) | 显式导出开关、行为来源、数据预算及可选场景种子 |
| [buoy.py](../src/buoy_network/buoy.py)、[environment.py](../src/buoy_network/environment.py) | 支持显式场景 RNG 种子；未指定仍使用历史 2568 |
| [dataset.py](../src/buoy_network/dataset.py) | 数据一致性审计、跨场景划分检查、WM 历史窗口读取 |
| [collect_tg_data.py](../scripts/collect_tg_data.py) | 独立进程、多行为/多种子采集，保存命令及原始 stdout/stderr |
| [audit_tg_dataset.py](../scripts/audit_tg_dataset.py) | 无需 Torch/Gym 的独立数据审计，导出可阅读 WM 样本 |
| [test_legacy_transition_data.py](../tests/test_legacy_transition_data.py) | 函数、原环境模块、独立进程测试；已接入 test_legacy_fixes.py |

默认 `--export_wm_data false --data_behavior tg-mappo --scenario_seed` 未指定，维持原基线行为。
`data_behavior=random/geographic` 必须同时启用导出，是数据采集模式，不能把其回合记作 TG-MAPPO 训练。
规则模式保留现有初始化/缓冲接口，但跳过 PPO 的 compute/train；其 value 字段为未使用的初始 Critic 预测，WM reader 不使用该字段。

## 2. 文件结构和样本含义

```text
一次采集目录/
  dataset_manifest.json          # 全部来源、场景种子、行为、train/validation/test 划分
  <behavior>_seed_<n>.command.json
  <behavior>_seed_<n>.stdout.log / stderr.log
  source_runs/runs/<behavior>_seed_<n>/moves_5/
    logs/                       # 原 TG 配置、训练/采集指标
    wm_data/
      manifest.json             # schema、配置、完成状态、真实计数、语义及预算
      snapshots.jsonl           # 一个联合时隙一份快照，不为每个 Agent 复制整个图
      transitions.jsonl         # t→t+1、动作、行为概率、事件、奖励与边界
      packet_traces.jsonl       # 包级历史，成功包删除后仍可读取
```

单个快照包含：

- `snapshot_id` 和 `step`。
- `actor_inputs`：原普通观测、节点编号、原全局图节点特征、距离矩阵和合法动作掩码。
- `diagnostic_only`：真实丢包率、包对象状态、旧历史奖励表等仿真真值；不进入 WM 输入。

一个转移引用 `state_t_ref`、`state_tp1_ref`；同一回合 T 个转移只有 T+1 个快照。
`behavior` 记录真实来源、策略版本、实际动作、旧 log probability 和价值；TG 模式在回合末更新 PPO，策略版本按更新边界记录。

每个节点事件记录：当前动作、时隙开始待服务包键、是否实际尝试、ACK、尝试前后双方包计数、本步实际投递、旧奖励分项。
队列的“本次尝试变化”和“整个联合时隙的最终变化”不同：其它节点也可能向本节点发送，训练时不能混淆这两个目标。
只有执行过的动作拥有真实后果标签；未执行候选动作没有仿真产生的反事实真值。

包键为 `(episode_id, packet_id)`，episode_id 包含当前来源 run 名；跨独立采集目录合并时还须保留采集目录标识，不能只按 packet_id 拼接。
包前缀按时序保存快照/事件引用和当前节点，最终 route/outcome 单独归档。构建 PM 前缀输入时仅读取对应时间之前的事件，不把完整未来 route 或最终 outcome 当作特征。

## 3. 标签约定

| 情况 | ACK 标签 | PM 最终 outcome | 说明 |
|---|---|---|---|
| WAIT、没有实际发送 | null，ack_loss_mask=false | 不据此决定 | 不伪装成发送失败 |
| 尝试失败 | false，ack_loss_mask=true | 未结束时仍是 null | ACK 失败可以重传，不等于包最终失败 |
| 实际成功转发 | true，ack_loss_mask=true | 非 Sink 时仍待定 | 以包真实 transmit 事件为准 |
| 实际到达 Sink | 对应实际尝试结果 | 1，censored=false | 删除包对象前归档，并记录时隙末投递时间 |
| 回合末仍未投递 | 取决于最后一次尝试 | null，censored=true | 结束原因为 collection_horizon_unresolved |

当前环境没有真实 TTL 递减/丢弃机制：`life=80` 作为静态 legacy 字段放在诊断信息中，不称为剩余 TTL。
本步不修改任务规则；目前不会生成 PM 的最终失败 outcome=0。
因此这些数据可用于 WM 预训练；PM 的成功/失败结果监督还需明确任务截止或真实失败定义，不能把删失样本直接标为失败。

`env_terminated=true` 对应原 Runner 的固定回合末 done；`rollout_truncated=false`，结束原因为 legacy_episode_horizon。
这记录原 PPO 的实际行为，不意味着每个包都已有最终成败；不宣称已实现未结束任务跨 rollout 续采。

旧奖励按 `step_penalty/history/distance_progress/link_quality/failed_transmission` 分项保存，并核对和等于实际返回值。
`reward_legacy` 是节点奖励求和后的团队奖励；当前 `reward_progress=0`，`reward_train=reward_legacy`。
没有将旧密集/历史奖励改写成“稀疏任务奖励”，也没有增加 PM 塑形。

## 4. 划分、预算和读取

- 同一个场景种子的所有行为、所有回合归入同一个 split；相邻窗口不会跨回合或跨集合。
- 种子 0/101/202 分别用于 train/validation/test，每个“行为+场景种子”独立启动进程，跨集合不共享训练状态或 added_value 表。
- 场景种子是本次新增的显式随机配置，控制原环境信道初始化等；布局仍为同一个 5×5 原浮标场景，不宣称覆盖多种网络规模。
- 默认总预算为 20,000 个联合时隙、10,000 个包和 256 MiB 原始 JSONL；本次仅采集小数据。采集超限明确失败并保留 incomplete 标记，不悄悄截断为完整数据。
- 当前只保存原始值，未拟合任何归一化器。第 4 步仅用 train 拟合，validation/test 不参与参数、归一化及阈值拟合。

WM reader 示例（在项目根目录，使用支持本项目的 Python）：

```python
from src.buoy_network.dataset import iter_wm_samples

sample = next(iter_wm_samples("results/tg_wm_mixed_data/<本次目录>", split="train", history_length=4))
print(sample["inputs"])
print(sample["targets"])
```

`inputs.history` 只有截至 t 的原 Actor 信息及本节点过去动作；`candidate_action` 是本节点当前动作。
`targets` 中才有下一观测、下一图特征、ACK/损失掩码、局部尝试队列变化、本步事件和旧奖励。
输入没有其它节点当前真实动作、未来状态、隐藏丢包率真值或最终包结果；历史长度不足时保留真实短历史，由后续模型处理 padding/mask。
返回的数据是 WM 的监督样本，不应拿旧数据直接替换 PPO 的当前策略 rollout。

## 5. 实际执行与验收

先完成 TG-only 采集：3 场景 × 5 回合 × 80 时隙 = 1200 时隙，360 个包，104 投递、256 未结束；数据独立审计通过。
证据：[TG-only 审计](../results/tg_wm_data/20261102_230328_494699/audit_report.json)。

随后按计划采集三种行为的混合数据；以下命令会创建新时间戳目录，不覆盖已有数据：

```powershell
& "D:\python3.6.8\python.exe" scripts/collect_tg_data.py --episodes 2 --episode-length 80 --seeds 0 101 202 --behaviors tg-mappo random geographic --output-root results/tg_wm_mixed_data
```

本次混合目录：`results/tg_wm_mixed_data/20261102_231446_057743`。
本次实际完成 9 个独立进程、18 回合、**1440 联合时隙、1458 快照、432 条包轨迹**。
223 条真实投递，209 条未结束；没有人为伪造最终失败标签。JSONL 总大小 53,670,677 bytes，约 51.18 MiB。

| 来源 | train：种子 0 | validation：种子 101 | test：种子 202 |
|---|---|---|---|
| TG-MAPPO | 160 时隙；14 投递/34 未结束 | 160；15/33 | 160；18/30 |
| 合法随机 | 160；8/40 | 160；7/41 | 160；17/31 |
| 地理规则 | 160；48/0 | 160；48/0 | 160；48/0 |

三个行为各占 480/1440 联合时隙；有效发送样本占比不同，应在 WM 训练采样时报告。
上述计数是采集数据的物理包事件统计，不是收敛性能或正式消融实验。

独立审计：[audit_report.json](../results/tg_wm_mixed_data/20261102_231446_057743/audit_report.json)。
包含六条可阅读的 WM 样本预览；检查了时序、合法动作、ACK/队列一致性、奖励分项、包守恒、前缀、划分和磁盘预算。
源码保留：[source_provenance.json](../results/tg_wm_mixed_data/20261102_231446_057743/source_provenance.json)及同目录 source_snapshot，126 个源码文件逐一核对 SHA256；解释器及依赖不在副本中。

人工核对的一条真实样本：train/tg-mappo_seed_0/episode_000000，t=0，发送者 0、动作 5、距离 1.0、掩码为 1。
实际尝试 ACK=false，双方包计数由 [1,1] 保持 [1,1]；节点奖励 -35.5，由 -35 步罚和 -0.5 发送失败项组成。
状态引用严格为 t=0→1，当前与下一普通观测均为 26 维；该包回合末仍未结束，outcome=null、censored=true，最后观测为 80。
它不是 PM 的失败标签，也没有把 t+1 观测提前放入 WM 输入。

复核现有数据（无需安装旧 Gym/PyG）：

```powershell
& "D:\python3.10.9\python.exe" scripts/audit_tg_dataset.py results/tg_wm_mixed_data/20261102_231446_057743
```

如需保存包含样本预览的审计 JSON，可加 `--output <新文件路径>`；为保护历史报告，输出文件要求不存在。

测试命令：

```powershell
& "D:\python3.6.8\python.exe" scripts/test_legacy_fixes.py --skip-smoke --output-root results/tg_step3_acceptance_regression
```

最终 TG 回归：函数 20、模块 24、联调 6，**50 项通过，失败/错误/跳过均为 0**。
包含导出开关前后训练 CSV 一致、数据损坏拒绝、WAIT/ACK、实际成功包归档、删失标签、源 run 去重、场景跨集合拒绝，以及规则回合不执行 PPO。
证据：[50 项报告](../results/tg_step3_acceptance_regression/20261102_231256_960712/test_report.json)。
参考框架另做 96 项回归通过，见 [报告](../results/tg_step3_research_regression_unrestricted/20261102_230423_725946/test_report.json)。
它用于检查共享旧入口兼容性，不将参考环境数据混入原 TG 数据集。

最终代码还以原基线配置开启导出运行 2×80 时隙；[CSV](../results/tg_step3_final_baseline_compat/runs/20261102_231444_672746/moves_5/logs/TG_MAPPO_metrics.csv)
与第 2 步 CSV 完全一致（投递率 0.375/0.25）。规则采集与 scenario_seed 均为显式配置，不把这些配置的结果混作原默认基线。

## 6. 已发现问题及处理

1. 原环境两处 RNG 固定为 2568，CLI policy seed 不等于场景种子。增加可选 scenario_seed，默认不变；采集时显式指定，测试默认兼容和场景差异。
2. 原入口拒绝一切 wm_ 前缀。仅允许四项导出设置，其余未实现 WM/PM 参数仍明确拒绝。
3. Windows 文本模式把 LF 转成 CRLF，造成字节计数偏差。已固定 JSONL 为 LF，并用文件真实字节数审计预算。早期 8 时隙原型产物仍保留，但不作为最终数据。
4. 参考框架测试在沙箱临时目录重命名时遭权限拒绝；获准沙箱外重跑后 96 项通过，没有为权限问题修改算法。

## 7. 下一步

第 3 步已完成数据导出、少量多行为采集和核验。数据位于本地 results 下，被 .gitignore 忽略；提交代码和文档不会自动备份这些数据，需另行保留实际采集目录及源码副本。

第 4 步先选择发送 ACK 和队列变化等最小预测头，用 train 的真实历史窗口做监督训练和小数据过拟合；用 validation 选模型、test 留作最终预测评估。
稀有事件、失败包标签不足、策略分布变化要如实报告；尚未实现 WM/PM 网络，不能把“数据采集可用”写成“模型训练完成”。
