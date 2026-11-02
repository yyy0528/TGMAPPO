# 小步优化与稳定性记录

本页为 2026-10-21 较早批次的绘图与预算优化记录；随后基础研究策略的实现与复测见
[Graph-MAPPO 优化记录](RESEARCH_POLICY_OPTIMIZATION.md)，当前评分见 [PROJECT_SCORECARD.md](PROJECT_SCORECARD.md)。

日期：2026-10-21（Asia/Shanghai）。本轮按“复现 → 解释原因 → 修改一点 → 测试通过 → 下一点”推进。
预算设计及绘图兼容性另外做了只读复核。未引入新的 WM/PM 网络或修改训练算法。

## 1. 单回合数据可见

原因：单个数据点没有第二点可连线，原 `plt.plot` 又未设置 marker，生成图片只有图例。

修改：`scripts/plot_comparison.py` 仅对单点设置圆点；多点曲线及数据值保持原状。
先运行 4 项测试，覆盖单点、多点、空输入目录、独立 CLI；全部通过，并查看了实际 PNG。

证据：[步骤 1](../results/optimization/20261021_130352_841024/step01_single_point/result.json)、
[单点示例图](../results/optimization/20261021_130352_841024/step01_single_point/moves_5/comparison_plots/delivery_rate_comparison.png)。
该示例是绘图测试数值，不作为训练或算法性能结果。

## 2. 训练预算真实生效

原因：旧 Runner 按 `num_episodes` 循环，`num_env_steps` 没有控制训练；统一入口和 CMD 还自动填入 200，
与默认 1000×80 的实际预算不一致。

修改：在 `src/legacy_cli.py` 增加纯标准库预算解析，主入口在创建环境前完成校验；
删除两个入口自动填入的 200。Runner 的真实训练循环未改。

| 用户参数 | 现在的行为 |
|---|---|
| 不给步数 | 按回合数×回合长度，统一入口默认仍为 1000×80 |
| 只给 `--num_env_steps 160`，回合长度 80 | 推导 2 回合 |
| 给 160 步和 2 回合，长度 80 | 一致，正常执行 |
| 给 160 步和 3 回合，长度 80 | 明确报错 |
| 给 200 步，长度 80 | 不是完整回合整数倍，明确报错 |

支持 `--参数=值` 和重复参数最后值；旧 TG 禁止参数缩写，避免缩写被接受后遗漏显式识别。
记录 `requested_num_env_steps`、`requested_num_episodes`、`effective_num_env_steps`、
`training_budget_source`、`training_step_unit=joint_slot` 到现有配置快照，并在启动时打印预算。

验证：

- 10 项预算测试通过，覆盖默认/仅回合/仅步数/一致/冲突/零负数/余数/重复参数/缩写/入口默认。
- 只传 8 步、长度 4，真实旧 TG 运行两回合，CSV 两行，配置记录有效预算 8。
- `run_training.cmd` 的同样调用也实际运行两回合；CMD 继续支持后置参数覆盖原默认参数。
- 原 16 项旧回归及 2×80 时隙真实训练通过后，再进入第三项优化。

证据：[预算测试](../results/optimization/20261021_130352_841024/step02_budget/tests.json)、
[仅步数真实训练](../results/optimization/20261021_130352_841024/step02_budget/steps_only_training.json)、
[CMD 真实训练](../results/optimization/20261021_130352_841024/cmd_smoke.json)。

边界：`num_env_steps` 只适用于旧 TG；统一 `all` 仍使用公共 `num_episodes`，因为其后七种基线不接收步数参数。
预算以联合时隙计数，不乘节点数；旧 TG 仍仅支持一个 rollout 环境。

## 3. 拒绝无效绘图数据

原因：只有表头的 CSV 被当成空数据，`float('nan')`/无穷值也能通过读取，可能生成空图或缺失片段。

修改：读完全部 CSV 并验证后才创建输出目录；拒绝空数据、非法数字、缺失单元格及非有限指标，
非法行报告源文件和物理行号。合法的 0 值仍然可用。

绘图测试扩展到 9 项，全部通过；Python 3.6 和 3.10 均运行通过。
兼容性扫描的 36 份已有统一 CSV、8113 行数据，拒绝 0 份。

证据：[步骤 3 测试](../results/optimization/20261021_130352_841024/step03_metric_validation/tests.json)、
[跨版本测试](../results/optimization/20261021_130352_841024/cross_version/test_plot_comparison.py.json)、
[CSV 兼容扫描](../results/optimization/20261021_130352_841024/historical_csv_compatibility.json)。

## 4. 最终回归与复测

新增预算及绘图用例已接入原 `scripts/test_legacy_fixes.py`，避免日后只运行旧 16 项而漏掉新增回归。

| 测试入口 | 函数 | 模块 | 联调 | 合计 |
|---|---:|---:|---:|---:|
| `scripts/test_legacy_fixes.py` | 16 | 16 | 3 | 35 |
| `scripts/test_research.py` | 31 | 34 | 9 | 74 |
| **总计** | **47** | **50** | **12** | **109** |

最终 109 项全部通过，失败/错误/跳过均为 0。跨版本重复执行不重复计入此数量。
真实旧 TG 额外完成 2×80 时隙训练；仅步数入口和 CMD 各完成 2×4 时隙预算验证。
这些是工程回归，不代表收敛、创新收益或 GPU 资源验证。

证据：[35 项与真实训练](../results/optimization/20261021_130352_841024/final_legacy_regression/20261021_131109_966533/test_report.json)、
[74 项研究回归](../results/optimization/20261021_130352_841024/final_research_regression/20261021_130939_462664/test_report.json)、
[本轮汇总](../results/optimization/20261021_130352_841024/optimization_summary.json)。

项目根目录复测：

```powershell
& D:\python3.6.8\python.exe scripts/test_legacy_fixes.py
& D:\python3.10.9\python.exe scripts/test_research.py
```

## 5. 保留事项

历史 README 备份、旧实验、2026-10-20 交付压缩包与验收报告保留；旧包是当日快照，未覆盖成最新源码。
本轮不安装或升级依赖，不声称 CUDA 可用，不上调历史项目评分。
研究 PPO、正式 WM/PM、完整 checkpoint/恢复、统一实验等仍需逐步实现和单独验收。
