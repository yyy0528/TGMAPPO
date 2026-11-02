# Buoy Network 交付索引

2026-10-21 当前源码：已新增研究 Graph-MAPPO 真实 PPO、权重保存与冻结评估；当前评分 **53/100**，
整体方案仍未通过验收，见 [C2 优化记录](docs/RESEARCH_POLICY_OPTIMIZATION.md)及
[当前评分](docs/PROJECT_SCORECARD.md)。短训效果尚不合格；WM/PM、完整恢复和正式对照实验仍缺失。
此前单点绘图、CSV 校验和旧 TG 预算修复见 [小步优化记录](docs/STABILITY_OPTIMIZATION_LOG.md)。
以下清单、压缩包、45 分及 90 项测试为 **2026-10-20 历史快照**，不代表最新源码；旧包未覆盖。

交付日期：2026-10-20（Asia/Shanghai）。验收基准：[原 README 备份](readme_before.md)及
[研究实现方案](WORLD_MODEL_PROGRESS_REWARD_PLAN.md)。备份内容与 Git 中原 `readme.md` 一致；本次不恢复或覆盖用户删除的原文件。

**交付状态：可运行研究框架与修正后的旧 TG-MAPPO；研究方案整体未完成，最终验收不通过。量化评分 45/100。**

## 交付清单

| 交付物 | 位置 | 状态 |
|---|---|---|
| 使用、运行环境与输出说明 | [DELIVERY_GUIDE.md](docs/DELIVERY_GUIDE.md) | 已补齐 |
| 最终验收及 README 对照表 | [FINAL_ACCEPTANCE_REPORT.md](docs/FINAL_ACCEPTANCE_REPORT.md) | 已补齐 |
| 未实现、失败项、原因及建议 | [UNIMPLEMENTED_AND_FAILED_TESTS.md](docs/UNIMPLEMENTED_AND_FAILED_TESTS.md) | 已补齐；本次不修复 |
| 评分规则及逐项得分 | [PROJECT_SCORECARD.md](docs/PROJECT_SCORECARD.md) | 已补齐 |
| 目录、职责、接口与 Python 入口 | [RESEARCH_ARCHITECTURE.md](docs/RESEARCH_ARCHITECTURE.md) | 已有，随包交付 |
| 定点修复及复测记录 | [TARGETED_FIX_REPORT.md](docs/TARGETED_FIX_REPORT.md) | 已有，随包交付 |
| 研究配置 | [research_v1.json](configs/research_v1.json) | WM/PM 默认关闭 |
| 测试及真实运行证据 | [本次执行索引](results/final_acceptance/20261020_210112_027890/execution_index.json)、[环境清单](results/final_acceptance/20261020_210112_027890/runtime_inventory.json) | 已生成 |
| 统一算法入口及绘图 | `run_algorithm.py`、`run_training.cmd`、`scripts/plot_comparison.py` | 八算法和三张示例图已验证 |
| 源码、文档、配置、测试和证据压缩包 | `delivery/20261020_final/buoy_network_delivery.zip` | 同目录提供 MANIFEST.json、SHA256 校验及打包验证报告 |
| policy/WM/PM 完整训练权重与恢复状态 | 方案要求的 `.pt` 文件 | 未实现，未制作占位文件 |
| 模型训练指标及 GPU 资源报告 | `model_metrics.csv`、峰值显存/耗时报告 | 未实现，不能以规则结果代替 |

## 最短复测

在项目根目录执行：

```powershell
& D:\python3.6.8\python.exe scripts/test_legacy_fixes.py
& D:\python3.10.9\python.exe scripts/test_research.py
& D:\python3.10.9\python.exe run_research.py describe
```

已运行：16 项定点测试、74 项研究回归均通过；真实旧 TG-MAPPO 完成 2×80 时隙 CPU 训练。
README 的八种算法完成短回合统一启动及 CSV/绘图验证。这些结果不表示 WM、PM 或研究训练闭环已经实现。

本交付包不附带解释器或依赖安装包，不声称已完成全新机器安装、GPU 训练或正式多种子算法比较。
