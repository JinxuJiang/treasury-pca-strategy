# 期货策略运行与回测说明

所有命令均在项目根目录执行。安装和快速运行见[项目README](../README.md)。

已有数据和预测时，一键更新两组仓位与回测，并汇总四套独立账户：

```bash
.venv/bin/python scripts/run_pipeline.py
```

完整刷新数据及预测：追加`--refresh-data --refit`；仅展示命令：追加`--dry-run`。默认只重跑信号和回测。最终对比在results/backtest/comparison_regression/。中途失败停止，前面已完成的输出可能已更新，旧最终汇总不代表本轮成功。

分别运行时，先准备数据与已有五日预测，再生成目标持仓，最后运行Backtrader。

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/run_forecasts.py --model ridge
.venv/bin/python scripts/generate_signals.py --config config/backtest_capacity.yaml
.venv/bin/python scripts/run_backtest.py --config config/backtest_capacity.yaml
.venv/bin/python scripts/generate_signals.py --config config/backtest_curve.yaml
.venv/bin/python scripts/run_backtest.py --config config/backtest_curve.yaml
```

两个回测入口默认使用PC配置；更改配置后必须重新生成目标持仓。

- PC组：PC1、PC2、PC3各自独立账户，输出在 results/signals/ridge_capacity_regression/ 和 results/backtest/ridge_capacity_regression/。
- 统一曲线：单个账户，输出在对应 ridge_curve_regression/。
- 当前两组初始资金均1,000万美元、五日标准差预算1%、保证金占用上限50%。
- factor_positions.py负责PC目标与共享回归、日历、整数定仓；curve_positions.py提供统一曲线风险权重。
- 周末最后交易日生成目标，下个交易时段开盘成交；订单为目标与当前持仓差额。
- 费用按每份每边2.5美元加半tick滑点费用；缺行情时只沿用估值、不成交。
- 不按日历剔除换月窗口，保留回归残差异常过滤。没有具体合约换月交易、现金利息或完整追保强平模拟。
- Yahoo连续序列选择和调整尚未验证，回归敏感度不是CTD定价DV01，当前是研究回测。

输出包含summary.csv、equity_comparison.png，以及各策略目录下的equity.csv、orders.csv、positions.csv、data_quality.csv、metrics.json。manifest.json保存配置和输入哈希。

详细方法见 regression_sensitivity.md。历史V1说明与当前回归版本分开阅读。

脚本 audit_backtest.py 已删除；独立盈亏记账检查保留在 tests/test_backtest_audit.py，直接检查实际Backtrader执行结果。

```bash
.venv/bin/python -m pytest tests -q
```
