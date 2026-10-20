# Interest Rate Investment

美国国债收益率曲线研究：使用2Y、5Y、10Y、30Y收益率变化构建PCA，含宏观特征的Ridge模型按月扩展训练，每日预测未来五个观察日的变化，再构建期货组合并进行周度回测。

## 环境与依赖

建议Python 3.12，与当前开发环境一致。在项目根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

所有依赖统一放在 `requirements.txt`，包含数据获取、PCA、Ridge、LightGBM、绘图、pytest、Backtrader和交易日历。一条安装命令即可；版本范围不是完整环境锁定文件。

在项目根目录的本地`.env`中配置`FRED_API_KEY=你的key`，也可通过同名环境变量提供。该文件被Git忽略，不要提交真实密钥。

## 主流程

先准备数据和含宏观预测：

```bash
python scripts/prepare_data.py
python scripts/run_forecasts.py --model ridge
```

`prepare_data.py`依次准备国债收益率、失业率/CPI/核心PCE历史发布与修订版本特征，以及Yahoo的ZT/ZF/ZN/ZB期货日线。`run_forecasts.py`重跑历史样本外预测，不只是生成最新一天；内部每月重训。

仅更新仓位与回测，一条命令跑完四套策略（复用已有预测和本地数据）：

```bash
python scripts/run_pipeline.py
```

默认按顺序运行两组信号生成及回测，更新配置指定的结果目录，四套独立账户的对比保存到`results/backtest/comparison_regression/`，包含`summary.csv`和`equity_comparison.png`。任一步失败立即停止，不生成新的最终汇总。已有输出可能部分更新，失败时不要把旧汇总误认为本轮完成结果。

需要从数据到预测完整刷新时：

```bash
python scripts/run_pipeline.py --refresh-data --refit
```

`--refresh-data --offline --refit`可用缓存重建；`--dry-run`仅显示步骤。默认不联网、不重新训练预测模型。

也可以分别执行以下命令。PC1、PC2、PC3三套独立账户：


```bash
python scripts/generate_signals.py --config config/backtest_capacity.yaml
python scripts/run_backtest.py --config config/backtest_capacity.yaml
```

统一曲线单个账户：

```bash
python scripts/generate_signals.py --config config/backtest_curve.yaml
python scripts/run_backtest.py --config config/backtest_curve.yaml
```

两个回测入口默认使用PC配置。修改配置后必须先重新生成信号。每套账户初始资金1,000万美元，五日盈亏标准差预算为初始资金1%，固定保证金占用上限50%。

## 当前方法与代码

- `src/data/`：收益率、宏观历史版本及期货数据获取。
- `src/models/`：PCA相关的滚动预测流程与模型。
- `src/portfolio/factor_positions.py`：PC策略入口、共享回归敏感度、交易日历和整数仓位计算。
- `src/portfolio/curve_positions.py`：统一曲线规则，目标风险权重为负的预测bp变化除以对应历史五日收益率标准差，复用公共定仓流程。
- `src/backtest/futures_engine.py`：按预先生成的目标持仓模拟成交、保证金占用、费用与盈亏。

回归使用截至信号日的历史收益率变化解释每份期货美元盈亏，形成敏感度矩阵；按实际期货五日盈亏协方差控制仓位规模。不按日历排除换月窗口，保留残差异常值过滤。

每周最后一个交易时段收盘后生成目标，下个交易时段开盘模拟成交。缺行情时只沿用估值、不成交；交易份数为目标持仓与当前持仓之差。手续费每份每边2.5美元，滑点半tick以额外费用扣除。

方法细节见[回归配仓说明](docs/regression_sensitivity.md)，运行与输出见[回测说明](docs/backtest.md)。

## 数据、结果与其他入口

| 位置 | 内容 |
|---|---|
| `data/raw/`、`data/processed/` | 原始缓存与处理后数据；期货原始JSON及CSV位于各自的futures子目录 |
| `results/forecast/ridge/` | Ridge预测、指标及loading |
| `results/signals/ridge_capacity_regression/` | PC组周度目标持仓及回归诊断 |
| `results/backtest/ridge_capacity_regression/` | PC组净值、订单、持仓、绩效与图片 |
| 对应的`ridge_curve_regression/`目录 | 统一曲线目标及回测结果 |

```bash
python scripts/prepare_data.py --offline
python scripts/run_pca.py
python scripts/plot_forecast_results.py --model ridge
python -m pytest tests -q
```

`--offline`只使用已有缓存，缺失会报错。数据日期、路径和品种主要在`config/base.yaml`；宏观数据在`config/macro.yaml`。`run_pca.py`用于压缩与重构诊断，主流程无需单独运行。绘图脚本读取已保存的预测。

`audit_backtest.py`已删除，独立盈亏记账核对保留在tests中。测试统一运行`tests`目录。

## 阶段结果（2026-10-20）

当前阶段暂告一段落，保留含宏观Ridge预测、三个PC策略和统一曲线策略。下表回测区间为 **2020-01-03至2026-10-05**，四套账户各自初始资金1,000万美元；五日标准差预算1%，保证金占用上限50%。收益已扣配置中的手续费与滑点，不计现金利息。

| 策略 | 累计收益 | 年化收益 | 年化波动 | 最大回撤 | 夏普（无风险利率0） | 累计成本 |
|---|---:|---:|---:|---:|---:|---:|
| PC1 | 45.15% | 5.67% | 7.67% | -15.43% | 0.74 | $481,101 |
| PC2 | 4.52% | 0.66% | 9.02% | -15.66% | 0.12 | $2,079,180 |
| PC3 | -26.20% | -4.40% | 12.24% | -29.05% | -0.30 | $2,460,988 |
| CURVE | 54.03% | 6.60% | 8.36% | -16.75% | 0.79 | $1,385,778 |

![四套独立策略净值与回撤](docs/results/equity_comparison.png)

[机器可读汇总](docs/results/summary.csv) · [运行时间与输入哈希](docs/results/provenance.json)

本次结果不与旧代理DV01版本直接比较：配仓映射、风险预算和保证金参数均曾改变。旧版本地结果已删除，当前完整运行结果仍在results/。docs/results/是本次发布摘要，后续重跑pipeline不会自动更新它。

统一曲线与PC1在本样本中盈利，PC2成本侵蚀明显，PC3亏损。尚未完成真实换月、时间口径和风险过滤偏差的验证，不视为实盘有效性结论。后续如继续研究，优先检查回归配比稳定性、交易成本、换月与压力风险。

## 研究边界与文档

Yahoo连续序列的切换与调整规则未核实；回归敏感度不等于CTD定价DV01。缺少真实换月交易与换月成本、现金利息、完整初始/维持保证金及追保强平模拟。统计异常过滤也可能低估极端风险。当前结果是探索性研究，不能视为已验证的可交易收益。

- 详细说明：[数据与预测](docs/data_and_forecasts.md)、[回测运行](docs/backtest.md)、[回归配仓](docs/regression_sensitivity.md)。
- `Treasury_PCA_Strategy_Roadmap.md`：早期研究目标与后续方向规划，非当前任务清单或运行入口。

Git提交代码、配置、依赖与相关测试/文档；不提交本地`.env`、环境、数据、结果、缓存，以及个人研究笔记`logic.md`和`利率PCA策略_中文说明.md`。忽略规则不移除历史提交中已有的内容。当前结果摘要和对比图位于docs/results/并随仓库保存；完整数据和运行结果仍仅保存在本地。
