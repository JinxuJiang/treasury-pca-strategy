# 滚动回归配仓（2026-10-19）

当前两份回测配置使用 sensitivity.method: rolling_regression。当前仅保留两个业务脚本：factor_positions.py（PC策略及共享回归、日历、定仓引擎）和curve_positions.py（统一曲线规则与入口）。旧平价债券配仓实现及regression_positions.py已删除。新信号和回测写入带 _regression 的目录。

## 回归与配仓

每个信号日，以当日及之前最近756个收益率观察日为窗口，联合回归四种期货的每日美元盈亏（收盘差乘合约乘数），解释变量是2Y、5Y、10Y、30Y收益率的日变化，含截距。beta矩阵每行为一种期货，每列为一个收益率节点，单位美元/bp/份；这不是CTD定价DV01。截距只用于估计，不计为可交易预测收益。

带方向收益率风险 e = -beta.T @ 合约份数；PC暴露 = loading @ e。PC策略求解目标PC方向、其余PC为零的连续篮子；曲线策略保留原先 -预测/五日收益率标准差 的目标节点风险方向，通过完整矩阵换算份数，而非要求每个期货的多空方向与同名期限一一对应。按实际每份期货历史五日美元盈亏协方差缩放整体仓位，检查邻近81个整数篮子、风险、保证金、份数、经验总绝对敏感度及PC非目标暴露限制。

原 gross_dv01_limit 字段为兼容保留，现在约束 sum(abs(份数) * 每份各节点beta绝对值之和)，不是CTD DV01。它避免各合约间净额抵消掩盖总敞口。审计输出beta各项、各合约R²、样本数、历史截止日期和组合风险。

## 清洗与失败处理

按用户要求（2026-10-20），已删除全部按日历排除换月窗口的逻辑；季度月份的正常记录与其他日期同等参与回归。仅用迭代残差MAD（6倍，三轮）剔除统计异常记录，这不等于确认数据源已处理换月。价格缺失不填充、不跨缺失收益率日期拼接；仅完整连续五观察日窗口进入风险协方差。

至少126条清洁日样本、60条五日盈亏窗口，每种期货R²至少0.30，收益率设计矩阵及beta条件数不超过10000；失败周空仓并记录原因，不回退平价债券。门槛是研究参数，不是已证明最优标准。

过滤会排除部分真实行情，也可能遗漏非窗口换月跳变；清洁样本风险可能低估极端行情风险。Yahoo合约选择和交易时段对齐仍未验证，日线不同收盘时点可能影响回归。正式验证需要具体合约价格与换月映射。本次回测仍使用原始连续序列计算实际收益，不会把过滤后的价格替换成交价格。

## 运行

```bash
.venv/bin/python scripts/generate_signals.py --config config/backtest_capacity.yaml
.venv/bin/python scripts/run_backtest.py --config config/backtest_capacity.yaml
.venv/bin/python scripts/generate_signals.py --config config/backtest_curve.yaml
.venv/bin/python scripts/run_backtest.py --config config/backtest_curve.yaml
.venv/bin/python -m pytest tests -q
```

## 当前配置与结构

PC组和统一曲线组各自初始资金1,000万美元，五日盈亏标准差预算均为1%，保证金占用上限50%。风险基数为初始资金，不随权益复利调整。

当前仅使用backtest_capacity.yaml和backtest_curve.yaml，旧小账户配置及平价债券字段已删除。信号入口分别调用generate_weekly_signals和generate_curve_signals；曲线模块复用factor_positions的公共引擎，不运行PC策略。

旧版实验、冻结快照及其说明已按用户要求删除。当前阶段结果见项目README与docs/results/；完整本地信号、交易和回测结果在results/。本次最终重跑已采用删除日历换月窗口后的逻辑，仍保留残差异常值过滤。
