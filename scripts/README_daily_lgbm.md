# 日频 LightGBM 含宏观预测

```bash
python -m pip install -r requirements-lgbm.txt
python scripts/prepare_data.py
python scripts/run_forecasts.py --model lgbm
```

只运行含宏观特征的预测，输出到 `results/forecast/lgbm/`，不再运行无宏观对照。历史对照结果仍保留在 `results/archive/daily_lgbm_experiment/`。

每日信号、未来五日累计 PC 目标、每月 expanding 重拟合和两年验证期不变。
六组候选参数按各 PC 验证 RMSE 选择，不使用 OOS 或 early stopping 选参数。
宏观数据使用当时可用的历史版本，缺失样本排除；PC 预测还原为期限 bp 变化后评估。
