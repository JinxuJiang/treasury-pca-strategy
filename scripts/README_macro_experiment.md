# 含宏观特征的日频预测

```bash
python scripts/prepare_data.py
python scripts/run_forecasts.py --model ridge
```

默认模型是 Ridge，可用 `--model lgbm` 切换。只读取本地准备好的数据，不在预测时下载。
现在仅运行含宏观特征的模型，不再运行无宏观对照或输出零预测对照指标。

每日信号预测未来五个交易日累计 PC 变化，每月扩展历史重新拟合，最近两年验证选参。
调参 PCA 与最终 PCA 分开拟合，保留训练标签完成日期检查；宏观缺失行在训练、验证、预测中排除，不插值，不改变五日标签时间跨度。

结果保存到 `results/forecast/<model>/`：

- `daily_oos_predictions.csv`：PC 预测及实际目标。
- `model_diagnostics.csv`、`expanding_pca_loadings.csv`：训练诊断和各轮 loading。
- `yield_change_predictions_bp.csv`：还原的四期限五日变化预测。
- `pc_metrics.csv`、`yield_metrics.csv`：该模型的完整标签评估。
- `sample_mask.csv`、`manifest.json`：样本筛选、配置和输入哈希。

还原公式为累计 PC 预测乘 loading，再加五倍 PCA 训练均值，第四个 PC 的预测 score 设为零。
重跑会更新上述目录；早期 `results/archive/macro_experiment` 和 `results/archive/daily_lgbm_experiment` 对照结果保持为历史记录。
