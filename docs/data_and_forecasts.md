# 数据与预测说明

以下命令均在项目根目录执行。环境安装和完整pipeline入口见[项目README](../README.md)。

数据准备入口依次运行国债收益率、宏观历史版本和四种期货日线准备；期货源数据保留缺失，不自行拼接或复权。基础数据与期货参数在config/base.yaml，宏观参数在config/macro.yaml。

## 宏观历史版本数据

数据准备已合并到 `prepare_data.py`，先生成国债样本，再对齐宏观特征。

在根目录 `.env` 配置 `FRED_API_KEY=你的key`，或设置同名环境变量；环境变量优先。不要提交或分享真实 key。

```bash
python scripts/prepare_data.py
python scripts/prepare_data.py --offline
```

基础配置为 `config/base.yaml`，宏观配置为 `config/macro.yaml`，分别用 `--config` 和 `--macro-config` 指定。合并入口以所选基础配置中的国债输出为准。

FRED 快照缓存在 `data/raw/fred_vintages`；宏观输出在 `data/processed/macro`，包括各指标快照、更新记录、`daily_macro_features.csv` 和审计用 `daily_macro_lineage.csv`。

特征包含失业率和月变化、CPI 与核心 PCE 的同比及环比。比值使用同一快照内数据，月份缺失不跨月替代、不插值。版本日期之后的第一个样本交易日才可使用；该规则不声称重建了官方日内发布时间。

宏观日期范围参考本次国债样本；默认观察起点为 2004 年，给同比计算留出历史。离线模式不联网，缺失缓存会报错。

## 含宏观特征的日频预测

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
重跑会更新上述目录；早期有/无宏观及其他模型的旧实验结果已删除。

## LightGBM补充

Ridge是默认模型，LightGBM可用同一个预测入口选择：

```bash
python scripts/run_forecasts.py --model lgbm
```

所有依赖统一在requirements.txt。

只运行含宏观特征的预测，输出到 `results/forecast/lgbm/`，不再运行无宏观对照。早期对照结果已在本次仓库整理中删除。

每日信号、未来五日累计 PC 目标、每月 expanding 重拟合和两年验证期不变。
六组候选参数按各 PC 验证 RMSE 选择，不使用 OOS 或 early stopping 选参数。
宏观数据使用当时可用的历史版本，缺失样本排除；PC 预测还原为期限 bp 变化后评估。
