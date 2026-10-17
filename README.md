# Interest Rate Investment

美国国债收益率曲线研究：用 2Y、5Y、10Y、30Y 收益率变化拟合 PCA，按月扩展训练，每日预测未来五个交易日的累计 PC 变化。

## 日常运行

在项目根目录激活环境：

```bash
source .venv/bin/activate
```

在根目录 `.env` 配置 `FRED_API_KEY=你的key`（已被 Git 忽略），或设置同名环境变量。不要提交或分享真实 key。

```bash
python scripts/prepare_data.py
python scripts/run_forecasts.py --model ridge
```

- `prepare_data.py` 先准备国债，再准备并对齐失业率、CPI、核心 PCE 历史版本特征；不再需要单独运行宏观准备脚本。
- `run_forecasts.py` 只运行含宏观特征的预测，不运行无宏观对照。默认 Ridge，仍支持 `--model lgbm`。
- 新结果分别保存到 `results/forecast/ridge/` 或 `results/forecast/lgbm/`。旧实验结果保留。
- 每次预测命令重跑历史样本外预测；内部每月重训，每日预测，不是仅预测最新月份。

## 其他入口

```bash
python scripts/prepare_data.py --offline
python scripts/run_pca.py
python -m pytest -q
```

`--offline` 仅使用缓存国债 XML 和 FRED 快照，缺失缓存会报错。基础配置使用 `--config`，宏观配置使用 `--macro-config`。
`run_pca.py` 用于解释方差、重建等诊断，预测前无需单独运行。

安装依赖：`python -m pip install -r requirements.txt`；LightGBM 可选依赖见 `requirements-lgbm.txt`。

## 结果目录

- `results/pca/`：PCA 压缩与重建诊断图片。
- `results/forecast/<model>/`：当前含宏观预测、指标和诊断；Ridge 已有结果，LightGBM 运行后生成。
- `results/archive/`：历史有／无宏观对照、月频、ARMA、ARMAX、alpha=0 及对比图片，保留原始文件。

读取已保存预测生成时序图、散点图和 PC 指标图：

```bash
python scripts/plot_forecast_results.py --model ridge
```

图片写入对应模型目录；指标图显示 RMSE、相关系数和 R²。历史 manifest 中路径是原运行位置，不改写其内容。
