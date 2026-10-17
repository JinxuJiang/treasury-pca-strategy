# 宏观历史版本数据

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
