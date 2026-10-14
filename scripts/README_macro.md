# 宏观历史版本数据（不改变训练流程）

## 运行

在 PowerShell 当前会话设置自己的 FRED API key（不要写进代码、配置或提交）：

```powershell
$env:FRED_API_KEY = "7e7f975d8de57acb2b0f4f766727b19a"
python scripts/prepare_macro_data.py
```

也可以在项目根目录的 `.env` 保存 `FRED_API_KEY=你的key`，然后直接运行脚本。
该文件已被Git忽略；环境变量优先于本地文件。不要分享或提交真实key。

全部缓存完成后可以不联网重算：

```powershell
python scripts/prepare_macro_data.py --offline
```

## 数据流

1. `src/data/fred_vintages.py` 查询版本日期；用 `output_type=2` 获取完整快照并缓存。
2. `src/features/macro_features.py` 在每份快照内计算特征：失业率及月变化（百分点）、CPI和核心PCE同比/环比（百分比）。月份缺失不跨月代替、不插值。
3. 快照版本日期之后的第一个财政部样本交易日才可用；不是实际下载之后一天。按已可用更新向后匹配到日频。
4. 训练可以读取 `daily_macro_features.csv`，但本次尚未接入或评估预测模型。

## 输出

`data/processed/macro/`：

- `*_snapshots.csv`：快照日期、数据月份、数值；每次版本保留完整历史。
- `*_updates.csv`：每份快照的最新月份和计算特征。
- `daily_macro_features.csv`：以国债样本交易日期为索引的数值特征。
- `daily_macro_lineage.csv`：各指标的快照日期、可用日期、所属月份，仅用于审计。

## 边界

- 初始化取样本开始当天快照，保守延后到下一样本交易日，所以第一行可能缺失。不会用未来值补齐。
- ALFRED版本日期不是精确发布时间；当前采用下一交易日规则，不声称重建了官方日内发布时刻。
- 同比/环比分子分母来自同一个快照；修订不会重写此前信号日特征。
- API key 缺失会明确报错。缓存可复用；新的结束日期会查询新的版本列表，已有快照不重复下载。
- `config/macro.yaml` 的观察起点为2004年，给2006年样本留出同比计算历史。版本结束日期默认为国债样本最后日期。
- 真实历史版本覆盖、缺失值以及与官方公告的抽样核验，必须在真实下载后检查；单元测试不能证明供应商覆盖完整。
