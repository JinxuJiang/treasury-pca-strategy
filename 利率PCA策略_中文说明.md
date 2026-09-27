# 利率 PCA 策略：中文项目说明

## 一、我们准备做什么？

这个项目想做的是一套可解释的美国国债利率策略：

1. 观察 2 年、5 年、10 年和 30 年美国国债收益率如何变化；
2. 用 PCA 把整条曲线的变化概括成 level、slope 和 curvature；
3. 用机器学习预测未来几个交易日这三个因子会怎样变化；
4. 把因子预测重新还原成各期限收益率预测；
5. 用 Treasury futures 表达交易观点；
6. 用 DV01 控制仓位风险；
7. 分析最终利润来自 level、slope 还是 curvature。

一句话概括：

> 先预测收益率曲线将怎样运动，再通过不同期限的美国国债期货建立对应仓位。

---

## 二、数据、预测目标和交易产品不是一回事

### 1. 用什么数据研究？

第一版使用固定期限的美国国债收益率：

- 2Y Treasury yield；
- 5Y Treasury yield；
- 10Y Treasury yield；
- 30Y Treasury yield。

模型处理的是这些收益率的每日或每周变化，而不是单只国债的成交价格。

### 2. 预测什么？

不直接预测“美联储下次是否降息”，而是预测：

- PC1：整条曲线整体上升或下降多少；
- PC2：短端与长端的相对变化；
- PC3：中间期限相对于两端的变化。

### 3. 最后交易什么？

最终计划使用不同期限的 Treasury futures 建仓，例如 2Y、5Y、10Y 和长债期货。

Fed Funds Futures 暂时不放进这个项目。它交易未来某个月的平均联邦基金有效利率，主要用于表达美联储政策路径观点，是另一个可以以后单独开发的策略。

---

## 三、PCA 在这里有什么用？

四个期限的收益率通常高度相关。如果美债市场发生一次整体抛售，2Y、5Y、10Y 和 30Y 收益率往往会一起上升，只是幅度不同。

PCA 将这些共同变化整理成少数几种标准形状：

### PC1：Level

四个期限大致同向变化。

- PC1 对应整体利率水平；
- 预测收益率整体下降，通常意味着做多 duration；
- 预测收益率整体上升，通常意味着做空 duration。

### PC2：Slope

短端和长端的变化不同。

- 短端上升得比长端多，曲线趋于变平；
- 长端上升得比短端多，曲线趋于变陡。

PC2 的正负号没有固定经济含义，必须查看实际 loading，再把预测还原为各期限变化。

### PC3：Curvature

中间期限相对于两端发生不同变化。

这类观点通常需要三个期限共同表达，也就是常说的 butterfly。第一版可以先计算和监控 PC3，但不必立即交易它。

PCA 不会自动创造赚钱信号。它只是把复杂的曲线变化整理成更容易预测、解释和风控的坐标。

---

## 四、PCA 的输入和输出

每天先计算四个期限的收益率变化：

$$
\Delta y_t=
\begin{pmatrix}
\Delta y_{2Y,t}\\
\Delta y_{5Y,t}\\
\Delta y_{10Y,t}\\
\Delta y_{30Y,t}
\end{pmatrix}.
$$

PCA 得到三条主要 loading：

$$
l_1,\quad l_2,\quad l_3.
$$

每条 loading 都包含四个数字，表示这个因子分别怎样影响 2Y、5Y、10Y 和 30Y。

某一天，每个 PC score 只是一个数字。整条曲线的变化可以近似写成：

$$
\Delta y_t
\approx
l_1\Delta PC_{1,t}
+l_2\Delta PC_{2,t}
+l_3\Delta PC_{3,t}.
$$

因此：

- loading 表示某种曲线运动的形状；
- PC score 表示当天沿着这种形状运动了多少。

---

## 五、机器学习究竟怎样预测三个 PC？

### 1. 这是时间序列监督学习，不是股票截面预测

每一行样本代表一个日期：

| 日期 | 当时可知的特征 | 未来 5 日 PC1 | 未来 5 日 PC2 | 未来 5 日 PC3 |
|---|---|---:|---:|---:|
| $t_1$ | lagged PCs、曲线形状、波动率等 | 数值 | 数值 | 数值 |
| $t_2$ | lagged PCs、曲线形状、波动率等 | 数值 | 数值 | 数值 |

它不是在同一天比较几百只股票，而是在历史时间轴上用今天的信息预测未来。

### 2. 树模型也可以做时间序列

“时间序列模型”和“树模型”不是互斥的概念。

- AR、VAR 直接描述变量随时间的动态；
- Ridge、Random Forest、XGBoost、LightGBM 使用人为构造的滞后和状态特征进行预测；
- LSTM、Transformer 则直接处理一段序列。

只要训练与验证按时间顺序进行，树模型完全可以用于时间序列预测。它不会因为使用树就自动变成截面模型。

### 3. 第一版建议：三个目标，三个模型

第一版最清楚的结构是：

$$
M_1(X_t)\rightarrow\widehat{\Delta PC}_{1,t+h},
$$

$$
M_2(X_t)\rightarrow\widehat{\Delta PC}_{2,t+h},
$$

$$
M_3(X_t)\rightarrow\widehat{\Delta PC}_{3,t+h}.
$$

三个模型可以使用同一组输入特征，而且每个模型都可以使用所有 PC 的历史信息。例如预测 PC1 时，也可以使用过去的 PC2 和 PC3。

分开建模的好处是：

- 容易看出哪个 PC 可预测；
- 不同 PC 可以选择不同参数；
- 某个 PC 没有效果时可以直接关闭；
- 便于单独进行误差和 P&L 归因。

最终交易时，不需要把它们做成三个完全独立的组合。三个预测可以一起还原为一条未来收益率曲线，再统一决定仓位。

“同一套特征，三个不同标签”的结构可以写成：

| 日期 | PC1 lag1 | PC1 mom5 | PC2 lag1 | PC2 mom5 | PC3 lag1 | 10Y-2Y | PC1 vol20 | 股票5日收益 | 未来5日PC1 | 未来5日PC2 | 未来5日PC3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $t_1$ | 数值 | 数值 | 数值 | 数值 | 数值 | 数值 | 数值 | 数值 | 标签1 | 标签2 | 标签3 |
| $t_2$ | 数值 | 数值 | 数值 | 数值 | 数值 | 数值 | 数值 | 数值 | 标签1 | 标签2 | 标签3 |

前三个模型看到的输入列完全相同：

```python
model_pc1.fit(X_train, y_pc1_train)
model_pc2.fit(X_train, y_pc2_train)
model_pc3.fit(X_train, y_pc3_train)
```

预测 PC1 的模型也可以使用 PC2、PC3 的历史信息；“三个模型”并不代表每个模型只能看自己的 PC。真正不同的是右侧标签：

$$
y_{PC1,t}=\sum_{j=1}^{5}\Delta PC_{1,t+j},
$$

$$
y_{PC2,t}=\sum_{j=1}^{5}\Delta PC_{2,t+j},
$$

$$
y_{PC3,t}=\sum_{j=1}^{5}\Delta PC_{3,t+j}.
$$

这与现有股票模型“同一套特征分别预测 5D、20D 标签”非常相似，但有一个区别：

- 股票例子主要是预测期限不同；
- 这里主要是预测对象不同，即 PC1、PC2、PC3。

因子和期限可以同时扩展。例如 3 个 PC 乘以 5D、20D 两个期限，会产生 6 个标签。但第一版固定为 5D，只训练 3 个目标，避免模型数量过早膨胀。

另一个关键区别是样本结构：

- 股票截面模型通常是“日期 × 股票”，同一天有很多资产样本；
- PC 模型通常每个日期只有一条收益率曲线，因此每个日期只有一行样本。

所以 PC 模型的有效样本量远小于股票截面模型，更需要限制特征数量和模型复杂度。这也是第一版优先使用 Ridge 和小型 LightGBM/XGBoost，而不直接使用大型深度时序模型的原因。

### 4. 推荐的模型顺序

不要一开始就上 LSTM 或 Transformer。日频利率数据的有效独立样本并没有看起来那么多，复杂深度模型很容易过拟合。

建议按下面的顺序建立 benchmark：

#### 第一层：最简单基准

- 零预测：未来 PC 变化等于 0；
- 历史均值；
- AR(1)；
- Ridge/Elastic Net。

这些模型用来判断复杂模型是否真的增加价值。

#### 第二层：树模型

- LightGBM；
- XGBoost；
- Random Forest 作为辅助比较。

树模型可以处理非线性关系和变量交互，但要严格限制深度、叶子数和特征数量。

#### 第三层：联合模型

如果单目标模型有效，再尝试：

- VAR：让三个 PC 相互预测；
- multi-output regression：一次输出三个 PC；
- multi-task neural network：共享底层特征，分别输出三个 PC。

#### 第四层：序列深度模型

只有在数据量、特征设计和 walk-forward 验证都成熟后，再测试：

- LSTM/GRU；
- Temporal Convolution；
- Transformer 类模型。

复杂模型必须战胜简单基准，才有保留价值。

### 5. 第一版推荐组合

第一版建议同时跑两类模型：

1. **Ridge**：作为稳定、容易解释的线性基准；
2. **LightGBM 或 XGBoost**：捕捉可能存在的非线性。

对 PC1、PC2、PC3 分别训练，然后比较样本外表现。若 PC3 没有稳定预测能力，就不必强制交易 PC3。

---

## 六、第一版特征可以用什么？

先保持简单：

### 因子自身信息

- PC1、PC2、PC3 的 1、2、5、10、20 日滞后变化；
- 各 PC 的短期动量；
- 各 PC 的滚动波动率；
- 各 PC 相对于滚动均值的 z-score。

### 当前曲线状态

- 2Y、5Y、10Y、30Y 当前收益率；
- 10Y-2Y slope；
- 10Y+2Y-2×5Y curvature；
- 当前曲线水平；
- 各期限近期波动率。

### 简单跨资产信息

- 股票指数近期收益；
- 股票市场波动率；
- 美元变化；
- 后续再增加宏观数据和政策预期。

第一版不要一次加入几十个宏观变量。先证明 PC 自身动态和简单市场状态是否具有预测力。

---

## 七、预测结果如何变成期限预测？

假设三个模型输出：

$$
\widehat f=
\begin{pmatrix}
\widehat{\Delta PC_1}\\
\widehat{\Delta PC_2}\\
\widehat{\Delta PC_3}
\end{pmatrix}.
$$

用 loading matrix 恢复各期限预测：

$$
\widehat{\Delta y}=V\widehat f.
$$

得到：

$$
\widehat{\Delta y}=
\begin{pmatrix}
\widehat{\Delta y}_{2Y}\\
\widehat{\Delta y}_{5Y}\\
\widehat{\Delta y}_{10Y}\\
\widehat{\Delta y}_{30Y}
\end{pmatrix}.
$$

这一步之后，我们才决定各期限 Treasury futures 做多还是做空。

---

## 八、如何从预测变成交易？

国债收益率与价格通常反向变化：

- 预测某期限收益率下降：倾向做多相应 Treasury future；
- 预测某期限收益率上升：倾向做空相应 Treasury future。

单个仓位的一阶 P&L 近似为：

$$
\Delta P_i\approx-N_iDV01_i\Delta y_i.
$$

DV01 用来统一不同期限的利率风险。不能简单地认为一份 2Y 期货与一份 30Y 期货风险相同。

第一版先使用简单仓位规则：

1. 根据预测方向决定多空；
2. 根据预测强度调整信号；
3. 用 DV01 调整不同期限的合约数量；
4. 设置最大仓位和总 DV01；
5. 最后进行目标波动率缩放。

统一组合优化器先保留，等简单策略能够正确运行、P&L 计算无误以后再加入。

---

## 九、必须记录的风险和归因

每个交易日或调仓日，需要记录：

- 三个 PC 的预测值；
- 还原后的四个期限收益率预测；
- 四种期货的目标仓位；
- 总 DV01；
- 各期限 Key Rate DV01；
- level、slope 和 curvature 暴露；
- 实际发生的 PC 变化；
- 各 PC 对当期 P&L 的贡献；
- 交易成本和换手率。

三个 PC 可以共同形成一个组合，但必须知道某一期的仓位主要在交易：

- 整体 duration；
- slope；
- curvature；
- 或者几者的混合。

---

## 十、项目代码架构

```text
Interest_rate_investment/
|
|-- README.md
|-- requirements.txt
|
|-- config/
|   `-- base.yaml
|
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- predictions/
|
|-- notebooks/
|   |-- 01_curve_exploration.ipynb
|   |-- 02_pca_diagnostics.ipynb
|   `-- 03_result_analysis.ipynb
|
|-- src/
|   |-- data/
|   |   |-- download_yields.py
|   |   |-- clean_yields.py
|   |   `-- build_dataset.py
|   |-- factors/
|   |   `-- pca_factors.py
|   |-- features/
|   |   `-- build_features.py
|   |-- models/
|   |   |-- baselines.py
|   |   |-- train_pc_models.py
|   |   `-- predict.py
|   |-- portfolio/
|   |   |-- signal_to_positions.py
|   |   |-- dv01.py
|   |   `-- risk.py
|   |-- backtest/
|   |   |-- engine.py
|   |   `-- attribution.py
|   `-- evaluation/
|       |-- forecast_metrics.py
|       `-- performance_metrics.py
|
|-- scripts/
|   |-- prepare_data.py
|   |-- train_models.py
|   `-- run_backtest.py
|
|-- tests/
|   |-- test_pca.py
|   |-- test_targets.py
|   |-- test_dv01.py
|   `-- test_no_leakage.py
|
|-- models/
|   `-- fitted/
|-- results/
|   |-- forecasts/
|   |-- positions/
|   `-- pnl/
`-- reports/
    |-- figures/
    `-- tables/
```

### `src/`：可复用的功能代码

`src/` 里的文件负责完成一个明确功能，例如：

- 下载和清洗收益率；
- 拟合 PCA、转换 scores、恢复收益率曲线；
- 构造特征和未来 PC 标签；
- 训练某一个 PC 模型；
- 把预测信号转换成仓位；
- 计算 DV01、风险和 P&L 归因。

如果某段代码需要被 script、notebook、测试或其他模块重复调用，就应该放在 `src/`。

例如：

```python
fit_pca(yield_changes, n_components=3)
build_pc_targets(pc_scores, horizon=5)
train_single_target_model(X_train, y_train, config)
pnl_from_yield_change(position, dv01, yield_change_bp)
```

`src/` 回答的是：

> 某个具体功能怎样实现？

### `scripts/`：可以直接运行的项目流程

`scripts/` 自己不重复实现 PCA、模型和回测逻辑。它负责：

1. 读取配置；
2. 按顺序调用 `src/` 中的功能；
3. 保存数据、模型和结果。

调用关系为：

```text
scripts/prepare_data.py
    -> 调用 src/data/*
    -> 保存清洗后的数据集

scripts/train_models.py
    -> 调用 PCA、特征、标签和模型模块
    -> 保存模型与预测

scripts/run_backtest.py
    -> 调用预测、仓位、风险、回测和归因模块
    -> 保存仓位、P&L与绩效结果
```

项目运行入口为：

```powershell
python scripts/prepare_data.py
python scripts/train_models.py
python scripts/run_backtest.py
```

`scripts/` 回答的是：

> 这次完整任务按照什么顺序运行？

因此可以记成：

$$
\boxed{\texttt{src/}=\text{积木},\qquad\texttt{scripts/}=\text{组装积木的流程}}
$$

### `notebooks/`：探索、画图和解释结果

Notebook 可以用来：

- 快速查看原始曲线数据；
- 画 PCA 解释度与 loading；
- 检查 PC scores；
- 展示预测效果；
- 分析回测和 P&L 归因。

但核心数据处理、标签生成、模型训练、仓位计算和回测逻辑应该放在 `src/`，避免 notebook 变成必须按神秘顺序运行的大型脚本。

### 其他目录

- `config/`：保存路径、预测期限、模型参数和风险限制；
- `data/raw/`：保存未经修改的下载数据；
- `data/processed/`：保存清洗后的曲线、PCA、特征与标签数据；
- `models/fitted/`：保存训练好的 PCA 和预测模型；
- `results/`：保存预测、仓位和 P&L；
- `reports/`：保存给人看的图表和汇总表；
- `tests/`：检查标签方向、PCA 重建、DV01 符号和数据泄漏。

## 十一、开发顺序

### 第一步：完成 PCA notebook

- 下载并清洗 2Y、5Y、10Y、30Y 收益率；
- 计算收益率变化；
- 划分训练集与测试集；
- 在训练集上拟合 PCA；
- 画出三个 loading；
- 检查解释度和曲线重建误差。

### 第二步：建立预测 benchmark

- 确定预测期限，例如未来 5 个交易日；
- 为三个 PC 构造未来累计变化目标；
- 先跑 AR(1)、Ridge；
- 再跑 LightGBM/XGBoost；
- 使用 walk-forward 方法评估。

### 第三步：完成简化经济回测

- 把 PC 预测还原成期限收益率预测；
- 用简化 DV01 计算策略收益；
- 加入最大风险和目标波动率；
- 分解 level、slope、curvature P&L。

### 第四步：接入真实 Treasury futures

- 获取连续期货价格；
- 处理换月；
- 获取或估算每份合约的 BPV/DV01；
- 后续处理 CTD 和 conversion factor；
- 加入交易成本。

### 第五步：加入统一优化器

只有在预测、产品映射和 P&L 均验证无误后，才让优化器综合考虑：

- 预期收益；
- 协方差；
- DV01；
- PC 风险；
- 换手率；
- 交易成本；
- 杠杆限制。

---

## 十二、目前的具体决定

1. 第一版研究 Treasury yield curve，而不是 Fed Funds Futures。
2. 使用 2Y、5Y、10Y 和 30Y 四个期限。
3. 对收益率变化做 PCA。
4. 先用三个独立模型预测三个 PC，输入特征可以共享。
5. 第一批模型使用 Ridge 与 LightGBM/XGBoost，暂时不使用深度时序模型。
6. 三个预测最终共同还原成一条未来曲线，并形成一个组合。
7. 第一版使用简单 DV01 调整规则，暂不立即上复杂优化器。
8. 所有结果必须使用时间顺序验证，并进行 PC 风险与 P&L 归因。

下一步只做第一件事：建立收益率曲线数据与 PCA notebook。先证明数据、loadings、scores 和曲线重建全部正确，再开始预测。

---

## 十三、已记录、以后分阶段处理的问题

这些问题暂时不要求现在做决定；到对应开发阶段时再逐项讨论和验证：

1. PCA 使用固定训练期坐标还是滚动坐标，以及滚动 loading 的符号和方向对齐；
2. 默认使用只中心化的 covariance PCA，并把标准化后的 correlation PCA 作为稳健性比较；
3. 未来 5 日标签采用每 5 日调仓还是每日建立重叠持仓，以及相应的 purge/embargo；
4. 固定期限收益率到真实 Treasury futures 的映射，包括 CTD、conversion factor、BPV 和换月；
5. 收益率、bp、loading、DV01、仓位、因子暴露和 P&L 的单位及正负号测试；
6. 同时使用样本外预测指标与扣除成本后的经济绩效，避免只依赖方向准确率。
