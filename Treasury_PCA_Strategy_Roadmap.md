# Treasury Yield-Curve PCA Strategy Roadmap

## 1. Project objective

Build an interpretable systematic interest-rate strategy with the following chain:

```text
Treasury yield-curve data
    -> PCA factor extraction
    -> ML forecasts of future PC movements
    -> reconstruction of maturity-level yield changes
    -> mapping into Treasury futures positions
    -> DV01 and factor-risk control
    -> P&L attribution
```

The first version focuses on the U.S. Treasury curve. It does **not** combine Fed Funds Futures, interest-rate options, HJM calibration, or complicated bond relative-value trades.

The principal research question is:

> Can Treasury yield-curve factors be forecast out of sample, and can those forecasts generate economically meaningful returns after risk adjustment and transaction costs?

---

## 2. Separate the data, prediction target, and trading product

These are three different objects:

| Layer | First-version choice | Purpose |
|---|---|---|
| Market data | Fixed-maturity Treasury yields | Describe the yield curve |
| Prediction target | Future PCA factor changes | Forecast curve movements |
| Trading product | Treasury futures | Express the forecast in a tradable portfolio |

### 2.1 Treasury yield data

Start with daily yields at:

```text
2Y, 5Y, 10Y, 30Y
```

The preferred research source is the Federal Reserve's Gürkaynak-Sack-Wright nominal zero-coupon yield curve. A simpler initial alternative is the U.S. Treasury daily par-yield curve.

Calculate yield changes in basis points:

$$
\Delta y_t = y_t-y_{t-1}.
$$

For a five-day forecast horizon, the target must be the cumulative future change:

$$
\Delta_h y_{t+h}=y_{t+h}-y_t.
$$

### 2.2 Treasury futures

The strategy will eventually be mapped into Treasury futures associated with the 2Y, 5Y, 10Y, and long-bond sectors.

Treasury futures track government-bond prices. Their prices generally move inversely to Treasury yields:

$$
\text{Yield rises} \Rightarrow \text{Treasury futures price falls}.
$$

### 2.3 Fed Funds Futures are not part of Version 1

Fed Funds Futures reflect the market-implied average Effective Federal Funds Rate for a future calendar month. They are mainly instruments for trading the Federal Reserve policy path and the very short end of the rate curve.

They are not the same as Treasury futures and should be treated as a separate future strategy:

```text
Fed Funds/SOFR data
    -> forecast the policy path
    -> trade short-rate futures
```

---

## 3. PCA representation of the Treasury curve

### 3.1 Yield-change matrix

With $T$ observations and four maturities, construct:

$$
X =
\begin{pmatrix}
\Delta y_{2Y,1} & \Delta y_{5Y,1} & \Delta y_{10Y,1} & \Delta y_{30Y,1} \\
\Delta y_{2Y,2} & \Delta y_{5Y,2} & \Delta y_{10Y,2} & \Delta y_{30Y,2} \\
\vdots & \vdots & \vdots & \vdots \\
\Delta y_{2Y,T} & \Delta y_{5Y,T} & \Delta y_{10Y,T} & \Delta y_{30Y,T}
\end{pmatrix}.
$$

Each row is one day's curve movement. Each column is one maturity.

Center each column using training-sample statistics. All preprocessing parameters must be estimated without future information.

### 3.2 Covariance matrix and eigendecomposition

Estimate:

$$
\Sigma_y = \frac{1}{T-1}X^\top X,
$$

then decompose:

$$
\Sigma_y = V\Lambda V^\top.
$$

- $V$ contains the PCA loadings.
- $\Lambda$ contains the variance explained by each component.
- Each loading vector has four entries, one for each maturity.
- Each PC score on a given date is a single number.

### 3.3 Economic interpretation

The first three components will normally resemble:

| Component | Typical loading shape | Interpretation |
|---|---|---|
| PC1 | Similar sign and size across maturities | Level / overall rate movement |
| PC2 | Opposite behavior at short and long maturities | Slope / steepening and flattening |
| PC3 | Middle differs from the two ends | Curvature |

The sign of a PCA vector is arbitrary. Therefore, never assume that a positive PC2 forecast automatically means steepening. Inspect the fitted loading signs and reconstruct maturity-level forecasts before deciding the trade.

### 3.4 Historical PC scores

Historical factor scores are:

$$
F=XV.
$$

For one observation:

$$
\Delta PC_t=V^\top\Delta y_t.
$$

These historical scores become the targets used to train the forecasting models.

---

## 4. Machine-learning forecast

### 4.0 Supervised-learning data structure

The ML table has one row per date rather than one row per date-security pair. A simplified feature matrix is:

| Date | PC1 lag 1 | PC1 momentum 5D | PC2 lag 1 | PC2 momentum 5D | PC3 lag 1 | 10Y-2Y slope | PC1 volatility 20D | Equity return 5D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| $t_1$ | value | value | value | value | value | value | value | value |
| $t_2$ | value | value | value | value | value | value | value | value |

The same feature matrix $X$ is paired with three different labels:

$$
y_{1,t}=\sum_{j=1}^{h}\Delta PC_{1,t+j},\qquad
y_{2,t}=\sum_{j=1}^{h}\Delta PC_{2,t+j},\qquad
y_{3,t}=\sum_{j=1}^{h}\Delta PC_{3,t+j}.
$$

With fixed PCA loadings, these labels can equivalently be produced by projecting the cumulative future yield change $y_{t+h}-y_t$ into PCA space.

The first implementation trains three single-target models:

```python
model_pc1.fit(X_train, y_pc1_train)
model_pc2.fit(X_train, y_pc2_train)
model_pc3.fit(X_train, y_pc3_train)
```

All three models use the same complete feature set. A PC1 model may use lagged PC2 and PC3 information; "separate models" does not mean restricting each model to its own factor history. Only the labels differ.

This resembles training separate equity models for 5-day and 20-day returns with the same features, except that the labels here differ primarily by **economic object** (PC1, PC2, PC3) rather than forecast horizon. If three factors and two horizons are tested, there are potentially six factor-horizon targets. Version 1 fixes one horizon to avoid unnecessary model proliferation.

The important difference from an equity cross-sectional dataset is sample structure:

- equity cross section: many securities for the same date, so observations are approximately date x security;
- PCA factor forecasting: one curve observation per date, so observations primarily come from time.

The effective sample size is therefore much smaller than in an equity cross-sectional model. Model complexity and feature count must be controlled accordingly.

### 4.1 Prediction horizon

Begin with a five-trading-day horizon rather than immediately attempting one-day prediction. The horizon must match:

- the ML target;
- the position holding period;
- the P&L measurement period;
- the assumed transaction costs.

### 4.2 Targets

Initially forecast:

$$
\widehat{\Delta PC}_{1,t\rightarrow t+h},\qquad
\widehat{\Delta PC}_{2,t\rightarrow t+h},\qquad
\widehat{\Delta PC}_{3,t\rightarrow t+h}.
$$

PC1, PC2, and PC3 may be modelled separately for diagnostic purposes, but the forecasts can later be combined into one portfolio.

The intended Version 1 structure is **three targets, three single-target models, one shared feature matrix**. A multi-output model is a later benchmark, not the initial design.

Start by testing PC1 and PC2. Add PC3 only if it demonstrates stable out-of-sample predictability or improves the final portfolio.

### 4.3 Initial feature set

Keep the first feature set small:

- lagged PC changes;
- current level, slope, and curvature;
- recent yield momentum and volatility;
- equity-index returns and volatility;
- later: inflation, labor-market, growth, and policy-expectation variables.

Macro data must eventually be aligned by release date and data vintage. Revised historical values must not be treated as information that was available in real time.

### 4.4 Validation

Use chronological train/validation/test periods or walk-forward validation. Do not randomly shuffle time-series observations.

Report for each PC:

- out-of-sample $R^2$;
- MAE and RMSE;
- sign accuracy;
- correlation between forecast and realization;
- performance stability across economic regimes.

PCA explains historical variance; it does not guarantee that the factor is predictable.

---

## 5. Reconstruct maturity-level yield forecasts

If the model predicts the factor vector

$$
\widehat f_{t+h}=
\begin{pmatrix}
\widehat{\Delta PC_1}\\
\widehat{\Delta PC_2}\\
\widehat{\Delta PC_3}
\end{pmatrix},
$$

reconstruct the yield changes using:

$$
\widehat{\Delta y}_{t+h}=V\widehat f_{t+h}.
$$

Equivalently:

$$
\widehat{\Delta y}
=l_1\widehat{\Delta PC_1}
+l_2\widehat{\Delta PC_2}
+l_3\widehat{\Delta PC_3}.
$$

For each maturity, add the contributions from all retained PCs. The result is:

$$
\widehat{\Delta y}_{t+h}=
\begin{pmatrix}
\widehat{\Delta y}_{2Y}\\
\widehat{\Delta y}_{5Y}\\
\widehat{\Delta y}_{10Y}\\
\widehat{\Delta y}_{30Y}
\end{pmatrix}.
$$

This makes the forecast economically interpretable. For example:

| Predicted movement | Interpretation |
|---|---|
| All yields rise similarly | Level rises; negative duration view |
| Short yields rise more than long yields | Bear flattening |
| Long yields rise more than short yields | Bear steepening |
| Middle maturity moves differently from both ends | Curvature movement |

---

## 6. Convert yield forecasts into product forecasts

For a long fixed-income position, the first-order P&L approximation is:

$$
\widehat{P\&L}_i\approx-N_i\,DV01_i\,\widehat{\Delta y_i},
$$

where:

- $N_i>0$ is a long position;
- $N_i<0$ is a short position;
- $DV01_i$ is the dollar value of a one-basis-point yield change;
- $\widehat{\Delta y_i}$ is measured in basis points.

Therefore:

| Forecast | Basic product direction |
|---|---|
| Yield expected to fall | Long the corresponding Treasury future |
| Yield expected to rise | Short the corresponding Treasury future |

For actual Treasury futures, contract BPV/DV01 must eventually reflect the cheapest-to-deliver bond and conversion factor.

---

## 7. Position construction

### 7.1 Version 1: simplified rule-based portfolio

Before building a full optimizer:

1. Convert each maturity's forecast into a long/short direction.
2. Scale the signal by forecast magnitude.
3. Divide by the contract's DV01 so that different maturities have comparable rate risk.
4. Apply a maximum position and total-risk limit.
5. Scale the complete portfolio to a target volatility.

A starting signal can be:

$$
s_i=-\frac{\widehat{\Delta y_i}}{\widehat\sigma(\Delta y_i)}.
$$

The negative sign reflects the inverse relationship between yield changes and bond-price returns.

### 7.2 Version 2: unified portfolio optimizer

The optimizer is retained for a later version. Its purpose is to choose all maturity positions jointly rather than treating PC1 and PC2 as permanently separate trading systems.

Define dollar-duration exposure:

$$
x_i=N_iDV01_i.
$$

Portfolio P&L is approximately:

$$
\Delta P\approx-x^\top\Delta y.
$$

A basic objective is:

$$
\max_x\left[
-x^\top\widehat{\Delta y}
-\frac{\lambda}{2}x^\top\Sigma_yx
\right].
$$

The optimizer balances expected return against risk while respecting constraints such as:

- maximum gross and net DV01;
- maximum exposure at each maturity;
- maximum PC1, PC2, and PC3 exposure;
- target portfolio volatility;
- leverage limits;
- turnover and estimated transaction costs.

The final output is a single position vector:

$$
N_t=(N_{2Y},N_{5Y},N_{10Y},N_{30Y}).
$$

The portfolio may simultaneously contain level and slope exposure. It does not need to be labelled as only one trade.

---

## 8. Risk management

### 8.1 Net DV01

$$
\text{Net DV01}=\sum_iN_iDV01_i.
$$

This approximates the portfolio P&L from a one-basis-point parallel curve move.

### 8.2 Key-rate DV01

Track exposure separately at each curve point:

$$
(KRD_{2Y},KRD_{5Y},KRD_{10Y},KRD_{30Y}).
$$

A portfolio can have zero net DV01 while retaining substantial slope or curvature risk.

### 8.3 PCA factor exposure

Given dollar-duration vector $x$ and loading vector $l_k$, factor exposure is:

$$
B_k=x^\top l_k.
$$

Track:

- $B_1$: level/duration exposure;
- $B_2$: slope exposure;
- $B_3$: curvature exposure.

Before placing a trade, record which risks the resulting portfolio actually carries.

### 8.4 Volatility targeting

Estimate portfolio volatility from historical or modelled P&L and scale positions:

$$
\text{Scale}_t=
\frac{\sigma_{\text{target}}}
{\widehat\sigma_{\text{portfolio},t}}.
$$

Always combine volatility targeting with a hard leverage or DV01 cap so that an unusually low volatility estimate cannot create excessive leverage.

### 8.5 Additional controls for later versions

- transaction costs and bid-ask spreads;
- futures roll schedule;
- liquidity limits;
- stress tests for parallel, steepening, flattening, and curvature shocks;
- model uncertainty and forecast decay;
- exposure limits around major macroeconomic releases and FOMC meetings.

---

## 9. P&L attribution

Forecasts may be combined into one portfolio, but prediction quality and realized P&L should still be decomposed by factor.

The approximate factor P&L is:

$$
P\&L\approx
-B_1\Delta PC_1
-B_2\Delta PC_2
-B_3\Delta PC_3.
$$

For every rebalance date, store:

| Category | Fields |
|---|---|
| Forecast | Predicted PC1, PC2, PC3 and reconstructed maturity changes |
| Position | Contract quantities and maturity DV01 exposures |
| Factor risk | Level, slope, and curvature exposure |
| Realization | Realized yield and PC changes |
| Attribution | P&L from each maturity and each PC |
| Costs | Turnover, commissions, spread, and roll costs |

This prevents an apparently sophisticated curve strategy from unknowingly becoming a persistent long- or short-duration position.

---

## 10. Backtest architecture and leakage controls

At each historical rebalance date:

1. Use only information available by that date.
2. Fit preprocessing and PCA on the permitted training window.
3. Generate historical PC targets using the same loading convention.
4. Fit the ML model using past observations only.
5. Predict future PC changes.
6. Reconstruct maturity-level yield forecasts.
7. Build the position using current DV01 and risk estimates.
8. Hold for the selected horizon or follow a clearly defined overlapping-position rule.
9. Measure realized P&L and transaction costs.
10. Record factor attribution and risk-limit utilization.

Important controls:

- Never fit PCA on the entire sample before backtesting.
- Never standardize using future observations.
- Avoid random train/test splits.
- Align macro variables with their actual publication dates.
- Treat overlapping five-day targets carefully when estimating statistics.
- Compare the strategy with simple no-skill baselines.

---

## 11. Research benchmarks

The PCA strategy should be compared against:

1. Zero forecast: $\widehat{\Delta y}=0$.
2. Historical-mean forecast.
3. Random walk / no-change yield forecast.
4. Simple momentum or reversal forecast.
5. Direct multi-output prediction of the four maturities.
6. PC1-only model.
7. PC1 plus PC2 model.
8. PC1, PC2, and PC3 model.

Direct maturity prediction remains an important benchmark even though PCA is the selected first approach. PCA is useful for denoising and interpretation, but it is not automatically superior.

---

## 12. Project architecture

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

### 12.1 Responsibilities of `src/`

`src/` contains reusable library code. Its modules implement individual capabilities such as downloading yields, fitting PCA, building labels, training one model, converting a signal into a position, or calculating P&L attribution.

Code belongs in `src/` when it should be callable from scripts, notebooks, tests, or another module. These files should not hard-code the complete project workflow.

Examples:

```python
fit_pca(yield_changes, n_components=3)
build_pc_targets(pc_scores, horizon=5)
train_single_target_model(X_train, y_train, config)
pnl_from_yield_change(position, dv01, yield_change_bp)
```

### 12.2 Responsibilities of `scripts/`

`scripts/` contains directly executable entry points. A script reads configuration, calls reusable functions from `src/` in the required order, and saves outputs.

```text
scripts/prepare_data.py
    -> calls src/data/* and writes processed datasets

scripts/train_models.py
    -> calls PCA, feature, target, and model functions
    -> saves fitted models and forecasts

scripts/run_backtest.py
    -> calls prediction, position, risk, backtest, and attribution functions
    -> saves positions, P&L, and performance results
```

The intended command-line workflow is:

```powershell
python scripts/prepare_data.py
python scripts/train_models.py
python scripts/run_backtest.py
```

### 12.3 Responsibilities of notebooks

Notebooks are research and visualization interfaces. They may inspect saved datasets and results, plot PCA loadings, review model diagnostics, and explain performance. Core data preparation, target construction, model training, position sizing, and backtesting logic should remain in `src/` rather than being hidden across notebook cells.

### 12.4 Other directories

- `config/`: parameters and paths, without embedding them throughout code.
- `data/raw/`: unmodified downloaded data.
- `data/processed/`: cleaned curve, PCA, and feature datasets.
- `models/fitted/`: serialized PCA and forecasting models.
- `results/`: machine-readable forecasts, positions, and P&L.
- `reports/`: charts and tables intended for human review.
- `tests/`: correctness checks, especially labels, PCA reconstruction, DV01 signs, and leakage prevention.

## 13. Development stages

### Stage A: curve and PCA notebook

- Download and clean Treasury yield data.
- Plot the curve through time.
- Calculate yield changes in basis points.
- Fit PCA on a training sample.
- Plot loadings and explained variance.
- Verify that PC1, PC2, and PC3 resemble level, slope, and curvature.
- Reconstruct historical yield changes and measure reconstruction error.

### Stage B: forecast notebook

- Select a five-day horizon.
- Construct PC targets.
- Establish simple time-series baselines.
- Train the first ML models.
- Perform walk-forward evaluation.
- Compare PC1, PC2, and PC3 predictability.

### Stage C: simplified economic backtest

- Reconstruct maturity-level yield forecasts.
- Convert yield forecasts into synthetic fixed-income returns using DV01.
- Apply basic long/short rules and volatility targeting.
- Calculate gross and net performance.
- Decompose P&L by PC factor.

### Stage D: Treasury futures implementation

- Acquire continuous Treasury futures data.
- Implement contract rolls.
- Calculate contract BPV/DV01 using CTD information.
- Map desired dollar-duration exposures into integer contract quantities.
- Include commissions, spread, slippage, and roll costs.

### Stage E: unified optimizer

- Estimate the maturity covariance matrix robustly.
- Combine all PC forecasts.
- Add DV01, factor-risk, leverage, turnover, and cost constraints.
- Compare optimized performance against the rule-based portfolio.

### Stage F: extensions

Only after the core strategy is validated:

- add carry and roll-down forecasts;
- add macroeconomic and policy-surprise features;
- test rolling or regime-dependent PCA;
- add PC3/butterfly positioning;
- develop a separate Fed Funds/SOFR policy-path strategy;
- integrate the rates strategy with the existing equity portfolio.

---

## 14. Current decisions

1. Start with the Treasury yield curve rather than Fed Funds Futures.
2. Use PCA because it provides an intuitive and interpretable curve representation.
3. Start with 2Y, 5Y, 10Y, and 30Y maturities.
4. Forecast PC factors with ML and reconstruct maturity-level yield changes.
5. Keep PC-level diagnostics, but allow forecasts to form one unified portfolio.
6. Use DV01 for product mapping, P&L approximation, and risk control.
7. Begin with a rule-based portfolio; retain the unified optimizer for a later version.
8. Postpone options, HJM calibration, complex butterflies, and policy-path futures.

---

## 15. Immediate next task

Build the **curve and PCA notebook** first. The notebook should answer:

1. What percentage of Treasury curve changes do the first three PCs explain?
2. Do their loadings look like level, slope, and curvature?
3. Are the loadings stable across different training periods?
4. How accurately can two or three PCs reconstruct the four observed maturity changes?
5. Which prediction horizon will be used for the first ML experiment?

Do not start product optimization until these questions have been answered.

---

## 16. Deferred implementation questions

The following questions have been deliberately recorded rather than decided in
advance. Review each item when the project reaches the corresponding stage.

1. **PCA basis stability and alignment** (`Stage A/B`): decide whether Version 1
   uses a PCA basis fixed from the initial training sample or a rolling basis.
   A rolling implementation must handle arbitrary loading-sign flips and possible
   rotations or swaps of nearby components before factor histories are combined.
2. **Covariance versus correlation PCA** (`Stage A`): use centered yield changes
   without variance standardization as the default economic-risk representation.
   Treat standardized/correlation PCA as a documented robustness comparison.
3. **Overlapping five-day labels and holdings** (`Stage B/C`): specify whether the
   strategy rebalances every five days or creates daily overlapping five-day
   position sleeves. Apply purging/embargo and overlap-aware inference where
   appropriate.
4. **Constant-maturity yields versus futures risk** (`Stage C/D`): label the first
   DV01 approximation as a synthetic yield-curve backtest. Real futures mapping
   must later account for CTD bonds, conversion factors, contract BPV, key-rate
   mapping, and rolls.
5. **Units and sign conventions** (`Stage A/C`): fix and test the units for yields,
   basis-point changes, PCA loadings, DV01, positions, factor exposures, and P&L.
   Include deterministic shock tests before interpreting backtest results.
6. **Economic versus statistical evaluation** (`Stage B/C`): do not rely on sign
   accuracy alone. Compare against zero, mean, AR, momentum/reversal, and direct
   maturity forecasts using out-of-sample forecast metrics as well as cost-adjusted
   P&L, turnover, drawdown, and regime stability.
