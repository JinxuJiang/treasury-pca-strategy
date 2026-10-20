# Next development phase: contract-level futures and forecast risk

Status: planned, not implemented. Updated October 20, 2026.

The current research stage is complete. Preserve its published results as an exploratory baseline. The next phase must replace continuous-series execution with actual dated Treasury futures contracts, implement explicit rolls, and then evaluate GARCH-based risk sizing. Keep the existing macro-augmented Ridge forecasts and both strategy families initially so that implementation changes can be assessed separately.

## Implementation order

1. Acquire and validate historical individual-contract data and contract specifications.
2. Define a causal contract-selection and roll policy, then implement executable rolls.
3. Rebuild empirical sensitivities and the historical-risk baseline using contract-level data.
4. Add GARCH volatility forecasts as an alternative risk estimator.
5. Compare variants at matched risk budgets, execution rules and costs before selecting defaults.

Do not calibrate a new risk model on unexplained continuous-series roll gaps. Do not change the allocation rule, roll policy and volatility model simultaneously when evaluating improvements.

## 1. Replace continuous futures execution data

### Required data

Obtain overlapping historical contracts for ZT, ZF, ZN and ZB, identified by product and delivery month/year. Cover each strategy signal, execution and roll date, including a sufficient warm-up period.

| Dataset | Required fields |
|---|---|
| Contract identity | Stable contract ID, product, delivery month/year, exchange |
| Daily prices | Session/trade date, open, high, low, close, settlement where available, volume, open interest where available |
| Lifecycle | Listing date, first notice date, last trading date, delivery dates and applicable trading calendar |
| Specifications | Contract multiplier, quotation conversion, minimum tick and effective dates of specification changes |
| Data provenance | Provider, retrieval time, timezone, timestamp/settlement conventions, missing-data status and file hashes |

Document access, licensing, historical coverage and revisions before selecting a source. No particular provider or historical availability is assumed in this plan. Convert Treasury fractional quotations consistently into decimal price points. Do not silently treat settlement and last-trade close as interchangeable.

### Changes

- Replace the execution inputs currently produced by `src/data/futures_prices.py` with contract-level acquisition and validation.
- Update `scripts/prepare_data.py` and configuration to prepare overlapping maturities rather than only four continuous series.
- Represent identity explicitly, such as `(product, contract_id, session_date)`, rather than one price column per product without contract metadata.
- Retain immutable raw responses and validated processed tables. Do not use interpolated prices for fills.
- Continuous or adjusted series may remain optional visualization/research aids with documented construction; they must not supply executable fills or cross-contract P&L.

### Acceptance criteria

- Every fill references a real dated contract and an observed price.
- Lifecycle and price coverage checks fail explicitly on unusable data.
- Dataset dates and publication times support the signal/execution chronology, including holiday and overnight sessions.
- Contract multipliers, tick conversions and dollar P&L are covered by deterministic tests.

## 2. Implement contract selection and rolls

### Policy to decide and freeze before testing

Choose one explicit policy, such as a fixed number of exchange sessions before first notice/last trade, or a volume/open-interest crossover with a lifecycle safety deadline. Record the exact rule and parameters in configuration. A crossover determined using a session's completed volume can only affect later execution; it must not select that same session's opening contract retrospectively.

- Determine the active dated contract for each product at each decision time.
- Roll before notice/delivery exposure and last-trade deadlines, using the applicable historical calendar.
- Specify roll execution timing, four-leg coordination, partial fills and handling of missing old/new contract prices.
- Decide whether to recompute target quantities using the incoming contract's sensitivity or retain quantities under an explicitly documented approximation.
- Distinguish signal changes from rolls, including weeks with no new prediction.

### Execution and accounting

`src/backtest/futures_engine.py` must hold and trade dated contracts:

1. Close the actual outgoing position at its observed executable price.
2. Open the incoming contract at its observed executable price.
3. Charge commissions, spread/slippage assumptions and any separately modeled execution costs on both transactions.
4. Record old/new contract IDs, roll trigger, decision time, planned/actual execution, quantities, prices, costs and delays.
5. Apply margin and available-cash checks to the transition, including temporary overlapping positions where relevant.

The difference between old and new quoted price levels is not an immediate trading profit or loss. Do not subtract a synthetic roll gap and also charge it as a loss. Attribute mark-to-market P&L within each held contract, and record execution costs separately. Do not fill at back-adjusted synthetic prices.

### Acceptance criteria

- A test with unchanged contract prices but different old/new price levels produces no artificial mark-to-market windfall on rolling; only actual modeled execution costs are charged.
- Long, short, reversal, missing-price and delayed-roll cases reconcile independently to account equity.
- No position remains beyond the configured lifecycle deadline without an explicit failure.
- Every contract-selection decision uses only information available at that decision time.

## 3. Rebuild sensitivities and the historical-risk baseline

The regression beta remains an empirical curve-sensitivity matrix, not CTD valuation DV01.

- Calculate daily dollar changes within the same dated contract; never difference an old-contract close against a new-contract close.
- Estimate the incoming contract's sensitivity using its own available history. Specify and test the policy for sparse early history; skip trading or use a clearly labeled, validated transfer estimator rather than a silent fallback.
- Recompute basket risk from economically valid per-contract or self-financing roll-aware P&L, with quantity changes and costs treated consistently.
- Review the existing residual filter. Report how exclusions change estimated risk, especially during stress; avoid assuming filtered quiet observations represent all tradable risk.
- Examine beta stability, matrix inversion sensitivity, gross offsets and turnover. High in-sample R-squared alone is insufficient.
- Retain PC1/PC2/PC3 and unified-curve targets initially, as well as current risk budgets, to isolate the data/execution change.

Acceptance: produce a new historical-volatility baseline with independent P&L reconciliation and matched-budget comparisons. Keep it distinct from the published continuous-series baseline; do not overwrite the stage snapshot without documenting the new version.

## 4. Add GARCH to risk sizing

### Purpose and integration point

GARCH forecasts conditional volatility for position sizing. It does not replace Ridge direction forecasts, PCA loadings or the regression sensitivity mapping.

Current flow:

```text
Forecast and allocation rule
    -> beta-based relative contract quantities
    -> historical five-day P&L covariance
    -> scale to risk budget
    -> integer, margin and exposure limits
```

Proposed alternative:

```text
Same forecast and allocation rule
    -> same sensitivity mapping
    -> forecast five-day P&L covariance
    -> same risk budget and position limits
```

Implement this as a selectable risk-estimation method in configuration, with historical covariance retained as the benchmark. The shared sizing code in `factor_positions.py` consumes the resulting covariance; a separate risk module is appropriate if implementation complexity warrants it.

### Initial specification

Start with a documented GARCH(1,1) candidate for each product/contract's appropriately defined P&L or return innovations. If fitting returns, convert forecast covariance back into per-contract dollar P&L units before sizing. Fit only observations available by the signal time.

For a simple initial portfolio model, combine marginal forecast variances with a trailing, positive-semidefinite correlation matrix. This does not make correlations dynamic; DCC-GARCH or other multivariate models are later alternatives, not initial requirements.

- Forecast the actual holding horizon in exchange sessions. The current fixed five-observation budget and weekly open-to-open holdings differ; resolve or explicitly retain and measure this mismatch.
- Aggregate multi-step variances under the chosen innovation assumptions; summing daily variances requires uncorrelated innovations and is not a universal rule.
- Avoid treating cross-contract level jumps as returns. Document the roll-aware series used to train the model and its relationship to executable holdings.
- Define convergence, sample-size, finite-variance and covariance-validity checks. Failures must be logged; select an explicit conservative historical fallback or zero position.
- Retain margin, quantity, gross empirical sensitivity and non-target-PC limits. A low forecast variance does not override these constraints.
- Add a package dependency only when implementing and testing the model; this document does not add one.

### Evaluation and acceptance criteria

Compare historical and GARCH risk estimates with the same forecasts, dates, budgets and costs:

- Forecast versus realized holding-period volatility and risk exceedances.
- Maximum drawdown, worst holding-period losses and stress behavior.
- Turnover, transaction costs, gross exposure, margin peaks and delayed execution.
- Performance across multiple periods, including volatile regimes.
- Sensitivity to fitting windows and model assumptions, using time-ordered validation.

Do not select a model solely for higher backtest return. Include tests that changing future inputs cannot alter earlier forecasts or targets, and verify that forecast dollar covariance yields the expected basket variance.

## 5. Additional execution and risk work

These remain separate from the three mandatory tasks above:

- Historical initial/maintenance and broker margin requirements, margin-call and forced-liquidation behavior, and portfolio offsets where supported by data.
- Cash/collateral interest and funding treatment.
- Liquidity, market impact and feasible basket execution rather than unconditional full fills.
- CTD/conversion-factor DV01 or key-rate analytics as a validation benchmark for empirical beta.
- Stress tests combining losses, correlation changes, margin increases and unavailable liquidity.
- Independent cash-flow/position/P&L reconciliation and reproducible data/parameter manifests.

## Completion checklist

- [ ] Validated historical dated-contract dataset with lifecycle metadata.
- [ ] Causal contract map and configured roll policy.
- [ ] Explicit old-contract close/new-contract open execution and both-side costs.
- [ ] No synthetic roll-gap P&L, with independent accounting tests.
- [ ] Rebuilt regression sensitivities and historical-risk baseline.
- [ ] Configurable GARCH risk forecast with horizon and unit checks.
- [ ] Matched-risk historical-versus-GARCH comparison and stress evaluation.
- [ ] Updated tests, README, provenance and a separately versioned result snapshot.

No runtime behavior changes are implemented by this plan. Resume with contract-data feasibility and coverage, not GARCH calibration.
