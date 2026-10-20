# Rolling-regression portfolio sizing

Both configurations use `sensitivity.method: rolling_regression`. `factor_positions.py` contains the PC strategy and shared regression, calendar and basket-sizing engine; `curve_positions.py` defines unified curve targets. The old par-bond allocation and separate regression module were removed. Current output directories end in `_regression`.

## Regression and allocation

At each signal date, use up to the latest 756 yield observations through that date. Regress each futures contract's daily dollar P&L (close difference times multiplier) on daily 2Y, 5Y, 10Y and 30Y yield changes, including an intercept. Rows of beta represent contracts; columns represent yield nodes. Units are USD per bp per contract. These are empirical sensitivities, not CTD valuation DV01. Intercepts are fitted but not used as tradable expected returns.

For contract quantities `n`, signed long-bond-style node exposures are `e = -beta.T @ n`; PC exposures are `loading @ e`. PC portfolios solve for the forecast direction of one PC with zero ideal exposure to the others. Curve portfolios target `-forecast / historical five-day yield standard deviation`. The full sensitivity matrix maps those targets to quantities; individual contracts need not have the direction of their similarly named yield node.

Scale quantities using historical five-day per-contract dollar P&L covariance. Search 81 neighboring integer baskets, enforcing risk, margin, quantity, gross-sensitivity and non-target-PC limits.

The retained configuration name `gross_dv01_limit` caps `sum(abs(quantity) * sum(abs(contract beta across nodes)))`, not CTD DV01. This avoids hiding gross exposure through offsetting positions. Diagnostics record beta, contract R-squared values, sample counts, historical end dates and portfolio risk.

## Filtering and failure handling

No dates are excluded solely because of a quarterly roll calendar. Three rounds of six-MAD residual filtering remove statistical outliers, followed by a final regression on retained observations. This does not establish that the data provider has removed roll gaps. Missing observations are not filled; only complete five-observation P&L windows enter covariance estimation.

Require at least 126 clean daily observations, 60 five-day P&L windows, R-squared of at least 0.30 for every contract, and condition numbers no greater than 10,000 for both the centered yield design and beta. Failed estimates produce zero targets with a recorded reason; there is no fallback to par-bond DV01. Thresholds are research choices, not validated optima.

Filtering may remove genuine market moves and miss roll artifacts, potentially understating tail risk. Yahoo contract selection and session alignment remain unverified; different daily close timestamps may distort estimates. Formal validation needs individual contract prices and a roll map. Execution P&L uses the original continuous prices, not a filtered replacement price series.

## Running

```bash
.venv/bin/python scripts/generate_signals.py --config config/backtest_capacity.yaml
.venv/bin/python scripts/run_backtest.py --config config/backtest_capacity.yaml
.venv/bin/python scripts/generate_signals.py --config config/backtest_curve.yaml
.venv/bin/python scripts/run_backtest.py --config config/backtest_curve.yaml
.venv/bin/python -m pytest tests -q
```

## Configuration and status

Each independent account starts with USD 10 million. Both strategy families use a 1% five-day P&L standard-deviation budget and a 50% margin-use cap, based on initial capital rather than compounding equity.

Current configurations are `backtest_capacity.yaml` and `backtest_curve.yaml`. The small-account configuration and par-bond fields were removed. Signal entry points call `generate_weekly_signals` or `generate_curve_signals`; the curve strategy reuses the shared engine without running PC strategies.

Obsolete experiments and frozen snapshots were removed. The README and `docs/results/` contain published stage results; full current local outputs remain in `results/`. The published run uses no calendar-based roll exclusions and retains residual outlier filtering.
