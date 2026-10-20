# Futures portfolio execution and backtesting

Run all commands from the repository root. See the [project README](../README.md) for setup.

With data and forecasts available:

```bash
.venv/bin/python scripts/run_pipeline.py
```

Add `--refresh-data --refit` for a full refresh, or `--dry-run` to preview commands. The default only regenerates targets and backtests. Combined results are written to `results/backtest/comparison_regression/`. Execution stops on failure; earlier outputs may already have changed, and an existing summary can belong to an older run.

Individual commands:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/run_forecasts.py --model ridge
.venv/bin/python scripts/generate_signals.py --config config/backtest_capacity.yaml
.venv/bin/python scripts/run_backtest.py --config config/backtest_capacity.yaml
.venv/bin/python scripts/generate_signals.py --config config/backtest_curve.yaml
.venv/bin/python scripts/run_backtest.py --config config/backtest_curve.yaml
```

Both backtest entry points default to the PC configuration. Regenerate targets after configuration changes.

- PC1, PC2 and PC3 are separate accounts. Signals and backtests use the `ridge_capacity_regression` directories.
- The unified curve strategy is a single account, using the `ridge_curve_regression` directories.
- Each account starts with USD 10 million, a 1% five-day P&L standard-deviation budget and a 50% margin-use cap, based on initial capital.
- `factor_positions.py` defines PC targets and shared regression, scheduling and integer sizing. `curve_positions.py` defines unified curve risk weights.
- Targets are formed at the final weekly session close; market orders execute at the next available session open. Orders adjust target minus actual positions.
- Commission is USD 2.50 per contract per side plus half a tick charged as a dollar cost. Missing bars carry valuation only and prevent basket fills.
- Calendar-based roll exclusions are disabled; residual outlier filtering remains. Explicit rolls, cash interest and full margin-call/liquidation procedures are not simulated.
- Continuous-contract selection and adjustment are unverified. Empirical sensitivities are not CTD valuation DV01.

Outputs include `summary.csv`, `equity_comparison.png`, and per-strategy `equity.csv`, `orders.csv`, `positions.csv`, `data_quality.csv` and `metrics.json`. Manifests record configuration and input hashes.

See [regression position sizing](regression_sensitivity.md) for equations and filtering rules. The standalone audit script was removed; `tests/test_backtest_audit.py` independently checks actual Backtrader accounting.

```bash
.venv/bin/python -m pytest tests -q
```
