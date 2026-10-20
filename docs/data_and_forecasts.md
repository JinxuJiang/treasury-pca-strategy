# Data preparation and forecasting

Run commands from the repository root. See the [project README](../README.md) for installation and the complete pipeline.

Preparation covers Treasury yields, historical macro vintages and four futures daily series. Missing futures observations are retained; this code does not splice or adjust contracts. Main settings are in `config/base.yaml`, macro settings in `config/macro.yaml`.

## Historical macro vintages

`prepare_data.py` first creates the Treasury sample, then aligns macro features to its dates.

Set `FRED_API_KEY=your_key` in the root `.env` or environment; the environment takes precedence. Never commit credentials.

```bash
python scripts/prepare_data.py
python scripts/prepare_data.py --offline
```

Use `--config` and `--macro-config` to select the respective configurations. Treasury output paths follow the selected base configuration.

FRED snapshots are cached in `data/raw/fred_vintages`. Macro outputs in `data/processed/macro` include indicator snapshots, update records, `daily_macro_features.csv` and provenance in `daily_macro_lineage.csv`.

Features include unemployment and its monthly change, plus year-over-year and month-over-month CPI and core PCE changes. Ratios use observations from the same vintage. Missing months are not interpolated or replaced with other months. Vintages become usable on a subsequent sample session according to the configured lag; this is not a reconstruction of official intraday release timestamps.

The vintage range follows the Treasury sample. The default observation start is 2004 to provide history for annual changes. Offline preparation uses caches only and fails when a required cache is absent.

## Daily macro-augmented forecasts

```bash
python scripts/prepare_data.py
python scripts/run_forecasts.py --model ridge
```

Ridge is the default; select LightGBM with `--model lgbm`. Forecasting reads prepared local data without downloading it. The current workflow runs macro-augmented forecasts only; historical macro/no-macro experiments were removed.

Daily targets are cumulative PC changes over the next five observations. Models are refitted monthly on expanding history, with the most recent two years used for validation. Tuning and final PCA fits are separate. Training labels must be complete by their cutoffs. Rows with missing macro features are excluded without interpolation or changing the five-observation label horizon.

Outputs in `results/forecast/<model>/`:

| File | Contents |
|---|---|
| `daily_oos_predictions.csv` | PC predictions and realized targets |
| `model_diagnostics.csv`, `expanding_pca_loadings.csv` | Training diagnostics and PCA vintages |
| `yield_change_predictions_bp.csv` | Reconstructed five-observation yield-change forecasts |
| `pc_metrics.csv`, `yield_metrics.csv` | Evaluation on completed labels |
| `sample_mask.csv`, `manifest.json` | Sample inclusion, configuration and input hashes |

Reconstruction multiplies cumulative PC forecasts by loadings and adds five times the PCA training mean; the fourth PC forecast is zero. Reruns update these outputs.

## LightGBM

```bash
python scripts/run_forecasts.py --model lgbm
```

Dependencies are included in the single `requirements.txt`. Outputs are written to `results/forecast/lgbm/`.

LightGBM uses the same daily five-observation targets, monthly expanding refits, two-year validation period and point-in-time macro features. Six parameter candidates are selected separately for each PC using validation RMSE, without out-of-sample selection or early stopping. Missing macro rows are excluded, and reconstructed maturity-level forecasts are evaluated in basis points.
