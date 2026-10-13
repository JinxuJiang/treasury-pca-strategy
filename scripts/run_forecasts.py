"""Run leakage-safe expanding PCA and daily Ridge forecasts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.walk_forward import run_monthly_expanding_forecast  # noqa: E402


def evaluation_table(predictions: pd.DataFrame, retained_components: int) -> pd.DataFrame:
    """Summarize completed daily out-of-sample Ridge forecasts."""
    rows = []
    for pc_index in range(1, retained_components + 1):
        pc = f"PC{pc_index}"
        completed = predictions[[f"actual_{pc}", f"ridge_{pc}"]].dropna()
        actual = completed[f"actual_{pc}"].to_numpy()
        predicted = completed[f"ridge_{pc}"].to_numpy()
        rows.append(
            {
                "component": pc,
                "observations": len(completed),
                "MAE": float(np.mean(np.abs(actual - predicted))),
                "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
                "correlation": float(np.corrcoef(actual, predicted)[0, 1]),
                "direction_accuracy": float(
                    np.mean(np.sign(actual) == np.sign(predicted))
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # yaml.safe_load() makes the yaml file as a dictionary
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    forecast_config = config["forecast"]
    maturities = config["data"]["maturities"]

    processed_dir = PROJECT_ROOT / paths["processed_dir"]
    output_dir = PROJECT_ROOT / paths["forecast_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    changes = pd.read_csv(
        processed_dir / "treasury_yield_changes_bp.csv", parse_dates=["date"]
    ).set_index("date")
    levels = pd.read_csv(
        processed_dir / "treasury_yields.csv", parse_dates=["date"]
    ).set_index("date")
    change_columns = [f"d_{maturity}_bp" for maturity in maturities]
    shared_dates = changes.index.intersection(levels.index)
    changes = changes.loc[shared_dates, change_columns].sort_index()
    levels = levels.loc[shared_dates, maturities].sort_index()

    retained_components = int(config["pca"]["retained_components"])
    result = run_monthly_expanding_forecast(
        changes=changes,
        levels=levels,
        maturities=maturities,
        first_oos_date=pd.Timestamp(forecast_config["first_oos_date"]),
        horizon_days=int(forecast_config["horizon_days"]),
        retained_components=retained_components,
        validation_years=int(forecast_config["validation_years"]),
        ridge_alphas=[float(value) for value in forecast_config["ridge_alphas"]],
    )

    result.predictions.to_csv(
        output_dir / "daily_oos_predictions.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    result.model_diagnostics.to_csv(
        output_dir / "model_diagnostics.csv", index=False, date_format="%Y-%m-%d"
    )
    result.pca_loadings.to_csv(
        output_dir / "expanding_pca_loadings.csv", index=False, date_format="%Y-%m-%d"
    )
    metrics = evaluation_table(result.predictions, retained_components)
    metrics.to_csv(output_dir / "oos_metrics.csv", index=False)

    print(
        f"Saved {len(result.predictions):,} daily OOS predictions "
        f"from {result.predictions['prediction_date'].min().date()} "
        f"to {result.predictions['prediction_date'].max().date()}."
    )
    print(f"Outputs: {output_dir}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
