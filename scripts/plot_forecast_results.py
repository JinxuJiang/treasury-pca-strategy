"""Plot current daily macro-augmented forecast diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]



def plot_prediction_series(predictions: pd.DataFrame, output: Path, model: str, horizon: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    for pc_index, ax in enumerate(axes, start=1):
        pc = f"PC{pc_index}"
        ax.plot(
            predictions["prediction_date"],
            predictions[f"actual_{pc}"],
            color="black",
            linewidth=0.9,
            label=f"Actual future {horizon}-day PC change",
        )
        ax.plot(
            predictions["prediction_date"],
            predictions[f"{model}_{pc}"],
            color="tab:blue",
            linewidth=0.9,
            alpha=0.9,
            label=f"{model.upper()} forecast",
        )
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_ylabel(pc)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"Daily {model.upper()} Forecasts with Macro vs Actual {horizon}-Day PC Changes")
    axes[-1].set_xlabel("Prediction date")
    fig.tight_layout()
    fig.savefig(output / "oos_prediction_series.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(predictions: pd.DataFrame, output: Path, model: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for pc_index, ax in enumerate(axes, start=1):
        pc = f"PC{pc_index}"
        data = predictions[[f"actual_{pc}", f"{model}_{pc}"]].dropna()
        actual = data[f"actual_{pc}"]
        forecast = data[f"{model}_{pc}"]
        correlation = actual.corr(forecast)
        ax.scatter(forecast, actual, s=12, alpha=0.4, color="tab:blue")
        lower = min(actual.min(), forecast.min())
        upper = max(actual.max(), forecast.max())
        ax.plot([lower, upper], [lower, upper], linestyle="--", color="grey", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(f"{pc}: correlation = {correlation:.3f}")
        ax.set_xlabel(f"{model.upper()} forecast")
        ax.set_ylabel("Actual")
        ax.grid(alpha=0.2)
    fig.suptitle(f"Daily {model.upper()} Forecasts with Macro: Predicted vs Actual", y=1.03)
    fig.tight_layout()
    fig.savefig(output / "oos_prediction_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame, output: Path, model: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics.plot.bar(x="target", y="RMSE", ax=axes[0], legend=False, rot=0)
    metrics.plot.bar(
        x="target", y="correlation", ax=axes[1], legend=False, rot=0
    )
    metrics.plot.bar(x="target", y="R2", ax=axes[2], legend=False, rot=0)
    axes[0].set_title("RMSE")
    axes[1].set_title("Prediction Correlation")
    axes[2].set_title("R-squared")
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Daily {model.upper()} Metrics with Macro", y=1.03)
    fig.tight_layout()
    fig.savefig(output / "oos_model_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['ridge', 'lgbm'], default='ridge')
    args = parser.parse_args()
    output = PROJECT_ROOT / 'results' / 'forecast' / args.model
    predictions = pd.read_csv(output / 'daily_oos_predictions.csv', parse_dates=['prediction_date'])
    metrics = pd.read_csv(output / 'pc_metrics.csv')
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    horizon = int(manifest['config']['forecast']['horizon_days'])
    plot_prediction_series(predictions, output, args.model, horizon)
    plot_prediction_scatter(predictions, output, args.model)
    plot_metrics(metrics, output, args.model)
    print(f'Saved forecast charts to {output}')


if __name__ == '__main__':
    main()
