"""Plot daily out-of-sample Ridge forecast diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "predictions"


def plot_prediction_series(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    for pc_index, ax in enumerate(axes, start=1):
        pc = f"PC{pc_index}"
        ax.plot(
            predictions["prediction_date"],
            predictions[f"actual_{pc}"],
            color="black",
            linewidth=0.9,
            label="Actual future 5-day PC change",
        )
        ax.plot(
            predictions["prediction_date"],
            predictions[f"ridge_{pc}"],
            color="tab:blue",
            linewidth=0.9,
            alpha=0.9,
            label="Ridge forecast",
        )
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_ylabel(pc)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Daily Out-of-Sample Ridge Forecasts vs Actual 5-Day PC Changes")
    axes[-1].set_xlabel("Prediction date")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "oos_prediction_series.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for pc_index, ax in enumerate(axes, start=1):
        pc = f"PC{pc_index}"
        data = predictions[[f"actual_{pc}", f"ridge_{pc}"]].dropna()
        actual = data[f"actual_{pc}"]
        forecast = data[f"ridge_{pc}"]
        correlation = actual.corr(forecast)
        ax.scatter(forecast, actual, s=12, alpha=0.4, color="tab:blue")
        lower = min(actual.min(), forecast.min())
        upper = max(actual.max(), forecast.max())
        ax.plot([lower, upper], [lower, upper], linestyle="--", color="grey", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(f"{pc}: correlation = {correlation:.3f}")
        ax.set_xlabel("Ridge forecast")
        ax.set_ylabel("Actual")
        ax.grid(alpha=0.2)
    fig.suptitle("Daily Ridge Forecasts: Predicted vs Actual", y=1.03)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "oos_prediction_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics.plot.bar(x="component", y="RMSE", ax=axes[0], legend=False, rot=0)
    metrics.plot.bar(
        x="component", y="correlation", ax=axes[1], legend=False, rot=0
    )
    (metrics.assign(direction_accuracy=metrics["direction_accuracy"] * 100)).plot.bar(
        x="component", y="direction_accuracy", ax=axes[2], legend=False, rot=0
    )
    axes[0].set_title("RMSE")
    axes[1].set_title("Prediction Correlation")
    axes[2].set_title("Direction Accuracy")
    axes[2].set_ylabel("Percent")
    axes[2].axhline(50, color="black", linestyle="--", linewidth=1)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Daily Out-of-Sample Ridge Metrics", y=1.03)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "oos_model_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    predictions = pd.read_csv(
        RESULTS_DIR / "daily_oos_predictions.csv", parse_dates=["prediction_date"]
    )
    metrics = pd.read_csv(RESULTS_DIR / "oos_metrics.csv")
    plot_prediction_series(predictions)
    plot_prediction_scatter(predictions)
    plot_metrics(metrics)
    print(f"Saved forecast charts to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
