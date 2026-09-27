"""Fit the Treasury curve PCA representation and save diagnostic outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factors.pca_factors import (  # noqa: E402
    explained_variance_table,
    fit_pca,
    loadings_table,
    reconstruction_error_tables,
    reconstruct_yield_changes,
    transform_to_pc_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pca_config = config["pca"]
    paths = config["paths"]

    input_path = PROJECT_ROOT / paths["processed_dir"] / "treasury_yield_changes_bp.csv"
    output_dir = PROJECT_ROOT / paths["pca_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    maturities = config["data"]["maturities"]
    change_columns = [f"d_{maturity}_bp" for maturity in maturities]
    sample_start = pd.Timestamp(pca_config["sample_start"])
    train_end = pd.Timestamp(pca_config["train_end"])
    n_components = int(pca_config["n_components"])
    retained_components = int(pca_config["retained_components"])

    changes = pd.read_csv(input_path, parse_dates=["date"])
    sample = (
        changes.loc[changes["date"] >= sample_start, ["date", *change_columns]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    train = sample.loc[sample["date"] <= train_end, change_columns]
    if train.empty:
        raise ValueError("PCA training sample is empty")
    if train_end >= sample["date"].max():
        raise ValueError("train_end must leave observations after the training period")

    pca = fit_pca(train, n_components=n_components, orient=True)
    scores = transform_to_pc_scores(pca, sample[change_columns])
    scores.insert(0, "date", sample["date"].to_numpy())

    loadings = loadings_table(pca)
    explained = explained_variance_table(pca)

    reconstruction_summary_rows = []
    retained_reconstruction = None
    retained_by_maturity = None
    score_columns = [column for column in scores.columns if column.startswith("PC")]
    score_values = scores[score_columns]

    for retained in range(1, n_components + 1):
        reconstructed = reconstruct_yield_changes(pca, score_values, retained)
        overall, by_maturity = reconstruction_error_tables(
            sample[change_columns], reconstructed
        )
        reconstruction_summary_rows.append(
            {
                "n_components": retained,
                "MAE_bp": float(overall.loc[0, "MAE_bp"]),
                "RMSE_bp": float(overall.loc[0, "RMSE_bp"]),
            }
        )
        if retained == retained_components:
            retained_reconstruction = reconstructed
            retained_by_maturity = by_maturity

    if retained_reconstruction is None or retained_by_maturity is None:
        raise ValueError("retained_components must be between 1 and n_components")

    reconstructed_output = retained_reconstruction.copy()
    reconstructed_output.columns = [
        column.replace("d_", "reconstructed_d_") for column in reconstructed_output.columns
    ]
    reconstructed_output.insert(0, "date", sample["date"].to_numpy())

    reconstruction_summary = pd.DataFrame(reconstruction_summary_rows)
    metadata = {
        "sample_start": sample["date"].min().strftime("%Y-%m-%d"),
        "sample_end": sample["date"].max().strftime("%Y-%m-%d"),
        "train_end": train_end.strftime("%Y-%m-%d"),
        "training_observations": int(len(train)),
        "total_observations": int(len(sample)),
        "n_components": n_components,
        "retained_components": retained_components,
        "input_unit": "basis_points",
        "scaling": "centered_training_mean_only_no_variance_standardization",
        "sign_convention": {
            "PC1": "positive means a broad rise in yields",
            "PC2": "positive means the short end rises relative to the long end",
            "PC3": "positive means the middle rises relative to the wings",
        },
        "training_means_bp": {
            column: float(value) for column, value in zip(change_columns, pca.mean_)
        },
    }

    scores.to_csv(output_dir / "pc_scores.csv", index=False, date_format="%Y-%m-%d")
    loadings.to_csv(output_dir / "pca_loadings.csv")
    explained.to_csv(output_dir / "explained_variance.csv")
    reconstruction_summary.to_csv(
        output_dir / "reconstruction_summary.csv", index=False
    )
    retained_by_maturity.to_csv(output_dir / "reconstruction_by_maturity.csv")
    reconstructed_output.to_csv(
        output_dir / "reconstructed_yield_changes_bp.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    (output_dir / "pca_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"PCA fitted on {len(train):,} observations through {train_end.date()}.")
    print(explained.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Saved PCA outputs to {output_dir}")


if __name__ == "__main__":
    main()
