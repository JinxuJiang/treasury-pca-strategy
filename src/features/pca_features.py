"""Build leakage-safe PCA forecasting features and forward targets."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.factors.pca_factors import transform_to_pc_scores


def forward_sum_and_end_date(
    values: pd.DataFrame,
    horizon_days: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return sums over t+1..t+h and the date on which each label completes. This to calculate the label"""
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("values must have a DatetimeIndex")
    if not values.index.is_monotonic_increasing or values.index.has_duplicates:
        raise ValueError("values index must be sorted and unique")

    future_sum = sum(values.shift(-step) for step in range(1, horizon_days + 1))
    future_sum.columns = [f"target_{column}_{horizon_days}d" for column in values.columns]
    end_date = pd.Series(
        values.index.to_series().shift(-horizon_days).to_numpy(),
        index=values.index,
        name="target_end_date",
    )
    return future_sum, end_date


def build_feature_target_frame(
    pca: PCA,
    changes: pd.DataFrame,
    levels: pd.DataFrame,
    maturities: Sequence[str],
    retained_components: int = 3,
    horizon_days: int = 5,
) -> pd.DataFrame:
    """Build one PCA-vintage dataset using only contemporaneous and past features."""
    if not changes.index.equals(levels.index):
        raise ValueError("changes and levels must have identical date indexes")

    score_columns = [f"PC{i}" for i in range(1, retained_components + 1)]
    all_scores = transform_to_pc_scores(pca, changes)
    all_scores.index = changes.index
    scores = all_scores[score_columns]
    frame = pd.DataFrame(index=changes.index)

    for pc in score_columns:
        frame[f"{pc}_current"] = scores[pc]
        for lag in (1, 2, 5, 10, 20):
            frame[f"{pc}_lag{lag}"] = scores[pc].shift(lag)
        frame[f"{pc}_sum5"] = scores[pc].rolling(5).sum()
        frame[f"{pc}_sum20"] = scores[pc].rolling(20).sum()
        frame[f"{pc}_mean20"] = scores[pc].rolling(20).mean()
        frame[f"{pc}_vol20"] = scores[pc].rolling(20).std()
        frame[f"{pc}_z20"] = (
            (scores[pc] - frame[f"{pc}_mean20"]) / frame[f"{pc}_vol20"]
        )

    level_columns = [f"{maturity}" for maturity in maturities]
    for maturity in level_columns:
        frame[f"yield_{maturity}"] = levels[maturity]
        frame[f"d_{maturity}_vol20"] = changes[f"d_{maturity}_bp"].rolling(20).std()

    frame["curve_level"] = levels[level_columns].mean(axis=1)
    frame["slope_10Y_2Y"] = levels["10Y"] - levels["2Y"]
    frame["curvature_2Y_5Y_10Y"] = levels["10Y"] + levels["2Y"] - 2 * levels["5Y"]

    targets, target_end = forward_sum_and_end_date(scores, horizon_days)
    frame = frame.join(targets)
    frame["target_end_date"] = target_end
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model feature columns from a feature/target frame."""
    return [
        column
        for column in frame.columns
        if not column.startswith("target_") and column != "target_end_date"
    ]
