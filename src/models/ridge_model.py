"""Ridge hyperparameter selection and final model fitting."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def choose_ridge_alpha(
    frame: pd.DataFrame,
    features: Sequence[str],
    target: str,
    training_mask: pd.Series,
    validation_mask: pd.Series,
    alphas: Sequence[float],
) -> tuple[float, float]:
    """Choose Ridge alpha using a chronological validation period."""
    if not training_mask.any() or not validation_mask.any():
        raise ValueError("training and validation periods must both contain observations")

    best_alpha = None
    best_rmse = float("inf")
    # for every pc, choose  the best alpha from different alphas.
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(frame.loc[training_mask, features], frame.loc[training_mask, target])
        # use the validation to choose the best alpha
        prediction = model.predict(frame.loc[validation_mask, features])
        rmse = mean_squared_error(frame.loc[validation_mask, target], prediction) ** 0.5
        if rmse < best_rmse:
            best_alpha = float(alpha)
            best_rmse = float(rmse)
    return float(best_alpha), best_rmse


def fit_ridge(
    frame: pd.DataFrame,
    features: Sequence[str],
    target: str,
    training_mask: pd.Series,
    alpha: float,
) -> Pipeline:
    """Fit the final standardized Ridge on all currently eligible rows."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(frame.loc[training_mask, features], frame.loc[training_mask, target])
    return model
