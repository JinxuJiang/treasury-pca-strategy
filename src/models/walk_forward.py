"""Monthly expanding-window Ridge forecasts with an expanding PCA representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.factors.pca_factors import fit_pca
from src.features.pca_features import build_feature_target_frame, feature_columns
from src.models.ridge_model import choose_ridge_alpha as _choose_ridge_alpha
from src.models.ridge_model import fit_ridge
from src.models.time_split import (
    chronological_validation_masks,
    eligible_training_rows,
    monthly_forecast_periods,
)


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: pd.DataFrame
    model_diagnostics: pd.DataFrame
    pca_loadings: pd.DataFrame


def run_monthly_expanding_forecast(
    changes: pd.DataFrame,
    levels: pd.DataFrame,
    maturities: Sequence[str],
    first_oos_date: pd.Timestamp,
    horizon_days: int = 5,
    retained_components: int = 3,
    validation_years: int = 2,
    ridge_alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    extra_features: pd.DataFrame | None = None,
    sample_mask: pd.Series | None = None,
    model_type: str = "ridge",
) -> WalkForwardResult:
    """Produce stitched daily OOS forecasts from monthly expanding refits."""
    if model_type not in ("ridge", "lgbm"):
        raise ValueError("model_type must be ridge or lgbm")
    if model_type == "lgbm":
        from src.models.lgbm_model import choose_lgbm_params, fit_lgbm
    if not changes.index.equals(levels.index):
        raise ValueError("changes and levels must have identical date indexes")
    if sample_mask is None:
        sample_mask = pd.Series(True, index=changes.index)
    elif not sample_mask.index.equals(changes.index) or sample_mask.isna().any():
        raise ValueError("sample mask must match dates and contain no missing values")

    # return ebery month training dates and prediction dates  (month_start, cutoff, prediction_dates)
    periods = monthly_forecast_periods(changes.index, first_oos_date)
    # collect every month daily forcast, model training imfor, PCA loading
    prediction_parts: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    loading_rows: list[dict[str, object]] = []

    for month_start, cutoff, prediction_dates in periods:
        if not sample_mask.loc[prediction_dates].any():
            continue
        target_columns = [
            f"target_PC{i}_{horizon_days}d"
            for i in range(1, retained_components + 1)
        ]

        # 1. Prepare tuning data with a PCA fitted before validation starts.
        # push two years ahead, making it as the start of validation
        validation_start = cutoff - pd.DateOffset(years=validation_years)
        # the dates of training data
        tuning_pca_dates = changes.index[changes.index < validation_start]
        if tuning_pca_dates.empty:
            raise ValueError("not enough observations before the validation period")
        tuning_pca_cutoff = tuning_pca_dates.max()
        # using the training data to fit pca, getting pca loadings
        tuning_pca = fit_pca(
            changes.loc[:tuning_pca_cutoff], # get the training data
            n_components=changes.shape[1],
            orient=True,
        )
        # using training pca loading, get the whole feacutures and tragets data.
        tuning_frame = build_feature_target_frame(
            pca=tuning_pca, # use the training pca
            extra_features=extra_features,
            changes=changes, # use the whole change, levels, which means it conduct the all data.
            levels=levels,
            maturities=maturities,
            retained_components=retained_components,
            horizon_days=horizon_days,
        )
        tuning_features = feature_columns(tuning_frame)
        tuning_complete_features = tuning_frame[tuning_features].notna().all(axis=1) # bollen series(mask)
        tuning_complete_targets = tuning_frame[target_columns].notna().all(axis=1)
        # tuning_eligible only choose normald datas.
        tuning_eligible = (
            eligible_training_rows(tuning_frame, cutoff)
            & tuning_complete_features
            & tuning_complete_targets
            & sample_mask
        )
        # split the whole data tuning_frame/ tuning_eligible into subtrain, validation
        subtrain, validation = chronological_validation_masks(
            tuning_frame,
            tuning_eligible,
            cutoff,
            validation_years,
        )

        # 2. Select each component's alpha before fitting the final models.
        # use a dic to store the best alpha and rmse for each pc.
        # {"PC1": (100.0, 20.8),"PC2": (10.0, 8.2),"PC3": (1.0, 2.5)}
        tuning_results: dict[str, tuple[float | dict, float]] = {}
        for pc_index in range(1, retained_components + 1):
            pc = f"PC{pc_index}"
            target = f"target_{pc}_{horizon_days}d"
            tuning_results[pc] = _choose_ridge_alpha(
                tuning_frame,
                tuning_features,
                target,
                subtrain,
                validation,
                ridge_alphas,
            ) if model_type == "ridge" else choose_lgbm_params(
                tuning_frame, tuning_features, target, subtrain, validation
            )

        # 3. Refit PCA through cutoff and rebuild the final feature/target frame.
        #prediction_date  model_cutoff_date  pca_version  target_end_date
        #2020-01-02       2019-12-31         2019-12-31   corresponding fifth future observation date
        #2020-01-03       2019-12-31         2019-12-31   corresponding fifth future observation date
        pca = fit_pca(
            changes.loc[:cutoff], # use the whole training and validation data to fit pca.
            n_components=changes.shape[1],
            orient=True,
        )
        frame = build_feature_target_frame(
            pca=pca,
            extra_features=extra_features,
            changes=changes,
            levels=levels,
            maturities=maturities,
            retained_components=retained_components,
            horizon_days=horizon_days,
        )
        features = feature_columns(frame)
        complete_features = frame[features].notna().all(axis=1)
        complete_targets = frame[target_columns].notna().all(axis=1)
        eligible = (
            eligible_training_rows(frame, cutoff)
            & complete_features
            & complete_targets
            & sample_mask
        )

        prediction_mask = frame.index.isin(prediction_dates) & complete_features & sample_mask # choose the complete features and prediction dates.
        prediction_frame = pd.DataFrame(index=frame.index[prediction_mask])
        prediction_frame.index.name = "prediction_date"
        prediction_frame["model_cutoff_date"] = cutoff
        prediction_frame["pca_version"] = cutoff
        prediction_frame["target_end_date"] = frame.loc[
            prediction_mask, "target_end_date"
        ]

        # 4. Fit final Ridge models with selected alphas and predict daily.
        for pc_index in range(1, retained_components + 1):
            pc = f"PC{pc_index}"
            target = f"target_{pc}_{horizon_days}d"
            alpha, validation_rmse = tuning_results[pc]
            ridge = fit_ridge(frame, features, target, eligible, alpha) if model_type == "ridge" else fit_lgbm(frame, features, target, eligible, alpha)
            # add a column to store the actual target value.
            prediction_frame[f"actual_{pc}"] = frame.loc[prediction_mask, target]
            # add a column to store the ridge predict value.
            prediction_frame[f"{model_type}_{pc}"] = ridge.predict(
                frame.loc[prediction_mask, features]
            )

            diagnostic_rows.append(
                {
                    "model_cutoff_date": cutoff,
                    "forecast_month": month_start.strftime("%Y-%m"),
                    "component": pc,
                    "training_observations": int(eligible.sum()),
                    "validation_observations": int(validation.sum()),
                    "validation_pca_cutoff": tuning_pca_cutoff,
                    "last_training_feature_date": frame.index[eligible].max(),
                    "last_training_target_end_date": frame.loc[
                        eligible, "target_end_date"
                    ].max(),
                    ("ridge_alpha" if model_type == "ridge" else "lgbm_params"): alpha,
                    "validation_rmse": validation_rmse,
                }
            )

        # store the pca loading for each component and maturity.
        '''
        pca_version   component   maturity    loading
        2019-12-31    PC1         d_2Y_bp     ...
        2019-12-31    PC1         d_5Y_bp     ...
        2019-12-31    PC1         d_10Y_bp    ...
        2019-12-31    PC1         d_30Y_bp    ...
        2019-12-31    PC2         d_2Y_bp     ...
        '''
        for component_index, component in enumerate(pca.components_, start=1):
            for maturity, loading in zip(changes.columns, component):
                loading_rows.append(
                    {
                        "pca_version": cutoff,
                        "component": f"PC{component_index}",
                        "maturity": maturity,
                        "loading": float(loading),
                    }
                )

        prediction_parts.append(prediction_frame.reset_index())

    if not prediction_parts:
        raise ValueError("no out-of-sample prediction periods were generated")
    # vertically concat different month predictions.
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        "prediction_date"
    )

    diagnostics = pd.DataFrame(diagnostic_rows)

    pca_loadings = pd.DataFrame(loading_rows)
    return WalkForwardResult(predictions, diagnostics, pca_loadings)
