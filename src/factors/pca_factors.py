"""PCA utilities for Treasury yield-curve changes measured in basis points."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _validate_yield_changes(yield_changes: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a floating-point copy of a yield-change matrix."""
    if not isinstance(yield_changes, pd.DataFrame):
        raise TypeError("yield_changes must be a pandas DataFrame")
    if yield_changes.empty:
        raise ValueError("yield_changes must contain at least one observation")
    if yield_changes.shape[1] < 2:
        raise ValueError("PCA requires at least two maturity columns")

    numeric = yield_changes.astype(float)
    if numeric.isna().any().any():
        raise ValueError("yield_changes contains missing values")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("yield_changes contains non-finite values")
    return numeric


def _orientation_statistic(component: np.ndarray, pc_index: int) -> float:
    """Return the deterministic sign statistic for one four-maturity component."""
    if pc_index == 0:
        # PC1 positive means a broad rise in yields.
        return float(component.sum())
    if pc_index == 1:
        # PC2 positive means the short end rises relative to the long end.
        return float(component[0] - component[-1])
    if pc_index == 2 and len(component) >= 4:
        # PC3 positive means the middle rises relative to the two wings.
        middle = component[1:-1].mean()
        wings = (component[0] + component[-1]) / 2.0
        return float(middle - wings)

    # Give any remaining component a repeatable sign.
    largest = int(np.argmax(np.abs(component)))
    return float(component[largest])


def orient_pca_loadings(pca: PCA) -> PCA:
    """Apply deterministic level, slope, and curvature sign conventions in place."""
    if not hasattr(pca, "components_"):
        raise ValueError("PCA must be fitted before its loadings can be oriented")

    for pc_index, component in enumerate(pca.components_):
        if _orientation_statistic(component, pc_index) < 0:
            pca.components_[pc_index] *= -1.0
    return pca


def fit_pca(
    yield_changes: pd.DataFrame,
    n_components: int | None = None,
    orient: bool = True,
) -> PCA:
    """Fit covariance PCA using training-sample centering and no variance scaling."""
    numeric = _validate_yield_changes(yield_changes)
    maximum = min(numeric.shape)
    requested = numeric.shape[1] if n_components is None else int(n_components)
    if requested < 1 or requested > maximum:
        raise ValueError(f"n_components must be between 1 and {maximum}")

    pca = PCA(n_components=requested)
    pca.fit(numeric)
    if orient:
        orient_pca_loadings(pca)
    return pca


def transform_to_pc_scores(pca: PCA, yield_changes: pd.DataFrame) -> pd.DataFrame:
    """Project maturity-level yield changes into the fitted PCA coordinate system."""
    numeric = _validate_yield_changes(yield_changes)
    if numeric.shape[1] != pca.n_features_in_:
        raise ValueError("yield_changes has a different number of maturities than the PCA")

    scores = pca.transform(numeric)
    columns = [f"PC{i}" for i in range(1, scores.shape[1] + 1)]
    return pd.DataFrame(scores, index=numeric.index, columns=columns)


def reconstruct_yield_changes(
    pca: PCA,
    pc_scores: pd.DataFrame,
    n_components: int | None = None,
) -> pd.DataFrame:
    """Reconstruct maturity changes using the requested number of fitted PCs."""
    if pc_scores.isna().any().any():
        raise ValueError("pc_scores contains missing values")
    total_components = pca.components_.shape[0]
    retained = total_components if n_components is None else int(n_components)
    if retained < 1 or retained > total_components:
        raise ValueError(f"n_components must be between 1 and {total_components}")
    if pc_scores.shape[1] != total_components:
        raise ValueError("pc_scores does not match the fitted PCA dimension")

    truncated = np.zeros_like(pc_scores.to_numpy(dtype=float))
    truncated[:, :retained] = pc_scores.iloc[:, :retained].to_numpy(dtype=float)
    reconstructed = pca.inverse_transform(truncated)
    feature_names = getattr(
        pca,
        "feature_names_in_",
        np.array([f"maturity_{i}" for i in range(pca.n_features_in_)]),
    )
    return pd.DataFrame(reconstructed, index=pc_scores.index, columns=feature_names)


def explained_variance_table(pca: PCA) -> pd.DataFrame:
    """Return individual and cumulative explained-variance ratios."""
    pc_names = [f"PC{i}" for i in range(1, len(pca.explained_variance_ratio_) + 1)]
    return pd.DataFrame(
        {
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_ratio": np.cumsum(pca.explained_variance_ratio_),
        },
        index=pd.Index(pc_names, name="component"),
    )


def loadings_table(pca: PCA) -> pd.DataFrame:
    """Return PCA loading vectors with maturity and component labels."""
    pc_names = [f"PC{i}" for i in range(1, pca.components_.shape[0] + 1)]
    feature_names = getattr(
        pca,
        "feature_names_in_",
        np.array([f"maturity_{i}" for i in range(pca.n_features_in_)]),
    )
    return pd.DataFrame(
        pca.components_,
        index=pd.Index(pc_names, name="component"),
        columns=feature_names,
    )


def reconstruction_error_tables(
    actual: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return overall and maturity-level reconstruction diagnostics."""
    actual_numeric = _validate_yield_changes(actual)
    reconstructed_numeric = _validate_yield_changes(reconstructed)
    if actual_numeric.shape != reconstructed_numeric.shape:
        raise ValueError("actual and reconstructed must have the same shape")

    overall = pd.DataFrame(
        {
            "MAE_bp": [mean_absolute_error(actual_numeric, reconstructed_numeric)],
            "RMSE_bp": [mean_squared_error(actual_numeric, reconstructed_numeric) ** 0.5],
        }
    )

    rows: list[Dict[str, float | str]] = []
    for column in actual_numeric.columns:
        observed = actual_numeric[column].to_numpy()
        fitted = reconstructed_numeric[column].to_numpy()
        rows.append(
            {
                "maturity": str(column),
                "MAE_bp": mean_absolute_error(observed, fitted),
                "RMSE_bp": mean_squared_error(observed, fitted) ** 0.5,
                "correlation": float(np.corrcoef(observed, fitted)[0, 1]),
            }
        )
    by_maturity = pd.DataFrame(rows).set_index("maturity")
    return overall, by_maturity
