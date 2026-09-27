import numpy as np
import pandas as pd
import pytest

from src.factors.pca_factors import (
    fit_pca,
    loadings_table,
    reconstruct_yield_changes,
    transform_to_pc_scores,
)


def sample_curve_changes() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    level = rng.normal(0, 5, 300)
    slope = rng.normal(0, 2, 300)
    curvature = rng.normal(0, 0.8, 300)
    noise = rng.normal(0, 0.05, (300, 4))
    matrix = (
        level[:, None] * np.array([0.45, 0.55, 0.55, 0.45])
        + slope[:, None] * np.array([0.70, 0.25, -0.25, -0.70])
        + curvature[:, None] * np.array([-0.55, 0.50, 0.35, -0.50])
        + noise
    )
    return pd.DataFrame(
        matrix,
        columns=["d_2Y_bp", "d_5Y_bp", "d_10Y_bp", "d_30Y_bp"],
    )


def test_pca_orientation_and_orthogonality() -> None:
    changes = sample_curve_changes()
    pca = fit_pca(changes, n_components=4)
    loadings = loadings_table(pca)

    assert loadings.loc["PC1"].sum() > 0
    assert loadings.loc["PC2", "d_2Y_bp"] > loadings.loc["PC2", "d_30Y_bp"]
    middle = loadings.loc["PC3", ["d_5Y_bp", "d_10Y_bp"]].mean()
    wings = loadings.loc["PC3", ["d_2Y_bp", "d_30Y_bp"]].mean()
    assert middle > wings
    np.testing.assert_allclose(
        pca.components_ @ pca.components_.T,
        np.eye(4),
        atol=1e-12,
    )


def test_all_components_reconstruct_exactly() -> None:
    changes = sample_curve_changes()
    pca = fit_pca(changes, n_components=4)
    scores = transform_to_pc_scores(pca, changes)
    reconstructed = reconstruct_yield_changes(pca, scores, n_components=4)
    np.testing.assert_allclose(reconstructed, changes, atol=1e-12)


def test_three_components_improve_over_one() -> None:
    changes = sample_curve_changes()
    pca = fit_pca(changes, n_components=4)
    scores = transform_to_pc_scores(pca, changes)
    one_pc = reconstruct_yield_changes(pca, scores, n_components=1)
    three_pc = reconstruct_yield_changes(pca, scores, n_components=3)
    one_rmse = np.sqrt(np.mean((changes.to_numpy() - one_pc.to_numpy()) ** 2))
    three_rmse = np.sqrt(np.mean((changes.to_numpy() - three_pc.to_numpy()) ** 2))
    assert three_rmse < one_rmse


def test_missing_values_are_rejected() -> None:
    changes = sample_curve_changes()
    changes.loc[0, "d_10Y_bp"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        fit_pca(changes)
