import numpy as np
import pandas as pd

from src.features.pca_features import forward_sum_and_end_date
from src.models.time_split import eligible_training_rows


def test_forward_target_uses_t_plus_1_through_t_plus_5() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    values = pd.DataFrame({"PC1": np.arange(1.0, 9.0)}, index=dates)

    targets, end_dates = forward_sum_and_end_date(values, horizon_days=5)

    assert targets.loc[dates[0], "target_PC1_5d"] == 2 + 3 + 4 + 5 + 6
    assert end_dates.loc[dates[0]] == dates[5]
    assert pd.isna(targets.loc[dates[-5], "target_PC1_5d"])


def test_training_rows_require_the_complete_label_by_cutoff() -> None:
    dates = pd.bdate_range("2024-01-01", periods=10)
    values = pd.DataFrame({"PC1": np.arange(10.0)}, index=dates)
    targets, end_dates = forward_sum_and_end_date(values, horizon_days=5)
    frame = targets.copy()
    frame["target_end_date"] = end_dates

    cutoff = dates[7]
    eligible = eligible_training_rows(frame, cutoff)

    assert list(frame.index[eligible]) == list(dates[:3])
    assert frame.loc[eligible, "target_end_date"].max() <= cutoff
