import pandas as pd

from src.models.time_split import (
    chronological_validation_masks,
    eligible_training_rows,
    monthly_forecast_periods,
)


def test_validation_purges_labels_crossing_its_start() -> None:
    dates = pd.bdate_range("2023-12-26", "2024-01-05")
    frame = pd.DataFrame(index=dates)
    frame["target_end_date"] = dates + pd.offsets.BDay(5)
    cutoff = pd.Timestamp("2025-01-01")

    eligible = eligible_training_rows(frame, cutoff)
    training, validation = chronological_validation_masks(frame, eligible, cutoff, 1)

    assert not training.loc[pd.Timestamp("2023-12-29")]
    assert validation.loc[pd.Timestamp("2024-01-01")]
    assert not (training & validation).any()


def test_first_forecast_month_starts_on_requested_oos_date() -> None:
    dates = pd.bdate_range("2019-12-30", "2020-02-05")
    periods = monthly_forecast_periods(dates, pd.Timestamp("2020-01-15"))

    assert periods[0][1] == pd.Timestamp("2019-12-31")
    assert periods[0][2].min() == pd.Timestamp("2020-01-15")
    assert periods[1][2].min() == pd.Timestamp("2020-02-03")
