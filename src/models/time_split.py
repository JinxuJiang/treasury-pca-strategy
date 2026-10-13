"""Time boundaries and leakage-safe sample masks for walk-forward forecasts."""

from __future__ import annotations

import pandas as pd


def eligible_training_rows(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Select rows whose complete forward label was observable by cutoff.
    frame: the dataframe of features and target

    date(index)   target_end_date    return
    2019-12-20    2019-12-27         True
    2019-12-27    2020-01-06         False
    2026-09-25    NaT                False
    """
    cutoff = pd.Timestamp(cutoff)
    return frame["target_end_date"].notna() & (frame["target_end_date"] <= cutoff)


def chronological_validation_masks(
    frame: pd.DataFrame,
    eligible: pd.Series, # it is come from eligible_training_rows, means from cutoff data to spilt the training and validation period
    cutoff: pd.Timestamp,
    validation_years: int,
) -> tuple[pd.Series, pd.Series]:
    """Create purged training and recent validation masks. It only return series of boolean."""
    validation_start = pd.Timestamp(cutoff) - pd.DateOffset(years=validation_years) # DateOffset push back time back
    validation = eligible & (frame.index >= validation_start)
    training = eligible & (frame["target_end_date"] < validation_start)
    return training, validation


def monthly_forecast_periods(
    dates: pd.DatetimeIndex,
    first_oos_date: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]]:
    """Return each forecast month, its preceding cutoff, and its daily dates.
    从第一个 OOS 月份开始，逐月往后走；每一轮找出该月之前最后一个交易日作为 cutoff，再找出该月所有要预测的交易日.
    返回 (月份, cutoff, 预测日期)。
    (2016-01, 2015年末最后交易日, 2016年1月所有交易日)
    (2016-02, 2016年1月末最后交易日, 2016年2月所有交易日)
    """
    first_oos_date = pd.Timestamp(first_oos_date)
    # find all the month start date from first_oos_date to the last date in dates
    month_starts = pd.date_range(
        first_oos_date.to_period("M").start_time,
        dates.max().to_period("M").start_time,
        freq="MS",
    )
    periods = []
    for month_start in month_starts:
        # find all the dates available for every month. This is for the training set
        # because we use the expanding window to train the model
        prior_dates = dates[dates < month_start]
        if prior_dates.empty:
            continue
        # this is for the forecast/test set. this select the end of the month as the forecast_end
        forecast_end = month_start + pd.offsets.MonthEnd(0)
        prediction_dates = dates[
            (dates >= month_start)
            & (dates <= forecast_end)
            & (dates >= first_oos_date)
        ]
        if not prediction_dates.empty:
            periods.append((month_start, prior_dates.max(), prediction_dates)) #prior_dates.max() is cutoff date(the last date of training + validation set)
    return periods
