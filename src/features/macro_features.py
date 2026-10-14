"""Calculate features within each vintage and align them forward to signal dates.
   Making snapshot features into daily featrue.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def snapshot_features(snapshot: pd.DataFrame, name: str, kind: str) -> dict:
    '''
    given a snapshot, return a row of daily features.
    '''
    if snapshot.empty or snapshot['snapshot_date'].nunique() != 1:
        raise ValueError('Expected one nonempty snapshot')
    months = pd.PeriodIndex(snapshot['observation_month'], freq='M')
    if months.has_duplicates:
        raise ValueError('Duplicate observation months in snapshot')
    values = pd.Series(snapshot['value'].to_numpy(dtype=float), index=months).sort_index()
    valid = values.dropna()
    if valid.empty:
        raise ValueError('Snapshot contains no known values')

    # Get the latest and previous value to calculate features.
    latest = valid.index.max()
    current = values.loc[latest]
    if not np.isfinite(current) or (kind == 'price_index' and current <= 0):
        raise ValueError('Invalid current macro value')
    row = {'snapshot_date': snapshot['snapshot_date'].iloc[0],
           'observation_month': latest.to_timestamp()}
    previous = values.get(latest - 1, np.nan)
    if kind == 'unemployment':
        row[f'{name}_rate'] = current
        row[f'{name}_change_1m_pp'] = current - previous
    elif kind == 'price_index':
        year_ago = values.get(latest - 12, np.nan)
        row[f'{name}_mom_pct'] = (current / previous - 1) * 100 if previous > 0 else np.nan
        row[f'{name}_yoy_pct'] = (current / year_ago - 1) * 100 if year_ago > 0 else np.nan
    else:
        raise ValueError(f'Unknown macro kind: {kind}')
    return row


def align_updates(updates: pd.DataFrame, trading_dates: pd.DatetimeIndex,
                  name: str, lag_sessions: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A date-only vintage is conservatively usable next sample trading session."""
    if lag_sessions < 1:
        raise ValueError('Date-only data requires at least one trading-session lag')
    if trading_dates.empty or trading_dates.has_duplicates or not trading_dates.is_monotonic_increasing:
        raise ValueError('Trading dates must be nonempty, sorted and unique')
    if updates.empty:
        raise ValueError('No feature updates')
    updates = updates.sort_values('snapshot_date').copy()
    if updates['snapshot_date'].duplicated().any():
        raise ValueError('Duplicate snapshot dates')
    # Strictly later session. If the vintage is a holiday, first following session.
    positions = trading_dates.searchsorted(pd.DatetimeIndex(updates['snapshot_date']), side='right') + lag_sessions - 1
    updates['available_date'] = [trading_dates[p] if p < len(trading_dates) else pd.NaT for p in positions]
    updates = updates.dropna(subset=['available_date'])
    if updates.empty:
        raise ValueError('No updates available within sample')
    
    # Multiple vintages may become usable on the same trading day: keep latest.
    right = updates.drop_duplicates('available_date', keep='last').sort_values('available_date')
    daily = pd.merge_asof(pd.DataFrame({'signal_date': trading_dates}), right,
                          left_on='signal_date', right_on='available_date', direction='backward')
    daily = daily.set_index('signal_date')
    columns = [c for c in updates if c.startswith(f'{name}_')]
    features = daily[columns]
    lineage = daily[['snapshot_date', 'available_date', 'observation_month']].add_prefix(f'{name}_')
    return features, lineage
