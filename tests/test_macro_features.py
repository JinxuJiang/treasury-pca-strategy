import json

import numpy as np
import pandas as pd

from src.data.fred_vintages import FredVintageClient
from src.features.macro_features import snapshot_features, align_updates


def snapshot(vintage, months, values):
    return pd.DataFrame({'snapshot_date': pd.Timestamp(vintage),
                         'observation_month': pd.to_datetime(months), 'value': values})


def test_revision_does_not_rewrite_previous_days():
    old = snapshot_features(snapshot('2020-02-07', ['2019-12-01', '2020-01-01'], [3.9, 4.0]), 'unemployment', 'unemployment')
    new = snapshot_features(snapshot('2020-03-06', ['2020-01-01', '2020-02-01'], [4.1, 4.2]), 'unemployment', 'unemployment')
    dates = pd.bdate_range('2020-02-06', '2020-03-10')
    before, _ = align_updates(pd.DataFrame([old]), dates, 'unemployment')
    after, lineage = align_updates(pd.DataFrame([old, new]), dates, 'unemployment')
    pd.testing.assert_frame_equal(before.loc[:'2020-03-06'], after.loc[:'2020-03-06'])
    assert after.loc['2020-03-06', 'unemployment_rate'] == 4.0
    assert after.loc['2020-03-09', 'unemployment_rate'] == 4.2
    assert np.isclose(after.loc['2020-03-09', 'unemployment_change_1m_pp'], .1)
    assert lineage.loc['2020-03-09', 'unemployment_available_date'] == pd.Timestamp('2020-03-09')
    assert after.iloc[0].isna().all()


def test_inflation_uses_calendar_months_in_same_snapshot():
    row = snapshot_features(snapshot('2020-03-06', ['2019-02-01', '2020-01-01', '2020-02-01'], [100, 108, 110]), 'cpi', 'price_index')
    assert np.isclose(row['cpi_yoy_pct'], 10)
    assert np.isclose(row['cpi_mom_pct'], (110 / 108 - 1) * 100)
    missing = snapshot_features(snapshot('2020-03-06', ['2019-02-01', '2020-02-01'], [100, 110]), 'cpi', 'price_index')
    assert np.isnan(missing['cpi_mom_pct'])


def test_weekend_versions_keep_latest_and_skip_future():
    updates = pd.DataFrame([{'snapshot_date': pd.Timestamp(day), 'observation_month': pd.Timestamp('2020-01-01'), 'cpi_yoy_pct': value}
                            for day, value in [('2020-03-06', 1), ('2020-03-08', 2), ('2020-03-11', 3)]])
    daily, _ = align_updates(updates, pd.bdate_range('2020-03-06', '2020-03-10'), 'cpi')
    assert daily.loc['2020-03-09', 'cpi_yoy_pct'] == 2
    assert daily.loc['2020-03-10', 'cpi_yoy_pct'] == 2


def test_cached_snapshot_is_available_offline(tmp_path):
    folder = tmp_path / 'UNRATE'
    folder.mkdir()
    (folder / 'snapshot_2020-02-07_from_2004-01-01.json').write_text(json.dumps([
        {'date': '2020-01-01', 'value': '4.0'}, {'date': '2020-02-01', 'value': '.'}]))
    client = FredVintageClient(None, tmp_path, offline=True)
    frame = client.snapshot('UNRATE', '2020-02-07', '2004-01-01')
    assert frame['value'].iloc[0] == 4.0
    assert pd.isna(frame['value'].iloc[1])


def test_api_requests_type_two_and_paginates(tmp_path):
    client = FredVintageClient('test-key', tmp_path)
    calls = []
    def fake_get(endpoint, params):
        calls.append(params)
        month = '2020-01-01' if params['offset'] == 0 else '2020-02-01'
        return {'count': 2, 'observations': [{'date': month, 'UNRATE_20200306': '4.0'}]}
    client._get = fake_get
    frame = client.snapshot('UNRATE', '2020-03-06', '2004-01-01')
    assert len(frame) == 2
    assert frame['value'].eq(4.0).all()
    assert all(call['output_type'] == 2 and call['vintage_dates'] == '2020-03-06' for call in calls)


def test_rate_limit_retries_without_real_sleep(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import src.data.fred_vintages as module
    sleeps = []
    monkeypatch.setattr(module.time, 'sleep', sleeps.append)
    responses = iter([
        SimpleNamespace(status_code=429, headers={'Retry-After': '90'}),
        SimpleNamespace(status_code=200, headers={}, json=lambda: {'ok': True}),
    ])
    client = FredVintageClient('test-key', tmp_path)
    monkeypatch.setattr(client.session, 'get', lambda *args, **kwargs: next(responses))
    assert client._get('series/observations', {}) == {'ok': True}
    assert 90 in sleeps
