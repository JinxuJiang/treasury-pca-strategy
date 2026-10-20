import hashlib
import json

import pandas as pd
import pytest

from src.data.futures_prices import parse_futures_chart, prepare_futures_data


def chart_payload():
    # UTC dates differ from New York trading dates: retain provider timezone.
    return {'chart': {'error': None, 'result': [{
        'meta': {'exchangeTimezoneName': 'America/New_York'},
        'timestamp': [1704153600, 1704240000],
        'indicators': {'quote': [{'open': [100., None], 'high': [102., None],
                                 'low': [99., None], 'close': [101., None],
                                 'volume': [200, None]}],
                       'adjclose': [{'adjclose': [101., None]}]},
    }]}}


def test_exchange_dates_and_missing_bars_are_preserved():
    frame = parse_futures_chart(chart_payload())
    assert frame.index.tolist() == [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')]
    assert frame.iloc[1].isna().all()
    assert frame.iloc[0]['close'] == 101.


def test_reject_duplicate_dates_and_provider_errors():
    payload = chart_payload()
    payload['chart']['result'][0]['timestamp'][1] = payload['chart']['result'][0]['timestamp'][0]
    with pytest.raises(ValueError, match='Duplicate'):
        parse_futures_chart(payload)
    with pytest.raises(ValueError, match='Yahoo chart error'):
        parse_futures_chart({'chart': {'error': {'description': 'Unavailable'}, 'result': None}})


def test_offline_rebuild_preserves_raw_and_marks_unverified(tmp_path, monkeypatch):
    raw = tmp_path/'raw'
    raw.mkdir()
    content = json.dumps(chart_payload()).encode()
    (raw/'ZN.json').write_bytes(content)
    def no_network(*args, **kwargs):
        raise AssertionError('Offline mode must not access network')
    monkeypatch.setattr('requests.Session.get', no_network)
    config = {'futures': {'symbols': ['ZN'], 'raw_dir': 'raw', 'output_dir': 'out',
                          'start_date': '2024-01-01', 'end_date': '2024-01-03'}}
    prepare_futures_data(config, tmp_path, offline=True)
    frame = pd.read_csv(tmp_path/'out/ZN.csv')
    assert len(frame) == 2 and frame.close.isna().sum() == 1
    manifest = json.loads((tmp_path/'out/manifest.json').read_text())
    assert 'UNVERIFIED' in manifest['status']
    assert manifest['series'][0]['raw_sha256'] == hashlib.sha256(content).hexdigest()
    assert (raw/'ZN.json').read_bytes() == content
    (raw/'ZN.json').unlink()
    with pytest.raises(FileNotFoundError, match='Missing cached futures'):
        prepare_futures_data(config, tmp_path, offline=True)


def test_invalid_download_does_not_replace_good_cache(tmp_path, monkeypatch):
    raw = tmp_path/'raw'
    raw.mkdir()
    original = json.dumps(chart_payload()).encode()
    (raw/'ZN.json').write_bytes(original)
    class Response:
        content = b'{"chart":{"error":{"code":"Not Found"},"result":null}}'
        def raise_for_status(self):
            pass
    monkeypatch.setattr('requests.Session.get', lambda *args, **kwargs: Response())
    config = {'futures': {'symbols': ['ZN'], 'raw_dir': 'raw', 'output_dir': 'out',
                          'start_date': '2024-01-01', 'end_date': '2024-01-03'}}
    with pytest.raises(ValueError, match='Yahoo chart error'):
        prepare_futures_data(config, tmp_path)
    assert (raw/'ZN.json').read_bytes() == original
    assert not (tmp_path/'out/ZN.csv').exists()
