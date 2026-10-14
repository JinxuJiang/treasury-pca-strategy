"""Download and cache complete ALFRED snapshots without exposing API keys."""
from __future__ import annotations

import json
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests


class FredVintageClient:
    def __init__(self, api_key: str | None, cache_dir: Path, offline: bool = False):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self.session = requests.Session()
        self._last_request = 0.0

    def _get(self, endpoint: str, params: dict) -> dict:
        if self.offline:
            raise RuntimeError('Offline mode: requested data is not cached')
        if not self.api_key:
            raise RuntimeError('Set FRED_API_KEY in your environment before downloading')
        for attempt in range(6):
            # Pace uncached requests even when preceding responses were fast.
            time.sleep(max(0.0, 1.0 - (time.monotonic() - self._last_request)))
            self._last_request = time.monotonic()
            try:
                response = self.session.get(
                    f'https://api.stlouisfed.org/fred/{endpoint}',
                    params={**params, 'api_key': self.api_key, 'file_type': 'json'},
                    timeout=60,
                )
            except requests.RequestException:
                if attempt == 5:
                    raise RuntimeError('FRED request failed; check network access. Cached snapshots are preserved.') from None
                delay = min(5 * 2 ** attempt, 60)
                print(f'FRED network error; retrying in {delay}s ({attempt + 1}/5)', flush=True)
                time.sleep(delay)
                continue
            if response.status_code not in (429, 500, 502, 503, 504) or attempt == 5:
                break
            delay = min((60 if response.status_code == 429 else 5) * 2 ** attempt, 300)
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    try:
                        delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - time.time())
                    except (TypeError, ValueError, OverflowError):
                        pass
            print(f'FRED HTTP {response.status_code}; retrying in {delay:.0f}s ({attempt + 1}/5)', flush=True)
            time.sleep(delay)
        # Do not include response URLs: they contain the API key.
        if response.status_code != 200:
            raise RuntimeError(f'FRED request failed with HTTP {response.status_code}')
        return response.json()

    def vintage_dates(self, series: str, start: str, end: str) -> list[str]:
        """Cache version dates through end; a new end date discovers new versions.
           return the date of addition or admention data.
        """
        # if there are data file, we directly read it.
        path = self.cache_dir / series / f'versions_{start}_{end}.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        # if there are no data file, we request it from FRED API.
        dates, offset = [], 0
        while True:
            result = self._get('series/vintagedates', dict(
                series_id=series, realtime_start=start, realtime_end=end,
                limit=1000, offset=offset, sort_order='asc'))
            page = result['vintage_dates']
            dates.extend(page)
            offset += len(page)
            if offset >= int(result['count']):
                break
            if not page:
                raise RuntimeError('FRED pagination returned an empty incomplete page')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dates), encoding='utf-8')
        return dates

    def snapshot(self, series: str, vintage: str, observation_start: str) -> pd.DataFrame:
        """Get type 2 full snapshot, including revisions known on this vintage date."""
        path = self.cache_dir / series / f'snapshot_{vintage}_from_{observation_start}.json'
        if path.exists():
            rows = json.loads(path.read_text(encoding='utf-8'))
        # if there are no data file, we request it from FRED API.
        else:
            rows, offset = [], 0
            while True:
                result = self._get('series/observations', dict(
                    series_id=series, vintage_dates=vintage, output_type=2,
                    units='lin', observation_start=observation_start,
                    limit=100000, offset=offset))
                page = result['observations']
                rows.extend(page)
                offset += len(page)
                if offset >= int(result['count']):
                    break
                if not page:
                    raise RuntimeError('FRED pagination returned an empty incomplete page')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows), encoding='utf-8')
        if not rows:
            raise ValueError(f'No archived observations for {series} at {vintage}')
        frame = pd.DataFrame(rows)
        # Type 2 uses a vintage-specific value field, e.g. UNRATE_20060209.
        # Keep support for ordinary observation responses/older local caches.
        if 'value' not in frame.columns:
            vintage_field = f"{series}_{pd.Timestamp(vintage).strftime('%Y%m%d')}"
            if vintage_field not in frame.columns:
                raise ValueError(f'Unexpected snapshot fields for {series} at {vintage}: {list(frame.columns)}')
            frame = frame.rename(columns={vintage_field: 'value'})
        frame = frame.rename(columns={'date': 'observation_month'})
        frame['observation_month'] = pd.to_datetime(frame['observation_month'])
        frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
        frame['snapshot_date'] = pd.Timestamp(vintage)
        return frame[['snapshot_date', 'observation_month', 'value']].sort_values('observation_month')
