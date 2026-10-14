"""Prepare cached historical macro snapshots and daily PIT features (no training).
work flow:
1. get historical vintage data when observation is added or revised. Also include the start
2. load or download data at each vintage date.
3. process data to get daily features.
4. align updates to daily features for every snapshot, making sure PIT is right
   Use the old vintage until the revision becomes available, then use the new one
5. Combine all indicators into daily feature and provenance tables.

Data flow:
    snapshot -> snapshots -> historical snapshot CSV
    snapshot -> feature row -> rows -> updates
    updates -> align_updates() -> daily, provenance
    daily -> features -> concatenate columns -> daily feature CSV
    provenance -> lineage -> concatenate columns -> daily provenance CSV
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.fred_vintages import FredVintageClient
from src.features.macro_features import snapshot_features, align_updates


def load_fred_api_key(env_path: Path = ROOT / '.env') -> str | None:
    """Read environment first, then a Git-ignored local .env; never print key."""
    key = os.environ.get('FRED_API_KEY', '').strip()
    if key:
        return key
    if env_path.exists():
        for line in env_path.read_text(encoding='utf-8-sig').splitlines():
            name, separator, value = line.strip().partition('=')
            if separator and name.strip() == 'FRED_API_KEY':
                return value.strip().strip('\"\'') or None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/macro.yaml')
    parser.add_argument('--offline', action='store_true', help='Use caches only')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    base = yaml.safe_load((ROOT / config['base_config']).read_text(encoding='utf-8'))
    levels = pd.read_csv(ROOT / base['paths']['processed_dir'] / 'treasury_yields.csv', parse_dates=['date'])
    dates = pd.DatetimeIndex(levels['date']).sort_values()
    start = dates.min().strftime('%Y-%m-%d')
    end = str(config['vintage_end'] or dates.max().strftime('%Y-%m-%d'))
    if pd.Timestamp(end) < dates.min():
        raise ValueError('vintage_end precedes sample start')
    
    # prepare download client, client is a FredVintageClient instance, use its methods to download data later.
    client = FredVintageClient(load_fred_api_key(), ROOT / config['cache_dir'], args.offline)
    output = ROOT / config['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    features, lineage = [], []

    # config['series'] is a dict of dict. Each item: 'unemployment': {'id': 'UNRATE', 'kind': 'unemployment'}
    for name, spec in config['series'].items():
        # use the FredVintageClient method vintage_dates to get list of the date of release and amendment.
        versions = client.vintage_dates(spec['id'], start, end)
        # Start-date snapshot initializes information already known at sample start.
        versions = sorted(set([start, *versions]))
        snapshots, rows = [], []

        for number, vintage in enumerate(versions, 1):
            # use the FredVintageClient method snapshot to get the real data of the vintage.
            snapshot = client.snapshot(spec['id'], vintage, str(config['observation_start']))
            snapshots.append(snapshot)
            # use snapshot_features to get the feature of the snapshot, one snapshot only have one feature.
            rows.append(snapshot_features(snapshot, name, spec['kind']))
            if number == 1 or number % 25 == 0 or number == len(versions):
                print(f'{spec["id"]}: {number}/{len(versions)} snapshots', flush=True)
        updates = pd.DataFrame(rows)

        # Match each signal date to the latest feature update available at that time.
        # Use the old vintage until the revision becomes available, then use the new one.
        daily, provenance = align_updates(updates, dates, name, int(config['availability_lag_sessions']))
        pd.concat(snapshots, ignore_index=True).to_csv(output / f'{name}_snapshots.csv', index=False)
        updates.to_csv(output / f'{name}_updates.csv', index=False)

        features.append(daily)
        lineage.append(provenance)
    # concat different macro data into one dataframe
    daily = pd.concat(features, axis=1)
    daily.to_csv(output / 'daily_macro_features.csv', index_label='date')
    pd.concat(lineage, axis=1).to_csv(output / 'daily_macro_lineage.csv', index_label='date')
    print(f'Saved {len(daily)} daily rows; {daily.notna().all(axis=1).sum()} complete. Outputs: {output}')


if __name__ == '__main__':
    main()
