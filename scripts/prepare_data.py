"""Prepare Treasury yields and point-in-time macro features in one run."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.treasury_yields import (  # noqa: E402
    build_analysis_datasets,
    combine_years,
    download_year,
    save_processed,
)


from src.data.fred_vintages import FredVintageClient
from src.features.macro_features import snapshot_features, align_updates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/base.yaml")
    parser.add_argument("--macro-config", type=Path, default=PROJECT_ROOT / "config/macro.yaml")
    parser.add_argument("--offline", action="store_true", help="Use cached Treasury XML and FRED snapshots only")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--overwrite-raw", action="store_true")
    return parser.parse_args()


def prepare_treasury(config, args) -> None:
    data_config = config["data"]
    path_config = config["paths"]

    start_year = args.start_year or int(data_config["start_year"])
    end_year = args.end_year or data_config.get("end_year") or date.today().year
    end_year = int(end_year)
    if start_year < 1990 or end_year < start_year or end_year > date.today().year:
        raise ValueError("Year range must be between 1990 and the current year")

    raw_dir = PROJECT_ROOT / path_config["raw_dir"]
    processed_dir = PROJECT_ROOT / path_config["processed_dir"]
    overwrite_all_raw = bool(args.overwrite_raw or data_config.get("overwrite_raw", False))
    current_year = date.today().year

    xml_paths = []
    for year in range(start_year, end_year + 1):
        destination = raw_dir / f"daily_treasury_yield_curve_{year}.xml"
        refresh_this_year = not args.offline and (overwrite_all_raw or year == current_year)
        if args.offline and not destination.exists():
            raise FileNotFoundError(f"Missing cached Treasury XML: {destination}")
        status = "refreshing" if refresh_this_year else "loading/downloading"
        print(f"[{year}] {status} {destination}")
        xml_paths.append(
            download_year(
                year,
                destination,
                timeout=int(data_config.get("request_timeout_seconds", 60)),
                overwrite=refresh_this_year,
            )
        )

    raw = combine_years(xml_paths)
    levels, changes, report = build_analysis_datasets(
        raw,
        data_config["maturities"],
        max_change_gap_days=int(data_config.get("max_change_gap_days", 7)),
    )
    save_processed(levels, changes, report, processed_dir)
    print(
        f"Saved {len(levels):,} complete observations "
        f"({report['first_complete_date']} to {report['last_complete_date']})."
    )


def load_fred_api_key(env_path: Path = PROJECT_ROOT / '.env') -> str | None:
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



def prepare_macro(config, base, offline=False):
    levels = pd.read_csv(PROJECT_ROOT / base['paths']['processed_dir'] / 'treasury_yields.csv', parse_dates=['date'])
    dates = pd.DatetimeIndex(levels['date']).sort_values()
    start = dates.min().strftime('%Y-%m-%d')
    end = str(config['vintage_end'] or dates.max().strftime('%Y-%m-%d'))
    if pd.Timestamp(end) < dates.min():
        raise ValueError('vintage_end precedes sample start')

    # prepare download client, client is a FredVintageClient instance, use its methods to download data later.
    client = FredVintageClient(load_fred_api_key(), PROJECT_ROOT / config['cache_dir'], offline)
    output = PROJECT_ROOT / config['output_dir']
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


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    macro_config = yaml.safe_load(args.macro_config.read_text(encoding="utf-8"))
    print("Preparing Treasury data ...", flush=True)
    prepare_treasury(config, args)
    print("Preparing macro data ...", flush=True)
    prepare_macro(macro_config, config, offline=args.offline)


if __name__ == "__main__":
    main()
