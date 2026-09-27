"""Download official Treasury yields and build the first research dataset."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/base.yaml")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--overwrite-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_config = config["data"]
    path_config = config["paths"]

    start_year = args.start_year or int(data_config["start_year"])
    end_year = args.end_year or data_config.get("end_year") or date.today().year
    end_year = int(end_year)
    if start_year < 1990 or end_year < start_year or end_year > date.today().year:
        raise ValueError("Year range must be between 1990 and the current year")

    raw_dir = PROJECT_ROOT / path_config["raw_dir"]
    processed_dir = PROJECT_ROOT / path_config["processed_dir"]
    overwrite = bool(args.overwrite_raw or data_config.get("overwrite_raw", False))

    xml_paths = []
    for year in range(start_year, end_year + 1):
        destination = raw_dir / f"daily_treasury_yield_curve_{year}.xml"
        print(f"[{year}] {'refreshing' if overwrite else 'loading/downloading'} {destination}")
        xml_paths.append(
            download_year(
                year,
                destination,
                timeout=int(data_config.get("request_timeout_seconds", 60)),
                overwrite=overwrite,
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


if __name__ == "__main__":
    main()
