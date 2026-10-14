"""Download and prepare official U.S. Treasury par-yield curve data."""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd
import requests


TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xml"
)

XML_FIELDS: Mapping[str, str] = {
    "BC_2YEAR": "2Y",
    "BC_5YEAR": "5Y",
    "BC_10YEAR": "10Y",
    "BC_30YEAR": "30Y",
}

ATOM_NS = "{http://www.w3.org/2005/Atom}"
META_NS = "{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}"
DATA_NS = "{http://schemas.microsoft.com/ado/2007/08/dataservices}"


def treasury_year_url(year: int) -> str:
    return (
        f"{TREASURY_URL}?data=daily_treasury_yield_curve"
        f"&field_tdr_date_value={year}"
    )


def download_year(
    year: int,
    output_path: Path,
    timeout: int = 60,
    overwrite: bool = False,
    max_attempts: int = 3,
) -> Path:
    """Download one calendar year, preserving the official XML response."""
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Interest-rate-research/1.0"}
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                treasury_year_url(year), headers=headers, timeout=timeout
            )
            response.raise_for_status()
            if b"<feed" not in response.content or b"<entry" not in response.content:
                raise ValueError(f"Treasury response for {year} contains no entries")
            output_path.write_bytes(response.content)
            return output_path
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Failed to download Treasury data for {year}") from last_error


def parse_treasury_xml(xml_path: Path) -> pd.DataFrame:
    """Parse selected maturities from one official Treasury Atom XML file."""
    root = ET.parse(xml_path).getroot()
    rows: List[Dict[str, object]] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        properties = entry.find(f"{ATOM_NS}content/{META_NS}properties")
        if properties is None:
            continue
        date_node = properties.find(f"{DATA_NS}NEW_DATE")
        if date_node is None or not date_node.text:
            continue
        row: Dict[str, object] = {"date": date_node.text}
        for xml_name, maturity in XML_FIELDS.items():
            node = properties.find(f"{DATA_NS}{xml_name}")
            row[maturity] = node.text if node is not None else None
        rows.append(row)

    frame = pd.DataFrame(rows, columns=["date", *XML_FIELDS.values()])
    if frame.empty:
        raise ValueError(f"No Treasury observations parsed from {xml_path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    for maturity in XML_FIELDS.values():
        frame[maturity] = pd.to_numeric(frame[maturity], errors="coerce")
    return frame


def combine_years(xml_paths: Iterable[Path]) -> pd.DataFrame:
    """Combine annual files, sorting dates and rejecting conflicting duplicates."""
    frames = [parse_treasury_xml(path) for path in xml_paths]
    combined = pd.concat(frames, ignore_index=True).sort_values("date")
    conflicts = combined.groupby("date")[list(XML_FIELDS.values())].nunique(dropna=True)
    if (conflicts > 1).any(axis=None):
        bad_dates = conflicts.index[(conflicts > 1).any(axis=1)].strftime("%Y-%m-%d")
        raise ValueError(f"Conflicting duplicate dates: {', '.join(bad_dates[:5])}")
    return combined.drop_duplicates("date", keep="last").reset_index(drop=True)


def build_analysis_datasets(
    raw: pd.DataFrame, maturities: Sequence[str], max_change_gap_days: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Return complete yield levels, bp changes, and a data-quality report."""
    unknown = sorted(set(maturities) - set(XML_FIELDS.values()))
    if unknown:
        raise ValueError(f"Unsupported maturities: {unknown}")

    ordered = raw[["date", *maturities]].sort_values("date").copy()
    missing_by_maturity = ordered[list(maturities)].isna().sum().astype(int).to_dict()
    complete = ordered.dropna(subset=list(maturities)).reset_index(drop=True)
    if complete.empty:
        raise ValueError("No dates contain all requested maturities")
    if complete["date"].duplicated().any() or not complete["date"].is_monotonic_increasing:
        raise ValueError("Processed Treasury dates must be unique and increasing")
    if not complete[list(maturities)].apply(lambda col: col.between(0, 25)).all(axis=None):
        raise ValueError("Yield outside the expected 0%-25% validation range")

    observation_gaps = complete["date"].diff().dt.days
    changes = complete.set_index("date")[list(maturities)].diff().mul(100.0)
    changes.columns = [f"d_{maturity}_bp" for maturity in maturities]
    long_gap_count = int((observation_gaps > max_change_gap_days).sum())
    changes = changes.loc[observation_gaps.to_numpy() <= max_change_gap_days]
    changes = changes.dropna().reset_index()

    report: Dict[str, object] = {
        "source": "U.S. Treasury Daily Treasury Par Yield Curve Rates",
        "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "requested_maturities": list(maturities),
        "raw_observations": int(len(ordered)),
        "complete_observations": int(len(complete)),
        "daily_change_observations": int(len(changes)),
        "max_change_gap_days": int(max_change_gap_days),
        "changes_removed_for_long_gaps": long_gap_count,
        "missing_by_maturity_before_complete_case_filter": missing_by_maturity,
        "first_complete_date": complete["date"].min().strftime("%Y-%m-%d"),
        "last_complete_date": complete["date"].max().strftime("%Y-%m-%d"),
        "fill_policy": "No interpolation or calendar-day filling",
        "yield_unit": "percent",
        "change_unit": "basis_points",
    }
    return complete, changes, report


def save_processed(
    levels: pd.DataFrame,
    changes: pd.DataFrame,
    report: Mapping[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    levels.to_csv(output_dir / "treasury_yields.csv", index=False, date_format="%Y-%m-%d")
    changes.to_csv(
        output_dir / "treasury_yield_changes_bp.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    (output_dir / "treasury_data_quality.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
