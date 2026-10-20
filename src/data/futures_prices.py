"""Fetch Yahoo Treasury futures candidates without assuming roll adjustment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def parse_futures_chart(payload: dict) -> pd.DataFrame:
    """Keep missing bars; label dates in the provider's exchange timezone."""
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Yahoo chart error: {chart.get('error')}")
    result = chart["result"][0]
    timestamps = result.get("timestamp", [])
    if not timestamps:
        raise ValueError("Yahoo futures response contains no daily bars")
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(
        result["meta"]["exchangeTimezoneName"]
    ).tz_localize(None).normalize()
    frame = pd.DataFrame(result["indicators"]["quote"][0], index=dates)
    frame.index.name = "date"
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(frame):
        raise ValueError("Yahoo futures response lacks required OHLCV fields")
    frame = frame[required].apply(pd.to_numeric, errors="raise")
    adjusted = result["indicators"].get("adjclose")
    if adjusted:
        frame["adj_close"] = adjusted[0]["adjclose"]
    if frame.index.has_duplicates:
        raise ValueError("Duplicate futures trading dates")
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any() or (frame["volume"].dropna() < 0).any():
        raise ValueError("Invalid futures prices or volume")
    frame = frame.sort_index()
    if frame["close"].notna().sum() == 0:
        raise ValueError("Yahoo futures response contains no valid closing prices")
    return frame


def prepare_futures_data(config: dict, root: Path, offline: bool = False) -> None:
    """Refresh candidates online, or rebuild from validated raw JSON offline."""
    settings = config.get("futures", {})
    symbols = settings.get("symbols", ["ZT", "ZF", "ZN", "ZB"])
    raw_dir = root / settings.get("raw_dir", "data/raw/futures/yahoo_candidates")
    output = root / settings.get("output_dir", "data/processed/futures_candidates")
    start = pd.Timestamp(str(settings.get("start_date", "2000-01-01")))
    # Today is exclusive so an in-progress daily bar is never used.
    today = datetime.now(ZoneInfo("America/New_York")).date()
    end = pd.Timestamp(str(settings.get("end_date") or today))
    if end <= start:
        raise ValueError("Futures end date must be later than start date")
    raw_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
    )))
    records = []
    try:
        for symbol in symbols:
            if symbol not in ("ZT", "ZF", "ZN", "ZB"):
                raise ValueError(f"Unsupported Treasury futures symbol: {symbol}")
            path = raw_dir / f"{symbol}.json"
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}=F"
            if offline:
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached futures response: {path}")
                content = path.read_bytes()
            else:
                response = session.get(url, params={
                    "period1": int(start.tz_localize("UTC").timestamp()),
                    "period2": int(end.tz_localize("UTC").timestamp()),
                    "interval": "1d",
                }, timeout=int(settings.get("request_timeout_seconds", 60)))
                response.raise_for_status()
                content = response.content
            payload = json.loads(content)
            frame = parse_futures_chart(payload)
            frame = frame.loc[(frame.index >= start) & (frame.index < end)]
            if frame.empty or not frame["close"].notna().any():
                raise ValueError(f"No valid {symbol} prices within the requested dates")
            # Validate before replacing a previous successful download.
            if not offline:
                temporary = path.with_suffix(".json.tmp")
                temporary.write_bytes(content)
                temporary.replace(path)
            frame.to_csv(output / f"{symbol}.csv", date_format="%Y-%m-%d")
            record = dict(
                symbol=f"{symbol}=F", source_url=url,
                requested_start=str(start.date()), end_exclusive=str(end.date()),
                raw_sha256=hashlib.sha256(content).hexdigest(),
                raw_file=str(path.resolve()),
                raw_modified_at_utc=datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
                exchange_timezone=payload["chart"]["result"][0]["meta"]["exchangeTimezoneName"],
                rows=len(frame), missing_close=int(frame["close"].isna().sum()),
                first_date=str(frame.index.min().date()),
                last_date=str(frame.index.max().date()),
            )
            records.append(record)
            print(f"{symbol}: {len(frame)} daily rows, {record['first_date']} to "
                  f"{record['last_date']}; missing close: {record['missing_close']}", flush=True)
    finally:
        session.close()
    manifest = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(), offline=offline,
        status="candidate data; roll selection and back-adjustment are UNVERIFIED",
        missing_policy="preserve missing observations; no interpolation or forward fill",
        adjustment_method="UNVERIFIED; adj_close does not establish futures roll adjustment",
        series=records,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
