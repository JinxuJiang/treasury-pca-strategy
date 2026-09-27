from pathlib import Path

import pandas as pd
import pytest

from src.data.treasury_yields import build_analysis_datasets, parse_treasury_xml


SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
 xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
 xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata">
 <entry><content type="application/xml"><m:properties>
  <d:NEW_DATE m:type="Edm.DateTime">2024-01-02T00:00:00</d:NEW_DATE>
  <d:BC_2YEAR m:type="Edm.Double">4.33</d:BC_2YEAR>
  <d:BC_5YEAR m:type="Edm.Double">3.93</d:BC_5YEAR>
  <d:BC_10YEAR m:type="Edm.Double">3.95</d:BC_10YEAR>
  <d:BC_30YEAR m:type="Edm.Double">4.08</d:BC_30YEAR>
 </m:properties></content></entry>
</feed>"""


def test_parse_official_xml_shape(tmp_path: Path) -> None:
    source = tmp_path / "sample.xml"
    source.write_text(SAMPLE_XML, encoding="utf-8")
    frame = parse_treasury_xml(source)
    assert list(frame.columns) == ["date", "2Y", "5Y", "10Y", "30Y"]
    assert frame.loc[0, "10Y"] == 3.95


def test_complete_case_and_basis_point_changes() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "2Y": [4.00, 4.01, 4.02],
            "5Y": [4.10, 4.08, 4.07],
            "10Y": [4.20, 4.20, 4.21],
            "30Y": [4.30, None, 4.33],
        }
    )
    levels, changes, report = build_analysis_datasets(raw, ["2Y", "5Y", "10Y", "30Y"])
    assert len(levels) == 2
    assert len(changes) == 1
    assert changes.loc[0, "d_2Y_bp"] == pytest.approx(2.0)
    assert changes.loc[0, "d_30Y_bp"] == pytest.approx(3.0)
    assert report["missing_by_maturity_before_complete_case_filter"]["30Y"] == 1


def test_long_gap_does_not_create_a_daily_change() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2002-02-15", "2006-02-09"]),
            "2Y": [3.0, 4.0],
            "5Y": [3.1, 4.1],
            "10Y": [3.2, 4.2],
            "30Y": [3.3, 4.3],
        }
    )
    _, changes, report = build_analysis_datasets(raw, ["2Y", "5Y", "10Y", "30Y"])
    assert changes.empty
    assert report["changes_removed_for_long_gaps"] == 1
