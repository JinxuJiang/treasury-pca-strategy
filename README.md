# Interest Rate Investment

Research project for an interpretable U.S. Treasury yield-curve PCA strategy.

## Current stage: Treasury curve data

The initial dataset uses the U.S. Department of the Treasury's official Daily
Treasury Par Yield Curve Rates. The pipeline retains the official raw XML and
builds analysis-ready 2Y, 5Y, 10Y, and 30Y datasets without filling missing
observations.

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python -m pytest
```

Outputs:

- `data/raw/treasury/`: one immutable XML response per requested year;
- `data/processed/treasury_yields.csv`: clean yield levels in percent;
- `data/processed/treasury_yield_changes_bp.csv`: daily changes in basis points;
- `data/processed/treasury_data_quality.json`: audit summary.

By default, the processed PCA sample contains only dates for which all four
selected maturities are available. Weekends and holidays are not synthesized.
