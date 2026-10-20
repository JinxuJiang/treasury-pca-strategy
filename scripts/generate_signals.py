"""Generate weekly PC or reconstructed-curve target positions before Backtrader."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.portfolio.factor_positions import generate_weekly_signals
from src.portfolio.curve_positions import generate_curve_signals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/backtest_capacity.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    forecast = ROOT/config['forecast_dir']
    processed = ROOT/config['processed_dir']
    forecast_manifest = json.loads((forecast/'manifest.json').read_text())
    if int(forecast_manifest['config']['forecast']['horizon_days']) != 5:
        raise ValueError('This first weekly strategy expects the existing five-day forecasts')
    if forecast_manifest['model'] != config['model']:
        raise ValueError('Forecast model does not match backtest config')
    paths = [forecast/'daily_oos_predictions.csv', forecast/'expanding_pca_loadings.csv',
             processed/'treasury_yield_changes_bp.csv']
    predictions = pd.read_csv(paths[0], parse_dates=['prediction_date']).set_index('prediction_date')
    loadings = pd.read_csv(paths[1])
    changes = pd.read_csv(paths[2], parse_dates=['date']).set_index('date')
    if config.get('sensitivity', {}).get('method') != 'rolling_regression':
        raise ValueError('Only rolling_regression is supported by current portfolio code')
    prices = {}
    for symbol in config['contracts']:
        path = ROOT/config['futures_dir']/f'{symbol}.csv'
        paths.append(path)
        prices[symbol] = pd.read_csv(path, parse_dates=['date']).set_index('date')
    if config.get('strategy', 'pc') == 'curve':
        yield_path = forecast/'yield_change_predictions_bp.csv'
        yield_forecasts = pd.read_csv(yield_path, parse_dates=['prediction_date']).set_index('prediction_date')
        paths.append(yield_path)
        signals = generate_curve_signals(predictions, yield_forecasts, changes, prices, config)
    else:
        signals = generate_weekly_signals(predictions, loadings, changes, prices, config)
    sizing = 'rolling multivariate USD P&L / yield-bp regression; basket risk from historical five-day futures P&L; fixed initial-cash budget'

    output = ROOT/config['signal_dir']
    output.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output/'weekly_targets.csv', index=False)
    manifest = dict(config=config,
                    status='EXPLORATORY: empirical curve sensitivities, not CTD DV01; no calendar roll exclusions; residual outlier filtering does not verify roll adjustment',
                    timing='week last exchange session close signal, next exchange session open order',
                    forecast_horizon_note='5 Treasury observations from signal close; weekly next-open holding is not the identical horizon',
                    sizing=sizing,
                    inputs=[dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                            for p in [*paths, forecast/'manifest.json', args.config]])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(signals.groupby(['strategy', 'status']).size().to_string())
    print(f'Signals: {output}')


if __name__ == '__main__':
    main()
