"""Run Backtrader on precomputed weekly PC or unified-curve target positions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtest.futures_engine import run_factor_backtest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/backtest_capacity.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    signal_dir = ROOT/config['signal_dir']
    signal_path = signal_dir/'weekly_targets.csv'
    signal_manifest = json.loads((signal_dir/'manifest.json').read_text())
    if signal_manifest['config'] != config:
        raise ValueError('Config changed after signal generation; regenerate signals first')
    # read signals and prices
    signals = pd.read_csv(signal_path)
    prices = {s: pd.read_csv(ROOT/config['futures_dir']/f'{s}.csv', parse_dates=['date']).set_index('date')
              for s in config['contracts']}
    output = ROOT/config['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    strategies = ['CURVE'] if config.get('strategy', 'pc') == 'curve' else ['PC1', 'PC2', 'PC3']
    for pc in strategies:
        eq, orders, positions, audit, metrics = run_factor_backtest(
            signals[signals.strategy == pc], prices, config)
        folder = output/pc
        folder.mkdir(exist_ok=True)
        eq.to_csv(folder/'equity.csv', index_label='date')
        orders.to_csv(folder/'orders.csv', index=False)
        positions.to_csv(folder/'positions.csv', index=False)
        audit.to_csv(folder/'data_quality.csv', index=False)
        (folder/'metrics.json').write_text(json.dumps(metrics, indent=2))
        summaries.append(dict(strategy=pc, **{k: v for k, v in metrics.items() if k != 'open_positions_at_end'}))
        axes[0].plot(eq.index, eq.nav, label=pc)
        axes[1].plot(eq.index, eq.drawdown, label=pc)
    axes[0].axhline(1, color='gray', lw=.8, label='Cash (zero interest)')
    axes[0].set_ylabel('NAV'); axes[1].set_ylabel('Drawdown')
    axes[0].set_title('Exploratory weekly strategies | regression sensitivities, unverified Yahoo rolls')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/'equity_comparison.png', dpi=150); plt.close(fig)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/'summary.csv', index=False)
    files = [signal_path, signal_dir/'manifest.json', args.config,
             *[ROOT/config['futures_dir']/f'{s}.csv' for s in config['contracts']]]
    manifest = dict(config=config, signal_manifest=signal_manifest,
                    limitations=['Yahoo roll selection/back-adjustment not verified; roll gaps may contaminate P&L',
                                 'Empirical regression sensitivities are not CTD/CF valuation DV01',
                                 'PC exposure diagnostics depend on the estimated regression mapping',
                                 'Assumed margins/fees are research parameters, not historical exchange schedules',
                                 'No explicit rollover transactions or rollover costs without a contract map',
                                 'Missing bars: carry previous valuation, no basket fills; orders may be delayed',
                                 'Weekly next-open holding differs from close-to-five-day model label',
                                 ('One reconstructed-curve portfolio' if len(strategies) == 1 else 'Three separate accounts; not a combined portfolio')],
                    inputs=[dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(summary.to_string(index=False)); print(f'Outputs: {output}')


if __name__ == '__main__':
    main()
