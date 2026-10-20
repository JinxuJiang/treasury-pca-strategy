"""Regenerate weekly positions, backtest both strategy families and compare results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def build_commands(args, configs):
    commands = []
    if args.refresh_data:
        commands.append(['prepare_data.py', *(['--offline'] if args.offline else [])])
    if args.refit:
        models = sorted({config['model'] for _, config in configs})
        commands.extend(['run_forecasts.py', '--model', model] for model in models)
    for path, _ in configs:
        commands.extend([[script, '--config', str(path)]
                         for script in ('generate_signals.py', 'run_backtest.py')])
    return [[sys.executable, str(ROOT/'scripts'/command[0]), *command[1:]]
            for command in commands]


def summarize(configs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows, sources = [], []
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    try:
        for path, config in configs:
            folder = ROOT/config['output_dir']
            summary = pd.read_csv(folder/'summary.csv')
            for row in summary.to_dict('records'):
                strategy = row['strategy']
                row.update(initial_cash=config['initial_cash'],
                           five_day_std_budget_fraction=config['risk']['five_day_std_budget_fraction'],
                           max_margin_fraction=config['risk']['max_margin_fraction'],
                           config_file=str(path), result_dir=str(folder/strategy))
                rows.append(row)
                equity = pd.read_csv(folder/strategy/'equity.csv', parse_dates=['date'])
                axes[0].plot(equity.date, equity.nav, label=strategy)
                axes[1].plot(equity.date, equity.drawdown, label=strategy)
            sources.append(dict(config_file=str(path), config=config,
                                manifest=str(folder/'manifest.json')))
        output.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(rows)
        table.to_csv(output/'summary.csv', index=False)
        axes[0].axhline(1, color='gray', lw=.8)
        axes[0].set_ylabel('NAV'); axes[1].set_ylabel('Drawdown')
        axes[0].set_title('Independent weekly strategies | regression sensitivities')
        for ax in axes:
            ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output/'equity_comparison.png', dpi=150)
        (output/'manifest.json').write_text(json.dumps(dict(
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            interpretation='Four independent accounts, not a combined portfolio; source manifests contain input hashes',
            sources=sources), indent=2))
        print('\nFour-strategy comparison:', flush=True)
        print(table[['strategy','total_return','annualized_return','annualized_volatility',
                     'max_drawdown','sharpe_zero_rf','total_cost']].to_string(index=False))
        print(f'Comparison saved: {output}', flush=True)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pc-config', type=Path, default=ROOT/'config/backtest_capacity.yaml')
    parser.add_argument('--curve-config', type=Path, default=ROOT/'config/backtest_curve.yaml')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/backtest/comparison_regression')
    parser.add_argument('--refresh-data', action='store_true', help='Run prepare_data first')
    parser.add_argument('--refit', action='store_true', help='Regenerate historical model forecasts first')
    parser.add_argument('--offline', action='store_true', help='Use cached data with --refresh-data')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running or writing outputs')
    args = parser.parse_args()
    if args.offline and not args.refresh_data:
        parser.error('--offline requires --refresh-data')
    configs = []
    for path in (args.pc_config, args.curve_config):
        path = (ROOT/path).resolve()
        configs.append((path, yaml.safe_load(path.read_text())))
    if configs[0][1].get('strategy','pc') != 'pc' or configs[1][1].get('strategy') != 'curve':
        parser.error('Expected PC configuration followed by curve configuration')
    for key in ('signal_dir','output_dir'):
        if (ROOT/configs[0][1][key]).resolve() == (ROOT/configs[1][1][key]).resolve():
            parser.error(f'Two strategy families must have distinct {key}')
    commands = build_commands(args, configs)
    for i, command in enumerate(commands, 1):
        print(f'[{i}/{len(commands)}] '+ ' '.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        summarize(configs, (ROOT/args.output_dir).resolve())


if __name__ == '__main__':
    main()
