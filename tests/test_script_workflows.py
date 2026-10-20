"""Check the public preparation and macro-only prediction workflows."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from scripts import prepare_data, run_forecasts
from src.factors.pca_factors import fit_pca


def test_combined_preparation_uses_same_base_config_in_order(tmp_path, monkeypatch):
    base = {'paths': {'processed_dir': 'custom'}}
    macro = {'output_dir': 'custom_macro'}
    base_path, macro_path = tmp_path/'base.yaml', tmp_path/'macro.yaml'
    base_path.write_text(yaml.safe_dump(base))
    macro_path.write_text(yaml.safe_dump(macro))
    monkeypatch.setattr(sys, 'argv', ['prepare_data.py', '--config', str(base_path),
                                    '--macro-config', str(macro_path), '--offline'])
    calls = []
    monkeypatch.setattr(prepare_data, 'prepare_treasury',
                        lambda config, args: calls.append(('treasury', config, args.offline)))
    monkeypatch.setattr(prepare_data, 'prepare_macro',
                        lambda config, base, offline: calls.append(('macro', config, base, offline)))
    monkeypatch.setattr(prepare_data, 'prepare_futures_data',
                        lambda config, root, offline: calls.append(('futures', config, offline)))
    prepare_data.main()
    assert calls == [('treasury', base, True), ('macro', macro, base, True),
                     ('futures', base, True)]


def test_macro_prediction_runs_once_and_restores_five_day_mean(tmp_path, monkeypatch):
    root = tmp_path
    (root/'config').mkdir()
    processed = root/'data/processed'
    processed.mkdir(parents=True)
    mats = ['2Y', '5Y', '10Y', '30Y']
    dates = pd.bdate_range('2019-01-01', periods=45, name='date')
    rng = np.random.default_rng(18)
    changes = pd.DataFrame(rng.normal(size=(45, 4)) + 2, index=dates,
                           columns=[f'd_{m}_bp' for m in mats])
    levels = pd.DataFrame(changes.to_numpy().cumsum(axis=0)/100+3, index=dates, columns=mats)
    changes.to_csv(processed/'treasury_yield_changes_bp.csv', index_label='date')
    levels.to_csv(processed/'treasury_yields.csv', index_label='date')
    config = dict(data=dict(maturities=mats), paths=dict(processed_dir='data/processed'),
                  pca=dict(retained_components=3), forecast=dict(horizon_days=5,
                  first_oos_date=str(dates[30].date()), validation_years=2, ridge_alphas=[1.]))
    (root/'config/base.yaml').write_text(yaml.safe_dump(config))
    macro_dir = processed/'macro'
    macro_dir.mkdir()
    extra = pd.DataFrame({'unemployment_rate': np.full(45, 4.)}, index=dates)
    extra.iloc[31] = np.nan
    extra.to_csv(macro_dir/'daily_macro_features.csv', index_label='date')
    lineage = pd.DataFrame(index=dates)
    for name in ['unemployment', 'cpi', 'core_pce']:
        lineage[f'{name}_available_date'] = dates[0]
        lineage[f'{name}_snapshot_date'] = dates[0]-pd.Timedelta(days=1)
    lineage.to_csv(macro_dir/'daily_macro_lineage.csv', index_label='date')
    calls = []
    signal_dates = dates[[30, 32, 33, 34]]
    cutoff = dates[29]
    pca = fit_pca(changes.loc[:cutoff], n_components=4)
    def forecast(*args, **kwargs):
        calls.append(kwargs)
        pd.testing.assert_frame_equal(kwargs['extra_features'], extra, check_freq=False)
        assert not kwargs['sample_mask'].iloc[31]
        pred = pd.DataFrame({'prediction_date': signal_dates, 'model_cutoff_date': cutoff})
        for i in range(1, 4):
            pred[f'actual_PC{i}'] = np.arange(4)+i
            pred[f'ridge_PC{i}'] = np.arange(4)*.1+i
        return SimpleNamespace(predictions=pred, model_diagnostics=pd.DataFrame({'n': [1]}),
                               pca_loadings=pd.DataFrame({'loading': [1.]}))
    monkeypatch.setattr(run_forecasts, 'ROOT', root)
    monkeypatch.setattr(run_forecasts, 'run_monthly_expanding_forecast', forecast)
    monkeypatch.setattr(sys, 'argv', ['run_forecasts.py', '--model', 'ridge'])
    run_forecasts.main()
    assert len(calls) == 1
    out = root/'results/forecast/ridge'
    predictions = pd.read_csv(out/'daily_oos_predictions.csv')
    restored = pd.read_csv(out/'yield_change_predictions_bp.csv')[mats]
    expected = predictions[[f'ridge_PC{i}' for i in range(1, 4)]].to_numpy() @ pca.components_[:3]+5*pca.mean_
    np.testing.assert_allclose(restored, expected)
    assert pd.read_csv(out/'pc_metrics.csv').model.unique().tolist() == ['ridge']
    assert pd.read_csv(out/'yield_metrics.csv').model.unique().tolist() == ['ridge']
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['excluded_dates'] == [str(dates[31].date())]
    assert not (root/'results/archive/macro_experiment').exists()
