"""Generate daily macro-augmented forecasts with monthly expanding refits."""
from pathlib import Path
import sys
import hashlib
import json
import argparse

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.walk_forward import run_monthly_expanding_forecast
from src.factors.pca_factors import fit_pca
from src.features.pca_features import forward_sum_and_end_date


def metrics(actual, predicted):
    '''calculate the metrics for comparing the actual and predicted values.'''
    return dict(R2=float(r2_score(actual, predicted)),
                MSE=float(np.mean((actual-predicted)**2)),
                RMSE=float(np.sqrt(np.mean((actual-predicted)**2))),
                MAE=float(np.mean(np.abs(actual-predicted))),
                correlation=float(np.corrcoef(actual,predicted)[0,1]) if np.std(predicted)>0 else None)


def load_macro(folder, dates):
    '''load and validate macro features.'''
    features = pd.read_csv(folder/'daily_macro_features.csv', parse_dates=['date']).set_index('date')
    lineage = pd.read_csv(folder/'daily_macro_lineage.csv', parse_dates=['date']).set_index('date')
    if features.index.has_duplicates or lineage.index.has_duplicates:
        raise ValueError('Duplicate macro dates')
    if not dates.isin(features.index).all() or not dates.isin(lineage.index).all():
        raise ValueError('Macro files do not cover all Treasury dates; rerun preparation')
    features, lineage = features.loc[dates], lineage.loc[dates]
    if not np.isfinite(features.to_numpy()[features.notna().to_numpy()]).all():
        raise ValueError('Nonfinite macro features')
    for name in ('unemployment','cpi','core_pce'):
        available = pd.to_datetime(lineage[f'{name}_available_date'])
        vintage = pd.to_datetime(lineage[f'{name}_snapshot_date'])
        known = available.notna()
        if not ((available[known] <= dates[known]).all() and (vintage[known] < available[known]).all()):
            raise ValueError('Macro lineage violates next-session availability')
    return features, features.notna().all(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['ridge','lgbm'], default='ridge')
    args = parser.parse_args()
    prediction_prefix = args.model
    config = yaml.safe_load((ROOT/'config/base.yaml').read_text(encoding='utf-8'))
    fc = config['forecast']
    maturities = config['data']['maturities']
    horizon = int(fc['horizon_days'])
    k = int(config['pca']['retained_components'])
    processed = ROOT/config['paths']['processed_dir']
    # load data
    changes = pd.read_csv(processed/'treasury_yield_changes_bp.csv',parse_dates=['date']).set_index('date')
    levels = pd.read_csv(processed/'treasury_yields.csv',parse_dates=['date']).set_index('date')
    dates = changes.index.intersection(levels.index).sort_values()
    changes = changes.loc[dates,[f'd_{m}_bp' for m in maturities]]
    levels = levels.loc[dates,maturities]
    extra, mask = load_macro(processed/'macro',dates)
    output = ROOT/'results/forecast'/args.model
    output.mkdir(parents=True,exist_ok=True)
    mask.rename('included').to_csv(output/'sample_mask.csv',index_label='date')
    paths = [processed/'treasury_yield_changes_bp.csv',processed/'treasury_yields.csv',
             processed/'macro/daily_macro_features.csv',processed/'macro/daily_macro_lineage.csv',ROOT/'config/base.yaml']
    manifest = dict(config=config, model=args.model, macro_columns=list(extra.columns), missing_policy='exclude rows with missing macro features; no interpolation',
                    excluded_dates=[str(d.date()) for d in dates[~mask]],
                    inputs=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    print(f'Running {args.model} with macro features ...', flush=True)

    # run the monthly expanding forecast, which is the core of the forecasting
    result = run_monthly_expanding_forecast(
        changes, levels, maturities, pd.Timestamp(fc['first_oos_date']),
        horizon_days=horizon, retained_components=k,
        validation_years=int(fc['validation_years']), ridge_alphas=fc['ridge_alphas'],
        extra_features=extra, sample_mask=mask, model_type=args.model,
    )
    result.predictions.to_csv(output/'daily_oos_predictions.csv', index=False)
    result.model_diagnostics.to_csv(output/'model_diagnostics.csv', index=False)
    result.pca_loadings.to_csv(output/'expanding_pca_loadings.csv', index=False)
    predictions = result.predictions.set_index('prediction_date')
    rows = []
    for i in range(1, k+1):
        pc = f'PC{i}'
        pair = predictions[[f'actual_{pc}', f'{prediction_prefix}_{pc}']].dropna()
        rows.append(dict(target=pc, model=args.model, n=len(pair),
                         **metrics(pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy())))
    pc_metrics = pd.DataFrame(rows)
    pc_metrics.to_csv(output/'pc_metrics.csv', index=False)
    targets, _ = forward_sum_and_end_date(changes, horizon)
    yields = pd.DataFrame(index=predictions.index, columns=maturities, dtype=float)

    # restore the predicted PCs to yield changes using the PCA loadings fitted on the training data
    for cutoff, group in predictions.groupby('model_cutoff_date'):
        pca = fit_pca(changes.loc[:pd.Timestamp(cutoff)], n_components=changes.shape[1])
        yields.loc[group.index] = (
            group[[f'{prediction_prefix}_PC{i}' for i in range(1, k+1)]].to_numpy() # remove the columns and index, changing to numpy array
            @ pca.components_[:k] + horizon*pca.mean_
        )
    yields.to_csv(output/'yield_change_predictions_bp.csv', index_label='prediction_date')

    # use metrics to evaluate the predicted yields against the actual yields
    rows = []
    for m in maturities:
        actual = targets.loc[predictions.index, f'target_d_{m}_bp_{horizon}d']
        valid = actual.notna()
        rows.append(dict(target=m, model=args.model, n=int(valid.sum()),
                         **metrics(actual[valid].to_numpy(), yields.loc[valid, m].to_numpy())))
    yield_metrics = pd.DataFrame(rows)
    yield_metrics.to_csv(output/'yield_metrics.csv', index=False)

    print('PC metrics\n'+pc_metrics.to_string(index=False), flush=True)
    print('Yield metrics (bp, MSE in bp squared)\n'+yield_metrics.to_string(index=False), flush=True)
    print(f'Predictions: {len(predictions)}, completed: {predictions.actual_PC1.notna().sum()}; '
          f'dates {predictions.index.min()} to {predictions.index.max()}', flush=True)
    print(f'Outputs: {output}', flush=True)


if __name__ == '__main__':
    main()
