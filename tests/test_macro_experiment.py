import numpy as np
import pandas as pd
import pytest

from scripts.run_forecasts import load_macro, metrics


def test_macro_loader_preserves_missing_and_rejects_future(tmp_path):
    dates=pd.bdate_range('2020-01-02',periods=3)
    pd.DataFrame({'unemployment_rate':[4.,np.nan,4.]},index=dates).to_csv(tmp_path/'daily_macro_features.csv',index_label='date')
    lineage=pd.DataFrame(index=dates)
    for name in ('unemployment','cpi','core_pce'):
        lineage[f'{name}_available_date']=dates[0]
        lineage[f'{name}_snapshot_date']=pd.Timestamp('2020-01-01')
    lineage.to_csv(tmp_path/'daily_macro_lineage.csv',index_label='date')
    features,mask=load_macro(tmp_path,dates)
    assert mask.tolist()==[True,False,True]
    assert features.iloc[1].isna().all()
    lineage.loc[dates[0],'cpi_available_date']=dates[-1]
    lineage.to_csv(tmp_path/'daily_macro_lineage.csv',index_label='date')
    with pytest.raises(ValueError,match='availability'):
        load_macro(tmp_path,dates)


def test_metrics_zero_baseline():
    result=metrics(np.array([-1.,1.]),np.array([0.,0.]))
    assert result['R2']==0
    assert result['MSE']==1
