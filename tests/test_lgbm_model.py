import numpy as np
import pandas as pd
import pytest

pytest.importorskip('lightgbm')
from src.models.lgbm_model import fit_lgbm, choose_lgbm_params, CANDIDATES


def test_future_rows_do_not_affect_fitting_or_tuning():
    rng=np.random.default_rng(9)
    frame=pd.DataFrame({'x':rng.normal(size=120),'target':rng.normal(size=120)})
    train=pd.Series(frame.index<80)
    validation=pd.Series((frame.index>=80)&(frame.index<100))
    params,error=choose_lgbm_params(frame,['x'],'target',train,validation)
    assert params in CANDIDATES and np.isfinite(error)
    altered=frame.copy(); altered.loc[100:,'target']=9999
    params2,error2=choose_lgbm_params(altered,['x'],'target',train,validation)
    assert params==params2 and error==error2
    model=fit_lgbm(frame,['x'],'target',train,params)
    assert model.predict(frame.loc[[100],['x']]).shape==(1,)


def test_daily_walk_forward_model_switch(monkeypatch):
    import src.models.lgbm_model as module
    from src.models.walk_forward import run_monthly_expanding_forecast
    monkeypatch.setattr(module,'CANDIDATES',[dict(num_leaves=3,min_child_samples=20,n_estimators=5)])
    dates=pd.bdate_range('2010-01-01',periods=1100)
    rng=np.random.default_rng(23)
    changes=pd.DataFrame(rng.normal(size=(len(dates),4)),index=dates,
                         columns=['d_2Y_bp','d_5Y_bp','d_10Y_bp','d_30Y_bp'])
    levels=pd.DataFrame(3+changes.to_numpy().cumsum(axis=0)/100,index=dates,
                        columns=['2Y','5Y','10Y','30Y'])
    result=run_monthly_expanding_forecast(changes,levels,list(levels.columns),dates[-2],model_type='lgbm')
    assert 'lgbm_PC1' in result.predictions
    assert 'ridge_PC1' not in result.predictions
    assert result.predictions.actual_PC1.isna().all()
    assert (pd.to_datetime(result.model_diagnostics.last_training_target_end_date)<=pd.to_datetime(result.model_diagnostics.model_cutoff_date)).all()
