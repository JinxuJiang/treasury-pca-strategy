import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from src.portfolio.factor_positions import fit_sensitivities, choose_basket, generate_weekly_signals


def case():
    c=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/backtest_capacity.yaml').read_text())
    c['risk']['five_day_std_budget_fraction']=0.005
    dates=pd.bdate_range('2016-01-01',periods=1100,name='date')
    rng=np.random.default_rng(851)
    x=pd.DataFrame(rng.normal(size=(len(dates),4)),index=dates,
                   columns=['d_2Y_bp','d_5Y_bp','d_10Y_bp','d_30Y_bp'])
    beta=-np.array([[40,3,0,0],[2,50,5,0],[0,4,70,10],[0,0,10,150.]])
    pnl=x.to_numpy()@beta.T
    prices={s:pd.DataFrame({'close':100+np.cumsum(pnl[:,i])/spec['multiplier']},index=dates)
            for i,(s,spec) in enumerate(c['contracts'].items())}
    return c,x,prices,beta


def test_multivariate_recovery_and_future_invariance():
    c,x,p,beta=case();day=x.index[950]
    a,cov,audit=fit_sensitivities(x,p,c,day)
    np.testing.assert_allclose(a,beta,atol=1e-9)
    changed=x.copy();changed.loc[changed.index>day]+=10000
    other={s:v.copy() for s,v in p.items()}
    for v in other.values():v.loc[v.index>day,'close']+=10000
    b,cov2,audit2=fit_sensitivities(changed,other,c,day)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(cov,cov2)
    assert audit==audit2 and pd.Timestamp(audit['regression_end_date'])<=day


def test_jump_filter_and_no_filling_missing_prices():
    c,x,p,beta=case()
    # A permanent jump is rejected by residual filtering, not calendar dates;
    # missing prices must not create a bridged return.
    day=x.index[-1]
    jump=x.index[(x.index.month==3)&(x.index.day==15)][-1]
    for v in p.values():
        v.loc[v.index>=jump,'close']+=10
        v.loc[x.index[-20],'close']=np.nan
    a,_,audit=fit_sensitivities(x,p,c,day)
    np.testing.assert_allclose(a,beta,atol=1e-8)
    assert audit['regression_min_r_squared']>.999


def test_pc_basket_uses_full_matrix_and_actual_pnl_risk():
    c,x,p,beta=case()
    _,cov,_=fit_sensitivities(x,p,c,x.index[-1])
    loading=np.array([[.5,.5,.5,.5],[.5,.5,-.5,-.5],[.5,-.5,-.5,.5],[.5,-.5,.5,-.5]])
    matrix=-loading@beta.T
    n,status=choose_basket(np.array([0.,-1,0,0]),matrix,cov,c,1,-beta.T)
    assert status=='active'
    exposure=matrix@n
    assert exposure[1]<0
    assert np.linalg.norm(np.delete(exposure,1))/abs(exposure[1])<=.35
    assert np.sqrt(n@cov@n)<=50000
    assert np.abs(n)@np.array([s['assumed_margin'] for s in c['contracts'].values()])<=5e6
    assert np.abs(n)@np.abs(beta).sum(axis=1)<=c['risk']['gross_dv01_limit']


def test_bad_fit_fails_instead_of_falling_back():
    c,x,p,_=case()
    for v in p.values(): v['close']=100.
    with pytest.raises(ValueError,match='unreliable_'):
        fit_sensitivities(x,p,c,x.index[-1])


def test_normal_quarterly_dates_are_not_excluded():
    c,x,p,beta=case()
    _,_,audit=fit_sensitivities(x,p,c,x.index[-1])
    # This exact linear sample has no outliers and a fully observed 756-day tail.
    # Quarterly dates must be retained just like all other dates.
    assert audit['regression_observations']==c['sensitivity']['window']
    assert audit['pnl_observations']==c['risk']['volatility_window']
