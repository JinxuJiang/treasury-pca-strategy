from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.portfolio.curve_positions import generate_curve_signals
from src.portfolio.factor_positions import generate_weekly_signals

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



def test_curve_signals_are_causal_and_respect_risk():
    c,x,prices,beta=case()
    c['strategy']='curve'
    dates=pd.bdate_range(x.index[-30],x.index[-10])
    predictions=pd.DataFrame({'model_cutoff_date':str(x.index[-50].date())},index=dates)
    forecasts=pd.DataFrame(np.tile([2.,-1.,0.,.5],(len(dates),1)),index=dates,columns=['2Y','5Y','10Y','30Y'])
    a=generate_curve_signals(predictions,forecasts,x,prices,c)
    day=a.loc[a.status=='active','signal_date'].iloc[0]
    changed=x.copy();changed.loc[changed.index>day]+=10000
    other={s:v.copy() for s,v in prices.items()}
    for v in other.values():v.loc[v.index>day,'close']+=10000
    b=generate_curve_signals(predictions,forecasts,changed,other,c)
    pd.testing.assert_frame_equal(a[a.signal_date==day].reset_index(drop=True),b[b.signal_date==day].reset_index(drop=True))
    assert (a.loc[a.status=='active','estimated_five_day_pnl_std']<=50000).all()
    assert (a[a.status=='terminal_flat'].filter(regex='^target_')==0).all().all()


def test_pc_entry_is_causal_and_generates_three_separate_strategies():
    c,x,prices,beta=case()
    dates=pd.bdate_range(x.index[-30],x.index[-10])
    cutoff=str(x.index[-50].date())
    predictions=pd.DataFrame({'model_cutoff_date':cutoff,'pca_version':cutoff,
                              'ridge_PC1':1.,'ridge_PC2':-1.,'ridge_PC3':1.},index=dates)
    l=np.array([[.5,.5,.5,.5],[.5,.5,-.5,-.5],[.5,-.5,-.5,.5],[.5,-.5,.5,-.5]])
    loadings=pd.DataFrame([dict(pca_version=cutoff,component=f'PC{i+1}',maturity=m,loading=v)
                          for i,row in enumerate(l) for m,v in zip(x.columns,row)])
    a=generate_weekly_signals(predictions,loadings,x,prices,c)
    assert set(a.strategy)=={'PC1','PC2','PC3'}
    day=a.loc[a.status=='active','signal_date'].iloc[0]
    changed=x.copy();changed.loc[changed.index>day]+=10000
    other={s:v.copy() for s,v in prices.items()}
    for v in other.values():v.loc[v.index>day,'close']+=10000
    b=generate_weekly_signals(predictions,loadings,changed,other,c)
    pd.testing.assert_frame_equal(a[a.signal_date==day].reset_index(drop=True),b[b.signal_date==day].reset_index(drop=True))
    assert (a.loc[a.status=='active','estimated_five_day_pnl_std']<=50000).all()
