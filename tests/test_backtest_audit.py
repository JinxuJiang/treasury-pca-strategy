"""Independent accounting check against actual Backtrader execution results."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.backtest.futures_engine import run_factor_backtest, prepare_feeds


def test_independent_accounting_with_open_reversal_and_costs():
    root=Path(__file__).resolve().parents[1]
    config=yaml.safe_load((root/'config/backtest_capacity.yaml').read_text())
    config['contracts']={'ZN':config['contracts']['ZN']}
    dates=pd.bdate_range('2024-01-04','2024-01-19',name='date')
    opens=100.+np.arange(len(dates))*.2
    prices={'ZN':pd.DataFrame(dict(open=opens,high=opens+1,low=opens-1,
                                  close=opens+.1,volume=1000),index=dates)}
    signals=pd.DataFrame(dict(signal_date=['2024-01-05','2024-01-11','2024-01-18'],
                              execution_date=['2024-01-08','2024-01-12','2024-01-19'],
                              target_ZN=[2,-1,0]))
    equity,orders,positions,_,metrics=run_factor_backtest(signals,prices,config)
    feeds,_=prepare_feeds(prices,config,equity.index.min(),equity.index.max())
    frame=feeds['ZN'].reindex(equity.index)
    held=positions.set_index(pd.to_datetime(positions.date))['ZN'].reindex(equity.index)
    overnight=held.shift(1).fillna(0)*(frame.open-frame.close.shift(1)).fillna(0)*1000
    intraday=held*(frame.close-frame.open)*1000
    costs=orders.groupby(pd.to_datetime(orders.execution_date)).total_cost.sum().reindex(equity.index,fill_value=0)
    expected=overnight+intraday-costs
    np.testing.assert_allclose(equity.equity.diff().fillna(0),expected,atol=1e-8)
    assert orders['size'].tolist()==[2,-3,1]
    assert metrics['open_positions_at_end']=={'ZN':0}
