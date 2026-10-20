import copy

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from src.portfolio.factor_positions import weekly_schedule
from src.backtest.futures_engine import run_factor_backtest
from src.factors.pca_factors import fit_pca

ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT/'config/backtest_capacity.yaml').read_text())


def test_real_calendar_holiday_week_and_truncated_week():
    s = weekly_schedule('CME_InterestRate', '2025-04-14', '2025-04-18')
    assert s.signal_date.tolist() == [pd.Timestamp('2025-04-17')]
    assert s.execution_date.tolist() == [pd.Timestamp('2025-04-21')]
    truncated = weekly_schedule('CME_InterestRate', '2025-04-14', '2025-04-16')
    assert truncated.empty


def small_case(missing_monday=False):
    c = config();c['contracts'] = {'ZN': c['contracts']['ZN']}
    c['execution']['slippage_ticks_per_side'] = 0
    dates = pd.bdate_range('2024-01-04','2024-01-16',name='date')
    opens = 100.+np.arange(len(dates))
    price = pd.DataFrame(dict(open=opens,high=opens+1,low=opens-1,close=opens+.5,volume=100),index=dates)
    if missing_monday:price.loc['2024-01-08'] = np.nan
    signals = pd.DataFrame(dict(signal_date=['2024-01-05','2024-01-12'],
                                 execution_date=['2024-01-08','2024-01-15'],target_ZN=[1,0]))
    return c, price, signals


def test_backtrader_next_open_multiplier_cost_and_final_liquidation():
    c,price,signals = small_case()
    equity,orders,positions,audit,metrics = run_factor_backtest(signals,{'ZN':price},c)
    assert orders.execution_date.astype(str).tolist() == ['2024-01-08','2024-01-15']
    expected_gross = (price.loc['2024-01-15','open']-price.loc['2024-01-08','open'])*1000
    assert metrics['net_pnl'] == pytest.approx(expected_gross-5.)
    assert metrics['total_cost'] == pytest.approx(5.)
    assert metrics['open_positions_at_end'] == {'ZN':0}
    same_day = signals.copy();same_day['execution_date'] = same_day.signal_date
    with pytest.raises(ValueError,match='Same-day'):
        run_factor_backtest(same_day,{'ZN':price},c)


def test_missing_bar_never_fills_on_carried_mark():
    c,price,signals = small_case(True)
    eq,orders,positions,audit,metrics = run_factor_backtest(signals,{'ZN':price},c)
    assert str(orders.execution_date.iloc[0]) == '2024-01-09'
    assert orders.price.iloc[0] == price.loc['2024-01-09','open']
    assert metrics['delayed_fills'] == 1
    assert 'missing_bar_no_execution_carry_mark' in audit.issue.tolist()
    assert np.isfinite(eq.equity).all()
