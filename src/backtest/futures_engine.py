"""Backtrader futures accounting for precomputed weekly target contract counts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import backtrader as bt
import pandas_market_calendars as mcal


class FuturesFeed(bt.feeds.PandasData):
    lines = ('tradable',)
    params = (('tradable', 'tradable'),)


class FuturesCosts(bt.CommInfoBase):
    params = (('stocklike', False), ('commtype', bt.CommInfoBase.COMM_FIXED),
              ('slippage_dollars', 0.))

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * (self.p.commission + self.p.slippage_dollars)


def prepare_feeds(prices, config, start, end):
    """Calendar gaps are flagged non-tradable and marked at the last known close."""
    sessions = mcal.get_calendar(config['calendar']).schedule(start, end).index
    prepared, audit = {}, []
    for symbol, source in prices.items():
        source = source.copy().sort_index()
        source.index = pd.DatetimeIndex(source.index)
        if source.index.has_duplicates:
            raise ValueError(f'Duplicate price dates for {symbol}')
        frame = source.reindex(sessions)[['open', 'high', 'low', 'close', 'volume']]
        valid = frame[['open', 'high', 'low', 'close']].notna().all(axis=1)
        values = frame.loc[valid, ['open', 'high', 'low', 'close']].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite prices for {symbol}')
        if len(values) and ((values[:, 1] < np.maximum(values[:, 0], values[:, 3])).any()
                            or (values[:, 2] > np.minimum(values[:, 0], values[:, 3])).any()):
            # Yahoo historical bars sometimes have settlement closes outside the range.
            # Keep prices, flag for audit; do not invent a new high/low or silently drop bars.
            audit.append(dict(symbol=symbol, issue='OHLC_range_inconsistency',
                              dates='see raw source; provider close may be settlement'))
        last_close = source.loc[source.index < pd.Timestamp(start), 'close'].dropna()
        marks = frame.close.where(valid).ffill()
        if len(last_close):
            marks = marks.fillna(last_close.iloc[-1])
        if marks.isna().any():
            raise ValueError(f'No initial valuation price for {symbol}')
        for field in ['open', 'high', 'low', 'close']:
            frame.loc[~valid, field] = marks.loc[~valid]
        frame['volume'] = frame.volume.fillna(0)
        frame['tradable'] = valid.astype(int)
        frame['openinterest'] = 0
        for date in frame.index[~valid]:
            audit.append(dict(symbol=symbol, issue='missing_bar_no_execution_carry_mark',
                              dates=str(date.date())))
        prepared[symbol] = frame
    return prepared, pd.DataFrame(audit, columns=['symbol', 'issue', 'dates'])


class WeeklyFactorStrategy(bt.Strategy):
    params = (('signals', None), ('specs', None))

    def __init__(self):
        self.events = {pd.Timestamp(row.signal_date).date(): row
                       for row in self.p.signals.itertuples(index=False)}
        self.equity_records, self.order_records, self.position_records = [], [], []
        self.pending = []

    def next(self):
        date = self.datetime.date(0)
        self.equity_records.append(dict(date=date, equity=self.broker.getvalue(), cash=self.broker.getcash()))
        self.position_records.append(dict(date=date, **{
            d._name: self.getposition(d).size for d in self.datas}))
        event = self.events.get(date)
        if event is None:
            return
        # A newer weekly target supersedes any unfilled orders from missing sessions.
        for order in self.pending:
            if order.alive():
                self.cancel(order)
        self.pending = []
        for data in self.datas:
            target = int(getattr(event, f'target_{data._name}'))
            order = self.order_target_size(data=data, target=target)
            if order is not None:
                order.addinfo(signal_date=str(date), planned_execution_date=str(event.execution_date.date()))
                self.pending.append(order)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Margin, order.Rejected]:
            raise RuntimeError(f'{order.data._name}: {order.getstatusname()} order; revisit risk/margin assumptions')
        if order.status == order.Completed:
            spec = self.p.specs[order.data._name]
            size = abs(order.executed.size)
            self.order_records.append(dict(
                signal_date=order.info.signal_date,
                planned_execution_date=order.info.planned_execution_date,
                execution_date=bt.num2date(order.executed.dt).date(),
                symbol=order.data._name, size=order.executed.size,
                price=order.executed.price, total_cost=order.executed.comm,
                slippage_cost=size * spec['slippage_dollars'],
                commission_cost=size * spec['commission'],
                delayed= str(bt.num2date(order.executed.dt).date()) != order.info.planned_execution_date,
            ))


def run_factor_backtest(signals, prices, config):
    """One account per PC; no optimization or signal computation inside Backtrader."""
    signals = signals.copy()
    for field in ['signal_date', 'execution_date']:
        signals[field] = pd.to_datetime(signals[field])
    if signals.signal_date.duplicated().any():
        raise ValueError('One target per strategy per signal date is required')
    if not (signals.execution_date > signals.signal_date).all():
        raise ValueError('Same-day close execution is not allowed')
    start = signals.signal_date.min()
    end = min(signals.execution_date.max(), min(pd.Timestamp(df.index.max()) for df in prices.values()))
    if end <= start:
        raise ValueError('Not enough prices after first signal')
    feeds, audit = prepare_feeds(prices, config, start, end)
    cerebro = bt.Cerebro(stdstats=False)
    specs = {}
    for symbol, spec in config['contracts'].items():
        cerebro.adddata(FuturesFeed(dataname=feeds[symbol]), name=symbol)
        execution = config['execution']
        slip = execution['slippage_ticks_per_side'] * spec['tick_size'] * spec['multiplier']
        comm = execution['commission_per_contract_side']
        cerebro.broker.addcommissioninfo(FuturesCosts(
            commission=comm, slippage_dollars=slip, margin=spec['assumed_margin'],
            mult=spec['multiplier']), name=symbol)
        specs[symbol] = dict(commission=comm, slippage_dollars=slip)
    # No fill on a missing-price session, for any leg of the basket.
    def filler(order, price, ago):
        if not all(data.tradable[0] for data in order.owner.datas):
            return 0
        return abs(order.executed.remsize)
    cerebro.broker.set_filler(filler)
    cerebro.broker.setcash(config['initial_cash'])
    cerebro.broker.set_coc(False)
    cerebro.addstrategy(WeeklyFactorStrategy, signals=signals, specs=specs)
    strategy = cerebro.run()[0]
    equity = pd.DataFrame(strategy.equity_records).set_index('date')
    equity.index = pd.to_datetime(equity.index)
    equity['return'] = equity.equity.pct_change().fillna(0)
    equity['nav'] = equity.equity / config['initial_cash']
    equity['drawdown'] = equity.nav / equity.nav.cummax() - 1
    orders = pd.DataFrame(strategy.order_records, columns=[
        'signal_date', 'planned_execution_date', 'execution_date', 'symbol', 'size',
        'price', 'total_cost', 'slippage_cost', 'commission_cost', 'delayed'])
    pnl = equity.equity.iloc[-1] - config['initial_cash']
    elapsed = max((equity.index[-1] - equity.index[0]).days / 365.25, 1/365.25)
    vol = equity['return'].std(ddof=1) * np.sqrt(252)
    metrics = dict(total_return=float(equity.nav.iloc[-1] - 1),
                   annualized_return=float(equity.nav.iloc[-1]**(1/elapsed) - 1) if equity.nav.iloc[-1] > 0 else None,
                   annualized_volatility=float(vol),
                   sharpe_zero_rf=float(equity['return'].mean()*252/vol) if vol > 0 else None,
                   max_drawdown=float(equity.drawdown.min()), final_equity=float(equity.equity.iloc[-1]),
                   net_pnl=float(pnl), total_cost=float(orders.total_cost.sum()),
                   gross_pnl_before_charged_costs=float(pnl + orders.total_cost.sum()),
                   fills=len(orders), contract_sides=int(orders['size'].abs().sum()),
                   delayed_fills=int(orders.delayed.sum()),
                   active_signal_count=int((signals.status == 'active').sum()) if 'status' in signals else None,
                   skipped_integer_baskets=int((signals.status == 'no_integer_basket_within_limits').sum()) if 'status' in signals else None,
                   open_positions_at_end={d._name: strategy.getposition(d).size for d in strategy.datas})
    return equity, orders, pd.DataFrame(strategy.position_records), audit, metrics
