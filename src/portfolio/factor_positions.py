"""PC portfolios plus shared regression, calendar and basket risk calculations."""
from itertools import product
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


def weekly_schedule(calendar_name: str, start, end) -> pd.DataFrame:
    """Use full calendar weeks, not the truncated set of prediction dates."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    # use the start and end to get real trading days from mcal.get_calendar
    schedule = mcal.get_calendar(calendar_name).schedule(
        start_date=start - pd.Timedelta(days=7), end_date=end + pd.Timedelta(days=21))
    dates = schedule.index
    week_ends = pd.Series(dates, index=dates).groupby(dates.to_period('W-FRI')).max()
    records = []
    # loop every trading date, which is the week_ends.
    for signal in week_ends:
        if start <= signal <= end:
            pos = dates.get_loc(signal)
            # The next trading session after the signal is the execution date.
            execution = dates[pos + 1]
            # The next futures session may open the evening before its trade date.
            if schedule.loc[execution, 'market_open'] <= schedule.loc[signal, 'market_close']:
                raise ValueError('Execution must follow the signal session close')
            records.append(dict(signal_date=signal, execution_date=execution,
                                signal_session_close_utc=schedule.loc[signal, 'market_close'],
                                execution_open_utc=schedule.loc[execution, 'market_open']))
    return pd.DataFrame(records)


def fit_sensitivities(changes, prices, config, day):
    """Rows of beta are contracts; columns are yield nodes, USD / bp / contract.

    No filling across missing sessions and no calendar-based roll exclusions.
    Robust residual trimming filters statistical outliers; it cannot establish
    that a continuous series is free of roll effects.
    """
    settings = config['sensitivity']
    symbols = list(config['contracts'])
    columns = [f"d_{spec['maturity']}_bp" for spec in config['contracts'].values()]
    x = changes.loc[:day, columns].copy()
    pnl = pd.DataFrame(index=x.index)
    for symbol in symbols:
        source = prices[symbol].loc[:day].reindex(x.index)
        # Differencing after alignment prevents bridging a missing yield/price day.
        pnl[symbol] = source.close.diff() * config['contracts'][symbol]['multiplier']
    data = x.join(pnl).replace([np.inf, -np.inf], np.nan).tail(settings['window'])
    data = data.dropna()
    if len(data) < settings['min_observations']:
        raise ValueError('insufficient_regression_history')
    X, Y = data[columns].to_numpy(), data[symbols].to_numpy()
    keep = np.ones(len(X), dtype=bool)
    for _ in range(3):
        design = np.column_stack([np.ones(keep.sum()), X[keep]])
        # The lstsq solution is equivalent to the OLS solution.
        coef = np.linalg.lstsq(design, Y[keep], rcond=None)[0]
        residual = Y - np.column_stack([np.ones(len(X)), X]) @ coef
        center = np.median(residual[keep], axis=0)
        mad = 1.4826*np.median(np.abs(residual[keep]-center), axis=0)
        keep &= (np.abs(residual-center) <= settings['residual_mad_limit']*np.maximum(mad, 1e-8)).all(axis=1)
    if keep.sum() < settings['min_observations']:
        raise ValueError('insufficient_clean_regression_history')
    X, Y = X[keep], Y[keep]
    condition = np.linalg.cond(X-X.mean(axis=0))
    if not np.isfinite(condition) or condition > settings['max_condition_number']:
        raise ValueError('unstable_yield_regression')
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), Y, rcond=None)[0]
    residual = Y-np.column_stack([np.ones(len(X)), X])@coef
    r2 = 1-(residual**2).sum(axis=0)/np.maximum(((Y-Y.mean(axis=0))**2).sum(axis=0), 1e-12)
    beta = coef[1:].T
    if r2.min() < settings['min_r_squared'] or np.linalg.cond(beta) > settings['max_condition_number']:
        raise ValueError('unreliable_futures_regression')
    # Actual consecutive five-observation dollar P&L windows. Excluded days stay
    # missing, so rolling does not join unrelated days across removed records.
    clean = pnl.copy()
    clean.loc[~clean.index.isin(data.index[keep])] = np.nan
    five = clean.rolling(5).sum().dropna().tail(config['risk']['volatility_window'])
    if len(five) < settings['min_five_day_observations']:
        raise ValueError('insufficient_clean_pnl_history')
    audit = dict(regression_observations=int(len(X)), regression_end_date=str(data.index[keep].max().date()),
                 regression_min_r_squared=float(r2.min()), regression_condition=float(condition),
                 pnl_observations=len(five))
    for i,s in enumerate(symbols):
        audit[f'r_squared_{s}'] = float(r2[i])
        for j,m in enumerate(columns): audit[f'beta_{s}_{m}'] = float(beta[i,j])
    return beta, five.cov().to_numpy(), audit


def choose_basket(target_exposure, exposure_matrix, pnl_cov, config, component, node_matrix=None):
    '''
    according to the beta from fit_sensitivities,
    choose a basket of contracts that is closest to the target exposure,
    while satisfying the risk constraints in config.
    '''
    risk, cash = config['risk'], config['initial_cash']
    margins = np.array([s['assumed_margin'] for s in config['contracts'].values()])
    # get a set of weights of 4 futures that fits the target exposure to only trade in a specific PC
    continuous = np.linalg.solve(exposure_matrix, target_exposure)
    # calculate the standard deviation of the P&L of the specific continuous weights
    std = np.sqrt(max(0., continuous@pnl_cov@continuous))
    if std <= 1e-12: return np.zeros(4,dtype=int), 'zero_signal'
    # scale the continuous weights to satisfy the risk constraints
    continuous *= cash*risk['five_day_std_budget_fraction']/std
    gross_per_contract = np.abs(exposure_matrix if node_matrix is None else node_matrix).sum(axis=0)
    continuous *= min(1., risk['max_contracts_per_leg']/max(np.abs(continuous).max(),1e-12),
                      cash*risk['max_margin_fraction']/max(np.abs(continuous)@margins,1e-12),
                      risk['gross_dv01_limit']/max(np.abs(continuous)@gross_per_contract,1e-12))
    desired = exposure_matrix@continuous
    best, error = None, np.inf
    for offsets in product((-1,0,1),repeat=4):
        n = np.rint(continuous).astype(int)+offsets
        e = exposure_matrix@n
        if not n.any() or np.abs(n).max()>risk['max_contracts_per_leg']: continue
        if np.abs(n)@margins>cash*risk['max_margin_fraction']: continue
        if np.abs(n)@gross_per_contract>risk['gross_dv01_limit']: continue
        if n@pnl_cov@n>(cash*risk['five_day_std_budget_fraction'])**2*(1+1e-10): continue
        if component is not None:
            if e[component]*desired[component]<=0: continue
            if np.linalg.norm(np.delete(e,component))/abs(e[component])>risk['max_other_pc_exposure_ratio']: continue
        distance = np.linalg.norm(e-desired)
        if distance<error: best,error=n,distance
    return (best,'active') if best is not None else (np.zeros(4,dtype=int),'no_integer_basket_within_limits')


def _generate_positions(predictions, changes, prices, config, strategies, build_targets):
    """Shared weekly execution schedule, regression, risk checks and audit output.

    build_targets supplies each strategy's loading basis and risk shape. Only the
    curve module defines the curve allocation rule; this engine is strategy-neutral.
    """
    symbols = list(config['contracts'])
    columns = [f"d_{spec['maturity']}_bp" for spec in config['contracts'].values()]
    changes = changes[columns]
    schedule = weekly_schedule(config['calendar'], predictions.index.min(), predictions.index.max())
    terminal = weekly_schedule(config['calendar'], predictions.index.max()+pd.Timedelta(days=1),
                               predictions.index.max()+pd.Timedelta(days=8)).iloc[:1]
    terminal_dates = set(terminal.signal_date)
    records = []
    for event in pd.concat([schedule, terminal], ignore_index=True).to_dict('records'):
        day = event['signal_date']
        fit = None
        status = 'terminal_flat' if day in terminal_dates else 'missing_prediction'
        if status != 'terminal_flat' and day in predictions.index:
            row = predictions.loc[day]
            if pd.Timestamp(row.model_cutoff_date) > day:
                raise ValueError('Future model cutoff')
            try:
                fit = fit_sensitivities(changes, prices, config, day)
            except ValueError as exc:
                if str(exc).startswith(('insufficient_', 'unstable_', 'unreliable_')):
                    status = str(exc)
                else:
                    raise
        targets = build_targets(day, row) if fit is not None else {}
        for strategy, component in strategies:
            result = dict(event, strategy=strategy, status=status)
            n = np.zeros(len(symbols), dtype=int)
            if fit is not None and strategy in targets:
                beta, cov, audit = fit
                loading, target = targets[strategy]
                matrix = -loading @ beta.T
                if not np.isfinite(target).all():
                    result['status'] = 'invalid_forecast'
                else:
                    n, result['status'] = choose_basket(target, matrix, cov, config, component, node_matrix=-beta.T)
                exposure = -beta.T @ n
                result.update(audit, model_cutoff_date=row.model_cutoff_date,
                              estimated_five_day_pnl_std=float(np.sqrt(max(0., n@cov@n))),
                              gross_dv01=float(np.abs(n)@np.abs(beta).sum(axis=1)),
                              net_dv01=float(exposure.sum()),
                              assumed_margin=float(np.abs(n)@np.array([s['assumed_margin'] for s in config['contracts'].values()])))
                keys = ([f'exposure_PC{i+1}' for i in range(4)] if component is not None else
                        [f"exposure_{s['maturity']}" for s in config['contracts'].values()])
                result.update(dict(zip(keys, map(float, loading@exposure))))
            result.update({f'target_{s}': int(v) for s, v in zip(symbols, n)})
            records.append(result)
    return pd.DataFrame(records)


def generate_weekly_signals(predictions, loading_table, changes, prices, config):
    """PC1/2/3: isolate the forecast PC direction using empirical sensitivities."""
    columns = [f"d_{s['maturity']}_bp" for s in config['contracts'].values()]

    def targets(day, row):
        cutoff = pd.Timestamp(row.pca_version)
        if cutoff > day:
            raise ValueError('Future PCA cutoff')
        table = loading_table[pd.to_datetime(loading_table.pca_version) == cutoff]
        loading = table.pivot(index='component', columns='maturity', values='loading').loc[
            ['PC1', 'PC2', 'PC3', 'PC4'], columns].to_numpy()
        if not np.allclose(loading@loading.T, np.eye(4), atol=1e-6):
            raise ValueError('Nonorthogonal PCA')
        result = {}
        for k in range(3):
            target = np.zeros(4)
            target[k] = -np.sign(float(row[f"{config['model']}_PC{k+1}"]))
            result[f'PC{k+1}'] = loading, target
        return result

    return _generate_positions(predictions, changes, prices, config,
                               [(f'PC{k+1}', k) for k in range(3)], targets)
