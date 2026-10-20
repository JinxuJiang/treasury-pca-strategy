"""Unified curve portfolio: reconstructed forecasts to futures positions."""
import numpy as np
from src.portfolio.factor_positions import _generate_positions


def generate_curve_signals(predictions, yield_forecasts, changes, prices, config):
    """Curve allocation rule; reuse the same regression and risk engine as PCs."""
    maturities = [s['maturity'] for s in config['contracts'].values()]
    columns = [f'd_{m}_bp' for m in maturities]

    def targets(day, row):
        if day not in yield_forecasts.index:
            return {}
        forecast = yield_forecasts.loc[day, maturities].to_numpy(dtype=float)
        volatility = changes.loc[:day, columns].rolling(5).sum().dropna().tail(
            config['risk']['volatility_window']).std().to_numpy()
        return {'CURVE': (np.eye(4), -forecast/volatility)}

    return _generate_positions(predictions, changes, prices, config, [('CURVE', None)], targets)
