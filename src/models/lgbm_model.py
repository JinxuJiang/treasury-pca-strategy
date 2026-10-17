"""Small validation-only LightGBM grid for PCA forecasting experiments."""
import numpy as np
from lightgbm import LGBMRegressor

CANDIDATES = [
    dict(num_leaves=leaves, min_child_samples=minimum, n_estimators=trees)
    for leaves, minimum in [(3, 20), (7, 20), (7, 40)]
    for trees in (50, 150)
]


def fit_lgbm(frame, features, target, training_mask, params):
    model = LGBMRegressor(objective='regression', learning_rate=0.03,
                          reg_lambda=10.0, random_state=42, n_jobs=1,
                          verbosity=-1, deterministic=True, force_col_wise=True,
                          **params)
    model.fit(frame.loc[training_mask, features], frame.loc[training_mask, target])
    return model


def choose_lgbm_params(frame, features, target, training_mask, validation_mask):
    best, error = None, float('inf')
    for params in CANDIDATES:
        model = fit_lgbm(frame, features, target, training_mask, params)
        prediction = model.predict(frame.loc[validation_mask, features])
        rmse = float(np.sqrt(np.mean((frame.loc[validation_mask, target].to_numpy()-prediction)**2)))
        if rmse < error:
            best, error = params.copy(), rmse
    return best, error
