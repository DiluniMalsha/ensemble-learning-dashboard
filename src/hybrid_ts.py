"""
Bonus: hybrid ARIMA + Random Forest for time series (a simple form of stacking).

Idea (Zhang, 2003 style hybrid):
    y_t = L_t (linear part, captured by SARIMA)  +  N_t (non-linear part)
1. Fit SARIMA on the training period -> one-step-ahead linear forecast  ŷ_arima
2. Residuals  r_t = y_t - ŷ_arima  contain what ARIMA could NOT model
3. Fit a Random Forest on lagged residuals + month + ŷ_arima  -> predicts r̂_t
4. Hybrid forecast = ŷ_arima + r̂_t

All forecasts are one-step-ahead: at month t we only use information up to t-1,
and SARIMA parameters are estimated on the training period only (no leakage).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _lags(s: pd.Series, lags=(1, 2, 3, 12), prefix="lag") -> pd.DataFrame:
    return pd.concat({f"{prefix}{k}": s.shift(k) for k in lags}, axis=1)


def _metrics(y, p):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": float(mean_absolute_error(y, p)),
            "MAPE %": float(np.mean(np.abs((y - p) / np.where(y == 0, 1, y))) * 100)}


def run_hybrid(series: pd.Series, test_size: int = 36, order=(1, 0, 1), seasonal_order=(1, 1, 1, 12),
               n_estimators: int = 300, random_state: int = 42):
    series = series.astype(float).asfreq("MS") if isinstance(series.index, pd.DatetimeIndex) else series.astype(float)
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]

    # 1) SARIMA on train, then apply the SAME parameters to the full series for one-step forecasts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        full = fit.apply(series)
    arima_pred = full.get_prediction().predicted_mean
    resid = series - arima_pred

    month = pd.Series(series.index.month if hasattr(series.index, "month") else np.arange(len(series)) % 12 + 1,
                      index=series.index, name="month")

    # 2) Random Forest alone (lags of the raw series)
    X_rf = pd.concat([_lags(series), month], axis=1)
    # 3) Hybrid: RF on residual lags + month + ARIMA forecast
    X_h = pd.concat([_lags(resid, prefix="res_lag"), month, arima_pred.rename("arima_pred")], axis=1)

    start = 13 + 12  # skip the SARIMA burn-in (seasonal differencing) and lag NaNs
    tr_idx = series.index[start:-test_size]
    te_idx = test.index

    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3, random_state=random_state, n_jobs=-1)
    rf.fit(X_rf.loc[tr_idx], series.loc[tr_idx])
    rf_pred = pd.Series(rf.predict(X_rf.loc[te_idx]), index=te_idx)

    rf_res = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=5, max_depth=6,
                                   random_state=random_state, n_jobs=-1)
    rf_res.fit(X_h.loc[tr_idx], resid.loc[tr_idx])
    hybrid_pred = arima_pred.loc[te_idx] + rf_res.predict(X_h.loc[te_idx])

    preds = pd.DataFrame({"Actual": test, "SARIMA": arima_pred.loc[te_idx],
                          "Random Forest": rf_pred, "Hybrid SARIMA+RF": hybrid_pred})
    scores = pd.DataFrame({m: _metrics(test.values, preds[m].values)
                           for m in ["SARIMA", "Random Forest", "Hybrid SARIMA+RF"]}).T
    return preds, scores, series
