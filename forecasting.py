"""Forecasting models, prediction intervals and backtesting."""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODELS = ["Seasonal naive", "Holt-Winters", "SARIMA"]
Z95 = 1.96


def forecast(y: pd.Series, h: int, model: str) -> pd.DataFrame:
    """Forecast h months ahead. Returns DataFrame with yhat, lower, upper."""
    y = y.asfreq("MS")
    idx = pd.date_range(y.index[-1], periods=h + 1, freq="MS")[1:]

    if model == "SARIMA":
        res = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred = res.get_forecast(h)
        ci = pred.conf_int(alpha=0.05)
        return pd.DataFrame({"yhat": pred.predicted_mean.values,
                             "lower": ci.iloc[:, 0].values,
                             "upper": ci.iloc[:, 1].values}, index=idx)

    if model == "Seasonal naive":
        yhat = np.resize(y.iloc[-12:].values, h)
        sigma = (y - y.shift(12)).dropna().std()
    else:  # Holt-Winters
        seasonal = "mul" if y.min() > 0 else "add"
        fit = ExponentialSmoothing(y, trend="add", seasonal=seasonal, seasonal_periods=12,
                                   initialization_method="estimated").fit()
        yhat = fit.forecast(h).values
        sigma = fit.resid.std()

    # Approximate interval: widens with the square root of the horizon
    width = Z95 * sigma * np.sqrt(np.arange(1, h + 1))
    return pd.DataFrame({"yhat": yhat, "lower": yhat - width, "upper": yhat + width}, index=idx)


def backtest(y: pd.Series, h: int = 6) -> pd.DataFrame:
    """Train on all but the last h months, score each model on the held-out months."""
    y = y.asfreq("MS")
    train, test = y.iloc[:-h], y.iloc[-h:]
    rows = []
    for m in MODELS:
        pred = forecast(train, h, m)["yhat"].values
        err = test.values - pred
        rows.append({"Model": m,
                     "MAE": np.abs(err).mean(),
                     "RMSE": np.sqrt((err ** 2).mean()),
                     "MAPE %": np.abs(err / test.values).mean() * 100})
    return pd.DataFrame(rows).sort_values("MAPE %").reset_index(drop=True)
