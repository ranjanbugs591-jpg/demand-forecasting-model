"""Forecast accuracy metrics: WMAPE, Bias, and Tracking Signal.

All three take aligned actual/forecast arrays for a single SKU's backtest
window and return a single number summarizing that window. Same sign
convention everywhere in this project (also used later in the Power BI
DAX measures): positive Bias = over-forecasting, negative = under-
forecasting.
"""
import numpy as np


def wmape(actual, forecast) -> float:
    """Weighted MAPE = sum(|Actual - Forecast|) / sum(|Actual|) * 100.

    Chosen over plain MAPE because MAPE divides by each *individual*
    actual, so a near-zero actual (routine at SKU level) can send the
    metric to thousands of percent and dominate any average. WMAPE
    weights every period by its share of total volume instead.

    Returns NaN (never silently 0) when sum(|Actual|) == 0 -- e.g. a test
    window with zero real demand throughout, where "percent error" simply
    isn't a defined concept.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual - forecast)) / denom * 100)


def bias(actual, forecast) -> float:
    """Bias = sum(Forecast - Actual) / sum(Actual) * 100.

    Positive = systematic over-forecasting (excess-inventory risk).
    Negative = systematic under-forecasting (stockout risk).

    Returns NaN when sum(Actual) == 0 (demand can't be negative, so this
    only happens when every actual in the window is zero).
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.sum(actual)
    if denom == 0:
        return np.nan
    return float(np.sum(forecast - actual) / denom * 100)


def tracking_signal(actual, forecast) -> float:
    """Tracking Signal = cumulative forecast error / MAD, where forecast
    error = Forecast - Actual (same sign convention as Bias).

    This is a control-chart-style diagnostic: it stays near zero for a
    well-behaved forecast and drifts toward +/- infinity when errors are
    persistently one-sided, even if each individual error is small. A
    widely used (not universally standardized) practical control limit is
    +/-4 -- see config.TRACKING_SIGNAL_CONTROL_LIMIT.

    Returns NaN when MAD == 0 (every forecast in the window was exactly
    correct -- 0/0 is undefined, not "perfectly in control").
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    errors = forecast - actual
    mad = np.mean(np.abs(errors))
    if mad == 0:
        return np.nan
    return float(np.sum(errors) / mad)
