"""Forecasting scaffolding: chronological train/test splitting and
rolling-origin (walk-forward) backtesting.

This is deliberately built and tested *before* any actual forecasting
method (SMA, SES, Holt-Winters, Croston) is added. Every method will share
the same interface -- a callable that takes a history series and returns
one number, the next-period forecast -- so this scaffolding doesn't need
to know or care which method is plugged into it.

Why not a random train/test split: a random split lets a model train on
data from *after* the period it's asked to forecast. A real demand
planner forecasting March never has April's actual demand available, so
every split here is strictly chronological -- the final slice of a SKU's
history is test, everything before it is train.

Why rolling-origin instead of one static holdout: a static holdout fits a
method once and forecasts the whole test window in one shot. Rolling-
origin instead re-forecasts one period at a time, using an *expanding*
window of everything observed so far (train + every previously-revealed
test period), which mirrors how planning actually works: forecast next
month, observe the actual, forecast the following month.
"""
import pandas as pd

from src import config


def chronological_split(series: pd.Series, test_size_fraction: float = None):
    """Split a single SKU's chronologically-sorted demand series into
    (train, test), assigning the final `test_size_fraction` of periods to
    test. Defaults to config.TEST_SIZE_FRACTION. Always keeps at least 1
    period in the test set."""
    if test_size_fraction is None:
        test_size_fraction = config.TEST_SIZE_FRACTION
    n = len(series)
    n_test = max(1, round(n * test_size_fraction))
    return series.iloc[: n - n_test], series.iloc[n - n_test:]


def rolling_origin_forecast(series: pd.Series, n_test: int, forecast_fn) -> pd.Series:
    """Walk-forward backtest over the final `n_test` periods of `series`.

    At step i of the test window, `forecast_fn` sees only
    series.iloc[:n_train + i] -- every training period plus every test
    period already "revealed" by earlier steps -- and returns a single
    one-step-ahead forecast for series.iloc[n_train + i]. It never sees
    that period's actual value or anything after it.

    forecast_fn: callable(history: pd.Series) -> float

    Returns a pd.Series of forecasts aligned to the test period's index.
    """
    n = len(series)
    n_train = n - n_test
    forecasts = [forecast_fn(series.iloc[: n_train + i]) for i in range(n_test)]
    test_index = series.index[n_train:]
    return pd.Series(forecasts, index=test_index, name="Forecast")


def naive_last_value_forecast(history: pd.Series) -> float:
    """Forecast = the most recent observed value. The simplest possible
    method, useful both as a sanity-check baseline for the scaffolding
    above and, later, as the final fallback when SES/Holt-Winters can't
    be fit (e.g. too little history). Returns 0.0 for empty history."""
    if len(history) == 0:
        return 0.0
    return float(history.iloc[-1])
