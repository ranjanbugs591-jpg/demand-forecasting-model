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
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

from src import config
from src.metrics import wmape


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


# ---------------------------------------------------------------------------
# The four forecasting methods. Every one of these matches the same
# forecast_fn(history: pd.Series) -> float interface rolling_origin_forecast
# expects, so any of them can be dropped straight into a backtest.
# ---------------------------------------------------------------------------


def sma_forecast(history: pd.Series, window: int) -> float:
    """Simple Moving Average: mean of the last `window` periods.

    Transparent, no fitting required, robust to a single noisy month --
    but it lags a real level shift or trend by roughly window/2 periods
    and can't represent seasonality at all.

    Falls back to the mean of all available history if there's less than
    `window` periods to work with (rather than crashing on short SKUs).
    """
    if len(history) == 0:
        return 0.0
    window = min(window, len(history))
    return float(history.iloc[-window:].mean())


def select_sma_window(train_series: pd.Series, candidate_windows=None) -> int:
    """Pick the best SMA window (from config.SMA_WINDOWS) for one SKU.

    Runs its own nested rolling-origin backtest *entirely within
    `train_series`* -- carving an inner test slice out of training data
    only -- and keeps whichever window scores the lowest WMAPE on that
    inner backtest. This can never leak real test-window information,
    because it never looks past the training data it's given.
    """
    if candidate_windows is None:
        candidate_windows = config.SMA_WINDOWS

    if len(train_series) < 4:
        # Too little training history to run a meaningful nested backtest;
        # default to the smallest window, which adapts fastest.
        return min(candidate_windows)

    inner_test_size = max(1, round(len(train_series) * config.TEST_SIZE_FRACTION))
    inner_test_size = min(inner_test_size, len(train_series) - 1)
    inner_actuals = train_series.iloc[-inner_test_size:].to_numpy()

    best_window, best_score = candidate_windows[0], np.inf
    for window in candidate_windows:
        inner_forecasts = rolling_origin_forecast(
            train_series, inner_test_size, lambda h, w=window: sma_forecast(h, w)
        )
        score = wmape(inner_actuals, inner_forecasts.to_numpy())
        if not np.isnan(score) and score < best_score:
            best_score, best_window = score, window
    return best_window


def ses_forecast(history: pd.Series) -> float:
    """Simple Exponential Smoothing: an exponentially-weighted average of
    all past observations, controlled by one smoothing parameter (alpha),
    optimized by maximum likelihood every time this is called. Adapts to
    a level shift faster than SMA, but has no trend or seasonal term.

    Falls back to a naive last-value forecast for empty, single-point, or
    perfectly constant history -- SES has nothing meaningful to fit in
    those cases and would either error or return a degenerate result.
    """
    values = history.dropna()
    if len(values) < 2 or values.nunique() == 1:
        return naive_last_value_forecast(values)
    try:
        model = SimpleExpSmoothing(
            values.to_numpy(), initialization_method="estimated"
        ).fit(optimized=True)
        return float(model.forecast(1)[0])
    except Exception:
        return naive_last_value_forecast(values)


def holt_winters_forecast(history: pd.Series, seasonal_period: int = None) -> float:
    """Holt-Winters / triple exponential smoothing: extends SES with an
    additive trend, and -- only when there's enough history to justify it
    -- an additive seasonal component.

    Seasonality is only attempted with at least
    config.MIN_HISTORY_FOR_SEASONALITY months of history (two full annual
    cycles); fitting a 12-month seasonal term on ~14 months of data would
    overfit a handful of degrees of freedom, so shorter SKUs deliberately
    fall back to Holt's trend-only method instead. If even that fit
    fails, falls back to SES, then to a naive forecast -- every fallback
    is a real, documented path, not a swallowed error.
    """
    if seasonal_period is None:
        seasonal_period = config.SEASONAL_PERIOD

    values = history.dropna()
    n = len(values)
    if n < 2 or values.nunique() == 1:
        return naive_last_value_forecast(values)

    use_seasonal = n >= config.MIN_HISTORY_FOR_SEASONALITY and n >= 2 * seasonal_period
    try:
        if use_seasonal:
            model = ExponentialSmoothing(
                values.to_numpy(),
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_period,
                initialization_method="estimated",
            ).fit(optimized=True)
        else:
            model = ExponentialSmoothing(
                values.to_numpy(), trend="add", seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
        return float(model.forecast(1)[0])
    except Exception:
        return ses_forecast(values)


def croston_forecast(history: pd.Series, alpha: float = None) -> float:
    """Classic Croston's method, implemented from scratch (no library
    covers it). SMA/SES/Holt-Winters smooth every period equally,
    including the zero-demand periods that dominate intermittent series,
    which drags their forecasts toward zero. Croston instead tracks two
    independent exponentially-smoothed series -- the *size* of demand
    when it occurs, and the *interval* between occurrences -- and
    forecasts their ratio (a demand rate).

    Known limitation, stated plainly: classic Croston carries a small
    positive bias (it doesn't fully correct for the geometric
    distribution of inter-demand intervals). Variants like SBA and TSB
    exist specifically to fix this and are intentionally out of scope
    here -- see docs/methodology.md.
    """
    if alpha is None:
        alpha = config.CROSTON_ALPHA

    values = history.to_numpy(dtype=float)
    nonzero_positions = np.flatnonzero(values > 0)

    if len(nonzero_positions) == 0:
        return 0.0  # no demand has ever occurred -- nothing to extrapolate from

    # Initialize from the first observed demand occurrence.
    z = values[nonzero_positions[0]]          # smoothed demand size
    p = float(nonzero_positions[0] + 1)       # smoothed inter-demand interval

    prev_position = nonzero_positions[0]
    for position in nonzero_positions[1:]:
        interval = position - prev_position
        z = alpha * values[position] + (1 - alpha) * z
        p = alpha * interval + (1 - alpha) * p
        prev_position = position

    return float(z / p) if p > 0 else float(z)
