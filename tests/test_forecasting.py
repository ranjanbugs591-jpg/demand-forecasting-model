"""Unit tests for src/forecasting.py's split + rolling-origin scaffolding."""
import pandas as pd

from src.forecasting import chronological_split, rolling_origin_forecast, naive_last_value_forecast


def _series(values):
    dates = pd.date_range("2020-01-01", periods=len(values), freq="MS")
    return pd.Series(values, index=dates)


def test_chronological_split_sizes_and_order():
    s = _series(range(1, 11))  # 10 periods: 1..10
    train, test = chronological_split(s, test_size_fraction=0.2)
    assert len(train) == 8
    assert len(test) == 2
    # train must be strictly earlier than test -- no leakage
    assert train.index.max() < test.index.min()
    assert list(train.values) == list(range(1, 9))
    assert list(test.values) == [9, 10]


def test_chronological_split_always_keeps_at_least_one_test_period():
    s = _series(range(1, 4))  # only 3 periods, 20% would round to ~1 anyway
    train, test = chronological_split(s, test_size_fraction=0.01)
    assert len(test) >= 1


def test_rolling_origin_forecast_expanding_window_with_naive_method():
    s = _series(range(1, 11))  # values 1..10
    forecasts = rolling_origin_forecast(s, n_test=3, forecast_fn=naive_last_value_forecast)

    # naive forecast at each step = the last value the model has seen so far
    assert list(forecasts.values) == [7.0, 8.0, 9.0]
    # forecasts must be aligned to the actual test-period dates
    assert list(forecasts.index) == list(s.index[7:])


def test_rolling_origin_forecast_uses_correct_expanding_window_length():
    # Recording exactly how much history forecast_fn sees at each step
    # directly proves the window expands by one period at a time and
    # never includes the period currently being forecast.
    s = _series(range(1, 11))
    n_test = 3
    n_train = len(s) - n_test
    seen_lengths = []

    def recording_forecast(history):
        seen_lengths.append(len(history))
        return naive_last_value_forecast(history)

    rolling_origin_forecast(s, n_test=n_test, forecast_fn=recording_forecast)
    assert seen_lengths == [n_train, n_train + 1, n_train + 2]


def test_naive_forecast_empty_history_returns_zero():
    assert naive_last_value_forecast(pd.Series([], dtype=float)) == 0.0
