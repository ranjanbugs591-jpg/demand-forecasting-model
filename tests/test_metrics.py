"""Unit tests for src/metrics.py -- WMAPE, Bias, Tracking Signal."""
import numpy as np
import pytest

from src.metrics import wmape, bias, tracking_signal


def test_wmape_perfect_forecast_is_zero():
    assert wmape([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0)


def test_wmape_all_zero_forecast():
    # error = 60 total, actual total = 60 -> 100% error
    assert wmape([10, 20, 30], [0, 0, 0]) == pytest.approx(100.0)


def test_wmape_undefined_when_actual_all_zero():
    assert np.isnan(wmape([0, 0, 0], [5, 0, 2]))


def test_bias_over_forecasting_is_positive():
    # forecast consistently 5 above actual of 10 -> +50%
    assert bias([10, 10], [15, 15]) == pytest.approx(50.0)


def test_bias_under_forecasting_is_negative():
    assert bias([10, 10], [5, 5]) == pytest.approx(-50.0)


def test_bias_undefined_when_actual_all_zero():
    assert np.isnan(bias([0, 0], [5, 5]))


def test_tracking_signal_persistent_over_forecast():
    # error = forecast - actual = +5 every period -> cumulative 15, MAD 5 -> TS 3
    assert tracking_signal([10, 10, 10], [15, 15, 15]) == pytest.approx(3.0)


def test_tracking_signal_errors_cancel_out():
    # errors [+5, -5] sum to 0 but MAD is still 5, so TS = 0 (not NaN)
    assert tracking_signal([10, 10], [15, 5]) == pytest.approx(0.0)


def test_tracking_signal_undefined_when_forecast_always_exact():
    assert np.isnan(tracking_signal([10, 20, 30], [10, 20, 30]))
