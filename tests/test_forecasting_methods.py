"""Unit tests for the four forecasting methods in src/forecasting.py.

Expected values here were independently computed (by hand-tracing the
Croston recursion, and from known closed-form behavior of SMA/SES/Holt-
Winters on trivial series) and cross-checked against the actual functions
before being written into these assertions.
"""
import pandas as pd
import pytest

from src.forecasting import (
    sma_forecast,
    select_sma_window,
    ses_forecast,
    holt_winters_forecast,
    croston_forecast,
)


def _series(values):
    dates = pd.date_range("2020-01-01", periods=len(values), freq="MS")
    return pd.Series(values, index=dates, dtype=float)


# --- SMA ---------------------------------------------------------------

def test_sma_uses_last_n_periods():
    s = _series([10, 20, 30, 40])
    assert sma_forecast(s, window=2) == pytest.approx(35.0)  # mean(30, 40)


def test_sma_falls_back_to_full_history_if_window_too_large():
    s = _series([10, 20, 30, 40])
    assert sma_forecast(s, window=10) == pytest.approx(25.0)  # mean of all 4


def test_select_sma_window_prefers_fastest_adapting_window_after_level_shift():
    # 20 months at level 10, then 10 months at level 50: the smallest
    # candidate window (3) adapts to the new level fastest and should win
    # the nested in-training backtest.
    step_change = _series([10] * 20 + [50] * 10)
    assert select_sma_window(step_change) == 3


# --- SES -----------------------------------------------------------------

def test_ses_converges_on_constant_series():
    s = _series([25.0] * 15)
    assert ses_forecast(s) == pytest.approx(25.0)


def test_ses_single_point_falls_back_to_naive():
    assert ses_forecast(_series([7.0])) == pytest.approx(7.0)


# --- Holt-Winters ----------------------------------------------------------

def test_holt_winters_continues_linear_trend():
    s = _series(list(range(1, 21)))  # 1, 2, ..., 20
    assert holt_winters_forecast(s) == pytest.approx(21.0, abs=0.5)


def test_holt_winters_single_point_falls_back_to_naive():
    assert holt_winters_forecast(_series([9.0])) == pytest.approx(9.0)


# --- Croston ---------------------------------------------------------------

def test_croston_all_zero_history_forecasts_zero():
    assert croston_forecast(_series([0] * 10)) == pytest.approx(0.0)


def test_croston_single_nonzero_occurrence():
    # demand=5 at position 2 (0-indexed) -> z=5, p=3 -> forecast = 5/3
    s = _series([0, 0, 5, 0, 0])
    assert croston_forecast(s) == pytest.approx(5 / 3)


def test_croston_matches_hand_traced_recursion():
    # nonzero occurrences at positions 2, 5, 9 with values 5, 10, 3, alpha=0.1
    s = _series([0, 0, 5, 0, 0, 10, 0, 0, 0, 3])
    assert croston_forecast(s, alpha=0.1) == pytest.approx(1.6935483870967742)
