"""Unit tests for src/demand_classification.py.

These use small hand-built series with known, obvious properties (not the
real M5 data) specifically so each test's expected answer can be verified
by hand -- that's the point of a unit test.
"""
import numpy as np
import pandas as pd
import pytest

from src.demand_classification import compute_adi, compute_cv2, classify_demand


def test_adi_every_period_has_demand():
    demand = pd.Series([5, 6, 4, 5, 6])
    assert compute_adi(demand) == 1.0


def test_adi_half_periods_zero():
    demand = pd.Series([5, 0, 5, 0, 5, 0])
    assert compute_adi(demand) == 2.0


def test_adi_all_zero_is_infinite():
    demand = pd.Series([0, 0, 0, 0])
    assert compute_adi(demand) == np.inf


def test_cv2_constant_nonzero_demand_is_zero():
    demand = pd.Series([10, 10, 10, 10])
    assert compute_cv2(demand) == pytest.approx(0.0)


def test_cv2_variable_demand_is_positive():
    demand = pd.Series([1, 50, 2, 60, 1])
    assert compute_cv2(demand) > 0.49


def test_cv2_undefined_with_fewer_than_two_nonzero_points():
    demand = pd.Series([0, 0, 5, 0, 0])
    assert np.isnan(compute_cv2(demand))


def test_cv2_undefined_with_all_zero():
    demand = pd.Series([0, 0, 0])
    assert np.isnan(compute_cv2(demand))


@pytest.mark.parametrize("adi,cv2,expected", [
    (1.0, 0.10, "Smooth"),        # low ADI, low CV2
    (1.0, 0.80, "Erratic"),       # low ADI, high CV2
    (2.0, 0.10, "Intermittent"),  # high ADI, low CV2
    (2.0, 0.80, "Lumpy"),         # high ADI, high CV2
])
def test_classify_demand_quadrants(adi, cv2, expected):
    assert classify_demand(adi, cv2) == expected


def test_classify_demand_boundary_values_go_to_high_bucket():
    # Exactly at the threshold counts as the ">=" (high) side, per
    # src/config.py's ADI_THRESHOLD / CV2_THRESHOLD definitions.
    assert classify_demand(1.32, 0.10) == "Intermittent"
    assert classify_demand(1.0, 0.49) == "Erratic"


def test_classify_demand_undefined_when_cv2_is_nan():
    assert classify_demand(np.inf, np.nan) == "Undefined"
