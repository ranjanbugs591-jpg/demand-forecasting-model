"""Unit tests for src/model_selection.py.

Uses small hand-built metrics tables (not real backtests) so each
scenario's expected decision is unambiguous and easy to verify by eye.
"""
import numpy as np

from src.model_selection import select_method_for_sku


def _m(wmape, bias=0.0, ts=0.0):
    return {"WMAPE": wmape, "Bias": bias, "Tracking_Signal": ts}


def test_picks_lowest_wmape_among_in_class_candidates():
    metrics = {
        "SMA": _m(40.0, ts=1.0),
        "SES": _m(30.0, ts=1.0),          # lowest WMAPE, in-control TS
        "Holt-Winters": _m(35.0, ts=1.0),
        "Croston": _m(90.0, ts=1.0),
    }
    result = select_method_for_sku(metrics, "Smooth")
    assert result["Selected_Method"] == "SES"
    assert result["Selected_WMAPE"] == 30.0
    assert result["Tracking_Signal_Guardrail_Applied"] is False
    assert result["Fallback_Used"] is False


def test_holt_winters_excluded_for_intermittent_even_if_best_score():
    # Holt-Winters has the best WMAPE here, but it's never a candidate
    # for Intermittent SKUs -- SES should win among the allowed set.
    metrics = {
        "SMA": _m(50.0, ts=1.0),
        "SES": _m(45.0, ts=1.0),
        "Holt-Winters": _m(10.0, ts=1.0),  # best overall, but not a candidate
        "Croston": _m(48.0, ts=1.0),
    }
    result = select_method_for_sku(metrics, "Intermittent")
    assert result["Selected_Method"] == "SES"


def test_guardrail_switches_to_stable_alternative_within_tolerance():
    # Best WMAPE (SES) has a tracking signal way out of control (10).
    # SMA is within 15% of SES's WMAPE and has an in-control TS -> should
    # be selected instead, with the guardrail flagged and SES demoted to
    # runner-up.
    metrics = {
        "SMA": _m(33.0, ts=1.5),           # within 10% of 30 and in control
        "SES": _m(30.0, ts=10.0),          # best WMAPE, but TS way out of control
        "Holt-Winters": _m(60.0, ts=1.0),
    }
    result = select_method_for_sku(metrics, "Smooth")
    assert result["Selected_Method"] == "SMA"
    assert result["Runner_Up_Method"] == "SES"
    assert result["Tracking_Signal_Guardrail_Applied"] is True


def test_guardrail_not_applied_when_no_alternative_is_close_enough():
    # SES's TS is out of control, but the only other candidate (SMA) is
    # far outside the WMAPE tolerance -- guardrail should NOT override.
    metrics = {
        "SES": _m(30.0, ts=10.0),
        "SMA": _m(90.0, ts=1.0),  # in control but way worse WMAPE
    }
    result = select_method_for_sku(metrics, "Smooth")
    assert result["Selected_Method"] == "SES"
    assert result["Tracking_Signal_Guardrail_Applied"] is False


def test_fallback_widens_search_when_no_in_class_candidate_is_valid():
    # Every Intermittent-class candidate (SMA/SES/Croston) has undefined
    # WMAPE; Holt-Winters (not normally a candidate for this class) does
    # have a valid score, so it should be used as a flagged fallback.
    metrics = {
        "SMA": _m(np.nan),
        "SES": _m(np.nan),
        "Croston": _m(np.nan),
        "Holt-Winters": _m(40.0, ts=1.0),
    }
    result = select_method_for_sku(metrics, "Intermittent")
    assert result["Selected_Method"] == "Holt-Winters"
    assert result["Fallback_Used"] is True


def test_no_eligible_method_when_everything_is_undefined():
    metrics = {
        "SMA": _m(np.nan),
        "SES": _m(np.nan),
        "Holt-Winters": _m(np.nan),
        "Croston": _m(np.nan),
    }
    result = select_method_for_sku(metrics, "Smooth")
    assert result["Selected_Method"] == "No Eligible Method (undefined WMAPE for all candidates)"
    assert np.isnan(result["Selected_WMAPE"])
    assert result["Runner_Up_Method"] == "No Runner-Up"


def test_runner_up_is_no_runner_up_string_when_only_one_candidate():
    metrics = {"Croston": _m(50.0, ts=1.0)}
    result = select_method_for_sku(metrics, "Lumpy")
    assert result["Selected_Method"] == "Croston"
    assert result["Runner_Up_Method"] == "No Runner-Up"


def test_undefined_demand_class_allows_every_method_as_candidate():
    metrics = {
        "SMA": _m(50.0, ts=1.0),
        "Holt-Winters": _m(20.0, ts=1.0),  # would be excluded under Intermittent/Lumpy
    }
    result = select_method_for_sku(metrics, "Undefined")
    assert result["Selected_Method"] == "Holt-Winters"
