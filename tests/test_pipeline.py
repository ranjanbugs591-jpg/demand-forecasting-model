"""Integration test for src/pipeline.py: runs the real end-to-end
run_backtests() function (not mocked) on a tiny synthetic dataset small
enough to run in under a second, specifically to exercise the
short-history path alongside two normally-evaluated SKUs in one pass.
"""
import numpy as np
import pandas as pd

from src.pipeline import run_backtests
from src import config


def _make_monthly_df():
    dates_long = pd.date_range("2020-01-01", periods=15, freq="MS")
    dates_short = pd.date_range("2020-01-01", periods=8, freq="MS")  # below MIN_HISTORY_FOR_EVALUATION

    rows = []
    # SKU_A: steady upward trend, always non-zero -> plenty of history.
    for date, demand in zip(dates_long, range(10, 25)):
        rows.append({"Date": date, "Product_Family": "TESTFAM_A", "SKU": "SKU_A", "Demand": demand})

    # SKU_B: intermittent pattern, plenty of history.
    demand_b = [0, 5, 0, 0, 8, 0, 0, 0, 6, 0, 0, 9, 0, 0, 7]
    for date, demand in zip(dates_long, demand_b):
        rows.append({"Date": date, "Product_Family": "TESTFAM_B", "SKU": "SKU_B", "Demand": demand})

    # SKU_SHORT: only 8 months -> should hit the short-history path.
    for date, demand in zip(dates_short, range(1, 9)):
        rows.append({"Date": date, "Product_Family": "TESTFAM_A", "SKU": "SKU_SHORT", "Demand": demand})

    return pd.DataFrame(rows)


def test_pipeline_runs_end_to_end_on_synthetic_data():
    monthly = _make_monthly_df()
    evaluation, selected_forecasts = run_backtests(monthly)

    assert len(evaluation) == 3
    assert set(evaluation["SKU"]) == {"SKU_A", "SKU_B", "SKU_SHORT"}

    # Every column the pipeline promises to produce is actually present.
    expected_cols = {
        "SKU", "Product_Family", "ADI", "CV2", "Demand_Class", "n_months",
        "Short_History_Flag", "n_train", "n_test", "SMA_Window",
        "Selected_Method", "Selected_WMAPE", "Runner_Up_Method",
        "Tracking_Signal_Guardrail_Applied", "Fallback_Used",
    }
    assert expected_cols.issubset(evaluation.columns)


def test_short_history_sku_is_flagged_not_dropped():
    monthly = _make_monthly_df()
    evaluation, selected_forecasts = run_backtests(monthly)

    short_row = evaluation.set_index("SKU").loc["SKU_SHORT"]
    assert short_row["Short_History_Flag"] == True  # noqa: E712
    assert np.isnan(short_row["Selected_WMAPE"])
    assert short_row["Selected_Method"].startswith("No Eligible Method")

    # A short-history SKU should never appear in selected_forecasts.
    assert "SKU_SHORT" not in set(selected_forecasts["SKU"])


def test_normal_skus_get_valid_selections_and_forecasts():
    monthly = _make_monthly_df()
    evaluation, selected_forecasts = run_backtests(monthly)

    for sku in ["SKU_A", "SKU_B"]:
        row = evaluation.set_index("SKU").loc[sku]
        assert row["Short_History_Flag"] == False  # noqa: E712
        assert not np.isnan(row["Selected_WMAPE"])
        assert not row["Selected_Method"].startswith("No Eligible")

        sku_forecasts = selected_forecasts[selected_forecasts["SKU"] == sku]
        assert len(sku_forecasts) == row["n_test"]
        assert (sku_forecasts["Method"] == row["Selected_Method"]).all()


def test_holt_winters_never_selected_for_intermittent_or_lumpy_skus():
    # SKU_B is intermittent -- whatever method wins, it must come from
    # the allowed candidate set, never Holt-Winters.
    monthly = _make_monthly_df()
    evaluation, _ = run_backtests(monthly)
    row = evaluation.set_index("SKU").loc["SKU_B"]
    if row["Demand_Class"] in ("Intermittent", "Lumpy"):
        assert row["Selected_Method"] != "Holt-Winters"
