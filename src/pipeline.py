"""End-to-end pipeline: classify every SKU's demand, backtest all four
forecasting methods against it with rolling-origin evaluation, apply
rule-based model selection, and save two output files:

    data/processed/model_evaluation.csv
        One row per SKU: its demand classification, every method's
        WMAPE / Bias / Tracking Signal, and the model-selection decision
        (selected method, runner-up, guardrail/fallback flags).

    data/processed/selected_forecasts.csv
        One row per (SKU, Date) in that SKU's test window, using only
        the SELECTED method's forecast -- i.e. what you'd actually hand
        a planner, not every method's output.

Run directly:
    python -m src.pipeline
"""
import numpy as np
import pandas as pd

from src import config
from src.demand_classification import classify_all_skus
from src.forecasting import (
    chronological_split,
    rolling_origin_forecast,
    sma_forecast,
    select_sma_window,
    ses_forecast,
    holt_winters_forecast,
    croston_forecast,
)
from src.metrics import wmape, bias, tracking_signal
from src.model_selection import select_method_for_sku

METHODS = ["SMA", "SES", "Holt-Winters", "Croston"]


def _get_sku_series(monthly: pd.DataFrame, sku: str) -> pd.Series:
    grp = monthly[monthly["SKU"] == sku].sort_values("Date")
    return pd.Series(
        grp["Demand"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(grp["Date"]),
        name=sku,
    )


def _build_forecasters(train: pd.Series):
    """Returns (chosen SMA window, {method_name: forecast_fn}). SMA's
    window is chosen once per SKU, from its training data only, before
    the real backtest runs."""
    sma_window = select_sma_window(train)
    forecasters = {
        "SMA": lambda h, w=sma_window: sma_forecast(h, w),
        "SES": ses_forecast,
        "Holt-Winters": holt_winters_forecast,
        "Croston": croston_forecast,
    }
    return sma_window, forecasters


def _short_history_row(base_row: dict) -> dict:
    """A SKU with too little history to evaluate isn't silently dropped
    -- it gets a flagged row with NaN metrics so it stays visible in the
    output, per config.MIN_HISTORY_FOR_EVALUATION."""
    nan_metrics = {}
    for m in METHODS:
        key = m.replace("-", "_")
        nan_metrics[f"WMAPE_{key}"] = np.nan
        nan_metrics[f"Bias_{key}"] = np.nan
        nan_metrics[f"TrackingSignal_{key}"] = np.nan
    return {
        **base_row,
        "Short_History_Flag": True,
        "n_train": np.nan, "n_test": np.nan, "SMA_Window": np.nan,
        **nan_metrics,
        "Selected_Method": "No Eligible Method (short history)",
        "Selected_WMAPE": np.nan,
        "Runner_Up_Method": "No Runner-Up",
        "Tracking_Signal_Guardrail_Applied": False,
        "Fallback_Used": False,
    }


def run_backtests(monthly: pd.DataFrame):
    """Run the full pipeline over every SKU present in `monthly`.
    Returns (evaluation, selected_forecasts) -- see module docstring."""
    classification = classify_all_skus(monthly).set_index("SKU")

    eval_rows = []
    forecast_rows = []

    for sku, class_row in classification.iterrows():
        series = _get_sku_series(monthly, sku)
        n = len(series)

        base_row = {
            "SKU": sku,
            "Product_Family": class_row["Product_Family"],
            "ADI": class_row["ADI"],
            "CV2": class_row["CV2"],
            "Demand_Class": class_row["Demand_Class"],
            "n_months": n,
        }

        if n < config.MIN_HISTORY_FOR_EVALUATION:
            eval_rows.append(_short_history_row(base_row))
            continue

        train, test = chronological_split(series)
        n_test = len(test)
        sma_window, forecasters = _build_forecasters(train)

        actual = test.to_numpy()
        metrics_by_method = {}
        forecasts_by_method = {}
        for name, fn in forecasters.items():
            fc = rolling_origin_forecast(series, n_test, fn)
            fc_vals = fc.to_numpy()
            metrics_by_method[name] = {
                "WMAPE": wmape(actual, fc_vals),
                "Bias": bias(actual, fc_vals),
                "Tracking_Signal": tracking_signal(actual, fc_vals),
            }
            forecasts_by_method[name] = fc

        selection = select_method_for_sku(metrics_by_method, class_row["Demand_Class"])

        row = {
            **base_row,
            "Short_History_Flag": False,
            "n_train": len(train),
            "n_test": n_test,
            "SMA_Window": sma_window,
            "Selected_Method": selection["Selected_Method"],
            "Selected_WMAPE": selection["Selected_WMAPE"],
            "Runner_Up_Method": selection["Runner_Up_Method"],
            "Tracking_Signal_Guardrail_Applied": selection["Tracking_Signal_Guardrail_Applied"],
            "Fallback_Used": selection["Fallback_Used"],
        }
        for name in METHODS:
            key = name.replace("-", "_")
            row[f"WMAPE_{key}"] = metrics_by_method[name]["WMAPE"]
            row[f"Bias_{key}"] = metrics_by_method[name]["Bias"]
            row[f"TrackingSignal_{key}"] = metrics_by_method[name]["Tracking_Signal"]
        eval_rows.append(row)

        # Record the SELECTED method's per-period forecasts -- what a
        # planner would actually see, not every method's output.
        selected_method = selection["Selected_Method"]
        if selected_method in forecasts_by_method:
            fc = forecasts_by_method[selected_method]
            for date, actual_val, forecast_val in zip(test.index, test.values, fc.values):
                forecast_rows.append({
                    "SKU": sku,
                    "Product_Family": class_row["Product_Family"],
                    "Date": date,
                    "Actual": actual_val,
                    "Forecast": forecast_val,
                    "Method": selected_method,
                })

    evaluation = pd.DataFrame(eval_rows)
    selected_forecasts = pd.DataFrame(forecast_rows)
    return evaluation, selected_forecasts


def main():
    monthly = pd.read_csv(config.PROCESSED_DIR / "monthly_demand.csv", parse_dates=["Date"])
    evaluation, selected_forecasts = run_backtests(monthly)

    print(f"Evaluated {len(evaluation)} SKUs\n")

    valid = evaluation[~evaluation["Selected_Method"].str.startswith("No Eligible")]
    print(f"{len(valid)} SKUs had an eligible selection "
          f"({len(evaluation) - len(valid)} did not).\n")

    if len(valid):
        print("Method win rate:")
        win_rate = valid["Selected_Method"].value_counts(normalize=True).mul(100).round(1)
        print(win_rate.astype(str) + "%")
        print()
        print("Average Selected_WMAPE by demand class:")
        print(valid.groupby("Demand_Class")["Selected_WMAPE"].mean().round(1))
        print()

    print(f"Tracking-signal guardrail applied: {int(evaluation['Tracking_Signal_Guardrail_Applied'].sum())} SKUs")
    print(f"Fallback used: {int(evaluation['Fallback_Used'].sum())} SKUs")
    print(f"Short-history SKUs: {int(evaluation['Short_History_Flag'].sum())}")

    eval_path = config.PROCESSED_DIR / "model_evaluation.csv"
    forecasts_path = config.PROCESSED_DIR / "selected_forecasts.csv"
    evaluation.to_csv(eval_path, index=False)
    selected_forecasts.to_csv(forecasts_path, index=False)
    print(f"\nSaved: {eval_path}")
    print(f"Saved: {forecasts_path}")
    return evaluation, selected_forecasts


if __name__ == "__main__":
    main()
