"""Rule-based, auditable model selection: given one SKU's backtest metrics
for every forecasting method that was tried, decide which method to
actually use for that SKU.

This module deliberately does NOT run any forecasts itself -- it takes
already-computed metrics as input (that's src/forecasting.py + a future
pipeline step's job) and only makes the selection decision. Keeping the
decision logic separate from the forecasting means it can be tested with
small, hand-built metric tables instead of needing real backtests every
time.

Decision logic (see docs/methodology.md for the full write-up):
    1. Candidate filtering by demand class (config.CANDIDATE_METHODS_BY_CLASS)
       -- e.g. Holt-Winters is never a candidate for Intermittent/Lumpy SKUs.
    2. Selection: the lowest-WMAPE candidate.
    3. Guardrail: if that pick's tracking signal is out of control (beyond
       +/- config.TRACKING_SIGNAL_CONTROL_LIMIT) AND a candidate within
       config.GUARDRAIL_WMAPE_TOLERANCE of its WMAPE has an in-control
       tracking signal, switch to that more stable candidate instead --
       and flag that this happened, rather than hiding it.
    4. Fallback: if every in-class candidate has an undefined (NaN) WMAPE,
       widen the search to every method that was actually evaluated. If
       even that finds nothing, report "No Eligible Method" rather than
       assigning an arbitrary default.
"""
import numpy as np

from src import config


def select_method_for_sku(metrics_by_method: dict, demand_class: str) -> dict:
    """
    metrics_by_method: {method_name: {"WMAPE": float, "Bias": float,
        "Tracking_Signal": float}} for every method actually evaluated
        for this SKU (any of WMAPE/Bias/Tracking_Signal may be NaN).
    demand_class: "Smooth" / "Erratic" / "Intermittent" / "Lumpy" /
        "Undefined" (from src.demand_classification.classify_demand).

    Returns:
        {
            "Selected_Method": str,
            "Selected_WMAPE": float,
            "Runner_Up_Method": str,
            "Tracking_Signal_Guardrail_Applied": bool,
            "Fallback_Used": bool,
        }
    """
    candidates = config.CANDIDATE_METHODS_BY_CLASS.get(
        demand_class, list(metrics_by_method.keys())
    )
    valid_candidates = [
        m for m in candidates
        if m in metrics_by_method and not np.isnan(metrics_by_method[m]["WMAPE"])
    ]

    fallback_used = False
    if not valid_candidates:
        # Nothing in-class has a usable score -- widen to every method
        # that was actually evaluated, regardless of demand class.
        valid_candidates = [
            m for m in metrics_by_method if not np.isnan(metrics_by_method[m]["WMAPE"])
        ]
        fallback_used = bool(valid_candidates)

    if not valid_candidates:
        return {
            "Selected_Method": "No Eligible Method (undefined WMAPE for all candidates)",
            "Selected_WMAPE": np.nan,
            "Runner_Up_Method": "No Runner-Up",
            "Tracking_Signal_Guardrail_Applied": False,
            "Fallback_Used": False,
        }

    ranked = sorted(valid_candidates, key=lambda m: metrics_by_method[m]["WMAPE"])
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    guardrail_applied = False
    best_ts = metrics_by_method[best]["Tracking_Signal"]
    if not np.isnan(best_ts) and abs(best_ts) > config.TRACKING_SIGNAL_CONTROL_LIMIT:
        best_wmape = metrics_by_method[best]["WMAPE"]
        for candidate in ranked[1:]:
            cand_wmape = metrics_by_method[candidate]["WMAPE"]
            cand_ts = metrics_by_method[candidate]["Tracking_Signal"]
            within_tolerance = cand_wmape <= best_wmape * (1 + config.GUARDRAIL_WMAPE_TOLERANCE)
            in_control = not np.isnan(cand_ts) and abs(cand_ts) <= config.TRACKING_SIGNAL_CONTROL_LIMIT
            if within_tolerance and in_control:
                runner_up = best
                best = candidate
                guardrail_applied = True
                break

    return {
        "Selected_Method": best,
        "Selected_WMAPE": metrics_by_method[best]["WMAPE"],
        "Runner_Up_Method": runner_up if runner_up else "No Runner-Up",
        "Tracking_Signal_Guardrail_Applied": guardrail_applied,
        "Fallback_Used": fallback_used,
    }
