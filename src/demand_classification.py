"""Classify each SKU's demand pattern using the ADI / CV^2 framework
(Syntetos & Boylan, 2005) so later modules know which forecasting methods
even make sense to try for a given SKU.

    ADI (Average Demand Interval) = periods / periods with non-zero demand
        Low ADI  -> demand shows up almost every period.
        High ADI -> demand is sparse; long gaps between non-zero periods.

    CV^2 (squared coefficient of variation) = (std / mean)^2 of the
    *non-zero* demand values only.
        Low CV^2  -> when demand happens, it's a consistent size.
        High CV^2 -> when demand happens, its size is all over the place.

                  CV^2 < 0.49      CV^2 >= 0.49
    ADI < 1.32     Smooth           Erratic
    ADI >= 1.32    Intermittent     Lumpy

Run directly:
    python -m src.demand_classification
"""
import numpy as np
import pandas as pd

from src import config


def compute_adi(demand: pd.Series) -> float:
    """periods / periods-with-nonzero-demand. np.inf if every period is zero
    (undefined in the literature -- there's no 'interval between demand
    occurrences' when demand never occurs)."""
    n_periods = len(demand)
    n_nonzero = int((demand > 0).sum())
    if n_nonzero == 0:
        return np.inf
    return n_periods / n_nonzero


def compute_cv2(demand: pd.Series) -> float:
    """(std / mean)^2 computed over non-zero periods only, per the standard
    convention in the intermittent-demand literature. NaN (not 0) with
    fewer than 2 non-zero observations -- you can't measure variability
    from a single point, and silently returning 0 would misrepresent that
    as 'perfectly consistent'."""
    nonzero = demand[demand > 0]
    if len(nonzero) < 2:
        return np.nan
    return (nonzero.std() / nonzero.mean()) ** 2


def classify_demand(adi: float, cv2: float) -> str:
    """Apply the Syntetos & Boylan (2005) 2x2 thresholds. Returns
    'Undefined' when cv2 is NaN -- either an all-zero SKU (ADI = inf) or a
    SKU with fewer than 2 non-zero months of history -- rather than forcing
    it into a bucket the math can't actually support."""
    if np.isnan(cv2):
        return "Undefined"
    if adi < config.ADI_THRESHOLD:
        return "Erratic" if cv2 >= config.CV2_THRESHOLD else "Smooth"
    return "Lumpy" if cv2 >= config.CV2_THRESHOLD else "Intermittent"


def classify_all_skus(monthly: pd.DataFrame) -> pd.DataFrame:
    """Run the ADI/CV^2 classification for every SKU in a long-format
    (Date, Product_Family, SKU, Demand) DataFrame."""
    rows = []
    for sku, grp in monthly.groupby("SKU"):
        grp = grp.sort_values("Date")
        demand = grp["Demand"]
        adi = compute_adi(demand)
        cv2 = compute_cv2(demand)
        rows.append({
            "SKU": sku,
            "Product_Family": grp["Product_Family"].iloc[0],
            "n_months": len(demand),
            "pct_zero_months": (demand == 0).mean(),
            "mean_demand": demand.mean(),
            "ADI": adi,
            "CV2": cv2,
            "Demand_Class": classify_demand(adi, cv2),
        })
    return pd.DataFrame(rows).sort_values("SKU").reset_index(drop=True)


def main() -> pd.DataFrame:
    monthly = pd.read_csv(config.PROCESSED_DIR / "monthly_demand.csv", parse_dates=["Date"])
    classified = classify_all_skus(monthly)

    print("Demand class counts:")
    print(classified["Demand_Class"].value_counts().to_string())
    print()
    print("By product family:")
    print(pd.crosstab(classified["Product_Family"], classified["Demand_Class"]))

    out_path = config.PROCESSED_DIR / "demand_classification.csv"
    classified.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return classified


if __name__ == "__main__":
    main()
