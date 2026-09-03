"""Load and reshape the raw M5 Forecasting Accuracy dataset into the
long-format monthly SKU-level table the rest of this project's pipeline
expects: one row per (Date, Product_Family, SKU, Demand).

The raw M5 files are wide and daily -- one row per item-store, one column
per day (d_1 ... d_1941). Real demand-planning tools work at a monthly,
long-format grain, so this module does that reshape once, up front, and
writes the result to data/processed/monthly_demand.csv.

Run directly:
    python -m src.data_loader
"""
import numpy as np
import pandas as pd

from src import config


def load_calendar() -> pd.DataFrame:
    """Load calendar.csv and keep only what we need: the mapping from each
    'd_###' column name in sales_train_evaluation.csv to a real calendar
    date."""
    calendar = pd.read_csv(config.RAW_DIR / "calendar.csv", usecols=["date", "d"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    return calendar


def load_sales() -> pd.DataFrame:
    """Load the raw M5 sales file and filter down to one store and one
    category (config.STORE_ID / config.CATEGORY_ID). The full file is
    ~30,490 rows x 1941+ day columns; filtering before reshaping keeps
    memory usage sane."""
    sales = pd.read_csv(config.RAW_DIR / "sales_train_evaluation.csv")
    sales = sales[
        (sales["store_id"] == config.STORE_ID)
        & (sales["cat_id"] == config.CATEGORY_ID)
    ].reset_index(drop=True)
    return sales


def wide_to_long(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Reshape from M5's native wide format (one row per item, one column
    per day) into long format (one row per item-day), then attach the real
    calendar date for each d_### column via a merge."""
    id_cols = ["item_id", "dept_id", "cat_id", "store_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    long_df = sales.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="units_sold",
    )
    long_df = long_df.merge(calendar, on="d", how="left")
    return long_df


def aggregate_monthly(long_df: pd.DataFrame) -> pd.DataFrame:
    """Roll daily unit sales up to monthly demand per item. Uses
    month-start timestamps so every SKU's history lines up on the same
    monthly grid downstream (needed for rolling-origin backtesting)."""
    long_df = long_df.copy()
    long_df["Month"] = long_df["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        long_df.groupby(["item_id", "dept_id", "Month"], as_index=False)["units_sold"]
        .sum()
        .rename(columns={
            "item_id": "SKU",
            "dept_id": "Product_Family",
            "units_sold": "Demand",
            "Month": "Date",
        })
    )
    monthly = monthly[["Date", "Product_Family", "SKU", "Demand"]]
    return monthly.sort_values(["SKU", "Date"]).reset_index(drop=True)


def sample_skus(monthly: pd.DataFrame) -> pd.DataFrame:
    """Reproducibly sample config.N_SKUS SKUs out of everything available
    in this store/category, using config.RANDOM_SEED so the same SKUs are
    chosen on every run (and by anyone else who clones this repo)."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    all_skus = monthly["SKU"].unique()
    n = min(config.N_SKUS, len(all_skus))
    chosen = rng.choice(all_skus, size=n, replace=False)
    sampled = monthly[monthly["SKU"].isin(chosen)].reset_index(drop=True)
    return sampled


def main() -> pd.DataFrame:
    calendar = load_calendar()
    sales = load_sales()
    print(f"Loaded {len(sales)} SKUs for store={config.STORE_ID}, category={config.CATEGORY_ID}")

    long_df = wide_to_long(sales, calendar)
    monthly = aggregate_monthly(long_df)
    sampled = sample_skus(monthly)

    print(f"Sampled {sampled['SKU'].nunique()} SKUs, {len(sampled)} SKU-month rows")
    print(f"Date range: {sampled['Date'].min().date()} to {sampled['Date'].max().date()}")
    print(f"Product families present: {sorted(sampled['Product_Family'].unique())}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DIR / "monthly_demand.csv"
    sampled.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return sampled


if __name__ == "__main__":
    main()
