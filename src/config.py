"""Project-wide configuration: paths and constants.

Centralizing these here means every other module imports config.X instead
of hardcoding paths or values. Changing which store/category we use, how
many SKUs to sample, or any of the forecasting thresholds later is a
one-line edit in this file rather than a hunt through the codebase.
"""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- Data selection (M5 Forecasting Accuracy dataset) ---------------------
# We only use one store and one category out of the full M5 dataset
# (10 stores x 3 categories) to keep this a portfolio-sized project
# rather than a full production-scale one.
STORE_ID = "CA_1"
CATEGORY_ID = "FOODS"

N_SKUS = 35            # how many SKUs to sample down to, out of everything
                        # available in STORE_ID + CATEGORY_ID
RANDOM_SEED = 42        # reproducibility: same 35 SKUs chosen every run

# --- Forecasting / evaluation (used by later modules) ---------------------
MIN_HISTORY_FOR_EVALUATION = 12
MIN_HISTORY_FOR_SEASONALITY = 24
TEST_SIZE_FRACTION = 0.2
TRACKING_SIGNAL_CONTROL_LIMIT = 4.0

# Demand classification thresholds (Syntetos & Boylan, 2005) --------------
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49
