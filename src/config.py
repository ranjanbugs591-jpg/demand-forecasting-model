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
SEASONAL_PERIOD = 12          # monthly data -> annual seasonality
SMA_WINDOWS = [3, 6, 12]      # candidate windows; best one picked per-SKU
CROSTON_ALPHA = 0.1           # classic Croston smoothing parameter

# Demand classification thresholds (Syntetos & Boylan, 2005) --------------
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

# --- Model selection --------------------------------------------------
# Which forecasting methods are even considered for each demand class.
# Holt-Winters is never a candidate for Intermittent/Lumpy SKUs by design:
# it smooths every period equally, which drags a sparse, spiky series
# toward zero. "Undefined"-classified SKUs (too little history to
# classify at all) get every method as a candidate, since classification
# itself couldn't tell us anything to filter on.
CANDIDATE_METHODS_BY_CLASS = {
    "Smooth": ["SMA", "SES", "Holt-Winters"],
    "Erratic": ["SMA", "SES", "Holt-Winters"],
    "Intermittent": ["SMA", "SES", "Croston"],
    "Lumpy": ["SMA", "SES", "Croston"],
    "Undefined": ["SMA", "SES", "Holt-Winters", "Croston"],
}

# Tracking-signal guardrail: how much worse (relative to the best
# candidate's WMAPE) a more-stable alternative is allowed to be and still
# override the top pick.
GUARDRAIL_WMAPE_TOLERANCE = 0.15
