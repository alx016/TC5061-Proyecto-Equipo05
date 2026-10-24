from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# data/ is not tracked by Git; this file is, and records which data version we have
DATA_MANIFEST_PATH = PROJ_ROOT / "references" / "viajes_manifest.json"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

INITIAL_YEAR = 2024
END_YEAR = 2026

# Classification study: predict if a station will be short of bikes or docks next hour
STUDY_YEAR = 2024
# A station is at risk when one hour moves at least 40% of its capacity
RISK_THRESHOLD = 0.40

# Temporal split: train in the first nine months, validate and test out-of-time
TRAIN_MONTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
VALIDATION_MONTHS = (10, 11)
TEST_MONTHS = (12,)

# Shared station-hour base and the classification feature table
CLASSIFICATION_PANEL_PATH = PROCESSED_DATA_DIR / "panel_station_hour.parquet"
CLASSIFICATION_FEATURES_PATH = PROCESSED_DATA_DIR / "classification_features.parquet"

# Prepared P1 and P3 artifacts, written by `features build --question p1|p3`
FEATURES_DIR = PROCESSED_DATA_DIR / "fe_2024_v4"

REGRESSION_FEATURES_DIR = FEATURES_DIR / "p1"
CLUSTERING_FEATURES_DIR = FEATURES_DIR / "p3"

# Shared seed for every model and split, so runs are repeatable
SEED = 42

# P2 temporal protocol: labels must be closed before the first scoring of the next split.
# The 10 minutes before the scoring are the budget to fit the model.
P2_FIT_AS_OF = "2024-09-30 23:40"
P2_SELECTION_AS_OF = "2024-11-30 23:40"

# P2 selection guardrail: the operational decision needs at least 85% of the shortages
# detected. The decision thresholds are searched jointly on validation to maximize macro-F1
# subject to that recall; there are no precision floors because 0.7% prevalence makes them
# unreachable (see the D2 report for the feasibility table).
P2_MIN_RECALL_SHORTAGE = 0.85
# 19 probability thresholds from 0.05 to 0.95 in steps of 0.05, shared by both risk classes.
P2_THRESHOLD_GRID = tuple(round(0.05 * index, 2) for index in range(1, 20))

# Small grid for the nested LightGBM search: 3 x 3 x 1 = 9 child runs.
P2_SEARCH_GRID = {
    "learning_rate": (0.03, 0.05, 0.1),
    "num_leaves": (31, 63, 127),
    "min_child_samples": (100,),
}
