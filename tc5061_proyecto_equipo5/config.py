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
