# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "deepchecks==0.19.1",
#     "scikit-learn==1.5.2",
#     "pandas==2.2.3",
#     "numpy==1.26.4",
#     "setuptools<81",
#     "plotly<6",
#     "pyarrow<22",
#     "ipywidgets>=7.6.5,<8",
# ]
# ///
from pathlib import Path

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from deepchecks.tabular.checks import DataDuplicates, PercentOfNulls
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Mirrors CLASSIFICATION_FEATURES_PATH in tc5061_proyecto_equipo5/config.py, which this
# isolated script cannot import
DATA_PATH = PROJ_ROOT / "data" / "processed" / "classification_features.parquet"

# Mirrors FEATURE_COLUMNS in tc5061_proyecto_equipo5/modeling/train_classification.py, 
# which this isolated script cannot import
FEATURE_COLUMNS = [
    "capacity",
    "latitude",
    "longitude",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "neighbor_net_flow",
    "net_flow_lag1",
    "net_flow_lag2",
    "net_flow_lag24",
    "arrivals_lag1",
    "arrivals_lag2",
    "arrivals_lag3",
    "departures_lag1",
    "departures_lag2",
    "departures_lag3",
]

CATEGORICAL_LIST = ["day_of_week", "is_weekend", "is_holiday"]
REPORTS_DIR = PROJ_ROOT / "reports" / "deepchecks"

df = pd.read_parquet(DATA_PATH)
print(df.shape)
print(df["split"].value_counts())

ds = Dataset(
    df,
    label="target",
    features=FEATURE_COLUMNS,
    cat_features=CATEGORICAL_LIST,
    datetime_name="hour",
    label_type="multiclass"    
)

print(ds.n_samples)
print(ds.label_type)

suite = data_integrity()
for check in suite.check.values():
    if isinstance(check, DataDuplicates):
        check.clean_conditions().add_condition_ratio_less_or_equal(0)
suite.add(PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0.01))
result = suite.run(ds)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
report_path = REPORTS_DIR / "data_integrity.html"
result.save_as_html(str(report_path))
print(f"Report saved to {report_path}")