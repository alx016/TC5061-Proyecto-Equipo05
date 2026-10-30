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
import argparse
import json
from pathlib import Path

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    DataDuplicates,
    PercentOfNulls,
)
from deepchecks.tabular.suites import data_integrity, train_test_validation
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]

# Mirrors CLASSIFICATION_FEATURES_PATH in tc5061_proyecto_equipo5/config.py
DEFAULT_DATA_PATH = PROJ_ROOT / "data" / "processed" / "classification_features.parquet"

# Mirrors FEATURE_COLUMNS in tc5061_proyecto_equipo5/modeling/train_classification.py
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
DEFAULT_REPORTS_DIR = PROJ_ROOT / "reports" / "deepchecks"


def save_suite_summary(result, html_path: Path, json_path: Path) -> dict:
    """Save DeepChecks suite results as HTML and structured JSON summary."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    if html_path.exists():
        html_path.unlink()
    result.save_as_html(str(html_path))
    print(f"Report saved to {html_path}")

    raw_json = json.loads(result.to_json(with_display=False))
    passed_checks = [c.header for c in result.get_passed_checks()]
    not_passed_checks = [c.header for c in result.get_not_passed_checks()]

    check_summaries = []
    for r in raw_json.get("results", []):
        conds = []
        for cr in r.get("conditions_results", []):
            conds.append(
                {
                    "name": cr.get("Condition") or cr.get("name"),
                    "status": cr.get("Status") or cr.get("status"),
                    "category": cr.get("Category") or cr.get("category"),
                    "details": cr.get("More Info") or cr.get("details"),
                }
            )
        check_summaries.append(
            {
                "header": r.get("header"),
                "check_name": (
                    r.get("check", {}).get("name")
                    if isinstance(r.get("check"), dict)
                    else str(r.get("check"))
                ),
                "type": r.get("type"),
                "conditions": conds,
                "value": r.get("value"),
            }
        )

    summary = {
        "suite_name": raw_json.get("name", getattr(result, "name", "suite")),
        "passed": bool(result.passed()),
        "total_checks": len(raw_json.get("results", [])),
        "passed_checks_count": len(passed_checks),
        "failed_checks_count": len(not_passed_checks),
        "passed_checks": passed_checks,
        "failed_checks": not_passed_checks,
        "check_summaries": check_summaries,
        "raw": raw_json,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary JSON saved to {json_path}")
    return summary


def _prepare_dataset(df: pd.DataFrame) -> Dataset:
    """Prepare a DeepChecks Dataset ensuring record_id, feature schema, and datetime metadata."""
    df = df.copy()
    if "record_id" not in df.columns:
        if "station_id" in df.columns and "hour" in df.columns:
            df["record_id"] = df["station_id"].astype(str) + "_" + df["hour"].astype(str)
        else:
            df["record_id"] = [f"rec_{i}" for i in range(len(df))]

    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    available_cat = [col for col in CATEGORICAL_LIST if col in available_features]
    datetime_col = "hour" if "hour" in df.columns else None

    return Dataset(
        df,
        label="target" if "target" in df.columns else None,
        features=available_features,
        cat_features=available_cat,
        datetime_name=datetime_col,
        index_name="record_id",
        label_type="multiclass" if "target" in df.columns else None,
    )


def run_data_integrity_check(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run data integrity suite and export HTML + JSON."""
    ds = _prepare_dataset(df)

    suite = data_integrity()
    for check in suite.checks.values():
        if isinstance(check, DataDuplicates):
            check.clean_conditions()
            check.add_condition_ratio_less_or_equal(0)
    suite.add(PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0.01))

    result = suite.run(ds)
    html_path = output_dir / "data_integrity.html"
    json_path = output_dir / "data_integrity.json"
    return save_suite_summary(result, html_path, json_path)


def run_train_test_drift_check(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run train/test drift suite (Jan-Sep train vs Dec test) and export HTML + JSON."""
    if (
        "split" not in df.columns
        or "train" not in df["split"].values
        or "test" not in df["split"].values
    ):
        raise ValueError(
            "DataFrame must contain a 'split' column with both 'train' and 'test' partitions."
        )

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    if "target" in train_df.columns:
        train_df = train_df.dropna(subset=["target"])
    if "target" in test_df.columns:
        test_df = test_df.dropna(subset=["target"])

    train_ds = _prepare_dataset(train_df)
    test_ds = _prepare_dataset(test_df)

    sample_size = min(50000, max(len(train_df), len(test_df)))
    suite = train_test_validation(n_samples=sample_size)
    suite.name = "train_test_drift"

    result = suite.run(train_dataset=train_ds, test_dataset=test_ds)
    html_path = output_dir / "train_test_drift.html"
    json_path = output_dir / "train_test_drift.json"
    return save_suite_summary(result, html_path, json_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepChecks data integrity and train/test drift suites."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to classification features parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory to save HTML and JSON reports",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)
    print(f"Loaded dataset: {df.shape}")
    if "split" in df.columns:
        print(df["split"].value_counts())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Running Data Integrity suite...")
    run_data_integrity_check(df, args.output_dir)

    print("Running Train/Test Drift suite (train Jan-Sep vs test Dec)...")
    run_train_test_drift_check(df, args.output_dir)

    print("All DeepChecks reports successfully generated.")


if __name__ == "__main__":
    main()
