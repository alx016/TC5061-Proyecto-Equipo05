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
    ConflictingLabels,
    DataDuplicates,
    DateTrainTestLeakageDuplicates,
    DateTrainTestLeakageOverlap,
    FeatureDrift,
    FeatureFeatureCorrelation,
    FeatureLabelCorrelation,
    IndexTrainTestLeakage,
    LabelDrift,
    MultivariateDrift,
    PercentOfNulls,
    TrainTestSamplesMix,
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
MAX_CONFLICT_RATIO = 0.005
MAX_SAMPLES_MIX_RATIO = 0.10


def validate_input_schema(df: pd.DataFrame) -> None:
    """Reject incomplete or malformed inputs before running any suite."""
    required = set(FEATURE_COLUMNS) | {"station_id", "hour", "target", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Dataset must not be empty")
    if not pd.api.types.is_datetime64_any_dtype(df["hour"]):
        raise ValueError("hour must be a datetime column")
    if df[["station_id", "hour"]].isna().any().any():
        raise ValueError("station_id and hour must not contain null values")
    bad_features = [
        col
        for col in FEATURE_COLUMNS
        if not (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]))
    ]
    if bad_features:
        raise ValueError(f"Features must be numeric or boolean: {', '.join(bad_features)}")
    splits = set(df["split"].dropna().unique())
    if not splits.issubset({"train", "validation", "test"}):
        raise ValueError("split contains unexpected partition names")
    if not {"train", "test"}.issubset(splits):
        raise ValueError("split must contain both train and test partitions")
    labeled = df[df["split"].isin(["train", "test"])]
    if labeled["target"].isna().any() or not labeled["target"].isin([0, 1, 2]).all():
        raise ValueError("train and test targets must contain only labels 0, 1 or 2")


def _check_result_name(cr) -> str:
    """Extract clean human-readable name from DeepChecks CheckResult."""
    header = getattr(cr, "header", None)
    if header:
        return str(header)
    check = getattr(cr, "check", None)
    if check is not None:
        name_attr = getattr(check, "name", None)
        if callable(name_attr):
            return str(name_attr())
        if isinstance(name_attr, str):
            return name_attr
        return check.__class__.__name__
    return str(cr)


def save_suite_summary(result, html_path: Path, json_path: Path) -> dict:
    """Save DeepChecks suite results as HTML and structured JSON summary."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    if html_path.exists():
        html_path.unlink()
    result.save_as_html(str(html_path))
    print(f"Report saved to {html_path}")

    raw_json = json.loads(result.to_json(with_display=False))
    passed_checks = [_check_result_name(c) for c in result.get_passed_checks()]
    not_passed_checks = [_check_result_name(c) for c in result.get_not_passed_checks()]
    not_run_checks = [_check_result_name(c) for c in result.get_not_ran_checks()]

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
        "passed": bool(result.passed(fail_if_check_not_run=True)),
        "total_checks": len(raw_json.get("results", [])),
        "passed_checks_count": len(passed_checks),
        "failed_checks_count": len(not_passed_checks),
        "passed_checks": passed_checks,
        "failed_checks": not_passed_checks,
        "not_run_checks": not_run_checks,
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

    return Dataset(
        df,
        label="target",
        features=FEATURE_COLUMNS,
        cat_features=CATEGORICAL_LIST,
        datetime_name="hour",
        index_name="record_id",
        label_type="multiclass",
    )


def run_data_integrity_check(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run data integrity suite and export HTML + JSON."""
    ds = _prepare_dataset(df)

    suite = data_integrity()
    for check in suite.checks.values():
        if isinstance(check, DataDuplicates):
            check.clean_conditions()
            check.add_condition_ratio_less_or_equal(0)
        elif isinstance(check, ConflictingLabels):
            check.clean_conditions()
            check.add_condition_ratio_of_conflicting_labels_less_or_equal(MAX_CONFLICT_RATIO)
        elif isinstance(check, FeatureFeatureCorrelation):
            check.clean_conditions()
            check.add_condition_max_number_of_pairs_above_threshold(threshold=0.9, n_pairs=0)
        elif isinstance(check, FeatureLabelCorrelation):
            check.clean_conditions()
            check.add_condition_feature_pps_less_than(threshold=0.9)
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

    for check in suite.checks.values():
        if isinstance(check, FeatureDrift):
            check.clean_conditions()
            check.add_condition_drift_score_less_than(
                max_allowed_categorical_score=0.20,
                max_allowed_numeric_score=0.20,
                allowed_num_features_exceeding_threshold=0,
            )
        elif isinstance(check, LabelDrift):
            check.clean_conditions()
            check.add_condition_drift_score_less_than(max_allowed_drift_score=0.15)
        elif isinstance(check, MultivariateDrift):
            check.clean_conditions()
            check.add_condition_overall_drift_value_less_than(max_drift_value=0.25)
        elif isinstance(check, IndexTrainTestLeakage):
            check.clean_conditions()
            check.add_condition_ratio_less_or_equal(0)
        elif isinstance(check, (DateTrainTestLeakageOverlap, DateTrainTestLeakageDuplicates)):
            check.clean_conditions()
            check.add_condition_leakage_ratio_less_or_equal(0)
        elif isinstance(check, TrainTestSamplesMix):
            check.clean_conditions()
            check.add_condition_duplicates_ratio_less_or_equal(MAX_SAMPLES_MIX_RATIO)

    result = suite.run(train_dataset=train_ds, test_dataset=test_ds)
    html_path = output_dir / "train_test_drift.html"
    json_path = output_dir / "train_test_drift.json"
    return save_suite_summary(result, html_path, json_path)


def print_summary(summaries: list[dict]) -> int:
    """Print the per-suite verdict and the conditions that did not pass.

    Returns the number of suites that did not pass, so `--strict` can turn it into an exit code.
    """
    print("\nResumen de suites:")
    failed_suites = 0
    for summary in summaries:
        verdict = "OK" if summary["passed"] else "NO PASA"
        print(
            f"  {summary['suite_name']}: {verdict} "
            f"({summary['passed_checks_count']}/{summary['total_checks']} checks)"
        )
        for name in summary["failed_checks"]:
            print(f"    - condición no cumplida: {name}")
        for name in summary["not_run_checks"]:
            print(f"    - check no ejecutado: {name}")
        for check in summary["check_summaries"]:
            value = check.get("value")
            if not isinstance(value, dict):
                continue
            if check.get("header") == "Conflicting Labels":
                ratio = value.get("percent_of_conflicting_samples")
                if isinstance(ratio, (int, float)) and ratio > 0:
                    print(f"    - alerta: etiquetas conflictivas {ratio:.2%} (límite 0.5%)")
            if check.get("header") == "Train Test Samples Mix":
                ratio = value.get("ratio")
                if isinstance(ratio, (int, float)) and ratio > 0.05:
                    print(f"    - alerta: mezcla train/test {ratio:.2%} (límite 10%)")
        failed_suites += 0 if summary["passed"] else 1
    if failed_suites:
        print(
            f"{failed_suites} suite(s) con condiciones no cumplidas; la interpretación está en "
            "reports/deepchecks/hallazgos.md."
        )
    return failed_suites


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when a suite has conditions that did not pass (default)",
    )
    mode.add_argument(
        "--report-only",
        action="store_true",
        help="Generate reports without failing the process for failed conditions",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)
    validate_input_schema(df)
    print(f"Loaded dataset: {df.shape}")
    if "split" in df.columns:
        print(df["split"].value_counts())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Running Data Integrity suite...")
    summaries = [run_data_integrity_check(df, args.output_dir)]

    print("Running Train/Test Drift suite (train Jan-Sep vs test Dec)...")
    summaries.append(run_train_test_drift_check(df, args.output_dir))

    print("All DeepChecks reports successfully generated.")
    failed_suites = print_summary(summaries)
    if not args.report_only and failed_suites:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
