from datetime import UTC, datetime
from pathlib import Path

from evidently import Dataset, Report
from evidently.core.datasets import DataDefinition
from evidently.core.metric_types import TestStatus
from evidently.presets import DataDriftPreset, DataSummaryPreset
import numpy as np
import pandas as pd
from pydantic import BaseModel

from tc5061_proyecto_equipo5.config import SEED

FEATURE_COLUMNS: list[str] = [
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

CATEGORICAL_FEATURES: list[str] = ["day_of_week", "is_weekend", "is_holiday"]
NUMERICAL_FEATURES: list[str] = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]

DEFAULT_NUMERICAL_METHOD: str = "wasserstein"
DEFAULT_CATEGORICAL_METHOD: str = "jensenshannon"
DEFAULT_DRIFT_THRESHOLD: float = 0.1
DEFAULT_DRIFT_SHARE: float = 0.5


class FeatureDriftResult(BaseModel):
    drift_detected: bool
    metric_name: str
    score: float
    threshold: float


class DriftSummaryMetadata(BaseModel):
    timestamp: str
    sample_size: int
    seed: int
    reference_rows: int | None = None
    current_rows: int | None = None


class DriftSummaryMetrics(BaseModel):
    dataset_drift: bool
    drifted_features_count: int
    drifted_features_share: float


class DriftSummaryReport(BaseModel):
    metadata: DriftSummaryMetadata
    summary: DriftSummaryMetrics
    features: dict[str, FeatureDriftResult]


def build_data_definition(
    columns: list[str] | None = None,
    df: pd.DataFrame | None = None,
) -> DataDefinition:
    """Build an Evidently DataDefinition ensuring correct feature types."""
    target_columns = (
        list(columns)
        if columns is not None
        else (list(df.columns) if df is not None else list(FEATURE_COLUMNS))
    )
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for col in target_columns:
        if col in CATEGORICAL_FEATURES or (
            df is not None
            and col in df.columns
            and (
                df[col].dtype == bool
                or isinstance(df[col].dtype, (pd.CategoricalDtype, pd.BooleanDtype))
                or df[col].dtype == object
            )
        ):
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return DataDefinition(
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
    )


def build_drift_report(
    columns: list[str] | None = None,
    num_method: str = DEFAULT_NUMERICAL_METHOD,
    num_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    cat_method: str = DEFAULT_CATEGORICAL_METHOD,
    cat_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    drift_share: float = DEFAULT_DRIFT_SHARE,
) -> Report:
    """Configure an Evidently Report with DataDriftPreset and DataSummaryPreset."""
    target_columns = list(columns) if columns is not None else list(FEATURE_COLUMNS)
    per_column_method = {
        col: (cat_method if col in CATEGORICAL_FEATURES else num_method) for col in target_columns
    }
    per_column_threshold = {
        col: (cat_threshold if col in CATEGORICAL_FEATURES else num_threshold)
        for col in target_columns
    }

    drift_preset = DataDriftPreset(
        columns=target_columns,
        drift_share=drift_share,
        num_method=num_method,
        num_threshold=num_threshold,
        cat_method=cat_method,
        cat_threshold=cat_threshold,
        per_column_method=per_column_method,
        per_column_threshold=per_column_threshold,
    )
    summary_preset = DataSummaryPreset(columns=target_columns)
    return Report(metrics=[drift_preset, summary_preset], include_tests=True)


def compute_drift_summary(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    columns: list[str] | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
    num_method: str = "wasserstein",
    num_threshold: float = 0.1,
    cat_method: str = "jensenshannon",
    cat_threshold: float = 0.1,
    drift_share: float = 0.5,
    output_path: Path | str | None = None,
) -> DriftSummaryReport:
    """Run drift detection between reference and current datasets and return summary."""
    effective_seed = seed if seed is not None else SEED

    if sample_size is not None:
        ref_df = (
            reference_data.sample(n=sample_size, random_state=effective_seed)
            if len(reference_data) > sample_size
            else reference_data
        )
        curr_df = (
            current_data.sample(n=sample_size, random_state=effective_seed)
            if len(current_data) > sample_size
            else current_data
        )
        effective_sample_size = sample_size
    else:
        ref_df = reference_data
        curr_df = current_data
        effective_sample_size = len(reference_data)

    if columns is not None:
        target_columns = list(columns)
    elif all(c in ref_df.columns and c in curr_df.columns for c in FEATURE_COLUMNS):
        target_columns = list(FEATURE_COLUMNS)
    else:
        target_columns = [
            c for c in FEATURE_COLUMNS if c in ref_df.columns and c in curr_df.columns
        ] or [c for c in ref_df.columns if c in curr_df.columns]

    data_def = build_data_definition(columns=target_columns, df=ref_df)
    ds_ref = Dataset.from_pandas(ref_df[target_columns], data_definition=data_def)
    ds_curr = Dataset.from_pandas(curr_df[target_columns], data_definition=data_def)

    report = build_drift_report(
        columns=target_columns,
        num_method=num_method,
        num_threshold=num_threshold,
        cat_method=cat_method,
        cat_threshold=cat_threshold,
        drift_share=drift_share,
    )
    snapshot = report.run(current_data=ds_curr, reference_data=ds_ref)
    snapshot_dict = snapshot.dict()

    tests_by_col: dict[str, bool] = {}
    for test in snapshot_dict.get("tests", []):
        params = test.get("metric_config", {}).get("params", {})
        if params.get("type") == "evidently:metric_v2:ValueDrift":
            col = params.get("column")
            if col:
                tests_by_col[col] = test.get("status") == TestStatus.FAIL

    features: dict[str, FeatureDriftResult] = {}
    for metric in snapshot_dict.get("metrics", []):
        config = metric.get("config", {})
        if config.get("type") == "evidently:metric_v2:ValueDrift":
            col = config.get("column")
            if not col or col not in target_columns:
                continue
            val = metric.get("value")
            score = 0.0 if val is None or np.isnan(val) else float(val)
            threshold = float(config.get("threshold", num_threshold))
            metric_name = str(config.get("method", ""))
            drift_detected = tests_by_col.get(col, score > threshold)
            features[col] = FeatureDriftResult(
                drift_detected=drift_detected,
                metric_name=metric_name,
                score=score,
                threshold=threshold,
            )

    features = {col: features[col] for col in target_columns if col in features}
    drifted_count = sum(1 for feat in features.values() if feat.drift_detected)
    drifted_share = (drifted_count / len(features)) if features else 0.0
    dataset_drift = bool(drifted_share >= drift_share)

    metadata = DriftSummaryMetadata(
        timestamp=datetime.now(UTC).isoformat(),
        sample_size=effective_sample_size,
        seed=effective_seed,
        reference_rows=len(reference_data),
        current_rows=len(current_data),
    )

    summary = DriftSummaryMetrics(
        dataset_drift=dataset_drift,
        drifted_features_count=drifted_count,
        drifted_features_share=drifted_share,
    )

    summary_report = DriftSummaryReport(
        metadata=metadata,
        summary=summary,
        features=features,
    )

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary_report.model_dump_json(indent=2), encoding="utf-8")

    return summary_report
