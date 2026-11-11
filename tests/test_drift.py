import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo5.monitoring import (
    DRIFT_MONTHS,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    REQUIRED_DRIFT_DATASET_COLUMNS,
    REQUIRED_METADATA_COLUMNS,
    DriftSummaryReport,
    compute_drift_summary,
    load_drift_datasets,
    stratified_monthly_sample,
    validate_drift_dataset_schema,
)


@pytest.fixture
def synthetic_reference_df() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    n = 150
    data = {c: rng.normal(10.0, 2.0, n) for c in NUMERICAL_FEATURES}
    data["day_of_week"] = rng.choice(range(7), n)
    data["is_weekend"] = rng.choice([0, 1], n, p=[0.7, 0.3])
    data["is_holiday"] = rng.choice([0, 1], n, p=[0.95, 0.05])
    return pd.DataFrame(data)[FEATURE_COLUMNS]


@pytest.fixture
def synthetic_temporal_drift_df() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    rows_per_month = 40
    records: list[pd.DataFrame] = []

    def _make_block(year: int, month: int, split_name: str, count: int) -> pd.DataFrame:
        hours = pd.date_range(
            f"{year}-{month:02d}-01 00:00:00",
            periods=count,
            freq="h",
        )
        block: dict[str, object] = {
            "station_id": np.arange(1, count + 1, dtype="int32"),
            "hour": hours,
        }
        for col in NUMERICAL_FEATURES:
            block[col] = rng.normal(10.0, 2.0, count)
        block["day_of_week"] = hours.dayofweek.astype("int8")
        block["is_weekend"] = (hours.dayofweek >= 5).astype("int8")
        block["is_holiday"] = np.zeros(count, dtype="int8")
        block["target"] = rng.choice([0, 1, 2], size=count)
        block["split"] = split_name
        return pd.DataFrame(block)[REQUIRED_DRIFT_DATASET_COLUMNS]

    for month in DRIFT_MONTHS:
        records.append(_make_block(2024, month, "train", rows_per_month))
        records.append(_make_block(2025, month, "current", rows_per_month))

    records.append(_make_block(2024, 10, "validation", 25))
    records.append(_make_block(2024, 12, "test", 25))
    records.append(_make_block(2025, 10, "current", 25))
    records.append(_make_block(2025, 3, "validation", 15))

    return pd.concat(records, ignore_index=True)


def test_negative_control_identical_data(synthetic_reference_df: pd.DataFrame) -> None:
    current_df = synthetic_reference_df.copy()
    report = compute_drift_summary(synthetic_reference_df, current_df)

    assert report.summary.dataset_drift is False
    assert report.summary.drifted_features_count == 0
    assert report.summary.drifted_features_share == 0.0
    assert len(report.features) == len(FEATURE_COLUMNS)

    for col, result in report.features.items():
        assert result.drift_detected is False, f"Feature {col} unexpectedly flagged as drifted"
        assert result.score <= result.threshold


def test_positive_control_continuous_feature_drift(
    synthetic_reference_df: pd.DataFrame,
) -> None:
    current_df = synthetic_reference_df.copy()
    current_df["capacity"] = (
        current_df["capacity"] + 5.0 * synthetic_reference_df["capacity"].std()
    )

    report = compute_drift_summary(synthetic_reference_df, current_df)

    assert report.features["capacity"].drift_detected is True
    assert report.features["capacity"].score > report.features["capacity"].threshold
    assert report.summary.drifted_features_count == 1

    for col, result in report.features.items():
        if col != "capacity":
            assert result.drift_detected is False, f"Feature {col} unexpectedly flagged"


def test_positive_control_categorical_feature_drift(
    synthetic_reference_df: pd.DataFrame,
) -> None:
    current_df = synthetic_reference_df.copy()
    current_df["is_weekend"] = 1

    report = compute_drift_summary(synthetic_reference_df, current_df)

    assert report.features["is_weekend"].drift_detected is True
    assert report.features["is_weekend"].score > report.features["is_weekend"].threshold
    assert report.features["is_weekend"].metric_name == "jensenshannon"


def test_artifact_export_and_deserialization(
    synthetic_reference_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    current_df = synthetic_reference_df.copy()
    current_df["capacity"] = (
        current_df["capacity"] + 5.0 * synthetic_reference_df["capacity"].std()
    )
    artifact_path = tmp_path / "nested" / "reports" / "drift_summary.json"

    report = compute_drift_summary(
        synthetic_reference_df,
        current_df,
        output_path=artifact_path,
    )

    assert artifact_path.exists()

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert "timestamp" in payload["metadata"]
    assert "sample_size" in payload["metadata"]
    assert "seed" in payload["metadata"]

    assert "dataset_drift" in payload["summary"]
    assert "drifted_features_count" in payload["summary"]
    assert "drifted_features_share" in payload["summary"]

    for col in FEATURE_COLUMNS:
        feature_entry = payload["features"][col]
        assert "drift_detected" in feature_entry
        assert "metric_name" in feature_entry
        assert "score" in feature_entry
        assert "threshold" in feature_entry

    parsed = DriftSummaryReport.model_validate(payload)
    assert parsed.summary.dataset_drift == report.summary.dataset_drift
    assert parsed.summary.drifted_features_count == report.summary.drifted_features_count
    assert parsed.metadata.seed == report.metadata.seed


def test_load_drift_datasets_filters_reference_and_current_partitions(
    synthetic_temporal_drift_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    pair = load_drift_datasets(synthetic_temporal_drift_df, sample_size=100_000, seed=42)
    ref_df, curr_df = pair

    assert len(ref_df) == 9 * 40
    assert len(curr_df) == 9 * 40
    assert set(ref_df["split"].unique()) == {"train"}
    assert set(curr_df["split"].unique()) == {"current"}
    assert set(ref_df["hour"].dt.year.unique()) == {2024}
    assert set(curr_df["hour"].dt.year.unique()) == {2025}
    assert set(ref_df["hour"].dt.month.unique()) == set(DRIFT_MONTHS)
    assert set(curr_df["hour"].dt.month.unique()) == set(DRIFT_MONTHS)

    parquet_path = tmp_path / "combined_features.parquet"
    synthetic_temporal_drift_df.to_parquet(parquet_path, index=False)
    from_disk = load_drift_datasets(parquet_path, sample_size=90, seed=42)
    assert len(from_disk.reference) == 90
    assert len(from_disk.current) == 90


def test_stratified_monthly_sampling_preserves_uniform_months(
    synthetic_temporal_drift_df: pd.DataFrame,
) -> None:
    ref_df, curr_df = load_drift_datasets(
        synthetic_temporal_drift_df,
        sample_size=90,
        seed=42,
    )

    assert len(ref_df) == 90
    assert len(curr_df) == 90

    ref_month_counts = ref_df["hour"].dt.month.value_counts().sort_index().to_dict()
    curr_month_counts = curr_df["hour"].dt.month.value_counts().sort_index().to_dict()

    expected_counts = {month: 10 for month in DRIFT_MONTHS}
    assert ref_month_counts == expected_counts
    assert curr_month_counts == expected_counts


def test_load_drift_datasets_returns_all_rows_when_within_sample_size(
    synthetic_temporal_drift_df: pd.DataFrame,
) -> None:
    exact_pair = load_drift_datasets(
        synthetic_temporal_drift_df,
        sample_size=360,
        seed=42,
    )
    assert len(exact_pair.reference) == 360
    assert len(exact_pair.current) == 360

    larger_pair = load_drift_datasets(
        synthetic_temporal_drift_df,
        sample_size=100_000,
        seed=42,
    )
    assert len(larger_pair.reference) == 360
    assert len(larger_pair.current) == 360


def test_stratified_sampling_is_deterministic_with_fixed_seed(
    synthetic_temporal_drift_df: pd.DataFrame,
) -> None:
    first_ref, first_curr = load_drift_datasets(
        synthetic_temporal_drift_df,
        sample_size=90,
        seed=42,
    )
    second_ref, second_curr = load_drift_datasets(
        synthetic_temporal_drift_df,
        sample_size=90,
        seed=42,
    )

    shuffled_input = synthetic_temporal_drift_df.sample(frac=1.0, random_state=777).reset_index(
        drop=True
    )
    shuffled_ref, shuffled_curr = load_drift_datasets(
        shuffled_input,
        sample_size=90,
        seed=42,
    )

    pd.testing.assert_frame_equal(first_ref, second_ref)
    pd.testing.assert_frame_equal(first_curr, second_curr)
    pd.testing.assert_frame_equal(first_ref, shuffled_ref)
    pd.testing.assert_frame_equal(first_curr, shuffled_curr)

    different_seed_ref, _ = load_drift_datasets(
        synthetic_temporal_drift_df,
        sample_size=90,
        seed=99,
    )
    assert not first_ref["station_id"].equals(different_seed_ref["station_id"])


@pytest.mark.parametrize(
    "missing_column",
    [
        "capacity",
        "neighbor_net_flow",
        "departures_lag3",
        *REQUIRED_METADATA_COLUMNS,
    ],
)
def test_validate_drift_dataset_schema_rejects_missing_columns(
    synthetic_temporal_drift_df: pd.DataFrame,
    missing_column: str,
) -> None:
    broken_df = synthetic_temporal_drift_df.drop(columns=[missing_column])

    with pytest.raises(ValueError, match=missing_column):
        validate_drift_dataset_schema(broken_df)

    with pytest.raises(ValueError, match=missing_column):
        load_drift_datasets(broken_df, sample_size=90, seed=42)


def test_stratified_monthly_sample_rejects_non_positive_sample_size(
    synthetic_temporal_drift_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="sample_size must be positive"):
        stratified_monthly_sample(synthetic_temporal_drift_df, sample_size=0, seed=42)
