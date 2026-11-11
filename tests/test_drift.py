import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo5.monitoring import (
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    DriftSummaryReport,
    compute_drift_summary,
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
