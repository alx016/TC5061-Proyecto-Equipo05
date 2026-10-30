import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

PROJ_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJ_ROOT / "scripts" / "data_checks.py"


@pytest.fixture
def synthetic_classification_parquet(tmp_path: Path) -> Path:
    """Generate a valid synthetic classification features dataset for testing."""
    n_samples = 200
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="h")

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "hour": dates,
            "station_id": rng.integers(1, 10, size=n_samples),
            "capacity": rng.integers(10, 30, size=n_samples).astype(float),
            "latitude": 20.67 + rng.normal(0, 0.01, size=n_samples),
            "longitude": -103.35 + rng.normal(0, 0.01, size=n_samples),
            "hour_sin": np.sin(2 * np.pi * dates.hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * dates.hour / 24.0),
            "day_of_week": dates.dayofweek.astype(int),
            "is_weekend": (dates.dayofweek >= 5).astype(int),
            "is_holiday": np.zeros(n_samples, dtype=int),
            "neighbor_net_flow": rng.normal(0, 1.5, size=n_samples),
            "net_flow_lag1": rng.normal(0, 1.0, size=n_samples),
            "net_flow_lag2": rng.normal(0, 1.0, size=n_samples),
            "net_flow_lag24": rng.normal(0, 1.0, size=n_samples),
            "arrivals_lag1": rng.poisson(2, size=n_samples).astype(float),
            "arrivals_lag2": rng.poisson(2, size=n_samples).astype(float),
            "arrivals_lag3": rng.poisson(2, size=n_samples).astype(float),
            "departures_lag1": rng.poisson(2, size=n_samples).astype(float),
            "departures_lag2": rng.poisson(2, size=n_samples).astype(float),
            "departures_lag3": rng.poisson(2, size=n_samples).astype(float),
            "target": rng.choice([0, 1, 2], size=n_samples, p=[0.8, 0.1, 0.1]),
            "split": ["train"] * 160 + ["test"] * 40,
        }
    )

    parquet_path = tmp_path / "synthetic_classification_features.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", compression="zstd", index=False)
    return parquet_path


@pytest.mark.slow
def test_data_checks_execution_on_synthetic_data(
    synthetic_classification_parquet: Path, tmp_path: Path
):
    """Execute scripts/data_checks.py via subprocess on synthetic data and verify JSON/HTML."""
    output_dir = tmp_path / "deepchecks_reports"

    cmd = [
        "uv",
        "run",
        str(SCRIPT_PATH),
        "--data-path",
        str(synthetic_classification_parquet),
        "--output-dir",
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"Execution failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    integrity_html = output_dir / "data_integrity.html"
    integrity_json = output_dir / "data_integrity.json"
    drift_html = output_dir / "train_test_drift.html"
    drift_json = output_dir / "train_test_drift.json"

    assert integrity_html.exists(), "data_integrity.html was not generated"
    assert integrity_json.exists(), "data_integrity.json was not generated"
    assert drift_html.exists(), "train_test_drift.html was not generated"
    assert drift_json.exists(), "train_test_drift.json was not generated"

    # Verify JSON content and structure for Data Integrity
    with open(integrity_json, "r", encoding="utf-8") as f:
        integrity_data = json.load(f)

    assert integrity_data["suite_name"] == "Data Integrity Suite"
    assert integrity_data["total_checks"] > 0
    assert "passed_checks" in integrity_data
    assert "failed_checks" in integrity_data
    assert "check_summaries" in integrity_data
    assert len(integrity_data["check_summaries"]) == integrity_data["total_checks"]

    # Verify JSON content and structure for Train/Test Drift
    with open(drift_json, "r", encoding="utf-8") as f:
        drift_data = json.load(f)

    assert drift_data["suite_name"] == "train_test_drift"
    assert drift_data["total_checks"] > 0
    assert "passed_checks" in drift_data
    assert "failed_checks" in drift_data
    assert "check_summaries" in drift_data
    assert len(drift_data["check_summaries"]) == drift_data["total_checks"]

    drift_headers = [cs["header"] for cs in drift_data["check_summaries"]]
    assert "Feature Drift" in drift_headers
    assert "Label Drift" in drift_headers
    assert "Multivariate Drift" in drift_headers


@pytest.mark.slow
def test_data_checks_detects_injected_defects(
    synthetic_classification_parquet: Path, tmp_path: Path
):
    """Verify that data_integrity detects deliberately injected duplicates and excess nulls."""
    df = pd.read_parquet(synthetic_classification_parquet)

    df_defective = df.copy()

    # Inject excess nulls (> 1%) in arrivals_lag1
    df_defective.loc[:30, "arrivals_lag1"] = np.nan

    # Inject duplicate rows
    dupes = df_defective.iloc[50:70].copy()
    df_defective = pd.concat([df_defective, dupes], ignore_index=True)

    defective_parquet = tmp_path / "defective_features.parquet"
    df_defective.to_parquet(defective_parquet, engine="pyarrow", compression="zstd", index=False)

    output_dir = tmp_path / "defective_reports"
    cmd = [
        "uv",
        "run",
        str(SCRIPT_PATH),
        "--data-path",
        str(defective_parquet),
        "--output-dir",
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Execution failed:\n{result.stderr}"

    with open(output_dir / "data_integrity.json", "r", encoding="utf-8") as f:
        integrity_data = json.load(f)

    assert integrity_data["passed"] is False

    # Verify that Data Duplicates detected the injected duplicates
    duplicate_check = next(
        (cs for cs in integrity_data["check_summaries"] if cs["header"] == "Data Duplicates"),
        None,
    )
    assert duplicate_check is not None
    assert duplicate_check["value"] > 0
    assert any(c["status"] in ("FAIL", "WARN") for c in duplicate_check["conditions"])

    # Verify that PercentOfNulls failed condition
    null_check = next(
        (cs for cs in integrity_data["check_summaries"] if cs["header"] == "PercentOfNulls"),
        None,
    )
    assert null_check is not None
    assert any(c["status"] == "FAIL" for c in null_check["conditions"])
