import json
from pathlib import Path
import subprocess
import textwrap

import numpy as np
import pandas as pd
import pytest

PROJ_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJ_ROOT / "scripts" / "data_checks.py"


@pytest.fixture
def synthetic_classification_parquet(tmp_path: Path) -> Path:
    """Generate a valid synthetic dataset that passes every condition of both suites.

    Six whole weeks keep hours and weekdays balanced between the chronological partitions
    (four weeks of train, two of test), and one holiday in each keeps is_holiday variable.
    """
    n_samples = 24 * 7 * 6
    n_train = 24 * 7 * 4
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
            "is_holiday": np.isin(dates.dayofyear, [3, 31]).astype(int),
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
            "split": ["train"] * n_train + ["test"] * (n_samples - n_train),
        }
    )

    parquet_path = tmp_path / "synthetic_classification_features.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", compression="zstd", index=False)
    return parquet_path


def run_data_checks_script(
    data_path: Path, output_dir: Path, *, report_only: bool = False
) -> subprocess.CompletedProcess:
    """Execute scripts/data_checks.py in isolated environment via subprocess."""
    cmd = [
        "uv",
        "run",
        str(SCRIPT_PATH),
        "--data-path",
        str(data_path),
        "--output-dir",
        str(output_dir),
    ]
    if report_only:
        cmd.append("--report-only")
    return subprocess.run(
        cmd,
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


# Checks from check_policy.md that a clean run must evaluate and pass, per report file
EXPECTED_PASSED_CHECKS = {
    "data_integrity": (
        "Data Integrity Suite",
        [
            "Data Duplicates",
            "PercentOfNulls",
            "Single Value in Column",
            "Mixed Data Types",
            "Special Characters",
            "Conflicting Labels",
        ],
    ),
    "train_test_drift": (
        "train_test_drift",
        [
            "Feature Drift",
            "Label Drift",
            "Multivariate Drift",
            "Train Test Samples Mix",
            "Date Train-Test Leakage (overlap)",
            "Index Train-Test Leakage",
        ],
    ),
}


@pytest.mark.slow
def test_data_checks_execution_on_synthetic_data(
    synthetic_classification_parquet: Path, tmp_path: Path
):
    """A clean dataset must pass every condition of both suites and exit with status 0."""
    output_dir = tmp_path / "deepchecks_reports"

    result = run_data_checks_script(synthetic_classification_parquet, output_dir)

    assert result.returncode == 0, (
        f"Execution failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    for report, (suite_name, expected_checks) in EXPECTED_PASSED_CHECKS.items():
        assert (output_dir / f"{report}.html").exists(), f"{report}.html was not generated"
        with open(output_dir / f"{report}.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["suite_name"] == suite_name
        assert data["passed"] is True
        assert data["failed_checks"] == []
        assert data["warned_checks"] == []
        assert data["not_run_checks"] == []
        assert all(
            isinstance(c, str) and not c.startswith("<bound method") for c in data["passed_checks"]
        )
        assert set(expected_checks) <= set(data["passed_checks"])

        assert len(data["check_summaries"]) == data["total_checks"]
        with_conditions = [cs for cs in data["check_summaries"] if cs["conditions"]]
        assert data["passed_checks_count"] == len(with_conditions)
        for cs in with_conditions:
            assert all(c["status"] == "PASS" for c in cs["conditions"]), cs["header"]


@pytest.mark.slow
def test_data_checks_detects_injected_defects(
    synthetic_classification_parquet: Path, tmp_path: Path
):
    """Verify that data_integrity detects deliberately injected duplicates and excess nulls."""
    df = pd.read_parquet(synthetic_classification_parquet)

    df_defective = df.copy()

    # Inject excess nulls (> 1%) in arrivals_lag1
    df_defective.loc[:30, "arrivals_lag1"] = np.nan

    # Inject rows that repeat the station_id and hour key with different feature values,
    # so only a key-based duplicate check can catch them
    dupes = df_defective.iloc[50:70].copy()
    dupes["arrivals_lag2"] += 1
    df_defective = pd.concat([df_defective, dupes], ignore_index=True)

    defective_parquet = tmp_path / "defective_features.parquet"
    df_defective.to_parquet(defective_parquet, engine="pyarrow", compression="zstd", index=False)

    output_dir = tmp_path / "defective_reports"
    result = run_data_checks_script(defective_parquet, output_dir, report_only=True)
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
    # DeepChecks reports this built-in condition as WARN, but it must still fail the suite
    assert any(c["status"] in ("FAIL", "WARN") for c in duplicate_check["conditions"])
    assert "Data Duplicates" in integrity_data["failed_checks"]
    assert "Data Duplicates" not in integrity_data["warned_checks"]

    # Verify that PercentOfNulls failed condition
    null_check = next(
        (cs for cs in integrity_data["check_summaries"] if cs["header"] == "PercentOfNulls"),
        None,
    )
    assert null_check is not None
    assert any(c["status"] == "FAIL" for c in null_check["conditions"])

    strict_result = run_data_checks_script(
        defective_parquet, tmp_path / "defective_reports_strict"
    )
    assert strict_result.returncode == 1, (
        f"Expected exit code 1 by default:\nSTDOUT:\n{strict_result.stdout}\n"
        f"STDERR:\n{strict_result.stderr}"
    )
    assert "NO PASA" in strict_result.stdout
    assert "condición no cumplida" in strict_result.stdout


@pytest.mark.slow
def test_data_checks_fails_without_split_partitions(
    synthetic_classification_parquet: Path, tmp_path: Path
):
    """Verify that script raises an error when split column lacks required partitions."""
    df = pd.read_parquet(synthetic_classification_parquet)
    df_no_split = df.drop(columns=["split"])
    no_split_parquet = tmp_path / "no_split_features.parquet"
    df_no_split.to_parquet(no_split_parquet, engine="pyarrow", compression="zstd", index=False)

    output_dir = tmp_path / "no_split_reports"
    result = run_data_checks_script(no_split_parquet, output_dir)
    assert result.returncode != 0
    assert "Missing required columns: split" in result.stderr


@pytest.mark.parametrize("missing", ["target", "hour", "station_id", "capacity"])
def test_data_checks_rejects_missing_required_columns(
    synthetic_classification_parquet: Path, tmp_path: Path, missing: str
):
    df = pd.read_parquet(synthetic_classification_parquet).drop(columns=[missing])
    path = tmp_path / "missing.parquet"
    df.to_parquet(path, index=False)
    result = run_data_checks_script(path, tmp_path / "reports")
    assert result.returncode != 0
    assert f"Missing required columns: {missing}" in result.stderr


@pytest.mark.parametrize(
    "defect", ["invalid_target", "missing_test", "invalid_hour", "invalid_split"]
)
def test_data_checks_rejects_invalid_contract(
    synthetic_classification_parquet: Path, tmp_path: Path, defect: str
):
    df = pd.read_parquet(synthetic_classification_parquet)
    if defect == "invalid_target":
        df.loc[0, "target"] = 9
        expected = "targets must contain only labels"
    elif defect == "missing_test":
        df["split"] = "train"
        expected = "split must contain both train and test"
    elif defect == "invalid_hour":
        df["hour"] = df["hour"].astype(str)
        expected = "hour must be a datetime column"
    else:
        df.loc[0, "split"] = "unexpected"
        expected = "split contains unexpected partition names"
    path = tmp_path / "invalid.parquet"
    df.to_parquet(path, index=False)
    result = run_data_checks_script(path, tmp_path / "reports")
    assert result.returncode != 0
    assert expected in result.stderr


def run_in_data_checks_env(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    """Run `body` in the script's isolated environment with the script loaded as `module`."""
    metadata = SCRIPT_PATH.read_text().split("# ///", 2)[0:2]
    script = tmp_path / "data_checks_probe.py"
    script.write_text(
        "# ///".join(metadata)
        + "# ///\n"
        + textwrap.dedent(
            f"""
            import importlib.util

            spec = importlib.util.spec_from_file_location('data_checks', {str(SCRIPT_PATH)!r})
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            """
        )
        + textwrap.dedent(body)
    )
    return subprocess.run(
        ["uv", "run", str(script)], cwd=tmp_path, capture_output=True, text=True, check=False
    )


def test_unrun_deepchecks_check_fails_the_gate(tmp_path: Path):
    """A DeepChecks exception must fail the suite instead of looking successful."""
    result = run_in_data_checks_env(
        tmp_path,
        """
        from pathlib import Path
        from deepchecks.core.check_result import CheckFailure
        from deepchecks.core.suite import SuiteResult
        from deepchecks.tabular.checks import DataDuplicates

        result = SuiteResult('broken', [CheckFailure(DataDuplicates(), ValueError('broken'))])
        summary = module.save_suite_summary(result, Path('broken.html'), Path('broken.json'))
        assert summary['passed'] is False
        assert summary['not_run_checks'] == ['Data Duplicates']
        assert module.print_summary([summary]) == 1
        """,
    )
    assert result.returncode == 0, result.stderr


def test_tiered_ratio_condition_passes_warns_and_fails(tmp_path: Path):
    """Ratios up to the original threshold pass, up to the operational limit warn, then fail."""
    result = run_in_data_checks_env(
        tmp_path,
        """
        from deepchecks.core import ConditionCategory

        def category(ratio):
            value = {'ratio': ratio}
            return module._tiered_ratio_condition(value, 'ratio', 0.05, 0.10).category

        assert category(0.05) == ConditionCategory.PASS
        assert category(0.0675) == ConditionCategory.WARN
        assert category(0.10) == ConditionCategory.WARN
        assert category(0.1001) == ConditionCategory.FAIL
        """,
    )
    assert result.returncode == 0, result.stderr
