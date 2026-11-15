import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo5.config import (
    CLASSIFICATION_FEATURES_PATH,
    CURRENT_FEATURES_PATH,
    DRIFT_CURRENT,
    DRIFT_REFERENCE,
    P2_FIT_AS_OF,
)
from tc5061_proyecto_equipo5.monitoring import (
    DATA_DRIFT_SHARE_TRIGGER,
    DEFAULT_TOP_IMPORTANCE_FEATURES,
    DRIFT_MONTHS,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    PREDICTION_COLUMN,
    REQUIRED_DRIFT_DATASET_COLUMNS,
    REQUIRED_METADATA_COLUMNS,
    TARGET_COLUMN,
    DriftSummaryReport,
    compute_drift_summary,
    export_drift_distribution_plots,
    extract_top_importance_features,
    filter_p2_reference,
    load_drift_datasets,
    run_drift_pipeline,
    score_dataset_predictions,
    stratified_monthly_sample,
    validate_drift_dataset_schema,
)
from tc5061_proyecto_equipo5.monitoring.drift import main


class MockPyFunc:
    def __init__(self, proba_fn):
        self.proba_fn = proba_fn

    def predict(self, df):
        return self.proba_fn(df)


class MockLoadedModel:
    def __init__(self, proba_fn, classes=None, thresholds=None):
        self.model = MockPyFunc(proba_fn)
        self.classes = classes if classes is not None else [0, 1, 2]
        self.thresholds = thresholds if thresholds is not None else {1: 0.2, 2: 0.95}


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
    report = compute_drift_summary(synthetic_reference_df, current_df, skip_prediction=True)

    assert report.summary.dataset_drift is False
    assert report.summary.drifted_features_count == 0
    assert report.summary.drifted_features_share == 0.0
    assert len(report.features) == len(FEATURE_COLUMNS)
    assert PREDICTION_COLUMN not in report.features

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

    report = compute_drift_summary(synthetic_reference_df, current_df, skip_prediction=True)

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

    report = compute_drift_summary(synthetic_reference_df, current_df, skip_prediction=True)

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
        skip_prediction=True,
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


def test_score_dataset_predictions_assigns_operational_classes(
    synthetic_reference_df: pd.DataFrame,
) -> None:
    sample_df = synthetic_reference_df.iloc[:3].copy()

    def _mock_proba(df: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                [0.90, 0.05, 0.05],
                [0.10, 0.80, 0.10],
                [0.01, 0.01, 0.98],
            ]
        )

    mock_model = MockLoadedModel(_mock_proba)
    scored = score_dataset_predictions(sample_df, mock_model)

    assert PREDICTION_COLUMN in scored.columns
    assert scored[PREDICTION_COLUMN].dtype == "int8"
    assert scored[PREDICTION_COLUMN].tolist() == [0, 1, 2]

    broken_df = sample_df.drop(columns=["capacity"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        score_dataset_predictions(broken_df, mock_model)


def test_compute_drift_summary_includes_target_and_prediction_drift(
    synthetic_reference_df: pd.DataFrame,
) -> None:
    ref_df = synthetic_reference_df.copy()
    ref_df[TARGET_COLUMN] = 0
    curr_df = synthetic_reference_df.copy()
    curr_df[TARGET_COLUMN] = 0

    def _mock_proba(df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        probs = np.zeros((n, 3))
        probs[:, 0] = 0.95
        probs[:, 1] = 0.03
        probs[:, 2] = 0.02
        return probs

    mock_model = MockLoadedModel(_mock_proba)
    report = compute_drift_summary(ref_df, curr_df, model=mock_model)

    assert TARGET_COLUMN in report.features
    assert PREDICTION_COLUMN in report.features
    assert report.features[TARGET_COLUMN].metric_name == "jensenshannon"
    assert report.features[PREDICTION_COLUMN].metric_name == "jensenshannon"
    assert len(report.features) == len(FEATURE_COLUMNS) + 2


def test_positive_prediction_drift_detection(
    synthetic_reference_df: pd.DataFrame,
) -> None:
    ref_df = synthetic_reference_df.copy()
    curr_df = synthetic_reference_df.copy()

    def _mock_proba(df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        probs = np.zeros((n, 3))
        is_curr = (df["capacity"] > 50.0).to_numpy()
        probs[~is_curr, 0] = 0.90
        probs[~is_curr, 1] = 0.05
        probs[~is_curr, 2] = 0.05
        probs[is_curr, 0] = 0.05
        probs[is_curr, 1] = 0.90
        probs[is_curr, 2] = 0.05
        return probs

    curr_df["capacity"] = 100.0
    mock_model = MockLoadedModel(_mock_proba)

    report = compute_drift_summary(ref_df, curr_df, model=mock_model)

    assert report.features[PREDICTION_COLUMN].drift_detected is True
    assert report.features[PREDICTION_COLUMN].score > report.features[PREDICTION_COLUMN].threshold
    assert report.features[PREDICTION_COLUMN].metric_name == "jensenshannon"


def test_positive_concept_drift_detection(
    synthetic_reference_df: pd.DataFrame,
) -> None:
    ref_df = synthetic_reference_df.copy()
    ref_df[TARGET_COLUMN] = 0
    curr_df = synthetic_reference_df.copy()
    curr_df[TARGET_COLUMN] = 2

    def _mock_proba(df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        probs = np.zeros((n, 3))
        probs[:, 0] = 0.95
        probs[:, 1] = 0.03
        probs[:, 2] = 0.02
        return probs

    mock_model = MockLoadedModel(_mock_proba)
    report = compute_drift_summary(ref_df, curr_df, model=mock_model)

    assert report.features[TARGET_COLUMN].drift_detected is True
    assert report.features[TARGET_COLUMN].score > report.features[TARGET_COLUMN].threshold
    assert report.features[TARGET_COLUMN].metric_name == "jensenshannon"


def test_skip_prediction_flag(
    synthetic_reference_df: pd.DataFrame,
) -> None:
    current_df = synthetic_reference_df.copy()
    report = compute_drift_summary(synthetic_reference_df, current_df, skip_prediction=True)

    assert PREDICTION_COLUMN not in report.features
    assert len(report.features) == len(FEATURE_COLUMNS)


def test_model_error_raises_exception_when_not_skipped(
    synthetic_reference_df: pd.DataFrame,
) -> None:
    current_df = synthetic_reference_df.copy()
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        compute_drift_summary(
            synthetic_reference_df,
            current_df,
            model_uri="invalid/path/nonexistent_model",
            skip_prediction=False,
        )


def test_drift_triggers_negative_control(
    synthetic_reference_df: pd.DataFrame,
) -> None:
    current_df = synthetic_reference_df.copy()
    report = compute_drift_summary(synthetic_reference_df, current_df, skip_prediction=True)

    assert report.summary.trigger_data_drift is False
    assert report.summary.trigger_label_drift is False


def test_trigger_data_drift_via_top_importance_feature(
    synthetic_reference_df: pd.DataFrame,
) -> None:
    current_df = synthetic_reference_df.copy()
    current_df["net_flow_lag24"] = (
        current_df["net_flow_lag24"] + 5.0 * synthetic_reference_df["net_flow_lag24"].std()
    )
    report = compute_drift_summary(synthetic_reference_df, current_df, skip_prediction=True)

    assert report.features["net_flow_lag24"].drift_detected is True
    assert report.summary.trigger_data_drift is True
    assert report.summary.trigger_label_drift is False


def test_trigger_data_drift_via_share_threshold(
    synthetic_reference_df: pd.DataFrame,
) -> None:
    current_df = synthetic_reference_df.copy()
    current_df["capacity"] = (
        current_df["capacity"] + 5.0 * synthetic_reference_df["capacity"].std()
    )

    report_below = compute_drift_summary(
        synthetic_reference_df,
        current_df,
        skip_prediction=True,
        top_importance_features=("latitude", "longitude"),
        data_drift_share_trigger=0.30,
    )
    assert report_below.features["capacity"].drift_detected is True
    assert report_below.summary.trigger_data_drift is False

    report_above = compute_drift_summary(
        synthetic_reference_df,
        current_df,
        skip_prediction=True,
        top_importance_features=("latitude", "longitude"),
        data_drift_share_trigger=0.04,
    )
    assert report_above.summary.trigger_data_drift is True


def test_trigger_label_drift_on_target_and_prediction(
    synthetic_reference_df: pd.DataFrame,
) -> None:
    ref_target_drift = synthetic_reference_df.copy()
    ref_target_drift[TARGET_COLUMN] = 0
    curr_target_drift = synthetic_reference_df.copy()
    curr_target_drift[TARGET_COLUMN] = 2

    report_target = compute_drift_summary(
        ref_target_drift,
        curr_target_drift,
        skip_prediction=True,
    )
    assert report_target.features[TARGET_COLUMN].drift_detected is True
    assert report_target.summary.trigger_label_drift is True
    assert report_target.summary.drifted_features_count == 0
    assert report_target.summary.drifted_features_share == 0.0

    ref_pred_drift = synthetic_reference_df.copy()
    curr_pred_drift = synthetic_reference_df.copy()

    def _mock_proba(df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        probs = np.zeros((n, 3))
        is_curr = (df["capacity"] > 50.0).to_numpy()
        probs[~is_curr, 0] = 0.90
        probs[~is_curr, 1] = 0.05
        probs[~is_curr, 2] = 0.05
        probs[is_curr, 0] = 0.05
        probs[is_curr, 1] = 0.90
        probs[is_curr, 2] = 0.05
        return probs

    curr_pred_drift["capacity"] = 100.0
    mock_model = MockLoadedModel(_mock_proba)

    report_pred = compute_drift_summary(
        ref_pred_drift,
        curr_pred_drift,
        model=mock_model,
    )
    assert report_pred.features[PREDICTION_COLUMN].drift_detected is True
    assert report_pred.summary.trigger_label_drift is True


def test_compute_drift_summary_html_export(
    synthetic_reference_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    current_df = synthetic_reference_df.copy()
    html_file = tmp_path / "reports" / "drift.html"

    report = compute_drift_summary(
        synthetic_reference_df,
        current_df,
        skip_prediction=True,
        html_output_path=html_file,
    )

    assert html_file.exists()
    assert html_file.stat().st_size > 1000
    html_content = html_file.read_text(encoding="utf-8")
    assert "<html" in html_content.lower()
    assert report.summary.drifted_features_count == 0


def test_export_drift_distribution_plots(
    synthetic_reference_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    ref_df = synthetic_reference_df.copy()
    ref_df[TARGET_COLUMN] = 0
    ref_df[PREDICTION_COLUMN] = 0

    curr_df = synthetic_reference_df.copy()
    curr_df[TARGET_COLUMN] = 1
    curr_df[PREDICTION_COLUMN] = 2
    curr_df["capacity"] = curr_df["capacity"] + 5.0 * ref_df["capacity"].std()

    report = compute_drift_summary(
        ref_df,
        curr_df,
        skip_prediction=True,
    )

    figs_dir = tmp_path / "figures"
    paths = export_drift_distribution_plots(
        reference_df=ref_df,
        current_df=curr_df,
        summary_report=report,
        output_dir=figs_dir,
        top_k=2,
    )

    assert len(paths) >= 3
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
        assert p.suffix == ".png"

    assert (figs_dir / f"drift_{TARGET_COLUMN}.png").exists()
    assert (figs_dir / f"drift_{PREDICTION_COLUMN}.png").exists()


def test_main_cli_end_to_end(
    synthetic_temporal_drift_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "classification_features.parquet"
    synthetic_temporal_drift_df.to_parquet(parquet_path, index=False)

    out_dir = tmp_path / "monitoring"
    figs_dir = tmp_path / "figures" / "entrega5"

    exit_code = main(
        [
            f"--reference-path={parquet_path}",
            f"--current-path={parquet_path}",
            f"--output-dir={out_dir}",
            f"--figures-dir={figs_dir}",
            "--skip-prediction",
            "--sample-size=90",
            "--seed=42",
            "--top-k-plots=2",
            "--html-filename=test_drift.html",
            "--json-filename=test_summary.json",
        ]
    )

    assert exit_code == 0

    json_file = out_dir / "test_summary.json"
    html_file = out_dir / "test_drift.html"

    assert json_file.exists()
    assert html_file.exists()
    assert html_file.stat().st_size > 1000

    payload = json.loads(json_file.read_text(encoding="utf-8"))
    parsed = DriftSummaryReport.model_validate(payload)
    assert parsed.metadata.sample_size == 90
    assert hasattr(parsed.summary, "trigger_data_drift")
    assert hasattr(parsed.summary, "trigger_label_drift")

    png_files = list(figs_dir.glob("drift_*.png"))
    assert len(png_files) >= 2
    for p in png_files:
        assert p.stat().st_size > 0


def test_run_drift_pipeline_programmatic(
    synthetic_temporal_drift_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    assert DATA_DRIFT_SHARE_TRIGGER == 0.30
    assert "longitude" in DEFAULT_TOP_IMPORTANCE_FEATURES
    assert "latitude" in DEFAULT_TOP_IMPORTANCE_FEATURES
    assert "net_flow_lag24" in DEFAULT_TOP_IMPORTANCE_FEATURES

    out_dir = tmp_path / "pipeline_reports"
    figs_dir = tmp_path / "pipeline_figures"

    report, figures = run_drift_pipeline(
        dataset=synthetic_temporal_drift_df,
        sample_size=90,
        seed=42,
        output_dir=out_dir,
        figures_dir=figs_dir,
        skip_prediction=True,
        top_k_plots=2,
    )

    assert isinstance(report, DriftSummaryReport)
    assert (out_dir / "drift_summary.json").exists()
    assert (out_dir / "drift_2024_vs_2025.html").exists()
    assert len(figures) >= 2
    for fig_path in figures:
        assert fig_path.exists()
        assert fig_path.stat().st_size > 0


def test_load_drift_datasets_with_separate_tables(
    synthetic_temporal_drift_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    ref_table = synthetic_temporal_drift_df[
        synthetic_temporal_drift_df["hour"].dt.year == 2024
    ].reset_index(drop=True)
    curr_table = synthetic_temporal_drift_df[
        synthetic_temporal_drift_df["hour"].dt.year == 2025
    ].reset_index(drop=True)

    ref_path = tmp_path / "reference_2024.parquet"
    curr_path = tmp_path / "current_2025.parquet"
    ref_table.to_parquet(ref_path, index=False)
    curr_table.to_parquet(curr_path, index=False)

    pair = load_drift_datasets(
        reference_path=ref_path, current_path=curr_path, sample_size=90, seed=42
    )
    assert len(pair.reference) == 90
    assert len(pair.current) == 90
    assert set(pair.reference["split"].unique()) == {"train"}
    assert set(pair.current["split"].unique()) == {"current"}
    assert set(pair.reference["hour"].dt.year.unique()) == {2024}
    assert set(pair.current["hour"].dt.year.unique()) == {2025}

    assert CURRENT_FEATURES_PATH.name == "classification_features_2025.parquet"
    out_dir = tmp_path / "cli_separate_reports"
    figs_dir = tmp_path / "cli_separate_figures"
    exit_code = main(
        [
            f"--reference-path={ref_path}",
            f"--current-path={curr_path}",
            f"--output-dir={out_dir}",
            f"--figures-dir={figs_dir}",
            "--skip-prediction",
            "--sample-size=60",
        ]
    )
    assert exit_code == 0
    assert (out_dir / "drift_summary.json").exists()


def test_evidently_handles_nullable_int8_target(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n = 120
    data = {c: rng.normal(10.0, 2.0, n) for c in NUMERICAL_FEATURES}
    data["day_of_week"] = rng.choice(range(7), n).astype("int8")
    data["is_weekend"] = rng.choice([0, 1], n).astype("int8")
    data["is_holiday"] = rng.choice([0, 1], n).astype("int8")
    data["station_id"] = np.arange(1, n + 1, dtype="int32")
    data["hour"] = pd.date_range("2024-03-01", periods=n, freq="h")
    data["split"] = "train"
    data["target"] = pd.Series(rng.choice([0, 1, 2], size=n), dtype="Int8")
    ref_df = pd.DataFrame(data)[REQUIRED_DRIFT_DATASET_COLUMNS]

    curr_data = dict(data)
    curr_data["hour"] = pd.date_range("2025-03-01", periods=n, freq="h")
    curr_data["split"] = "current"
    curr_data["target"] = pd.Series(rng.choice([0, 1, 2], size=n), dtype="Int8")
    curr_df = pd.DataFrame(curr_data)[REQUIRED_DRIFT_DATASET_COLUMNS]

    json_path = tmp_path / "drift_summary.json"
    html_path = tmp_path / "drift_report.html"

    report = compute_drift_summary(
        ref_df,
        curr_df,
        output_path=json_path,
        html_output_path=html_path,
        skip_prediction=True,
    )
    assert report is not None
    assert json_path.exists()
    assert html_path.exists()
    assert html_path.stat().st_size > 1000

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    assert "reference_period" in metadata
    assert "current_period" in metadata
    assert metadata["reference_period"] == list(DRIFT_REFERENCE)
    assert metadata["current_period"] == list(DRIFT_CURRENT)


def test_reference_mask_parity_with_p2_training_protocol() -> None:
    hours = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-09-30 21:00",
            "2024-09-30 22:00",
            "2024-09-30 23:00",
        ]
    )
    n = len(hours)
    data = {c: [10.0] * n for c in NUMERICAL_FEATURES}
    data["day_of_week"] = [0] * n
    data["is_weekend"] = [False] * n
    data["is_holiday"] = [False] * n
    data["station_id"] = [1] * n
    data["hour"] = hours
    data["target"] = pd.Series([0, 1, 2, 0], dtype="Int8")
    data["split"] = "train"
    df = pd.DataFrame(data)[REQUIRED_DRIFT_DATASET_COLUMNS]

    filtered = filter_p2_reference(df, fit_as_of=P2_FIT_AS_OF)
    assert len(filtered) == 2
    assert filtered["hour"].tolist() == [
        pd.Timestamp("2024-01-01 00:00"),
        pd.Timestamp("2024-09-30 21:00"),
    ]

    if CLASSIFICATION_FEATURES_PATH.exists():
        from tc5061_proyecto_equipo5.modeling.train_classification import (
            load_rows,
            split_masks,
        )

        raw_df = pd.read_parquet(CLASSIFICATION_FEATURES_PATH)
        filtered_ref = filter_p2_reference(raw_df)
        assert len(filtered_ref) == 2_344_900

        loaded = load_rows(CLASSIFICATION_FEATURES_PATH)
        masks = split_masks(loaded)
        expected = loaded.loc[masks["train"]]
        assert len(filtered_ref) == len(expected) == 2_344_900


def test_dynamic_top_importance_features_and_ranking_change() -> None:
    top_default = extract_top_importance_features(None, top_k=3)
    assert top_default == ("longitude", "latitude", "net_flow_lag24")

    class MockBooster:
        def feature_name(self):
            return ["capacity", "latitude", "longitude", "hour_sin"]

        def feature_importance(self, importance_type="gain"):
            return np.array([5000.0, 100.0, 20.0, 8000.0])

    class MockSklearn:
        def __init__(self):
            self.booster_ = MockBooster()

    class MockModelImpl:
        def __init__(self):
            self.sklearn_model = MockSklearn()

    class MockLoadedModel:
        def __init__(self):
            self._model_impl = MockModelImpl()

    mock_model = MockLoadedModel()
    custom_top = extract_top_importance_features(mock_model, top_k=3)
    assert custom_top == ("hour_sin", "capacity", "latitude")

    ref_df = pd.DataFrame(
        {
            "capacity": [10.0, 10.0] * 50,
            "latitude": [20.0, 20.0] * 50,
            "longitude": [30.0, 30.0] * 50,
            "hour_sin": [0.0, 0.0] * 50,
            "day_of_week": [1] * 100,
            "is_weekend": [False] * 100,
            "is_holiday": [False] * 100,
            "target": [0] * 100,
            "prediction": [0] * 100,
        }
    )
    for col in NUMERICAL_FEATURES:
        if col not in ref_df:
            ref_df[col] = [5.0] * 100

    curr_df = ref_df.copy()
    curr_df["hour_sin"] = [1.0, 1.0] * 50

    report = compute_drift_summary(
        ref_df,
        curr_df,
        model=mock_model,
        skip_prediction=True,
        data_drift_share_trigger=0.99,
    )
    assert report.summary.trigger_data_drift is True
    assert "hour_sin" in report.metadata.top_importance_features
