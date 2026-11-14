import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

from evidently import Dataset, Report
from evidently.core.datasets import DataDefinition, MulticlassClassification
from evidently.core.metric_types import TestStatus
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import numpy as np
import pandas as pd
from pydantic import BaseModel

from tc5061_proyecto_equipo5.api.model import (
    SIGNATURE_DTYPES,
    LoadedModel,
    load_model,
)
from tc5061_proyecto_equipo5.config import (
    CLASSIFICATION_FEATURES_PATH,
    CURRENT_FEATURES_PATH,
    DRIFT_CURRENT,
    DRIFT_REFERENCE,
    FIGURES_DIR,
    REPORTS_DIR,
    SEED,
)
from tc5061_proyecto_equipo5.modeling.train_classification import apply_thresholds

TARGET_COLUMN: str = "target"
PREDICTION_COLUMN: str = "prediction"

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

DEFAULT_TOP_IMPORTANCE_FEATURES: tuple[str, ...] = (
    "longitude",
    "latitude",
    "net_flow_lag24",
)
DATA_DRIFT_SHARE_TRIGGER: float = 0.30


def extract_top_importance_features(
    model: Any | None = None,
    top_k: int = 3,
    fallback: Sequence[str] = DEFAULT_TOP_IMPORTANCE_FEATURES,
) -> tuple[str, ...]:
    """Extract top-K features by gain/importance from the model, or use audited fallback."""
    if model is None:
        return tuple(fallback[:top_k])

    raw = getattr(model, "model", model)
    impl = getattr(raw, "_model_impl", raw)
    skl = getattr(impl, "sklearn_model", impl)
    booster = getattr(skl, "booster_", None)

    feature_names: list[str] | None = None
    importances: np.ndarray | list[float] | None = None

    if (
        booster is not None
        and hasattr(booster, "feature_importance")
        and hasattr(booster, "feature_name")
    ):
        try:
            importances = booster.feature_importance(importance_type="gain")
            feature_names = booster.feature_name()
        except (AttributeError, ValueError, TypeError):
            importances = None
            feature_names = None

    if importances is None and hasattr(skl, "feature_importances_"):
        try:
            importances = skl.feature_importances_
            feature_names = getattr(skl, "feature_name_", None) or getattr(
                skl, "feature_names_in_", None
            )
        except (AttributeError, ValueError, TypeError):
            importances = None

    if (
        importances is not None
        and feature_names is not None
        and len(importances) == len(feature_names)
    ):
        ranking = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        top_features = tuple(feat for feat, _ in ranking[:top_k])
        if len(top_features) == top_k:
            return top_features

    return tuple(fallback[:top_k])


@dataclass
class ColumnMapping:
    target: str | None = TARGET_COLUMN
    prediction: str | None = PREDICTION_COLUMN
    numerical_features: list[str] | None = None
    categorical_features: list[str] | None = None


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
    reference_period: tuple[str, str] | list[str] | None = None
    current_period: tuple[str, str] | list[str] | None = None
    top_importance_features: tuple[str, ...] | list[str] | None = None


class DriftSummaryMetrics(BaseModel):
    dataset_drift: bool
    drifted_features_count: int
    drifted_features_share: float
    trigger_data_drift: bool = False
    trigger_label_drift: bool = False


class DriftSummaryReport(BaseModel):
    metadata: DriftSummaryMetadata
    summary: DriftSummaryMetrics
    features: dict[str, FeatureDriftResult]


def score_dataset_predictions(
    df: pd.DataFrame,
    loaded_model: LoadedModel | Any,
    prediction_column: str = PREDICTION_COLUMN,
) -> pd.DataFrame:
    """Score dataset records with the champion model and append operational class labels."""
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for prediction: {missing}")

    out_df = df.copy()
    if out_df.empty:
        out_df[prediction_column] = pd.Series(dtype="int8")
        return out_df

    features_df = out_df[FEATURE_COLUMNS].astype(SIGNATURE_DTYPES)
    proba = loaded_model.model.predict(features_df)
    labels = apply_thresholds(proba, loaded_model.classes, loaded_model.thresholds)
    out_df[prediction_column] = labels.astype("int8")
    return out_df


def load_champion_model(uri: str | None = None) -> LoadedModel:
    """Load the champion model using the API model loader."""
    return load_model(uri)


def build_data_definition(
    columns: list[str] | None = None,
    df: pd.DataFrame | None = None,
    column_mapping: ColumnMapping | None = None,
) -> DataDefinition:
    """Build an Evidently DataDefinition ensuring correct feature and target/prediction types."""
    effective_cm = column_mapping if column_mapping is not None else ColumnMapping()
    target_columns = (
        list(columns)
        if columns is not None
        else (list(df.columns) if df is not None else list(FEATURE_COLUMNS))
    )
    extra_cats = set(CATEGORICAL_FEATURES)
    if effective_cm.categorical_features:
        extra_cats.update(effective_cm.categorical_features)
    if effective_cm.target:
        extra_cats.add(effective_cm.target)
    if effective_cm.prediction:
        extra_cats.add(effective_cm.prediction)

    num_cols: list[str] = []
    cat_cols: list[str] = []
    for col in target_columns:
        if col in extra_cats or (
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

    classification = None
    target_in = bool(effective_cm.target and effective_cm.target in target_columns)
    pred_in = bool(effective_cm.prediction and effective_cm.prediction in target_columns)
    if target_in or pred_in:
        classification = [
            MulticlassClassification(
                target=effective_cm.target if target_in else (effective_cm.target or "target"),
                prediction_labels=effective_cm.prediction
                if pred_in
                else (effective_cm.prediction or "prediction"),
            )
        ]

    return DataDefinition(
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
        classification=classification,
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
    categorical_set = set(CATEGORICAL_FEATURES) | {TARGET_COLUMN, PREDICTION_COLUMN}
    per_column_method = {
        col: (cat_method if col in categorical_set else num_method) for col in target_columns
    }
    per_column_threshold = {
        col: (cat_threshold if col in categorical_set else num_threshold) for col in target_columns
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
    html_output_path: Path | str | None = None,
    column_mapping: ColumnMapping | None = None,
    model: LoadedModel | Any | None = None,
    model_uri: str | None = None,
    skip_prediction: bool = False,
    top_importance_features: Sequence[str] = DEFAULT_TOP_IMPORTANCE_FEATURES,
    data_drift_share_trigger: float = DATA_DRIFT_SHARE_TRIGGER,
) -> DriftSummaryReport:
    """Run drift detection between reference and current datasets and return summary."""
    effective_seed = seed if seed is not None else SEED
    effective_cm = column_mapping if column_mapping is not None else ColumnMapping()
    target_col = effective_cm.target or TARGET_COLUMN
    pred_col = effective_cm.prediction or PREDICTION_COLUMN

    if sample_size is not None:
        ref_df = (
            reference_data.sample(n=sample_size, random_state=effective_seed)
            if len(reference_data) > sample_size
            else reference_data.copy()
        )
        curr_df = (
            current_data.sample(n=sample_size, random_state=effective_seed)
            if len(current_data) > sample_size
            else current_data.copy()
        )
        effective_sample_size = sample_size
    else:
        ref_df = reference_data.copy()
        curr_df = current_data.copy()
        effective_sample_size = len(reference_data)

    effective_model = model
    if not skip_prediction and (pred_col not in ref_df.columns or pred_col not in curr_df.columns):
        if effective_model is None:
            try:
                effective_model = load_champion_model(model_uri)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load champion model from '{model_uri}': {exc}"
                ) from exc

        try:
            ref_df = score_dataset_predictions(ref_df, effective_model, prediction_column=pred_col)
            curr_df = score_dataset_predictions(
                curr_df, effective_model, prediction_column=pred_col
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to score predictions with champion model: {exc}") from exc

    if columns is not None:
        target_columns = list(columns)
    elif all(c in ref_df.columns and c in curr_df.columns for c in FEATURE_COLUMNS):
        target_columns = list(FEATURE_COLUMNS)
    else:
        target_columns = [
            c for c in FEATURE_COLUMNS if c in ref_df.columns and c in curr_df.columns
        ] or [c for c in ref_df.columns if c in curr_df.columns]

    if columns is None:
        if (
            target_col in ref_df.columns
            and target_col in curr_df.columns
            and target_col not in target_columns
        ):
            target_columns.append(target_col)
        if (
            not skip_prediction
            and pred_col in ref_df.columns
            and pred_col in curr_df.columns
            and pred_col not in target_columns
        ):
            target_columns.append(pred_col)

    data_def = build_data_definition(
        columns=target_columns,
        df=ref_df,
        column_mapping=effective_cm,
    )
    ref_evidently = ref_df[target_columns].copy()
    curr_evidently = curr_df[target_columns].copy()
    for col in target_columns:
        if str(ref_evidently[col].dtype).lower() in ("int8", "uint8") or (
            col in (target_col, pred_col, TARGET_COLUMN, PREDICTION_COLUMN)
            and pd.api.types.is_integer_dtype(ref_evidently[col].dtype)
        ):
            ref_evidently[col] = ref_evidently[col].astype("int32")
        if str(curr_evidently[col].dtype).lower() in ("int8", "uint8") or (
            col in (target_col, pred_col, TARGET_COLUMN, PREDICTION_COLUMN)
            and pd.api.types.is_integer_dtype(curr_evidently[col].dtype)
        ):
            curr_evidently[col] = curr_evidently[col].astype("int32")

    ds_ref = Dataset.from_pandas(ref_evidently, data_definition=data_def)
    ds_curr = Dataset.from_pandas(curr_evidently, data_definition=data_def)

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

    label_cols = {target_col, pred_col, TARGET_COLUMN, PREDICTION_COLUMN}
    input_feature_cols = [c for c in features if c not in label_cols]
    input_drifted_count = sum(1 for c in input_feature_cols if features[c].drift_detected)
    input_drifted_share = (
        (input_drifted_count / len(input_feature_cols)) if input_feature_cols else 0.0
    )

    resolved_top_importance = extract_top_importance_features(
        model=effective_model,
        top_k=len(top_importance_features) if top_importance_features else 3,
        fallback=top_importance_features,
    )
    top_importance_drifted = any(
        features[c].drift_detected for c in resolved_top_importance if c in features
    )
    trigger_data_drift = bool(
        input_drifted_share > data_drift_share_trigger or top_importance_drifted
    )

    trigger_label_drift = any(
        features[c].drift_detected
        for c in (target_col, pred_col, TARGET_COLUMN, PREDICTION_COLUMN)
        if c in features
    )

    ref_period: list[str] = list(DRIFT_REFERENCE)
    if "hour" in ref_df.columns and not ref_df["hour"].empty:
        ref_h = pd.to_datetime(ref_df["hour"])
        ref_period = [str(ref_h.min().date()), str(ref_h.max().date())]

    curr_period: list[str] = list(DRIFT_CURRENT)
    if "hour" in curr_df.columns and not curr_df["hour"].empty:
        curr_h = pd.to_datetime(curr_df["hour"])
        curr_period = [str(curr_h.min().date()), str(curr_h.max().date())]

    metadata = DriftSummaryMetadata(
        timestamp=datetime.now(UTC).isoformat(),
        sample_size=effective_sample_size,
        seed=effective_seed,
        reference_rows=len(reference_data),
        current_rows=len(current_data),
        reference_period=ref_period,
        current_period=curr_period,
        top_importance_features=list(resolved_top_importance),
    )

    summary = DriftSummaryMetrics(
        dataset_drift=dataset_drift,
        drifted_features_count=drifted_count,
        drifted_features_share=drifted_share,
        trigger_data_drift=trigger_data_drift,
        trigger_label_drift=trigger_label_drift,
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

    if html_output_path is not None:
        html_path = Path(html_output_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))

    return summary_report


def run_drift_pipeline(
    reference_path: Path | str | None = None,
    current_path: Path | str | None = None,
    dataset: pd.DataFrame | Path | str | None = None,
    sample_size: int | None = None,
    seed: int = SEED,
    output_dir: Path | str | None = None,
    figures_dir: Path | str | None = None,
    html_filename: str = "drift_2024_vs_2025.html",
    json_filename: str = "drift_summary.json",
    model_uri: str | None = None,
    skip_prediction: bool = False,
    top_k_plots: int = 3,
    top_importance_features: Sequence[str] = DEFAULT_TOP_IMPORTANCE_FEATURES,
    data_drift_share_trigger: float = DATA_DRIFT_SHARE_TRIGGER,
) -> tuple[DriftSummaryReport, list[Path]]:
    """Execute complete drift monitoring pipeline with reports and distribution figures."""
    from tc5061_proyecto_equipo5.monitoring.datasets import (
        DEFAULT_SAMPLE_SIZE,
        load_drift_datasets,
    )
    from tc5061_proyecto_equipo5.monitoring.plots import export_drift_distribution_plots

    effective_sample_size = sample_size if sample_size is not None else DEFAULT_SAMPLE_SIZE
    effective_output_dir = (
        Path(output_dir) if output_dir is not None else (REPORTS_DIR / "monitoring")
    )
    effective_figures_dir = (
        Path(figures_dir) if figures_dir is not None else (FIGURES_DIR / "entrega5")
    )
    effective_output_dir.mkdir(parents=True, exist_ok=True)
    effective_figures_dir.mkdir(parents=True, exist_ok=True)

    html_output_path = effective_output_dir / html_filename
    json_output_path = effective_output_dir / json_filename

    effective_ref_path = (
        reference_path if reference_path is not None else CLASSIFICATION_FEATURES_PATH
    )
    effective_curr_path = current_path if current_path is not None else CURRENT_FEATURES_PATH

    pair = load_drift_datasets(
        dataset=dataset,
        reference_path=effective_ref_path if dataset is None else None,
        current_path=effective_curr_path if dataset is None else None,
        sample_size=effective_sample_size,
        seed=seed,
    )

    ref_df = pair.reference
    curr_df = pair.current

    effective_model = None
    if not skip_prediction:
        try:
            effective_model = load_champion_model(model_uri)
            ref_df = score_dataset_predictions(ref_df, effective_model)
            curr_df = score_dataset_predictions(curr_df, effective_model)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to score predictions with champion model from '{model_uri}': {exc}"
            ) from exc

    summary_report = compute_drift_summary(
        reference_data=ref_df,
        current_data=curr_df,
        sample_size=None,
        seed=seed,
        output_path=json_output_path,
        html_output_path=html_output_path,
        model=effective_model,
        skip_prediction=skip_prediction,
        top_importance_features=top_importance_features,
        data_drift_share_trigger=data_drift_share_trigger,
    )

    figures: list[Path] = []
    if top_k_plots > 0:
        figures = export_drift_distribution_plots(
            reference_df=ref_df,
            current_df=curr_df,
            summary_report=summary_report,
            output_dir=effective_figures_dir,
            top_k=top_k_plots,
        )

    return summary_report, figures


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI entrypoint for drift monitoring and artifact generation."""
    from tc5061_proyecto_equipo5.monitoring.datasets import (
        DEFAULT_SAMPLE_SIZE,
        DEFAULT_SEED,
    )

    parser = argparse.ArgumentParser(
        prog="python -m tc5061_proyecto_equipo5.monitoring.drift",
        description="Run Evidently data, prediction, and concept drift monitoring pipeline.",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=CLASSIFICATION_FEATURES_PATH,
        help="Path to reference dataset (2024 train partition) or shared features parquet.",
    )
    parser.add_argument(
        "--current-path",
        type=Path,
        default=CURRENT_FEATURES_PATH,
        help="Path to current dataset (2025 current partition). Defaults to CURRENT_FEATURES_PATH.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Maximum stratified sample size per dataset partition.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for sampling reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "monitoring",
        help="Directory where HTML and JSON drift reports will be saved.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR / "entrega5",
        help="Directory where comparison distribution plots (PNG) will be saved.",
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default=None,
        help="MLflow model URI for champion model. If omitted, uses default champion.",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        default=False,
        help="Skip model prediction scoring and prediction drift evaluation.",
    )
    parser.add_argument(
        "--top-k-plots",
        type=int,
        default=3,
        help="Number of top drifted features to plot in addition to target and prediction.",
    )
    parser.add_argument(
        "--html-filename",
        type=str,
        default="drift_2024_vs_2025.html",
        help="Filename for the interactive HTML Evidently report.",
    )
    parser.add_argument(
        "--json-filename",
        type=str,
        default="drift_summary.json",
        help="Filename for the structured JSON summary report.",
    )

    args = parser.parse_args(argv)

    try:
        report, figures = run_drift_pipeline(
            reference_path=args.reference_path,
            current_path=args.current_path,
            sample_size=args.sample_size,
            seed=args.seed,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
            html_filename=args.html_filename,
            json_filename=args.json_filename,
            model_uri=args.model_uri,
            skip_prediction=args.skip_prediction,
            top_k_plots=args.top_k_plots,
        )
        logger.info(
            f"Drift monitoring pipeline succeeded: dataset_drift={report.summary.dataset_drift}, "
            f"trigger_data_drift={report.summary.trigger_data_drift}, "
            f"trigger_label_drift={report.summary.trigger_label_drift}, "
            f"figures={len(figures)}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Drift monitoring pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
