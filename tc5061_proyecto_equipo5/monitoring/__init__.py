"""Monitoring tools, loaded on demand so A/B testing does not require Evidently."""

from importlib import import_module

_DATASET_NAMES = {"CURRENT_YEAR", "DEFAULT_SAMPLE_SIZE", "DEFAULT_SEED", "DRIFT_MONTHS",
                  "REFERENCE_YEAR", "REQUIRED_DRIFT_DATASET_COLUMNS", "REQUIRED_METADATA_COLUMNS",
                  "DriftDatasetPair", "filter_p2_reference", "load_drift_datasets",
                  "stratified_monthly_sample", "validate_drift_dataset_schema"}

__all__ = [
    "CATEGORICAL_FEATURES",
    "CURRENT_YEAR",
    "DATA_DRIFT_SHARE_TRIGGER",
    "DEFAULT_CATEGORICAL_METHOD",
    "DEFAULT_DRIFT_SHARE",
    "DEFAULT_DRIFT_THRESHOLD",
    "DEFAULT_NUMERICAL_METHOD",
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_SEED",
    "DEFAULT_TOP_IMPORTANCE_FEATURES",
    "DRIFT_MONTHS",
    "FEATURE_COLUMNS",
    "NUMERICAL_FEATURES",
    "PREDICTION_COLUMN",
    "REFERENCE_YEAR",
    "REQUIRED_DRIFT_DATASET_COLUMNS",
    "REQUIRED_METADATA_COLUMNS",
    "TARGET_COLUMN",
    "ColumnMapping",
    "DriftDatasetPair",
    "DriftSummaryMetadata",
    "DriftSummaryMetrics",
    "DriftSummaryReport",
    "FeatureDriftResult",
    "build_data_definition",
    "build_drift_report",
    "compute_drift_summary",
    "export_drift_distribution_plots",
    "extract_top_importance_features",
    "filter_p2_reference",
    "load_champion_model",
    "load_drift_datasets",
    "run_drift_pipeline",
    "score_dataset_predictions",
    "stratified_monthly_sample",
    "validate_drift_dataset_schema",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = "datasets" if name in _DATASET_NAMES else "plots" if name == "export_drift_distribution_plots" else "drift"
    return getattr(import_module(f"{__name__}.{module}"), name)
