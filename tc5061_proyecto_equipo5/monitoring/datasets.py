from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from tc5061_proyecto_equipo5.config import (
    CLASSIFICATION_FEATURES_PATH,
    SEED,
    TRAIN_MONTHS,
)
from tc5061_proyecto_equipo5.monitoring.drift import FEATURE_COLUMNS

DEFAULT_SAMPLE_SIZE: int = 100_000
DEFAULT_SEED: int = SEED
REFERENCE_YEAR: int = 2024
CURRENT_YEAR: int = 2025
DRIFT_MONTHS: tuple[int, ...] = TRAIN_MONTHS

REQUIRED_METADATA_COLUMNS: list[str] = ["station_id", "hour", "target", "split"]
REQUIRED_DRIFT_DATASET_COLUMNS: list[str] = [
    "station_id",
    "hour",
    *FEATURE_COLUMNS,
    "target",
    "split",
]


class DriftDatasetPair(NamedTuple):
    reference: pd.DataFrame
    current: pd.DataFrame


def validate_drift_dataset_schema(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_DRIFT_DATASET_COLUMNS,
) -> None:
    """Validate that the dataset contains all required feature and metadata columns."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Dataset is missing required drift columns: {missing_str}")


def _canonical_sort_columns(df: pd.DataFrame) -> list[str]:
    if "station_id" in df.columns:
        return ["hour", "station_id"]
    return ["hour"]


def _allocate_monthly_quotas(
    month_counts: dict[pd.Period, int],
    sample_size: int,
) -> dict[pd.Period, int]:
    total_rows = sum(month_counts.values())
    months = sorted(month_counts.keys())
    raw_quotas = {m: sample_size * (month_counts[m] / total_rows) for m in months}
    quotas = {m: int(np.floor(raw_quotas[m])) for m in months}
    remainder = sample_size - sum(quotas.values())

    ranked_months = sorted(
        months,
        key=lambda m: (-(raw_quotas[m] - quotas[m]), m),
    )
    for month in ranked_months:
        if remainder <= 0:
            break
        if quotas[month] < month_counts[month]:
            quotas[month] += 1
            remainder -= 1

    return quotas


def stratified_monthly_sample(
    df: pd.DataFrame,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Draw a deterministic monthly-stratified sample without replacement."""
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if "hour" not in df.columns:
        raise ValueError("Dataset is missing required 'hour' column for monthly stratification")

    work = df.copy()
    work["hour"] = pd.to_datetime(work["hour"])
    sort_cols = _canonical_sort_columns(work)
    work = work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if len(work) <= sample_size:
        return work

    month_periods = work["hour"].dt.to_period("M")
    months = sorted(month_periods.unique())
    month_counts = {month: int((month_periods == month).sum()) for month in months}
    quotas = _allocate_monthly_quotas(month_counts, sample_size)

    rng = np.random.default_rng(seed)
    sampled_parts: list[pd.DataFrame] = []
    for month in months:
        quota = quotas[month]
        if quota <= 0:
            continue
        month_df = work.loc[month_periods == month]
        chosen_indices = np.sort(rng.choice(len(month_df), size=quota, replace=False))
        sampled_parts.append(month_df.iloc[chosen_indices])

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _coerce_dataframe(source: pd.DataFrame | Path | str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_parquet(Path(source))


def _filter_partition(
    df: pd.DataFrame,
    *,
    expected_split: str,
    expected_year: int,
    months: Sequence[int] = DRIFT_MONTHS,
) -> pd.DataFrame:
    work = df.copy()
    work["hour"] = pd.to_datetime(work["hour"])
    mask = (
        (work["split"] == expected_split)
        & (work["hour"].dt.year == expected_year)
        & (work["hour"].dt.month.isin(months))
    )
    return work.loc[mask].reset_index(drop=True)


def load_drift_datasets(
    dataset: pd.DataFrame | Path | str | None = None,
    current_dataset: pd.DataFrame | Path | str | None = None,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    reference_data: pd.DataFrame | Path | str | None = None,
    current_data: pd.DataFrame | Path | str | None = None,
    reference_path: Path | str | None = None,
    current_path: Path | str | None = None,
) -> DriftDatasetPair:
    """Load, validate, filter, and monthly-stratify reference (2024) and current (2025) datasets."""
    ref_source = (
        reference_data
        if reference_data is not None
        else (reference_path if reference_path is not None else dataset)
    )
    curr_source = (
        current_data
        if current_data is not None
        else (current_path if current_path is not None else current_dataset)
    )

    if ref_source is None and curr_source is None:
        ref_source = CLASSIFICATION_FEATURES_PATH

    if curr_source is None:
        if ref_source is None:
            ref_source = CLASSIFICATION_FEATURES_PATH
        combined_df = _coerce_dataframe(ref_source)
        validate_drift_dataset_schema(combined_df)
        ref_raw = _filter_partition(
            combined_df,
            expected_split="train",
            expected_year=REFERENCE_YEAR,
        )
        curr_raw = _filter_partition(
            combined_df,
            expected_split="current",
            expected_year=CURRENT_YEAR,
        )
    else:
        effective_ref_source = (
            ref_source if ref_source is not None else CLASSIFICATION_FEATURES_PATH
        )
        ref_input = _coerce_dataframe(effective_ref_source)
        curr_input = _coerce_dataframe(curr_source)
        validate_drift_dataset_schema(ref_input)
        validate_drift_dataset_schema(curr_input)
        ref_raw = _filter_partition(
            ref_input,
            expected_split="train",
            expected_year=REFERENCE_YEAR,
        )
        curr_raw = _filter_partition(
            curr_input,
            expected_split="current",
            expected_year=CURRENT_YEAR,
        )

    if len(ref_raw) == 0:
        raise ValueError(
            f"No reference records found for year {REFERENCE_YEAR} with split='train' "
            f"and months {DRIFT_MONTHS}."
        )
    if len(curr_raw) == 0:
        raise ValueError(
            f"No current records found for year {CURRENT_YEAR} with split='current' "
            f"and months {DRIFT_MONTHS}. Provide a dataset containing 2025 current data or pass current_path."
        )

    reference_sampled = stratified_monthly_sample(ref_raw, sample_size=sample_size, seed=seed)
    current_sampled = stratified_monthly_sample(curr_raw, sample_size=sample_size, seed=seed)
    return DriftDatasetPair(reference=reference_sampled, current=current_sampled)
