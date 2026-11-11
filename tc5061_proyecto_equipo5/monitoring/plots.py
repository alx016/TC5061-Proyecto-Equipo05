from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tc5061_proyecto_equipo5.monitoring.drift import (
    CATEGORICAL_FEATURES,
    PREDICTION_COLUMN,
    TARGET_COLUMN,
)

if TYPE_CHECKING:
    from tc5061_proyecto_equipo5.monitoring.drift import DriftSummaryReport

CLASS_NAMES: dict[int, str] = {
    0: "Balanceado",
    1: "Riesgo_Desabasto",
    2: "Riesgo_Saturacion",
}


def export_drift_distribution_plots(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    summary_report: "DriftSummaryReport",
    output_dir: Path | str,
    top_k: int = 3,
    extra_columns: Sequence[str] = (TARGET_COLUMN, PREDICTION_COLUMN),
) -> list[Path]:
    """Export comparative distribution plots (PNG) for top drifted features and extra columns."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_set = set(extra_columns)
    candidate_features = [
        col
        for col in summary_report.features
        if col not in extra_set and col in reference_df.columns and col in current_df.columns
    ]
    sorted_features = sorted(
        candidate_features,
        key=lambda c: summary_report.features[c].score,
        reverse=True,
    )
    selected_columns: list[str] = sorted_features[: max(0, top_k)]

    for col in extra_columns:
        if (
            col in reference_df.columns
            and col in current_df.columns
            and col not in selected_columns
        ):
            selected_columns.append(col)

    created_paths: list[Path] = []

    for col in selected_columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            is_categorical = (
                col in extra_set
                or col in CATEGORICAL_FEATURES
                or reference_df[col].dtype == object
                or reference_df[col].dtype == bool
                or isinstance(reference_df[col].dtype, (pd.CategoricalDtype, pd.BooleanDtype))
            )

            if is_categorical:
                if col in (TARGET_COLUMN, PREDICTION_COLUMN):
                    ref_series = reference_df[col].map(lambda v: CLASS_NAMES.get(v, str(v)))
                    curr_series = current_df[col].map(lambda v: CLASS_NAMES.get(v, str(v)))
                    categories = [
                        name
                        for name in ("Balanceado", "Riesgo_Desabasto", "Riesgo_Saturacion")
                        if name in ref_series.values or name in curr_series.values
                    ]
                    observed_other = sorted(
                        set(ref_series.unique()) | set(curr_series.unique()) - set(categories)
                    )
                    categories.extend(observed_other)
                else:
                    ref_series = reference_df[col].astype(str)
                    curr_series = current_df[col].astype(str)
                    try:
                        categories = sorted(
                            set(reference_df[col].dropna().unique())
                            | set(current_df[col].dropna().unique())
                        )
                        categories = [str(c) for c in categories]
                    except TypeError:
                        categories = sorted(set(ref_series.unique()) | set(curr_series.unique()))

                ref_counts = ref_series.value_counts(normalize=True).to_dict()
                curr_counts = curr_series.value_counts(normalize=True).to_dict()

                ref_props = [ref_counts.get(cat, 0.0) for cat in categories]
                curr_props = [curr_counts.get(cat, 0.0) for cat in categories]

                x = np.arange(len(categories))
                width = 0.35
                ax.bar(
                    x - width / 2,
                    ref_props,
                    width,
                    label="Referencia (2024)",
                    color="#1f77b4",
                    alpha=0.85,
                )
                ax.bar(
                    x + width / 2,
                    curr_props,
                    width,
                    label="Actual (2025)",
                    color="#ff7f0e",
                    alpha=0.85,
                )
                ax.set_xticks(x)
                rotate = any(len(str(cat)) > 8 for cat in categories)
                ax.set_xticklabels(categories, rotation=15 if rotate else 0)
                ax.set_ylabel("Proporción")
            else:
                ref_vals = reference_df[col].dropna().to_numpy()
                curr_vals = current_df[col].dropna().to_numpy()

                if len(ref_vals) == 0 or len(curr_vals) == 0:
                    min_val, max_val = 0.0, 1.0
                else:
                    min_val = float(min(ref_vals.min(), curr_vals.min()))
                    max_val = float(max(ref_vals.max(), curr_vals.max()))

                if min_val == max_val:
                    bins = np.linspace(min_val - 1.0, max_val + 1.0, 30)
                else:
                    bins = np.linspace(min_val, max_val, 30)

                ax.hist(
                    ref_vals,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    label="Referencia (2024)",
                    color="#1f77b4",
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.hist(
                    curr_vals,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    label="Actual (2025)",
                    color="#ff7f0e",
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.set_ylabel("Densidad")

            metric_info = ""
            if col in summary_report.features:
                feat = summary_report.features[col]
                metric_info = f" (Score: {feat.score:.4f}, Drift: {feat.drift_detected})"

            ax.set_title(f"Distribución: {col}{metric_info}")
            ax.set_xlabel(col)
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            fig.tight_layout()

            out_path = out_dir / f"drift_{col}.png"
            fig.savefig(out_path, dpi=150)
            created_paths.append(out_path)
        finally:
            plt.close(fig)

    return created_paths
