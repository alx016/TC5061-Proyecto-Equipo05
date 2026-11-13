"""A/B test between two versions of the P2 model on the same evaluation period.

Each version predicts with its own thresholds. The difference challenger - champion gets a 95%
interval from a bootstrap that resamples whole days, and the decision follows section 4.2 of
reports/entrega5_monitoreo.md, which was fixed before seeing any result.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tc5061_proyecto_equipo5.api.model import SIGNATURE_DTYPES, LoadedModel
from tc5061_proyecto_equipo5.config import CURRENT_FEATURES_PATH
from tc5061_proyecto_equipo5.modeling.train_classification import (
    FEATURE_COLUMNS,
    alerts_per_day,
    apply_thresholds,
    load_rows,
    metrics,
)

EVALUATION_SPLIT = "evaluation"


@dataclass(frozen=True)
class Scored:
    """Probabilities and labels of one version, with the class order of its model."""

    proba: np.ndarray
    labels: np.ndarray
    classes: list[int]


def load_evaluation(
    path: Path = CURRENT_FEATURES_PATH, split: str = EVALUATION_SPLIT
) -> pd.DataFrame:
    """Read the feature table and keep the rows of the evaluation split."""
    frame = load_rows(path)
    rows = frame[frame["split"].eq(split).fillna(False)].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No rows with split == {split!r} in {path}")
    # Same day definition as build_data, so alerts_per_day matches the training runs
    rows["day"] = rows["hour"].dt.date
    return rows


def score(loaded: LoadedModel, rows: pd.DataFrame) -> Scored:
    """Predict with the model and turn its probabilities into labels with its own thresholds."""
    features = rows[FEATURE_COLUMNS].astype(SIGNATURE_DTYPES)
    proba = np.asarray(loaded.model.predict(features))
    labels = apply_thresholds(proba, loaded.classes, loaded.thresholds)
    return Scored(proba=proba, labels=labels, classes=loaded.classes)


def version_metrics(y_true: np.ndarray, scored: Scored, days: pd.Series) -> dict[str, float]:
    """Return the P2 metrics of one version plus its alerts per day."""
    result = metrics(y_true, scored.labels, scored.proba, scored.classes, prefix="")
    n_alerts = int(np.count_nonzero(scored.labels != 0))
    result["alerts_per_day"] = alerts_per_day(n_alerts, int(days.nunique()))
    return result