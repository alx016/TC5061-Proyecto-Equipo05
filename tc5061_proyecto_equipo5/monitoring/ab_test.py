"""A/B test between two versions of the P2 model on the same evaluation period.

Each version predicts with its own thresholds. The difference challenger - champion gets a 95%
interval from a bootstrap that resamples whole days, and the decision follows section 4.2 of
reports/entrega5_monitoreo.md, which was fixed before seeing any result.
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score

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
BOOTSTRAP_METRICS = ("pr_auc_minority_mean", "macro_f1", "recall_shortage", "alerts_per_day")


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


def version_metrics(y_true: np.ndarray, scored: Scored, n_dates: int) -> dict[str, float]:
    """Return the P2 metrics of one version plus its alerts per day."""
    result = metrics(y_true, scored.labels, scored.proba, scored.classes, prefix="")
    n_alerts = int(np.count_nonzero(scored.labels != 0))
    result["alerts_per_day"] = alerts_per_day(n_alerts, n_dates)
    return result


def bootstrap_metrics(y_true: np.ndarray, scored: Scored, n_dates: int) -> dict[str, float]:
    """The four metrics of the decision rule, computed with the same functions as metrics().

    It skips the metrics the rule does not use, which makes each bootstrap replicate faster.
    """
    positions = {int(cls): index for index, cls in enumerate(scored.classes)}
    minority = [
        average_precision_score(y_true == cls, scored.proba[:, positions[cls]])
        for cls in (1, 2)
        if (y_true == cls).any()
    ]
    return {
        "pr_auc_minority_mean": float(np.mean(minority)),
        "macro_f1": f1_score(y_true, scored.labels, average="macro", zero_division=0),
        "recall_shortage": recall_score(y_true == 1, scored.labels == 1, zero_division=0),
        "alerts_per_day": alerts_per_day(int(np.count_nonzero(scored.labels != 0)), n_dates),
    }


def _subset(scored: Scored, index: np.ndarray) -> Scored:
    """Keep the given rows of one version's scores."""
    return Scored(proba=scored.proba[index], labels=scored.labels[index], classes=scored.classes)


def day_groups(days: pd.Series) -> list[np.ndarray]:
    """Return the row positions of each day, in date order."""
    codes, _ = pd.factorize(days, sort=True)
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.cumsum(np.bincount(codes))[:-1])


def paired_bootstrap(
    y_true: np.ndarray,
    champion: Scored,
    challenger: Scored,
    days: pd.Series,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """95% percentile interval of challenger - champion, resampling whole days.

    Both versions are scored on the same resampled rows in every replicate, so the comparison is
    paired. Rows of the same day are correlated, so days are resampled, not rows. All the draws
    are made first, so the result depends only on the seed.
    """
    groups = day_groups(days)
    n_dates = len(groups)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_dates, size=(n_boot, n_dates))
    differences = np.empty((n_boot, len(BOOTSTRAP_METRICS)))
    for replicate, draw in enumerate(draws):
        index = np.concatenate([groups[day] for day in draw])
        a = bootstrap_metrics(y_true[index], _subset(champion, index), n_dates)
        b = bootstrap_metrics(y_true[index], _subset(challenger, index), n_dates)
        differences[replicate] = [b[name] - a[name] for name in BOOTSTRAP_METRICS]
        if (replicate + 1) % 100 == 0:
            logger.info(f"Bootstrap: {replicate + 1}/{n_boot} replicates")

    intervals = {}
    for column, name in enumerate(BOOTSTRAP_METRICS):
        low, high = np.percentile(differences[:, column], [2.5, 97.5])
        intervals[name] = {"low": float(low), "high": float(high)}
    return intervals