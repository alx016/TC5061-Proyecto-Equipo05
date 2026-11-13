"""Tests for the A/B test between two P2 versions (Entrega 5, WP-3)."""

import copy

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo5.api.model import LoadedModel
from tc5061_proyecto_equipo5.modeling.train_classification import FEATURE_COLUMNS
from tc5061_proyecto_equipo5.monitoring.ab_test import (
    ALERTS_MAX_RATIO,
    MACRO_F1_MARGIN,
    Scored,
    bootstrap_metrics,
    decide,
    mcnemar,
    paired_bootstrap,
    score,
    version_metrics,
)

CLASSES = [0, 1, 2]


def _synthetic(n_days: int = 10, rows_per_day: int = 50, seed: int = 0):
    """Labels, probabilities and days for a small evaluation period."""
    rng = np.random.default_rng(seed)
    n_rows = n_days * rows_per_day
    days = pd.Series(np.repeat(pd.date_range("2025-10-01", periods=n_days).date, rows_per_day))
    y_true = rng.choice(CLASSES, size=n_rows, p=[0.8, 0.1, 0.1])
    proba = rng.dirichlet([4, 1, 1], size=n_rows)
    return y_true, proba, days


def _scored(proba: np.ndarray, labels: np.ndarray) -> Scored:
    return Scored(proba=proba, labels=labels, classes=CLASSES)


class _FixedModel:
    """Stands in for the MLflow model: always returns the same probabilities."""

    def __init__(self, proba: np.ndarray):
        self.proba = proba

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.proba[: len(features)]


def test_bootstrap_is_reproducible_with_the_seed():
    y_true, proba, days = _synthetic()
    champion = _scored(proba, proba.argmax(axis=1))
    challenger = _scored(proba[::-1].copy(), proba[::-1].argmax(axis=1))

    first = paired_bootstrap(y_true, champion, challenger, days, n_boot=30, seed=42)
    again = paired_bootstrap(y_true, champion, challenger, days, n_boot=30, seed=42)
    other = paired_bootstrap(y_true, champion, challenger, days, n_boot=30, seed=7)

    assert first == again
    assert first != other


def test_identical_versions_have_a_zero_interval():
    y_true, proba, days = _synthetic()
    version = _scored(proba, proba.argmax(axis=1))

    intervals = paired_bootstrap(y_true, version, version, days, n_boot=20)

    for interval in intervals.values():
        assert interval == {"low": 0.0, "high": 0.0}


def test_bootstrap_metrics_match_the_training_metrics():
    y_true, proba, days = _synthetic()
    version = _scored(proba, proba.argmax(axis=1))
    n_dates = int(days.nunique())

    full = version_metrics(y_true, version, n_dates)
    fast = bootstrap_metrics(y_true, version, n_dates)

    for name, value in fast.items():
        assert value == pytest.approx(full[name])


def test_each_version_uses_its_own_thresholds():
    # Shortage probability 0.3: above a 0.2 threshold, below a 0.4 threshold
    proba = np.tile([0.6, 0.3, 0.1], (4, 1))
    rows = pd.DataFrame({column: [0.0] * 4 for column in FEATURE_COLUMNS})
    low = LoadedModel(
        info={}, model=_FixedModel(proba), classes=CLASSES, thresholds={1: 0.2, 2: 0.95}
    )
    high = LoadedModel(
        info={}, model=_FixedModel(proba), classes=CLASSES, thresholds={1: 0.4, 2: 0.95}
    )

    assert (score(low, rows).labels == 1).all()
    assert (score(high, rows).labels == 0).all()


def test_mcnemar_counts_rows_where_only_one_version_is_right():
    y_true = np.array([1, 1, 2, 2, 0])
    empty = np.zeros((5, 3))
    champion = _scored(empty, np.array([1, 0, 2, 0, 0]))
    challenger = _scored(empty, np.array([0, 1, 2, 2, 1]))

    result = mcnemar(y_true, champion, challenger)

    # Only the risk rows count: the champion alone is right in row 0, the challenger in rows 1 and 3
    assert result["only_champion_correct"] == 1
    assert result["only_challenger_correct"] == 2


PASSING = {
    "champion": {"recall_shortage": 0.86, "alerts_per_day": 100.0},
    "challenger": {"recall_shortage": 0.86, "alerts_per_day": 110.0},
    "intervals": {
        "pr_auc_minority_mean": {"low": 0.01, "high": 0.03},
        "macro_f1": {"low": -0.005, "high": 0.01},
    },
}


def test_challenger_wins_when_every_check_passes():
    result = decide(PASSING["champion"], PASSING["challenger"], PASSING["intervals"])

    assert all(result["checks"].values())
    assert result["winner"] == "challenger"


@pytest.mark.parametrize(
    ("check", "change"),
    [
        (
            "challenger_recall_shortage",
            lambda case: case["challenger"].update(recall_shortage=0.84),
        ),
        (
            "pr_auc_improvement",
            lambda case: case["intervals"]["pr_auc_minority_mean"].update(low=0.0),
        ),
        (
            "macro_f1_non_inferior",
            lambda case: case["intervals"]["macro_f1"].update(low=-MACRO_F1_MARGIN - 0.001),
        ),
        (
            "alerts_non_inferior",
            lambda case: case["challenger"].update(alerts_per_day=100.0 * ALERTS_MAX_RATIO + 1),
        ),
    ],
)
def test_champion_stays_when_one_check_fails(check, change):
    case = copy.deepcopy(PASSING)
    change(case)

    result = decide(case["champion"], case["challenger"], case["intervals"])

    assert result["checks"][check] is False
    assert sum(not passed for passed in result["checks"].values()) == 1
    assert result["winner"] == "champion"
