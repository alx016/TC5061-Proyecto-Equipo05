"""Unit tests for the P2 probability thresholds, metrics and selection rule."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo5.modeling import train_classification


def test_apply_thresholds_maps_classes_by_value_not_by_position():
    # The estimator may return classes in any order; the mapping must use the values.
    classes = np.array([2, 0, 1])
    proba = np.array(
        [
            [0.1, 0.2, 0.7],  # shortage above 0.5
            [0.8, 0.1, 0.1],  # saturation above 0.5
            [0.6, 0.2, 0.6],  # both; the higher one wins (tie -> shortage)
            [0.1, 0.8, 0.1],  # none -> balanced
        ]
    )

    prediction = train_classification.apply_thresholds(proba, classes, thresholds={1: 0.5, 2: 0.5})

    assert prediction.tolist() == [1, 2, 1, 0]


def test_calibrate_thresholds_respects_the_precision_floor():
    y_true = np.array([1] * 20 + [2] * 20 + [0] * 60)
    proba = np.zeros((100, 3))
    proba[:20, 1] = 0.9
    proba[20:40, 2] = 0.9
    proba[40:, 0] = 0.9

    thresholds = train_classification.calibrate_thresholds(
        y_true, proba, classes=np.array([0, 1, 2])
    )

    assert 0.05 <= thresholds[1] <= 0.95
    assert 0.05 <= thresholds[2] <= 0.95


def test_metrics_use_the_class_order_of_the_estimator():
    classes = np.array([2, 0, 1])
    y_true = np.array([1, 1, 2, 2, 0, 0])
    y_pred = np.array([1, 0, 2, 0, 0, 0])
    proba = np.zeros((6, 3))
    proba[:, 0] = [0.1, 0.1, 0.8, 0.7, 0.1, 0.2]  # class 2
    proba[:, 1] = [0.1, 0.2, 0.1, 0.1, 0.8, 0.7]  # class 0
    proba[:, 2] = [0.8, 0.7, 0.1, 0.2, 0.1, 0.1]  # class 1

    result = train_classification.metrics(y_true, y_pred, proba, classes, "val_")

    assert result["val_recall_shortage"] == pytest.approx(0.5)
    assert result["val_recall_saturation"] == pytest.approx(0.5)
    assert "val_pr_auc_shortage" in result
    assert "val_pr_auc_macro" in result


def test_choose_run_applies_the_recall_and_precision_guardrails():
    runs = pd.DataFrame(
        [
            {
                "run_id": "best-f1",
                "tags.question": "p2",
                "tags.stage_of_work": "candidate",
                "tags.git_dirty": "false",
                "tags.git_commit": "abc123",
                "metrics.val_macro_f1": 0.95,
                "metrics.val_recall_shortage": 0.40,
                "metrics.val_precision_shortage": 0.90,
                "metrics.val_precision_saturation": 0.90,
            },
            {
                "run_id": "good-enough",
                "tags.question": "p2",
                "tags.stage_of_work": "candidate",
                "tags.git_dirty": "false",
                "tags.git_commit": "abc123",
                "metrics.val_macro_f1": 0.80,
                "metrics.val_recall_shortage": 0.90,
                "metrics.val_precision_shortage": 0.60,
                "metrics.val_precision_saturation": 0.60,
            },
            {
                "run_id": "other-question",
                "tags.question": "p1",
                "tags.stage_of_work": "candidate",
                "tags.git_dirty": "false",
                "tags.git_commit": "abc123",
                "metrics.val_macro_f1": 0.99,
                "metrics.val_recall_shortage": 1.0,
                "metrics.val_precision_shortage": 1.0,
                "metrics.val_precision_saturation": 1.0,
            },
        ]
    )

    chosen = train_classification.choose_run(runs)

    assert chosen["run_id"] == "good-enough"


def test_choose_run_falls_back_to_the_best_macro_f1():
    runs = pd.DataFrame(
        [
            {
                "run_id": "best-f1",
                "tags.question": "p2",
                "tags.stage_of_work": "baseline",
                "tags.git_dirty": "false",
                "tags.git_commit": "abc123",
                "metrics.val_macro_f1": 0.60,
                "metrics.val_recall_shortage": 0.10,
                "metrics.val_precision_shortage": 0.10,
                "metrics.val_precision_saturation": 0.10,
            },
            {
                "run_id": "second",
                "tags.question": "p2",
                "tags.stage_of_work": "candidate",
                "tags.git_dirty": "false",
                "tags.git_commit": "abc123",
                "metrics.val_macro_f1": 0.50,
                "metrics.val_recall_shortage": 0.10,
                "metrics.val_precision_shortage": 0.10,
                "metrics.val_precision_saturation": 0.10,
            },
        ]
    )

    chosen = train_classification.choose_run(runs)

    assert chosen["run_id"] == "best-f1"


def test_choose_run_ignores_dirty_and_other_commit_runs():
    def candidate(run_id, dirty, commit, f1):
        return {
            "run_id": run_id,
            "tags.question": "p2",
            "tags.stage_of_work": "candidate",
            "tags.git_dirty": dirty,
            "tags.git_commit": commit,
            "metrics.val_macro_f1": f1,
            "metrics.val_recall_shortage": 0.90,
            "metrics.val_precision_shortage": 0.60,
            "metrics.val_precision_saturation": 0.60,
        }

    runs = pd.DataFrame(
        [
            candidate("dirty-best", "true", "abc123", 0.95),
            candidate("other-commit", "false", "def456", 0.90),
            candidate("clean-head", "false", "abc123", 0.80),
        ]
    )

    chosen = train_classification.choose_run(runs, git_commit="abc123")

    assert chosen["run_id"] == "clean-head"


def test_choose_run_allows_dirty_runs_for_local_work():
    runs = pd.DataFrame(
        [
            {
                "run_id": "dirty-best",
                "tags.question": "p2",
                "tags.stage_of_work": "candidate",
                "tags.git_dirty": "true",
                "tags.git_commit": "abc123",
                "metrics.val_macro_f1": 0.95,
                "metrics.val_recall_shortage": 0.90,
                "metrics.val_precision_shortage": 0.60,
                "metrics.val_precision_saturation": 0.60,
            }
        ]
    )

    chosen = train_classification.choose_run(runs, allow_dirty=True)

    assert chosen["run_id"] == "dirty-best"


def test_persistence_baseline_uses_the_normalized_lag_two_flow():
    # net_flow_lag2 is already divided by capacity in features.py; capacity must not be used again.
    X = pd.DataFrame(
        {
            "capacity": [20, 20, 20, 20],
            "net_flow_lag2": [-0.5, 0.0, 0.5, 0.39],
        }
    )
    model = train_classification.PersistenceRiskClassifier(tau=0.4).fit(X, [0, 0, 0, 0])

    assert model.predict(X).tolist() == [1, 0, 2, 0]
    assert model.predict_proba(X).shape == (4, 3)


def test_persistence_baseline_accepts_arrays_in_feature_column_order():
    X = pd.DataFrame(
        {column: [0.0, 0.0, 0.0, 0.0] for column in train_classification.FEATURE_COLUMNS}
    )
    X["capacity"] = 20.0
    X["net_flow_lag2"] = [-0.5, 0.0, 0.5, 0.39]
    model = train_classification.PersistenceRiskClassifier(tau=0.4).fit(X, [0, 0, 0, 0])

    assert model.predict(X.to_numpy()).tolist() == [1, 0, 2, 0]


def test_persistence_baseline_requires_the_normalized_flow_column():
    X = pd.DataFrame({"capacity": [20.0]})
    model = train_classification.PersistenceRiskClassifier(tau=0.4).fit(X, [0])

    with pytest.raises(ValueError, match="net_flow_lag2"):
        model.predict(X)


@pytest.mark.parametrize("value", [None, "", "none", "NONE", " none "])
def test_normalize_run_id_maps_mlflow_placeholders_to_none(value):
    assert train_classification.normalize_run_id(value) is None


def test_normalize_run_id_keeps_a_real_run_id():
    assert train_classification.normalize_run_id("abc123") == "abc123"


def test_split_masks_ignore_rows_without_a_split():
    frame = pd.DataFrame(
        {
            "station_id": [1, 1],
            "target_time": pd.to_datetime(["2024-12-01 00:00", "2024-12-02 00:00"]),
            "label_available_at": pd.to_datetime(["2024-12-01 01:00", "2024-12-02 01:00"]),
            "split": pd.Series(["test", pd.NA], dtype="string"),
        }
    )

    masks = train_classification.split_masks(frame)

    # A row without a target has no split; it must not raise or land in test.
    assert masks["test"].tolist() == [True, False]
