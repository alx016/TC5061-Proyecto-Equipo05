"""Unit tests for the P2 probability thresholds, metrics and selection rule."""

from types import SimpleNamespace

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


def test_calibrate_thresholds_meets_the_recall_floor_and_returns_no_fallback():
    y_true = np.array([1] * 20 + [2] * 20 + [0] * 60)
    proba = np.zeros((100, 3))
    proba[:20, 1] = np.linspace(0.98, 0.30, 20)
    proba[20:40, 2] = np.linspace(0.98, 0.30, 20)
    proba[40:, 0] = 0.99

    thresholds, fallback = train_classification.calibrate_thresholds(
        y_true, proba, classes=np.array([0, 1, 2])
    )
    prediction = train_classification.apply_thresholds(proba, np.array([0, 1, 2]), thresholds)

    assert fallback is False
    assert (prediction == 1).sum() >= 17  # at least the configured 85% of the 20 shortages
    assert thresholds[1] in train_classification.P2_THRESHOLD_GRID
    assert thresholds[2] in train_classification.P2_THRESHOLD_GRID


def test_calibrate_thresholds_matches_a_brute_force_search():
    rng = np.random.default_rng(7)
    y_true = rng.choice([0, 1, 2], size=200, p=[0.7, 0.15, 0.15])
    proba = rng.dirichlet([1.0, 0.3, 0.3], size=200)
    classes = np.array([0, 1, 2])

    expected_best = None
    for threshold_shortage in train_classification.P2_THRESHOLD_GRID:
        for threshold_saturation in train_classification.P2_THRESHOLD_GRID:
            prediction = train_classification.apply_thresholds(
                proba,
                classes,
                {1: threshold_shortage, 2: threshold_saturation},
            )
            macro = train_classification.f1_score(
                y_true, prediction, labels=[0, 1, 2], average="macro", zero_division=0
            )
            recall = train_classification.recall_score(
                y_true == 1, prediction == 1, zero_division=0
            )
            if recall >= train_classification.P2_MIN_RECALL_SHORTAGE:
                if expected_best is None or macro > expected_best[0]:
                    expected_best = (
                        macro,
                        threshold_shortage,
                        threshold_saturation,
                    )

    thresholds, fallback = train_classification.calibrate_thresholds(y_true, proba, classes)

    assert fallback is (expected_best is None)
    if expected_best is not None:
        assert thresholds == {1: expected_best[1], 2: expected_best[2]}


def test_calibrate_thresholds_falls_back_when_recall_is_unreachable():
    # No probability ever crosses the grid, so no pair can recall the shortage class.
    y_true = np.array([1] * 10 + [2] * 10 + [0] * 40)
    proba = np.zeros((60, 3))
    proba[:, 0] = 1.0

    thresholds, fallback = train_classification.calibrate_thresholds(
        y_true, proba, classes=np.array([0, 1, 2])
    )

    assert fallback is True
    assert thresholds == {1: 0.05, 2: 0.05}


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
    assert result["val_pr_auc_minority_mean"] == pytest.approx(
        (result["val_pr_auc_shortage"] + result["val_pr_auc_saturation"]) / 2
    )


def _run(run_id, minority_ap, macro_f1, *, question="p2", stage="candidate", dirty="false"):
    return {
        "run_id": run_id,
        "tags.question": question,
        "tags.stage_of_work": stage,
        "tags.git_dirty": dirty,
        "metrics.val_pr_auc_minority_mean": minority_ap,
        "metrics.val_macro_f1": macro_f1,
    }


def test_choose_run_ranks_clean_candidates_by_minority_ap():
    runs = pd.DataFrame(
        [
            _run("dirty-best", 0.60, 0.90, dirty="true"),
            _run("search-child", 0.55, 0.80, stage="search"),
            _run("other-question", 0.54, 0.70, question="p1"),
            _run("clean-best-ap", 0.40, 0.45),
            _run("clean-best-f1", 0.35, 0.90),
        ]
    )

    chosen = train_classification.choose_run(runs)

    assert chosen["run_id"] == "clean-best-ap"


def test_choose_run_breaks_minority_ap_ties_with_macro_f1():
    runs = pd.DataFrame(
        [
            _run("lower-f1", 0.40, 0.45),
            _run("higher-f1", 0.40, 0.50),
        ]
    )

    chosen = train_classification.choose_run(runs)

    assert chosen["run_id"] == "higher-f1"


def test_choose_run_rejects_a_frame_without_clean_runs():
    runs = pd.DataFrame([_run("dirty", 0.60, 0.90, dirty="true")])

    with pytest.raises(ValueError, match="No clean P2"):
        train_classification.choose_run(runs)


def test_selection_summary_counts_alerts_and_per_day_rate():
    test_metrics = {
        "test_recall_shortage": 0.80,
        "test_precision_shortage": 0.20,
        "test_recall_saturation": 0.70,
        "test_precision_saturation": 0.30,
        "test_macro_f1": 0.55,
        "test_pr_auc_minority_mean": 0.35,
    }
    prediction = np.array([0, 1, 2, 0, 1, 0])

    summary = train_classification.selection_summary(
        run_id="run-1",
        thresholds={1: 0.2, 2: 0.3},
        fallback=False,
        test_metrics=test_metrics,
        prediction=prediction,
        n_test_dates=2,
    )

    assert summary["run_id"] == "run-1"
    assert summary["n_alerts"] == 3
    assert summary["alerts_per_day"] == pytest.approx(1.5)
    assert summary["thresholds"] == {"shortage": 0.2, "saturation": 0.3}
    assert summary["threshold_fallback"] is False
    assert summary["pr_auc_minority_mean"] == pytest.approx(0.35)
    assert summary["recall_shortage"] == pytest.approx(0.80)


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


class _FakeRun:
    def __init__(self, tracking, run_id):
        self.tracking = tracking
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        self.tracking.active.append(self.info.run_id)
        return self

    def __exit__(self, *_exception):
        assert self.tracking.active.pop() == self.info.run_id
        return False


class _FakeTracking:
    """Record the MLflow calls that ``search`` makes, without touching a backend store."""

    def __init__(self):
        self.runs = {}
        self.active = []
        self.tables = {}
        self._counter = 0

    def start_run(self, run_name=None, nested=False):
        self._counter += 1
        run_id = f"run-{self._counter:03d}"
        self.runs[run_id] = {
            "run_id": run_id,
            "name": run_name,
            "nested": nested,
            "params": {},
            "tags": {},
            "metrics": {},
        }
        return _FakeRun(self, run_id)

    def set_tags(self, tags):
        for key, value in tags.items():
            self.set_tag(self._current, key, value)

    def set_tag(self, run_id, key, value):
        self.runs[run_id]["tags"][key] = value

    def log_param(self, key, value):
        self.set_param(self._current, key, value)

    def set_param(self, run_id, key, value):
        self.runs[run_id]["params"][key] = str(value)

    def log_params(self, params):
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key, value):
        self.runs[self._current]["metrics"][key] = value

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            self.log_metric(key, value)

    def log_artifact(self, path, artifact_path=None):
        self.tables[artifact_path] = pd.read_csv(path)

    def log_dict(self, dictionary, artifact_file):
        self.tables[artifact_file] = dictionary

    def MlflowClient(self):
        return SimpleNamespace(set_tag=self.set_tag)

    @property
    def _current(self):
        return self.active[-1]


def _install_fake_tracking(monkeypatch):
    tracking = _FakeTracking()
    calls = []

    def base_tags(question, model_family, stage):
        return {
            "question": question,
            "model_family": model_family,
            "stage_of_work": stage,
            "git_dirty": "false",
            "git_commit": "0" * 40,
        }

    def build_model(name, seed, hyperparams):
        calls.append({"build_model": (name, seed, dict(hyperparams))})
        return (name, seed, tuple(sorted(hyperparams.items()))), {
            "model": name,
            "seed": seed,
            **hyperparams,
        }

    def train_and_log(
        name,
        model,
        model_params,
        data,
        seed,
        stage,
        model_family=None,
        nested=False,
        log_model=True,
    ):
        calls.append(
            {
                "name": name,
                "model": model,
                "model_params": model_params,
                "data": data,
                "seed": seed,
                "stage": stage,
                "nested": nested,
                "log_model": log_model,
            }
        )
        run = tracking.start_run(name, nested)
        with run:
            tracking.set_tags(base_tags("p2", model_family or model_params["model"], stage))
            tracking.log_params(
                {"data_version": train_classification.data_version(), **model_params}
            )
            val_macro_f1 = (
                0.40
                + float(model_params["learning_rate"])
                + (float(model_params["num_leaves"]) / 1000)
            )
            tracking.log_metrics(
                {
                    "val_macro_f1": val_macro_f1,
                    "val_pr_auc_minority_mean": val_macro_f1,
                    "val_recall_shortage": 0.90,
                    "val_precision_shortage": 0.60,
                    "val_precision_saturation": 0.60,
                }
            )
        return {
            "run_id": run.info.run_id,
            "run_name": name,
            "model": model,
            "thresholds": {1: 0.5, 2: 0.5},
            "threshold_fallback": False,
            "val_macro_f1": val_macro_f1,
            "val_pr_auc_minority_mean": val_macro_f1,
            "val_recall_shortage": 0.90,
            "val_precision_shortage": 0.60,
            "val_precision_saturation": 0.60,
            "fit_seconds": 0.0,
        }

    # ``search`` opens the parent run itself, so a fresh fake run must be created for it.
    def start_run(run_name=None, nested=False):
        return tracking.start_run(run_name, nested)

    monkeypatch.setattr(train_classification, "setup_experiment", lambda: None)
    monkeypatch.setattr(train_classification, "load_rows", lambda path: pd.DataFrame())
    monkeypatch.setattr(
        train_classification,
        "build_data",
        lambda frame, source: {
            "X": {"train": pd.DataFrame(), "validation": pd.DataFrame()},
            "y": {"train": pd.Series(dtype=int), "validation": pd.Series(dtype=int)},
            "n": {"train": 0, "validation": 0, "test": 0},
            "source": source,
            "source_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(train_classification, "base_tags", base_tags)
    monkeypatch.setattr(train_classification, "build_model", build_model)
    monkeypatch.setattr(train_classification, "train_and_log", train_and_log)
    monkeypatch.setattr(train_classification, "data_version", lambda: "synthetic-v1")
    for attribute in (
        "start_run",
        "set_tags",
        "log_param",
        "log_params",
        "log_metric",
        "log_metrics",
        "log_artifact",
        "log_dict",
    ):
        monkeypatch.setattr(train_classification.mlflow, attribute, getattr(tracking, attribute))
    monkeypatch.setattr(train_classification.mlflow, "MlflowClient", tracking.MlflowClient)
    return tracking, calls


def _runs_frame(tracking):
    rows = []
    for run in tracking.runs.values():
        row = {"run_id": run["run_id"]}
        for key, value in run["tags"].items():
            row[f"tags.{key}"] = value
        for key, value in run["metrics"].items():
            row[f"metrics.{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_search_promotes_the_winning_grid_config_as_a_selectable_candidate(monkeypatch):
    tracking, calls = _install_fake_tracking(monkeypatch)
    seed = 7

    parent_id = train_classification.search(seed=seed)

    searches = [call for call in calls if call.get("stage") == "search"]
    candidates = [call for call in calls if call.get("stage") == "candidate"]
    assert len(searches) == 9
    assert all(call["log_model"] is False and call["nested"] is True for call in searches)
    assert [call["name"] for call in searches] == [
        f"p2-search-lightgbm-{index:02d}" for index in range(9)
    ]

    # The trials keep their hyperparameters next to their results.
    table = tracking.tables["search"]
    assert len(table) == 9
    assert "hyperparams" in table.columns
    trial_ids = {
        run_id
        for run_id, run in tracking.runs.items()
        if run["name"] in {call["name"] for call in searches}
    }
    assert set(table["run_id"]) == trial_ids
    assert table["hyperparams"].nunique() == 9

    winner = {
        "learning_rate": 0.1,
        "num_leaves": 127,
        "min_child_samples": 100,
    }
    winning_call = next(
        call
        for call in searches
        if call["model_params"] == {"model": "lightgbm", "seed": seed, **winner}
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["log_model"] is True
    assert candidate["nested"] is True
    assert candidate["seed"] == seed
    assert candidate["model_params"] == {"model": "lightgbm", "seed": seed, **winner}
    # Search never rebuilds its data per run, so every call sees the same train/validation split.
    logged = [call for call in calls if "data" in call]
    assert all(call["data"] is logged[0]["data"] for call in logged)

    # Lineage: the candidate points at its parent and at the winning trial, and vice versa.
    candidate_id = next(
        run_id for run_id, run in tracking.runs.items() if run["name"] == candidate["name"]
    )
    winning_id = next(
        run_id for run_id, run in tracking.runs.items() if run["name"] == winning_call["name"]
    )
    assert tracking.runs[candidate_id]["tags"]["search_parent_run_id"] == parent_id
    assert tracking.runs[candidate_id]["tags"]["source_trial_run_id"] == winning_id
    assert tracking.runs[candidate_id]["tags"]["stage_of_work"] == "candidate"
    assert tracking.runs[parent_id]["params"]["best_candidate_run_id"] == candidate_id
    assert tracking.runs[parent_id]["params"]["best_run_id"] == winning_id

    # The parent records the data version that the trials and the candidate log.
    assert tracking.runs[parent_id]["params"]["data_version"] == "synthetic-v1"
    assert tracking.runs[winning_id]["params"]["data_version"] == "synthetic-v1"
    assert tracking.runs[candidate_id]["params"]["data_version"] == "synthetic-v1"

    # Search compares on validation only: no run carries a test metric.
    for run in tracking.runs.values():
        assert not any(key.startswith("test_") for key in run["metrics"])

    # The promoted candidate is a normal candidate for `choose_run`.
    chosen = train_classification.choose_run(_runs_frame(tracking), git_commit="0" * 40)
    assert chosen["run_id"] == candidate_id
