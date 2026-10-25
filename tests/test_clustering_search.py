"""Unit tests for the P3 nested search over K and the run selection.

The MLflow test uses a throwaway SQLite store and a fake ``train_and_log``, so it checks the
parent/child structure without real data, training or plots.
"""

import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from typer.testing import CliRunner

from tc5061_proyecto_equipo5.modeling import search as search_cli
from tc5061_proyecto_equipo5.modeling import train_clustering
from tc5061_proyecto_equipo5.modeling.common import log_model_with_signature
from tc5061_proyecto_equipo5.modeling.train_clustering import (
    K_VALUES,
    choose_run,
    search,
    select,
    select_winner,
)

NAN = float("nan")


def test_select_winner_picks_the_highest_validation_silhouette():
    rows = [{"k": 3, "val_silhouette": 0.1}, {"k": 4, "val_silhouette": 0.3}]

    assert select_winner(rows)["k"] == 4


def test_select_winner_breaks_ties_with_the_smaller_k():
    rows = [{"k": 5, "val_silhouette": 0.2}, {"k": 3, "val_silhouette": 0.2}]

    assert select_winner(rows)["k"] == 3


def test_select_winner_ignores_nan_silhouettes():
    rows = [{"k": 3, "val_silhouette": NAN}, {"k": 4, "val_silhouette": 0.1}]

    assert select_winner(rows)["k"] == 4


def test_select_winner_without_a_finite_silhouette_raises():
    with pytest.raises(ValueError, match="no Trial with a finite validation silhouette"):
        select_winner([{"k": 3, "val_silhouette": NAN}])


def test_search_rejects_an_unknown_model_before_loading_data(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not run for an unknown model")

    monkeypatch.setattr(train_clustering, "setup_experiment", forbidden)
    monkeypatch.setattr(train_clustering, "load_data", forbidden)

    with pytest.raises(ValueError, match="model must be one of"):
        search(model="lightgbm")


@pytest.fixture
def tracking_store(tmp_path, monkeypatch):
    """Point MLflow at a throwaway SQLite store and restore the previous URI afterwards."""
    previous = mlflow.get_tracking_uri()
    # Outside `mlflow run` there is no entry-point run to resume
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    # Artifacts in tmp_path too, so the tests leave nothing in the repo's mlruns/
    mlflow.set_experiment(
        experiment_id=mlflow.create_experiment(
            "p3-search-test", artifact_location=(tmp_path / "artifacts").as_uri()
        )
    )
    yield MlflowClient()
    mlflow.set_tracking_uri(previous)


def test_search_logs_one_nested_trial_per_k_and_tags_the_winner(tracking_store, monkeypatch):
    # K=4 wins; the real Trial would train, plot and log a model, the fake only opens its run
    silhouettes = {3: 0.10, 4: 0.30, 5: 0.20, 6: NAN}

    def fake_train_and_log(name, estimator, kind, data, seed, **kwargs):
        assert kwargs["nested"] is True and kwargs["stage"] == "search"
        k = kwargs["model_params"]["n_clusters"]
        with mlflow.start_run(run_name=name, nested=kwargs["nested"]) as run:
            return {
                "run_id": run.info.run_id,
                "run_name": name,
                "val_silhouette": silhouettes[k],
                "val_davies_bouldin": 1.0,
                "val_temporal_ari": 0.7,
            }

    monkeypatch.setattr(train_clustering, "setup_experiment", lambda: None)
    monkeypatch.setattr(train_clustering, "load_data", lambda base_dir: {})
    monkeypatch.setattr(train_clustering, "train_and_log", fake_train_and_log)

    study_id = search(model="kmeans")

    client = tracking_store
    study = client.get_run(study_id)
    trials = client.search_runs(
        [study.info.experiment_id], f"tags.mlflow.parentRunId = '{study_id}'"
    )
    assert len(trials) == len(K_VALUES)
    assert study.data.tags["mlflow.runName"] == "p3-search-kmeans"
    assert study.data.tags["stage_of_work"] == "search"
    assert study.data.metrics["best_k"] == 4
    winner = {run.info.run_id for run in trials if run.data.tags.get("search_winner") == "true"}
    assert winner == {study.data.tags["best_run_id"]}
    # selected is reserved for the single run chosen by select
    assert not any("selected" in run.data.tags for run in trials)
    assert [artifact.path for artifact in client.list_artifacts(study_id, "search")] == [
        "search/search_results.csv"
    ]


def test_p3_is_dispatched_by_the_search_entry_point(monkeypatch):
    calls = []

    def searcher(model):
        calls.append(model)
        return "p3-study"

    monkeypatch.setitem(search_cli.SEARCHERS, "p3", searcher)

    result = CliRunner().invoke(search_cli.app, ["--question", "p3", "--model", "kmeans"])

    assert result.exit_code == 0, result.output
    assert calls == ["kmeans"]


def test_p3_search_with_the_default_lightgbm_model_fails_clearly(monkeypatch):
    # The search entry point defaults to lightgbm (P1), so P3 must ask for its model explicitly
    monkeypatch.setattr(train_clustering, "setup_experiment", lambda: None)

    result = CliRunner().invoke(search_cli.app, ["--question", "p3"])

    assert result.exit_code != 0
    assert "model must be one of" in result.output


def p3_runs(**overrides) -> pd.DataFrame:
    """Three clean P3 runs on one commit; ward has the best validation silhouette."""
    frame = pd.DataFrame(
        {
            "run_id": ["kmeans", "ward", "baseline"],
            "tags.question": ["p3", "p3", "p3"],
            "tags.stage_of_work": ["candidate", "candidate", "baseline"],
            "tags.git_dirty": ["false", "false", "false"],
            "tags.git_commit": ["abc", "abc", "abc"],
            "params.n_clusters": ["3", "3", "4"],
            "metrics.val_silhouette": [0.17, 0.20, 0.12],
            "metrics.val_davies_bouldin": [1.7, 1.5, 1.9],
        }
    )
    return frame.assign(**overrides)


def test_choose_run_picks_the_best_validation_silhouette():
    assert choose_run(p3_runs())["run_id"] == "ward"


def test_choose_run_breaks_ties_with_davies_bouldin_then_k():
    runs = p3_runs(**{"metrics.val_silhouette": [0.2, 0.2, 0.2]})
    assert choose_run(runs)["run_id"] == "ward"
    runs = runs.assign(
        **{"metrics.val_davies_bouldin": [1.5, 1.5, 1.5], "params.n_clusters": ["5", "4", "3"]}
    )
    assert choose_run(runs)["run_id"] == "baseline"


def test_choose_run_ignores_search_trials_other_questions_and_nan():
    runs = p3_runs(
        **{
            "tags.question": ["p3", "p3", "p2"],
            "tags.stage_of_work": ["candidate", "search", "baseline"],
            "metrics.val_silhouette": [0.10, 0.20, 0.90],
        }
    )
    assert choose_run(runs)["run_id"] == "kmeans"
    runs = runs.assign(**{"metrics.val_silhouette": [np.nan, 0.20, 0.90]})
    with pytest.raises(ValueError, match="No eligible"):
        choose_run(runs)


def test_choose_run_skips_dirty_runs_and_other_commits():
    runs = p3_runs(**{"tags.git_dirty": ["false", "true", "false"]})
    assert choose_run(runs)["run_id"] == "kmeans"
    assert choose_run(runs, allow_dirty=True)["run_id"] == "ward"
    assert choose_run(p3_runs(), git_commit="abc")["run_id"] == "ward"
    with pytest.raises(ValueError, match="No eligible"):
        choose_run(p3_runs(), git_commit="zzz")


def select_data() -> dict:
    """Two well separated groups of stations, the same in every window."""
    names = ["departures_hour_08", "arrivals_hour_18"]
    matrix = pd.DataFrame(
        np.vstack([np.zeros((5, 2)), np.full((5, 2), 10.0)])
        + np.random.default_rng(0).normal(scale=0.1, size=(10, 2)),
        columns=names,
    )
    keys = pd.DataFrame({"station_id": range(10)})
    return {
        "manifest": {"data_version": "v1"},
        "matrices": {split: matrix for split in ("train", "validation", "test")},
        "keys": {split: keys for split in ("train", "validation", "test")},
    }


def log_p3_run(data: dict, selected: bool = False, features_data_version: str = "v1") -> str:
    """Log a clean P3 candidate with a real K-Means model, as train_and_log would."""
    train = data["matrices"]["train"]
    with mlflow.start_run() as run:
        mlflow.set_tags({"question": "p3", "stage_of_work": "candidate", "git_dirty": "false"})
        if selected:
            mlflow.set_tag("selected", "true")
        mlflow.log_param("features_data_version", features_data_version)
        log_model_with_signature(
            KMeans(n_clusters=2, n_init=10, random_state=0).fit(train),
            train,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    return run.info.run_id


def test_select_scores_december_once_and_moves_the_selected_tag(tracking_store, monkeypatch):
    data = select_data()
    monkeypatch.setattr(train_clustering, "setup_experiment", lambda: None)
    monkeypatch.setattr(train_clustering, "load_data", lambda base_dir: data)
    old = log_p3_run(data, selected=True)
    new = log_p3_run(data)

    assert select(run_id=new) == new

    client = tracking_store
    chosen = client.get_run(new)
    assert {"test_silhouette", "test_davies_bouldin", "test_temporal_ari"} <= set(
        chosen.data.metrics
    )
    assert chosen.data.metrics["test_temporal_ari"] == 1.0
    assert chosen.data.tags["selected"] == "true"
    assert "selected" not in client.get_run(old).data.tags
    assert "config/selection.json" in [
        artifact.path for artifact in client.list_artifacts(new, "config")
    ]
    # A second call must not score December again
    select(run_id=new)
    assert len(client.get_metric_history(new, "test_silhouette")) == 1


def test_select_refuses_a_run_trained_on_other_features(tracking_store, monkeypatch):
    data = select_data()
    monkeypatch.setattr(train_clustering, "setup_experiment", lambda: None)
    monkeypatch.setattr(train_clustering, "load_data", lambda base_dir: data)
    stale = log_p3_run(data, features_data_version="v0")

    with pytest.raises(RuntimeError, match="features changed"):
        select(run_id=stale)

    run = tracking_store.get_run(stale)
    assert "selected" not in run.data.tags
    assert not any(name.startswith("test_") for name in run.data.metrics)
