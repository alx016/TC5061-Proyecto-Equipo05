"""Search contracts observed through P1 search and a real temporary Backend Store."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from test_regression_costs import training_store as training_store  # noqa: PLC0414 - pytest fixture
from test_regression_training import make_prepared

from tc5061_proyecto_equipo5.modeling import train_regression


def results(client, run_id):
    return pd.read_csv(client.download_artifacts(run_id, "search/search_results.csv"))


def test_search_logs_eight_complete_trials_and_selects_first_tie(training_store, monkeypatch):
    _, client, X, keys = training_store
    fitted = []
    predicted = []
    original_fit = train_regression.NonNegativeRegressor.fit
    original_predict = train_regression.NonNegativeRegressor.predict

    def fit(model, X, y):
        fitted.append(mlflow.active_run().info.run_id)
        return original_fit(model, X, y)

    def predict(model, X):
        predicted.append(X.copy())
        return original_predict(model, X)

    monkeypatch.setattr(train_regression.NonNegativeRegressor, "fit", fit)
    monkeypatch.setattr(train_regression.NonNegativeRegressor, "predict", predict)
    study_id = train_regression.search(seed=7)
    study = client.get_run(study_id)
    trials = client.search_runs(
        [study.info.experiment_id], f"tags.mlflow.parentRunId = '{study_id}'"
    )
    table = results(client, study_id)

    assert len(trials) == len(table) == len(fitted) == 8
    assert set(fitted) == set(table.run_id)
    assert set(table.columns) == {
        "run_id",
        "status",
        "learning_rate",
        "num_leaves",
        "n_estimators",
        "val_mae",
        "val_rmse",
        "val_wape",
        "fit_seconds",
        "predict_ms_per_1k",
        "selected",
    }
    assert table[["learning_rate", "num_leaves", "n_estimators"]].values.tolist() == [
        [0.03, 31, 200],
        [0.03, 31, 300],
        [0.03, 63, 200],
        [0.03, 63, 300],
        [0.05, 31, 200],
        [0.05, 31, 300],
        [0.05, 63, 200],
        [0.05, 63, 300],
    ]
    assert table.status.eq("FINISHED").all()
    assert np.isfinite(table[["fit_seconds", "predict_ms_per_1k", "val_wape"]]).all().all()
    # Four training rows cannot split with min_child_samples=100: all trials predict
    # the same training mean. Deterministic tie breaking therefore picks grid position 0.
    assert table.val_wape.nunique() == 1
    assert table.selected.tolist() == [True, False, False, False, False, False, False, False]
    assert study.data.tags["best_run_id"] == table.run_id.iloc[0]
    assert study.data.metrics["best_val_wape"] == pytest.approx(table.val_wape.iloc[0])
    assert study.data.tags["mlflow.runName"] == "p1-search-lightgbm"
    assert study.data.tags["objective"] == "minimize val_wape"
    assert study.data.tags["hypothesis"]
    assert json.loads(study.data.params["search_space"])["learning_rate"] == [0.03, 0.05]
    assert study.data.params["seed"] == "7"
    assert study.info.status == "FINISHED"
    assert mlflow.active_run() is None
    assert not any(frame.equals(X.loc[keys.split.eq("test")]) for frame in predicted)
    for trial in trials:
        assert trial.info.status == "FINISHED"
        assert trial.data.tags["question"] == "p1"
        assert trial.data.tags["stage_of_work"] == "search"
        assert trial.data.tags["model_family"] == "lightgbm"
        assert trial.data.params["objective"] == "poisson"
        assert not any(metric.startswith("test_") for metric in trial.data.metrics)
        assert {a.path for a in client.list_artifacts(trial.info.run_id, "plots")} == {
            f"plots/val_{name}.png"
            for name in ("real_vs_predicted", "residuals_vs_predicted", "residuals_histogram")
        }
        assert mlflow.models.get_model_info(f"runs:/{trial.info.run_id}/model").signature
        assert client.list_artifacts(trial.info.run_id, "data")
        assert client.list_artifacts(trial.info.run_id, "config")


@pytest.mark.parametrize("demand,status", [(0.0, "WAPE_UNDEFINED"), (5e-324, "WAPE_NONFINITE")])
def test_search_fails_without_fictitious_winner_when_wape_is_undefined(
    training_store, demand, status
):
    _, client, _, keys = training_store
    folder = train_regression.REGRESSION_FEATURES_DIR
    y = pd.read_parquet(folder / "y.parquet").astype(float)
    y.loc[keys.split.eq("validation"), :] = demand
    y.to_parquet(folder / "y.parquet", index=False)

    with pytest.raises(ValueError, match="no selectable Trial"):
        train_regression.search()

    experiment = client.get_experiment_by_name("mibici-equipo05")
    (study,) = client.search_runs(
        [experiment.experiment_id], "tags.mlflow.runName = 'p1-search-lightgbm'"
    )
    table = results(client, study.info.run_id)
    assert study.info.status == "FAILED"
    assert "best_run_id" not in study.data.tags
    assert "best_val_wape" not in study.data.metrics
    assert len(table) == 8
    assert not np.isfinite(table.val_wape).any()
    assert not table.selected.any()
    assert table.status.eq(status).all()
    assert mlflow.active_run() is None


def test_search_aborts_failed_fit_and_preserves_partial_table(training_store, monkeypatch):
    _, client, _, _ = training_store
    original_fit = train_regression.NonNegativeRegressor.fit
    calls = []

    def fit(model, X, y):
        calls.append(mlflow.active_run().info.run_id)
        if len(calls) == 3:
            raise RuntimeError("deliberate third Trial failure")
        return original_fit(model, X, y)

    monkeypatch.setattr(train_regression.NonNegativeRegressor, "fit", fit)
    with pytest.raises(RuntimeError, match="third Trial failure"):
        train_regression.search()

    experiment = client.get_experiment_by_name("mibici-equipo05")
    (study,) = client.search_runs(
        [experiment.experiment_id], "tags.mlflow.runName = 'p1-search-lightgbm'"
    )
    table = results(client, study.info.run_id)
    assert len(table) == 3
    assert table.run_id.tolist() == calls
    assert table.status.tolist() == ["FINISHED", "FINISHED", "FAILED"]
    assert table.selected.tolist() == [False, False, False]
    assert pd.isna(table.val_wape.iloc[2])
    assert client.get_run(calls[-1]).info.status == "FAILED"
    assert study.info.status == "FAILED"
    assert "best_run_id" not in study.data.tags
    assert "best_val_wape" not in study.data.metrics
    assert mlflow.active_run() is None


@pytest.mark.parametrize("context", ["active", "project", "identified-study"])
def test_search_adopts_study_without_orphans_or_closing_caller(
    training_store, monkeypatch, context
):
    _, client, _, _ = training_store
    train_regression.setup_experiment()
    experiment = client.get_experiment_by_name("mibici-equipo05")
    if context in {"active", "identified-study"}:
        tags = (
            {"question": "p1", "model_family": "lightgbm", "stage_of_work": "search"}
            if context == "identified-study"
            else {}
        )
        existing = mlflow.start_run(run_name="entrypoint", tags=tags)
    else:
        existing = client.create_run(experiment.experiment_id)
    # Covers the environment hint both before and after the project run is activated.
    if context == "identified-study":
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_RUN_ID", existing.info.run_id)

    study_id = train_regression.search(seed=19)

    assert study_id == existing.info.run_id
    assert len(client.search_runs([experiment.experiment_id])) == 9
    children = client.search_runs(
        [experiment.experiment_id], f"tags.mlflow.parentRunId = '{study_id}'"
    )
    assert len(children) == 8
    assert all(child.info.status == "FINISHED" for child in children)
    assert client.get_run(study_id).data.tags["mlflow.runName"] == "p1-search-lightgbm"
    if context in {"active", "identified-study"}:
        assert mlflow.active_run().info.run_id == study_id
        assert client.get_run(study_id).info.status == "RUNNING"
        mlflow.log_param("caller_still_owns_run", True)
        mlflow.end_run()
    else:
        assert mlflow.active_run() is None
        assert client.get_run(study_id).info.status == "FINISHED"


@pytest.mark.parametrize("model", ["ridge", "lightgbm_tweedie", "unknown"])
def test_search_rejects_unsupported_model_without_loading_or_opening_runs(
    training_store, monkeypatch, model
):
    _, client, _, _ = training_store

    def forbidden_load(*args):
        pytest.fail("unsupported models must fail before prepared data loading")

    monkeypatch.setattr(train_regression, "load_prepared", forbidden_load)
    with pytest.raises(ValueError, match="supports only 'lightgbm'"):
        train_regression.search(model=model)
    assert client.get_experiment_by_name("mibici-equipo05") is None


@pytest.mark.parametrize("tags", [{"question": "p2"}, {}])
def test_search_rejects_unrelated_caller_without_closing_it(training_store, monkeypatch, tags):
    _, client, _, _ = training_store
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    train_regression.setup_experiment()
    with mlflow.start_run(run_name="unrelated-caller", tags=tags) as caller:
        with pytest.raises(ValueError, match="not a compatible P1 Study"):
            train_regression.search()
        assert mlflow.active_run().info.run_id == caller.info.run_id
        assert client.get_run(caller.info.run_id).info.status == "RUNNING"
        assert client.get_run(caller.info.run_id).data.tags == caller.data.tags
        assert len(client.search_runs([caller.info.experiment_id])) == 1


def test_search_preserves_fit_error_if_partial_table_cannot_be_persisted(
    training_store, monkeypatch
):
    def fail_fit(model, X, y):
        raise RuntimeError("original training failure")

    original_log = mlflow.log_artifact

    def fail_summary(local_path, artifact_path=None, **kwargs):
        if artifact_path == "search":
            raise OSError("diagnostic artifact unavailable")
        return original_log(local_path, artifact_path, **kwargs)

    monkeypatch.setattr(train_regression.NonNegativeRegressor, "fit", fail_fit)
    monkeypatch.setattr(mlflow, "log_artifact", fail_summary)
    with pytest.raises(RuntimeError, match="original training failure"):
        train_regression.search()


def test_search_selects_lowest_wape_when_trials_differ(training_store):
    _, client, _, _ = training_store
    hours = pd.date_range("2024-09-01", periods=240, freq="h").union(
        pd.date_range("2024-10-01", periods=48, freq="h")
    )
    X, keys, _ = make_prepared(hours)
    y = (2 * X.departures_lag_24 + X.hour).rename("demand")
    folder = train_regression.REGRESSION_FEATURES_DIR
    X.to_parquet(folder / "X.parquet", index=False)
    keys.to_parquet(folder / "keys.parquet", index=False)
    y.to_frame().to_parquet(folder / "y.parquet", index=False)

    study_id = train_regression.search()
    study = client.get_run(study_id)
    table = results(client, study_id)

    assert table.val_wape.nunique() > 1
    (selected,) = table.loc[table.selected, "run_id"]
    assert selected == study.data.tags["best_run_id"]
    assert client.get_run(selected).data.metrics["val_wape"] == pytest.approx(table.val_wape.min())
    assert study.data.metrics["best_val_wape"] == pytest.approx(table.val_wape.min())
