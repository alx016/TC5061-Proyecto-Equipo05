"""Observe P1 profiling through train_one and a temporary MLflow Backend Store."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from test_regression_training import make_prepared

from tc5061_proyecto_equipo5.features import FEATURE_VERSION
from tc5061_proyecto_equipo5.modeling import train_regression


@pytest.fixture
def training_store(tmp_path, monkeypatch):
    X, keys, y = make_prepared(
        pd.to_datetime(
            [
                "2024-09-15 08:00",
                "2024-09-16 08:00",
                "2024-10-15 08:00",
                "2024-10-16 08:00",
                "2024-12-15 08:00",
            ]
        )
    )
    folder = tmp_path / "p1"
    folder.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_regression, "REGRESSION_FEATURES_DIR", folder)
    monkeypatch.setattr(train_regression, "data_version", lambda: "synthetic")

    def prepare(keep_splits=None):
        keep = keys["split"].isin(keep_splits) if keep_splits else np.ones(len(X), dtype=bool)
        X.loc[keep].to_parquet(folder / "X.parquet", index=False)
        keys.loc[keep].to_parquet(folder / "keys.parquet", index=False)
        y.loc[keep].to_frame().to_parquet(folder / "y.parquet", index=False)
        (folder / "features_manifest.json").write_text(
            json.dumps({"feature_version": FEATURE_VERSION, "data_version": "synthetic"})
        )

    prepare()
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield prepare, mlflow.MlflowClient(), X, keys
    mlflow.end_run()
    mlflow.set_tracking_uri(previous_uri)


def test_train_one_measures_complete_pipeline_inside_run(training_store, monkeypatch):
    _, client, X, keys = training_store
    fit_run_ids = []
    prediction_inputs = []
    original_fit = train_regression.NonNegativeRegressor.fit
    original_predict = train_regression.NonNegativeRegressor.predict

    def fit(model, X, y):
        active = mlflow.active_run()
        fit_run_ids.append(active.info.run_id if active else None)
        return original_fit(model, X, y)

    def predict(model, X):
        prediction_inputs.append(X.copy())
        return original_predict(model, X)

    monkeypatch.setattr(train_regression.NonNegativeRegressor, "fit", fit)
    monkeypatch.setattr(train_regression.NonNegativeRegressor, "predict", predict)
    ticks = iter([10.0, 12.0, 20.0, 20.125])
    monkeypatch.setattr(train_regression, "perf_counter", lambda: next(ticks), raising=False)

    run_id = train_regression.train_one("ridge", "candidate", 7, {})
    run = client.get_run(run_id)

    assert fit_run_ids == [run_id]
    assert run.info.status == "FINISHED"
    assert run.data.metrics["fit_seconds"] == 2.0
    # Four validation rows take 125 ms: 31,250 ms per thousand rows.
    assert run.data.metrics["predict_ms_per_1k"] == 31250.0
    assert np.isfinite(list(run.data.metrics.values())).all()
    validation = X.loc[keys["split"].eq("validation")]
    assert sum(frame.equals(validation) for frame in prediction_inputs) == 1
    test = X.loc[keys["split"].eq("test")]
    assert not any(frame.equals(test) for frame in prediction_inputs)
    assert not any(name.startswith("test_") for name in run.data.metrics)
    assert run.data.params["predict_batch_size"] == "4"
    config = json.loads(
        Path(client.download_artifacts(run_id, "config/run_config.json")).read_text()
    )
    assert config["predict_batch_size"] == 4
    assert "perf_counter" in config["cost_measurement_protocol"]
    assert "validation" in config["cost_measurement_protocol"]


@pytest.mark.parametrize("missing_split", ["train", "validation"])
def test_train_one_rejects_empty_required_split_before_fit(
    training_store, monkeypatch, missing_split
):
    prepare, client, _, _ = training_store
    prepare({"train", "validation", "test"} - {missing_split})
    fit_called = False

    def fit(model, X, y):
        nonlocal fit_called
        fit_called = True
        raise AssertionError("fit must not start with invalid splits")

    monkeypatch.setattr(train_regression.NonNegativeRegressor, "fit", fit)
    with pytest.raises(ValueError, match=f"{missing_split}.*empty"):
        train_regression.train_one("ridge", "candidate", 7, {})
    assert not fit_called
    experiment = client.get_experiment_by_name("mibici-equipo05")
    assert client.search_runs([experiment.experiment_id]) == []


def test_train_one_propagates_fit_failure_and_marks_run_failed(training_store, monkeypatch):
    _, client, _, _ = training_store
    failed_run_ids = []

    def fail_fit(model, X, y):
        active = mlflow.active_run()
        failed_run_ids.append(active.info.run_id if active else None)
        raise RuntimeError("deliberate pipeline fit failure")

    monkeypatch.setattr(train_regression.NonNegativeRegressor, "fit", fail_fit)
    with pytest.raises(RuntimeError, match="deliberate pipeline fit failure"):
        train_regression.train_one("ridge", "candidate", 7, {})

    assert len(failed_run_ids) == 1
    assert failed_run_ids[0] is not None
    failed = client.get_run(failed_run_ids[0])
    assert failed.info.status == "FAILED"
    assert not {"fit_seconds", "predict_ms_per_1k", "val_wape"} & failed.data.metrics.keys()
    assert mlflow.active_run() is None


def test_train_one_accepts_empty_test_and_records_finite_costs(training_store):
    prepare, client, _, _ = training_store
    prepare({"train", "validation"})

    run_id = train_regression.train_one("ridge", "candidate", 7, {})
    run = client.get_run(run_id)

    assert run.info.status == "FINISHED"
    assert run.data.params["n_test"] == "0"
    for name in ("fit_seconds", "predict_ms_per_1k"):
        assert np.isfinite(run.data.metrics[name])
        assert run.data.metrics[name] >= 0
    assert not any(name.startswith("test_") for name in run.data.metrics)
