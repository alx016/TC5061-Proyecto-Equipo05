"""Candidate contracts at the existing estimator and train_one seams."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from test_regression_training import make_prepared, write_prepared
from typer.testing import CliRunner

from tc5061_proyecto_equipo5.modeling import train, train_regression
from tc5061_proyecto_equipo5.modeling.train_regression import candidates


def test_tweedie_has_explicit_loss_and_reproducible_seed():
    specs = {params["model"]: (build, params) for build, params in candidates(7).values()}
    build, params = specs["lightgbm_tweedie"]
    X, _, y = make_prepared(pd.date_range("2024-03-01", periods=48, freq="h"))
    model = build(X).fit(X, y)
    assert params["objective"] == "tweedie"
    assert params["tweedie_variance_power"] == 1.5
    assert params["random_state"] == 7
    assert np.isfinite(model.predict(X)).all()
    assert (model.predict(X) >= 0).all()


def test_hist_boosting_supports_many_nominal_stations_and_unseen_categories():
    specs = {params["model"]: (build, params) for build, params in candidates(7).values()}
    build, params = specs["hist_gradient_boosting"]
    # More stations than max_bins: fit must not treat station IDs as numeric distances
    # nor materialize one dense column per station across all rows.
    X, _, y = make_prepared(
        pd.date_range("2024-03-01", periods=2, freq="h"), stations=tuple(range(300))
    )
    X["municipality"] = pd.Series(["A", "B"] * 300, dtype="string")
    model = build(X).fit(X, y)
    assert params["loss"] == "poisson"
    assert params["early_stopping"] is False
    assert params["categorical_features"] == "from_dtype"
    assert params["random_state"] == 7
    assert params["categorical_max_categories"] == 255
    assert model.estimator_.steps[-1][1].is_categorical_.sum() == 2
    transformed = model.estimator_.steps[0][1].transform(X)
    assert transformed.shape == X.shape
    # Explicit unseen categories must work without learning anything from validation.
    unseen = X.iloc[:2].copy()
    unseen["station_id"] = "new-station"
    unseen["municipality"] = "new-municipality"
    predicted = model.predict(unseen)
    assert np.isfinite(predicted).all()
    assert (predicted >= 0).all()
    assert not hasattr(build(X), "estimator_")


@pytest.mark.parametrize(
    ("family", "hyperparams", "expected"),
    [
        ("lightgbm_tweedie", {}, {"objective": "tweedie", "tweedie_variance_power": "1.5"}),
        ("hist_gradient_boosting", {}, {"loss": "poisson", "early_stopping": "False"}),
        ("ridge", {"alpha": 10.0}, {"alpha": "10.0", "numeric_scaling": "robust"}),
    ],
)
def test_candidate_cli_logs_reloadable_model_and_effective_configuration(
    tmp_path, monkeypatch, family, hyperparams, expected
):
    folder = write_prepared(tmp_path)
    X = pd.read_parquet(folder / "X.parquet")
    keys = pd.read_parquet(folder / "keys.parquet")
    X.loc[keys["split"].eq("validation"), "station_id"] = "validation-only"
    X.to_parquet(folder / "X.parquet", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_regression, "REGRESSION_FEATURES_DIR", folder)
    monkeypatch.setattr(train_regression, "data_version", lambda: "synthetic")
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        args = ["--question", "p1", "--model", family, "--stage", "candidate", "--seed", "7"]
        if hyperparams:
            args += ["--alpha", "10.0"]
        result = CliRunner().invoke(train.app, args)
        assert result.exit_code == 0, result.output + str(result.exception)
        runs = mlflow.search_runs(experiment_names=["mibici-equipo05"])
        assert len(runs) == 1
        run_id = runs.iloc[0].run_id
        run = mlflow.get_run(run_id)
        assert run.data.tags["model_family"] == family
        assert run.data.tags["stage_of_work"] == "candidate"
        for name, value in expected.items():
            assert run.data.params[name] == value
        assert "test_wape" not in run.data.metrics
        assert run.data.params["train_end"] < run.data.params["val_start"]
        assert run.inputs.dataset_inputs[0].dataset.name == "p1-train"
        info = mlflow.models.get_model_info(f"runs:/{run_id}/model")
        assert info.signature is not None
        assert info.saved_input_example_info is not None
        loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        X = pd.read_parquet(folder / "X.parquet")
        # Inference uses raw feature inputs, including unseen station labels.
        example = X.iloc[:5].copy()
        example["station_id"] = "unseen-station"
        prediction = loaded.predict(example)
        assert np.isfinite(prediction).all()
        assert (prediction >= 0).all()
        estimator = loaded.estimator_.steps[-1][1]
        for name, value in estimator.get_params().items():
            assert run.data.params[name] == str(value), name
        paths = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id)}
        assert {"data", "config"} <= paths
        # Fits of an individual candidate never consume validation station categories.
        if family == "hist_gradient_boosting":
            encoder = loaded.estimator_.steps[0][1].named_transformers_["categorical"]
            assert encoder.steps[1][1].categories_[0].tolist() == ["1", "2"]
    finally:
        mlflow.set_tracking_uri(previous_uri)
