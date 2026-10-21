"""Validation evidence through training and a real temporary MLflow Backend Store."""

import json
from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pytest
from test_regression_training import make_prepared

from tc5061_proyecto_equipo5.features import FEATURE_VERSION
from tc5061_proyecto_equipo5.modeling import train_regression


def test_training_logs_aligned_validation_diagnostics_without_scoring_test(tmp_path, monkeypatch):
    # Repeated validation hour covers the network aggregation; irregular index catches
    # accidental pandas alignment against a freshly created prediction Series.
    X, keys, y = make_prepared(
        pd.to_datetime(
            [
                "2024-03-01 08:00",
                "2024-03-08 08:00",
                "2024-10-01 08:00",
                "2024-10-01 08:00",
                "2024-10-02 09:00",
                "2024-12-01 08:00",
            ]
        ),
        stations=(1,),
    )
    index = [40, 7, 90, 12, 62, 5]
    X.index = keys.index = y.index = index
    y[:] = [2, 2, 5, 0, 2, 999]
    folder = tmp_path / "prepared"
    folder.mkdir()
    X.to_parquet(folder / "X.parquet")
    keys.to_parquet(folder / "keys.parquet")
    y.to_frame().to_parquet(folder / "y.parquet")
    (folder / "features_manifest.json").write_text(
        json.dumps(
            {
                "feature_version": FEATURE_VERSION,
                "data_version": "synthetic",
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_regression, "REGRESSION_FEATURES_DIR", folder)
    monkeypatch.setattr(train_regression, "data_version", lambda: "synthetic")
    observations = {}
    predicted_indices = []
    predict = train_regression.NonNegativeRegressor.predict

    def observe_predict(model, frame):
        predicted_indices.append(list(frame.index))
        return predict(model, frame)

    monkeypatch.setattr(train_regression.NonNegativeRegressor, "predict", observe_predict)
    scatter, hist, plot = Axes.scatter, Axes.hist, Axes.plot

    def observe_scatter(axis, x, y, *args, **kwargs):
        observations["scatter"] = (np.asarray(x), np.asarray(y))
        return scatter(axis, x, y, *args, **kwargs)

    def observe_hist(axis, x, *args, **kwargs):
        observations["histogram"] = np.asarray(x)
        return hist(axis, x, *args, **kwargs)

    def observe_plot(axis, *args, **kwargs):
        if kwargs.get("label") in {"Real", "Predicción"}:
            observations[kwargs["label"]] = np.asarray(args[1])
        return plot(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", observe_scatter)
    monkeypatch.setattr(Axes, "hist", observe_hist)
    monkeypatch.setattr(Axes, "plot", observe_plot)
    previous_uri = mlflow.get_tracking_uri()
    previous_figures = plt.get_fignums()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        run_id = train_regression.train_one("seasonal_mean", "candidate", 7, {})
        client = mlflow.MlflowClient()
        artifacts = {a.path for a in client.list_artifacts(run_id, "plots")}
        assert artifacts == {
            "plots/val_real_vs_predicted.png",
            "plots/val_residuals_vs_predicted.png",
            "plots/val_residuals_histogram.png",
        }
        for artifact in artifacts:
            path = Path(client.download_artifacts(run_id, artifact))
            pixels = mpimg.imread(path)
            assert pixels.ndim == 3
            assert min(pixels.shape[:2]) > 100
        run = client.get_run(run_id)
        assert run.data.metrics["val_mae"] == pytest.approx(5 / 3)
        assert run.data.metrics["val_wape"] == pytest.approx(5 / 7)
        assert not any(name.startswith("test_") for name in run.data.metrics)
        assert {a.path for a in client.list_artifacts(run_id)} >= {"data", "config", "plots"}
        assert mlflow.models.get_model_info(f"runs:/{run_id}/model").signature is not None
    finally:
        mlflow.set_tracking_uri(previous_uri)
    np.testing.assert_allclose(observations["scatter"][0], [2, 2, 2])
    np.testing.assert_allclose(observations["scatter"][1], [3, -2, 0])
    np.testing.assert_allclose(observations["histogram"], [3, -2, 0])
    np.testing.assert_allclose(observations["Real"], [5, 2])
    np.testing.assert_allclose(observations["Predicción"], [4, 2])
    assert predicted_indices.count([90, 12, 62]) == 1
    assert all(5 not in indices for indices in predicted_indices)
    assert plt.get_fignums() == previous_figures
