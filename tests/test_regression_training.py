"""Unit tests for the P1 regression training script.

All tests use small synthetic DataFrames, so they do not need the real dataset.
"""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tc5061_proyecto_equipo5.features import FEATURE_VERSION
from tc5061_proyecto_equipo5.modeling import train_regression
from tc5061_proyecto_equipo5.modeling.train_regression import (
    NonNegativeRegressor,
    SeasonalMeanRegressor,
    candidates,
    metrics,
    split_masks,
    split_ranges,
)


def make_prepared(hours, stations=(1, 2)):
    """Return X, keys and y shaped like the prepared P1 artifacts, one row per station-hour."""
    rows = [(station, hour) for station in stations for hour in hours]
    target_time = pd.to_datetime([hour for _, hour in rows])
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "station_id": [str(station) for station, _ in rows],
            "hour_sin": np.sin(2 * np.pi * target_time.hour / 24),
            "hour_cos": np.cos(2 * np.pi * target_time.hour / 24),
            "hour": target_time.hour.astype("int8"),
            "day_of_week": target_time.dayofweek.astype("int8"),
            "departures_lag_24": rng.poisson(3, len(rows)).astype(float),
        }
    )
    keys = pd.DataFrame(
        {
            "station_id": [station for station, _ in rows],
            "target_time": target_time,
            "issue_time": target_time - pd.Timedelta(minutes=10),
            "label_available_at": target_time + pd.Timedelta(hours=1),
            "split": pd.cut(
                target_time,
                bins=pd.to_datetime(["2024-01-01", "2024-10-01", "2024-12-01", "2025-01-01"]),
                labels=["train", "validation", "test"],
                right=False,
            ),
            "history_complete": True,
        }
    )
    y = pd.Series(rng.poisson(3, len(rows)), name="demand")
    return X, keys, y


def test_seasonal_mean_uses_train_means_and_falls_back():
    X = pd.DataFrame(
        {"station_id": ["1", "1", "1", "2"], "day_of_week": [0, 0, 1, 0], "hour": [8, 8, 8, 8]}
    )
    model = SeasonalMeanRegressor().fit(X, [2, 4, 10, 6])

    unseen = pd.DataFrame(
        {"station_id": ["1", "1", "3"], "day_of_week": [0, 5, 0], "hour": [8, 8, 8]}
    )
    predicted = model.predict(unseen)

    # Known group: mean of 2 and 4.  Unseen weekday: station 1 mean.  Unseen station: global.
    assert predicted.tolist() == pytest.approx([3.0, 16 / 3, 5.5])


def test_non_negative_regressor_clips_at_zero():
    class Negative:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.array([-2.0, 0.5])

        def get_params(self, deep=True):
            return {}

    model = NonNegativeRegressor(Negative()).fit(None, None)

    assert model.predict(None).tolist() == [0.0, 0.5]


def test_metrics_leave_wape_undefined_when_demand_is_zero():
    defined = metrics([0, 2, 2], [1, 1, 2], "val_")
    undefined = metrics([0, 0], [1, 0], "val_")

    assert defined["val_mae"] == pytest.approx(2 / 3)
    assert defined["val_wape"] == pytest.approx(2 / 4)
    assert "val_wape" not in undefined
    assert undefined["val_mae"] == pytest.approx(0.5)


def test_split_masks_respect_cutoffs_and_warmup():
    hours = pd.to_datetime(
        [
            "2024-03-01 08:00",  # plain train row
            "2024-09-30 23:00",  # train, but its label closes after the 23:40 fit cut-off
            "2024-10-15 08:00",  # validation
            "2024-11-30 23:00",  # validation, label closes after the selection cut-off
            "2024-12-10 08:00",  # test
        ]
    )
    X, keys, _ = make_prepared(hours, stations=(1,))
    warm = X.copy()
    warm.loc[0, "departures_lag_24"] = np.nan  # warm-up row without history

    masks = split_masks(X, keys)
    warm_masks = split_masks(warm, keys)

    assert masks["train"].tolist() == [True, False, False, False, False]
    assert masks["validation"].tolist() == [False, False, True, False, False]
    assert masks["test"].tolist() == [False, False, False, False, True]
    assert not warm_masks["train"].iloc[0]


def test_split_ranges_follow_the_rows_that_entered_each_split():
    hours = pd.to_datetime(
        [
            "2024-01-01 05:00",  # warm-up row without history, so train starts on the next one
            "2024-03-01 08:00",
            "2024-09-30 22:00",  # last train row: its label closes at 23:00, before 23:40
            "2024-09-30 23:00",  # excluded, its label closes after the fit cut-off
            "2024-10-15 08:00",
            "2024-11-30 22:00",
            "2024-12-10 08:00",
            "2024-12-31 23:00",
        ]
    )
    X, keys, _ = make_prepared(hours, stations=(1,))
    X.loc[0, "departures_lag_24"] = np.nan

    ranges = split_ranges(keys, split_masks(X, keys))

    assert ranges == {
        "train_start": "2024-03-01 08:00:00",
        "train_end": "2024-09-30 22:00:00",
        "val_start": "2024-10-15 08:00:00",
        "val_end": "2024-11-30 22:00:00",
        "test_start": "2024-12-10 08:00:00",
        "test_end": "2024-12-31 23:00:00",
    }


def test_logged_params_match_the_trained_estimator():
    X, _, y = make_prepared(pd.date_range("2024-03-01", periods=48, freq="h"))

    for name, (build, params) in candidates(seed=7).items():
        if params["model"] == "seasonal_mean":
            continue
        model = build(X).fit(X, y)
        estimator = model.estimator_.steps[-1][1]
        # Every logged hyperparameter is the one the fitted estimator really used
        for key, value in estimator.get_params().items():
            assert params[key] == value, f"{name}: {key}"


def test_cli_trains_three_candidates_and_tests_only_the_selected(tmp_path, monkeypatch):
    hours = pd.date_range("2024-09-20", "2024-12-05", freq="6h")
    X, keys, y = make_prepared(hours)
    folder = tmp_path / "p1"
    folder.mkdir()
    X.to_parquet(folder / "X.parquet", index=False)
    keys.to_parquet(folder / "keys.parquet", index=False)
    y.to_frame().to_parquet(folder / "y.parquet", index=False)
    manifest = {"feature_version": FEATURE_VERSION, "data_version": "synthetic"}
    (folder / "features_manifest.json").write_text(json.dumps(manifest))
    # MLflow writes artifacts to ./mlruns, so run from tmp_path to keep the repo folder clean
    monkeypatch.chdir(tmp_path)
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(train_regression, "data_version", lambda: "synthetic")

    try:
        result = CliRunner().invoke(
            train_regression.app, ["regression", "--features-dir", str(folder)]
        )
        assert result.exit_code == 0, result.output + str(result.exception)
        runs = mlflow.search_runs(experiment_names=["mibici-equipo05"])
    finally:
        mlflow.set_tracking_uri(previous_uri)

    assert len(runs) == 3
    assert runs["metrics.val_wape"].notna().all()
    # Every run records the hours that really entered each split
    for column in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end"):
        assert runs[f"params.{column}"].notna().all()
    assert (runs["params.train_end"] < runs["params.val_start"]).all()
    # The commit is tagged by us, not only by MLflow's own source detection
    assert runs["tags.git_commit"].str.len().eq(40).all()
    selected = runs["tags.selected"].eq("true")
    assert selected.sum() == 1
    # Only the selected candidate was scored on the December test
    assert runs.loc[selected, "metrics.test_wape"].notna().all()
    assert runs.loc[~selected, "metrics.test_wape"].isna().all()
