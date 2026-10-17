"""Behavioral contracts: exact hours, point-in-time invariance and recipe isolation."""

import json

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from typer.testing import CliRunner

from tc5061_proyecto_equipo5.features import (
    app,
    build_demand_features,
    build_hourly_panel,
    build_risk_features,
    build_station_profiles,
    make_preprocessor,
    training_mask,
)


def panel(start="2024-09-20", end="2024-12-02"):
    grid = pd.MultiIndex.from_product(
        [[1, 2], pd.date_range(start, end, freq="h", inclusive="left")],
        names=["station_id", "target_time"],
    ).to_frame(index=False)
    grid["departures"] = np.arange(len(grid)) % 11
    grid["arrivals"] = np.arange(len(grid)) % 7
    return grid


def stations():
    return pd.DataFrame({"station_id": [1, 2], "capacity": [10, 20]})


def risk(data, **kwargs):
    return build_risk_features(data, stations(), catalog_known_at="2023-01-01", **kwargs)


def test_dense_grid_counts_arrivals_at_end_not_start_and_half_open_intervals():
    trips = pd.DataFrame(
        {
            "viaje_id": [1, 2, 3],
            "origen_id": [1, 2, 1],
            "destino_id": [2, 1, 2],
            "inicio_viaje": pd.to_datetime(
                ["2023-12-31 23:50", "2024-01-01 00:15", "2024-01-01 02:00"]
            ),
            "fin_viaje": pd.to_datetime(
                ["2024-01-01 01:01", "2024-01-01 01:00", "2024-01-01 03:00"]
            ),
        }
    )
    result = build_hourly_panel(trips, [1, 2, 3], start="2024-01-01", end="2024-01-01 02:00")
    assert len(result) == 6
    assert result.departures.sum() == 1
    assert result.arrivals.sum() == 2
    assert result.loc[result.station_id.eq(3), ["departures", "arrivals"]].eq(0).all().all()
    assert result.loc[result.target_time.eq(pd.Timestamp("2024-01-01")), "arrivals"].eq(0).all()
    with pytest.raises(ValueError, match="duplicate"):
        build_hourly_panel(pd.concat([trips, trips]), [1, 2], start="2024-01-01", end="2024-01-02")
    with pytest.raises(ValueError, match="absent"):
        build_hourly_panel(trips, [1], start="2024-01-01", end="2024-01-02")


@pytest.mark.parametrize("builder", [build_demand_features, risk])
def test_future_and_unfinished_hour_perturbations_do_not_change_predictors(builder):
    data = panel()
    before = data.copy(deep=True)
    target = pd.Timestamp("2024-10-02 10:00")
    baseline = builder(data)
    mask = data.target_time.ge(target - pd.Timedelta(hours=1))
    data.loc[mask, ["departures", "arrivals"]] = [1000, 2000]
    changed = builder(data)
    rows = baseline.keys.target_time.le(target)
    assert_frame_equal(baseline.X.loc[rows], changed.X.loc[rows])
    assert not baseline.y.equals(changed.y)
    # Recipes are pure and do not mutate the caller's input.
    assert_frame_equal(before, panel())


def test_lags_rolling_and_station_isolation_are_exact():
    data = panel("2024-01-01", "2024-01-10")
    original = data.copy(deep=True)
    result = build_demand_features(data)
    row = 170
    assert result.X.loc[row, "departures_lag_2"] == data.departures.iloc[row - 2]
    assert result.X.loc[row, "departures_lag_168"] == data.departures.iloc[row - 168]
    past = data.departures.iloc[row - 4 : row - 1]
    assert result.X.loc[row, "departures_mean_3"] == pytest.approx(past.mean())
    assert result.X.loc[row, "departures_std_3"] == pytest.approx(past.std())
    assert result.X.loc[216, "departures_lag_2"] != result.X.loc[216, "departures_lag_2"]  # NaN
    assert_frame_equal(data, original)
    assert "demand" not in result.X and "target_time" not in result.X
    custom = build_demand_features(data, lags=(24,), windows=())
    assert "departures_lag_2" not in custom.X
    assert "departures_lag_24" in custom.X
    with pytest.raises(ValueError, match=">= 2"):
        build_demand_features(data, lags=(1,))


def test_sparse_duplicate_negative_and_timezone_panels_fail():
    data = panel()
    with pytest.raises(ValueError, match="every hour"):
        build_demand_features(data.drop(index=10))
    with pytest.raises(ValueError, match="unique"):
        build_demand_features(pd.concat([data, data.iloc[:1]]))
    data.loc[0, "departures"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        build_demand_features(data)
    data = panel()
    data["target_time"] = data.target_time.dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone"):
        build_demand_features(data)


def test_risk_inclusive_thresholds_and_unbounded_pressure():
    data = panel("2024-01-01", "2024-01-02")
    data.loc[:3, "departures"] = [4, 0, 3, 100]
    data.loc[:3, "arrivals"] = [0, 4, 0, 0]
    result = risk(data)
    assert result.y.iloc[:4].tolist() == [
        "Riesgo_Desabasto",
        "Riesgo_Saturacion",
        "Balanceado",
        "Riesgo_Desabasto",
    ]
    assert result.X.loc[5, "net_ratio_lag_2"] == -10
    assert "net_flow" not in result.X and "risk" not in result.X
    assert result.X.loc[0, "capacity"] == 10
    assert result.X.loc[24, "capacity"] == 20
    assert "arrivals_lag_2" not in build_demand_features(data).X


@pytest.mark.parametrize("capacity", [0, -2, float("nan"), float("inf")])
def test_risk_rejects_invalid_capacity(capacity):
    cat = stations()
    cat["capacity"] = cat.capacity.astype(float)
    cat.loc[0, "capacity"] = capacity
    with pytest.raises(ValueError):
        build_risk_features(panel(), cat, catalog_known_at="2023-01-01")


def test_catalog_provenance_missing_ids_and_duplicate_ids_fail_closed():
    data = panel()
    with pytest.raises(ValueError, match="not known"):
        build_risk_features(data, stations(), catalog_known_at="2026-09-01")
    with pytest.raises(ValueError, match="capacity"):
        build_risk_features(data, stations().iloc[:1], catalog_known_at="2023-01-01")
    with pytest.raises(ValueError, match="unique"):
        build_risk_features(
            data, pd.concat([stations(), stations()]), catalog_known_at="2023-01-01"
        )


def test_splits_follow_target_window_and_issue_is_minute_50():
    result = build_demand_features(panel())
    keys = result.keys.set_index(["station_id", "target_time"])
    assert keys.loc[(1, pd.Timestamp("2024-09-30 23:00")), "split"] == "train"
    assert keys.loc[(1, pd.Timestamp("2024-10-01")), "split"] == "validation"
    assert keys.loc[(1, pd.Timestamp("2024-12-01")), "split"] == "test"
    assert keys.loc[(1, pd.Timestamp("2024-10-01")), "issue_time"] == pd.Timestamp(
        "2024-09-30 23:50"
    )


def test_p3_has_49_features_normalizes_and_ignores_future():
    data = panel("2024-01-01", "2024-11-01")
    result = build_station_profiles(data)
    assert result.X.shape == (2, 49)
    for prefix in ("departures_hour_", "arrivals_hour_"):
        assert np.allclose(result.X.filter(like=prefix).sum(axis=1), 1)
    assert result.X.asymmetry.between(-1, 1).all()
    assert result.y is None
    assert result.keys.window_end.eq(pd.Timestamp("2024-10-01")).all()
    data.loc[data.target_time.ge("2024-10-01"), ["departures", "arrivals"]] = 9999
    assert_frame_equal(result.X, build_station_profiles(data).X)


def test_p3_filters_low_activity_in_any_month_and_rejects_missing_coverage():
    data = panel("2024-01-01", "2024-10-01")
    data.loc[data.station_id.eq(2) & data.target_time.dt.month.eq(3), "departures"] = 0
    assert build_station_profiles(data).keys.station_id.tolist() == [1]
    data["arrivals"] = 0
    assert build_station_profiles(data).X.empty
    with pytest.raises(ValueError, match="coverage"):
        build_station_profiles(data.loc[data.target_time.ge("2024-02-01")])
    with pytest.raises(ValueError, match="whole calendar"):
        build_station_profiles(data, start="2024-01-02")


def test_model_preprocessor_does_not_learn_validation_statistics():
    X = pd.DataFrame({"station_id": ["1", "2", "future"], "count": [0.0, 2.0, 1000.0]})
    transformer = make_preprocessor(X)
    assert not hasattr(transformer, "transformers_")
    transformer.fit(X.iloc[:2])
    scaler = transformer.named_transformers_["numeric"].steps[-1][1]
    assert scaler.mean_[0] == 1
    assert transformer.transform(X.iloc[2:]).shape[0] == 1
    unscaled = make_preprocessor(X, scale_numeric=False)
    assert len(unscaled.transformers[0][1].steps) == 1


def test_cli_end_to_end_and_deterministic_receipt(tmp_path):
    root = tmp_path / "viajes" / "year=2024"
    root.mkdir(parents=True)
    for month in range(1, 13):
        time = pd.Timestamp(2024, month, 1, 12)
        pd.DataFrame(
            {
                "viaje_id": [month],
                "origen_id": [1],
                "destino_id": [1],
                "inicio_viaje": [time],
                "fin_viaje": [time + pd.Timedelta(minutes=15)],
            }
        ).to_parquet(root / f"viajes_2024_{month:02}.parquet", index=False)
    cat = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": [1], "capacity": [10]}).to_csv(cat, index=False)
    runner = CliRunner()
    base = [
        "--stations-path",
        str(cat),
        "--catalog-known-at",
        "2023-01-01",
        "--input-dir",
        str(root.parent),
        "--output-dir",
        str(tmp_path / "processed"),
    ]
    for problem in ("p1", "p2", "p3"):
        args = ["--problem", problem, *base]
        first = runner.invoke(app, args)
        assert first.exit_code == 0, first.output + str(first.exception)
        target = tmp_path / "processed" / problem
        receipt = (target / "features_manifest.json").read_bytes()
        assert runner.invoke(app, args).exit_code == 0
        assert (target / "features_manifest.json").read_bytes() == receipt
        meta = json.loads(receipt)
        assert len(meta["input_sha256"]) == 13
        assert meta["feature_version"] == "4"
        if problem != "p3":
            keys = pd.read_parquet(target / "keys.parquet")
            assert keys.label_available_at.eq(keys.target_time + pd.Timedelta(hours=1)).all()
        assert (target / "X.parquet").is_file() and (target / "keys.parquet").is_file()
        assert (target / "y.parquet").is_file() == (problem != "p3")
    (root / "viajes_2024_12.parquet").unlink()
    missing = runner.invoke(app, ["--problem", "p1", *base])
    assert missing.exit_code != 0 and "Missing monthly coverage" in missing.output
    bad = runner.invoke(app, ["--problem", "p2", *base, "--catalog-known-at", "2026-09-01"])
    assert bad.exit_code != 0 and "catalog" in bad.output


def dense(start, end):
    times = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DataFrame({"station_id": 1, "target_time": times, "departures": 4, "arrivals": 0})


@pytest.mark.parametrize(
    "boundary,next_split", [("2024-10-01", "validation"), ("2024-12-01", "test")]
)
def test_all_prior_split_labels_are_available_before_first_next_prediction(boundary, next_split):
    boundary = pd.Timestamp(boundary)
    result = build_demand_features(
        dense(boundary - pd.Timedelta(days=2), boundary + pd.Timedelta(days=1))
    )
    evaluation_issue = result.keys.loc[result.keys["split"].eq(next_split), "issue_time"].min()
    previous_split = "train" if next_split == "validation" else "validation"
    eligible = training_mask(result.keys, fit_as_of=evaluation_issue, splits=(previous_split,))
    assert eligible.any()
    assert result.keys.loc[eligible, "label_available_at"].le(evaluation_issue).all()
    last_previous = result.keys.target_time.eq(boundary - pd.Timedelta(hours=1))
    assert not eligible.loc[last_previous].any()
    assert result.keys.loc[last_previous, "split"].eq(previous_split).all()
    known_previous = result.keys.target_time.eq(boundary - pd.Timedelta(hours=2))
    assert eligible.loc[known_previous].all()


@pytest.mark.parametrize("column", ["inicio_viaje", "fin_viaje"])
def test_datetime_strings_parsing_to_nat_are_rejected(column):
    trip = {
        "viaje_id": [1],
        "origen_id": [1],
        "destino_id": [1],
        "inicio_viaje": ["2024-01-01 01:00"],
        "fin_viaje": ["2024-01-01 01:10"],
    }
    trip[column] = ["NaT"]
    with pytest.raises(ValueError):
        build_hourly_panel(pd.DataFrame(trip), [1], start="2024-01-01", end="2024-01-02")


def test_valid_unsigned_counts_do_not_reverse_risk_direction():
    data = dense("2024-01-01", "2024-01-03")
    data[["departures", "arrivals"]] = data[["departures", "arrivals"]].astype("uint32")
    result = build_risk_features(
        data, pd.DataFrame({"station_id": [1], "capacity": [10]}), catalog_known_at="2023-01-01"
    )
    assert result.y.eq("Riesgo_Desabasto").all(), result.y.unique().tolist()
    assert result.X.loc[2, "net_ratio_lag_2"] == -0.4


def test_valid_unsigned_counts_preserve_negative_profile_asymmetry():
    data = dense("2024-01-01", "2024-02-01")
    data["arrivals"] = 1
    data[["departures", "arrivals"]] = data[["departures", "arrivals"]].astype("uint32")
    result = build_station_profiles(data, start="2024-01-01", end="2024-02-01")
    assert result.X.loc[0, "asymmetry"] == pytest.approx(-0.6)


@pytest.mark.parametrize("builder", [build_demand_features, risk])
def test_counts_outside_safe_aggregation_range_are_rejected(builder):
    data = dense("2024-01-01", "2024-01-03")
    data["departures"] = np.uint64(2**63)
    with pytest.raises(ValueError, match="safe signed"):
        builder(data)


def test_training_mask_allows_only_available_labels_in_permitted_splits():
    result = build_demand_features(panel())
    keys = result.keys.copy(deep=True)
    fit_as_of = "2024-11-30 23:50"
    eligible = training_mask(keys, fit_as_of=fit_as_of, splits=("train", "validation"))
    assert keys.loc[eligible, "split"].isin(["train", "validation"]).all()
    assert keys.loc[eligible, "label_available_at"].le(pd.Timestamp(fit_as_of)).all()
    assert not eligible.loc[keys["split"].eq("test")].any()
    assert eligible.loc[keys["split"].eq("train")].all()
    assert_frame_equal(keys, result.keys)
    with pytest.raises(ValueError, match="never test"):
        training_mask(keys, fit_as_of=fit_as_of, splits=("test",))


def test_training_mask_honors_delayed_labels_and_rejects_backdating():
    keys = build_demand_features(dense("2024-09-30 22:00", "2024-10-01 01:00")).keys
    keys.loc[0, "label_available_at"] = pd.Timestamp("2024-10-01 00:05")
    mask = training_mask(keys, fit_as_of="2024-09-30 23:50")
    assert not mask.any()  # Even the older target has not been published yet.
    keys.loc[0, "label_available_at"] = pd.Timestamp("2024-09-30 22:59")
    with pytest.raises(ValueError, match="before their target"):
        training_mask(keys, fit_as_of="2024-09-30 23:50")


@pytest.mark.parametrize("cutoff", ["NaT", "2024-09-30T23:50:00Z"])
def test_training_cutoff_requires_nonnull_local_time(cutoff):
    keys = build_demand_features(dense("2024-09-30", "2024-10-02")).keys
    with pytest.raises(ValueError, match="timezone-naive"):
        training_mask(keys, fit_as_of=cutoff)


def test_future_training_label_cannot_change_first_validation_prediction():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    data = dense("2024-09-27", "2024-10-02")
    changed = data.copy(deep=True)
    # One extra group of departures in the label bucket still open at 23:50.
    changed.loc[changed.target_time.eq("2024-09-30 23:00"), "departures"] = 1004

    def predict_first(frame, *, purge):
        features = build_demand_features(frame, lags=(2,), windows=())
        val = features.keys["split"].eq("validation")
        first = features.keys.loc[val].index[:1]
        cutoff = features.keys.loc[first, "issue_time"].iloc[0]
        eligible = (
            training_mask(features.keys, fit_as_of=cutoff)
            if purge
            else features.keys["split"].eq("train")
        )
        model = make_pipeline(make_preprocessor(features.X), Ridge(solver="lsqr"))
        model.fit(features.X.loc[eligible], features.y.loc[eligible])
        return model.predict(features.X.loc[first])[0]

    assert predict_first(data, purge=True) == pytest.approx(predict_first(changed, purge=True))
    # Positive control: this test would detect the original documentation's leaky fit.
    assert predict_first(data, purge=False) != pytest.approx(predict_first(changed, purge=False))
