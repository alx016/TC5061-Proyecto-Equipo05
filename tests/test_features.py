"""Behavioral contracts: exact hours, point-in-time invariance and recipe isolation."""

import json

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from typer.testing import CliRunner

from tc5061_proyecto_equipo5.features import (
    CLASSIFICATION_COLUMNS,
    HOLIDAYS_2024,
    HOLIDAYS_BY_YEAR,
    add_calendar,
    add_lags,
    add_neighbors,
    add_split,
    app,
    build_classification_panel,
    build_demand_features,
    build_feature_table,
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
        args = ["main", "--problem", problem, *base]
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
    missing = runner.invoke(app, ["main", "--problem", "p1", *base])
    assert missing.exit_code != 0 and "Missing monthly coverage" in missing.output
    bad = runner.invoke(
        app, ["main", "--problem", "p2", *base, "--catalog-known-at", "2026-09-01"]
    )
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


# ---------------------------------------------------------------------------
# Entrega 5: 2025 current table, challenger table and per-year calendar
# ---------------------------------------------------------------------------


def _catalog(station_ids):
    return pd.DataFrame(
        {
            "station_id": station_ids,
            "capacity": [10] * len(station_ids),
            "latitude": [20.0 + 0.01 * index for index in range(len(station_ids))],
            "longitude": [-103.0 - 0.01 * index for index in range(len(station_ids))],
        }
    )


def _monthly_trips(year, station_ids):
    rows = []
    viaje_id = 0
    for station in station_ids:
        for month in range(1, 13):
            start = pd.Timestamp(year=year, month=month, day=1, hour=6)
            viaje_id += 1
            rows.append(
                {
                    "viaje_id": viaje_id,
                    "inicio_viaje": start,
                    "fin_viaje": start + pd.Timedelta(minutes=10),
                    "origen_id": station,
                    "destino_id": station_ids[0],
                }
            )
    return pd.DataFrame(rows)


def test_holidays_by_year():
    hours = pd.to_datetime(
        [
            "2024-02-05 12:00",
            "2025-02-03 12:00",
            "2026-02-02 12:00",
            "2025-10-01 12:00",
            "2024-10-01 12:00",
        ]
    )
    frame = pd.DataFrame({"station_id": 1, "hour": hours})
    flags = add_calendar(frame)["is_holiday"].tolist()
    assert flags == [True, True, True, False, True]
    assert set(HOLIDAYS_BY_YEAR) == {2024, 2025, 2026}
    # The 2024 transfer of the Federal Executive is not repeated in 2025 or 2026.
    assert pd.Timestamp("2024-10-01") in HOLIDAYS_2024
    assert pd.Timestamp("2025-10-01") not in HOLIDAYS_BY_YEAR[2025]
    unknown = pd.DataFrame({"station_id": 1, "hour": pd.to_datetime(["2027-01-01"])})
    with pytest.raises(ValueError, match="No holiday list"):
        add_calendar(unknown)


def test_current_split_uses_target_hour():
    hours = pd.to_datetime(
        [
            "2025-09-30 21:00",
            "2025-09-30 22:00",
            "2025-09-30 23:00",
            "2025-10-01 00:00",
            "2025-12-31 23:00",
        ]
    )
    frame = pd.DataFrame(
        {"station_id": 1, "hour": hours, "target": pd.Series([0, 0, 0, 0, pd.NA], dtype="Int8")}
    )
    result = add_split(
        frame,
        boundaries=("2025-01-01", "2025-10-01", "2026-01-01"),
        labels=("current", "evaluation"),
    )
    assert result["split"].dtype == "string"
    assert result["split"].tolist()[:4] == ["current", "current", "evaluation", "evaluation"]
    assert pd.isna(result["split"].iloc[4])
    # The 23:00 row is evaluation because its target hour is 2025-10-01 00:00.
    assert result.loc[2, "hour"] + pd.Timedelta(hours=1) == pd.Timestamp("2025-10-01")


def test_challenger_split_and_training_masks():
    hours = pd.to_datetime(
        [
            "2025-07-31 21:00",
            "2025-07-31 22:00",
            "2025-07-31 23:00",
            "2025-08-01 00:00",
            "2025-09-30 22:00",
            "2025-09-30 23:00",
            "2025-10-01 00:00",
        ]
    )
    frame = pd.DataFrame(
        {"station_id": 1, "hour": hours, "target": pd.Series(0, index=range(7), dtype="Int8")}
    )
    result = add_split(
        frame,
        boundaries=("2024-01-01", "2025-08-01", "2025-10-01", "2026-01-01"),
        labels=("train", "validation", "test"),
    )
    assert result["split"].tolist() == [
        "train",
        "train",
        "validation",
        "validation",
        "validation",
        "test",
        "test",
    ]
    keys = result[["hour", "split"]].rename(columns={"hour": "target_time"})
    keys["target_time"] = keys["target_time"] + pd.Timedelta(hours=1)
    keys["label_available_at"] = keys["target_time"] + pd.Timedelta(hours=1)

    train = training_mask(keys, fit_as_of="2025-07-31 23:40", splits=("train",))
    assert keys.loc[train, "split"].eq("train").all()
    assert keys.loc[train, "label_available_at"].le(pd.Timestamp("2025-07-31 23:40")).all()
    assert not keys.loc[train, "target_time"].ge(pd.Timestamp("2025-10-01")).any()
    assert train.tolist() == [True, False, False, False, False, False, False]

    validation = training_mask(keys, fit_as_of="2025-09-30 23:40", splits=("validation",))
    assert keys.loc[validation, "split"].eq("validation").all()
    # Only the August rows are selectable: the 2025-09-30 labels close after the cutoff.
    assert validation.tolist() == [False, False, True, True, False, False, False]
    with pytest.raises(ValueError, match="never test"):
        training_mask(keys, fit_as_of="2025-09-30 23:40", splits=("test",))


def test_continuous_lags_cross_year_boundary():
    hours = pd.date_range("2024-12-31 00:00", "2025-01-01 01:00", freq="h")
    rows = []
    for station_id, offset in ((1, 0.0), (2, 100.0)):
        for position, hour in enumerate(hours):
            rows.append(
                {
                    "station_id": station_id,
                    "hour": hour,
                    "departures": 0,
                    "arrivals": 0,
                    "net_flow": offset + position,
                    "latitude": 20.0 + 0.01 * station_id,
                    "longitude": -103.0 - 0.01 * station_id,
                }
            )
    frame = pd.DataFrame(rows)
    lagged = add_lags(add_neighbors(frame))
    new_year = lagged[lagged["hour"].eq(pd.Timestamp("2025-01-01 00:00"))].set_index("station_id")
    assert new_year.loc[1, "net_flow_lag24"] == 0.0  # 2024-12-31 00:00, station 1
    assert new_year.loc[2, "net_flow_lag24"] == 100.0
    # Neighbour flow is the other station one hour earlier: 2024-12-31 23:00.
    assert new_year.loc[1, "neighbor_net_flow"] == 100.0 + 23
    assert new_year.loc[2, "neighbor_net_flow"] == 0.0 + 23


def test_current_table_contract_synthetic(tmp_path):
    station_ids = [1, 2, 3]
    trips = _monthly_trips(2025, station_ids)
    catalog = _catalog(station_ids).astype({"capacity": "int32"})
    frame = build_feature_table(
        trips=trips,
        stations=catalog,
        keep_station_ids=None,
        start="2025-01-01",
        end="2026-01-01",
        split_boundaries=("2025-01-01", "2025-10-01", "2026-01-01"),
        split_labels=("current", "evaluation"),
        panel_path=tmp_path / "panel.parquet",
        features_path=tmp_path / "features.parquet",
    )
    assert list(frame.columns) == CLASSIFICATION_COLUMNS
    expected_dtypes = {
        "station_id": "int64",
        "hour": "datetime64[us]",
        "departures": "int32",
        "arrivals": "int32",
        "capacity": "int32",
        "day_of_week": "int8",
        "is_weekend": "bool",
        "is_holiday": "bool",
        "target": "Int8",
        "split": "string",
    }
    expected_dtypes.update(
        {column: "float64" for column in CLASSIFICATION_COLUMNS if column not in expected_dtypes}
    )
    assert {column: str(dtype) for column, dtype in frame.dtypes.items()} == expected_dtypes
    assert set(frame["split"].dropna().unique()) == {"current", "evaluation"}
    assert frame["split"].isna().sum() == len(station_ids)
    current = frame["split"].eq("current")
    target_time = frame["hour"] + pd.Timedelta(hours=1)
    assert not target_time[current].ge(pd.Timestamp("2025-10-01")).any()
    assert frame.loc[current, "target"].notna().all()
    assert (target_time[frame["split"].eq("evaluation")] >= pd.Timestamp("2025-10-01")).all()
    assert (target_time[frame["split"].eq("evaluation")] < pd.Timestamp("2026-01-01")).all()
    # The panel is dense and the table round-trips through Parquet.
    assert len(frame) == len(station_ids) * 8760
    assert_frame_equal(pd.read_parquet(tmp_path / "features.parquet"), frame)
    with pytest.raises(ValueError, match="keep_station_ids"):
        build_classification_panel(
            pd.concat([trips, _monthly_trips(2024, station_ids)], ignore_index=True),
            _catalog(station_ids),
            start="2024-01-01",
            end="2026-01-01",
        )


def test_2024_default_behavior_unchanged():
    hours = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    frame = pd.DataFrame({"station_id": 1, "hour": hours})
    target = pd.Series(0, index=frame.index, dtype="Int8")
    target.iloc[-1] = pd.NA
    frame["target"] = target
    result = add_split(frame)
    target_time = frame["hour"] + pd.Timedelta(hours=1)
    expected = pd.Series(pd.NA, index=frame.index, dtype="string")
    in_year = target_time.dt.year.eq(2024)
    month = target_time.dt.month
    expected[in_year & month.isin((1, 2, 3, 4, 5, 6, 7, 8, 9))] = "train"
    expected[in_year & month.isin((10, 11))] = "validation"
    expected[in_year & month.isin((12,))] = "test"
    assert_frame_equal(result[["split"]], expected.to_frame("split"))
    calendar = add_calendar(frame)
    assert calendar["is_holiday"].equals(calendar["hour"].dt.normalize().isin(HOLIDAYS_2024))
    assert calendar["is_holiday"].dtype == bool


def test_legacy_split_uses_requested_year():
    """The legacy annual builder retains its train/validation/test contract for other years."""
    frame = pd.DataFrame(
        {
            "hour": pd.to_datetime(["2025-06-01", "2025-10-01", "2025-12-01"]),
            "target": pd.Series([0, 0, 0], dtype="Int8"),
        }
    )
    assert add_split(frame, study_year=2025)["split"].tolist() == ["train", "validation", "test"]


def test_current_cli_without_2024_reference(tmp_path, monkeypatch):
    from tc5061_proyecto_equipo5 import features

    trips = _monthly_trips(2025, [1, 2, 3])
    monkeypatch.setattr(features, "CLASSIFICATION_FEATURES_PATH", tmp_path / "missing.parquet")

    def load_trips(year, trips_dir):
        if year == 2024:
            raise FileNotFoundError("No 2024 trips")
        return trips

    monkeypatch.setattr(features, "load_trips", load_trips)
    monkeypatch.setattr(features, "load_stations", lambda _: _catalog([1, 2, 3]))
    output = tmp_path / "current.parquet"
    result = CliRunner().invoke(
        app,
        [
            "current",
            "--trips-dir",
            str(tmp_path),
            "--stations-dir",
            str(tmp_path),
            "--panel-path",
            str(tmp_path / "panel.parquet"),
            "--features-path",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert len(pd.read_parquet(output)) == 3 * 8760


def test_missing_target_has_no_split_and_cross_year_requires_station_universe():
    frame = pd.DataFrame(
        {"hour": pd.to_datetime(["2025-02-01"]), "target": pd.Series([pd.NA], dtype="Int8")}
    )
    assert pd.isna(add_split(frame, 2025)["split"].iloc[0])
    assert pd.isna(
        add_split(frame, boundaries=("2025-01-01", "2026-01-01"), labels=("current",))[
            "split"
        ].iloc[0]
    )
    with pytest.raises(ValueError, match="keep_station_ids"):
        build_classification_panel(
            _monthly_trips(2025, [1, 2]), _catalog([1, 2]), start="2025-12-01", end="2026-02-01"
        )
