"""EDA-grounded feature contracts through builders, transformers and generation CLI."""

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from tc5061_proyecto_equipo5.features import build_demand_features


def hourly():
    times = pd.date_range("2024-09-25", "2024-10-04", freq="h", inclusive="left")
    frame = pd.MultiIndex.from_product(
        [[1, 2, 3], times], names=["station_id", "target_time"]
    ).to_frame(index=False)
    frame["departures"] = np.arange(len(frame)) % 9
    frame["arrivals"] = np.arange(len(frame)) % 7
    return frame


def test_extended_calendar_and_log_features_are_causal():
    panel = hourly()
    before = panel.copy(deep=True)
    result = build_demand_features(panel, extended=True)
    assert {
        "month_sin",
        "month_cos",
        "hour",
        "day_of_week",
        "hour_sin_weekend",
        "log1p_departures_lag_2",
    }.issubset(result.X.columns)
    assert result.X.loc[10, "log1p_departures_lag_2"] == pytest.approx(
        np.log1p(panel.departures.iloc[8])
    )
    assert_frame_equal(panel, before)
    changed = panel.copy()
    cut = pd.Timestamp("2024-10-02 10:00")
    changed.loc[changed.target_time.ge(cut - pd.Timedelta(hours=1)), "departures"] = 999
    rows = result.keys.target_time.le(cut)
    assert_frame_equal(
        result.X.loc[rows], build_demand_features(changed, extended=True).X.loc[rows]
    )


def test_external_calendar_is_not_part_of_feature_contract():
    import inspect

    from tc5061_proyecto_equipo5.features import (
        build_risk_features,
        main,
        prepare_feature_dataset,
    )

    for function in (build_demand_features, build_risk_features, main, prepare_feature_dataset):
        assert not any(
            "holiday" in name or "calendar" in name
            for name in inspect.signature(function).parameters
        )
    result = build_demand_features(hourly(), extended=True)
    forbidden = {
        "is_known_holiday",
        "is_known_bank_holiday",
        "is_operational_calendar_day",
        "is_weekday_regular",
    }
    assert forbidden.isdisjoint(result.X.columns)


def test_spatial_features_use_valid_metadata_and_past_neighbor_flows():
    stations = pd.DataFrame(
        {
            "station_id": [1, 2, 3],
            "capacity": [10, 20, 10],
            "latitude": [20.0, 20.001, 20.1],
            "longitude": [-103.0] * 3,
            "municipality": ["A", "A", "B"],
        }
    )
    result = build_demand_features(
        hourly(), extended=True, stations=stations, catalog_known_at="2023-01-01"
    )
    assert {
        "capacity",
        "latitude",
        "longitude",
        "municipality",
        "neighbor_count_500m",
        "neighbor_net_ratio_lag_2",
    }.issubset(result.X.columns)
    assert result.X.loc[0, "neighbor_count_500m"] == 1
    assert result.X.loc[0, "capacity"] == 10
    assert np.isnan(result.X.loc[0, "neighbor_net_ratio_lag_2"])
    with pytest.raises(ValueError, match="known"):
        build_demand_features(
            hourly(), extended=True, stations=stations, catalog_known_at="2026-09-01"
        )


def test_preprocessor_learns_missing_scaling_encoding_and_selection_only_on_train():
    from tc5061_proyecto_equipo5.features import make_preprocessor

    train = pd.DataFrame(
        {
            "station_id": ["1", "2", "1", "2"],
            "municipality": ["A", None, "A", "B"],
            "count": [0.0, 2.0, np.nan, 100.0],
            "constant": [5.0] * 4,
        }
    )
    proc = make_preprocessor(
        train, numeric_scaling="robust", select_variance=True, missing_indicators=True
    )
    proc.fit(train)
    before = (
        proc.transform(train).toarray()
        if hasattr(proc.transform(train), "toarray")
        else proc.transform(train)
    )
    new = pd.DataFrame(
        {
            "station_id": ["future"],
            "municipality": ["unseen"],
            "count": [100000.0],
            "constant": [999.0],
        }
    )
    assert proc.transform(new).shape[1] == before.shape[1]
    after = (
        proc.transform(train).toarray()
        if hasattr(proc.transform(train), "toarray")
        else proc.transform(train)
    )
    assert np.allclose(before, after)
    assert not any("constant" in name for name in proc.get_feature_names_out())


def test_ready_cli_needs_no_future_station_catalog_for_demand(tmp_path):
    from typer.testing import CliRunner

    from tc5061_proyecto_equipo5.features import app

    root = tmp_path / "input" / "year=2024"
    root.mkdir(parents=True)
    for month in range(1, 13):
        start = pd.Timestamp(2024, month, 1)
        times = pd.date_range(start, periods=200, freq="h")
        pd.DataFrame(
            {
                "viaje_id": np.arange(200) + month * 1000,
                "inicio_viaje": times,
                "fin_viaje": times + pd.Timedelta(minutes=15),
                "origen_id": np.arange(200) % 2 + 1,
                "destino_id": np.arange(200) % 2 + 1,
            }
        ).to_parquet(root / f"viajes_2024_{month:02}.parquet", index=False)
    result = CliRunner().invoke(
        app,
        [
            "--problem",
            "p1",
            "--prepare-ready",
            "--input-dir",
            str(root.parent),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    import json

    target = tmp_path / "out" / "p1"
    receipt = json.loads((target / "features_manifest.json").read_text())
    assert receipt["feature_version"] == "4"
    assert "calendar" not in receipt["input_sha256"]
    assert not any("holiday" in c or "operational_calendar" in c for c in receipt["columns"])
    assert receipt["prepared"]["fit_as_of"] < "2024-09-30 23:50:00"
    assert (target / "preprocessor.pkl").exists()
    assert receipt["prepared"]["splits"]["test"]["rows"] > 0
    assert receipt["quality"]["source_rows_2024"] == 2400
    assert "capacity" in receipt["unavailable_features"]
    assert len(receipt["prepared"]["selected_columns"]) > 0


def test_future_neighbor_activity_cannot_change_current_features():
    from tc5061_proyecto_equipo5.features import build_risk_features

    stations = pd.DataFrame(
        {
            "station_id": [1, 2, 3],
            "capacity": [10, 20, 10],
            "latitude": [20.0, 20.001, 20.1],
            "longitude": [-103.0] * 3,
        }
    )
    data = hourly()
    original = build_risk_features(data, stations, catalog_known_at="2023-01-01", extended=True)
    cutoff = pd.Timestamp("2024-10-02 10:00")
    data.loc[data.target_time.ge(cutoff - pd.Timedelta(hours=1)), ["departures", "arrivals"]] = [
        200,
        400,
    ]
    changed = build_risk_features(data, stations, catalog_known_at="2023-01-01", extended=True)
    rows = original.keys.target_time.le(cutoff)
    assert_frame_equal(original.X.loc[rows], changed.X.loc[rows])


@pytest.mark.parametrize(
    "bad", [{"latitude": 100.0}, {"valid_from": "2024-10-01"}, {"valid_to": "2024-10-01"}]
)
def test_invalid_spatial_metadata_is_rejected(bad):
    stations = pd.DataFrame(
        {
            "station_id": [1, 2, 3],
            "capacity": [10, 20, 10],
            "latitude": [20.0, 20.001, 20.1],
            "longitude": [-103.0] * 3,
        }
    )
    for col, value in bad.items():
        stations[col] = value
    with pytest.raises(ValueError):
        build_demand_features(hourly(), stations=stations, catalog_known_at="2023-01-01")


def test_metadata_failure_does_not_publish_partial_dataset(tmp_path):
    from tc5061_proyecto_equipo5.features import prepare_feature_dataset

    with pytest.raises(ValueError, match="historical capacity"):
        prepare_feature_dataset("p2", tmp_path / "missing", tmp_path / "output")
    assert not (tmp_path / "output" / "p2").exists()


def test_constant_profile_does_not_fabricate_ready_training_columns(tmp_path):
    from tc5061_proyecto_equipo5.features import make_preprocessor

    frame = pd.DataFrame({"x": [1.0, 1.0], "y": [0.0, 0.0]})
    with pytest.raises(ValueError):
        make_preprocessor(frame, select_variance=True).fit(frame)


def test_capacity_only_metadata_checks_effective_dates():
    from tc5061_proyecto_equipo5.features import build_risk_features

    catalog = pd.DataFrame(
        {"station_id": [1, 2, 3], "capacity": [10, 20, 10], "valid_from": ["2026-01-01"] * 3}
    )
    with pytest.raises(ValueError, match="cover"):
        build_risk_features(hourly(), catalog, catalog_known_at="2023-01-01", extended=True)


def test_future_first_observation_cannot_change_historical_spatial_universe(tmp_path):
    from tc5061_proyecto_equipo5.features import prepare_feature_dataset

    source = tmp_path / "input" / "year=2024"
    source.mkdir(parents=True)
    for month in range(1, 13):
        times = pd.date_range(pd.Timestamp(2024, month, 1), periods=200, freq="h")
        pd.DataFrame(
            {
                "viaje_id": np.arange(200) + month * 1000,
                "inicio_viaje": times,
                "fin_viaje": times + pd.Timedelta(minutes=15),
                "origen_id": np.arange(200) % 2 + 1,
                "destino_id": np.arange(200) % 2 + 1,
            }
        ).to_parquet(source / f"viajes_2024_{month:02}.parquet", index=False)
    catalog = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "station_id": [1, 2, 3],
            "capacity": [10, 20, 10],
            "latitude": [20.0, 20.001, 20.002],
            "longitude": [-103.0] * 3,
        }
    ).to_csv(catalog, index=False)
    for run in ("before", "after"):
        prepare_feature_dataset(
            "p1",
            source.parent,
            tmp_path / run,
            stations_path=catalog,
            catalog_known_at="2023-01-01",
        )
        if run == "before":
            july = source / "viajes_2024_07.parquet"
            trips = pd.read_parquet(july)
            trips.loc[0, ["origen_id", "destino_id"]] = 3
            trips.to_parquet(july, index=False)

    def earlier(run):
        keys = pd.read_parquet(tmp_path / run / "p1" / "keys.parquet")
        X = pd.read_parquet(tmp_path / run / "p1" / "X.parquet")
        return X.loc[keys.target_time.lt("2024-07-01")].reset_index(drop=True)

    assert_frame_equal(earlier("before"), earlier("after"))
