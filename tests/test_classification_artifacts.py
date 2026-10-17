"""Unit tests for the P2 classification artifacts built on top of the shared hourly panel.

All tests use small synthetic DataFrames, so they do not need the real dataset.
"""

import pandas as pd
import pytest

from tc5061_proyecto_equipo5.features import (
    add_calendar,
    add_lags,
    add_neighbors,
    add_split,
    add_target,
    build_classification_panel,
    full_year_station_ids,
    load_stations,
)

YEAR = 2024
STATION_HOURS = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31 23:00", freq="h")


def make_stations():
    """Return a tiny station catalog with two stations."""
    return pd.DataFrame(
        {
            "station_id": [1, 2],
            "name": ["A", "B"],
            "latitude": [19.43, 19.44],
            "longitude": [-99.13, -99.14],
            "capacity": [10, 20],
        }
    )


def add_monthly_activity(rows, trip_id, station_ids=(1, 2)):
    """Append one self-loop trip per month per station, so the stations last the whole year."""
    for month in range(1, 13):
        for station_id in station_ids:
            start = pd.Timestamp(year=YEAR, month=month, day=15, hour=3)
            rows.append(
                {
                    "viaje_id": trip_id,
                    "inicio_viaje": start,
                    "fin_viaje": start + pd.Timedelta(minutes=20),
                    "origen_id": station_id,
                    "destino_id": station_id,
                }
            )
            trip_id += 1
    return trip_id


def make_trips():
    """Return two base trips plus one trip per month, so both stations last the whole year."""
    rows = [
        {
            "viaje_id": 1,
            "inicio_viaje": pd.Timestamp("2024-01-01 00:10"),
            "fin_viaje": pd.Timestamp("2024-01-01 00:30"),
            "origen_id": 1,
            "destino_id": 2,
        },
        {
            "viaje_id": 2,
            "inicio_viaje": pd.Timestamp("2024-01-01 01:20"),
            "fin_viaje": pd.Timestamp("2024-01-01 01:40"),
            "origen_id": 2,
            "destino_id": 1,
        },
    ]
    add_monthly_activity(rows, 3)
    return pd.DataFrame(rows)


def make_flow_panel(hours, flows_by_station):
    """Build a minimal panel from {station_id: [net_flow values]}."""
    rows = []
    for station_id, flows in flows_by_station.items():
        for hour, flow in zip(hours, flows, strict=True):
            rows.append(
                {
                    "station_id": station_id,
                    "hour": hour,
                    "arrivals": 0,
                    "departures": 0,
                    "net_flow": flow,
                    "capacity": 10,
                    "latitude": 19.43 + station_id / 100,
                    "longitude": -99.13 - station_id / 100,
                }
            )
    return pd.DataFrame(rows)


def test_partial_stations_are_dropped():
    # Station 1 moves bikes every month; station 2 only appears in January.
    rows = [
        {
            "viaje_id": 1,
            "inicio_viaje": pd.Timestamp("2024-01-20 05:00"),
            "fin_viaje": pd.Timestamp("2024-01-20 05:20"),
            "origen_id": 2,
            "destino_id": 1,
        }
    ]
    add_monthly_activity(rows, 2, station_ids=(1,))
    trips = pd.DataFrame(rows)

    ids = full_year_station_ids(trips)

    assert ids.tolist() == [1]


def test_panel_full_grid():
    panel = build_classification_panel(make_trips(), make_stations(), YEAR)

    # Every station must have one row for every hour of the whole year.
    assert len(panel) == 2 * len(STATION_HOURS)
    assert panel["departures"].notna().all()
    assert panel["arrivals"].notna().all()
    # Two base trips plus one trip per month per station (2 x 12) create activity.
    assert panel["departures"].sum() == 26
    assert panel["arrivals"].sum() == 26
    quiet = panel[(panel["station_id"] == 1) & (panel["hour"] == pd.Timestamp("2024-06-15 12:00"))]
    assert quiet["departures"].item() == 0
    assert quiet["arrivals"].item() == 0


def test_net_flow_formula():
    # Station 1 has capacity 10, 3 departures and 7 arrivals in the 08:00 hour.
    rows = []
    trip_id = 1
    for _ in range(3):
        rows.append(
            {
                "viaje_id": trip_id,
                "inicio_viaje": pd.Timestamp("2024-03-01 08:05"),
                "fin_viaje": pd.Timestamp("2024-03-01 08:20"),
                "origen_id": 1,
                "destino_id": 2,
            }
        )
        trip_id += 1
    for _ in range(7):
        rows.append(
            {
                "viaje_id": trip_id,
                "inicio_viaje": pd.Timestamp("2024-03-01 08:10"),
                "fin_viaje": pd.Timestamp("2024-03-01 08:40"),
                "origen_id": 2,
                "destino_id": 1,
            }
        )
        trip_id += 1
    add_monthly_activity(rows, trip_id)

    panel = build_classification_panel(pd.DataFrame(rows), make_stations(), YEAR)
    row = panel[(panel["station_id"] == 1) & (panel["hour"] == pd.Timestamp("2024-03-01 08:00"))]

    assert row["departures"].item() == 3
    assert row["arrivals"].item() == 7
    assert row["net_flow"].item() == pytest.approx((7 - 3) / 10)


def test_lags_use_past_only():
    hours = pd.date_range("2024-01-01 00:00", periods=30, freq="h")
    panel = make_flow_panel(
        hours, {1: [i / 10 for i in range(30)], 2: [100 + i / 10 for i in range(30)]}
    )
    lags = add_lags(panel).set_index(["station_id", "hour"])
    original = panel.set_index(["station_id", "hour"])
    now = hours[27]

    for station_id in (1, 2):
        assert (
            lags.loc[(station_id, now), "net_flow_lag1"]
            == original.loc[(station_id, hours[26]), "net_flow"]
        )
        assert (
            lags.loc[(station_id, now), "net_flow_lag2"]
            == original.loc[(station_id, hours[25]), "net_flow"]
        )
        assert (
            lags.loc[(station_id, now), "net_flow_lag24"]
            == original.loc[(station_id, hours[3]), "net_flow"]
        )
        # The lag must never be the value of the current hour.
        assert (
            lags.loc[(station_id, now), "net_flow_lag1"]
            != original.loc[(station_id, now), "net_flow"]
        )

    # The first hour of each station has no past, so it cannot borrow from another station.
    assert pd.isna(lags.loc[(1, hours[0]), "net_flow_lag1"])
    assert pd.isna(lags.loc[(2, hours[0]), "net_flow_lag1"])


def test_target_thresholds():
    net_flows = [-0.60, -0.50, -0.40, 0.0, 0.40, 0.50, 0.0]
    hours = pd.date_range("2024-01-01 00:00", periods=len(net_flows), freq="h")
    panel = make_flow_panel(hours, {1: net_flows})

    targets = add_target(panel)["target"].tolist()

    # target[t] uses net_flow[t+1]: below, exactly -0.40, balanced, exactly +0.40, above, balanced.
    assert targets[:6] == [1, 1, 0, 2, 2, 0]
    assert pd.isna(targets[6])


def test_split_months():
    hours = pd.to_datetime(
        [
            "2024-01-15 12:00",
            "2024-09-30 23:00",
            "2024-10-01 00:00",
            "2024-11-30 23:00",
            "2024-12-01 00:00",
            "2024-12-31 23:00",
        ]
    )
    panel = make_flow_panel(hours, {1: [0.0] * len(hours)})

    split = add_split(panel)["split"].tolist()

    assert split == ["train", "train", "validation", "validation", "test", "test"]


def test_calendar():
    hours = pd.to_datetime(
        [
            "2024-05-01 12:00",  # Wednesday, Mexican holiday
            "2024-05-04 12:00",  # Saturday
            "2024-05-06 12:00",  # Monday, regular day
        ]
    )
    panel = make_flow_panel(hours, {1: [0.0] * len(hours)})

    calendar = add_calendar(panel)

    assert calendar["is_weekend"].tolist() == [False, True, False]
    assert calendar["is_holiday"].tolist() == [True, False, False]
    assert {"hour_sin", "hour_cos", "day_of_week"}.issubset(calendar.columns)


def test_neighbor_is_lagged():
    hours = pd.date_range("2024-01-01 00:00", periods=5, freq="h")
    panel = make_flow_panel(hours, {1: [0.1] * 5, 2: [0.9] * 5})

    neighbors = add_neighbors(panel, k=1).set_index(["station_id", "hour"])

    # At hour t the neighbor value is the other station's flow at hour t-1.
    for station_id, expected in ((1, 0.9), (2, 0.1)):
        assert pd.isna(neighbors.loc[(station_id, hours[0]), "neighbor_net_flow"])
        for hour in hours[1:]:
            assert neighbors.loc[(station_id, hour), "neighbor_net_flow"] == pytest.approx(
                expected
            )


def test_load_stations_renames_capacity(tmp_path):
    stations_dir = tmp_path / "stations"
    stations_dir.mkdir()
    csv_path = stations_dir / "nomenclatura_2024_01.csv"
    csv_path.write_text(
        '"id","name","latitude","longitude","dpcapacity"\n'
        '1,"A",19.43,-99.13,10\n'
        '2,"B",19.44,-99.14,20\n',
        encoding="utf-8",
    )

    stations = load_stations(stations_dir)

    assert "capacity" in stations.columns
    assert "dpcapacity" not in stations.columns
    assert {"station_id", "name", "latitude", "longitude", "capacity"} == set(stations.columns)
    assert stations["capacity"].tolist() == [10, 20]


def test_no_label_leak_in_features():
    panel = build_classification_panel(make_trips(), make_stations(), YEAR)
    features = add_split(add_target(add_lags(add_neighbors(add_calendar(panel), k=1))))

    assert "net_flow_next" not in features.columns
    assert "target" in features.columns
    assert "split" in features.columns
    for column in ("net_flow_lag1", "net_flow_lag2", "net_flow_lag24", "neighbor_net_flow"):
        assert column in features.columns
