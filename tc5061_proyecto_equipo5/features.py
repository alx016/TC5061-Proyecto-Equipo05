"""Feature engineering for the station-hour risk classification task.

The question is: given everything known up to hour ``t``, will a station run out
of bikes (shortage) or out of docks (saturation) during hour ``t + 1``?  Every
function builds one piece of that table and keeps the time direction one-way:
features only look backwards, the label looks one hour ahead.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from tc5061_proyecto_equipo5.config import (
    CLASSIFICATION_FEATURES_PATH,
    CLASSIFICATION_PANEL_PATH,
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    RISK_THRESHOLD,
    STUDY_YEAR,
    TEST_MONTHS,
    TRAIN_MONTHS,
    VALIDATION_MONTHS,
)

app = typer.Typer()

# The eight columns stored in the cleaned trip Parquet files
TRIP_COLUMNS = [
    "viaje_id",
    "usuario_id",
    "genero",
    "anio_nacimiento",
    "inicio_viaje",
    "fin_viaje",
    "origen_id",
    "destino_id",
]
# Final shape of the shared station-hour panel
PANEL_COLUMNS = [
    "station_id",
    "hour",
    "departures",
    "arrivals",
    "net_flow",
    "capacity",
    "latitude",
    "longitude",
]
# Final shape of the classification table (net_flow_next is only the label source)
FINAL_COLUMNS = [
    "station_id",
    "hour",
    "departures",
    "arrivals",
    "net_flow",
    "capacity",
    "latitude",
    "longitude",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "neighbor_net_flow",
    "net_flow_lag1",
    "net_flow_lag2",
    "net_flow_lag24",
    "arrivals_lag1",
    "arrivals_lag2",
    "arrivals_lag3",
    "departures_lag1",
    "departures_lag2",
    "departures_lag3",
    "target",
    "split",
]
# Federal holidays of Mexico in 2024: demand does not follow a normal weekday
HOLIDAYS_2024 = pd.to_datetime(
    [
        "2024-01-01",
        "2024-02-05",
        "2024-03-18",
        "2024-05-01",
        "2024-09-16",
        "2024-11-18",
        "2024-12-25",
    ]
)


def load_trips(year: int, trips_dir: Path = INTERIM_DATA_DIR / "viajes") -> pd.DataFrame:
    """Read the cleaned trips of one year and keep the eight trip columns.

    Trips are only a counting source (departures and arrivals per hour); the
    user columns are kept for traceability but never become features, because
    the question is operational and per station.
    """
    paths = sorted(trips_dir.glob(f"year={year}/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files for {year} in {trips_dir}")
    trips = pd.concat(
        [pd.read_parquet(path, columns=TRIP_COLUMNS) for path in paths],
        ignore_index=True,
    )
    logger.info(f"Loaded {len(trips):,} trips from {len(paths)} files for {year}")
    return trips


def load_stations(stations_dir: Path = EXTERNAL_DATA_DIR / "estaciones") -> pd.DataFrame:
    """Read the newest station catalog and normalize its columns.

    The catalog is the only source of capacity and coordinates: capacity lets us
    express flow as a fraction of the station size, and coordinates let us build
    the neighbour feature.  The raw column for capacity is called ``dpcapacity``.
    """
    catalogs = sorted(stations_dir.glob("nomenclatura_*.csv"))
    if not catalogs:
        raise FileNotFoundError(f"No nomenclatura_*.csv file in {stations_dir}")
    path = catalogs[-1]
    stations = pd.read_csv(path).rename(columns={"id": "station_id", "dpcapacity": "capacity"})
    stations = stations[["station_id", "name", "latitude", "longitude", "capacity"]].copy()
    stations["station_id"] = stations["station_id"].astype("int32")
    stations["capacity"] = stations["capacity"].astype("int32")
    logger.info(f"Loaded {len(stations)} stations from {path.name}")
    return stations


def station_hours(year: int) -> pd.DatetimeIndex:
    """Return every hour of the year (8,784 hours in 2024, a leap year)."""
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    return pd.date_range(start, end, freq="h", inclusive="left")


def select_stations(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Keep the catalog stations that moved at least one bike during the year.

    A station that never appears in the trips is not operating, so filling it
    with zeros would add noise; a station without capacity cannot be normalized.
    """
    used = pd.Index(sorted(set(trips["origen_id"]) | set(trips["destino_id"])))
    active = stations[stations["station_id"].isin(used)].copy()
    unknown = used.difference(active["station_id"])
    if len(unknown):
        logger.warning(f"{len(unknown)} station ids in trips are not in the catalog")
    no_capacity = active["capacity"] <= 0
    if no_capacity.any():
        logger.warning(f"Dropping {int(no_capacity.sum())} stations with capacity <= 0")
        active = active[~no_capacity]
    return active.sort_values("station_id").reset_index(drop=True)


def arrival_column(trips: pd.DataFrame) -> str:
    """Return the trip end column, falling back to the start column.

    Real cleaned trips always carry ``fin_viaje``; the fallback only keeps the
    function usable with minimal tables that have just ``inicio_viaje``.
    """
    return "fin_viaje" if "fin_viaje" in trips.columns else "inicio_viaje"


def build_panel(trips: pd.DataFrame, stations: pd.DataFrame, year: int) -> pd.DataFrame:
    """Count departures and arrivals per station-hour and fill the full grid.

    The panel crosses every active station with every hour of the year, so the
    table has one row per station-hour and hours without trips become zero.  The
    complete grid keeps the series evenly spaced, which makes ``shift`` on the
    hour axis an exact one-hour lag.  Dividing by capacity turns the flow into a
    comparable percentage of the station size.
    """
    hours = station_hours(year)
    active = select_stations(trips, stations)

    departures = (
        trips.assign(hour=trips["inicio_viaje"].dt.floor("h"))
        .groupby(["origen_id", "hour"])
        .size()
        .rename("departures")
    )
    arrivals = (
        trips.assign(hour=trips[arrival_column(trips)].dt.floor("h"))
        .groupby(["destino_id", "hour"])
        .size()
        .rename("arrivals")
    )
    departures.index = departures.index.set_names(["station_id", "hour"])
    arrivals.index = arrivals.index.set_names(["station_id", "hour"])

    index = pd.MultiIndex.from_product([active["station_id"], hours], names=["station_id", "hour"])
    panel = pd.DataFrame(index=index).join([departures, arrivals])
    panel[["departures", "arrivals"]] = panel[["departures", "arrivals"]].fillna(0).astype("int32")
    panel = panel.reset_index().merge(active, on="station_id", how="left")
    panel["net_flow"] = (panel["arrivals"] - panel["departures"]) / panel["capacity"]
    panel = panel[PANEL_COLUMNS].sort_values(["station_id", "hour"]).reset_index(drop=True)
    logger.info(f"Panel: {len(panel):,} rows ({len(active)} stations x {len(hours)} hours)")
    return panel


def add_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic hour, weekday and holiday flags.

    Bike demand follows the clock and the working calendar, so the model needs
    to know the time of day and whether the day is special.  All of these come
    from the timestamp of the row, which is known in advance.
    """
    panel = panel.copy()
    hour_of_day = panel["hour"].dt.hour
    panel["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    panel["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)
    panel["day_of_week"] = panel["hour"].dt.dayofweek.astype("int8")
    panel["is_weekend"] = panel["day_of_week"] >= 5
    panel["is_holiday"] = panel["hour"].dt.normalize().isin(HOLIDAYS_2024)
    return panel


def neighbor_indices(panel: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the sorted station ids and the k nearest neighbours of each one.

    Distance is the plain Euclidean distance between latitude/longitude, which
    is good enough for a city-wide grid and easy to explain.
    """
    stations = (
        panel[["station_id", "latitude", "longitude"]]
        .drop_duplicates("station_id")
        .sort_values("station_id")
    )
    station_ids = stations["station_id"].to_numpy()
    k = min(k, len(station_ids) - 1)
    if k < 1:
        raise ValueError("At least two stations are needed to compute neighbours")
    coords = stations[["latitude", "longitude"]].to_numpy()
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)  # a station is not its own neighbour
    return station_ids, np.argsort(distances, axis=1)[:, :k]


def add_neighbors(panel: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Add the mean flow of the k nearest stations, lagged by one hour.

    Nearby stations share trips, so their flow helps predict the next hour of a
    station.  The neighbours are fixed per station (same three for every hour),
    but their flow is taken from hour ``t - 1``: using hour ``t`` or ``t + 1``
    would leak information that is not available when the prediction is made.
    """
    panel = panel.sort_values(["station_id", "hour"]).reset_index(drop=True)
    station_ids, neighbours = neighbor_indices(panel, k)
    flow = panel.pivot(index="hour", columns="station_id", values="net_flow")
    flow = flow.reindex(columns=station_ids)
    values = flow.to_numpy()
    neighbor_mean = values[:, neighbours].mean(axis=2)  # (hours, stations)
    lagged = np.vstack([np.full((1, neighbor_mean.shape[1]), np.nan), neighbor_mean[:-1]])
    neighbor_table = pd.DataFrame(lagged, index=flow.index, columns=station_ids)
    keys = pd.MultiIndex.from_arrays(
        [panel["hour"], panel["station_id"]], names=["hour", "station_id"]
    )
    panel["neighbor_net_flow"] = neighbor_table.stack().reindex(keys).to_numpy()
    return panel


def add_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """Add past values of flow, arrivals and departures per station.

    Lag 1 and 2 describe the last two hours (short-term pressure) and lag 24
    describes the same hour yesterday (daily pattern).  Arrivals and departures
    are lagged separately because an imbalance can come from one side only.
    Every lag uses ``shift``, so a row never sees its own hour or the future.
    """
    panel = panel.sort_values(["station_id", "hour"]).reset_index(drop=True)
    grouped = panel.groupby("station_id", sort=False)
    for lag in (1, 2, 24):
        panel[f"net_flow_lag{lag}"] = grouped["net_flow"].shift(lag)
    for lag in (1, 2, 3):
        panel[f"arrivals_lag{lag}"] = grouped["arrivals"].shift(lag)
        panel[f"departures_lag{lag}"] = grouped["departures"].shift(lag)
    return panel


def add_target(panel: pd.DataFrame) -> pd.DataFrame:
    """Label each row with the risk class of the NEXT hour.

    ``net_flow_next`` is the flow of hour ``t + 1``: shortage if it drops to or
    below ``-RISK_THRESHOLD``, saturation if it rises to or above
    ``+RISK_THRESHOLD``, balanced otherwise.  This is the only column that looks
    forward, and it is the value to predict, so it is dropped before returning:
    it must never reach the model as a feature.  The last hour of the year has
    no ``t + 1``, so its label is left empty instead of invented.
    """
    panel = panel.copy()
    next_flow = panel.groupby("station_id", sort=False)["net_flow"].shift(-1)
    target = pd.Series(0, index=panel.index, dtype="Int8")
    target[next_flow <= -RISK_THRESHOLD] = 1
    target[next_flow >= RISK_THRESHOLD] = 2
    target[next_flow.isna()] = pd.NA
    panel["target"] = target
    panel = panel.drop(columns="net_flow_next", errors="ignore")
    counts = panel["target"].value_counts(dropna=False).sort_index().to_dict()
    logger.info(f"Target counts (0 balanced, 1 shortage, 2 saturation): {counts}")
    return panel


def add_split(panel: pd.DataFrame) -> pd.DataFrame:
    """Assign each row to train, validation or test by calendar month.

    The split is temporal (whole months, not random rows) so the test set is
    really out-of-time: December is never seen during training or validation.
    """
    panel = panel.copy()
    month = panel["hour"].dt.month
    panel["split"] = pd.Series("unknown", index=panel.index, dtype="string")
    panel.loc[month.isin(TRAIN_MONTHS), "split"] = "train"
    panel.loc[month.isin(VALIDATION_MONTHS), "split"] = "validation"
    panel.loc[month.isin(TEST_MONTHS), "split"] = "test"
    if (panel["split"] == "unknown").any():
        raise ValueError("Some rows fall outside the train/validation/test months")
    logger.info(f"Split counts: {panel['split'].value_counts().to_dict()}")
    return panel


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as Parquet (pyarrow + zstd) and log its size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    logger.info(f"Saved {len(frame):,} rows to {path}")


def build_panel_file(
    year: int = STUDY_YEAR,
    trips_dir: Path = INTERIM_DATA_DIR / "viajes",
    stations_dir: Path = EXTERNAL_DATA_DIR / "estaciones",
    output_path: Path = CLASSIFICATION_PANEL_PATH,
) -> pd.DataFrame:
    """Build the shared station-hour panel from trips and stations, then save it."""
    trips = load_trips(year, trips_dir)
    stations = load_stations(stations_dir)
    panel = build_panel(trips, stations, year)
    save_frame(panel, output_path)
    return panel


def build_classification_features(
    year: int = STUDY_YEAR,
    trips_dir: Path = INTERIM_DATA_DIR / "viajes",
    stations_dir: Path = EXTERNAL_DATA_DIR / "estaciones",
    panel_path: Path = CLASSIFICATION_PANEL_PATH,
    features_path: Path = CLASSIFICATION_FEATURES_PATH,
) -> pd.DataFrame:
    """Run the whole pipeline and save the classification feature table.

    Order matters: the panel comes first, then the features that only look back
    (calendar, neighbours, lags) and finally the label and the temporal split.
    """
    panel = build_panel_file(year, trips_dir, stations_dir, panel_path)
    features = add_calendar(panel)
    features = add_neighbors(features)
    features = add_lags(features)
    features = add_target(features)
    features = add_split(features)
    features = features[FINAL_COLUMNS].sort_values(["station_id", "hour"]).reset_index(drop=True)
    save_frame(features, features_path)
    return features


@app.command()
def panel(
    year: int = STUDY_YEAR,
    trips_dir: Path = INTERIM_DATA_DIR / "viajes",
    stations_dir: Path = EXTERNAL_DATA_DIR / "estaciones",
    output_path: Path = CLASSIFICATION_PANEL_PATH,
):
    """Build and save only the station-hour panel."""
    logger.info(f"Building the station-hour panel for {year}")
    frame = build_panel_file(year, trips_dir, stations_dir, output_path)
    logger.success(f"Panel complete: {len(frame):,} rows")


@app.command()
def classification(
    year: int = STUDY_YEAR,
    trips_dir: Path = INTERIM_DATA_DIR / "viajes",
    stations_dir: Path = EXTERNAL_DATA_DIR / "estaciones",
    panel_path: Path = CLASSIFICATION_PANEL_PATH,
    features_path: Path = CLASSIFICATION_FEATURES_PATH,
):
    """Build and save the final classification features (panel + lags + target)."""
    logger.info(f"Building the classification features for {year}")
    frame = build_classification_features(year, trips_dir, stations_dir, panel_path, features_path)
    logger.success(f"Classification features complete: {len(frame):,} rows")


if __name__ == "__main__":
    app()
