"""Point-in-time feature recipes for MiBici, independent of any fitted estimator.

All timestamps are timezone-naive Guadalajara wall time. A supervised row describes
[target_time, target_time + 1h), issued ten minutes before target_time. Therefore
only buckets starting at target_time - 2h or earlier are observable. These are
rolling one-hour forecasts, NOT a fixed-origin forecast for the whole test month.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
import pandas as pd
import typer

from tc5061_proyecto_equipo5.config import (
    CLASSIFICATION_FEATURES_PATH,
    CLASSIFICATION_PANEL_PATH,
    EXTERNAL_DATA_DIR,
    FEATURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RISK_THRESHOLD,
    STUDY_YEAR,
    TEST_MONTHS,
    TRAIN_MONTHS,
    VALIDATION_MONTHS,
)

app = typer.Typer()
FEATURE_VERSION = "4"
KEYS = ["station_id", "target_time"]


@dataclass(frozen=True)
class FeatureSet:
    """Keep identifiers/splits, predictors and labels separate to prevent target leakage."""

    keys: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series | None = None


def _timestamp(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None:
        raise ValueError("Use non-null, timezone-naive Guadalajara timestamps")
    return result


def _bounds(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = _timestamp(start), _timestamp(end)
    if start >= end or start != start.floor("h") or end != end.floor("h"):
        raise ValueError("Expected increasing, hour-aligned [start, end) bounds")
    return start, end


def build_hourly_panel(trips: pd.DataFrame, station_ids, *, start, end) -> pd.DataFrame:
    """Aggregate departures by start and arrivals by end, independently, on a dense grid.

    Explicit station_ids must come from an as-of-valid network, not future activity.
    Zero filling assumes the caller has verified complete source coverage in [start,end).
    It does NOT prove a station was open. Unknown stations fail rather than disappear.
    No user demographics or trip duration enter the model. Supply cross-boundary trips
    when counting arrivals; the input must not have been filtered by departure time.
    """
    start, end = _bounds(start, end)
    ids = pd.Index(station_ids, name="station_id")
    if ids.empty or ids.has_duplicates or ids.isna().any():
        raise ValueError("station_ids must be nonempty, unique and non-null")
    required = ["viaje_id", "inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
    data = trips[required].copy()
    if data.isna().any().any() or data.viaje_id.duplicated().any():
        raise ValueError("Trips contain missing values or duplicate viaje_id")
    for column in ("inicio_viaje", "fin_viaje"):
        data[column] = pd.to_datetime(data[column], errors="raise")
        if data[column].isna().any():
            raise ValueError(f"{column} contains missing timestamps after parsing")
        if data[column].dt.tz is not None:
            raise ValueError("Trip times must be timezone-naive Guadalajara time")
    if (data.fin_viaje < data.inicio_viaje).any():
        raise ValueError("Trip end precedes start")
    grid = pd.MultiIndex.from_product(
        [ids.sort_values(), pd.date_range(start, end, freq="h", inclusive="left")],
        names=KEYS,
    )
    panel = pd.DataFrame(index=grid)
    for name, station, time in (
        ("departures", "origen_id", "inicio_viaje"),
        ("arrivals", "destino_id", "fin_viaje"),
    ):
        events = data.loc[data[time].ge(start) & data[time].lt(end), [station, time]].copy()
        if not events[station].isin(ids).all():
            raise ValueError("Event station absent from the as-of station universe")
        events[time] = events[time].dt.floor("h")
        counts = events.groupby([station, time]).size()
        counts.index.names = KEYS
        panel[name] = counts.reindex(grid, fill_value=0).astype("int32")
    return panel.reset_index()


def _panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate exact hours and normalize counts for signed, overflow-safe arithmetic."""
    data = panel[KEYS + ["departures", "arrivals"]].copy()
    if data.empty or data.isna().any().any() or data.duplicated(KEYS).any():
        raise ValueError("Expected nonempty, non-null, unique station-hour panel")
    data["target_time"] = pd.to_datetime(data.target_time)
    if data.target_time.dt.tz is not None:
        raise ValueError("Panel times must be timezone-naive")
    data = data.sort_values(KEYS).reset_index(drop=True)
    if not data.target_time.eq(data.target_time.dt.floor("h")).all():
        raise ValueError("Panel times must be hour aligned")
    gaps = data.groupby("station_id").target_time.diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Panel must include every hour; do not shift sparse event rows")
    values = data[["departures", "arrivals"]].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError("Counts must be finite nonnegative integers")
    # Bound both directional sums together, not just individual subtractions.
    # Python integers avoid overflow/float rounding while checking uint64 inputs.
    limit = np.iinfo(np.int64).max // (2 * len(data))
    for column in ("departures", "arrivals"):
        if int(data[column].max()) > limit:
            raise ValueError("Counts exceed the safe signed aggregation range")
        data[column] = data[column].astype("int64")
    return data


def _calendar(data: pd.DataFrame) -> pd.DataFrame:
    time = data.target_time.dt
    return pd.DataFrame(
        {
            "station_id": data.station_id.astype(str),
            "hour_sin": np.sin(2 * np.pi * time.hour / 24),
            "hour_cos": np.cos(2 * np.pi * time.hour / 24),
            "weekday_sin": np.sin(2 * np.pi * time.dayofweek / 7),
            "weekday_cos": np.cos(2 * np.pi * time.dayofweek / 7),
            "is_weekend": (time.dayofweek >= 5).astype("int8"),
        }
    )


def _extended_calendar(data: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Known calendar and weekend interactions reflect EDA weekday/seasonal patterns."""
    time = data.target_time.dt
    X["hour"] = time.hour.astype("int8")
    X["day_of_week"] = time.dayofweek.astype("int8")
    X["month_sin"] = np.sin(2 * np.pi * (time.month - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (time.month - 1) / 12)
    for component in ("sin", "cos"):
        X[f"hour_{component}_weekend"] = X[f"hour_{component}"] * X.is_weekend
    return X


def _log_history(X: pd.DataFrame) -> pd.DataFrame:
    """Compress heavy-tailed predictor magnitudes, never clip demand/pressure targets."""
    for column in list(X):
        if column.startswith(("departures_", "arrivals_")):
            X[f"log1p_{column}"] = np.log1p(X[column])
        elif column.startswith("net_ratio_lag_"):
            X[f"signed_log1p_{column}"] = np.sign(X[column]) * np.log1p(X[column].abs())
    return X


def _validate_station_window(data, catalog, catalog_known_at):
    """Reject explicit effective dates or row-level notices incompatible with the panel."""
    earliest_issue = data.target_time.min() - pd.Timedelta(minutes=10)
    if _timestamp(catalog_known_at) > earliest_issue:
        raise ValueError("Station metadata must be known before the earliest issue")
    if "known_at" in catalog:
        notices = pd.to_datetime(catalog.known_at, errors="raise")
        if notices.isna().any() or notices.dt.tz is not None or notices.gt(earliest_issue).any():
            raise ValueError("Station metadata row was not known before the earliest issue")
    for column, bound, invalid in (
        ("valid_from", data.target_time.min(), "after"),
        ("valid_to", data.target_time.max() + pd.Timedelta(hours=1), "before"),
    ):
        if column in catalog:
            dates = pd.to_datetime(catalog[column], errors="raise")
            if dates.isna().any() or dates.dt.tz is not None:
                raise ValueError("Station validity must use non-null local dates")
            if (dates.gt(bound) if invalid == "after" else dates.lt(bound)).any():
                raise ValueError("Station snapshot does not cover the entire panel")


def _require_positive_capacity(capacity: pd.Series) -> None:
    """Fail when a station lacks capacity or has zero/negative capacity.

    Capacity divides the flow, so a missing value would silently label the row balanced and a
    zero would turn the flow into +/-inf and label it as risk.
    """
    if capacity.isna().any() or not np.isfinite(capacity).all() or (capacity <= 0).any():
        raise ValueError("Every station needs finite positive historical capacity")


def _station_features(data, X, stations, catalog_known_at) -> pd.DataFrame:
    """Static as-of-valid geography plus lagged neighbor pressure in the analyzed network.

    A single snapshot must cover the entire supplied panel; SCD2 joins are not
    inferred from trip activity. Neighbors exclude self, not co-located stations.
    Missing history stays NaN. No contemporaneous neighbor flow enters X.
    """
    if catalog_known_at is None or _timestamp(catalog_known_at) > (
        data.target_time.min() - pd.Timedelta(minutes=10)
    ):
        raise ValueError("Station metadata must be known before the earliest issue")
    ids = pd.Index(sorted(data.station_id.unique()))
    catalog = stations.set_index("station_id")
    if catalog.index.has_duplicates:
        raise ValueError("Station metadata must have unique IDs")
    catalog = catalog.reindex(ids)
    required = ["capacity", "latitude", "longitude"]
    values = catalog[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (catalog.capacity <= 0).any():
        raise ValueError("Every station needs finite capacity and coordinates")
    if (
        not catalog.latitude.between(-90, 90).all()
        or not catalog.longitude.between(-180, 180).all()
    ):
        raise ValueError("Invalid latitude/longitude range")
    _validate_station_window(data, catalog, catalog_known_at)
    for column in required:
        X[column] = data.station_id.map(catalog[column])
    if "municipality" in catalog:
        X["municipality"] = data.station_id.map(catalog.municipality).astype("string")
    latitude, longitude = np.deg2rad(values[:, 1]), np.deg2rad(values[:, 2])
    a = np.sin((latitude[:, None] - latitude) / 2) ** 2
    a += (
        np.cos(latitude[:, None])
        * np.cos(latitude)
        * np.sin((longitude[:, None] - longitude) / 2) ** 2
    )
    distances = 6371000 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(distances, np.inf)
    count = pd.Series((distances <= 500).sum(axis=1), index=ids)
    X["neighbor_count_500m"] = data.station_id.map(count)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, : min(3, len(ids) - 1)]
    if neighbors.shape[1] == 0:
        X["neighbor_net_ratio_lag_2"] = np.nan
    else:
        ratio = (data.arrivals - data.departures) / data.station_id.map(catalog.capacity)
        recent = ratio.groupby(data.station_id).shift(2)
        wide = data.assign(recent=recent).pivot(
            index="target_time", columns="station_id", values="recent"
        )
        position = ids.get_indexer(data.station_id)
        time = wide.index.get_indexer(data.target_time)
        X["neighbor_net_ratio_lag_2"] = (
            wide.reindex(columns=ids).to_numpy()[time[:, None], neighbors[position]].mean(axis=1)
        )
    return X


def _lags(data, columns, lags) -> pd.DataFrame:
    if not lags or any(not isinstance(lag, int) or lag < 2 for lag in lags):
        raise ValueError("Lags must be integer hours >= 2 for minute-50 issuance")
    result = {}
    for column in columns:
        group = data.groupby("station_id")[column]
        for lag in dict.fromkeys(lags):
            result[f"{column}_lag_{lag}"] = group.shift(lag)
    return pd.DataFrame(result, index=data.index)


def _keys(data: pd.DataFrame) -> pd.DataFrame:
    keys = data[KEYS].copy()
    keys["issue_time"] = keys.target_time - pd.Timedelta(minutes=10)
    keys["label_available_at"] = keys.target_time + pd.Timedelta(hours=1)
    # The label window closes one hour after its start; split by the target window.
    keys["split"] = pd.cut(
        keys.target_time,
        bins=pd.to_datetime(["2024-01-01", "2024-10-01", "2024-12-01", "2025-01-01"]),
        labels=["train", "validation", "test"],
        right=False,
    )
    return keys


def training_mask(
    keys: pd.DataFrame, *, fit_as_of, splits: tuple[str, ...] = ("train",)
) -> pd.Series:
    """Select allowed supervised labels already observable at the model's fit cutoff.

    Set fit_as_of no later than the earliest scoring issue_time (allowing time to fit).
    Split membership alone is insufficient at minute-50 boundaries. The default
    label_available_at assumes zero reporting latency after bucket close; callers
    may move it later for actual ingestion latency, never earlier than label close.
    Test labels cannot be selected. This mask does not replace train-only fitting
    of imputers/scalers, and does not mutate partition labels or the supplied keys.
    """
    cutoff = _timestamp(fit_as_of)
    required = {"target_time", "split", "label_available_at"}
    if not required.issubset(keys.columns):
        raise ValueError("Expected supervised keys with label availability")
    if isinstance(splits, str) or not splits or set(splits) - {"train", "validation"}:
        raise ValueError("Training splits must be train and/or validation, never test")
    available = pd.to_datetime(keys.label_available_at, errors="raise")
    targets = pd.to_datetime(keys.target_time, errors="raise")
    for times in (available, targets):
        if times.isna().any() or times.dt.tz is not None:
            raise ValueError("Label times must be non-null, timezone-naive timestamps")
    if available.lt(targets + pd.Timedelta(hours=1)).any():
        raise ValueError("Labels cannot be available before their target window closes")
    return keys["split"].isin(splits) & available.le(cutoff)


def build_demand_features(
    panel: pd.DataFrame,
    *,
    lags: tuple[int, ...] = (2, 3, 24, 168),
    windows: tuple[int, ...] = (3, 6),
    extended: bool = False,
    stations: pd.DataFrame | None = None,
    catalog_known_at=None,
) -> FeatureSet:
    """P1: demand history plus known calendar, with raw departure count as separate y.

    Each estimator may request different lags/windows without duplicating aggregation.
    Rolling means/std use only complete past buckets (shift before rolling). Warm-up
    stays NaN, never backfilled from the future; handle it in a train-fitted pipeline.
    """
    data = _panel(panel)
    X = pd.concat([_calendar(data), _lags(data, ["departures"], lags)], axis=1)
    for window in dict.fromkeys(windows):
        if not isinstance(window, int) or window < 2:
            raise ValueError("Rolling windows must be integers >= 2")
        group = data.groupby("station_id").departures
        for stat in ("mean", "std"):
            X[f"departures_{stat}_{window}"] = group.transform(
                lambda s, window=window, stat=stat: getattr(
                    s.shift(2).rolling(window, min_periods=window), stat
                )()
            )
    if extended:
        X = _log_history(_extended_calendar(data, X))
    if stations is not None:
        X = _station_features(data, X, stations, catalog_known_at)
    return FeatureSet(_keys(data), X, data.departures.rename("demand"))


def build_risk_features(
    panel: pd.DataFrame,
    stations: pd.DataFrame,
    *,
    catalog_known_at,
    lags: tuple[int, ...] = (2, 3, 24),
    tau: float = 0.40,
    extended: bool = False,
) -> FeatureSet:
    """P2: past flows/capacity predict future pressure, NOT actual bicycle inventory.

    stations requires station_id and positive capacity, valid unchanged for this window.
    catalog_known_at is caller-attested availability, not an arbitrary backdated stamp.
    Future normalized flow belongs ONLY to y. Ratios are unbounded: throughput can exceed
    physical capacity in one hour; do not clip them to [-1,1]. Inclusive thresholds apply.
    """
    data = _panel(panel)
    known_at = _timestamp(catalog_known_at)
    if known_at > data.target_time.min() - pd.Timedelta(minutes=10):
        raise ValueError("Station catalog was not known before the earliest issue time")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    catalog = stations[["station_id", "capacity"]]
    if catalog.isna().any().any() or catalog.station_id.duplicated().any():
        raise ValueError("Station catalog must be unique and non-null")
    _validate_station_window(
        data, stations.set_index("station_id").reindex(data.station_id.unique()), known_at
    )
    capacity = data.station_id.map(catalog.set_index("station_id").capacity)
    _require_positive_capacity(capacity)
    data["net_flow"] = data.arrivals - data.departures
    X = pd.concat(
        [_calendar(data), _lags(data, ["departures", "arrivals", "net_flow"], lags)], axis=1
    )
    X["capacity"] = capacity
    for lag in dict.fromkeys(lags):
        X[f"net_ratio_lag_{lag}"] = X[f"net_flow_lag_{lag}"] / capacity
    ratio = data.net_flow / capacity
    y = pd.Series(
        np.select(
            [ratio <= -tau, ratio >= tau],
            ["Riesgo_Desabasto", "Riesgo_Saturacion"],
            default="Balanceado",
        ),
        name="risk",
    )
    if extended:
        X = _log_history(_extended_calendar(data, X))
        if {"latitude", "longitude"}.issubset(stations.columns):
            X = _station_features(data, X, stations, known_at)
    return FeatureSet(_keys(data), X, y)


def build_station_profiles(
    panel: pd.DataFrame,
    *,
    start="2024-01-01",
    end="2024-10-01",
    min_monthly_trips: int = 100,
) -> FeatureSet:
    """P3: 24 outgoing + 24 incoming L1 proportions and asymmetry, no fitted scaler.

    Only [start,end) is observed; the profile is available at end, never before. Default
    train window excludes validation/test. Require complete calendar months and >=100
    departures in EACH month, not 100 annual trips. Zero-activity profiles are excluded.
    """
    start, end = _bounds(start, end)
    if start.day != 1 or end.day != 1 or start.hour or end.hour:
        raise ValueError("Profiles require whole calendar months")
    if not isinstance(min_monthly_trips, int) or min_monthly_trips < 0:
        raise ValueError("min_monthly_trips must be a nonnegative integer")
    data = _panel(panel)
    data = data.loc[data.target_time.ge(start) & data.target_time.lt(end)].copy()
    hours = int((end - start) / pd.Timedelta(hours=1))
    if data.empty or not data.groupby("station_id").size().eq(hours).all():
        raise ValueError("Profile window lacks complete hourly coverage")
    monthly = data.groupby(["station_id", data.target_time.dt.to_period("M")]).departures.sum()
    eligible = monthly.groupby(level=0).min()
    ids = eligible[eligible >= min_monthly_trips].index
    totals = data.groupby("station_id")[["departures", "arrivals"]].sum()
    ids = ids.intersection(totals.index[(totals > 0).all(axis=1)]).sort_values()
    X = pd.DataFrame(index=ids)
    for column in ("departures", "arrivals"):
        counts = data.groupby(["station_id", data.target_time.dt.hour])[column].sum()
        profile = counts.unstack(fill_value=0).reindex(index=ids, columns=range(24), fill_value=0)
        profile = profile.div(profile.sum(axis=1), axis=0)
        profile.columns = [f"{column}_hour_{hour:02}" for hour in range(24)]
        X = X.join(profile)
    X["asymmetry"] = ((totals.arrivals - totals.departures) / totals.sum(axis=1)).reindex(ids)
    keys = pd.DataFrame({"station_id": ids, "window_start": start, "window_end": end})
    return FeatureSet(keys, X.reset_index(drop=True))


def _category_text(X):
    """Deterministic missing token; does not learn from evaluation categories."""
    return X.astype("string").fillna("__missing__").astype(str)


def make_preprocessor(
    X: pd.DataFrame,
    *,
    scale_numeric: bool = True,
    numeric_scaling: str | None = None,
    select_variance: bool = False,
    missing_indicators: bool = False,
):
    """Create an UNFITTED per-estimator transformer; fit only inside its train pipeline.

    Linear/distance models typically scale, tree models can opt out. Station identity
    is categorical (unknown stations ignored), never a numeric spatial distance. This
    factory inspects column names only; it learns no statistics from X.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import (
        FunctionTransformer,
        OneHotEncoder,
        RobustScaler,
        StandardScaler,
    )

    from tc5061_proyecto_equipo5.features import _category_text as normalize_categories

    scaling = numeric_scaling or ("standard" if scale_numeric else "none")
    if scaling not in {"standard", "robust", "none"}:
        raise ValueError("numeric_scaling must be standard, robust or none")
    categorical = [
        c
        for c in X
        if c in {"station_id", "municipality"}
        or pd.api.types.is_string_dtype(X[c].dtype)
        or isinstance(X[c].dtype, pd.CategoricalDtype)
    ]
    numeric = [c for c in X.columns if c not in categorical]
    steps = [
        SimpleImputer(
            strategy="median", keep_empty_features=True, add_indicator=missing_indicators
        )
    ]
    if scaling == "standard":
        steps.append(StandardScaler())
    elif scaling == "robust":
        steps.append(RobustScaler())
    transformers = [("numeric", make_pipeline(*steps), numeric)]
    if categorical:
        transformers.append(
            (
                "station",
                make_pipeline(
                    FunctionTransformer(normalize_categories, feature_names_out="one-to-one"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical,
            )
        )
    transformer = ColumnTransformer(transformers, remainder="drop")
    return make_pipeline(transformer, VarianceThreshold()) if select_variance else transformer


def _load_feature_trips(input_dir: Path):
    """Audit exact event identities; keep zero/long durations, never target-trim peaks."""
    from io import BytesIO

    columns = ["viaje_id", "inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
    paths = [input_dir / "year=2024" / f"viajes_2024_{m:02}.parquet" for m in range(1, 13)]
    context = input_dir / "year=2023" / "viajes_2023_12.parquet"
    if context.exists():
        paths.insert(0, context)
    frames, hashes, monthly = [], {}, []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Missing monthly coverage: {path}")
        payload = path.read_bytes()
        frame = pd.read_parquet(BytesIO(payload), columns=columns)
        for column in ("inicio_viaje", "fin_viaje"):
            frame[column] = pd.to_datetime(frame[column], errors="raise")
        if frame.isna().any().any():
            raise ValueError(f"Null event identities/timestamps in {path.name}")
        month = path.stem.replace("viajes_", "").replace("_", "-")
        year_number, month_number = map(int, month.split("-"))
        if not (
            frame.inicio_viaje.dt.year.eq(year_number)
            & frame.inicio_viaje.dt.month.eq(month_number)
        ).all():
            raise ValueError(f"Departure month does not match source: {path.name}")
        hashes[path.name] = hashlib.sha256(payload).hexdigest()
        monthly.append(
            {
                "month": month,
                "rows": len(frame),
                "stations": int(pd.concat([frame.origen_id, frame.destino_id]).nunique()),
            }
        )
        frames.append(frame)
    trips = pd.concat(frames, ignore_index=True)
    source_rows = int(trips.inicio_viaje.dt.year.eq(2024).sum())
    dedup = trips.drop_duplicates(columns)
    if dedup.viaje_id.duplicated().any():
        raise ValueError(
            "Conflicting records for the same viaje_id; no arbitrary correction chosen"
        )
    duration = (dedup.fin_viaje - dedup.inicio_viaje).dt.total_seconds() / 60
    if (duration < 0).any() or (dedup[["origen_id", "destino_id"]] <= 0).any().any():
        raise ValueError(
            "Invalid event station or negative duration; quarantine requires investigation"
        )
    quality = {
        "source_rows_2024": source_rows,
        "source_rows_with_context": len(trips),
        "identical_duplicates_removed": len(trips) - len(dedup),
        "monthly": monthly,
        "zero_duration_kept": int(duration.eq(0).sum()),
        "duration_over_180m_kept": int(duration.gt(180).sum()),
        "duration_policy": "audit only; no future end/duration based filtering of departures",
        "context_december_2023": context.exists(),
        "personal_columns": "excluded: user_id, gender, birth year",
        "duration_outliers": "not used as predictors or target filters",
    }
    return dedup.reset_index(drop=True), hashes, quality


def _write_ready(result, target, *, fit_as_of, eligible=None, profiles=None):
    """Fit once on eligible train; export bounded-size CSR batches with aligned keys."""
    import pickle

    from scipy import sparse

    if profiles is None:
        masks = {
            split: result.keys["split"].eq(split) & eligible
            for split in ("train", "validation", "test")
        }
        masks["train"] &= training_mask(result.keys, fit_as_of=fit_as_of)
        masks["validation"] &= training_mask(
            result.keys, fit_as_of="2024-11-30 23:40", splits=("validation",)
        )
        train_X = result.X.loc[masks["train"]]
    else:
        train_X = profiles["train"].X
    if len(train_X) < 2 or not train_X.nunique(dropna=False).gt(1).any():
        return {"status": "blocked", "reason": "Insufficient eligible training observations"}
    processor = make_preprocessor(
        train_X,
        numeric_scaling="none" if profiles else "robust",
        select_variance=True,
        missing_indicators=True,
    )
    processor.fit(train_X)
    names = processor.get_feature_names_out().tolist()
    with (target / "preprocessor.pkl").open("wb") as handle:
        pickle.dump(processor, handle)
    prepared = {
        "status": "ready",
        "fit_as_of": str(fit_as_of),
        "selection_as_of": None if profiles else "2024-11-30 23:40:00",
        "selected_columns": names,
        "input_columns": result.X.columns.tolist(),
        "scaling": "none" if profiles else "robust",
        "selection": "train-only nonzero variance",
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        folder = target / "ready" / split
        folder.mkdir(parents=True, exist_ok=True)
        if profiles is None:
            X, keys, y = (
                result.X.loc[masks[split]],
                result.keys.loc[masks[split]],
                result.y.loc[masks[split]],
            )
        else:
            X, keys, y = profiles[split].X, profiles[split].keys, None
        keys.to_parquet(folder / "keys.parquet", index=False)
        if y is not None:
            y.to_frame().to_parquet(folder / "y.parquet", index=False)
        batches = []
        for start in range(0, len(X), 100000):
            stop = min(start + 100000, len(X))
            matrix = sparse.csr_matrix(processor.transform(X.iloc[start:stop]), dtype=np.float32)
            if not np.isfinite(matrix.data).all():
                raise ValueError("Nonfinite transformed features")
            path = folder / f"X_{start:08}.npz"
            sparse.save_npz(path, matrix)
            batches.append(
                {
                    "path": path.relative_to(target).as_posix(),
                    "start": start,
                    "stop": stop,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        prepared["splits"][split] = {"rows": len(X), "columns": len(names), "batches": batches}
    return prepared


def _demand_ablation(result, eligible, fit_as_of):
    """Fixed Ridge comparison on deterministic samples, no test targets or tuned selection."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.pipeline import make_pipeline

    train = result.keys.index[eligible & training_mask(result.keys, fit_as_of=fit_as_of)]
    validation = result.keys.index[
        eligible & training_mask(result.keys, fit_as_of="2024-11-30 23:40", splits=("validation",))
    ]
    train = train[:: max(1, int(np.ceil(len(train) / 50000)))]
    validation = validation[:: max(1, int(np.ceil(len(validation) / 20000)))]
    if len(train) < 2 or len(validation) < 2:
        return {
            "status": "not_evaluated",
            "reason": "Insufficient comparable train/validation rows",
        }
    baseline = [
        c
        for c in (
            "station_id",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "is_weekend",
            "departures_lag_24",
        )
        if c in result.X
    ]
    report: dict = {
        "protocol": "fixed alpha=1; deterministic <=50k train/<=20k validation; same rows; no test",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "comparisons": {},
    }
    for name, columns in (("calendar_lag24", baseline), ("engineered", result.X.columns.tolist())):
        X = result.X.loc[:, columns]
        model = make_pipeline(
            make_preprocessor(X, numeric_scaling="robust"),
            Ridge(alpha=1, solver="lsqr", max_iter=200),
        )
        model.fit(X.loc[train], result.y.loc[train])
        predicted = np.maximum(0, model.predict(X.loc[validation]))
        actual = result.y.loc[validation].to_numpy()
        denominator = actual.sum()
        report["comparisons"][name] = {
            "input_columns": columns,
            "val_mae": float(mean_absolute_error(actual, predicted)),
            "val_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "val_wape": float(np.abs(actual - predicted).sum() / denominator)
            if denominator
            else None,
            "postprocessing": "clip predictions at zero; target unchanged",
        }
    return report


def _profile_comparison(panel, profiles):
    """Compare volume versus shape labels in the SAME validation profile space; no test."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler

    if min(len(profiles[s].X) for s in ("train", "validation")) < 5:
        return {
            "status": "not_evaluated",
            "reason": "Need at least five eligible stations per window",
        }
    original = {}
    for split in ("train", "validation"):
        profile = profiles[split]
        start, end = profile.keys.window_start.iloc[0], profile.keys.window_end.iloc[0]
        totals = (
            panel.loc[panel.target_time.ge(start) & panel.target_time.lt(end)]
            .groupby("station_id")[["departures", "arrivals"]]
            .sum()
        )
        original[split] = totals.reindex(profile.keys.station_id).reset_index(drop=True)
    result: dict = {
        "protocol": "fixed K=4,n_init=10,seed=42; same eligible validation stations; silhouette/DB measured in common 49D profile space; no test",
        "comparisons": {},
    }
    common = profiles["validation"].X
    for name, train_X, val_X, scale in (
        ("original_volume", original["train"], original["validation"], True),
        ("engineered_profile", profiles["train"].X, common, False),
    ):
        cluster = KMeans(n_clusters=4, n_init=10, random_state=42)
        model = make_pipeline(RobustScaler(), cluster) if scale else cluster
        model.fit(train_X)
        labels = model.predict(val_X)
        groups = len(np.unique(labels))
        valid = 1 < groups < len(labels)
        result["comparisons"][name] = {
            "train_stations": len(train_X),
            "validation_stations": len(val_X),
            "val_groups": groups,
            "val_profile_silhouette": float(silhouette_score(common, labels)) if valid else None,
            "val_profile_davies_bouldin": float(davies_bouldin_score(common, labels))
            if valid
            else None,
            "undefined_reason": None if valid else "Insufficient nonempty groups",
        }
    return result


def prepare_feature_dataset(
    problem: str,
    input_dir: Path,
    output_dir: Path,
    *,
    stations_path: Path | None = None,
    catalog_known_at=None,
) -> dict:
    """Materialize audited real/synthetic sources and train-fitted, per-split ready matrices.

    Station identity becomes eligible only after its first observed event, not its
    future existence. This is evidence of observation, NOT proof of installation,
    continuous operation or complete source coverage. Historical metadata is optional
    for P1/P3; P2 fails closed rather than inventing its capacity-normalized labels.
    """
    from io import BytesIO
    import os
    from tempfile import TemporaryDirectory

    if problem not in {"p1", "p2", "p3"}:
        raise ValueError("problem must be p1, p2 or p3")
    if problem == "p2" and (stations_path is None or catalog_known_at is None):
        raise ValueError("P2 blocked: historical capacity and provenance are required")
    trips, hashes, quality = _load_feature_trips(input_dir)
    stations = None
    if stations_path is not None:
        payload = stations_path.read_bytes()
        stations = pd.read_csv(BytesIO(payload))
        hashes["stations"] = hashlib.sha256(payload).hexdigest()
        if catalog_known_at is None:
            raise ValueError("Station metadata requires a documented known_at")
    events = pd.concat(
        [
            trips[["origen_id", "inicio_viaje"]].rename(
                columns={"origen_id": "station_id", "inicio_viaje": "time"}
            ),
            trips[["destino_id", "fin_viaje"]].rename(
                columns={"destino_id": "station_id", "fin_viaje": "time"}
            ),
        ]
    )
    # Exclude events outside the study/context when defining its observed universe.
    start = "2023-12-01" if quality["context_december_2023"] else "2024-01-01"
    events = events.loc[events.time.ge(start) & events.time.lt("2025-01-01")]
    first_seen = events.groupby("station_id").time.min()
    # An attested static snapshot, not eventual trip activity, defines spatial membership.
    panel_ids = stations.station_id if stations is not None else first_seen.index
    panel = build_hourly_panel(trips, panel_ids, start=start, end="2025-01-01")
    del trips, events
    final_target = output_dir / problem
    # Reject accidental overwrite: one output directory is one immutable preparation run.
    if final_target.exists() and any(final_target.iterdir()):
        raise ValueError(f"Output already exists: {final_target}; use a new run directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{problem}-pending-", dir=output_dir) as folder:
        target = Path(folder)
        profiles = None
        if problem == "p1":
            result = build_demand_features(
                panel,
                extended=True,
                stations=stations,
                catalog_known_at=catalog_known_at,
            )
        elif problem == "p2":
            result = build_risk_features(
                panel,
                stations,
                catalog_known_at=catalog_known_at,
                lags=(2, 3, 4, 24),
                extended=True,
            )
        else:
            profiles = {
                split: build_station_profiles(panel, start=left, end=right)
                for split, left, right in (
                    ("train", "2024-01-01", "2024-10-01"),
                    ("validation", "2024-10-01", "2024-12-01"),
                    ("test", "2024-12-01", "2025-01-01"),
                )
            }
            result = profiles["train"]
        eligible = None
        fit_as_of = "2024-10-01 00:00:00" if profiles else "2024-09-30 23:40:00"
        if profiles is None:
            assert result.y is not None
            observed = result.keys.station_id.map(first_seen)
            keep = result.keys.target_time.ge("2024-01-01") & observed.le(result.keys.issue_time)
            positions = result.keys.index[keep]
            result = FeatureSet(
                result.keys.loc[positions].reset_index(drop=True),
                result.X.loc[positions].reset_index(drop=True),
                result.y.loc[positions].reset_index(drop=True),
            )
            lookback = 168 if problem == "p1" else 24
            result.keys["history_complete"] = result.keys.target_time.ge(
                result.keys.station_id.map(first_seen).dt.ceil("h") + pd.Timedelta(hours=lookback)
            )
            eligible = result.keys.history_complete & result.X.select_dtypes(
                include="number"
            ).notna().all(axis=1)
            quality["warmup_excluded_rows"] = int((~eligible).sum())
            quality["eligible_rows_by_split"] = (
                result.keys.loc[eligible, "split"].value_counts().to_dict()
            )
        result.X.to_parquet(target / "X.parquet", index=False)
        result.keys.to_parquet(target / "keys.parquet", index=False)
        if result.y is not None:
            result.y.to_frame().to_parquet(target / "y.parquet", index=False)
        quality["feature_summary_train"] = {}
        train_X = (
            result.X
            if profiles
            else result.X.loc[eligible & training_mask(result.keys, fit_as_of=fit_as_of)]
        )
        for column in train_X:
            series = train_X[column]
            description = {
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique()),
            }
            if pd.api.types.is_numeric_dtype(series.dtype):
                quantiles = series.quantile([0, 0.01, 0.5, 0.99, 1])
                description["quantiles"] = {
                    str(q): float(v) if pd.notna(v) else None for q, v in quantiles.items()
                }
            quality["feature_summary_train"][column] = description
        receipt = {
            "feature_version": FEATURE_VERSION,
            "problem": problem,
            "input_sha256": hashes,
            "rows": len(result.X),
            "columns": result.X.columns.tolist(),
            "quality": quality,
            "availability": "event time replay; true publication latency unobserved",
            "station_universe": (
                "attested static catalog; focal rows gated by first observed event <= issue time"
                if stations is not None
                else "first observed event <= issue time, not an installation date"
            ),
            "unavailable_features": (
                ["capacity", "coordinates", "municipality", "spatial_neighbors"]
                if stations is None and problem != "p3"
                else []
            ),
            "prepared": _write_ready(
                result, target, fit_as_of=fit_as_of, eligible=eligible, profiles=profiles
            ),
        }
        if problem == "p1":
            receipt["ablation"] = _demand_ablation(result, eligible, fit_as_of)
        elif problem == "p3":
            receipt["ablation"] = _profile_comparison(panel, profiles)
        receipt["data_version"] = hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest()
        (target / "features_manifest.json").write_text(
            json.dumps(receipt, indent=2, default=int) + "\n"
        )
        os.replace(target, final_target)
        return receipt


@app.command()
def build(
    question: Annotated[str, typer.Option(help="p1, p2 or p3")],
):
    """Build the features of one question with the paths of config.py (MLflow entry point)."""

    if question not in {"p1", "p2", "p3"}:
        raise typer.BadParameter("question must be p1, p2 or p3")

    if question == "p2":
        build_classification_features()
    else:
        path = FEATURES_DIR / question
        if path.exists() and any(path.iterdir()):
            logger.info(f"Already exists... skipping {path}")
            return
        try:
            prepare_feature_dataset(question, INTERIM_DATA_DIR / "viajes", FEATURES_DIR)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc


@app.command()
def main(
    problem: Annotated[str, typer.Option(help="p1, p2 or p3")],
    stations_path: Annotated[
        Path | None, typer.Option(exists=True, help="As-of CSV: station_id,capacity")
    ] = None,
    catalog_known_at: Annotated[
        str | None, typer.Option(help="Audited historical catalog availability")
    ] = None,
    input_dir: Path = INTERIM_DATA_DIR / "viajes",
    output_dir: Path = PROCESSED_DATA_DIR,
    prepare_ready: bool = False,
):
    """Build baseline 2024 artifacts from all 12 monthly Parquets; never download data.

    Capacity is required only for P2. Catalog IDs are the explicitly attested 2024
    station universe. A 2026 catalog will be rejected by its true availability date.
    Input hashes and recipe version are recorded alongside separate X/y/keys artifacts.
    """
    if prepare_ready:
        try:
            receipt = prepare_feature_dataset(
                problem,
                input_dir,
                output_dir,
                stations_path=stations_path,
                catalog_known_at=catalog_known_at,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(
            f"{problem}: {receipt['rows']} raw rows; ready status={receipt['prepared']['status']}"
        )
        return
    if stations_path is None or catalog_known_at is None:
        raise typer.BadParameter("Supply historical stations or use --prepare-ready for P1/P3")
    if problem not in {"p1", "p2", "p3"}:
        raise typer.BadParameter("problem must be p1, p2 or p3")
    known_at = _timestamp(catalog_known_at)
    if known_at > pd.Timestamp("2023-12-31 23:50"):
        raise typer.BadParameter("Baseline needs a catalog known before the first 2024 forecast")
    paths = [input_dir / "year=2024" / f"viajes_2024_{m:02}.parquet" for m in range(1, 13)]
    for path in paths:
        if not path.is_file():
            raise typer.BadParameter(f"Missing monthly coverage: {path}")
    # Read/hash the same bytes so the receipt describes exactly the consumed snapshot.
    from io import BytesIO

    hashes = {}
    frames = []
    for path in paths:
        payload = path.read_bytes()
        hashes[path.name] = hashlib.sha256(payload).hexdigest()
        frames.append(pd.read_parquet(BytesIO(payload)))
    payload = stations_path.read_bytes()
    hashes["stations"] = hashlib.sha256(payload).hexdigest()
    stations = pd.read_csv(BytesIO(payload))
    panel = build_hourly_panel(
        pd.concat(frames, ignore_index=True),
        stations.station_id,
        start="2024-01-01",
        end="2025-01-01",
    )
    if problem == "p1":
        result = build_demand_features(panel)
    elif problem == "p2":
        result = build_risk_features(panel, stations, catalog_known_at=known_at)
    else:
        result = build_station_profiles(panel)
    target = output_dir / problem
    target.mkdir(parents=True, exist_ok=True)
    result.X.to_parquet(target / "X.parquet", index=False)
    result.keys.to_parquet(target / "keys.parquet", index=False)
    if result.y is not None:
        result.y.to_frame().to_parquet(target / "y.parquet", index=False)
    receipt = {
        "feature_version": FEATURE_VERSION,
        "problem": problem,
        "input_sha256": hashes,
        "catalog_known_at": str(known_at),
        "rows": len(result.X),
        "columns": result.X.columns.tolist(),
        "forecast_issue_offset_minutes": -10,
        "earliest_lag_hours": 2,
        "label_availability": "target_time + 1h; zero reporting latency assumed",
        "p3_window": ["2024-01-01", "2024-10-01"],
        "evaluation": "rolling hourly; assumes completed buckets available immediately",
        "coverage": "12 source files; row-level completeness not independently certified",
        "left_censored_arrivals": "January lacks trips departing before 2024; warm-up required",
    }
    (target / "features_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    typer.echo(f"{problem}: {len(result.X)} rows written to {target}")


# ---------------------------------------------------------------------------
# P2 classification artifacts
#
# The recipes above are the strict point-in-time contract.  The open data portal publishes only the
# current station catalog, so the risk recipe cannot be materialized without one explicit
# assumption: capacity is treated as static between the study year and the catalog snapshot.  That
# assumption is logged here and must be reported with the results.  The functions below reuse the
# dense hourly panel and build the table that modeling/train_classification.py and plots.py consume.
# ---------------------------------------------------------------------------

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
# Shared station-hour panel of the classification study
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
CLASSIFICATION_COLUMNS = [
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
# A station must operate in every month; partial stations would add fake zero rows
FULL_YEAR_MONTHS = 12
# Federal holidays of Mexico in 2024: demand does not follow a normal weekday
HOLIDAYS_2024 = pd.to_datetime(
    [
        "2024-01-01",
        "2024-02-05",
        "2024-03-18",
        "2024-05-01",
        "2024-09-16",
        "2024-10-01",  # transfer of the Federal Executive (every six years)
        "2024-11-18",
        "2024-12-25",
    ]
)


def load_trips(
    year: int = STUDY_YEAR, trips_dir: Path = INTERIM_DATA_DIR / "viajes"
) -> pd.DataFrame:
    """Read the cleaned trips of one year and keep the eight trip columns.

    Trips are only a counting source (departures and arrivals per hour); the user columns are kept
    for traceability but never become features, because the question is operational and per station.
    """
    paths = sorted(trips_dir.glob(f"year={year}/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files for {year} in {trips_dir}")
    trips = pd.concat(
        [pd.read_parquet(path, columns=TRIP_COLUMNS) for path in paths], ignore_index=True
    )
    logger.info(f"Loaded {len(trips):,} trips from {len(paths)} files for {year}")
    return trips


def load_stations(stations_dir: Path = EXTERNAL_DATA_DIR / "estaciones") -> pd.DataFrame:
    """Read the newest station catalog and normalize its columns.

    The catalog is the only source of capacity and coordinates; the raw capacity column is called
    ``dpcapacity``.  Capacity is assumed static for the study year (see the block note above).
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


def active_months(trips: pd.DataFrame) -> pd.Series:
    """Count the months in which each station moved at least one bike.

    A station counts when it appears as the origin of a departure or as the destination of an
    arrival, so the number reflects real operation and not a zero-filled grid.
    """
    departures = pd.DataFrame({"station_id": trips["origen_id"], "when": trips["inicio_viaje"]})
    arrivals = pd.DataFrame({"station_id": trips["destino_id"], "when": trips["fin_viaje"]})
    both = pd.concat([departures, arrivals], ignore_index=True)
    both["month"] = both["when"].dt.month
    return both.groupby("station_id")["month"].nunique()


def full_year_station_ids(trips: pd.DataFrame) -> pd.Index:
    """Return the stations that operated in every month of the study year.

    A station that only appears in part of the year would contribute zero-filled station-hours for
    the months when it did not exist, and those fake rows are all labeled ``balanced``.  Requiring
    activity in every month keeps the panel honest and gives every station the same history in the
    three splits.
    """
    months = active_months(trips)
    partial = months[months < FULL_YEAR_MONTHS]
    if len(partial):
        logger.info(
            f"Dropping {len(partial)} stations that did not operate the whole year: "
            f"{sorted(partial.index.tolist())}"
        )
    return pd.Index(sorted(months[months == FULL_YEAR_MONTHS].index), name="station_id")


def build_classification_panel(
    trips: pd.DataFrame, stations: pd.DataFrame, year: int = STUDY_YEAR
) -> pd.DataFrame:
    """Build the dense station-hour panel with flow, capacity and coordinates.

    The aggregation and the full grid come from ``build_hourly_panel``; that builder needs every
    station that appears in the trips, so the union is passed and the rows of the partial stations
    are dropped afterwards.  Their trips still count as arrivals or departures of the stations that
    stay, because a bike that arrived did arrive.
    """
    event_ids = pd.Index(
        sorted(set(trips["origen_id"]) | set(trips["destino_id"])), name="station_id"
    )
    panel = build_hourly_panel(trips, event_ids, start=f"{year}-01-01", end=f"{year + 1}-01-01")
    panel = panel.rename(columns={"target_time": "hour"})
    keep = full_year_station_ids(trips)
    panel = panel[panel["station_id"].isin(keep)]
    panel = panel.merge(
        stations[["station_id", "capacity", "latitude", "longitude"]], on="station_id", how="left"
    )
    _require_positive_capacity(panel["capacity"])
    panel["net_flow"] = (panel["arrivals"] - panel["departures"]) / panel["capacity"]
    panel = panel[PANEL_COLUMNS].sort_values(["station_id", "hour"]).reset_index(drop=True)
    logger.info(
        f"Panel: {len(panel):,} rows ({len(keep)} stations x {panel['hour'].nunique()} hours)"
    )
    return panel


def add_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic hour, weekday and holiday flags.

    Bike demand follows the clock and the working calendar, so the model needs to know the time of
    day and whether the day is special.  All of these come from the timestamp of the row, which is
    known in advance.
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

    Distance is the plain Euclidean distance between latitude/longitude, which is good enough for a
    city-wide grid and easy to explain.
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

    Nearby stations share trips, so their flow helps predict the next hour of a station.  The
    neighbours are fixed per station (same three for every hour), but their flow is taken from hour
    ``t - 1``: using hour ``t`` or ``t + 1`` would leak information that is not available when the
    prediction is made.
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

    Lag 1 and 2 describe the last two hours (short-term pressure) and lag 24 describes the same hour
    yesterday (daily pattern).  Arrivals and departures are lagged separately because an imbalance
    can come from one side only.  Every lag uses ``shift``, so a row never sees its own hour or the
    future.
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

    ``net_flow_next`` is the flow of hour ``t + 1``: shortage if it drops to or below
    ``-RISK_THRESHOLD``, saturation if it rises to or above ``+RISK_THRESHOLD``, balanced otherwise.
    This is the only column that looks forward, and it is the value to predict, so it is dropped
    before returning: it must never reach the model as a feature.  The last hour of the year has no
    ``t + 1``, so its label is left empty instead of invented.
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

    The split is temporal (whole months, not random rows) so the test set is really out-of-time:
    December is never seen during training or validation.
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


def build_classification_features(
    year: int = STUDY_YEAR,
    trips_dir: Path = INTERIM_DATA_DIR / "viajes",
    stations_dir: Path = EXTERNAL_DATA_DIR / "estaciones",
    panel_path: Path = CLASSIFICATION_PANEL_PATH,
    features_path: Path = CLASSIFICATION_FEATURES_PATH,
) -> pd.DataFrame:
    """Run the whole classification pipeline and save the panel and the feature table.

    Order matters: the panel comes first, then the features that only look back (calendar,
    neighbours, lags) and finally the label and the temporal split.
    """
    trips = load_trips(year, trips_dir)
    stations = load_stations(stations_dir)
    panel = build_classification_panel(trips, stations, year)
    save_frame(panel, panel_path)
    features = add_calendar(panel)
    features = add_neighbors(features)
    features = add_lags(features)
    features = add_target(features)
    features = add_split(features)
    features = (
        features[CLASSIFICATION_COLUMNS].sort_values(["station_id", "hour"]).reset_index(drop=True)
    )
    save_frame(features, features_path)
    return features


@app.command()
def panel(
    year: int = STUDY_YEAR,
    trips_dir: Path = INTERIM_DATA_DIR / "viajes",
    stations_dir: Path = EXTERNAL_DATA_DIR / "estaciones",
    output_path: Path = CLASSIFICATION_PANEL_PATH,
):
    """Build and save only the station-hour panel of the classification study."""
    logger.info(f"Building the station-hour panel for {year}")
    trips = load_trips(year, trips_dir)
    stations = load_stations(stations_dir)
    frame = build_classification_panel(trips, stations, year)
    save_frame(frame, output_path)
    typer.echo(f"Panel complete: {len(frame):,} rows")


@app.command()
def classification(
    year: int = STUDY_YEAR,
    trips_dir: Path = INTERIM_DATA_DIR / "viajes",
    stations_dir: Path = EXTERNAL_DATA_DIR / "estaciones",
    panel_path: Path = CLASSIFICATION_PANEL_PATH,
    features_path: Path = CLASSIFICATION_FEATURES_PATH,
):
    """Build and save the classification table (panel + calendar + neighbours + lags + target)."""
    logger.info(f"Building the classification features for {year}")
    frame = build_classification_features(year, trips_dir, stations_dir, panel_path, features_path)
    typer.echo(f"Classification features complete: {len(frame):,} rows")


if __name__ == "__main__":
    app()
