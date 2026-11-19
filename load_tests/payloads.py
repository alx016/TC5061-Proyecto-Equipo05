"""Deterministic MiBici station payload generators for Locust load tests.

Uses only Python standard library modules (no ``locust`` import) so unit tests
can validate every generated payload under ``uv sync --frozen`` without the
optional ``perf`` dependency group.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

# MiBici Guadalajara active station count (Entrega 5) and PredictRequest schema limits.
MIBICI_ACTIVE_STATIONS = 365
MAX_BATCH_SIZE = 500
BATCH_SIZES = (1, 20, 365)

# Realistic Guadalajara bounding box (strictly within StationFeatures bounds).
_LAT_MIN = 20.625
_LAT_MAX = 20.715
_LON_MIN = -103.415
_LON_MAX = -103.285
_CAPACITIES = (11, 15, 19, 23, 27, 31)

# Precompute a deterministic station catalog so multi-station batches represent distinct docks.
_STATION_CATALOG: tuple[tuple[int, int, float, float], ...] = tuple(
    (
        idx + 2,
        _CAPACITIES[idx % len(_CAPACITIES)],
        round(_LAT_MIN + ((idx * 17) % 100) / 100.0 * (_LAT_MAX - _LAT_MIN), 6),
        round(_LON_MIN + ((idx * 29) % 100) / 100.0 * (_LON_MAX - _LON_MIN), 6),
    )
    for idx in range(MIBICI_ACTIVE_STATIONS)
)


def build_station_features(
    station_index: int = 0,
    hour: int | None = None,
    day_of_week: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Build a valid ``StationFeatures`` payload dict for a MiBici station."""
    rand = rng if rng is not None else random.Random(station_index)  # nosec B311

    cat_idx = station_index % len(_STATION_CATALOG)
    base_station_id, capacity, latitude, longitude = _STATION_CATALOG[cat_idx]
    station_id = base_station_id if station_index < MIBICI_ACTIVE_STATIONS else station_index + 2

    resolved_hour = (hour if hour is not None else rand.randrange(24)) % 24
    resolved_day = (day_of_week if day_of_week is not None else rand.randrange(7)) % 7
    angle = 2.0 * math.pi * resolved_hour / 24.0
    hour_sin = round(max(-1.0, min(1.0, math.sin(angle))), 6)
    hour_cos = round(max(-1.0, min(1.0, math.cos(angle))), 6)

    arrivals_1 = round(rand.uniform(0.0, 6.0), 2)
    arrivals_2 = round(rand.uniform(0.0, 6.0), 2)
    arrivals_3 = round(rand.uniform(0.0, 8.0), 2)
    departures_1 = round(rand.uniform(0.0, 6.0), 2)
    departures_2 = round(rand.uniform(0.0, 6.0), 2)
    departures_3 = round(rand.uniform(0.0, 8.0), 2)

    return {
        "station_id": station_id,
        "capacity": capacity,
        "latitude": latitude,
        "longitude": longitude,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_of_week": resolved_day,
        "is_weekend": resolved_day >= 5,
        "is_holiday": rand.random() < 0.03,
        "neighbor_net_flow": round(rand.uniform(-1.5, 1.5), 4),
        "net_flow_lag1": round(arrivals_1 - departures_1, 2),
        "net_flow_lag2": round(arrivals_2 - departures_2, 2),
        "net_flow_lag24": round(rand.uniform(-4.0, 4.0), 2),
        "arrivals_lag1": arrivals_1,
        "arrivals_lag2": arrivals_2,
        "arrivals_lag3": arrivals_3,
        "departures_lag1": departures_1,
        "departures_lag2": departures_2,
        "departures_lag3": departures_3,
    }


def build_predict_payload(
    batch_size: int = 1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Build a valid ``PredictRequest`` dict for ``1 <= batch_size <= 500`` stations."""
    if not isinstance(batch_size, int) or batch_size < 1 or batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}, got {batch_size!r}")

    rng = random.Random(seed)  # nosec B311
    hour = rng.randrange(24)
    day_of_week = rng.randrange(7)
    offset = rng.randrange(MIBICI_ACTIVE_STATIONS)

    instances = [
        build_station_features(
            station_index=(offset + i) % MIBICI_ACTIVE_STATIONS
            if batch_size <= MIBICI_ACTIVE_STATIONS
            else i,
            hour=hour,
            day_of_week=day_of_week,
            rng=rng,
        )
        for i in range(batch_size)
    ]
    return {"instances": instances}


def build_payload_pool(
    batch_size: int,
    pool_size: int = 8,
    base_seed: int = 2026,
) -> tuple[str, ...]:
    """Pre-serialize a ring of valid JSON request bodies to minimize Locust client CPU."""
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size!r}")
    return tuple(
        json.dumps(build_predict_payload(batch_size=batch_size, seed=base_seed + idx))
        for idx in range(pool_size)
    )
