"""Validate that Locust payload builders satisfy API schemas without importing locust."""

import json
from pathlib import Path

from load_tests.payloads import (
    BATCH_SIZES,
    MAX_BATCH_SIZE,
    MIBICI_ACTIVE_STATIONS,
    build_payload_pool,
    build_predict_payload,
    build_station_features,
)
import pytest

from tc5061_proyecto_equipo5.api.schemas import PredictRequest, StationFeatures


def test_payloads_module_does_not_require_locust():
    payloads_source = (
        Path(__file__).resolve().parents[1] / "load_tests" / "payloads.py"
    ).read_text(encoding="utf-8")
    assert "import locust" not in payloads_source
    assert "from locust" not in payloads_source


def test_build_station_features_validates_against_schema():
    for idx in (0, 17, 150, 364, 400):
        station = build_station_features(station_index=idx)
        validated = StationFeatures.model_validate(station)
        assert validated.is_weekend == (validated.day_of_week >= 5)


@pytest.mark.parametrize("batch_size", [*BATCH_SIZES, MAX_BATCH_SIZE])
def test_build_predict_payload_validates_batches(batch_size):
    payload = build_predict_payload(batch_size=batch_size, seed=123 + batch_size)
    request = PredictRequest.model_validate(payload)
    assert len(request.instances) == batch_size


def test_full_mibici_batch_uses_distinct_stations():
    payload = build_predict_payload(batch_size=MIBICI_ACTIVE_STATIONS, seed=99)
    station_ids = [inst["station_id"] for inst in payload["instances"]]
    assert len(station_ids) == 365
    assert len(set(station_ids)) == 365


def test_build_predict_payload_is_deterministic_per_seed():
    first = build_predict_payload(batch_size=20, seed=2026)
    second = build_predict_payload(batch_size=20, seed=2026)
    third = build_predict_payload(batch_size=20, seed=2027)
    assert first == second
    assert first != third


@pytest.mark.parametrize("invalid_batch", [0, -1, MAX_BATCH_SIZE + 1])
def test_build_predict_payload_rejects_out_of_bounds_batch_size(invalid_batch):
    with pytest.raises(ValueError, match="batch_size"):
        build_predict_payload(batch_size=invalid_batch)


def test_build_payload_pool_returns_valid_json_bodies():
    pool = build_payload_pool(batch_size=20, pool_size=4, base_seed=500)
    assert len(pool) == 4
    for raw_json in pool:
        parsed = json.loads(raw_json)
        request = PredictRequest.model_validate(parsed)
        assert len(request.instances) == 20
