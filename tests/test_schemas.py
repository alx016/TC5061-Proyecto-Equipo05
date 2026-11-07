import math

from pydantic import ValidationError
import pytest

from tc5061_proyecto_equipo5.api.schemas import (
    ModelInfo,
    Prediction,
    PredictRequest,
    PredictResponse,
    StationFeatures,
)


@pytest.fixture
def valid_station_payload() -> dict:
    return {
        "station_id": 100,
        "capacity": 15,
        "latitude": 20.666378,
        "longitude": -103.34882,
        "hour_sin": 0.0,
        "hour_cos": 1.0,
        "day_of_week": 1,
        "is_weekend": False,
        "is_holiday": False,
        "neighbor_net_flow": 0.0222,
        "net_flow_lag1": -1.5,
        "net_flow_lag2": 2.0,
        "net_flow_lag24": 0.0,
        "arrivals_lag1": 0.0,
        "arrivals_lag2": 1.0,
        "arrivals_lag3": 4.0,
        "departures_lag1": 0.0,
        "departures_lag2": 2.0,
        "departures_lag3": 4.0,
    }


def test_valid_single_station(valid_station_payload):
    station = StationFeatures(**valid_station_payload)
    assert station.station_id == 100
    assert station.capacity == 15
    assert math.isclose(station.latitude, 20.666378)
    assert math.isclose(station.longitude, -103.34882)
    assert station.day_of_week == 1
    assert station.is_weekend is False


def test_station_id_optional_handling(valid_station_payload):
    payload_without_id = dict(valid_station_payload)
    del payload_without_id["station_id"]
    station_default = StationFeatures(**payload_without_id)
    assert station_default.station_id is None

    payload_with_none = dict(valid_station_payload, station_id=None)
    station_none = StationFeatures(**payload_with_none)
    assert station_none.station_id is None

    payload_with_int = dict(valid_station_payload, station_id=42)
    station_int = StationFeatures(**payload_with_int)
    assert station_int.station_id == 42


def test_valid_batch_of_500_instances(valid_station_payload):
    instances = [StationFeatures(**valid_station_payload) for _ in range(500)]
    request = PredictRequest(instances=instances)
    assert len(request.instances) == 500


def test_batch_empty_rejected():
    with pytest.raises(ValidationError) as exc_info:
        PredictRequest(instances=[])
    assert "instances" in str(exc_info.value)


def test_batch_501_rejected(valid_station_payload):
    instances = [StationFeatures(**valid_station_payload) for _ in range(501)]
    with pytest.raises(ValidationError) as exc_info:
        PredictRequest(instances=instances)
    assert "instances" in str(exc_info.value)


def test_extra_forbidden_fields_rejected(valid_station_payload):
    with pytest.raises(ValidationError):
        StationFeatures(**valid_station_payload, unexpected_field="disallowed")

    station = StationFeatures(**valid_station_payload)
    with pytest.raises(ValidationError):
        PredictRequest(instances=[station], unexpected_field="disallowed")


@pytest.mark.parametrize(
    "missing_key",
    [
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
    ],
)
def test_missing_required_fields_rejected(valid_station_payload, missing_key):
    payload = dict(valid_station_payload)
    del payload[missing_key]
    with pytest.raises(ValidationError):
        StationFeatures(**payload)


@pytest.mark.parametrize("capacity", [0, -1, -50])
def test_out_of_bounds_capacity_rejected(valid_station_payload, capacity):
    with pytest.raises(ValidationError):
        StationFeatures(**dict(valid_station_payload, capacity=capacity))


@pytest.mark.parametrize(
    ("lat", "lon", "is_valid"),
    [
        (20.4, -103.6, True),
        (21.0, -103.1, True),
        (20.399, -103.35, False),
        (21.001, -103.35, False),
        (20.7, -103.601, False),
        (20.7, -103.099, False),
    ],
)
def test_latitude_longitude_bounds(valid_station_payload, lat, lon, is_valid):
    payload = dict(valid_station_payload, latitude=lat, longitude=lon)
    if is_valid:
        station = StationFeatures(**payload)
        assert station.latitude == lat
        assert station.longitude == lon
    else:
        with pytest.raises(ValidationError):
            StationFeatures(**payload)


@pytest.mark.parametrize(
    ("hour_sin", "hour_cos", "is_valid"),
    [
        (-1.0, 1.0, True),
        (1.0, -1.0, True),
        (0.0, 0.0, True),
        (-1.001, 0.0, False),
        (1.001, 0.0, False),
        (0.0, -1.001, False),
        (0.0, 1.001, False),
    ],
)
def test_cyclic_hour_bounds(valid_station_payload, hour_sin, hour_cos, is_valid):
    payload = dict(valid_station_payload, hour_sin=hour_sin, hour_cos=hour_cos)
    if is_valid:
        station = StationFeatures(**payload)
        assert station.hour_sin == hour_sin
        assert station.hour_cos == hour_cos
    else:
        with pytest.raises(ValidationError):
            StationFeatures(**payload)


@pytest.mark.parametrize("day_of_week", [-1, 7, 10])
def test_out_of_bounds_day_of_week_rejected(valid_station_payload, day_of_week):
    with pytest.raises(ValidationError):
        StationFeatures(**dict(valid_station_payload, day_of_week=day_of_week))


@pytest.mark.parametrize(
    ("day_of_week", "is_weekend", "should_succeed"),
    [
        (0, False, True),
        (1, False, True),
        (2, False, True),
        (3, False, True),
        (4, False, True),
        (0, True, False),
        (1, True, False),
        (2, True, False),
        (3, True, False),
        (4, True, False),
        (5, True, True),
        (6, True, True),
        (5, False, False),
        (6, False, False),
    ],
)
def test_temporal_consistency_weekend_validation(
    valid_station_payload, day_of_week, is_weekend, should_succeed
):
    payload = dict(valid_station_payload, day_of_week=day_of_week, is_weekend=is_weekend)
    if should_succeed:
        station = StationFeatures(**payload)
        assert station.day_of_week == day_of_week
        assert station.is_weekend == is_weekend
    else:
        with pytest.raises(ValidationError):
            StationFeatures(**payload)


@pytest.mark.parametrize(
    "lag_field",
    [
        "arrivals_lag1",
        "arrivals_lag2",
        "arrivals_lag3",
        "departures_lag1",
        "departures_lag2",
        "departures_lag3",
    ],
)
def test_negative_arrival_and_departure_lags_rejected(valid_station_payload, lag_field):
    with pytest.raises(ValidationError):
        StationFeatures(**dict(valid_station_payload, **{lag_field: -0.01}))

    valid_zero = StationFeatures(**dict(valid_station_payload, **{lag_field: 0.0}))
    assert getattr(valid_zero, lag_field) == 0.0


@pytest.mark.parametrize("flow_val", [-100.5, 0.0, 42.75])
def test_net_flow_positive_and_negative_allowed(valid_station_payload, flow_val):
    payload = dict(
        valid_station_payload,
        neighbor_net_flow=flow_val,
        net_flow_lag1=flow_val,
        net_flow_lag2=flow_val,
        net_flow_lag24=flow_val,
    )
    station = StationFeatures(**payload)
    assert station.neighbor_net_flow == flow_val
    assert station.net_flow_lag1 == flow_val
    assert station.net_flow_lag2 == flow_val
    assert station.net_flow_lag24 == flow_val


def test_prediction_probabilities_sum_validation():
    valid = Prediction(
        station_id=1,
        label="Riesgo_Desabasto",
        label_id=1,
        probabilities={"Balanceado": 0.1, "Riesgo_Desabasto": 0.8, "Riesgo_Saturacion": 0.1},
    )
    assert valid.label_id == 1

    valid_tolerance = Prediction(
        label="Balanceado",
        label_id=0,
        probabilities={"Balanceado": 0.995, "Riesgo_Desabasto": 0.005, "Riesgo_Saturacion": 0.005},
    )
    assert valid_tolerance.label == "Balanceado"

    with pytest.raises(ValidationError):
        Prediction(
            label="Riesgo_Desabasto",
            label_id=1,
            probabilities={"Balanceado": 0.1, "Riesgo_Desabasto": 0.2, "Riesgo_Saturacion": 0.1},
        )

    with pytest.raises(ValidationError):
        Prediction(
            label="Balanceado",
            label_id=3,
            probabilities={"Balanceado": 1.0, "Riesgo_Desabasto": 0.0, "Riesgo_Saturacion": 0.0},
        )

    with pytest.raises(ValidationError):
        Prediction(
            label="Balanceado",
            label_id=0,
            probabilities={"Balanceado": 1.0, "Riesgo_Desabasto": 0.0, "Riesgo_Saturacion": 0.0},
            unexpected="forbid",
        )


def test_predict_response_validation():
    prediction = Prediction(
        station_id=10,
        label="Balanceado",
        label_id=0,
        probabilities={"Balanceado": 0.8, "Riesgo_Desabasto": 0.1, "Riesgo_Saturacion": 0.1},
    )
    response = PredictResponse(
        model_name="p2-riesgo",
        model_version="1",
        thresholds={"shortage": 0.35, "saturation": 0.40},
        predictions=[prediction],
    )
    assert response.model_name == "p2-riesgo"
    assert response.predictions[0].station_id == 10

    with pytest.raises(ValidationError):
        PredictResponse(predictions=[prediction], unexpected="forbidden")


def test_model_info_validation():
    info = ModelInfo(
        model_name="p2-riesgo",
        model_version="2",
        model_alias="champion",
        source_run_id="run-123",
        git_commit="abcdef01",
        data_version="v1.0.0",
        thresholds={"shortage": 0.35, "saturation": 0.40},
        exported_at="2026-11-03T00:00:00Z",
    )
    assert info.model_name == "p2-riesgo"
    assert info.model_alias == "champion"

    info_defaults = ModelInfo()
    assert info_defaults.model_name is None
    assert info_defaults.thresholds == {}

    with pytest.raises(ValidationError):
        ModelInfo(unexpected_field="forbid")


@pytest.mark.parametrize(
    "changes",
    [
        {"label": "wrong"},
        {"label_id": 2},
        {"probabilities": {"Balanceado": 2, "Riesgo_Desabasto": -1, "Riesgo_Saturacion": 0}},
        {"probabilities": {"wrong": 1}},
        {
            "probabilities": {
                "Balanceado": float("nan"),
                "Riesgo_Desabasto": 0,
                "Riesgo_Saturacion": 0,
            }
        },
    ],
)
def test_invalid_prediction_contract_rejected(changes):
    payload = {
        "label": "Balanceado",
        "label_id": 0,
        "probabilities": {"Balanceado": 0.8, "Riesgo_Desabasto": 0.1, "Riesgo_Saturacion": 0.1},
    }
    payload.update(changes)
    with pytest.raises(ValidationError):
        Prediction(**payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("capacity", 2147483648),
        ("neighbor_net_flow", float("inf")),
        ("net_flow_lag1", float("nan")),
    ],
)
def test_non_finite_and_overflow_inputs_rejected(valid_station_payload, field, value):
    with pytest.raises(ValidationError):
        StationFeatures(**dict(valid_station_payload, **{field: value}))
