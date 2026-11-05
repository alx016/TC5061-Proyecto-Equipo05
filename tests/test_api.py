import math

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pytest

from tc5061_proyecto_equipo5.api.main import app, get_model
from tc5061_proyecto_equipo5.api.model import LoadedModel


class DummyPyFuncModel:
    def predict(self, df):
        return np.tile([0.1, 0.8, 0.1], (len(df), 1))


@pytest.fixture
def mock_lineage_info() -> dict:
    return {
        "model_name": "p2-riesgo-lgbm",
        "model_version": "1",
        "model_alias": "champion",
        "source_run_id": "mock-run-id-12345",
        "git_commit": "abcdef123456",
        "data_version": "v1.0.0",
        "thresholds": {"shortage": 0.35, "saturation": 0.40},
        "exported_at": "2026-11-03T00:00:00Z",
    }


@pytest.fixture
def mock_loaded_model(mock_lineage_info) -> LoadedModel:
    return LoadedModel(
        info=mock_lineage_info,
        model=DummyPyFuncModel(),
        classes=[0, 1, 2],
        thresholds={1: 0.35, 2: 0.40},
    )


@pytest.fixture
def client(mock_loaded_model):
    app.dependency_overrides[get_model] = lambda: mock_loaded_model
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def valid_station() -> dict:
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


def test_health_returns_200_when_model_is_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_returns_503_when_model_not_loaded(client):
    def mock_missing_model():
        raise HTTPException(status_code=503, detail="Model not loaded; check the API logs")

    app.dependency_overrides[get_model] = mock_missing_model
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded; check the API logs"


def test_health_returns_503_when_app_state_unloaded(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_URI", str(tmp_path / "missing-model"))
    monkeypatch.setattr("tc5061_proyecto_equipo5.api.model.MODELS_DIR", tmp_path)
    app.dependency_overrides.clear()
    app.state.loaded_model = None
    with TestClient(app, raise_server_exceptions=False) as unconfigured_client:
        response = unconfigured_client.get("/health")
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]


def test_model_info_returns_complete_lineage(client, mock_lineage_info):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == mock_lineage_info["model_name"]
    assert data["model_version"] == mock_lineage_info["model_version"]
    assert data["model_alias"] == mock_lineage_info["model_alias"]
    assert data["source_run_id"] == mock_lineage_info["source_run_id"]
    assert data["git_commit"] == mock_lineage_info["git_commit"]
    assert data["data_version"] == mock_lineage_info["data_version"]
    assert data["thresholds"] == mock_lineage_info["thresholds"]
    assert data["exported_at"] == mock_lineage_info["exported_at"]


def test_predict_returns_200_with_valid_predictions(client, valid_station):
    payload = {"instances": [valid_station]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["model_name"] == "p2-riesgo-lgbm"
    assert body["model_version"] == "1"
    assert body["thresholds"] == {"shortage": 0.35, "saturation": 0.40}
    assert len(body["predictions"]) == 1

    prediction = body["predictions"][0]
    assert prediction["station_id"] == 100
    assert prediction["label"] == "Riesgo_Desabasto"
    assert prediction["label_id"] == 1
    assert math.isclose(sum(prediction["probabilities"].values()), 1.0, abs_tol=0.01)


def test_predict_returns_200_with_multiple_instances(client, valid_station):
    station_a = dict(valid_station, station_id=1)
    station_b = dict(valid_station, station_id=2)
    payload = {"instances": [station_a, station_b]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["station_id"] == 1
    assert predictions[1]["station_id"] == 2
    for pred in predictions:
        assert pred["label"] in ["Balanceado", "Riesgo_Desabasto", "Riesgo_Saturacion"]
        assert math.isclose(sum(pred["probabilities"].values()), 1.0, abs_tol=0.01)


@pytest.mark.parametrize(
    ("mutate_key", "mutate_val"),
    [
        ("capacity", 0),
        ("capacity", -5),
        ("latitude", 20.39),
        ("latitude", 21.01),
        ("longitude", -103.61),
        ("longitude", -103.09),
        ("hour_sin", 1.5),
        ("hour_cos", -1.5),
        ("day_of_week", 7),
        ("is_weekend", True),
        ("arrivals_lag1", -0.5),
        ("departures_lag2", -1.0),
    ],
)
def test_predict_returns_422_on_out_of_range_values(
    client, valid_station, mutate_key, mutate_val
):
    invalid_station = dict(valid_station, **{mutate_key: mutate_val})
    response = client.post("/predict", json={"instances": [invalid_station]})
    assert response.status_code == 422


def test_predict_returns_422_on_extra_fields(client, valid_station):
    station_with_extra = dict(valid_station, extra_key="disallowed")
    response_station_extra = client.post("/predict", json={"instances": [station_with_extra]})
    assert response_station_extra.status_code == 422

    response_request_extra = client.post(
        "/predict", json={"instances": [valid_station], "unexpected_key": 123}
    )
    assert response_request_extra.status_code == 422


def test_predict_returns_422_on_missing_required_fields(client, valid_station):
    incomplete_station = dict(valid_station)
    del incomplete_station["capacity"]
    response = client.post("/predict", json={"instances": [incomplete_station]})
    assert response.status_code == 422


def test_predict_returns_422_on_empty_instances(client):
    response = client.post("/predict", json={"instances": []})
    assert response.status_code == 422


def test_predict_returns_422_on_exceeded_instances(client, valid_station):
    response = client.post("/predict", json={"instances": [valid_station] * 501})
    assert response.status_code == 422


def test_dependency_override_mechanism(mock_loaded_model):
    custom_info = dict(mock_loaded_model.info, model_alias="experimental")
    custom_model = LoadedModel(
        info=custom_info,
        model=mock_loaded_model.model,
        classes=mock_loaded_model.classes,
        thresholds=mock_loaded_model.thresholds,
    )

    app.dependency_overrides[get_model] = lambda: custom_model
    with TestClient(app) as test_client:
        response = test_client.get("/model-info")
        assert response.status_code == 200
        assert response.json()["model_alias"] == "experimental"
    app.dependency_overrides.clear()
