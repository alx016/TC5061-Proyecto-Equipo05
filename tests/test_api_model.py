"""API model loading: registry metadata matches the response schemas."""

from types import SimpleNamespace

import pytest

from tc5061_proyecto_equipo5.api import model


class FakeClient:
    """Stand-in for MlflowClient; the sqlite backend returns the version as an int."""

    def get_model_version_by_alias(self, name, alias):
        return SimpleNamespace(version=3)

    def get_model_version(self, name, version):
        return SimpleNamespace(version=int(version))


@pytest.fixture
def fake_mlflow(monkeypatch):
    monkeypatch.setattr(model, "MlflowClient", FakeClient)
    monkeypatch.setattr(model, "_run_metadata", lambda run_id: {"model_alias": None})


def test_registry_metadata_by_alias_returns_version_as_string(fake_mlflow):
    info = model._registry_metadata("models:/p2-riesgo@champion", "run-123")

    assert info["model_name"] == "p2-riesgo"
    assert info["model_alias"] == "champion"
    assert info["model_version"] == "3"


def test_registry_metadata_by_version_returns_version_as_string(fake_mlflow):
    info = model._registry_metadata("models:/p2-riesgo/5", "run-123")

    assert info["model_name"] == "p2-riesgo"
    assert info["model_alias"] is None
    assert info["model_version"] == "5"


def test_real_export_serves_smoke_request(monkeypatch):
    """The checked-in champion can be loaded and used through the HTTP contract."""
    import json
    from pathlib import Path

    from fastapi.testclient import TestClient

    from tc5061_proyecto_equipo5.api.main import app

    root = Path(__file__).resolve().parents[1]
    bundle = root / "models" / "p2-riesgo"
    if not (bundle / "MLmodel").is_file():
        pytest.skip("Run make prepare-model to test the real registry export")
    monkeypatch.setenv("MODEL_URI", str(bundle))
    with TestClient(app) as client:
        info = client.get("/model-info")
        response = client.post(
            "/predict", json=json.loads((root / "scripts/smoke_request.json").read_text())
        )
    assert info.status_code == response.status_code == 200
    assert info.json()["model_name"] == "mibici-equipo05-p2-riesgo"
    prediction = response.json()["predictions"][0]
    assert prediction["label"] == "Balanceado"
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
