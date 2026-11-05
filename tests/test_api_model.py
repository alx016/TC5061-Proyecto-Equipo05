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
