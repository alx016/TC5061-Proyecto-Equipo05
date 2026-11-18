"""Contract tests for single, medium, and full-network batch predictions (1, 20, 365 stations)."""

import json
import math
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from tc5061_proyecto_equipo5.api.main import app, get_model
from tc5061_proyecto_equipo5.api.model import LoadedModel


class DummyPyFuncModel:
    """In-memory test double that dynamically outputs probabilities for any batch length."""

    def predict(self, df):
        return np.tile([0.1, 0.8, 0.1], (len(df), 1))


@pytest.fixture
def mock_lineage_info() -> dict:
    return {
        "model_name": "mibici-equipo05-p2-riesgo",
        "model_version": "3",
        "model_alias": "champion",
        "source_run_id": "mock-run-id-batch-contract",
        "git_commit": "mock-commit-hash",
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
def reference_station() -> dict:
    root = Path(__file__).resolve().parents[1]
    smoke_file = root / "scripts" / "smoke_request.json"
    data = json.loads(smoke_file.read_text(encoding="utf-8"))
    return data["instances"][0]


@pytest.mark.parametrize("batch_size", [1, 20, 365])
def test_batch_prediction_cardinality_and_station_traceability(
    client, reference_station, batch_size
):
    instances = [dict(reference_station, station_id=sid) for sid in range(1, batch_size + 1)]
    payload = {"instances": instances}

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    predictions = data["predictions"]
    assert len(predictions) == batch_size

    valid_labels = {"Balanceado", "Riesgo_Desabasto", "Riesgo_Saturacion"}
    for idx, (expected_sid, pred) in enumerate(
        zip(range(1, batch_size + 1), predictions, strict=True)
    ):
        assert pred["station_id"] == expected_sid, (
            f"Index {idx} expected station_id {expected_sid}, got {pred['station_id']}"
        )
        assert pred["label"] in valid_labels
        assert pred["label_id"] in {0, 1, 2}
        assert set(pred["probabilities"].keys()) == valid_labels
        assert math.isclose(sum(pred["probabilities"].values()), 1.0, abs_tol=0.01)


def test_dependency_overrides_cleaned_up_after_tests():
    assert len(app.dependency_overrides) == 0
