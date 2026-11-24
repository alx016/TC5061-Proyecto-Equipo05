"""Tests of the Registry seed against a local MLflow in tmp_path and a small bundle."""

import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import pytest
from scripts import seed_registry
from sklearn.linear_model import LogisticRegression

from tc5061_proyecto_equipo5.api.model import SIGNATURE_DTYPES, load_model, predict
from tc5061_proyecto_equipo5.modeling.train_classification import FEATURE_COLUMNS

NAME = "mibici-equipo05-p2-riesgo"
ORIGINAL_RUN = "run-original-v3"


def write_metadata(bundle: Path, source_run_id: str = ORIGINAL_RUN) -> None:
    """registry_metadata.json in the format export_model.py writes."""
    metadata = {
        "model_name": NAME,
        "model_version": "3",
        "model_alias": "champion",
        "source_run_id": source_run_id,
        "git_commit": "commit-de-prueba",
        "data_version": "4c54003c",
        "thresholds": {"shortage": 0.2, "saturation": 0.95},
        "exported_at": "2026-11-05T01:06:39Z",
    }
    (bundle / seed_registry.METADATA_FILE).write_text(json.dumps(metadata))


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A local tracking server in tmp_path; the global MLflow URIs are restored afterwards."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    yield uri
    mlflow.set_tracking_uri(previous[0])
    mlflow.set_registry_uri(previous[1])


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """A three-class sklearn model saved like the exported champion, with its metadata."""
    rng = np.random.default_rng(0)
    features = pd.DataFrame({column: rng.normal(size=60) for column in FEATURE_COLUMNS})
    features = features.astype(SIGNATURE_DTYPES)
    model = LogisticRegression(max_iter=500).fit(features, np.arange(60) % 3)
    path = tmp_path / "p2-riesgo"
    mlflow.sklearn.save_model(
        model,
        path,
        signature=infer_signature(features, model.predict_proba(features)),
        input_example=features.head(2),
        pyfunc_predict_fn="predict_proba",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    write_metadata(path)
    return path


def test_seed_registers_the_bundle_as_champion(tracking: str, bundle: Path):
    version = seed_registry.seed(bundle, tracking)
    client = MlflowClient()
    assert str(version.version) == "1"
    assert str(client.get_model_version_by_alias(NAME, "champion").version) == "1"
    assert version.tags["source_run_id_original"] == ORIGINAL_RUN
    assert version.tags["source_registry_version"] == "3"
    assert version.tags["validation_status"] == "approved"


def test_the_seeded_run_carries_what_the_api_reads(tracking: str, bundle: Path):
    version = seed_registry.seed(bundle, tracking)
    run = MlflowClient().get_run(version.tags["source_run_id"])
    assert run.data.params["threshold_shortage"] == "0.2"
    assert run.data.params["threshold_saturation"] == "0.95"
    assert run.data.params["data_version"] == "4c54003c"
    assert run.data.tags["git_commit"] == "commit-de-prueba"


def test_the_api_loads_and_scores_the_seeded_champion(tracking: str, bundle: Path):
    version = seed_registry.seed(bundle, tracking)
    loaded = load_model(f"models:/{NAME}@champion")
    assert loaded.info["model_version"] == "1"
    assert loaded.info["model_alias"] == "champion"
    # The new MLmodel points to the seeded run, not to the original one
    assert loaded.info["source_run_id"] == version.tags["source_run_id"]
    assert loaded.thresholds == {1: 0.2, 2: 0.95}
    record = {column: 1 for column in FEATURE_COLUMNS}
    assert len(predict(loaded, [record])) == 1


def test_seeding_twice_changes_nothing(tracking: str, bundle: Path):
    first = seed_registry.seed(bundle, tracking)
    second = seed_registry.seed(bundle, tracking)
    assert second.version == first.version
    assert len(MlflowClient().search_model_versions(f"name='{NAME}'")) == 1


def test_a_champion_from_another_source_is_not_moved(tracking: str, bundle: Path):
    seed_registry.seed(bundle, tracking)
    write_metadata(bundle, source_run_id="otro-run")
    with pytest.raises(seed_registry.SeedError, match="--force"):
        seed_registry.seed(bundle, tracking)
    assert str(MlflowClient().get_model_version_by_alias(NAME, "champion").version) == "1"


def test_force_moves_the_champion(tracking: str, bundle: Path):
    seed_registry.seed(bundle, tracking)
    write_metadata(bundle, source_run_id="otro-run")
    version = seed_registry.seed(bundle, tracking, force=True)
    assert str(version.version) == "2"
    assert str(MlflowClient().get_model_version_by_alias(NAME, "champion").version) == "2"


def test_missing_metadata_stops_with_a_clear_message(tracking: str, bundle: Path):
    (bundle / seed_registry.METADATA_FILE).unlink()
    with pytest.raises(seed_registry.SeedError, match="prepare-model"):
        seed_registry.seed(bundle, tracking)


def test_incomplete_metadata_names_the_missing_fields(tracking: str, bundle: Path):
    (bundle / seed_registry.METADATA_FILE).write_text(json.dumps({"model_version": "3"}))
    with pytest.raises(seed_registry.SeedError, match="source_run_id"):
        seed_registry.seed(bundle, tracking)


def test_a_server_that_is_down_fails_fast():
    with pytest.raises(seed_registry.SeedError, match="make up"):
        seed_registry.check_server("http://127.0.0.1:9")
