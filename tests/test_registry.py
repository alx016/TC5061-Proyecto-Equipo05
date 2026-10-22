"""Model Registry: lineage tags, aliases and clear errors for the register entry point."""

from types import SimpleNamespace

import mlflow
from mlflow import MlflowClient
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from typer.testing import CliRunner

from tc5061_proyecto_equipo5.modeling import registry

P1_NAME = "mibici-equipo05-p1-demanda"


# --- Unit tests: no MLflow, a fake run with the same shape as a real one ---


def fake_run():
    """Minimal stand-in for an MLflow run (idea taken from Oscar's draft)."""
    data = SimpleNamespace(
        params={"data_version": "abc123", "features_data_version": "f456"},
        tags={
            "model_family": "lightgbm",
            "task": "regression",
            "git_commit": "deadbeef",
            "git_dirty": "false",
        },
    )
    return SimpleNamespace(data=data, info=SimpleNamespace(run_id="run-123", experiment_id="7"))


def test_model_names_follow_the_team_convention():
    assert registry.MODEL_NAMES == {
        "p1": "mibici-equipo05-p1-demanda",
        "p2": "mibici-equipo05-p2-riesgo",
        "p3": "mibici-equipo05-p3-grupos",
    }


def test_lineage_tags_preserve_run_data_and_code():
    tags = registry.lineage_tags("p1", fake_run(), "challenger")

    assert tags["question"] == "p1"
    assert tags["source_run_id"] == "run-123"
    assert tags["experiment_id"] == "7"
    assert tags["data_version"] == "abc123"
    assert tags["features_data_version"] == "f456"
    assert tags["git_commit"] == "deadbeef"
    assert tags["validation_status"] == "pending"


def test_champion_is_approved():
    tags = registry.lineage_tags("p1", fake_run(), "champion")

    assert tags["validation_status"] == "approved"


# --- End-to-end tests: a temporary tracking store, no server, nothing written to the repo ---


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    """Point MLflow to a throwaway SQLite store and hide the variables loaded from .env."""
    monkeypatch.chdir(tmp_path)
    for variable in ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_TRACKING_URI"):
        monkeypatch.delenv(variable, raising=False)
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield
    mlflow.set_tracking_uri(previous)


def logged_run(tags: dict) -> str:
    """Log a tiny model with the tags the training scripts would set."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    with mlflow.start_run() as run:
        mlflow.set_tags(tags)
        mlflow.log_param("data_version", "abc123")
        mlflow.sklearn.log_model(LinearRegression().fit(X, [0.0, 1.0, 2.0]), name="model")
    return run.info.run_id


def register(*args: str):
    return CliRunner().invoke(registry.app, list(args))


def test_register_adds_tags_and_challenger_alias(tracking):
    run_id = logged_run(
        {
            "question": "p1",
            "model_family": "ridge",
            "git_commit": "abc123",
            "git_dirty": "false",
            "selection_rule": "min val_wape",
        }
    )

    result = register("--run-id", run_id)

    assert result.exit_code == 0, result.output
    version = MlflowClient().get_model_version_by_alias(P1_NAME, "challenger")
    assert version.run_id == run_id
    assert version.tags["question"] == "p1"
    assert version.tags["validation_status"] == "pending"
    assert version.tags["data_version"] == "abc123"
    assert version.tags["selection_rule"] == "min val_wape"
    assert run_id in version.description


def test_new_champion_moves_the_alias(tracking):
    first = logged_run({"question": "p1", "git_dirty": "false"})
    second = logged_run({"question": "p1", "git_dirty": "false"})

    assert register("--run-id", first, "--alias", "champion").exit_code == 0
    assert register("--run-id", second, "--alias", "champion").exit_code == 0

    version = MlflowClient().get_model_version_by_alias(P1_NAME, "champion")
    assert version.run_id == second
    assert version.tags["validation_status"] == "approved"


def test_promoting_a_challenger_reuses_its_version(tracking):
    run_id = logged_run({"question": "p1", "git_dirty": "false"})

    assert register("--run-id", run_id).exit_code == 0
    assert register("--run-id", run_id, "--alias", "champion").exit_code == 0

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{P1_NAME}'")
    assert len(versions) == 1
    # Stores differ on the version type (int in SQLite, str over REST), so compare as text
    aliases = client.get_registered_model(P1_NAME).aliases
    assert {alias: str(number) for alias, number in aliases.items()} == {
        "champion": str(versions[0].version)
    }
    assert (
        client.get_model_version_by_alias(P1_NAME, "champion").tags["validation_status"]
        == "approved"
    )


def test_promoting_keeps_a_challenger_of_another_version(tracking):
    promoted = logged_run({"question": "p1", "git_dirty": "false"})
    newer = logged_run({"question": "p1", "git_dirty": "false"})

    # Both start as challengers; the alias ends on the newer one, then the older is promoted
    assert register("--run-id", promoted).exit_code == 0
    assert register("--run-id", newer).exit_code == 0
    assert register("--run-id", promoted, "--alias", "champion").exit_code == 0

    client = MlflowClient()
    assert len(client.search_model_versions(f"name='{P1_NAME}'")) == 2
    assert client.get_model_version_by_alias(P1_NAME, "champion").run_id == promoted
    assert client.get_model_version_by_alias(P1_NAME, "challenger").run_id == newer


def test_rejects_demoting_the_champion_to_challenger(tracking):
    run_id = logged_run({"question": "p1", "git_dirty": "false"})

    assert register("--run-id", run_id, "--alias", "champion").exit_code == 0
    assert register("--run-id", run_id).exit_code == 2

    client = MlflowClient()
    assert "challenger" not in client.get_registered_model(P1_NAME).aliases
    champion = client.get_model_version_by_alias(P1_NAME, "champion")
    assert champion.tags["validation_status"] == "approved"


def test_rejects_an_unknown_alias(tracking):
    run_id = logged_run({"question": "p1"})

    assert register("--run-id", run_id, "--alias", "production").exit_code == 2


def test_rejects_a_run_without_question(tracking):
    run_id = logged_run({"model_family": "ridge"})

    assert register("--run-id", run_id).exit_code == 2


def test_rejects_a_missing_run(tracking):
    assert register("--run-id", "does-not-exist").exit_code == 2


def test_rejects_a_run_without_model_and_registers_nothing(tracking):
    # Same shape as a search Study: the question tag is there, the model lives in its Trials
    with mlflow.start_run() as study:
        mlflow.set_tags({"question": "p1", "git_dirty": "false"})

    result = register("--run-id", study.info.run_id)

    assert result.exit_code == 2
    assert MlflowClient().search_registered_models() == []
