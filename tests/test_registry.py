"""Model Registry: lineage tags, aliases and clear errors for the register entry point."""

from types import SimpleNamespace

import mlflow
from mlflow import MlflowClient
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from typer.testing import CliRunner

from tc5061_proyecto_equipo5.modeling import registry, train_clustering

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


def test_features_data_version_falls_back_to_feature_data_version():
    run = fake_run()
    run.data.params = {"data_version": "abc123", "feature_data_version": "f789"}

    tags = registry.lineage_tags("p1", run, "challenger")

    assert tags["features_data_version"] == "f789"


def test_champion_is_approved():
    tags = registry.lineage_tags("p1", fake_run(), "champion")

    assert tags["validation_status"] == "approved"


# --- End-to-end tests: a temporary tracking store, no server, nothing written to the repo ---


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    """Point MLflow to a throwaway SQLite store and hide the variables loaded from .env."""
    monkeypatch.chdir(tmp_path)
    for variable in (
        "MLFLOW_RUN_ID",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_REGISTRY_URI",
    ):
        monkeypatch.delenv(variable, raising=False)
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield
    mlflow.set_tracking_uri(previous)


def logged_run(tags: dict, *, fitted=True, example=True, data_version="abc123") -> str:
    """Log a tiny model with the tags the training scripts would set."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    with mlflow.start_run() as run:
        mlflow.set_tags({"git_commit": "abc123", **tags})
        if data_version is not None:
            mlflow.log_param("data_version", data_version)
        model = LinearRegression()
        if fitted:
            model.fit(X, [0.0, 1.0, 2.0])
        mlflow.sklearn.log_model(model, name="model", input_example=X if example else None)
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
    model = MlflowClient().get_registered_model(P1_NAME)
    assert "WAPE" in model.description
    assert model.tags["question"] == "p1"


def test_new_champion_moves_the_alias(tracking):
    first = logged_run({"question": "p1", "git_dirty": "false", "selected": "true"})
    second = logged_run({"question": "p1", "git_dirty": "false", "selected": "true"})

    assert register("--run-id", first, "--alias", "champion").exit_code == 0
    assert register("--run-id", second, "--alias", "champion").exit_code == 0

    version = MlflowClient().get_model_version_by_alias(P1_NAME, "champion")
    assert version.run_id == second
    assert version.tags["validation_status"] == "approved"


def test_promoting_a_challenger_reuses_its_version(tracking):
    run_id = logged_run({"question": "p1", "git_dirty": "false", "selected": "true"})

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


def test_reusing_a_version_keeps_its_description(tracking):
    run_id = logged_run({"question": "p1", "git_dirty": "false", "selected": "true"})
    assert register("--run-id", run_id).exit_code == 0
    client = MlflowClient()
    version = client.get_model_version_by_alias(P1_NAME, "challenger").version
    client.update_model_version(P1_NAME, version, description="Revisado por el equipo")

    assert register("--run-id", run_id, "--alias", "champion").exit_code == 0

    assert client.get_model_version(P1_NAME, version).description == "Revisado por el equipo"


def test_promoting_keeps_a_challenger_of_another_version(tracking):
    promoted = logged_run({"question": "p1", "git_dirty": "false", "selected": "true"})
    newer = logged_run({"question": "p1", "git_dirty": "false", "selected": "true"})

    # Both start as challengers; the alias ends on the newer one, then the older is promoted
    assert register("--run-id", promoted).exit_code == 0
    assert register("--run-id", newer).exit_code == 0
    assert register("--run-id", promoted, "--alias", "champion").exit_code == 0

    client = MlflowClient()
    assert len(client.search_model_versions(f"name='{P1_NAME}'")) == 2
    assert client.get_model_version_by_alias(P1_NAME, "champion").run_id == promoted
    assert client.get_model_version_by_alias(P1_NAME, "challenger").run_id == newer


def test_rejects_demoting_the_champion_to_challenger(tracking):
    run_id = logged_run({"question": "p1", "git_dirty": "false", "selected": "true"})

    assert register("--run-id", run_id, "--alias", "champion").exit_code == 0
    assert register("--run-id", run_id).exit_code == 2

    client = MlflowClient()
    assert "challenger" not in client.get_registered_model(P1_NAME).aliases
    champion = client.get_model_version_by_alias(P1_NAME, "champion")
    assert champion.tags["validation_status"] == "approved"


def test_champion_loads_by_alias_and_predicts(tracking):
    run_id = logged_run({"question": "p1", "git_dirty": "false", "selected": "true"})
    assert register("--run-id", run_id, "--alias", "champion").exit_code == 0

    model = mlflow.pyfunc.load_model(f"models:/{P1_NAME}@champion")

    assert list(model.predict(pd.DataFrame({"x": [3.0]}))) == pytest.approx([3.0])


def test_rejects_a_dirty_champion(tracking):
    run_id = logged_run({"question": "p1", "git_dirty": "true", "selected": "true"})

    assert register("--run-id", run_id, "--alias", "champion").exit_code == 2
    assert MlflowClient().search_registered_models() == []


def test_rejects_an_unselected_champion(tracking):
    run_id = logged_run({"question": "p1", "git_dirty": "false"})

    assert register("--run-id", run_id, "--alias", "champion").exit_code == 2
    assert MlflowClient().search_registered_models() == []


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


@pytest.mark.parametrize("field", ["git_commit", "data_version"])
@pytest.mark.parametrize("value", [None, "", "unknown"])
def test_rejects_incomplete_lineage_before_registration(tracking, field, value):
    tags = {"question": "p1", "git_dirty": "false", "selected": "true"}
    client = MlflowClient()
    if field == "data_version":
        run_id = logged_run(tags, data_version=value)
    else:
        run_id = logged_run({**tags, "git_commit": value or ""})
        if value is None:
            client.delete_tag(run_id, "git_commit")
    result = register("--run-id", run_id, "--alias", "champion")
    assert result.exit_code == 2, result.output
    assert field in result.output
    assert client.search_registered_models() == []


@pytest.mark.parametrize("dirty", [None, "unknown", ""])
def test_rejects_champion_without_confirmed_clean_code(tracking, dirty):
    tags = {"question": "p1", "selected": "true"}
    if dirty is not None:
        tags["git_dirty"] = dirty
    run_id = logged_run(tags)
    assert register("--run-id", run_id, "--alias", "champion").exit_code == 2
    assert MlflowClient().search_registered_models() == []


@pytest.mark.parametrize("fitted,example", [(False, True), (True, False)])
def test_failed_preflight_preserves_existing_aliases(tracking, fitted, example):
    tags = {"question": "p1", "git_dirty": "false", "selected": "true"}
    champion = logged_run(tags)
    assert register("--run-id", champion, "--alias", "champion").exit_code == 0
    candidate = logged_run(tags, fitted=fitted, example=example)
    assert register("--run-id", candidate).exit_code == 0
    client = MlflowClient()
    before = client.get_registered_model(P1_NAME).aliases

    result = register("--run-id", candidate, "--alias", "champion")

    assert result.exit_code == 2, result.output
    assert "pre-flight" in result.output
    assert client.get_registered_model(P1_NAME).aliases == before
    challenger = client.get_model_version_by_alias(P1_NAME, "challenger")
    assert challenger.tags["validation_status"] == "pending"


def test_p3_explicit_selection_can_be_promoted(tracking):
    run_id = logged_run({"question": "p3", "git_dirty": "false"})
    assert register("--run-id", run_id, "--alias", "champion").exit_code == 2
    result = CliRunner().invoke(
        train_clustering.app,
        ["select", "--run-id", run_id, "--reason", "Reviewed validation silhouette and profiles"],
    )
    assert result.exit_code == 0, result.output
    assert register("--run-id", run_id, "--alias", "champion").exit_code == 0
    version = MlflowClient().get_model_version_by_alias(registry.MODEL_NAMES["p3"], "champion")
    assert version.tags["selected"] == "true"
    assert version.tags["selection_reason"] == "Reviewed validation silhouette and profiles"


def test_p3_training_logs_an_operational_example(tracking, monkeypatch):
    import numpy as np
    from sklearn.cluster import KMeans

    from tc5061_proyecto_equipo5.modeling import common

    matrix = np.array([[0.0], [0.1], [5.0], [5.1]])
    monkeypatch.setattr(
        common, "git_state", lambda: {"git_commit": "abc123", "git_dirty": "false"}
    )
    monkeypatch.setattr(train_clustering, "data_version", lambda: "abc123")
    monkeypatch.setattr(train_clustering, "save_artifacts", lambda *args: None)
    data = {
        "matrices": {split: matrix for split in train_clustering.SPLITS},
        "windows": {split: "2024" for split in train_clustering.SPLITS},
        "manifest": {"feature_version": "v1", "data_version": "features123"},
        "k_table": pd.DataFrame({"k": [2]}),
        "keys": {"train": pd.DataFrame()},
        "names": ["profile"],
        "stations": pd.DataFrame(),
    }
    record = train_clustering.train_and_log(
        "p3-test",
        KMeans(n_clusters=2, n_init=1, random_state=42),
        "kmeans",
        data,
        42,
        {},
        "Synthetic profiles",
        True,
    )
    uri = f"runs:/{record['run_id']}/model"
    model = mlflow.pyfunc.load_model(uri)
    assert model.metadata.signature is not None
    example = model.metadata.load_input_example(uri)
    assert example is not None
    assert len(model.predict(example)) == len(matrix)


@pytest.mark.parametrize("question,reason", [("p1", "reviewed"), ("p3", " ")])
def test_p3_selection_rejects_wrong_question_or_empty_reason(tracking, question, reason):
    run_id = logged_run({"question": question, "git_dirty": "false"})
    result = CliRunner().invoke(
        train_clustering.app, ["select", "--run-id", run_id, "--reason", reason]
    )
    assert result.exit_code == 2, result.output
    assert "selected" not in MlflowClient().get_run(run_id).data.tags


def test_p3_selection_rejects_a_run_without_model(tracking):
    with mlflow.start_run() as run:
        mlflow.set_tag("question", "p3")
    result = CliRunner().invoke(
        train_clustering.app, ["select", "--run-id", run.info.run_id, "--reason", "reviewed"]
    )
    assert result.exit_code == 2, result.output
    assert "selected" not in MlflowClient().get_run(run.info.run_id).data.tags


def test_reused_version_repairs_unknown_lineage_from_known_source(tracking):
    run_id = logged_run({"question": "p1", "git_dirty": "false", "selected": "true"})
    assert register("--run-id", run_id).exit_code == 0
    client = MlflowClient()
    version = client.get_model_version_by_alias(P1_NAME, "challenger").version
    client.set_model_version_tag(P1_NAME, version, "git_commit", "unknown")
    assert register("--run-id", run_id, "--alias", "champion").exit_code == 0
    assert client.get_model_version(P1_NAME, version).tags["git_commit"] == "abc123"
