"""Unit tests for the helpers shared by the training scripts."""

from pathlib import Path
import subprocess
from urllib.error import URLError

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from tc5061_proyecto_equipo5.modeling import common
from tc5061_proyecto_equipo5.modeling.common import base_tags, git_state


def test_git_state_reads_the_commit_of_the_repo():
    state = git_state()

    assert len(state["git_commit"]) == 40
    assert state["git_dirty"] in {"true", "false"}


def test_git_state_marks_uncommitted_changes(monkeypatch):
    def fake_run(command, **kwargs):
        stdout = "a" * 40 + "\n" if command[1] == "rev-parse" else " M features.py\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(common.subprocess, "run", fake_run)

    assert git_state() == {"git_commit": "a" * 40, "git_dirty": "true"}


def test_git_state_is_unknown_without_git(monkeypatch):
    def missing_git(command, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(common.subprocess, "run", missing_git)

    assert git_state() == {"git_commit": "unknown", "git_dirty": "unknown"}


def test_base_tags_carry_the_git_state():
    tags = base_tags("p1", "lightgbm", "baseline")

    assert tags["team"] == "equipo05"
    assert tags["question"] == "p1"
    assert {"git_commit", "git_dirty"} <= tags.keys()


def test_base_tags_derive_the_task_from_the_question():
    assert base_tags("p1", "lightgbm", "baseline")["task"] == "regression"
    assert base_tags("p3", "kmeans", "search")["task"] == "clustering"


@pytest.mark.parametrize("question, stage", [("p9", "baseline"), ("p1", "final")])
def test_base_tags_reject_unknown_values(question, stage):
    with pytest.raises(ValueError):
        base_tags(question, "lightgbm", stage)


def test_setup_experiment_fails_fast_without_server(monkeypatch):
    def unreachable(url, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(common.mlflow, "get_tracking_uri", lambda: "http://127.0.0.1:5001")
    monkeypatch.setattr(common, "urlopen", unreachable)

    with pytest.raises(RuntimeError, match="not reachable"):
        common.setup_experiment()


def test_log_run_context_writes_data_and_config(monkeypatch):
    logged = []
    monkeypatch.setattr(
        common.mlflow,
        "log_artifact",
        lambda path, artifact_path=None: logged.append((Path(path).name, artifact_path)),
    )
    monkeypatch.setattr(common.mlflow, "log_dict", lambda d, file: logged.append((file, d)))

    common.log_run_context({"question": "p1", "features_dir": Path("/tmp/x")})

    assert ("viajes_manifest.json", "data") in logged
    assert ("MLproject", "config") in logged
    assert ("python_env.yaml", "config") in logged
    assert ("config/run_config.json", {"question": "p1", "features_dir": "/tmp/x"}) in logged


def test_log_training_input_drops_personal_columns(monkeypatch):
    df = pd.DataFrame({"station_id": [1], "genero": ["F"], "anio_nacimiento": [1990]})
    received = []

    def fake_log_input(dataset, context=None):
        received.append(dataset)

    monkeypatch.setattr(common.mlflow, "log_input", fake_log_input)
    common.log_training_input(df, Path("/tmp/x.parquet"), "p1-train")

    assert len(received) == 1
    assert received[0].schema.input_names() == ["station_id"]
    assert received[0].name == "p1-train"


def test_log_model_with_signature_sends_signature_example_and_code(monkeypatch):
    X = pd.DataFrame({"hour": range(10), "lag_24": range(10, 20)})
    model = LinearRegression().fit(X, range(10))

    received = {}

    def fake_log_model(model, **kwargs):
        received["model"] = model
        received.update(kwargs)

    monkeypatch.setattr(common.mlflow.sklearn, "log_model", fake_log_model)

    common.log_model_with_signature(model, X, serialization_format="cloudpickle")

    assert received["model"] is model
    assert received["name"] == "model"
    assert received["signature"].inputs.input_names() == ["hour", "lag_24"]
    assert len(received["input_example"]) == 5
    assert received["code_paths"][0].endswith("tc5061_proyecto_equipo5")
    assert received["serialization_format"] == "cloudpickle"


def test_log_model_with_signature_rejects_unknown_flavor():
    with pytest.raises(ValueError):
        common.log_model_with_signature(None, None, flavor="lgbm")
