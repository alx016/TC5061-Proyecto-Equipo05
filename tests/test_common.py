"""Unit tests for the helpers shared by the training scripts."""

import subprocess
from urllib.error import URLError

import pytest
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