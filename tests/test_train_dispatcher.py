"""Unit tests for the dispatcher behind the MLflow ``train`` entry point."""

from typer.testing import CliRunner

from tc5061_proyecto_equipo5.config import SEED
from tc5061_proyecto_equipo5.modeling import train


def fake_trainer(calls):
    def trainer(model, stage, seed, hyperparams):
        calls.append((model, stage, seed, hyperparams))
        return "run-123"

    return trainer


def test_only_the_given_hyperparameters_reach_the_trainer(monkeypatch):
    calls = []
    monkeypatch.setitem(train.TRAINERS, "p1", fake_trainer(calls))

    result = CliRunner().invoke(
        train.app,
        ["--question", "p1", "--model", "lightgbm", "--learning-rate", "0.03", "--num-leaves=-1"],
    )

    assert result.exit_code == 0, result.output
    # -1 means "not set", so num_leaves is left out and the seed comes from config.py
    assert calls == [("lightgbm", "candidate", SEED, {"learning_rate": 0.03})]


def test_unknown_question_stage_or_missing_trainer_is_rejected(monkeypatch):
    monkeypatch.setitem(train.TRAINERS, "p1", fake_trainer([]))

    for args in (
        ["--question", "p9", "--model", "x"],
        ["--question", "p2", "--model", "lightgbm"],
        ["--question", "p1", "--model", "ridge", "--stage", "final"],
    ):
        result = CliRunner().invoke(train.app, args)
        assert result.exit_code != 0, args
