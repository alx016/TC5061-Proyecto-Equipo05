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


def test_p2_dispatches_to_the_classification_trainer(monkeypatch):
    calls = []
    monkeypatch.setitem(train.TRAINERS, "p2", fake_trainer(calls))

    result = CliRunner().invoke(train.app, ["--question", "p2", "--model", "lightgbm"])

    assert result.exit_code == 0, result.output
    assert calls == [("lightgbm", "candidate", SEED, {})]


def test_integer_options_are_cast_from_mlflow_floats(monkeypatch):
    calls = []
    monkeypatch.setitem(train.TRAINERS, "p2", fake_trainer(calls))

    result = CliRunner().invoke(
        train.app,
        ["--question", "p2", "--model", "lightgbm", "--n-estimators=300.0", "--max-depth=6.0"],
    )

    assert result.exit_code == 0, result.output
    # MLflow formats numeric parameters as floats; the trainer must receive integers.
    assert calls == [("lightgbm", "candidate", SEED, {"n_estimators": 300, "max_depth": 6})]


def test_p1_declares_its_own_integer_parameters(monkeypatch):
    calls = []
    monkeypatch.setitem(train.TRAINERS, "p1", fake_trainer(calls))

    result = CliRunner().invoke(
        train.app, ["--question", "p1", "--model", "lightgbm", "--num-leaves=31.0"]
    )

    assert result.exit_code == 0, result.output
    assert calls == [("lightgbm", "candidate", SEED, {"num_leaves": 31})]


def test_fractional_integer_option_is_rejected(monkeypatch):
    calls = []
    monkeypatch.setitem(train.TRAINERS, "p2", fake_trainer(calls))

    result = CliRunner().invoke(
        train.app, ["--question", "p2", "--model", "lightgbm", "--n-estimators=300.5"]
    )

    assert result.exit_code != 0
    assert "n_estimators must be an integer" in result.output
    assert calls == []


def test_unknown_question_stage_or_missing_trainer_is_rejected(monkeypatch):
    monkeypatch.setitem(train.TRAINERS, "p1", fake_trainer([]))

    for args in (
        ["--question", "p9", "--model", "x"],
        ["--question", "p3", "--model", "kmeans"],
        ["--question", "p1", "--model", "ridge", "--stage", "final"],
    ):
        result = CliRunner().invoke(train.app, args)
        assert result.exit_code != 0, args
