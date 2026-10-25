"""Unit tests for the dispatcher behind the MLflow ``select`` entry point."""

from typer.testing import CliRunner

from tc5061_proyecto_equipo5.modeling import select, train_classification


def test_p3_reaches_the_clustering_select_and_none_means_no_run(monkeypatch):
    calls = []
    monkeypatch.setitem(
        select.SELECTORS, "p3", lambda run_id, allow_dirty: calls.append((run_id, allow_dirty))
    )

    result = CliRunner().invoke(select.app, ["--question", "p3", "--run-id", "none"])

    assert result.exit_code == 0, result.output
    assert calls == [(None, False)]


def test_p3_passes_an_explicit_run_id(monkeypatch):
    calls = []
    monkeypatch.setitem(
        select.SELECTORS, "p3", lambda run_id, allow_dirty: calls.append((run_id, allow_dirty))
    )

    result = CliRunner().invoke(select.app, ["--question", "p3", "--run-id", "abc123"])

    assert result.exit_code == 0, result.output
    assert calls == [("abc123", False)]


def test_p2_still_reaches_the_classification_select(monkeypatch):
    calls = []
    monkeypatch.setattr(train_classification, "select", lambda **kwargs: calls.append(kwargs))

    # p2 is the default, as it was before the dispatcher existed
    result = CliRunner().invoke(select.app, [])

    assert result.exit_code == 0, result.output
    assert calls == [{"question": "p2", "run_id": None, "allow_dirty": False}]


def test_question_without_select_is_rejected():
    result = CliRunner().invoke(select.app, ["--question", "p1"])

    assert result.exit_code != 0
    assert "p1 has no select available" in result.output
