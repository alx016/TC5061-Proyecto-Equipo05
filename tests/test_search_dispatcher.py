"""Unit tests for the dispatcher behind the MLflow ``search`` entry point."""

import pytest
import typer

from tc5061_proyecto_equipo5.modeling import search


def test_search_rejects_questions_without_an_implementation():
    with pytest.raises(typer.BadParameter, match="p1"):
        search.main(question="p1", model="lightgbm")


def test_search_dispatches_to_the_registered_implementation(monkeypatch):
    calls = []
    monkeypatch.setitem(search.SEARCHERS, "p2", lambda model: calls.append(model))

    search.main(question="p2", model="lightgbm")

    assert calls == ["lightgbm"]
