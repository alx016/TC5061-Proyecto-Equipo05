"""Search entrypoint behavior at its CLI seam."""

from typer.testing import CliRunner

from tc5061_proyecto_equipo5.modeling import search


def test_p1_defaults_to_lightgbm_and_preserves_explicit_model(monkeypatch):
    calls = []

    def searcher(model="lightgbm"):
        calls.append(model)
        return "study-123"

    monkeypatch.setitem(search.SEARCHERS, "p1", searcher)
    for options in ([], ["--model", "lightgbm"]):
        result = CliRunner().invoke(search.app, ["--question", "p1", *options])
        assert result.exit_code == 0, result.output
    assert calls == ["lightgbm", "lightgbm"]


def test_p2_keeps_its_searcher_defaults_without_p1_seed(monkeypatch):
    calls = []

    def classification_search(model, seed=91):
        calls.append((model, seed))
        return "p2-study"

    monkeypatch.setitem(search.SEARCHERS, "p2", classification_search)
    result = CliRunner().invoke(search.app, ["--question", "p2", "--model", "logistic"])
    assert result.exit_code == 0, result.output
    assert calls == [("logistic", 91)]


def test_unknown_or_unavailable_questions_never_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(search, "SEARCHERS", {"p1": lambda **kwargs: calls.append(kwargs)})
    for question, message in (
        ("p9", "question must be"),
        ("p2", "no search available"),
        ("p3", "no search available"),
    ):
        result = CliRunner().invoke(search.app, ["--question", question])
        assert result.exit_code != 0
        assert message in result.output
    assert calls == []


def test_invalid_p1_model_is_a_clear_cli_error():
    result = CliRunner().invoke(search.app, ["--question", "p1", "--model", "ridge"])
    assert result.exit_code != 0
    assert "supports only 'lightgbm'" in result.output
