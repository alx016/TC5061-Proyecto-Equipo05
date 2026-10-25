"""Dispatch one question's search; the selected module owns its Study and Trials."""

from typing import Annotated

from loguru import logger
import typer

from tc5061_proyecto_equipo5.modeling import (
    train_classification,
    train_clustering,
    train_regression,
)
from tc5061_proyecto_equipo5.modeling.common import TASKS

app = typer.Typer()

SEARCHERS = {"p1": train_regression.search, "p3": train_clustering.search}
# Classification joins as soon as its experiment work package provides the searcher.
if callable(getattr(train_classification, "search", None)):
    SEARCHERS["p2"] = train_classification.search


@app.command()
def main(
    question: Annotated[str, typer.Option(help="Question to search (p1, p2 or p3)")],
    model: Annotated[str, typer.Option(help="Model family to search")] = "lightgbm",
):
    """Run a search with the selected question's own configuration and seed defaults."""
    if question not in TASKS:
        raise typer.BadParameter(f"question must be one of {sorted(TASKS)}, got {question!r}")
    if question not in SEARCHERS:
        raise typer.BadParameter(f"{question} has no search available (its experiment WP adds it)")
    try:
        run_id = SEARCHERS[question](model=model)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    logger.success(f"Study {run_id} logged")


if __name__ == "__main__":
    app()
