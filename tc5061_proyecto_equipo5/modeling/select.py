"""Dispatch one question's selection; the selected module scores December once."""

from typing import Annotated

from loguru import logger
import typer

from tc5061_proyecto_equipo5.modeling import train_classification, train_clustering
from tc5061_proyecto_equipo5.modeling.common import TASKS

app = typer.Typer()

# P1 has no select of its own: its regression command selects and scores December in one go
SELECTORS = {
    "p2": lambda run_id, allow_dirty: train_classification.select(
        question="p2", run_id=run_id, allow_dirty=allow_dirty
    ),
    "p3": train_clustering.select,
}


@app.command()
def main(
    question: Annotated[str, typer.Option(help="Question to select (p2 or p3)")] = "p2",
    run_id: Annotated[str, typer.Option(help="Run to select; none lets the rule choose")] = "none",
    allow_dirty: bool = False,
):
    """Choose one question's run on validation and score December once."""
    if question not in TASKS:
        raise typer.BadParameter(f"question must be one of {sorted(TASKS)}, got {question!r}")
    if question not in SELECTORS:
        raise typer.BadParameter(f"{question} has no select available")
    # MLproject cannot pass an empty value, so "none" means "let the selection rule choose"
    chosen = None if run_id.strip().lower() in {"", "none"} else run_id
    SELECTORS[question](run_id=chosen, allow_dirty=allow_dirty)
    logger.success(f"{question}: selection finished")


if __name__ == "__main__":
    app()
