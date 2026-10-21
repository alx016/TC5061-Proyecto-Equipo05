"""Small dispatcher for the MLflow ``search`` entry point.

Only P2 has a nested search for now.  P1 and P3 can add their own function here when their
experiment work packages need it.
"""

import typer

from tc5061_proyecto_equipo5.modeling import train_classification


def main(
    question: str = typer.Option(..., help="p1, p2 or p3"),
    model: str = typer.Option("lightgbm", help="Model name"),
) -> None:
    """Run the nested search of one question and model."""
    if question != "p2":
        raise typer.BadParameter("search is implemented for p2 only")
    train_classification.search(model)


if __name__ == "__main__":
    typer.run(main)
