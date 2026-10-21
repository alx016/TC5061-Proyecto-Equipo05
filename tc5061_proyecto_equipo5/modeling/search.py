"""Run the P2 hyperparameter search through the MLflow entry point."""

import typer

from tc5061_proyecto_equipo5.modeling import train_classification


def main(
    question: str = typer.Option(..., help="p1, p2 or p3"),
    model: str = typer.Option("lightgbm", help="Model name"),
) -> None:
    """Run the nested search of one question and model."""
    if question != "p2":
        raise typer.BadParameter("question must be p2")
    train_classification.search(model)


if __name__ == "__main__":
    typer.run(main)
