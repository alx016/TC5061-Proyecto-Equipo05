"""Run the hyperparameter search of one question through the MLflow entry point."""

import typer

from tc5061_proyecto_equipo5.modeling import train_classification

# Each question registers its search implementation here, as ``train.py`` does with TRAINERS.
SEARCHERS = {
    "p2": train_classification.search,
}


def main(
    question: str = typer.Option(..., help="p1, p2 or p3"),
    model: str = typer.Option("lightgbm", help="Model name"),
) -> None:
    """Run the nested search of one question and model."""
    if question not in SEARCHERS:
        raise typer.BadParameter(
            f"search is not implemented for {question!r}; available: {sorted(SEARCHERS)}"
        )
    SEARCHERS[question](model)


if __name__ == "__main__":
    typer.run(main)
