"""Train one model of one question: the command behind the MLflow ``train`` entry point.

``mlflow run . -e train -P question=p1 -P model=lightgbm -P hp_learning_rate=0.03`` ends here.
MLproject cannot leave a parameter empty, so every hyperparameter defaults to -1, which means "not
set": the model keeps its own default and only the values given on the command line override it.
The MLproject names carry an ``hp_`` prefix because ``mlflow run`` logs them as run params, and
the training script logs the resolved values (seed=42, learning_rate=0.05...) under the plain
names; MLflow refuses to change a param that is already logged.

Each question exposes ``train_one(model, stage, seed, hyperparams) -> run_id`` in its own script
and is listed in ``TRAINERS``.  One call is one run: inside ``mlflow run`` the first
``start_run`` resumes the run that MLflow already opened for the entry point, so a loop of models
here would leave the project tags on the first model only.
"""

from typing import Annotated

from loguru import logger
import typer

from tc5061_proyecto_equipo5.config import SEED
from tc5061_proyecto_equipo5.modeling import train_classification, train_regression
from tc5061_proyecto_equipo5.modeling.common import STAGES, TASKS

app = typer.Typer()

TRAINERS = {
    "p1": train_regression.train_one,
    "p2": train_classification.train_one,
}
# The "not set" value of every numeric option (seeds and hyperparameters are never negative)
UNSET = -1
# MLflow passes every numeric value as text; these ones must reach sklearn as integers.
INTEGER_PARAMS = {
    "n_estimators",
    "num_leaves",
    "min_child_samples",
    "max_depth",
    "min_samples_leaf",
    "max_leaf_nodes",
    "n_clusters",
}


@app.command()
def main(
    question: Annotated[str, typer.Option(help="p1, p2 or p3")],
    model: Annotated[str, typer.Option(help="Model name, e.g. seasonal_mean or lightgbm")],
    stage: Annotated[str, typer.Option(help="baseline, search or candidate")] = "candidate",
    seed: int = UNSET,
    n_estimators: float = UNSET,
    learning_rate: float = UNSET,
    num_leaves: float = UNSET,
    min_child_samples: float = UNSET,
    max_depth: float = UNSET,
    min_samples_leaf: float = UNSET,
    max_leaf_nodes: float = UNSET,
    alpha: float = UNSET,
    c: Annotated[float, typer.Option("--c", help="Inverse regularization (C)")] = UNSET,
    n_clusters: float = UNSET,
):
    """Train and log one model of one question in its own MLflow run."""
    if question not in TASKS:
        raise typer.BadParameter(f"question must be one of {sorted(TASKS)}, got {question!r}")
    if question not in TRAINERS:
        raise typer.BadParameter(f"{question} has no train_one yet (its experiment WP adds it)")
    if stage not in STAGES:
        raise typer.BadParameter(f"stage must be one of {sorted(STAGES)}, got {stage!r}")

    given = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_leaf_nodes": max_leaf_nodes,
        "alpha": alpha,
        "C": c,
        "n_clusters": n_clusters,
    }
    hyperparams = {
        name: int(value) if name in INTEGER_PARAMS else value
        for name, value in given.items()
        if value != UNSET
    }
    run_seed = SEED if seed == UNSET else int(seed)

    logger.info(f"Training {question} {model} ({stage}) with {hyperparams or 'its defaults'}")
    run_id = TRAINERS[question](model, stage, run_seed, hyperparams)
    logger.success(f"Run {run_id} logged")


if __name__ == "__main__":
    app()
