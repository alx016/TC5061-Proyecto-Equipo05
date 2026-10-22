"""Register a trained run in the MLflow Model Registry with tags, descriptions and an alias."""

import os
from typing import Annotated

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import typer

from tc5061_proyecto_equipo5.modeling.common import EXPERIMENT_NAME

app = typer.Typer()
# One registered model per question (names agreed in the team plan)
MODEL_NAMES = {
    "p1": f"{EXPERIMENT_NAME}-p1-demanda",
    "p2": f"{EXPERIMENT_NAME}-p2-riesgo",
    "p3": f"{EXPERIMENT_NAME}-p3-grupos",
}
# champion serves; challenger waits for comparison. A new version starts as challenger.
ALIASES = {"champion", "challenger"}


@app.command()
def main(
    run_id: Annotated[str, typer.Option(help="Run that logged the model")],
    alias: Annotated[str, typer.Option(help="champion or challenger")] = "challenger",
):
    """Register the model of one run and point an alias to the new version."""
    if alias not in ALIASES:
        raise typer.BadParameter(f"Alias must be one of {sorted(ALIASES)}, got {alias!r}")

    client = MlflowClient()
    try:
        run = client.get_run(run_id)
    except MlflowException as error:
        raise typer.BadParameter(f"Run {run_id} not found in tracking server") from error

    # The run says which question it answers; the registered name follows from it
    question = run.data.tags.get("question")

    if question not in MODEL_NAMES:
        raise typer.BadParameter(f"Run {run_id} has no valid question tag, got {question!r}")

    if run.data.tags.get("git_dirty") == "true":
        logger.warning(f"Run {run_id} was trained with uncommitted changes (git_dirty=true)")

    name = MODEL_NAMES[question]
    logger.info(f"Run {run_id} answers {question}: registering it as {name} with alias {alias}")
    version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=name)
    tags = {
        "question": question,
        "model_family": run.data.tags.get("model_family"),
        "git_commit": run.data.tags.get("git_commit"),
        "git_dirty": run.data.tags.get("git_dirty"),
        "data_version": run.data.params.get("data_version"),
        "features_data_version": run.data.params.get("features_data_version"),
        "validation_status": "approved" if alias == "champion" else "pending",
        "source_run_id": run_id,
        "experiment_id": run.info.experiment_id,
        "task": run.data.tags.get("task"),
    }
    # Lineage tags copied from the run, so the version can be traced without opening it
    for key, value in tags.items():
        if value is not None:
            client.set_model_version_tag(name, version.version, key, value)

    # Selection decisions belong to each question's select step; the registry only carries them
    for key, value in run.data.tags.items():
        if key.startswith("selection_"):
            client.set_model_version_tag(name, version.version, key, value)

    client.update_model_version(
        name,
        version.version,
        description=(
            f"{tags['model_family']} trained in run {run_id} "
            f"(commit {tags['git_commit']}, data_version {tags['data_version']})."
        ),
    )

    # The alias is the stable address that consumers load: models:/<name>@<alias>
    client.set_registered_model_alias(name, alias, version.version)

    #  inside `mlflow run -e register`, label the bookkeeping run so it can be found later
    register_run_id = os.environ.get("MLFLOW_RUN_ID")
    if register_run_id:
        client.set_tag(register_run_id, "mlflow.runName", f"register-{name}-v{version.version}")
        client.set_tag(register_run_id, "registered_model", name)
        client.set_tag(register_run_id, "registered_version", version.version)
        client.set_tag(register_run_id, "source_run_id", run_id)

    logger.success(f"Registered run {run_id} as {name} version {version.version} (@{alias})")


if __name__ == "__main__":
    app()
