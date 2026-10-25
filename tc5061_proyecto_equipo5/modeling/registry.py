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
# What each registered model answers, shown first in the Model Registry UI
MODEL_DESCRIPTIONS = {
    "p1": "P1 demanda: viajes que iniciaran en cada estacion la siguiente hora. Metrica: WAPE.",
    "p2": (
        "P2 riesgo: desabasto, balanceada o saturacion de cada estacion la siguiente hora. "
        "Metricas: macro-F1 y PR-AUC."
    ),
    "p3": (
        "P3 grupos: estaciones agrupadas por su perfil horario de uso. "
        "Metricas: silhouette y Davies-Bouldin."
    ),
}
# champion serves; challenger waits for comparison. A new version starts as challenger.
ALIASES = {"champion", "challenger"}
# Training scripts have used several names for the feature table version; first non-empty wins
FEATURES_VERSION_PARAMS = ("features_data_version", "feature_data_version", "feature_table_sha256")
UNKNOWN_LINEAGE_VALUES = {"", "unknown", "none", "null"}


def require_lineage(run) -> None:
    """Do not create versions whose code or data origin cannot be audited."""
    for key, value in (
        ("git_commit", run.data.tags.get("git_commit")),
        ("data_version", run.data.params.get("data_version")),
    ):
        if not value or value.strip().lower() in UNKNOWN_LINEAGE_VALUES:
            raise typer.BadParameter(f"Run {run.info.run_id} needs a known {key}")


def preflight(name: str, version: str) -> None:
    """Prove the immutable candidate can predict before changing any serving alias."""
    uri = f"models:/{name}/{version}"
    try:
        model = mlflow.pyfunc.load_model(uri)
        example = model.metadata.load_input_example(uri)
        if example is None:
            raise ValueError("model has no saved input_example; log it again with an example")
        model.predict(example)
    except Exception as error:
        raise typer.BadParameter(f"Model {uri} failed pre-flight: {error}") from error


def lineage_tags(question: str, run, alias: str) -> dict:
    """Tags copied from the run so a version can be traced without opening it."""
    return {
        "question": question,
        "model_family": run.data.tags.get("model_family"),
        "git_commit": run.data.tags.get("git_commit"),
        "git_dirty": run.data.tags.get("git_dirty"),
        "data_version": run.data.params.get("data_version"),
        "features_data_version": next(
            (run.data.params[key] for key in FEATURES_VERSION_PARAMS if run.data.params.get(key)),
            None,
        ),
        "validation_status": "approved" if alias == "champion" else "pending",
        "source_run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "task": run.data.tags.get("task"),
        "selected": run.data.tags.get("selected"),
    }


def version_description(tags: dict, run_id: str) -> str:
    """Where a version came from, leaving out what the run did not record."""
    origin = f"entrenado en el run {run_id}"
    origin = f"{tags['model_family']} {origin}" if tags["model_family"] else origin.capitalize()
    details = [
        f"{label} {tags[key]}"
        for label, key in (("commit", "git_commit"), ("data_version", "data_version"))
        if tags[key]
    ]
    return f"{origin} ({', '.join(details)})." if details else f"{origin}."


@app.command()
def main(
    run_id: Annotated[str, typer.Option(help="Run that logged the model")],
    alias: Annotated[str, typer.Option(help="champion or challenger")] = "challenger",
):
    """Register the model of one run and point an alias to its new version."""
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

    require_lineage(run)

    # A champion must be reproducible and chosen by its select step; a challenger only warns
    problems = []
    if run.data.tags.get("git_dirty") != "false":
        problems.append("has no confirmed clean code (git_dirty must be false)")
    if run.data.tags.get("selected") != "true":
        problems.append("was not marked by its select step (selected!=true)")
    for problem in problems:
        if alias == "champion":
            raise typer.BadParameter(f"Run {run_id} {problem}; it cannot be champion")
        logger.warning(f"Run {run_id} {problem}")

    # Check before register_model, which creates the registered model before it looks for the
    # artifact and would leave it empty. A search Study carries the question tag but no model
    model_uri = f"runs:/{run_id}/model"
    try:
        mlflow.models.get_model_info(model_uri)
    except MlflowException as error:
        raise typer.BadParameter(
            f"Run {run_id} has no logged model ({error}); "
            "if it is a search Study, register its winning Trial (tag selected=true)"
        ) from error

    name = MODEL_NAMES[question]
    logger.info(f"Run {run_id} answers {question}: registering it as {name} with alias {alias}")
    # Promoting a challenger to champion moves the alias; registering again would duplicate it
    existing = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
    if existing:
        version = max(existing, key=lambda found: int(found.version))
        # The default alias must not demote the version in production
        aliases = client.get_registered_model(name).aliases
        if alias == "challenger" and int(aliases.get("champion", -1)) == int(version.version):
            raise typer.BadParameter(
                f"Run {run_id} is already {name} version {version.version} (@champion)"
            )
        logger.info(f"Run {run_id} is already {name} version {version.version}; reusing it")
    else:
        version = mlflow.register_model(model_uri=model_uri, name=name)

    # Failed validation leaves the previous champion, challenger and approval tags untouched.
    if alias == "champion":
        preflight(name, version.version)

    # The registered model says what it answers; each version says where it came from
    client.update_registered_model(name, description=MODEL_DESCRIPTIONS[question])
    client.set_registered_model_tag(name, "question", question)
    client.set_registered_model_tag(name, "team", "equipo05")
    tags = lineage_tags(question, run, alias)

    # A reused version keeps what it already says: only missing tags and description are added,
    # except validation_status, which always follows the alias it is getting
    version = client.get_model_version(name, version.version)

    # Lineage tags copied from the run, so the version can be traced without opening it
    for key, value in tags.items():
        missing = key not in version.tags
        if key in {"git_commit", "data_version"}:
            missing = version.tags.get(key, "").strip().lower() in UNKNOWN_LINEAGE_VALUES
        if value is not None and (missing or key == "validation_status"):
            client.set_model_version_tag(name, version.version, key, value)

    # Selection decisions belong to each question's select step; the registry only carries them
    for key, value in run.data.tags.items():
        if key.startswith("selection_") and key not in version.tags:
            client.set_model_version_tag(name, version.version, key, value)

    if not version.description:
        client.update_model_version(
            name, version.version, description=version_description(tags, run_id)
        )

    # The alias is the stable address that consumers load: models:/<name>@<alias>
    client.set_registered_model_alias(name, alias, version.version)
    # A promoted challenger stops being one, so @challenger never points at the champion
    if alias == "champion":
        aliases = client.get_registered_model(name).aliases
        if int(aliases.get("challenger", -1)) == int(version.version):
            client.delete_registered_model_alias(name, "challenger")

    # Inside `mlflow run -e register`, label the bookkeeping run so it can be found later
    register_run_id = os.environ.get("MLFLOW_RUN_ID")
    if register_run_id:
        client.set_tag(register_run_id, "mlflow.runName", f"register-{name}-v{version.version}")
        client.set_tag(register_run_id, "registered_model", name)
        client.set_tag(register_run_id, "registered_version", version.version)
        client.set_tag(register_run_id, "source_run_id", run_id)

    logger.success(f"Registered run {run_id} as {name} version {version.version} (@{alias})")


if __name__ == "__main__":
    app()
