"""Seed the exported P2 champion into an MLflow server, such as the one in Docker Compose.

The MLflow server of ``docker-compose.yml`` starts empty, so the API can only load
``models:/mibici-equipo05-p2-riesgo@champion`` after a version exists there. This script takes
the bundle in ``models/p2-riesgo/`` (``make prepare-model`` or ``make export-model``) and:

1. Logs it again in a new run of the project experiment. The API reads the ``run_id`` written
   in ``MLmodel`` and then the run's ``threshold_*`` and ``data_version`` params and its
   ``git_commit`` tag, so copying the files is not enough: the bundle points to the original
   run, which does not exist in this server.
2. Copies those values from ``registry_metadata.json`` to the new run and keeps the real
   lineage in tags (``source_run_id_original``, ``source_registry_version``).
3. Registers the model, tags the version like ``registry.py`` does and points ``@champion`` to
   it.

It is idempotent: if ``@champion`` already comes from the same original run, nothing changes.
If ``@champion`` points to another model, it stops instead of moving the alias, unless
``--force`` is given.

Usage:
    make up
    make seed-registry
    uv run python scripts/seed_registry.py --tracking-uri http://127.0.0.1:5001
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import Model

from tc5061_proyecto_equipo5.modeling.common import EXPERIMENT_NAME
from tc5061_proyecto_equipo5.modeling.registry import MODEL_DESCRIPTIONS, MODEL_NAMES

PROJ_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = PROJ_ROOT / "models" / "p2-riesgo"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
METADATA_FILE = "registry_metadata.json"

QUESTION = "p2"
ALIAS = "champion"
ORIGIN_TAG = "source_run_id_original"
REQUIRED_METADATA = ("model_version", "source_run_id", "git_commit", "data_version")


class SeedError(Exception):
    """A condition that stops the seed with a clear message instead of a traceback."""


def read_metadata(bundle: Path) -> dict[str, Any]:
    """Lineage of the exported champion, with every field the API needs."""
    path = bundle / METADATA_FILE
    if not path.exists():
        raise SeedError(f"No existe {path}. Corre 'make prepare-model' o 'make export-model'.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_METADATA if not metadata.get(key)]
    thresholds = metadata.get("thresholds") or {}
    missing += [f"thresholds.{key}" for key in ("shortage", "saturation") if key not in thresholds]
    if missing:
        raise SeedError(f"{path} no trae {', '.join(missing)}")
    return metadata


def check_server(tracking_uri: str) -> None:
    """Fail fast when the server is down; MLflow's own retries take minutes."""
    if not tracking_uri.startswith(("http://", "https://")):
        return
    try:
        with urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=5):
            pass
    except (URLError, OSError) as error:
        raise SeedError(
            f"MLflow no responde en {tracking_uri} ({error}). ¿Corriste 'make up'?"
        ) from error


def current_champion(client: MlflowClient, name: str) -> ModelVersion | None:
    """The version behind @champion, or None when the model or the alias do not exist.

    Absence is checked without relying on exceptions: a missing alias comes back as a generic
    INVALID_PARAMETER_VALUE, the same code as other errors. Any MLflow error stops the seed,
    so a failed lookup never ends up moving the alias without --force.
    """
    try:
        if not client.search_registered_models(filter_string=f"name = '{name}'"):
            return None
        version = client.get_registered_model(name).aliases.get(ALIAS)
        return client.get_model_version(name, str(version)) if version is not None else None
    except MlflowException as error:
        raise SeedError(
            f"No se pudo consultar @{ALIAS} de {name} ({error}); no se cambio nada."
        ) from error


def experiment_id(client: MlflowClient) -> str:
    """Id of the project experiment, created on the server when it is new."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(EXPERIMENT_NAME)


def log_seed_run(bundle: Path, metadata: dict[str, Any], exp_id: str) -> tuple[str, str]:
    """Log the bundle again in a new run that carries what the API reads from it."""
    source = Model.load(str(bundle))
    sk_model = mlflow.sklearn.load_model(str(bundle))
    example = source.load_input_example(str(bundle))
    requirements = bundle / "requirements.txt"
    code_dir = bundle / "code" / "tc5061_proyecto_equipo5"

    with mlflow.start_run(
        experiment_id=exp_id, run_name=f"seed-{QUESTION}-champion-v{metadata['model_version']}"
    ) as run:
        mlflow.log_params(
            {
                "data_version": metadata["data_version"],
                "threshold_shortage": metadata["thresholds"]["shortage"],
                "threshold_saturation": metadata["thresholds"]["saturation"],
            }
        )
        mlflow.set_tags(
            {
                "question": QUESTION,
                "git_commit": metadata["git_commit"],
                ORIGIN_TAG: metadata["source_run_id"],
                "source_registry_version": str(metadata["model_version"]),
                "seeded_from_bundle": "true",
            }
        )
        info = mlflow.sklearn.log_model(
            sk_model,
            name="model",
            signature=source.signature,
            input_example=example,
            pyfunc_predict_fn="predict_proba",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            pip_requirements=str(requirements) if requirements.exists() else None,
            code_paths=[str(code_dir)] if code_dir.exists() else None,
        )
    return run.info.run_id, info.model_uri


def verify(model_uri: str, bundle: Path) -> None:
    """Load the new model and score the bundle's input example before registering it."""
    example = Model.load(str(bundle)).load_input_example(str(bundle))
    proba = mlflow.pyfunc.load_model(model_uri).predict(example)
    if proba.shape[0] != len(example):
        raise SeedError(f"{model_uri} devolvió {proba.shape[0]} filas para {len(example)}")


def seed(bundle: Path, tracking_uri: str, force: bool = False) -> ModelVersion:
    """Register the bundle as @champion on ``tracking_uri``; return the champion version."""
    check_server(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    metadata = read_metadata(bundle)
    client = MlflowClient()
    name = MODEL_NAMES[QUESTION]

    champion = current_champion(client, name)
    if champion is not None:
        if champion.tags.get(ORIGIN_TAG) == metadata["source_run_id"]:
            logger.info(
                f"{name} v{champion.version} (@{ALIAS}) ya viene de este bundle; no cambia"
            )
            return champion
        if not force:
            raise SeedError(
                f"@{ALIAS} ya apunta a {name} v{champion.version}, que no viene de este bundle. "
                "Usa --force para moverlo."
            )

    exp_id = experiment_id(client)
    run_id, model_uri = log_seed_run(bundle, metadata, exp_id)
    verify(model_uri, bundle)
    version = mlflow.register_model(model_uri=model_uri, name=name)

    # Same registered-model and version tags as registry.py, plus the original lineage
    client.update_registered_model(name, description=MODEL_DESCRIPTIONS[QUESTION])
    client.set_registered_model_tag(name, "question", QUESTION)
    client.set_registered_model_tag(name, "team", "equipo05")
    tags = {
        "question": QUESTION,
        "git_commit": metadata["git_commit"],
        "data_version": metadata["data_version"],
        "validation_status": "approved",
        "source_run_id": run_id,
        "experiment_id": exp_id,
        ORIGIN_TAG: metadata["source_run_id"],
        "source_registry_version": str(metadata["model_version"]),
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, version.version, key, value)
    client.update_model_version(
        name,
        version.version,
        description=(
            f"Champion v{metadata['model_version']} sembrado desde el bundle exportado "
            f"(run original {metadata['source_run_id']}, commit {metadata['git_commit']}, "
            f"data_version {metadata['data_version']})."
        ),
    )
    client.set_registered_model_alias(name, ALIAS, version.version)
    logger.success(f"Registrado {name} v{version.version} (@{ALIAS}) en el run {run_id}")
    return client.get_model_version(name, version.version)


def main() -> None:
    """Seed the champion and print how to point the API at it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument(
        "--force", action="store_true", help="Move @champion even if it points to another model"
    )
    args = parser.parse_args()

    try:
        version = seed(args.bundle, args.tracking_uri, force=args.force)
    except SeedError as error:
        logger.error(str(error))
        sys.exit(1)

    print(
        f"{version.name} v{version.version} @{ALIAS} listo. Para servirlo desde el Registry:\n"
        f"  MODEL_URI=models:/{version.name}@{ALIAS} docker compose up -d api"
    )


if __name__ == "__main__":
    main()
