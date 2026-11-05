"""Export the P2 champion from the Model Registry to models/p2-riesgo/.

Reads ``models:/mibici-equipo05-p2-riesgo@champion``, downloads the concrete version's
files next to ``MLmodel`` and writes ``registry_metadata.json`` in the format the API
reads.  Creates no MLflow run.

Usage:
    uv run python scripts/export_model.py
    uv run python scripts/export_model.py --output-dir /tmp/p2-riesgo

``MLFLOW_TRACKING_URI`` is read from the environment; when it is unset and the
champion's files live behind the server's artifact proxy, the project ``.env`` is
used as a fallback.  Models registered through ``make mlflow-server`` need that
server running during the export.
"""

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from urllib.parse import urlparse

from dotenv import dotenv_values
import mlflow
from mlflow.tracking import MlflowClient

PROJ_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "mibici-equipo05-p2-riesgo"
MODEL_ALIAS = "champion"
DEFAULT_OUTPUT_DIR = PROJ_ROOT / "models" / "p2-riesgo"
METADATA_FILENAME = "registry_metadata.json"
REQUIRED_MODEL_FILES = ("MLmodel", "model.pkl")


def _required(values: dict[str, Any], key: str, source: str, run_id: str) -> str:
    """Return a lineage value or fail with the key the export contract requires."""
    value = values.get(key)
    if value is None or value == "":
        raise ValueError(f"Run {run_id} has no {source}[{key!r}] for registry_metadata.json")
    return str(value)


def _required_float(values: dict[str, Any], key: str, run_id: str) -> float:
    """Read a numeric run parameter, naming the parameter when it is not a number."""
    value = _required(values, key, "params", run_id)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Run {run_id} has a non-numeric params[{key!r}]: {value!r}") from exc


def build_metadata(client: MlflowClient, model_name: str, alias: str) -> dict:
    """Resolve the alias and read the lineage the API needs from the source run."""
    version = client.get_model_version_by_alias(model_name, alias)
    if not version.run_id:
        raise ValueError(f"Model {model_name}@{alias} has no source run id")
    run = client.get_run(version.run_id)
    params = run.data.params
    tags = run.data.tags
    return {
        "model_name": model_name,
        # MlflowClient returns the version as text already; keep it text.
        "model_version": str(version.version),
        "model_alias": alias,
        "source_run_id": version.run_id,
        "git_commit": _required(tags, "git_commit", "tags", version.run_id),
        "data_version": _required(params, "data_version", "params", version.run_id),
        "thresholds": {
            "shortage": _required_float(params, "threshold_shortage", version.run_id),
            "saturation": _required_float(params, "threshold_saturation", version.run_id),
        },
        "exported_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _download_tracking_uri(client: MlflowClient, model_name: str, version: str) -> str | None:
    """Return the HTTP tracking URI proxied artifacts need, or None for local files."""
    download_uri = client.get_model_version_download_uri(model_name, version)
    if urlparse(download_uri).scheme != "mlflow-artifacts":
        return None
    tracking_uri = mlflow.get_tracking_uri()
    if urlparse(tracking_uri).scheme in {"http", "https"}:
        return tracking_uri
    # Fall back to the project .env only for proxied artifacts, so exports backed by
    # local mlruns keep working without a running server.  An exported URI still wins.
    if "MLFLOW_TRACKING_URI" not in os.environ:
        from_env = dotenv_values(PROJ_ROOT / ".env").get("MLFLOW_TRACKING_URI")
        if from_env and urlparse(from_env).scheme in {"http", "https"}:
            return from_env
    raise ValueError(
        f"Model {model_name} v{version} is stored in the MLflow server ({download_uri}), "
        f"but MLFLOW_TRACKING_URI is {tracking_uri!r}. Start the server with "
        "`make mlflow-server` and set MLFLOW_TRACKING_URI=http://127.0.0.1:5001 "
        "in the environment or in .env before exporting."
    )


def _validate_staged_model(staged: Path) -> None:
    """Fail before replacing a good export when the download is incomplete or corrupt."""
    missing = [name for name in REQUIRED_MODEL_FILES if not (staged / name).is_file()]
    if missing:
        raise ValueError(f"Downloaded model is incomplete; missing {', '.join(missing)}")
    # Loading is the cheapest proof that the pickle and the MLmodel agree.
    mlflow.pyfunc.load_model(str(staged))


def _replace_output(staged: Path, output_dir: Path) -> None:
    """Swap the staged model into place, restoring the previous export if the move fails."""
    backup: Path | None = None
    if output_dir.exists():
        backup = output_dir.with_name(f".{output_dir.name}.backup")
        shutil.rmtree(backup, ignore_errors=True)
        shutil.move(str(output_dir), str(backup))
    try:
        shutil.move(str(staged), str(output_dir))
    except Exception:
        if backup is not None and backup.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
            shutil.move(str(backup), str(output_dir))
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)


def export_model(
    model_name: str,
    alias: str,
    output_dir: Path,
    client: MlflowClient | None = None,
) -> dict:
    """Download the aliased version by concrete number and write its metadata file."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    client = client or MlflowClient()
    metadata = build_metadata(client, model_name, alias)
    tracking_uri = _download_tracking_uri(client, model_name, metadata["model_version"])

    # Download by concrete version (immutable), not by alias.
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as staging:
        staged = Path(staging) / "model"
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}/{metadata['model_version']}",
            dst_path=str(staged),
            tracking_uri=tracking_uri,
        )
        _validate_staged_model(staged)
        (staged / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
        _replace_output(staged, output_dir)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the P2 champion and its registry_metadata.json."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory receiving the model files and registry_metadata.json",
    )
    args = parser.parse_args()

    metadata = export_model(MODEL_NAME, MODEL_ALIAS, args.output_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
