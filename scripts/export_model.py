"""Export the P2 champion from the Model Registry to models/p2-riesgo/.

Reads ``models:/mibici-equipo05-p2-riesgo@champion``, downloads the concrete version's
files next to ``MLmodel`` and writes ``registry_metadata.json`` in the format the API
reads (``local/entrega4.md`` §2.7).  Creates no MLflow run.
"""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import tempfile

import mlflow
from mlflow.tracking import MlflowClient

PROJ_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "mibici-equipo05-p2-riesgo"
MODEL_ALIAS = "champion"
DEFAULT_OUTPUT_DIR = PROJ_ROOT / "models" / "p2-riesgo"
METADATA_FILENAME = "registry_metadata.json"


def build_metadata(client: MlflowClient, model_name: str, alias: str) -> dict:
    """Resolve the alias and read the lineage the API needs from the source run."""
    version = client.get_model_version_by_alias(model_name, alias)
    run = client.get_run(version.run_id)
    return {
        "model_name": model_name,
        # MlflowClient returns the version as text already; keep it text.
        "model_version": str(version.version),
        "model_alias": alias,
        "source_run_id": version.run_id,
        "git_commit": run.data.tags["git_commit"],
        "data_version": run.data.params["data_version"],
        "thresholds": {
            "shortage": float(run.data.params["threshold_shortage"]),
            "saturation": float(run.data.params["threshold_saturation"]),
        },
        "exported_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def export_model(model_name: str, alias: str, output_dir: Path) -> dict:
    """Download the aliased version by concrete number and write its metadata file."""
    client = MlflowClient()
    metadata = build_metadata(client, model_name, alias)

    # Download by concrete version (immutable), not by alias.
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as staging:
        staged = Path(staging) / "model"
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}/{metadata['model_version']}",
            dst_path=str(staged),
        )
        (staged / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.move(str(staged), output_dir)

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

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    metadata = export_model(MODEL_NAME, MODEL_ALIAS, args.output_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
