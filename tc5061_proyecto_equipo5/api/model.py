from dataclasses import dataclass
import json
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import pandas as pd

from tc5061_proyecto_equipo5.config import MODELS_DIR
from tc5061_proyecto_equipo5.modeling.train_classification import (
    CLASS_NAMES_ES,
    FEATURE_COLUMNS,
    apply_thresholds,
)

SIGNATURE_DTYPES = {
    **{column: "float64" for column in FEATURE_COLUMNS},
    "capacity": "int32",
    "day_of_week": "int32",
    "is_weekend": "bool",
    "is_holiday": "bool",
}
METADATA_FILE = "registry_metadata.json"


@dataclass(frozen=True)
class LoadedModel:
    """The loaded model with what the API needs to turn its probabilities into classes."""

    info: dict
    model: mlflow.pyfunc.PyFuncModel
    classes: list[int]
    thresholds: dict[int, float]


def _run_metadata(run_id: str) -> dict:
    """Build the export metadata from a training run when there is no exported folder."""
    client = MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    tags = run.data.tags
    return {
        "model_name": None,
        "model_version": None,
        "model_alias": None,
        "exported_at": None,
        "source_run_id": run_id,
        "git_commit": tags.get("git_commit"),
        "data_version": params.get("data_version"),
        "thresholds": {
            "shortage": float(params["threshold_shortage"]),
            "saturation": float(params["threshold_saturation"]),
        },
    }


def _file_metadata(path: str) -> dict:
    """Read the metadata that export_model.py saved next to the exported model."""
    file = Path(path) / METADATA_FILE
    data = json.loads(file.read_text(encoding="utf-8"))
    return data


def load_model(uri: str | None = None) -> LoadedModel:
    """Load the model from uri, MODEL_URI or models/p2-riesgo, in that order."""
    uri = uri or os.environ.get("MODEL_URI", str(MODELS_DIR / "p2-riesgo"))
    model = mlflow.pyfunc.load_model(uri)
    classes = [int(c) for c in model.get_raw_model().classes_]

    if Path(uri).is_dir():
        info = _file_metadata(uri)
    else:
        info = _run_metadata(model.metadata.run_id)
        
    # The metadata names the thresholds; apply_thresholds expects them by class id
    thresholds = {
        1: float(info["thresholds"]["shortage"]),
        2: float(info["thresholds"]["saturation"]),
    }
    return LoadedModel(info=info, model=model, classes=classes, thresholds=thresholds)


def predict(loaded: LoadedModel, records: list[dict]) -> list[dict]:
    """Predict the risk class of each record with the thresholds chosen in validation."""
    df = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    df = df.astype(SIGNATURE_DTYPES)
    proba = loaded.model.predict(df)
    labels = apply_thresholds(proba, loaded.classes, loaded.thresholds)
    predictions = []
    for label_id, row in zip(labels, proba):
        prediction = {
            "label_id": int(label_id),
            "label": CLASS_NAMES_ES[label_id],
            "probabilities": {
                CLASS_NAMES_ES[cls]: float(p) for cls, p in zip(loaded.classes, row)
            },
        }
        predictions.append(prediction)

    return predictions
