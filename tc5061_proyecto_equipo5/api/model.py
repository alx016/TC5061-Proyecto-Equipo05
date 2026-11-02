from dataclasses import dataclass
import os

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


@dataclass(frozen=True)
class LoadedModel:
    """The loaded model with what the API needs to turn its probabilities into classes."""

    model: mlflow.pyfunc.PyFuncModel
    classes: list[int]
    thresholds: dict[int, float]


def _run_thresholds(run_id: str) -> dict[int, float]:
    """Read the decision thresholds that the training run logged as params."""
    client = MlflowClient()
    params = client.get_run(run_id).data.params
    return {1: float(params["threshold_shortage"]), 2: float(params["threshold_saturation"])}


def load_model(uri: str | None = None) -> LoadedModel:
    """Load the model from uri, MODEL_URI or models/p2-riesgo, in that order."""
    uri = uri or os.environ.get("MODEL_URI", str(MODELS_DIR / "p2-riesgo"))
    model = mlflow.pyfunc.load_model(uri)
    classes = [int(c) for c in model.get_raw_model().classes_]
    thresholds = _run_thresholds(model.metadata.run_id)
    return LoadedModel(model=model, classes=classes, thresholds=thresholds)


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
