from dataclasses import dataclass
import os

import mlflow
from mlflow import MlflowClient

from tc5061_proyecto_equipo5.config import MODELS_DIR


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
    """load the model from uri, MODEL_URI or models/p2-riesgo, in that order."""
    uri = uri or os.environ.get("MODEL_URI", str(MODELS_DIR / "p2-riesgo"))
    model = mlflow.pyfunc.load_model(uri)
    classes = [int(c) for c in model.get_raw_model().classes_]
    thresholds = _run_thresholds(model.metadata.run_id)
    return LoadedModel(model=model, classes=classes, thresholds=thresholds)
