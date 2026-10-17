"""Small helpers shared by the training scripts of the three project questions.

Each question has its own training script (P1 regression, P2 classification, P3 clustering) so the
team can work in parallel without touching the same file, but reproducibility is identical for all
of them: the same experiment and the same data version.  Those two pieces live here, and each script
only adds its own model, features and metrics.

The MLflow tracking stays local by default (no tracking URI is set), so metadata goes to mlflow.db
and artifacts stay inside the ignored mlruns/ folder.  MLflow tags every run with the Git commit,
the branch and the repo URL on its own (tags ``mlflow.source.git.*``), so there is no need to read
the commit by hand.
"""

import json

import mlflow

from tc5061_proyecto_equipo5.config import DATA_MANIFEST_PATH

# One shared experiment for the whole team; the rubric asks for the team number in the name
EXPERIMENT_NAME = "mibici-equipo05"
TEAM = "equipo05"


def data_version() -> str:
    """Read the data version hash written by the dataset manifest command."""
    return json.loads(DATA_MANIFEST_PATH.read_text())["data_version"]


def setup_experiment() -> None:
    """Point MLflow at the shared experiment (default local tracking)."""
    mlflow.set_experiment(EXPERIMENT_NAME)


def base_tags(question: str) -> dict[str, str]:
    """Return the tags every run must carry: the team and the question."""
    return {"team": TEAM, "question": question}
