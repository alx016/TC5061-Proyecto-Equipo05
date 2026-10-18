"""Small helpers shared by the training scripts of the three project questions.

Each question has its own training script (P1 regression, P2 classification, P3 clustering) so the
team can work in parallel without touching the same file, but reproducibility is identical for all
of them: the same experiment and the same data version.  Those two pieces live here, and each
script only adds its own model, features and metrics.

The MLflow tracking stays local by default (no tracking URI is set), so metadata goes to mlflow.db
and artifacts stay inside the ignored mlruns/ folder.  MLflow tags the Git commit on its own
(``mlflow.source.git.*``) only when the run starts from a script inside the repo; from a notebook,
``python -c`` or a copy without .git it silently skips it.  So every run also carries our own
``git_commit`` and ``git_dirty`` tags, read from the project root.
"""

import json
import subprocess

import mlflow

from tc5061_proyecto_equipo5.config import DATA_MANIFEST_PATH, PROJ_ROOT

# One shared experiment for the whole team; the rubric asks for the team number in the name
EXPERIMENT_NAME = "mibici-equipo05"
TEAM = "equipo05"


def data_version() -> str:
    """Read the data version hash written by the dataset manifest command."""
    return json.loads(DATA_MANIFEST_PATH.read_text())["data_version"]


def setup_experiment() -> None:
    """Point MLflow at the shared experiment (default local tracking)."""
    mlflow.set_experiment(EXPERIMENT_NAME)


def git_state() -> dict[str, str]:
    """Commit hash and whether the working tree had uncommitted changes when the run started."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        # Without git the run still trains; the tag says the code version is unknown
        return {"git_commit": "unknown", "git_dirty": "unknown"}
    return {"git_commit": commit, "git_dirty": str(bool(status.strip())).lower()}


def base_tags(question: str) -> dict[str, str]:
    """Return the tags every run must carry: the team, the question and the Git state."""
    return {"team": TEAM, "question": question, **git_state()}
