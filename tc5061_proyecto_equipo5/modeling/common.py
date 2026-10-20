"""Small helpers shared by the training scripts of the three project questions.

Each question has its own training script (P1 regression, P2 classification, P3 clustering) so the
team can work in parallel without touching the same file, but reproducibility is identical for all
of them: the same experiment and the same data version.  Those two pieces live here, and each
script only adds its own model, features and metrics.

Runs go to the tracking server in ``MLFLOW_TRACKING_URI`` (``make mlflow-server`` starts it on
port 5001, with metadata in mlflow.db and artifacts in the ignored mlartifacts/ folder); config.py
loads the variable from .env.  Without it, MLflow falls back to local tracking, which is fine for
quick tests but not for the evidence runs, which must all share one database.

MLflow tags the Git commit on its own (``mlflow.source.git.*``) only when the run starts from a
script inside the repo; from a notebook, ``python -c`` or a copy without .git it silently skips
it.  So every run also carries our own ``git_commit`` and ``git_dirty`` tags, read from the
project root.
"""

import json
import subprocess
from urllib.request import urlopen

import mlflow

from tc5061_proyecto_equipo5.config import DATA_MANIFEST_PATH, PROJ_ROOT

# One shared experiment for the whole team; the rubric asks for the team number in the name
EXPERIMENT_NAME = "mibici-equipo05"
STAGES = {"baseline", "search", "candidate"}
TASKS = {"p1": "regression", "p2": "classification", "p3": "clustering"}
TEAM = "equipo05"


def data_version() -> str:
    """Read the data version hash written by the dataset manifest command."""
    return json.loads(DATA_MANIFEST_PATH.read_text())["data_version"]


def setup_experiment() -> None:
    """Point MLflow at the shared experiment, failing fast if the tracking server is down.

    Without this check MLflow keeps retrying an unreachable server for minutes before it shows a
    connection error, so a run that forgot ``make mlflow-server`` looks frozen instead of failing.
    """
    uri = mlflow.get_tracking_uri()

    if uri.startswith("http"):
        try:
            urlopen(f"{uri.rstrip('/')}/health", timeout=3)
        except OSError as exc:
            raise RuntimeError(
                f"MLflow server not reachable at {uri}. Start it with `make mlflow-server`."
            ) from exc

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


def base_tags(question: str, model_family: str, stage: str) -> dict[str, str]:
    """Return the tags every run must carry, so runs can be filtered by question, family and stage.

    Raises ValueError for an unknown question or stage, since a misspelled tag would hide the run
    from the UI filters.
    """
    if question not in TASKS:
        raise ValueError(f"question must be one of {sorted(TASKS)}, got {question!r}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}, got {stage!r}")
    return {
        "team": TEAM,
        "question": question,
        "task": TASKS[question],
        "model_family": model_family,
        "stage_of_work": stage,
        **git_state(),
    }
