"""Execute the declarative MLflow Project against synthetic P1 data and SQLite."""

import json
import os
from pathlib import Path
import shlex
import shutil
import sys

import mlflow
import pandas as pd
from test_regression_training import make_prepared

from tc5061_proyecto_equipo5.config import SEED
from tc5061_proyecto_equipo5.features import FEATURE_VERSION


def test_project_entrypoint_is_the_study_with_eight_trials(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(source / "MLproject", project / "MLproject")
    shutil.copy(source / "python_env.yaml", project / "python_env.yaml")
    shutil.copytree(source / "tc5061_proyecto_equipo5", project / "tc5061_proyecto_equipo5")
    folder = project / "data/processed/fe_2024_v4/p1"
    folder.mkdir(parents=True)
    X, keys, y = make_prepared(pd.to_datetime(["2024-09-15", "2024-10-15", "2024-12-15"]))
    X.to_parquet(folder / "X.parquet", index=False)
    keys.to_parquet(folder / "keys.parquet", index=False)
    y.to_frame().to_parquet(folder / "y.parquet", index=False)
    manifest = {"data_version": "synthetic", "feature_version": FEATURE_VERSION}
    (folder / "features_manifest.json").write_text(json.dumps(manifest))
    (project / "references").mkdir()
    (project / "references/viajes_manifest.json").write_text(json.dumps(manifest))
    previous_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'project.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", str(project))
    # macOS SIP strips DYLD_* when MLflow starts system bash. Restore the caller
    # runtime setting inside that shell, without changing the project command.
    if sys.platform == "darwin" and os.environ.get("DYLD_LIBRARY_PATH"):
        shell_env = tmp_path / "bash_env"
        shell_env.write_text(
            "export DYLD_LIBRARY_PATH=" + shlex.quote(os.environ["DYLD_LIBRARY_PATH"]) + "\n"
        )
        monkeypatch.setenv("BASH_ENV", str(shell_env))
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    experiment = client.create_experiment("mibici-equipo05", str(tmp_path / "artifacts"))
    try:
        submitted = mlflow.projects.run(
            str(project),
            entry_point="search",
            parameters={"question": "p1"},
            experiment_id=experiment,
            env_manager="local",
            synchronous=True,
        )
        study = client.get_run(submitted.run_id)
        runs = client.search_runs([experiment])
        children = [
            run for run in runs if run.data.tags.get("mlflow.parentRunId") == study.info.run_id
        ]
        assert len(runs) == 9
        assert len(children) == 8
        assert study.info.status == "FINISHED"
        assert study.data.tags["mlflow.runName"] == "p1-search-lightgbm"
        assert study.data.params["model"] == "lightgbm"
        assert study.data.params["seed"] == str(SEED)
        assert study.data.tags["best_run_id"] in {run.info.run_id for run in children}
        assert {
            run.info.run_id for run in children if run.data.tags.get("selected") == "true"
        } == {study.data.tags["best_run_id"]}
        assert all(run.info.status == "FINISHED" for run in children)
        assert all(not any(key.startswith("test_") for key in run.data.metrics) for run in runs)
        assert client.list_artifacts(study.info.run_id, "search")
    finally:
        mlflow.set_tracking_uri(previous_uri)
