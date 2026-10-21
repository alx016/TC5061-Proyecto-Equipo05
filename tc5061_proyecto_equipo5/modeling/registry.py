"""Register a selected training run in the MLflow Model Registry.

The D2 workflow is: train candidates (``train``), run the nested search (``search``), evaluate
December test once (``select``) and only then register the selected run.  Registration adds the
description, the lineage tags and the ``champion`` alias that serving uses.  The aliases API is
used instead of the deprecated stages API because an alias is mutable, atomic and can be promoted
or rolled back without changing the model URI.

Usage:
    python -m tc5061_proyecto_equipo5.modeling.registry --question p2 --run-id <RUN_ID>
"""

import mlflow
from mlflow import MlflowClient
import pandas as pd
import typer

from tc5061_proyecto_equipo5.config import P2_MIN_RECALL_SHORTAGE
from tc5061_proyecto_equipo5.modeling.common import EXPERIMENT_NAME, TASKS, setup_experiment

MODEL_NAMES = {
    "p1": "mibici-equipo05-p1-demanda",
    "p2": "mibici-equipo05-p2-riesgo",
    "p3": "mibici-equipo05-p3-grupos",
}
# Ranking metric of the clean baseline used in the description of the P2 model
P2_RANK_METRIC = "val_pr_auc_minority_mean"


def model_name(question: str) -> str:
    """Return the registry name agreed with the team for one question."""
    if question not in MODEL_NAMES:
        raise ValueError(f"question must be one of {sorted(MODEL_NAMES)}, got {question!r}")
    return MODEL_NAMES[question]


def selected_run_id(question: str) -> str:
    """Find the single run marked ``selected=true`` for one question."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.question = '{question}' and tags.selected = 'true'",
    )
    if runs.empty:
        raise ValueError(f"No selected {question} run; run `select` first")
    if len(runs) > 1:
        runs = runs.sort_values("start_time", ascending=False)
    return runs.iloc[0]["run_id"]


def best_baseline(question: str) -> pd.Series | None:
    """Return the best clean baseline of one question, or None when there is no baseline."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=(
            f"tags.question = '{question}' and tags.stage_of_work = 'baseline' "
            "and tags.git_dirty = 'false'"
        ),
    )
    if runs.empty:
        return None
    if question == "p2":
        return runs.sort_values(
            [f"metrics.{P2_RANK_METRIC}", "metrics.val_macro_f1"],
            ascending=False,
            na_position="last",
        ).iloc[0]
    if question == "p1":
        return runs.sort_values("metrics.val_wape", ascending=True, na_position="last").iloc[0]
    return runs.sort_values("metrics.val_silhouette", ascending=False, na_position="last").iloc[0]


def _number(metrics: dict, key: str, digits: int = 4) -> str:
    value = metrics.get(key)
    try:
        return "n/a" if value is None else f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def describe(run, baseline: pd.Series | None = None) -> str:
    """Build the model-version description with metrics, baseline and lineage."""
    params = run.data.params
    metrics = run.data.metrics
    tags = run.data.tags
    family = tags.get("model_family", "unknown")
    thresholds = (
        f"{_number(params, 'threshold_shortage')}/{_number(params, 'threshold_saturation')}"
    )

    lines = [
        (
            "P2 risk classifier for MiBici station shortages and saturations one hour ahead. "
            f"Model family: {family}. Decision thresholds (shortage/saturation): {thresholds}. "
            f"Threshold fallback: {params.get('threshold_fallback', 'n/a')}. "
            f"Recall shortage guardrail: {P2_MIN_RECALL_SHORTAGE:.2f}."
        ),
        (
            f"Validation: macro-F1 {_number(metrics, 'val_macro_f1')}, "
            f"AP shortage {_number(metrics, 'val_pr_auc_shortage')}, "
            f"AP saturation {_number(metrics, 'val_pr_auc_saturation')}, "
            f"minority mean AP {_number(metrics, P2_RANK_METRIC)}, "
            f"recall shortage {_number(metrics, 'val_recall_shortage')}, "
            f"precision shortage {_number(metrics, 'val_precision_shortage')}."
        ),
        (
            "Test (December, evaluated once by select): macro-F1 "
            f"{_number(metrics, 'test_macro_f1')}, AP shortage "
            f"{_number(metrics, 'test_pr_auc_shortage')}, AP saturation "
            f"{_number(metrics, 'test_pr_auc_saturation')}, recall shortage "
            f"{_number(metrics, 'test_recall_shortage')}, precision shortage "
            f"{_number(metrics, 'test_precision_shortage')}."
        ),
    ]

    if baseline is not None:
        baseline_name = baseline.get("tags.model_family", "baseline")
        baseline_ap = baseline.get(f"metrics.{P2_RANK_METRIC}")
        baseline_f1 = baseline.get("metrics.val_macro_f1")
        delta_ap = (
            metrics.get(P2_RANK_METRIC, 0.0) - baseline_ap
            if baseline_ap is not None and pd.notna(baseline_ap)
            else None
        )
        delta_f1 = (
            metrics.get("val_macro_f1", 0.0) - baseline_f1
            if baseline_f1 is not None and pd.notna(baseline_f1)
            else None
        )
        lines.append(
            f"Baseline comparison ({baseline_name}): validation minority mean AP "
            f"{baseline_ap if baseline_ap is None else f'{baseline_ap:.4f}'}, macro-F1 "
            f"{baseline_f1 if baseline_f1 is None else f'{baseline_f1:.4f}'}; "
            f"delta AP {delta_ap if delta_ap is None else f'{delta_ap:+.4f}'}, "
            f"delta macro-F1 {delta_f1 if delta_f1 is None else f'{delta_f1:+.4f}'}."
        )

    lines.append(
        f"Lineage: data_version {params.get('data_version', 'n/a')}, git_commit "
        f"{tags.get('git_commit', 'n/a')}, source run {run.info.run_id}, experiment "
        f"{run.info.experiment_id}."
    )
    return "\n\n".join(lines)


def lineage_tags(question: str, run) -> dict[str, str]:
    """Return the version tags that preserve the run, data and code lineage."""
    params = run.data.params
    tags = run.data.tags
    values = {
        "question": question,
        "task": tags.get("task", TASKS[question]),
        "model_family": tags.get("model_family", ""),
        "data_version": params.get("data_version", ""),
        "git_commit": tags.get("git_commit", ""),
        "source_run_id": run.info.run_id,
        "experiment_id": str(run.info.experiment_id),
        "selection_rule": tags.get("selection_rule", ""),
        "threshold_shortage": params.get("threshold_shortage", ""),
        "threshold_saturation": params.get("threshold_saturation", ""),
        "threshold_fallback": params.get("threshold_fallback", ""),
        "validation_status": "selected",
    }
    return {key: str(value) for key, value in values.items()}


def register_model(question: str, run_id: str = "", alias: str = "champion") -> int:
    """Register the selected run and move the requested alias to the new version."""
    if question not in TASKS:
        raise ValueError(f"question must be one of {sorted(TASKS)}, got {question!r}")

    setup_experiment()
    client = MlflowClient()
    run_id = run_id or selected_run_id(question)
    run = client.get_run(run_id)

    if run.data.tags.get("question") != question:
        raise ValueError(f"Run {run_id} is not a {question} run")
    if run.data.tags.get("selected") != "true":
        raise ValueError(f"Run {run_id} is not selected; run `select` first")
    if run.data.tags.get("git_dirty") != "false":
        raise ValueError(
            f"Run {run_id} has git_dirty=true; a registry version must be reproducible"
        )
    if not any(key.startswith("test_") for key in run.data.metrics):
        raise ValueError(f"Run {run_id} has no test metrics; run `select` first")

    name = model_name(question)
    description = describe(run, best_baseline(question))
    model_version = mlflow.register_model(f"runs:/{run_id}/model", name)
    client.update_registered_model(
        name=name,
        description=(
            f"MiBici Equipo 05 {question.upper()} model. Lifecycle uses the alias `champion` "
            "for the serving version and `challenger` for the next candidate; the deprecated "
            "stages API is not used."
        ),
    )
    client.update_model_version(name=name, version=model_version.version, description=description)
    for key, value in lineage_tags(question, run).items():
        client.set_model_version_tag(name, model_version.version, key, value)

    client.set_registered_model_alias(name, alias, model_version.version)
    client.set_tag(run_id, "registered_model", name)
    client.set_tag(run_id, "registered_version", model_version.version)
    client.set_tag(run_id, "registered_alias", alias)

    print(f"Registered {name} version {model_version.version} with alias @{alias}")
    print(f"Load with: mlflow.pyfunc.load_model('models:/{name}@{alias}')")
    return int(model_version.version)


def register(
    question: str = typer.Option(..., help="p1, p2 or p3"),
    run_id: str = typer.Option("", help="Selected run id; empty finds tags.selected=true"),
    alias: str = typer.Option("champion", help="Registry alias to move to the new version"),
) -> None:
    """Register the selected model and update its lifecycle alias."""
    register_model(question, run_id, alias)


if __name__ == "__main__":
    typer.run(register)
