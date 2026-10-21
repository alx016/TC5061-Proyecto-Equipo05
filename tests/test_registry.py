"""Unit tests for the Model Registry description, lineage and naming helpers."""

from types import SimpleNamespace

import pandas as pd
import pytest

from tc5061_proyecto_equipo5.modeling import registry


def fake_run():
    data = SimpleNamespace(
        params={
            "data_version": "abc123",
            "threshold_shortage": "0.20",
            "threshold_saturation": "0.30",
            "threshold_fallback": "false",
        },
        metrics={
            "val_macro_f1": 0.47,
            "val_pr_auc_shortage": 0.24,
            "val_pr_auc_saturation": 0.31,
            "val_pr_auc_minority_mean": 0.275,
            "val_recall_shortage": 0.86,
            "val_precision_shortage": 0.12,
            "test_macro_f1": 0.46,
            "test_pr_auc_shortage": 0.23,
            "test_pr_auc_saturation": 0.30,
            "test_recall_shortage": 0.84,
            "test_precision_shortage": 0.11,
        },
        tags={
            "model_family": "lightgbm",
            "task": "classification",
            "git_commit": "deadbeef",
            "selection_rule": "max val_pr_auc_minority_mean",
        },
    )
    return SimpleNamespace(data=data, info=SimpleNamespace(run_id="run-123", experiment_id="7"))


def test_registry_names_follow_the_team_convention():
    assert registry.model_name("p2") == "mibici-equipo05-p2-riesgo"
    assert registry.model_name("p1") == "mibici-equipo05-p1-demanda"
    assert registry.model_name("p3") == "mibici-equipo05-p3-grupos"
    with pytest.raises(ValueError, match="question must be one of"):
        registry.model_name("p9")


def test_description_contains_metrics_baseline_and_lineage():
    baseline = pd.Series(
        {
            "tags.model_family": "dummy_most_frequent",
            "metrics.val_pr_auc_minority_mean": 0.25,
            "metrics.val_macro_f1": 0.33,
        }
    )

    description = registry.describe(fake_run(), baseline)

    assert "macro-F1 0.4700" in description
    assert "minority mean AP 0.2750" in description
    assert "Test (December" in description
    assert "dummy_most_frequent" in description
    assert "delta macro-F1 +0.1400" in description
    assert "data_version abc123" in description
    assert "git_commit deadbeef" in description
    assert "run-123" in description


def test_lineage_tags_preserve_run_data_and_code():
    tags = registry.lineage_tags("p2", fake_run())

    assert tags["source_run_id"] == "run-123"
    assert tags["experiment_id"] == "7"
    assert tags["data_version"] == "abc123"
    assert tags["git_commit"] == "deadbeef"
    assert tags["question"] == "p2"
    assert tags["threshold_shortage"] == "0.20"
    assert tags["validation_status"] == "selected"


def test_selected_run_id_fails_clearly_without_a_selected_run(monkeypatch):
    monkeypatch.setattr(registry.mlflow, "search_runs", lambda **kwargs: pd.DataFrame())

    with pytest.raises(ValueError, match="No selected p2 run"):
        registry.selected_run_id("p2")
