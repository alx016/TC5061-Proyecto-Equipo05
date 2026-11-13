"""A/B test between two versions of the P2 model on the same evaluation period.

Each version predicts with its own thresholds. The difference challenger - champion gets a 95%
interval from a bootstrap that resamples whole days, and the decision follows section 4.2 of
reports/entrega5_monitoreo.md, which was fixed before seeing any result.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, f1_score, recall_score
import typer

from tc5061_proyecto_equipo5.api.model import SIGNATURE_DTYPES, LoadedModel, load_model
from tc5061_proyecto_equipo5.config import (
    CLASSIFICATION_FEATURES_PATH,
    CURRENT_FEATURES_PATH,
    P2_MIN_RECALL_SHORTAGE,
    REPORTS_DIR,
)
from tc5061_proyecto_equipo5.modeling.common import base_tags, setup_experiment
from tc5061_proyecto_equipo5.modeling.train_classification import (
    FEATURE_COLUMNS,
    alerts_per_day,
    apply_thresholds,
    load_rows,
    metrics,
)

EVALUATION_SPLIT = "evaluation"
BOOTSTRAP_METRICS = ("pr_auc_minority_mean", "macro_f1", "recall_shortage", "alerts_per_day")
MACRO_F1_MARGIN = 0.01
ALERTS_MAX_RATIO = 1.20


@dataclass(frozen=True)
class Scored:
    """Probabilities and labels of one version, with the class order of its model."""

    proba: np.ndarray
    labels: np.ndarray
    classes: list[int]


def load_evaluation(
    path: Path = CURRENT_FEATURES_PATH, split: str = EVALUATION_SPLIT
) -> pd.DataFrame:
    """Read the feature table and keep the rows of the evaluation split."""
    frame = load_rows(path)
    rows = frame[frame["split"].eq(split).fillna(False)].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No rows with split == {split!r} in {path}")
    # Same day definition as build_data, so alerts_per_day matches the training runs
    rows["day"] = rows["hour"].dt.date
    return rows


def score(loaded: LoadedModel, rows: pd.DataFrame) -> Scored:
    """Predict with the model and turn its probabilities into labels with its own thresholds."""
    features = rows[FEATURE_COLUMNS].astype(SIGNATURE_DTYPES)
    proba = np.asarray(loaded.model.predict(features))
    labels = apply_thresholds(proba, loaded.classes, loaded.thresholds)
    return Scored(proba=proba, labels=labels, classes=loaded.classes)


def version_metrics(y_true: np.ndarray, scored: Scored, n_dates: int) -> dict[str, float]:
    """Return the P2 metrics of one version plus its alerts per day."""
    result = metrics(y_true, scored.labels, scored.proba, scored.classes, prefix="")
    n_alerts = int(np.count_nonzero(scored.labels != 0))
    result["alerts_per_day"] = alerts_per_day(n_alerts, n_dates)
    return result


def bootstrap_metrics(y_true: np.ndarray, scored: Scored, n_dates: int) -> dict[str, float]:
    """The four metrics of the decision rule, computed with the same functions as metrics().

    It skips the metrics the rule does not use, which makes each bootstrap replicate faster.
    """
    positions = {int(cls): index for index, cls in enumerate(scored.classes)}
    minority = [
        average_precision_score(y_true == cls, scored.proba[:, positions[cls]])
        for cls in (1, 2)
        if (y_true == cls).any()
    ]
    return {
        "pr_auc_minority_mean": float(np.mean(minority)),
        "macro_f1": f1_score(y_true, scored.labels, average="macro", zero_division=0),
        "recall_shortage": recall_score(y_true == 1, scored.labels == 1, zero_division=0),
        "alerts_per_day": alerts_per_day(int(np.count_nonzero(scored.labels != 0)), n_dates),
    }


def _subset(scored: Scored, index: np.ndarray) -> Scored:
    """Keep the given rows of one version's scores."""
    return Scored(proba=scored.proba[index], labels=scored.labels[index], classes=scored.classes)


def day_groups(days: pd.Series) -> list[np.ndarray]:
    """Return the row positions of each day, in date order."""
    codes, _ = pd.factorize(days, sort=True)
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.cumsum(np.bincount(codes))[:-1])


def paired_bootstrap(
    y_true: np.ndarray,
    champion: Scored,
    challenger: Scored,
    days: pd.Series,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """95% percentile interval of challenger - champion, resampling whole days.

    Both versions are scored on the same resampled rows in every replicate, so the comparison is
    paired. Rows of the same day are correlated, so days are resampled, not rows. All the draws
    are made first, so the result depends only on the seed.
    """
    groups = day_groups(days)
    n_dates = len(groups)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_dates, size=(n_boot, n_dates))
    differences = np.empty((n_boot, len(BOOTSTRAP_METRICS)))
    for replicate, draw in enumerate(draws):
        index = np.concatenate([groups[day] for day in draw])
        a = bootstrap_metrics(y_true[index], _subset(champion, index), n_dates)
        b = bootstrap_metrics(y_true[index], _subset(challenger, index), n_dates)
        differences[replicate] = [b[name] - a[name] for name in BOOTSTRAP_METRICS]
        if (replicate + 1) % 100 == 0:
            logger.info(f"Bootstrap: {replicate + 1}/{n_boot} replicates")

    intervals = {}
    for column, name in enumerate(BOOTSTRAP_METRICS):
        low, high = np.percentile(differences[:, column], [2.5, 97.5])
        intervals[name] = {"low": float(low), "high": float(high)}
    return intervals


def mcnemar(y_true: np.ndarray, champion: Scored, challenger: Scored) -> dict[str, float]:
    """McNemar test on the hits of each version over the rows with a risk class.

    Complementary to the bootstrap: rows of the same day are not independent, so the p-value is
    only a reference and the decision uses the bootstrap intervals.
    """
    risk = np.isin(y_true, (1, 2))
    champion_hit = champion.labels[risk] == y_true[risk]
    challenger_hit = challenger.labels[risk] == y_true[risk]
    only_champion = int(np.count_nonzero(champion_hit & ~challenger_hit))
    only_challenger = int(np.count_nonzero(~champion_hit & challenger_hit))
    discordant = only_champion + only_challenger
    statistic = (abs(only_champion - only_challenger) - 1) ** 2 / discordant if discordant else 0.0
    return {
        "only_champion_correct": only_champion,
        "only_challenger_correct": only_challenger,
        "statistic": float(statistic),
        "p_value": float(chi2.sf(statistic, df=1)) if discordant else 1.0,
    }


def breakdown(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    champion: Scored,
    challenger: Scored,
    reference_stations: set[int],
) -> dict[str, dict[str, dict]]:
    """Point metrics of both versions by target month and by common vs new stations."""
    groups = {
        "month": rows["target_time"].dt.strftime("%Y-%m"),
        "stations": np.where(rows["station_id"].isin(reference_stations), "common", "new"),
    }
    result = {}
    for name, keys in groups.items():
        result[name] = {}
        for key in sorted(pd.unique(keys)):
            index = np.flatnonzero(np.asarray(keys) == key)
            n_dates = int(rows["day"].iloc[index].nunique())
            result[name][str(key)] = {
                "rows": len(index),
                "champion": bootstrap_metrics(y_true[index], _subset(champion, index), n_dates),
                "challenger": bootstrap_metrics(
                    y_true[index], _subset(challenger, index), n_dates
                ),
            }
    return result


def decide(
    champion_metrics: dict[str, float],
    challenger_metrics: dict[str, float],
    intervals: dict[str, dict[str, float]],
) -> dict:
    """Apply the decision rule of section 4.2: the challenger wins only if every check passes."""
    checks = {
        "challenger_recall_shortage": bool(
            challenger_metrics["recall_shortage"] >= P2_MIN_RECALL_SHORTAGE
        ),
        "pr_auc_improvement": bool(intervals["pr_auc_minority_mean"]["low"] > 0),
        "macro_f1_non_inferior": bool(intervals["macro_f1"]["low"] >= -MACRO_F1_MARGIN),
        "alerts_non_inferior": bool(
            challenger_metrics["alerts_per_day"]
            <= ALERTS_MAX_RATIO * champion_metrics["alerts_per_day"]
        ),
    }
    return {"checks": checks, "winner": "challenger" if all(checks.values()) else "champion"}


AB_SUMMARY_PATH = REPORTS_DIR / "monitoring" / "ab_summary.json"
REPORT_METRICS = (
    "pr_auc_minority_mean",
    "macro_f1",
    "recall_shortage",
    "recall_saturation",
    "precision_shortage",
    "alerts_per_day",
)

app = typer.Typer()


def build_summary(
    champion_uri: str,
    challenger_uri: str,
    features_path: Path,
    split: str,
    n_boot: int,
    seed: int,
) -> dict:
    """Run the whole A/B and return everything the report needs."""
    rows = load_evaluation(features_path, split)
    y_true = rows["target"].astype(int).to_numpy()
    n_dates = int(rows["day"].nunique())

    loaded = {"champion": load_model(champion_uri), "challenger": load_model(challenger_uri)}
    scored = {name: score(model, rows) for name, model in loaded.items()}
    point = {name: version_metrics(y_true, scored[name], n_dates) for name in scored}
    intervals = paired_bootstrap(
        y_true, scored["champion"], scored["challenger"], rows["day"], n_boot=n_boot, seed=seed
    )
    reference = set(
        pd.read_parquet(CLASSIFICATION_FEATURES_PATH, columns=["station_id"])["station_id"]
    )

    return {
        "evaluation": {
            "features_path": str(features_path),
            "split": split,
            "rows": len(rows),
            "days": n_dates,
            "first_target_time": str(rows["target_time"].min()),
            "last_target_time": str(rows["target_time"].max()),
        },
        "versions": {
            name: {
                "uri": uri,
                "info": loaded[name].info,
                "metrics": {key: point[name][key] for key in REPORT_METRICS},
            }
            for name, uri in (("champion", champion_uri), ("challenger", challenger_uri))
        },
        "difference": {
            name: {
                "value": point["challenger"][name] - point["champion"][name],
                **intervals[name],
            }
            for name in BOOTSTRAP_METRICS
        },
        "bootstrap": {"replicates": n_boot, "seed": seed, "unit": "day", "interval": "95%"},
        "mcnemar": mcnemar(y_true, scored["champion"], scored["challenger"]),
        "breakdown": breakdown(rows, y_true, scored["champion"], scored["challenger"], reference),
        "decision": decide(point["champion"], point["challenger"], intervals),
    }


def log_summary(summary: dict, summary_path: Path) -> str:
    """Log the A/B as one MLflow run with both versions' metrics and the decision."""
    setup_experiment()
    # base_tags only accepts training stages; the A/B keeps its tags and changes the stage
    tags = {**base_tags("p2", "lightgbm", "candidate"), "stage_of_work": "ab_test"}
    with mlflow.start_run(run_name="p2-ab-test", tags=tags) as run:
        versions = summary["versions"]
        mlflow.log_params(
            {
                "champion_uri": versions["champion"]["uri"],
                "challenger_uri": versions["challenger"]["uri"],
                "champion_version": versions["champion"]["info"].get("model_version"),
                "challenger_version": versions["challenger"]["info"].get("model_version"),
                "bootstrap_replicates": summary["bootstrap"]["replicates"],
                "bootstrap_seed": summary["bootstrap"]["seed"],
                "winner": summary["decision"]["winner"],
            }
        )
        for name, version in versions.items():
            mlflow.log_metrics(
                {f"{name}_{key}": value for key, value in version["metrics"].items()}
            )
        for name, difference in summary["difference"].items():
            mlflow.log_metrics(
                {
                    f"diff_{name}": difference["value"],
                    f"diff_{name}_low": difference["low"],
                    f"diff_{name}_high": difference["high"],
                }
            )
        mlflow.log_artifact(str(summary_path), artifact_path="ab_test")
        return run.info.run_id


@app.command()
def main(
    champion: str = "models:/mibici-equipo05-p2-riesgo@champion",
    challenger: str = "models:/mibici-equipo05-p2-riesgo@challenger",
    features_path: Path = CURRENT_FEATURES_PATH,
    split: str = EVALUATION_SPLIT,
    n_boot: int = 1000,
    seed: int = 42,
    output: Path = AB_SUMMARY_PATH,
    log_mlflow: bool = True,
):
    """Compare champion and challenger on the evaluation split and write ab_summary.json.

    --champion and --challenger accept models:/ URIs or exported folders (registry_metadata.json),
    so the A/B can be repeated on a machine without the shared Registry.
    """
    summary = build_summary(champion, challenger, features_path, split, n_boot, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info(f"A/B summary written to {output}")
    logger.info(f"Decision: {summary['decision']}")
    if log_mlflow:
        run_id = log_summary(summary, output)
        logger.info(f"A/B logged to MLflow run {run_id}")


if __name__ == "__main__":
    app()
