"""Train the classification models for the MiBici station risk question (P2).

P2 predicts whether a station will be short of bikes, short of docks or balanced during the
next hour, using only information available at minute 50 of the current hour.  The classes are
very imbalanced (about 98.7% balanced), so we report macro-F1, balanced accuracy and PR-AUC
instead of plain accuracy.

The script has three jobs:

* ``train_one`` trains one model per MLflow run (used by the MLproject ``train`` entry point).
* ``search`` runs a small nested LightGBM grid (used by the ``search`` entry point).
* ``select`` chooses the best validation run and evaluates it once on December test.

Candidate runs never evaluate test: the test month is used only by ``select`` after the model
and the decision thresholds are frozen.
"""

import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
import time

from lightgbm import LGBMClassifier
from loguru import logger
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from tc5061_proyecto_equipo5.config import (
    CLASSIFICATION_FEATURES_PATH,
    P2_FIT_AS_OF,
    P2_MIN_PRECISION_SATURATION,
    P2_MIN_PRECISION_SHORTAGE,
    P2_MIN_RECALL_SHORTAGE,
    P2_SEARCH_GRID,
    P2_SELECTION_AS_OF,
    RISK_THRESHOLD,
    SEED,
    STUDY_YEAR,
)
from tc5061_proyecto_equipo5.features import training_mask
from tc5061_proyecto_equipo5.modeling.common import (
    EXPERIMENT_NAME,
    base_tags,
    data_version,
    git_state,
    log_model_with_signature,
    log_run_context,
    log_training_input,
    setup_experiment,
)

# matplotlib has to switch to a non-interactive backend before pyplot is imported
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = typer.Typer()

FEATURE_SET = "classification_lags"
SPLITS = ("train", "validation", "test")
# Figures and the text report are labeled in Spanish; identifiers stay English.
CLASS_NAMES_ES = ["Balanceado", "Riesgo_Desabasto", "Riesgo_Saturacion"]
# Models from the first delivery; they are rerun as the baseline of this delivery.
BASELINES = {"persistence", "dummy_most_frequent", "logistic_regression", "lightgbm"}
# MLflow sends every numeric parameter as text; these must reach the tree estimators as integers.
INTEGER_HYPERPARAMS = {
    "n_estimators",
    "num_leaves",
    "min_child_samples",
    "max_depth",
    "min_samples_leaf",
    "max_leaf_nodes",
}

# Anti-leakage decision point: the forecast is made at minute 50 of hour t, so the departures,
# arrivals and net_flow of the current hour are not fully known yet and are never used as
# features.  Only lagged flows, calendar columns, station size and location are kept.
FEATURE_COLUMNS = [
    "capacity",
    "latitude",
    "longitude",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "neighbor_net_flow",
    "net_flow_lag1",
    "net_flow_lag2",
    "net_flow_lag24",
    "arrivals_lag1",
    "arrivals_lag2",
    "arrivals_lag3",
    "departures_lag1",
    "departures_lag2",
    "departures_lag3",
]


class PersistenceRiskClassifier(BaseEstimator, ClassifierMixin):
    """Simple baseline: reuse the risk implied by the flow of two hours ago.

    ``net_flow_lag2`` is already normalized by capacity in ``features.py``, so it is compared
    directly with the same ``tau`` used for the target; there are no learned parameters.  ``X``
    may be a DataFrame with that column or an array whose columns follow ``FEATURE_COLUMNS``, so
    scikit-learn utilities that pass arrays can use the estimator too.
    """

    def __init__(self, tau: float = RISK_THRESHOLD):
        self.tau = tau

    def fit(self, X, y):
        self.classes_ = np.array([0, 1, 2], dtype=np.int8)
        return self

    def _normalized_flow(self, X) -> np.ndarray:
        """Return ``net_flow_lag2`` for a DataFrame or a FEATURE_COLUMNS array."""
        if isinstance(X, pd.DataFrame):
            if "net_flow_lag2" not in X.columns:
                raise ValueError("X is missing columns: ['net_flow_lag2']")
            flow = X["net_flow_lag2"].to_numpy(dtype=float)
        else:
            values = np.asarray(X, dtype=float)
            if values.ndim != 2 or values.shape[1] != len(FEATURE_COLUMNS):
                raise ValueError(
                    f"X must be a DataFrame or an array with {len(FEATURE_COLUMNS)} columns "
                    "in FEATURE_COLUMNS order"
                )
            flow = values[:, FEATURE_COLUMNS.index("net_flow_lag2")]
        return flow

    def predict(self, X):
        flow = self._normalized_flow(X)
        prediction = np.zeros(len(flow), dtype=np.int8)
        prediction[flow <= -self.tau] = 1
        prediction[flow >= self.tau] = 2
        return prediction

    def predict_proba(self, X):
        prediction = self.predict(X)
        proba = np.zeros((len(prediction), 3), dtype=float)
        proba[np.arange(len(prediction)), prediction] = 1.0
        return proba


def load_rows(path: Path) -> pd.DataFrame:
    """Read the feature table and add the timestamps needed by the temporal masks."""
    frame = pd.read_parquet(path)
    frame["hour"] = pd.to_datetime(frame["hour"])
    logger.info(f"Loaded {len(frame):,} rows from {path}")

    # The last hour of the year has no next hour and January starts with empty lags, so both are dropped.
    frame = frame.dropna(subset=["target"]).dropna(subset=FEATURE_COLUMNS)
    frame["target_time"] = frame["hour"] + pd.Timedelta(hours=1)
    frame["label_available_at"] = frame["target_time"] + pd.Timedelta(hours=1)
    logger.info(f"Kept {len(frame):,} rows with target and features")
    return frame


def split_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the train, validation and test masks under the temporal protocol.

    ``training_mask`` removes labels that were not closed before the next split starts; test is
    never passed through it because it is not used for training or selection.
    """
    keys = frame[["station_id", "target_time", "label_available_at", "split"]]
    return {
        "train": training_mask(keys, fit_as_of=P2_FIT_AS_OF, splits=("train",)),
        "validation": training_mask(keys, fit_as_of=P2_SELECTION_AS_OF, splits=("validation",)),
        "test": keys["split"].eq("test").fillna(False).astype(bool),
    }


def build_data(frame: pd.DataFrame, source: Path) -> dict:
    """Build X/y per split and log the row counts once."""
    masks = split_masks(frame)
    data = {"X": {}, "y": {}, "n": {}, "source": source}
    for split in SPLITS:
        mask = masks[split]
        X = frame.loc[mask, FEATURE_COLUMNS]
        y = frame.loc[mask, "target"].astype(int)
        if X.empty:
            raise ValueError(f"The {split} split is empty after applying the temporal masks")
        data["X"][split] = X
        data["y"][split] = y
        data["n"][split] = len(X)
        logger.info(f"{split}: {len(X):,} rows")
    data["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return data


def candidates(seed: int) -> dict:
    """Return the baseline and candidate models, each with the parameters that built it.

    Every value is ``(builder, params)``; the builder returns a new unfitted estimator for each run.
    """
    return {
        "persistence": (
            lambda: PersistenceRiskClassifier(RISK_THRESHOLD),
            {"model": "persistence", "tau": RISK_THRESHOLD},
        ),
        "dummy_most_frequent": (
            lambda: DummyClassifier(strategy="most_frequent"),
            {"model": "dummy_most_frequent", "strategy": "most_frequent"},
        ),
        "logistic_regression": (
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=300,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                "model": "logistic_regression",
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 300,
            },
        ),
        "lightgbm": (
            lambda: LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
            {
                "model": "lightgbm",
                "n_estimators": 200,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "min_child_samples": 20,
                "class_weight": "balanced",
            },
        ),
        "lightgbm_unweighted": (
            lambda: LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
            {
                "model": "lightgbm_unweighted",
                "n_estimators": 200,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "min_child_samples": 20,
            },
        ),
        "hist_gradient_boosting": (
            lambda: HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=seed,
            ),
            {
                "model": "hist_gradient_boosting",
                "max_iter": 200,
                "learning_rate": 0.1,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "class_weight": "balanced",
            },
        ),
        "random_forest": (
            lambda: RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
            {
                "model": "random_forest",
                "n_estimators": 200,
                "min_samples_leaf": 20,
                "class_weight": "balanced_subsample",
            },
        ),
    }


def with_hyperparams(model, model_params: dict, hyperparams: dict) -> tuple:
    """Apply CLI hyperparameters to a pipeline or to a plain estimator."""
    if hasattr(model, "steps"):
        step_name, estimator = model.steps[-1]
        prefix = f"{step_name}__"
    else:
        estimator = model
        prefix = ""

    unknown = set(hyperparams) - set(estimator.get_params())
    if unknown:
        raise ValueError(f"{model_params['model']} has no hyperparameter {sorted(unknown)}")
    if hyperparams:
        model.set_params(**{f"{prefix}{key}": value for key, value in hyperparams.items()})
    return model, {**model_params, **hyperparams}


def build_model(name: str, seed: int, hyperparams: dict) -> tuple:
    """Build one model by name and apply the requested hyperparameters."""
    available = candidates(seed)
    if name not in available:
        raise ValueError(f"model must be one of {sorted(available)}, got {name!r}")
    builder, model_params = available[name]
    return with_hyperparams(builder(), model_params, hyperparams)


def calibrate_thresholds(
    y_true,
    proba,
    classes,
    min_precision_shortage: float = P2_MIN_PRECISION_SHORTAGE,
    min_precision_saturation: float = P2_MIN_PRECISION_SATURATION,
) -> dict[int, float]:
    """Choose one probability threshold per risk class using validation data only.

    For each class we look for the threshold with the largest recall whose precision is at
    least the configured floor.  If no threshold reaches the floor we fall back to 0.5 and the
    run keeps working; ``select`` can still report the real precision and recall.
    """
    positions = {int(cls): index for index, cls in enumerate(classes)}
    thresholds = {}
    floors = {1: min_precision_shortage, 2: min_precision_saturation}

    for cls in (1, 2):
        truth = np.asarray(y_true) == cls
        scores = proba[:, positions[cls]]
        best = None
        for threshold in np.linspace(0.05, 0.95, 19):
            prediction = scores >= threshold
            if not prediction.any():
                continue
            precision = precision_score(truth, prediction, zero_division=0)
            recall = recall_score(truth, prediction, zero_division=0)
            if precision >= floors[cls] and (
                best is None or (recall, precision) > (best[0], best[1])
            ):
                best = (recall, precision, float(threshold))
        thresholds[cls] = best[2] if best else 0.50
    return thresholds


def apply_thresholds(proba, classes, thresholds: dict[int, float]) -> np.ndarray:
    """Turn class probabilities into labels using the two risk thresholds."""
    positions = {int(cls): index for index, cls in enumerate(classes)}
    shortage = proba[:, positions[1]]
    saturation = proba[:, positions[2]]
    above_shortage = shortage >= thresholds[1]
    above_saturation = saturation >= thresholds[2]

    prediction = np.zeros(len(proba), dtype=np.int8)
    prediction[above_shortage & ~above_saturation] = 1
    prediction[above_saturation & ~above_shortage] = 2
    both = above_shortage & above_saturation
    prediction[both] = np.where(shortage[both] >= saturation[both], 1, 2)
    return prediction


def metrics(y_true, y_pred, proba, classes, prefix: str) -> dict[str, float]:
    """Return the P2 metrics with the ``val_`` or ``test_`` prefix.

    The probability columns are mapped with ``model.classes_``; never assume class 1 is column 1.
    PR-AUC is one-vs-rest and macro-averaged over the classes with positive examples.
    """
    positions = {int(cls): index for index, cls in enumerate(classes)}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba)

    result = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    for cls, name in ((1, "shortage"), (2, "saturation")):
        result[f"{prefix}precision_{name}"] = precision_score(
            y_true == cls, y_pred == cls, zero_division=0
        )
        result[f"{prefix}recall_{name}"] = recall_score(
            y_true == cls, y_pred == cls, zero_division=0
        )
        if (y_true == cls).any():
            result[f"{prefix}pr_auc_{name}"] = average_precision_score(
                (y_true == cls).astype(int), proba[:, positions[cls]]
            )

    average_precisions = []
    for cls in (0, 1, 2):
        if (y_true == cls).any():
            average_precisions.append(
                average_precision_score((y_true == cls).astype(int), proba[:, positions[cls]])
            )
    if average_precisions:
        result[f"{prefix}pr_auc_macro"] = float(np.mean(average_precisions))
    return result


def save_plots(folder: Path, y_true, y_pred, proba, classes, title: str) -> None:
    """Write the classification report, two confusion matrices and the two PR curves."""
    folder.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES_ES, zero_division=0
    )
    (folder / "classification_report.txt").write_text(report)

    for filename, normalize, values_format in (
        ("confusion_matrix.png", None, "d"),
        ("confusion_matrix_normalized.png", "true", ".2f"),
    ):
        figure, axis = plt.subplots(figsize=(6.5, 4.5))
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            display_labels=CLASS_NAMES_ES,
            normalize=normalize,
            values_format=values_format,
            ax=axis,
            colorbar=False,
        )
        axis.set_title(f"Matriz de confusión ({title})")
        axis.set_xlabel("Etiqueta predicha")
        axis.set_ylabel("Etiqueta real")
        axis.tick_params(axis="x", rotation=15)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
        figure.tight_layout()
        figure.savefig(folder / filename, bbox_inches="tight")
        plt.close(figure)

    positions = {int(cls): index for index, cls in enumerate(classes)}
    for cls, name in ((1, "shortage"), (2, "saturation")):
        precision, recall, _ = precision_recall_curve(
            (np.asarray(y_true) == cls).astype(int), proba[:, positions[cls]]
        )
        average = average_precision_score(
            (np.asarray(y_true) == cls).astype(int), proba[:, positions[cls]]
        )
        figure, axis = plt.subplots(figsize=(6, 4.5))
        axis.plot(recall, precision, label=f"AP = {average:.3f}")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title(f"Curva precisión-recall: {name} ({title})")
        axis.legend()
        figure.tight_layout()
        figure.savefig(folder / f"pr_curve_{name}.png", bbox_inches="tight")
        plt.close(figure)


def predict_ms_per_1k(model, X) -> float:
    """Measure prediction time for 1,000 rows (or fewer) in milliseconds."""
    rows = min(1000, len(X))
    started = time.perf_counter()
    model.predict_proba(X.iloc[:rows])
    elapsed = time.perf_counter() - started
    return elapsed * 1000.0 * 1000.0 / rows


def train_and_log(
    name: str,
    model,
    model_params: dict,
    data: dict,
    seed: int,
    stage: str,
    model_family: str | None = None,
    nested: bool = False,
    log_model: bool = True,
) -> dict:
    """Fit one model, evaluate validation only and log the complete run."""
    X_train, y_train = data["X"]["train"], data["y"]["train"]
    X_val, y_val = data["X"]["validation"], data["y"]["validation"]
    model_family = model_family or model_params.get("model", name)

    with mlflow.start_run(run_name=name, nested=nested) as run:
        mlflow.set_tags(base_tags("p2", model_family, stage))

        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        proba_val = model.predict_proba(X_val)
        thresholds = calibrate_thresholds(y_val.to_numpy(), proba_val, model.classes_)
        prediction_val = apply_thresholds(proba_val, model.classes_, thresholds)
        val_metrics = metrics(y_val.to_numpy(), prediction_val, proba_val, model.classes_, "val_")
        val_metrics["fit_seconds"] = fit_seconds
        val_metrics["predict_ms_per_1k"] = predict_ms_per_1k(model, X_val)

        params = {
            "study_year": STUDY_YEAR,
            "risk_threshold": RISK_THRESHOLD,
            "seed": seed,
            "n_train": data["n"]["train"],
            "n_val": data["n"]["validation"],
            "n_test": data["n"]["test"],
            "n_features": len(FEATURE_COLUMNS),
            "data_version": data_version(),
            "feature_set": FEATURE_SET,
            "feature_table_sha256": data["source_sha256"],
            "fit_as_of": P2_FIT_AS_OF,
            "selection_as_of": P2_SELECTION_AS_OF,
            "threshold_shortage": thresholds[1],
            "threshold_saturation": thresholds[2],
            "refit": False,
            **model_params,
        }
        mlflow.log_params(params)
        mlflow.log_metrics(val_metrics)
        log_run_context(params)
        log_training_input(X_train, data["source"], "p2-train")

        with tempfile.TemporaryDirectory() as folder:
            save_plots(
                Path(folder),
                y_val.to_numpy(),
                prediction_val,
                proba_val,
                model.classes_,
                f"{name}, validación",
            )
            mlflow.log_artifacts(folder, artifact_path="plots/val")

        if log_model:
            log_model_with_signature(
                model,
                X_train,
                flavor="sklearn",
                predict_fn="predict_proba",
                pyfunc_predict_fn="predict_proba",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
        run_id = run.info.run_id

    logger.success(f"{name}: run {run_id} logged")
    return {
        "run_id": run_id,
        "run_name": name,
        "model": model,
        "thresholds": thresholds,
        **val_metrics,
    }


def train_one(model: str, stage: str, seed: int, hyperparams: dict) -> str:
    """Train and log one P2 model: the unit of the MLproject ``train`` entry point."""
    setup_experiment()
    frame = load_rows(CLASSIFICATION_FEATURES_PATH)
    data = build_data(frame, CLASSIFICATION_FEATURES_PATH)
    estimator, model_params = build_model(model, seed, hyperparams)
    record = train_and_log(
        name=f"p2-{stage}-{model}",
        model=estimator,
        model_params=model_params,
        data=data,
        seed=seed,
        stage=stage,
        model_family=model,
    )
    return record["run_id"]


def choose_run(
    runs: pd.DataFrame,
    min_recall_shortage: float = P2_MIN_RECALL_SHORTAGE,
    min_precision_shortage: float = P2_MIN_PRECISION_SHORTAGE,
    min_precision_saturation: float = P2_MIN_PRECISION_SATURATION,
    allow_dirty: bool = False,
    git_commit: str | None = None,
) -> pd.Series:
    """Pick the best validation run, applying the guardrails and the evidence filters.

    Unless ``allow_dirty`` is set, only runs trained from a clean tree are eligible.  When
    ``git_commit`` is given, runs from other commits are excluded, so a clean run from another
    branch cannot become the selected evidence by accident.
    """
    if "tags.question" not in runs.columns or "tags.stage_of_work" not in runs.columns:
        raise ValueError("No P2 baseline or candidate runs found")
    if not allow_dirty and "tags.git_dirty" not in runs.columns:
        raise ValueError("Runs are missing the git_dirty tag")
    if git_commit is not None and "tags.git_commit" not in runs.columns:
        raise ValueError("Runs are missing the git_commit tag")
    candidates_frame = runs[
        (runs["tags.question"] == "p2")
        & (runs["tags.stage_of_work"].isin(["baseline", "candidate"]))
    ].copy()
    if not allow_dirty:
        candidates_frame = candidates_frame[candidates_frame["tags.git_dirty"] == "false"]
        if git_commit is not None:
            candidates_frame = candidates_frame[candidates_frame["tags.git_commit"] == git_commit]
    if candidates_frame.empty:
        clean = "clean " if not allow_dirty else ""
        commit_note = f" from {git_commit[:7]}" if git_commit else ""
        raise ValueError(f"No {clean}P2 baseline or candidate runs found{commit_note}")
    if "metrics.val_macro_f1" not in candidates_frame:
        candidates_frame["metrics.val_macro_f1"] = 0.0
    candidates_frame = candidates_frame.sort_values(
        "metrics.val_macro_f1", ascending=False, na_position="last"
    )

    for _, row in candidates_frame.iterrows():
        recall = row.get("metrics.val_recall_shortage", 0.0)
        precision_shortage = row.get("metrics.val_precision_shortage", 0.0)
        precision_saturation = row.get("metrics.val_precision_saturation", 0.0)
        if (
            pd.notna(recall)
            and pd.notna(precision_shortage)
            and pd.notna(precision_saturation)
            and recall >= min_recall_shortage
            and precision_shortage >= min_precision_shortage
            and precision_saturation >= min_precision_saturation
        ):
            return row
    return candidates_frame.iloc[0]


def search(model: str = "lightgbm", seed: int = SEED) -> str:
    """Run a small nested grid search and promote its winner to a candidate.

    Trials compare configurations on validation only and do not keep a model.  The winning
    combination is rebuilt with the same seed and trained once more as a ``candidate`` run that
    logs its model, so ``select`` can compare it like any other candidate.
    """
    available = candidates(seed)
    if model not in available:
        raise ValueError(f"model must be one of {sorted(available)}, got {model!r}")

    setup_experiment()
    frame = load_rows(CLASSIFICATION_FEATURES_PATH)
    data = build_data(frame, CLASSIFICATION_FEATURES_PATH)

    keys = list(P2_SEARCH_GRID)
    combinations = list(itertools.product(*(P2_SEARCH_GRID[key] for key in keys)))

    with mlflow.start_run(run_name=f"p2-search-{model}") as parent:
        mlflow.set_tags(base_tags("p2", model, "search"))
        mlflow.log_param("search_space", json.dumps(P2_SEARCH_GRID))
        mlflow.log_param("n_trials", len(combinations))
        mlflow.log_param("data_version", data_version())
        parent_id = parent.info.run_id

        records = []
        for index, values in enumerate(combinations):
            hyperparams = dict(zip(keys, values))
            estimator, model_params = build_model(model, seed, hyperparams)
            record = train_and_log(
                name=f"p2-search-{model}-{index:02d}",
                model=estimator,
                model_params=model_params,
                data=data,
                seed=seed,
                stage="search",
                model_family=model,
                nested=True,
                log_model=False,
            )
            records.append({**record, "hyperparams": hyperparams})

        table = pd.DataFrame(records)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "search_results.csv"
            table.drop(columns=["model", "thresholds"], errors="ignore").to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="search")

        best = table.sort_values("val_macro_f1", ascending=False).iloc[0]
        mlflow.log_metric("best_val_macro_f1", float(best["val_macro_f1"]))
        mlflow.log_param("best_run_id", best["run_id"])

        # Retrain the winning configuration as a selectable candidate (validation only).
        estimator, model_params = build_model(model, seed, best["hyperparams"])
        candidate = train_and_log(
            name=f"p2-candidate-{model}-search-winner",
            model=estimator,
            model_params=model_params,
            data=data,
            seed=seed,
            stage="candidate",
            model_family=model,
            nested=True,
            log_model=True,
        )
        client = mlflow.MlflowClient()
        client.set_tag(candidate["run_id"], "search_parent_run_id", parent_id)
        client.set_tag(candidate["run_id"], "source_trial_run_id", best["run_id"])
        mlflow.log_param("best_candidate_run_id", candidate["run_id"])

    logger.success(
        f"Search finished in run {parent_id}; best child {best['run_id']} "
        f"promoted to candidate {candidate['run_id']}"
    )
    return parent_id


@app.callback()
def main():
    """Train, search and select the MiBici P2 risk models."""


@app.command()
def classification(
    features_path: Path = CLASSIFICATION_FEATURES_PATH,
    seed: int = SEED,
):
    """Train all P2 baselines and candidates once (no test evaluation)."""
    setup_experiment()
    frame = load_rows(features_path)
    data = build_data(frame, features_path)

    records = []
    for name in candidates(seed):
        stage = "baseline" if name in BASELINES else "candidate"
        estimator, model_params = build_model(name, seed, {})
        records.append(
            train_and_log(
                name=f"p2-{stage}-{name}",
                model=estimator,
                model_params=model_params,
                data=data,
                seed=seed,
                stage=stage,
                model_family=name,
            )
        )

    table = pd.DataFrame(records)
    columns = ["run_name", "run_id"] + [
        column for column in table.columns if column.startswith("val_")
    ]
    logger.success("Validation comparison:")
    print(table[columns].to_string(index=False))


def normalize_run_id(run_id: str | None) -> str | None:
    """Map MLflow's placeholder for an omitted run id to ``None``.

    MLflow project parameters cannot be empty, so the ``select`` entry point passes ``none``
    when the user does not choose a run.  Direct CLI use can omit the option entirely.
    """
    if run_id is None or run_id.strip().lower() in {"", "none"}:
        return None
    return run_id


@app.command()
def select(
    question: str = "p2",
    run_id: str | None = None,
    allow_dirty: bool = False,
):
    """Choose the best validation run and evaluate December test once."""
    if question != "p2":
        raise typer.BadParameter("select is implemented for p2 only")

    run_id = normalize_run_id(run_id)
    # MLflow sets this variable for the project run opened by ``mlflow run``.  Keep it to link
    # the select run with the candidate that holds the test metrics.
    selection_run_id = os.environ.get("MLFLOW_RUN_ID")

    setup_experiment()
    frame = load_rows(CLASSIFICATION_FEATURES_PATH)
    data = build_data(frame, CLASSIFICATION_FEATURES_PATH)

    if run_id is None:
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
        chosen = choose_run(runs, allow_dirty=allow_dirty, git_commit=git_state()["git_commit"])
        run_id = chosen["run_id"]

    run = mlflow.get_run(run_id)
    if run.data.tags.get("selected") == "true":
        logger.info(f"Run {run_id} is already selected; test was already evaluated once.")
        if selection_run_id and selection_run_id != run_id:
            mlflow.MlflowClient().set_tag(selection_run_id, "selected_run_id", run_id)
        return
    if run.data.tags.get("git_dirty") == "true" and not allow_dirty:
        raise RuntimeError(
            "The selected run has git_dirty=true. Commit or stash your changes, or use "
            "--allow-dirty for a local test."
        )
    expected_hash = run.data.params.get("feature_table_sha256")
    if expected_hash and expected_hash != data["source_sha256"]:
        raise RuntimeError(
            "The feature table changed since this run was trained. Rerun the candidate "
            "with the current data/processed/classification_features.parquet."
        )

    thresholds = {
        1: float(run.data.params.get("threshold_shortage", 0.50)),
        2: float(run.data.params.get("threshold_saturation", 0.50)),
    }
    fallback = (
        run.data.metrics.get("val_recall_shortage", 0.0) < P2_MIN_RECALL_SHORTAGE
        or run.data.metrics.get("val_precision_shortage", 0.0) < P2_MIN_PRECISION_SHORTAGE
        or run.data.metrics.get("val_precision_saturation", 0.0) < P2_MIN_PRECISION_SATURATION
    )
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    proba_test = model.predict_proba(data["X"]["test"])
    prediction_test = apply_thresholds(proba_test, model.classes_, thresholds)
    test_metrics = metrics(
        data["y"]["test"].to_numpy(), prediction_test, proba_test, model.classes_, "test_"
    )

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("selection_fallback", str(fallback).lower())
        if selection_run_id and selection_run_id != run_id:
            mlflow.set_tag("selection_run_id", selection_run_id)
        mlflow.log_metrics(test_metrics)
        with tempfile.TemporaryDirectory() as folder:
            save_plots(
                Path(folder),
                data["y"]["test"].to_numpy(),
                prediction_test,
                proba_test,
                model.classes_,
                "test, diciembre",
            )
            mlflow.log_artifacts(folder, artifact_path="plots/test")
        mlflow.log_dict(
            {
                "run_id": run_id,
                "rule": "best val_macro_f1 subject to recall/precision guardrails",
                "fallback": fallback,
                "threshold_shortage": thresholds[1],
                "threshold_saturation": thresholds[2],
            },
            "config/selection.json",
        )
        if selection_run_id and selection_run_id != run_id:
            client = mlflow.MlflowClient()
            client.set_tag(selection_run_id, "selected_run_id", run_id)
            client.set_tag(selection_run_id, "selection_fallback", str(fallback).lower())
        # Mark the run only after all test evidence is logged, so a failure keeps it retryable.
        mlflow.set_tag("selected", "true")

    logger.success(f"Selected run {run_id}")
    print(pd.Series(test_metrics).to_string())


if __name__ == "__main__":
    app()
