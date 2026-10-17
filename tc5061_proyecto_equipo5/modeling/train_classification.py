"""Train the two classification baselines for the MiBici station risk question.

The question is whether a station will be short of bikes, short of docks or balanced during hour t+1, using only what
is known at minute 50 of hour t.  The classes are severely imbalanced (about 98.7% balanced), so the baselines are
scored with macro-F1, balanced accuracy and PR-AUC instead of plain accuracy.  Both runs use the shared helpers in
modeling/common.py (same experiment, data version and Git commit) and the default local tracking, so the artifacts stay
inside the ignored mlruns/ folder.
"""

from pathlib import Path
import tempfile

from loguru import logger
import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from tc5061_proyecto_equipo5.config import (
    CLASSIFICATION_FEATURES_PATH,
    RISK_THRESHOLD,
    STUDY_YEAR,
)
from tc5061_proyecto_equipo5.modeling.common import base_tags, data_version, setup_experiment

# matplotlib has to switch to a non-interactive backend before pyplot is imported
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = typer.Typer()

FEATURE_SET = "classification_lags"
SPLITS = ("train", "validation", "test")
# Figures and the text report are labeled in Spanish; identifiers stay English.
CLASS_NAMES_ES = ["Balanceado", "Riesgo_Desabasto", "Riesgo_Saturacion"]

# Anti-leakage decision point: the forecast is made at minute 50 of hour t, so the departures, arrivals and net_flow of
# the current hour are not fully known yet and are never used as features.  Only the lagged flows, the calendar columns,
# the station size and its location are kept; station_id, hour, target and split are identifiers or the label itself.
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


def load_rows(path: Path) -> pd.DataFrame:
    """Read the feature table and drop the rows that cannot be used."""
    frame = pd.read_parquet(path)
    logger.info(f"Loaded {len(frame):,} rows from {path}")
    # The last hour of the year has no next hour (358 empty targets) and the first hours of January have empty lags
    # (warm-up); both are dropped, never filled.
    clean = frame.dropna(subset=["target"]).dropna(subset=FEATURE_COLUMNS)
    logger.info(f"Dropped {len(frame) - len(clean):,} rows with missing target or features")
    return clean


def metrics(y_true, y_pred, y_proba, prefix: str) -> dict[str, float]:
    """Return the metrics of one model, with the given ``val_`` or ``test_`` prefix.

    Accuracy is useless when one class holds 98.7% of the rows, so macro-F1 and balanced accuracy are logged too.
    The two PR-AUC values treat shortage and saturation one-vs-rest, which is the honest view of a rare-event model.
    """
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}recall_shortage": recall_score(
            y_true, y_pred, labels=[1], average="macro", zero_division=0
        ),
        f"{prefix}recall_saturation": recall_score(
            y_true, y_pred, labels=[2], average="macro", zero_division=0
        ),
        f"{prefix}pr_auc_shortage": average_precision_score(
            (y_true == 1).astype(int), y_proba[:, 1]
        ),
        f"{prefix}pr_auc_saturation": average_precision_score(
            (y_true == 2).astype(int), y_proba[:, 2]
        ),
    }


def save_artifacts(folder: Path, y_true, y_pred, title: str) -> None:
    """Write the classification report and the confusion matrix as files."""
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES_ES, zero_division=0
    )
    (folder / "classification_report.txt").write_text(report)
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1, 2], display_labels=CLASS_NAMES_ES, ax=axis, colorbar=False
    )
    axis.set_title(f"Matriz de confusión ({title})")
    axis.set_xlabel("Etiqueta predicha")
    axis.set_ylabel("Etiqueta real")
    axis.tick_params(axis="x", rotation=15)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    figure.tight_layout()
    figure.savefig(folder / "confusion_matrix.png", bbox_inches="tight")
    plt.close(figure)


def train_and_log(name, model, model_params, data, seed) -> dict:
    """Fit one baseline, evaluate it and log the whole run in MLflow."""
    features, labels = data["X"], data["y"]
    model.fit(features["train"], labels["train"])
    params = {
        "study_year": STUDY_YEAR,
        "risk_threshold": RISK_THRESHOLD,
        "seed": seed,
        "n_train": len(features["train"]),
        "n_val": len(features["validation"]),
        "n_test": len(features["test"]),
        "n_features": len(FEATURE_COLUMNS),
        "data_version": data_version(),
        "feature_set": FEATURE_SET,
        **model_params,
    }
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        mlflow.set_tags(base_tags("classification"))
        test_metrics = {}
        for split in ("validation", "test"):
            prefix = "val_" if split == "validation" else "test_"
            split_metrics = metrics(
                labels[split].to_numpy(),
                model.predict(features[split]),
                model.predict_proba(features[split]),
                prefix,
            )
            mlflow.log_metrics(split_metrics)
            if split == "test":
                test_metrics = split_metrics
        with tempfile.TemporaryDirectory() as folder:
            save_artifacts(
                Path(folder), labels["test"].to_numpy(), model.predict(features["test"]), name
            )
            mlflow.log_artifacts(folder)
        mlflow.sklearn.log_model(model, name="model")
        run_id = run.info.run_id
    logger.success(f"{name}: run {run_id} logged")
    return {"run_id": run_id, "run_name": name, **test_metrics}


@app.callback()
def main():
    """Train the MiBici classification baselines (keeps ``classification`` a subcommand)."""


@app.command()
def classification(
    features_path: Path = CLASSIFICATION_FEATURES_PATH,
    seed: int = 42,
):
    """Train the majority and logistic baselines and log them in MLflow."""
    frame = load_rows(features_path)
    for split in SPLITS:
        logger.info(f"{split}: {len(frame[frame['split'] == split]):,} rows")
    data = {
        "X": {split: frame.loc[frame["split"] == split, FEATURE_COLUMNS] for split in SPLITS},
        "y": {split: frame.loc[frame["split"] == split, "target"].astype(int) for split in SPLITS},
    }
    models = {
        "classification-baseline-majority": (
            DummyClassifier(strategy="most_frequent"),
            {"model": "dummy_most_frequent", "strategy": "most_frequent"},
        ),
        "classification-baseline-logistic": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=200, class_weight="balanced", random_state=seed
                        ),
                    ),
                ]
            ),
            {"model": "logistic_regression", "C": 1.0},
        ),
    }
    # Default local tracking comes from modeling/common.py: one shared experiment for the team
    setup_experiment()
    records = []
    for name, (model, model_params) in models.items():
        logger.info(f"Training {name}")
        records.append(train_and_log(name, model, model_params, data, seed))
    table = pd.DataFrame(records).set_index("run_name")
    columns = [column for column in table.columns if column.startswith("test_")]
    logger.success("Comparison of the two baseline runs:")
    print(table[["run_id", *columns]].to_string())


if __name__ == "__main__":
    app()
