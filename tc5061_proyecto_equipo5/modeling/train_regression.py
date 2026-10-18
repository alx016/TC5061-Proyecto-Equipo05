"""Train the regression models for the MiBici station demand question (P1).

The question is how many trips will start at each station during hour h, using only what is known at minute 50 of
hour h-1.  The rows come from the prepared P1 artifacts (``features main --problem p1 --prepare-ready``): raw X, keys
and y, never the already-fitted ready matrices, so every candidate fits its own new pipeline on train rows only.

Protocol (docs/docs/feature-engineering.md, section 4):

* Warm-up rows without complete history are excluded with the same rule as the prepared dataset, never backfilled.
* Train rows come from ``training_mask`` with a cut-off before the first October scoring (labels still open at 23:50
  of September 30 are purged).  Validation rows are the labels available before the first December scoring.
* Candidates are compared on validation WAPE, the primary metric of the ML Canvas.  Only the selected candidate is
  scored on the December test, once, and no refit is done.
* Predictions are clipped at zero inside the logged model, so offline metrics and scoring use the same policy.

All runs use the shared helpers in modeling/common.py (same experiment, data version and Git commit) and the default
local tracking, so the artifacts stay inside the ignored mlruns/ folder.
"""

import json
from pathlib import Path
import tempfile

from lightgbm import LGBMRegressor
from loguru import logger
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
import typer

from tc5061_proyecto_equipo5.config import REGRESSION_FEATURES_DIR, STUDY_YEAR
from tc5061_proyecto_equipo5.features import FEATURE_VERSION, make_preprocessor, training_mask
from tc5061_proyecto_equipo5.modeling.common import base_tags, data_version, setup_experiment

# matplotlib has to switch to a non-interactive backend before pyplot is imported
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = typer.Typer()

FEATURE_SET = "demand_extended_v4"
# The first validation scoring is issued at 2024-09-30 23:50; ten minutes are left to fit
FIT_AS_OF = "2024-09-30 23:40"
# The first test scoring is issued at 2024-11-30 23:50; selection only sees labels closed before it
SELECTION_AS_OF = "2024-11-30 23:40"
WARMUP_POLICY = "exclude rows without complete 168h history or with missing numeric features"
POSTPROCESSING = "clip predictions at zero; target unchanged"
# Hours of the system-wide series drawn in the evidence figure (two weeks)
SERIES_HOURS = 14 * 24
# The seasonal baseline groups by station, weekday and hour (the first column is the station)
SEASONAL_COLUMNS = ("station_id", "day_of_week", "hour")


class SeasonalMeanRegressor(RegressorMixin, BaseEstimator):
    """Baseline of the ML Canvas: mean train demand per station, weekday and hour.

    The mean is learned only from the rows passed to ``fit``.  An unseen combination falls back to the mean of its
    station and then to the global train mean, so the baseline never needs future data to answer.
    """

    def __init__(self, columns=SEASONAL_COLUMNS):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y):
        columns = list(self.columns)
        frame = X.loc[:, columns].reset_index(drop=True)
        frame["y"] = np.asarray(y, dtype=float)
        self.seasonal_means_ = frame.groupby(columns)["y"].mean().rename("seasonal").reset_index()
        self.station_means_ = frame.groupby(columns[0])["y"].mean()
        self.global_mean_ = float(frame["y"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        columns = list(self.columns)
        keys = X.loc[:, columns].reset_index(drop=True)
        # A left merge keeps the order of the rows to predict
        seasonal = keys.merge(self.seasonal_means_, on=columns, how="left")["seasonal"]
        station = keys[columns[0]].map(self.station_means_)
        return seasonal.fillna(station).fillna(self.global_mean_).to_numpy()


class NonNegativeRegressor(RegressorMixin, BaseEstimator):
    """Wrap a regressor and clip its predictions at zero: a trip count cannot be negative."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.maximum(0, self.estimator_.predict(X))


def load_prepared(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Read the raw P1 artifacts and the manifest written by the prepared feature run."""
    X = pd.read_parquet(folder / "X.parquet")
    keys = pd.read_parquet(folder / "keys.parquet")
    y = pd.read_parquet(folder / "y.parquet")["demand"]
    manifest = json.loads((folder / "features_manifest.json").read_text())
    if not (len(X) == len(keys) == len(y)):
        raise ValueError("X, keys and y must be aligned row by row")
    if manifest["feature_version"] != FEATURE_VERSION:
        raise ValueError(
            f"Prepared features are version {manifest['feature_version']}, code expects {FEATURE_VERSION}"
        )
    logger.info(f"Loaded {len(X):,} rows and {X.shape[1]} columns from {folder}")
    return X, keys, y, manifest


def eligible_rows(X: pd.DataFrame, keys: pd.DataFrame) -> pd.Series:
    """Rows with complete history: the same warm-up rule used to build the prepared dataset."""
    return keys["history_complete"] & X.select_dtypes(include="number").notna().all(axis=1)


def split_masks(X: pd.DataFrame, keys: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the train, validation and test rows under the temporal protocol."""
    eligible = eligible_rows(X, keys)
    masks = {
        "train": eligible & training_mask(keys, fit_as_of=FIT_AS_OF),
        "validation": eligible
        & training_mask(keys, fit_as_of=SELECTION_AS_OF, splits=("validation",)),
        "test": eligible & keys["split"].eq("test"),
    }
    for split, mask in masks.items():
        logger.info(f"{split}: {int(mask.sum()):,} rows")
    logger.info(f"Excluded {int((~eligible).sum()):,} warm-up rows")
    return masks


def metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    """Return MAE, RMSE and WAPE with the given ``val_`` or ``test_`` prefix.

    WAPE divides by the real total demand; when that total is zero the metric is undefined and is left out instead of
    being patched with an epsilon.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    result = {
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    total = y_true.sum()
    if total > 0:
        result[f"{prefix}wape"] = float(np.abs(y_true - y_pred).sum() / total)
    return result


def save_series_figure(path: Path, keys: pd.DataFrame, y_true, y_pred, title: str) -> None:
    """Plot the real and predicted system-wide demand of the first two weeks of the split."""
    series = (
        pd.DataFrame(
            {
                "target_time": keys["target_time"].to_numpy(),
                "real": np.asarray(y_true),
                "pred": np.asarray(y_pred),
            }
        )
        .groupby("target_time")[["real", "pred"]]
        .sum()
        .sort_index()
        .iloc[:SERIES_HOURS]
    )
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(series.index, series["real"], label="Real", color="tab:blue", linewidth=1.2)
    axis.plot(series.index, series["pred"], label="Predicción", color="tab:orange", linewidth=1.2)
    axis.set_title(f"Salidas por hora en toda la red: real vs. predicción ({title})")
    axis.set_xlabel("Hora objetivo")
    axis.set_ylabel("Viajes iniciados")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def log_evaluation(model, X, y, keys, mask, prefix: str, title: str) -> dict[str, float]:
    """Score one split, log its metrics and its real-vs-predicted figure in the active run."""
    predicted = model.predict(X.loc[mask])
    split_metrics = metrics(y.loc[mask], predicted, prefix)
    mlflow.log_metrics(split_metrics)
    if f"{prefix}wape" not in split_metrics:
        mlflow.set_tag(f"{prefix}wape_undefined", "real total demand is zero")
    with tempfile.TemporaryDirectory() as folder:
        figure = Path(folder) / f"{prefix}real_vs_predicted.png"
        save_series_figure(figure, keys.loc[mask], y.loc[mask], predicted, title)
        mlflow.log_artifact(str(figure))
    return split_metrics


def train_and_log(name, model, model_params, data, seed, manifest) -> dict:
    """Fit one candidate on train rows, evaluate it on validation and log the run in MLflow."""
    X, y, keys, masks = data["X"], data["y"], data["keys"], data["masks"]
    model.fit(X.loc[masks["train"]], y.loc[masks["train"]])
    params = {
        "study_year": STUDY_YEAR,
        "seed": seed,
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["validation"].sum()),
        "n_test": int(masks["test"].sum()),
        "n_features": X.shape[1],
        "data_version": data_version(),
        "features_data_version": manifest["data_version"],
        "feature_version": FEATURE_VERSION,
        "feature_set": FEATURE_SET,
        "fit_as_of": FIT_AS_OF,
        "selection_as_of": SELECTION_AS_OF,
        "warmup_policy": WARMUP_POLICY,
        "postprocessing": POSTPROCESSING,
        "refit": False,
        **model_params,
    }
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        mlflow.set_tags(base_tags("regression"))
        val_metrics = log_evaluation(
            model, X, y, keys, masks["validation"], "val_", f"{name}, validación"
        )
        # skops rejects the custom estimators and the station-text function of make_preprocessor;
        # the bundle is loaded only from our own trusted runs (feature-engineering.md, section 6)
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = run.info.run_id
    logger.success(f"{name}: run {run_id} logged")
    return {"run_id": run_id, "run_name": name, "model": model, **val_metrics}


def candidates(seed: int) -> dict:
    """Return the three candidates of the ML Canvas, each with its own new pipeline."""

    def pipeline(estimator, **preprocessing):
        def build(X):
            return NonNegativeRegressor(
                make_pipeline(make_preprocessor(X, **preprocessing), estimator)
            )

        return build

    return {
        "regression-baseline-seasonal-mean": (
            lambda X: NonNegativeRegressor(SeasonalMeanRegressor()),
            {"model": "seasonal_mean", "group_by": ",".join(SEASONAL_COLUMNS)},
        ),
        "regression-baseline-ridge": (
            pipeline(Ridge(alpha=1.0, solver="lsqr", random_state=seed), numeric_scaling="robust"),
            {"model": "ridge", "alpha": 1.0, "solver": "lsqr", "numeric_scaling": "robust"},
        ),
        # Candidate from the ML Canvas: gradient boosting with a Poisson loss for non-negative counts
        "regression-candidate-lightgbm": (
            pipeline(
                LGBMRegressor(
                    objective="poisson",
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=63,
                    min_child_samples=100,
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1,
                ),
                numeric_scaling="none",
            ),
            {
                "model": "lightgbm",
                "objective": "poisson",
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_child_samples": 100,
                "numeric_scaling": "none",
            },
        ),
    }


@app.callback()
def main():
    """Train the MiBici regression baselines (keeps ``regression`` a subcommand)."""


@app.command()
def regression(
    features_dir: Path = REGRESSION_FEATURES_DIR,
    seed: int = 42,
):
    """Train the seasonal baseline, Ridge and LightGBM, select on validation and test the winner once."""
    X, keys, y, manifest = load_prepared(features_dir)
    data = {"X": X, "y": y, "keys": keys, "masks": split_masks(X, keys)}
    # Default local tracking comes from modeling/common.py: one shared experiment for the team
    setup_experiment()
    records = []
    for name, (build, model_params) in candidates(seed).items():
        logger.info(f"Training {name}")
        records.append(train_and_log(name, build(X), model_params, data, seed, manifest))
    table = pd.DataFrame(records).set_index("run_name")
    selected = table["val_wape"].idxmin()
    logger.success(f"Selected on validation WAPE: {selected}")
    # The December test is scored once, for the selected candidate only, after the selection is frozen
    record = table.loc[selected]
    with mlflow.start_run(run_id=record["run_id"]):
        mlflow.set_tag("selected", "true")
        test_metrics = log_evaluation(
            record["model"], X, y, keys, data["masks"]["test"], "test_", f"{selected}, test"
        )
    columns = [column for column in table.columns if column.startswith("val_")]
    logger.success("Comparison of the runs on validation:")
    print(table[["run_id", *columns]].to_string())
    logger.success(f"Test of {selected}: {test_metrics}")


if __name__ == "__main__":
    app()
