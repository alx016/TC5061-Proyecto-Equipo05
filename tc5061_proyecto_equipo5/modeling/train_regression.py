"""Train the regression models for the MiBici station demand question (P1).

The question is how many trips will start at each station during hour h, using only what is
known at minute 50 of hour h-1.  The rows come from the prepared P1 artifacts
(``features main --problem p1 --prepare-ready``): raw X, keys and y, never the already-fitted
ready matrices, so every candidate fits its own new pipeline on train rows only.

Protocol (docs/docs/feature-engineering.md, section 4):

* Warm-up rows without complete history are excluded with the same rule as the prepared dataset,
  never backfilled.
* Train rows come from ``training_mask`` with a cut-off before the first October scoring (labels
  still open at 23:50 of September 30 are purged).  Validation rows are the labels available
  before the first December scoring.
* Candidates are compared on validation WAPE, the primary metric of the ML Canvas.  Only the
  selected candidate is scored on the December test, once, and no refit is done.
* Predictions are clipped at zero inside the logged model, so offline metrics and scoring use the
  same policy.

All runs use the shared helpers in modeling/common.py: same experiment, tags, data version and
run context, and a model logged with its signature.  ``regression`` trains the catalog
in one go; ``train_one`` trains a single model per run for the MLflow ``train`` entry point.
"""

from contextlib import contextmanager, nullcontext
from itertools import product
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

from lightgbm import LGBMRegressor
from loguru import logger
import matplotlib
import mlflow
from mlflow.entities import ViewType
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
import typer

from tc5061_proyecto_equipo5.config import REGRESSION_FEATURES_DIR, SEED, STUDY_YEAR
from tc5061_proyecto_equipo5.features import (
    FEATURE_VERSION,
    _category_text,
    make_preprocessor,
    training_mask,
)
from tc5061_proyecto_equipo5.modeling.common import (
    EXPERIMENT_NAME,
    base_tags,
    data_version,
    log_model_with_signature,
    log_run_context,
    log_training_input,
    setup_experiment,
)

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

    The mean is learned only from the rows passed to ``fit``.  An unseen combination falls back to
    the mean of its station and then to the global train mean, so the baseline never needs future
    data to answer.
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
            f"Prepared features are version {manifest['feature_version']}, "
            f"code expects {FEATURE_VERSION}"
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


def split_ranges(keys: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, str]:
    """First and last target hour of every split, as they really entered the run."""
    ranges = {}
    for split, prefix in (("train", "train"), ("validation", "val"), ("test", "test")):
        target_time = keys.loc[masks[split], "target_time"]
        ranges[f"{prefix}_start"] = str(target_time.min())
        ranges[f"{prefix}_end"] = str(target_time.max())
    return ranges


def metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    """Return MAE, RMSE and WAPE with the given ``val_`` or ``test_`` prefix.

    WAPE divides by the real total demand; when that total is zero the metric is undefined and is
    left out instead of being patched with an epsilon.
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


def _save_residual_figures(folder: Path, y_true, y_pred, prefix: str, title: str) -> None:
    """Draw station-hour residuals, using all rows in split order (no sampling).

    Arrays deliberately discard the pandas index: predictions are in input row order,
    including when the split keeps a non-consecutive index from the prepared dataset.
    """
    predicted = np.asarray(y_pred, dtype=float)
    residuals = np.asarray(y_true, dtype=float) - predicted
    figure, axis = plt.subplots(figsize=(8, 4.5))
    try:
        axis.scatter(predicted, residuals, s=8, alpha=0.25, rasterized=True)
        axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set_title(f"Residuales vs. predicción ({title})")
        axis.set_xlabel("Predicción de viajes iniciados")
        axis.set_ylabel("Residual (real − predicho)")
        figure.tight_layout()
        figure.savefig(folder / f"{prefix}residuals_vs_predicted.png", bbox_inches="tight")
    finally:
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    try:
        axis.hist(residuals, bins=50, color="tab:blue", edgecolor="white")
        axis.set_title(f"Distribución de residuales ({title})")
        axis.set_xlabel("Residual (real − predicho)")
        axis.set_ylabel("Frecuencia")
        figure.tight_layout()
        figure.savefig(folder / f"{prefix}residuals_histogram.png", bbox_inches="tight")
    finally:
        plt.close(figure)


def log_evaluation(
    model, X, y, keys, mask, prefix: str, title: str, *, predicted=None
) -> dict[str, float]:
    """Score one split and log coherent metrics and diagnostics in the active run."""
    if predicted is None:
        predicted = model.predict(X.loc[mask])
    split_metrics = metrics(y.loc[mask], predicted, prefix)
    mlflow.log_metrics(split_metrics)
    if f"{prefix}wape" not in split_metrics:
        mlflow.set_tag(f"{prefix}wape_undefined", "real total demand is zero")
    with tempfile.TemporaryDirectory() as folder:
        figure = Path(folder) / f"{prefix}real_vs_predicted.png"
        save_series_figure(figure, keys.loc[mask], y.loc[mask], predicted, title)
        _save_residual_figures(Path(folder), y.loc[mask], predicted, prefix, title)
        mlflow.log_artifacts(folder, artifact_path="plots")
    return split_metrics


def train_and_log(
    name, model, model_params, data, seed, manifest, stage="baseline", *, _run=None
) -> dict:
    """Fit one candidate on train rows, evaluate it on validation and log the run in MLflow."""
    X, y, keys, masks = data["X"], data["y"], data["keys"], data["masks"]
    for split in ("train", "validation"):
        if not masks[split].any():
            raise ValueError(f"{split} split is empty after temporal eligibility filtering")
    X_train = X.loc[masks["train"]]
    y_train = y.loc[masks["train"]]
    X_validation = X.loc[masks["validation"]]
    params = {
        "study_year": STUDY_YEAR,
        "seed": seed,
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["validation"].sum()),
        "n_test": int(masks["test"].sum()),
        **split_ranges(keys, masks),
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
        "predict_batch_size": len(X_validation),
        "cost_measurement_protocol": (
            "perf_counter; one full pipeline fit on train; one full pipeline predict on "
            "validation; predict_ms_per_1k = seconds * 1000000 / validation rows; "
            "excludes data loading, logging, plots and signature inference; no warm-up"
        ),
        **model_params,
    }
    # The search owns its Trial context; individual training owns its own run.
    with nullcontext(_run) if _run is not None else mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        mlflow.set_tags(base_tags("p1", model_params["model"], stage))
        log_run_context(params)
        log_training_input(X_train, data["source"], "p1-train")
        fit_started = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - fit_started
        predict_started = perf_counter()
        predicted = model.predict(X_validation)
        predict_seconds = perf_counter() - predict_started
        costs = {
            "fit_seconds": fit_seconds,
            "predict_ms_per_1k": predict_seconds * 1_000_000 / len(X_validation),
        }
        mlflow.log_metrics(costs)
        val_metrics = log_evaluation(
            model,
            X,
            y,
            keys,
            masks["validation"],
            "val_",
            f"{name}, validación",
            predicted=predicted,
        )
        # skops rejects the custom estimators and the station-text function of make_preprocessor;
        # the bundle is loaded only from our own trusted runs (feature-engineering.md, section 6)
        log_model_with_signature(
            model,
            X_train,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = run.info.run_id
    logger.success(f"{name}: run {run_id} logged")
    return {"run_id": run_id, "run_name": name, "model": model, **val_metrics, **costs}


def _nominal_categories(frame):
    """Preserve ordinal codes as nominal labels, not a numeric station ordering."""
    return frame.astype("category")


def _hist_preprocessor(X):
    """Train-only, bounded-width representation local to histogram boosting.

    Native categorical splits avoid a dense station one-hot matrix. When a feature has
    more than 255 categories, infrequent train categories share the last bin; unseen
    categories encode to -1 and are handled as missing by histogram boosting.
    """
    categorical = [
        column
        for column in X
        if column in {"station_id", "municipality"}
        or pd.api.types.is_string_dtype(X[column].dtype)
        or isinstance(X[column].dtype, pd.CategoricalDtype)
    ]
    numeric = [column for column in X if column not in categorical]
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            (
                "categorical",
                make_pipeline(
                    FunctionTransformer(_category_text, feature_names_out="one-to-one"),
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        max_categories=255,
                    ),
                    FunctionTransformer(_nominal_categories, feature_names_out="one-to-one"),
                ),
                categorical,
            ),
        ],
        remainder="drop",
    ).set_output(transform="pandas")


def candidates(seed: int) -> dict:
    """Return reference configurations and additional families, with fresh pipelines."""

    def pipeline(model_name, estimator, **preprocessing):
        def build(X):
            return NonNegativeRegressor(
                make_pipeline(make_preprocessor(X, **preprocessing), estimator)
            )

        # Params come from the estimator itself, so the run logs exactly what was trained
        return build, {"model": model_name, **estimator.get_params(), **preprocessing}

    def build_hist(X):
        return NonNegativeRegressor(make_pipeline(_hist_preprocessor(X), hist_estimator))

    hist_estimator = HistGradientBoostingRegressor(
        loss="poisson",
        random_state=seed,
        early_stopping=False,
        categorical_features="from_dtype",
    )
    return {
        "regression-baseline-seasonal-mean": (
            lambda X: NonNegativeRegressor(SeasonalMeanRegressor()),
            {"model": "seasonal_mean", "group_by": ",".join(SEASONAL_COLUMNS)},
        ),
        # lsqr is deterministic, so Ridge takes no random_state
        "regression-baseline-ridge": pipeline(
            "ridge", Ridge(alpha=1.0, solver="lsqr"), numeric_scaling="robust"
        ),
        # Delivery 1 reference: gradient boosting with a Poisson loss for non-negative counts.
        "regression-baseline-lightgbm": pipeline(
            "lightgbm",
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
        "regression-candidate-lightgbm-tweedie": pipeline(
            "lightgbm_tweedie",
            LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=1.5,
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
        "regression-candidate-hist-gradient-boosting": (
            build_hist,
            {
                "model": "hist_gradient_boosting",
                **hist_estimator.get_params(),
                "numeric_scaling": "none",
                "categorical_encoding": "ordinal_native",
                "categorical_max_categories": 255,
                "categorical_unknown_value": -1,
            },
        ),
    }


def with_hyperparams(model, model_params: dict, hyperparams: dict) -> tuple:
    """Override hyperparameters of the final estimator; the logged params follow the change."""
    pipeline = model.estimator
    if not hasattr(pipeline, "steps"):
        raise ValueError(f"{model_params['model']} takes no hyperparameters")
    step, estimator = pipeline.steps[-1]
    # LightGBM's set_params accepts any name and ignores the unknown ones, so check them here
    unknown = set(hyperparams) - set(estimator.get_params())
    if unknown:
        raise ValueError(f"{model_params['model']} has no hyperparameter {sorted(unknown)}")
    model.set_params(**{f"estimator__{step}__{key}": value for key, value in hyperparams.items()})
    return model, {**model_params, **hyperparams}


def _prepared_data(features_dir: Path) -> tuple[dict, dict]:
    """Assemble the shared prepared rows, temporal masks and manifest."""
    X, keys, y, manifest = load_prepared(features_dir)
    data = {
        "X": X,
        "y": y,
        "keys": keys,
        "masks": split_masks(X, keys),
        "source": features_dir / "X.parquet",
    }
    return data, manifest


def train_one(model: str, stage: str, seed: int, hyperparams: dict) -> str:
    """Train and log one P1 model in one run: the unit of the MLflow ``train`` entry point.

    Unlike ``regression``, it does not score the test: the selection happens later, across runs.
    """
    specs = {params["model"]: (build, params) for build, params in candidates(seed).values()}
    if model not in specs:
        raise ValueError(f"model must be one of {sorted(specs)}, got {model!r}")
    # Before loading 2M rows, so a stopped tracking server fails at once
    setup_experiment()
    data, manifest = _prepared_data(REGRESSION_FEATURES_DIR)
    build, model_params = specs[model]
    estimator = build(data["X"])
    if hyperparams:
        estimator, model_params = with_hyperparams(estimator, model_params, hyperparams)
    record = train_and_log(
        f"p1-{stage}-{model}", estimator, model_params, data, seed, manifest, stage
    )
    return record["run_id"]


@contextmanager
def _search_study():
    """Adopt only an unused Study, retaining ownership of a caller's active context."""
    active = mlflow.active_run()
    project_id = os.environ.get("MLFLOW_RUN_ID")
    if active is not None and project_id not in {None, active.info.run_id}:
        raise ValueError("Active run does not match the MLflow Project run")
    existing_id = active.info.run_id if active is not None else project_id
    # ActiveRun metadata is a snapshot; fetch tags/metrics written since start_run.
    existing = mlflow.get_run(existing_id) if existing_id else None
    if existing is not None:
        expected = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
        if existing.info.experiment_id != expected:
            raise ValueError("Study must belong to the shared experiment")
        tags = existing.data.tags
        identity = {
            "question": "p1",
            "model_family": "lightgbm",
            "stage_of_work": "search",
        }
        identified_study = all(tags.get(key) == value for key, value in identity.items())
        if (
            "mlflow.parentRunId" in tags
            or any(tags.get(key, value) != value for key, value in identity.items())
            or (project_id is None and not identified_study)
        ):
            raise ValueError("Active/project run is not a compatible P1 Study")
        # A prior attempt must keep its own winner and immutable Trial history.
        # Include deleted Trials: deleting them does not make a Study unused.
        client = mlflow.MlflowClient()
        trials = client.search_runs(
            [expected],
            f"tags.mlflow.parentRunId = '{existing.info.run_id}'",
            run_view_type=ViewType.ALL,
            max_results=1,
        )
        has_selection = "best_run_id" in tags or "best_val_wape" in existing.data.metrics
        if trials or has_selection or client.list_artifacts(existing.info.run_id, "search"):
            raise ValueError("Study already contains Trials or results; start a new Study")
    if active is None:
        # start_run consumes MLFLOW_RUN_ID when MLflow Project supplied it, so the
        # following nested starts create Trials rather than resuming the Study.
        with mlflow.start_run(run_name="p1-search-lightgbm") as study:
            yield study
    else:
        # A caller may already have activated the project run without consuming the
        # environment hint. Hide it from the nested starts, then restore caller state.
        inherited = os.environ.pop("MLFLOW_RUN_ID", None)
        try:
            yield active
        except Exception:
            mlflow.MlflowClient().set_terminated(active.info.run_id, status="FAILED")
            raise
        finally:
            if inherited is not None:
                os.environ["MLFLOW_RUN_ID"] = inherited


def _log_search_results(rows):
    columns = [
        "run_id",
        "status",
        "learning_rate",
        "num_leaves",
        "n_estimators",
        "val_mae",
        "val_rmse",
        "val_wape",
        "fit_seconds",
        "predict_ms_per_1k",
        "selected",
    ]
    with tempfile.TemporaryDirectory() as folder:
        table_path = Path(folder) / "search_results.csv"
        pd.DataFrame(rows, columns=columns).to_csv(table_path, index=False)
        mlflow.log_artifact(str(table_path), artifact_path="search")


def search(model: str = "lightgbm", seed: int = SEED) -> str:
    """Compare eight Poisson Trials on validation, returning the Study run identifier.

    Models are logged in their Trials, not retained or refitted for selection. December
    is never scored by search; winning this grid does not promote a champion.
    An existing Study must have no Trials or previous search results.
    """
    if model != "lightgbm":
        raise ValueError(f"P1 search supports only 'lightgbm', got {model!r}")
    setup_experiment()
    data, manifest = _prepared_data(REGRESSION_FEATURES_DIR)
    build, reference_params = next(
        spec for spec in candidates(seed).values() if spec[1]["model"] == model
    )
    space = {"learning_rate": [0.03, 0.05], "num_leaves": [31, 63], "n_estimators": [200, 300]}
    rows = []
    with _search_study() as study:
        mlflow.set_tags(
            {
                **base_tags("p1", model, "search"),
                "mlflow.runName": "p1-search-lightgbm",
                "hypothesis": "Poisson tree complexity and learning rate improve validation WAPE",
                "objective": "minimize val_wape",
            }
        )
        mlflow.log_params({"seed": seed, "search_space": json.dumps(space)})
        try:
            for index, values in enumerate(product(*space.values()), start=1):
                hyperparams = dict(zip(space, values))
                estimator, params = with_hyperparams(
                    build(data["X"]), reference_params, hyperparams
                )
                name = f"p1-search-lightgbm-trial-{index:02d}"
                with mlflow.start_run(
                    run_name=name, nested=True, parent_run_id=study.info.run_id
                ) as trial:
                    try:
                        record = train_and_log(
                            name, estimator, params, data, seed, manifest, "search", _run=trial
                        )
                    except Exception:
                        rows.append(
                            {
                                "run_id": trial.info.run_id,
                                "status": "FAILED",
                                **hyperparams,
                                "selected": False,
                            }
                        )
                        raise
                # Do not retain fitted pipelines while comparing scalar results.
                del estimator
                record.pop("model")
                record.pop("run_name")
                if "val_wape" not in record:
                    status = "WAPE_UNDEFINED"
                else:
                    status = "FINISHED" if np.isfinite(record["val_wape"]) else "WAPE_NONFINITE"
                rows.append({**record, "status": status, **hyperparams, "selected": False})
            selectable = [row for row in rows if row["status"] == "FINISHED"]
            if not selectable:
                raise ValueError("P1 search has no selectable Trial with finite validation WAPE")
            winner = min(selectable, key=lambda row: row["val_wape"])
            winner["selected"] = True
            mlflow.log_metric("best_val_wape", winner["val_wape"])
            mlflow.set_tag("best_run_id", winner["run_id"])
        except Exception:
            try:
                _log_search_results(rows)
            except Exception:  # noqa: BLE001 - Diagnostics must not replace the training error.
                logger.exception("Could not persist partial P1 search table")
            raise
        _log_search_results(rows)
        # Mark the grid winner without reopening its finished Trial or promoting a champion.
        mlflow.MlflowClient().set_tag(winner["run_id"], "selected", "true")
        return study.info.run_id


@app.callback()
def main():
    """Train the MiBici regression baselines (keeps ``regression`` a subcommand)."""


@app.command()
def regression(
    features_dir: Path = REGRESSION_FEATURES_DIR,
    seed: int = SEED,
):
    """Train the catalog, select on validation WAPE and test the winner once."""
    data, manifest = _prepared_data(features_dir)
    X, y, keys = data["X"], data["y"], data["keys"]
    # The tracking server and the shared experiment come from modeling/common.py
    setup_experiment()
    records = []
    for name, (build, model_params) in candidates(seed).items():
        logger.info(f"Training {name}")
        stage = (
            "baseline"
            if model_params["model"] in {"seasonal_mean", "ridge", "lightgbm"}
            else "candidate"
        )
        records.append(train_and_log(name, build(X), model_params, data, seed, manifest, stage))
    table = pd.DataFrame(records).set_index("run_name")
    selected = table["val_wape"].idxmin()
    logger.success(f"Selected on validation WAPE: {selected}")
    # The December test is scored once, for the selected candidate only, after the selection is
    # frozen
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
