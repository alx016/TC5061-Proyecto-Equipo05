"""Train the clustering models for the MiBici station profile question.

P3 asks which groups of stations share usage patterns and where they are.  The features prepared by
``features.py`` are the 24 hourly departure proportions, the 24 hourly arrival proportions and the
global asymmetry of each station; the frozen preprocessor keeps 46 of those 49 columns because the
02-04 departure hours have zero variance in the fitting window.

The three splits are the SAME network observed in three disjoint windows (January-September,
October-November and December), so the evaluation measures temporal stability of the groups, not
performance on new stations.  A fixed-K K-Means is the reference, the number of groups is chosen on
the October-November validation window (the Canvas asks for the elbow, the silhouette and the
Davies-Bouldin index), and an agglomerative Ward model is trained for comparison.  Every run goes
to the shared MLflow experiment with the helpers in modeling/common.py.
"""

import json
from pathlib import Path
import tempfile
from time import perf_counter

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
import typer

from tc5061_proyecto_equipo5.config import (
    CLUSTERING_FEATURES_DIR,
    EXTERNAL_DATA_DIR,
    SEED,
    STUDY_YEAR,
)
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

app = typer.Typer()

SPLITS = ("train", "validation", "test")
MODELS = ("gmm", "kmeans", "kmeans_scaled", "minibatch_kmeans", "ward")
# How each model's predict labels the validation and December stations, logged with every run
ASSIGNMENTS = {
    "gmm": "highest posterior probability",
    "kmeans": "nearest centroid",
    "kmeans_scaled": "nearest centroid in standardized space",
    "minibatch_kmeans": "nearest centroid",
    "ward": "nearest train cluster mean",
}
# The ML Canvas expects between 3 and 6 functional groups
K_VALUES = (3, 4, 5, 6)
# Fixed K of the ablation already published in features_manifest.json, kept for comparability
REFERENCE_K = 4
VALID_HYPERPARAMETERS = {"n_clusters"}
# MLflow sends every numeric parameter as text; these must reach LightGBM as integers.
INTEGER_HYPERPARAMETERS = {"n_clusters"}


class WardClusterer(ClusterMixin, BaseEstimator):
    """Agglomerative Ward clustering with a ``predict`` method, so it can be logged and registered.

    ``AgglomerativeClustering`` only labels the rows it was fitted on.  This wrapper keeps Ward's
    train labels and assigns new stations (validation, December, production) to the nearest train
    cluster mean, the same rule K-Means uses, so both models are compared like for like.  Because
    Ward groups by hierarchy, ``predict`` on the train rows can disagree with ``labels_`` for a few
    border stations; the evaluation never uses ``labels_``, every split (train included) is
    labeled with ``predict``.  Ward needs O(n^2) memory: fine for a few hundred stations, not for
    P1 rows.
    """

    def __init__(self, n_clusters=REFERENCE_K):
        self.n_clusters = n_clusters

    def fit(self, X: np.ndarray, y=None):
        """Cluster the train rows with Ward and keep the mean of each cluster as its centroid.

        ``y`` is ignored; it is accepted only to follow the scikit-learn API.
        """
        self.model_ = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="ward")
        self.labels_ = self.model_.fit_predict(X, y)
        self.n_features_in_ = X.shape[1]

        self.cluster_centers_ = np.vstack(
            [X[self.labels_ == group].mean(axis=0) for group in sorted(set(self.labels_.tolist()))]
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the label of the nearest train centroid (Euclidean distance) for every row."""
        check_is_fitted(self)
        return euclidean_distances(X, self.cluster_centers_).argmin(axis=1)


def load_manifest(base_dir: Path) -> dict:
    """Read the feature manifest that describes the prepared profiles."""
    manifest_path = base_dir / "features_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No feature manifest in {base_dir}; run the p3 preparation command first"
        )
    return json.loads(manifest_path.read_text())


def column_names(manifest: dict) -> list[str]:
    """Return the selected column names without the transformer prefix."""
    return [name.split("__", 1)[-1] for name in manifest["prepared"]["selected_columns"]]


def load_split(base_dir: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the keys and the frozen profile matrix of one split.

    The matrix is already transformed by the frozen preprocessor of the preparation step, so it is
    used as-is and never scaled again.  It arrives as CSR batches; it is turned dense once because
    Ward only accepts dense input and the profile table is tiny (a few hundred stations).
    """
    folder = base_dir / "ready" / split
    keys = pd.read_parquet(folder / "keys.parquet")
    batches = sorted(folder.glob("X_*.npz"))
    if not batches:
        raise FileNotFoundError(f"No CSR batches in {folder}")
    matrix = sparse.vstack([sparse.load_npz(path) for path in batches]).toarray()
    if matrix.shape[0] != len(keys):
        raise ValueError(f"{split}: {matrix.shape[0]} rows but {len(keys)} keys")
    logger.info(f"{split}: {matrix.shape[0]} stations x {matrix.shape[1]} columns")
    return keys, matrix


def window_labels(keys: pd.DataFrame) -> str:
    """Return the [start, end) calendar window of one split, for the run parameters."""
    start = pd.to_datetime(keys["window_start"]).min()
    end = pd.to_datetime(keys["window_end"]).max()
    return f"{start:%Y-%m-%d}..{end:%Y-%m-%d}"


def cluster_metrics(matrix: np.ndarray, labels: np.ndarray, prefix: str) -> dict[str, float]:
    """Return the silhouette and Davies-Bouldin scores of one split.

    Every P3 model is scored on the frozen profile matrix, the common space where the columns
    keep their business meaning (share of trips per hour), so the scores compare across models
    even when a model clusters in another space, like the standardized ``kmeans_scaled``.
    """
    groups = len(set(labels.tolist()))
    if groups < 2 or groups >= matrix.shape[0]:
        return {f"{prefix}silhouette": float("nan"), f"{prefix}davies_bouldin": float("nan")}
    return {
        f"{prefix}silhouette": float(silhouette_score(matrix, labels)),
        f"{prefix}davies_bouldin": float(davies_bouldin_score(matrix, labels)),
    }


def model_space_metrics(model, matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Return the validation silhouette in the space the model clustered, for pipelines only.

    A pipeline transforms the profiles before clustering (``kmeans_scaled`` standardizes them),
    so this describes how well it separates the groups by its own rule.  It is descriptive: it
    does not compare across models and selection never uses it.
    """
    if not isinstance(model, Pipeline):
        return {}
    transformed = model[:-1].transform(matrix)
    groups = len(set(labels.tolist()))
    if groups < 2 or groups >= transformed.shape[0]:
        return {"val_silhouette_model_space": float("nan")}
    return {"val_silhouette_model_space": float(silhouette_score(transformed, labels))}


def temporal_ari(
    keys_a: pd.DataFrame, labels_a: np.ndarray, keys_b: pd.DataFrame, labels_b: np.ndarray
) -> float:
    """Return the adjusted Rand index between two windows' labels of the same stations.

    Every train station also has a validation and a December profile, so this asks whether the
    stations keep their group when the season changes.  Rows are matched by ``station_id`` (the
    splits are not in the same order and later windows add new stations), and the index ignores
    the label names: a whole group renamed from 0 to 2 still counts as stable.
    """
    merged = pd.DataFrame(
        {"station_id": keys_a["station_id"].to_numpy(), "label_a": labels_a}
    ).merge(
        pd.DataFrame({"station_id": keys_b["station_id"].to_numpy(), "label_b": labels_b}),
        on="station_id",
        how="inner",
    )
    if len(merged) < 2:
        return float("nan")
    return float(adjusted_rand_score(merged["label_a"], merged["label_b"]))


def choose_k(
    train_matrix: np.ndarray,
    validation_matrix: np.ndarray,
    k_values: tuple[int, ...],
    seed: int,
) -> tuple[int, pd.DataFrame]:
    """Pick the number of groups with the best validation silhouette and record the usual curves.

    The Canvas asks for the elbow method, the silhouette and the Davies-Bouldin index, so all three
    are computed for every candidate K.  Following the protocol documented in
    docs/docs/feature-engineering.md, K is chosen with the October-November window; December never
    takes part in the choice.
    """
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(train_matrix)
        validation_labels = model.predict(validation_matrix)
        rows.append(
            {
                "k": k,
                "train_silhouette": float(silhouette_score(train_matrix, labels)),
                "train_davies_bouldin": float(davies_bouldin_score(train_matrix, labels)),
                "train_inertia": float(model.inertia_),
                "validation_silhouette": float(
                    silhouette_score(validation_matrix, validation_labels)
                ),
            }
        )
        logger.info(
            f"K={k}: train silhouette {rows[-1]['train_silhouette']:.4f} | "
            f"validation silhouette {rows[-1]['validation_silhouette']:.4f} | "
            f"Davies-Bouldin {rows[-1]['train_davies_bouldin']:.4f} | "
            f"inertia {rows[-1]['train_inertia']:.1f}"
        )
    table = pd.DataFrame(rows)
    best = int(table.loc[table["validation_silhouette"].idxmax(), "k"])
    return best, table


def predict_ms_per_1k(model, X: np.ndarray) -> float:
    """Measure prediction time for 1,000 rows (or fewer) in milliseconds."""
    rows = min(1000, len(X))
    start = perf_counter()
    model.predict(X[:rows])
    return (perf_counter() - start) * 1000.0 * 1000.0 / rows


def labels_for_splits(
    model, matrices: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], float]:
    """Fit the model on train and assign every split with its predict method.

    Train, validation and December all use the same rule (for Ward, the nearest train cluster
    mean), the rule the logged model applies in production, so the temporal ARI compares like
    with like.  Returns the labels per split and the seconds spent in ``fit`` alone.
    """
    start = perf_counter()
    model.fit(matrices["train"])
    fit_seconds = perf_counter() - start

    labels = {split: model.predict(matrices[split]) for split in SPLITS}
    return labels, fit_seconds


def centroid_frame(matrix: np.ndarray, labels: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Return the mean value of every column per cluster."""
    frame = pd.DataFrame(matrix, columns=names)
    return frame.groupby(labels).mean().sort_index()


def zone_from_name(name: str) -> str:
    """Return the zone code of a station name, e.g. ``(GDL-001) ...`` -> ``GDL``."""
    match = str(name).split("-", 1)[0].strip("( ").upper()
    return match if 2 <= len(match) <= 4 else "OTRA"


def profile_summary(
    centroids: pd.DataFrame, labels_train: np.ndarray, located: pd.DataFrame
) -> pd.DataFrame:
    """Summarize each cluster: size, peak hours, mean asymmetry and most common zone.

    The business question asks which groups exist and where they are, so the summary names the peak
    hour of departures and arrivals, the average asymmetry and the zone with most stations.
    """
    rows = []
    for cluster, centroid in centroids.iterrows():
        departures = {
            int(name.rsplit("_", 1)[-1]): centroid[name]
            for name in centroids.columns
            if name.startswith("departures_hour_")
        }
        arrivals = {
            int(name.rsplit("_", 1)[-1]): centroid[name]
            for name in centroids.columns
            if name.startswith("arrivals_hour_")
        }
        members = located[located["cluster"] == cluster]
        shares = members["zone"].value_counts(normalize=True).mul(100).round(0)
        rows.append(
            {
                "cluster": int(cluster),
                "stations": int((labels_train == cluster).sum()),
                "share_percent": round(100 * float((labels_train == cluster).mean()), 2),
                "peak_departure_hour": max(departures, key=departures.get),
                "peak_arrival_hour": max(arrivals, key=arrivals.get),
                "mean_asymmetry": round(float(centroid.get("asymmetry", np.nan)), 3),
                "top_zone": shares.index[0] if len(shares) else "",
                "zone_shares": " / ".join(
                    f"{zone} {share:.0f}%" for zone, share in shares.head(3).items()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster")


def save_silhouette_plot(folder: Path, matrix: np.ndarray, labels: np.ndarray, title: str) -> None:
    """Draw the silhouette of every station, grouped by cluster, with the mean as a dashed line.

    Wide groups above zero are well separated; negative bars are stations closer to another group.
    It is drawn on the raw profile matrix, the same space as the logged silhouette metrics, and
    skipped when the split has a single group (the silhouette is undefined there).
    """
    groups = sorted(set(labels.tolist()))
    if len(groups) < 2 or len(groups) >= matrix.shape[0]:
        logger.warning(f"Silhouette plot skipped: {len(groups)} groups for {matrix.shape[0]} rows")
        return
    values = silhouette_samples(matrix, labels)
    figure, axis = plt.subplots(figsize=(7, 6))
    bottom, centers = 0, []
    for cluster in groups:
        cluster_values = np.sort(values[labels == cluster])
        top = bottom + len(cluster_values)
        axis.fill_betweenx(np.arange(bottom, top), 0, cluster_values, alpha=0.8)
        centers.append((bottom + top) / 2)
        # Leave a gap between groups so each one reads as its own block
        bottom = top + 10
    mean = float(values.mean())
    axis.axvline(mean, color="black", linestyle="--", label=f"Media = {mean:.3f}")
    axis.set_xlabel("Coeficiente de silhouette")
    # Group names on the axis, not inside the plot, where negative bars would cover them
    axis.set_yticks(centers, [f"Clúster {cluster}" for cluster in groups])
    axis.set_title(title)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "silhouette_validation.png", bbox_inches="tight")
    plt.close(figure)


def save_artifacts(
    plots: Path,
    tables: Path,
    names: list[str],
    matrices: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    keys_train: pd.DataFrame,
    stations: pd.DataFrame,
    title: str,
) -> None:
    """Write the figures to ``plots`` and the CSV tables to ``tables``.

    Figures: hourly centroids, station map and the validation silhouette per station.  Tables:
    cluster sizes in train and validation and the profile summary.
    """
    sizes = (
        pd.DataFrame(
            {
                split: pd.Series(labels[split]).value_counts().sort_index()
                for split in ("train", "validation")
            }
        )
        .fillna(0)
        .astype(int)
    )
    sizes.index.name = "cluster"
    sizes.to_csv(tables / "cluster_sizes.csv")

    centroids = centroid_frame(matrices["train"], labels["train"], names)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cluster, row in centroids.iterrows():
        for axis, prefix in zip(axes, ("departures_hour_", "arrivals_hour_")):
            # Reindex to the 24 hours so the hours dropped by the preprocessor show a gap
            values = row.filter(like=prefix).reindex([f"{prefix}{hour:02d}" for hour in range(24)])
            axis.plot(range(24), values.to_numpy(), marker="o", label=f"Clúster {cluster}")
    for axis, label in zip(axes, ("Salidas", "Llegadas")):
        axis.set_xlabel("Hora del día")
        axis.set_ylabel("Proporción media del perfil")
        axis.set_title(f"Perfil horario medio de {label.lower()} por clúster")
        axis.set_xticks(range(0, 24, 3))
        axis.legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(plots / "cluster_centroids.png", bbox_inches="tight")
    plt.close(figure)

    train_keys = pd.DataFrame({"station_id": keys_train["station_id"], "cluster": labels["train"]})
    located = train_keys.merge(stations, on="station_id", how="inner")
    located["zone"] = located["name"].map(zone_from_name)
    missing = len(train_keys) - len(located)
    if missing:
        logger.warning(
            f"{missing} train stations have no coordinates in the catalog and are skipped"
        )
    profile_summary(centroids, labels["train"], located).to_csv(
        tables / "cluster_profile_summary.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(7, 7))
    for cluster, group in located.groupby("cluster"):
        axis.scatter(
            group["longitude"],
            group["latitude"],
            s=18,
            alpha=0.8,
            label=f"Clúster {cluster} (n={len(group)})",
        )
    for zone, group in located.groupby("zone"):
        if len(group) >= 30:
            axis.annotate(
                f"{zone} ({len(group)})",
                (group["longitude"].mean(), group["latitude"].mean()),
                fontsize=9,
                weight="bold",
                ha="center",
            )
    # Longitude degrees are shorter than latitude degrees at this latitude
    axis.set_aspect(1 / np.cos(np.radians(located["latitude"].mean())))
    axis.set_xlabel("Longitud")
    axis.set_ylabel("Latitud")
    axis.set_title(f"Estaciones de la ventana de ajuste por clúster (n={len(located)})")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / "cluster_map.png", bbox_inches="tight")
    plt.close(figure)

    save_silhouette_plot(
        plots,
        matrices["validation"],
        labels["validation"],
        f"Silhouette por estación, validación (K={len(set(labels['train'].tolist()))})",
    )


def load_stations() -> pd.DataFrame:
    """Read the newest station catalog for the map.

    The portal publishes only the current snapshot, so the coordinates are assumed static; they are
    used for the map only and never as model features.
    """
    catalogs = sorted((EXTERNAL_DATA_DIR / "estaciones").glob("nomenclatura_*.csv"))
    if not catalogs:
        raise FileNotFoundError("No station catalog found; run the dataset download first")
    stations = pd.read_csv(catalogs[-1]).rename(columns={"id": "station_id"})
    return stations[["station_id", "name", "latitude", "longitude"]]


def select_winner(rows: list[dict]) -> dict:
    """Pick the Trial with the best validation silhouette; ties go to the smaller K.

    Trials whose silhouette is NaN (a K that leaves one group in validation) are ignored.
    """

    finite = [row for row in rows if np.isfinite(row["val_silhouette"])]
    if not finite:
        raise ValueError("P3 search has no Trial with a finite validation silhouette")
    best = max(finite, key=lambda row: (row["val_silhouette"], -row["k"]))

    return best


def choose_run(
    runs: pd.DataFrame, allow_dirty: bool = False, git_commit: str | None = None
) -> pd.Series:
    """Pick the P3 baseline or candidate run with the best validation silhouette.

    Ties go to the lower validation Davies-Bouldin, then to the smaller K.  Runs with a NaN
    silhouette are ignored.  Unless ``allow_dirty`` is set, only runs trained from a clean tree
    are eligible; when ``git_commit`` is given, runs from other commits are excluded, so a run
    from another branch cannot become the selected model by accident.
    """
    needed = {"tags.question", "tags.stage_of_work", "metrics.val_silhouette"}
    if not needed <= set(runs.columns):
        raise ValueError("No P3 baseline or candidate runs found")
    candidates = runs[
        (runs["tags.question"] == "p3")
        & runs["tags.stage_of_work"].isin(["baseline", "candidate"])
        & np.isfinite(runs["metrics.val_silhouette"])
    ]
    if not allow_dirty:
        candidates = candidates[candidates["tags.git_dirty"].astype(str).str.lower().eq("false")]
    if git_commit is not None:
        candidates = candidates[candidates["tags.git_commit"] == git_commit]
    if candidates.empty:
        raise ValueError("No eligible P3 baseline or candidate run found")
    ranked = candidates.assign(k=pd.to_numeric(candidates["params.n_clusters"])).sort_values(
        ["metrics.val_silhouette", "metrics.val_davies_bouldin", "k"],
        ascending=[False, True, True],
    )
    return ranked.iloc[0]


def train_and_log(
    name: str,
    model,
    kind: str,
    data: dict,
    seed: int,
    model_params: dict,
    figure_title: str,
    log_model: bool,
    stage: str = "baseline",
    nested: bool = False,
) -> dict:
    """Fit one model on train, evaluate every split and log the run in MLflow.

    ``nested=True`` opens the run as a child of the active one (a search Trial).
    """
    with mlflow.start_run(run_name=name, nested=nested) as run:
        labels, fit_seconds = labels_for_splits(model, data["matrices"])
        mlflow.log_metrics(
            {
                "fit_seconds": fit_seconds,
                "predict_ms_per_1k": predict_ms_per_1k(model, data["matrices"]["validation"]),
            }
        )
        metrics = {}
        # December is scored once, by select, for the chosen run only
        for split, prefix in (("train", "train_"), ("validation", "val_")):
            metrics.update(cluster_metrics(data["matrices"][split], labels[split], prefix))
        metrics.update(
            model_space_metrics(model, data["matrices"]["validation"], labels["validation"])
        )
        # Stability: do the train stations keep their group in the validation window?
        metrics["val_temporal_ari"] = temporal_ari(
            data["keys"]["train"],
            labels["train"],
            data["keys"]["validation"],
            labels["validation"],
        )
        params = {
            "study_year": STUDY_YEAR,
            "seed": seed,
            "model": kind,
            "n_clusters": len(set(labels["train"].tolist())),
            "n_train": data["matrices"]["train"].shape[0],
            "n_val": data["matrices"]["validation"].shape[0],
            "n_test": data["matrices"]["test"].shape[0],
            "n_features": data["matrices"]["train"].shape[1],
            "train_window": data["windows"]["train"],
            "val_window": data["windows"]["validation"],
            "test_window": data["windows"]["test"],
            "feature_set": "station_profiles",
            "feature_version": data["manifest"]["feature_version"],
            "features_data_version": data["manifest"]["data_version"],
            "data_version": data_version(),
            "k_selection": "validation silhouette; December never used"
            if kind == "kmeans"
            else "inherited from the K-Means choice",
            "assignment": ASSIGNMENTS[kind],
            **model_params,
        }
        mlflow.log_params(params)
        mlflow.set_tags(base_tags("p3", kind, stage))
        log_run_context(params)
        train = pd.DataFrame(data["matrices"]["train"], columns=data["names"])
        log_training_input(train, data["source"], "p3-train")
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as folder:
            plots, tables = Path(folder) / "plots", Path(folder) / "tables"
            plots.mkdir()
            tables.mkdir()
            save_artifacts(
                plots,
                tables,
                data["names"],
                data["matrices"],
                labels,
                data["keys"]["train"],
                data["stations"],
                figure_title,
            )
            # A search Trial has no choose_k table: the search itself is the K comparison
            if "k_table" in data:
                data["k_table"].to_csv(tables / "k_selection.csv", index=False)
            mlflow.log_artifacts(str(plots), artifact_path="plots")
            mlflow.log_artifacts(str(tables), artifact_path="tables")
        if log_model:
            log_model_with_signature(
                model,
                data["matrices"]["train"],
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
        run_id = run.info.run_id
    logger.success(f"{name}: run {run_id} logged")
    return {"run_id": run_id, "run_name": name, **metrics}


@app.callback()
def main() -> None:
    """Train the MiBici clustering models (keeps ``clustering`` a subcommand)."""


def load_data(base_dir: Path) -> dict:
    """Load the manifest, the three splits, their windows and the station catalog."""
    manifest = load_manifest(base_dir)
    data = {
        "manifest": manifest,
        "names": column_names(manifest),
        "matrices": {},
        "keys": {},
        "stations": load_stations(),
        "source": base_dir / "ready" / "train",
    }
    for split in SPLITS:
        data["keys"][split], matrix = load_split(base_dir, split)
        data["matrices"][split] = pd.DataFrame(matrix, columns=data["names"])
    data["windows"] = {split: window_labels(keys) for split, keys in data["keys"].items()}

    return data


def build_estimator(model: str, k: int, seed: int, k_selection: str) -> tuple[ClusterMixin, dict]:
    """Return the unfitted estimator of one P3 model and the params to log with its run.

    Only the requested model is built.  The params describe that model alone (a Ward run never
    shows K-Means settings) and carry ``k_selection``, so the run says whether K came from the
    validation silhouette or from the CLI.
    """
    if model == "gmm":
        estimator = GaussianMixture(
            n_components=k, covariance_type="diag", n_init=10, random_state=seed
        )
        params = {
            "n_components": k,
            "covariance_type": "diag",
            "n_init": 10,
            "k_selection": k_selection,
        }
    elif model == "kmeans":
        estimator = KMeans(n_clusters=k, n_init=10, random_state=seed)
        params = {"n_init": 10, "init": "k-means++", "n_clusters": k, "k_selection": k_selection}
    elif model == "kmeans_scaled":
        estimator = make_pipeline(
            StandardScaler(), KMeans(n_clusters=k, n_init=10, random_state=seed)
        )
        params = {
            "n_init": 10,
            "init": "k-means++",
            "n_clusters": k,
            "k_selection": k_selection,
            "scaler": "standard",
        }
    elif model == "minibatch_kmeans":
        estimator = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=seed)
        params = {"n_init": 10, "n_clusters": k, "batch_size": 1024, "k_selection": k_selection}
    elif model == "ward":
        estimator = WardClusterer(n_clusters=k)
        params = {"n_clusters": k, "linkage": "ward", "k_selection": k_selection}
    else:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}.")

    return estimator, params


def train_one(model: str, stage: str, seed: int, hyperparams: dict) -> str:
    """Train and log one P3 model in one run: the unit of the MLflow ``train`` entry point."""

    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")

    diff_hip = set(hyperparams) - set(VALID_HYPERPARAMETERS)
    if diff_hip:
        raise ValueError(
            f"P3 models only accept {sorted(VALID_HYPERPARAMETERS)}, got {sorted(diff_hip)}"
        )

    setup_experiment()
    data = load_data(CLUSTERING_FEATURES_DIR)
    best_k, k_table = choose_k(
        data["matrices"]["train"], data["matrices"]["validation"], K_VALUES, seed
    )
    data["k_table"] = k_table

    if "n_clusters" in hyperparams:
        k_selection = "given on CLI"
        k = hyperparams["n_clusters"]
    else:
        k_selection = "validation silhouette; December never used"
        k = best_k

    estimator, params = build_estimator(model, k, seed, k_selection)

    result = train_and_log(
        f"p3-{stage}-{model}-k{k}",
        estimator,
        model,
        data,
        seed,
        model_params=params,
        figure_title=f"Perfiles horarios medios por clúster - {model}, K={k} (ventana de ajuste)",
        log_model=True,
        stage=stage,
    )

    return result["run_id"]


def search(model: str = "kmeans", seed: int = SEED) -> str:
    """Train one nested Trial per K in ``K_VALUES`` and return the Study (parent) run id.

    The winner is the Trial with the best validation silhouette (ties go to the smaller K, see
    ``select_winner``); no Trial scores December, that is left to ``select``.  The Study
    keeps the search space, the winner and the comparison table in ``search/``.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
    setup_experiment()
    data = load_data(CLUSTERING_FEATURES_DIR)
    space = {"n_clusters": list(K_VALUES)}
    rows = []
    # Inside `mlflow run` this resumes the entry point's run, which ignores run_name: name it by tag
    with mlflow.start_run() as study:
        mlflow.set_tags(
            {
                **base_tags("p3", model, "search"),
                "mlflow.runName": f"p3-search-{model}",
                "hypothesis": "The K that best separates the hourly profiles lies between 3 and 6",
                "objective": "maximize val_silhouette",
            }
        )
        mlflow.log_params({"seed": seed, "search_space": json.dumps(space)})
        for k in K_VALUES:
            estimator, params = build_estimator(model, k, seed, k_selection="search trial")
            record = train_and_log(
                f"p3-search-{model}-k{k:02d}",
                estimator,
                model,
                data,
                seed,
                model_params=params,
                figure_title=f"Perfiles horarios medios por clúster — {model}, K={k} (ventana de ajuste)",
                log_model=True,
                stage="search",
                nested=True,
            )
            rows.append({"k": k, **record})
        winner = select_winner(rows)
        mlflow.log_metrics(
            {"best_val_silhouette": winner["val_silhouette"], "best_k": winner["k"]}
        )
        mlflow.set_tag("best_run_id", winner["run_id"])
        columns = ["run_id", "k", "val_silhouette", "val_davies_bouldin", "val_temporal_ari"]
        table = pd.DataFrame(rows)[columns]
        table["search_winner"] = table["run_id"] == winner["run_id"]
        with tempfile.TemporaryDirectory() as folder:
            table.to_csv(Path(folder) / "search_results.csv", index=False)
            mlflow.log_artifacts(folder, artifact_path="search")
    # The winning Trial is already closed; the client tags it without reopening it
    mlflow.MlflowClient().set_tag(winner["run_id"], "search_winner", "true")
    logger.success(
        f"p3-search-{model}: K={winner['k']} wins (val_silhouette {winner['val_silhouette']:.4f})"
    )
    return study.info.run_id


def select(run_id: str | None = None, allow_dirty: bool = False) -> str:
    """Choose the P3 run on validation, score December once and make it the only selected run.

    ``run_id`` skips ``choose_run``.  A run that is already selected is left as is, so December is
    never scored twice.  The test evidence is logged before the ``selected`` tag moves, so a
    failure keeps the previous choice and the call can be retried.
    """
    setup_experiment()
    client = mlflow.MlflowClient()
    if run_id is None:
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
        chosen = choose_run(runs, allow_dirty=allow_dirty, git_commit=git_state()["git_commit"])
        run_id = chosen["run_id"]
    run = client.get_run(run_id)
    if run.data.tags.get("question") != "p3":
        raise ValueError(f"Run {run_id} is not a P3 run")
    if run.data.tags.get("selected") == "true":
        logger.info(f"Run {run_id} is already selected; December was already scored once")
        return run_id
    if run.data.tags.get("git_dirty") == "true" and not allow_dirty:
        raise RuntimeError(
            "The run was trained from a dirty tree. Commit and retrain, or use --allow-dirty "
            "for a local test."
        )
    data = load_data(CLUSTERING_FEATURES_DIR)
    if run.data.params.get("features_data_version") != data["manifest"]["data_version"]:
        raise RuntimeError("The P3 features changed since this run was trained; retrain it")

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    labels = {split: model.predict(data["matrices"][split]) for split in ("train", "test")}
    metrics = cluster_metrics(data["matrices"]["test"], labels["test"], "test_")
    metrics["test_temporal_ari"] = temporal_ari(
        data["keys"]["train"], labels["train"], data["keys"]["test"], labels["test"]
    )
    rule = "max val_silhouette; ties: min val_davies_bouldin, then smaller K"
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("selection_rule", rule)
        mlflow.log_dict({"run_id": run_id, "rule": rule, **metrics}, "config/selection.json")

    previous = client.search_runs(
        [run.info.experiment_id], "tags.question = 'p3' and tags.selected = 'true'"
    )
    for old in previous:
        client.delete_tag(old.info.run_id, "selected")
    client.set_tag(run_id, "selected", "true")
    logger.success(f"Selected P3 run {run_id}: test silhouette {metrics['test_silhouette']:.4f}")
    return run_id


@app.command()
def clustering(
    base_dir: Path = CLUSTERING_FEATURES_DIR,
    seed: int = SEED,
):
    """Train the fixed-K reference, the best-K candidate and the Ward comparison."""
    # One shared experiment for the team, from modeling/common.py
    setup_experiment()

    data = load_data(base_dir)

    best_k, k_table = choose_k(
        data["matrices"]["train"], data["matrices"]["validation"], K_VALUES, seed
    )
    data["k_table"] = k_table
    logger.info(f"Chosen K on the validation window: {best_k}")

    records = [
        train_and_log(
            f"clustering-baseline-kmeans-k{REFERENCE_K}",
            KMeans(n_clusters=REFERENCE_K, n_init=10, random_state=seed),
            "kmeans",
            data,
            seed,
            model_params={"n_init": 10, "init": "k-means++"},
            figure_title=f"Perfiles horarios medios por clúster — K-Means, K={REFERENCE_K} (ventana de ajuste)",
            log_model=True,
            stage="baseline",
        )
    ]
    records.append(
        train_and_log(
            f"clustering-candidate-kmeans-k{best_k}",
            KMeans(n_clusters=best_k, n_init=10, random_state=seed),
            "kmeans",
            data,
            seed,
            model_params={"n_init": 10, "init": "k-means++"},
            figure_title=f"Perfiles horarios medios por clúster — K-Means, K={best_k} (ventana de ajuste)",
            log_model=True,
            stage="candidate",
        )
    )
    records.append(
        train_and_log(
            f"clustering-candidate-ward-k{best_k}",
            WardClusterer(n_clusters=best_k),
            "ward",
            data,
            seed,
            model_params={"linkage": "ward"},
            figure_title=f"Perfiles horarios medios por clúster — Ward, K={best_k} (ventana de ajuste)",
            log_model=True,
            stage="candidate",
        )
    )
    table = pd.DataFrame(records).set_index("run_name")
    logger.success("Comparison of the clustering runs:")
    print(
        table[
            [
                column
                for column in table.columns
                if column.endswith(("silhouette", "davies_bouldin"))
            ]
        ].to_string()
    )


if __name__ == "__main__":
    app()
