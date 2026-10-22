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
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
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
    base_tags,
    data_version,
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


class WardClusterer(ClusterMixin, BaseEstimator):
    """Agglomerative Ward clustering with a ``predict`` method, so it can be logged and registered.

    ``AgglomerativeClustering`` only labels the rows it was fitted on.  This wrapper keeps Ward's
    train labels and assigns new stations (validation, December, production) to the nearest train
    cluster mean, the same rule K-Means uses, so both models are compared like for like.  Because
    Ward groups by hierarchy, ``predict`` on the train rows can disagree with ``labels_`` for a few
    border stations.  Ward needs O(n^2) memory: fine for a few hundred stations, not for P1 rows.
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
    """Return the silhouette and Davies-Bouldin scores of one split."""
    groups = len(set(labels.tolist()))
    if groups < 2 or groups >= matrix.shape[0]:
        return {f"{prefix}silhouette": float("nan"), f"{prefix}davies_bouldin": float("nan")}
    return {
        f"{prefix}silhouette": float(silhouette_score(matrix, labels)),
        f"{prefix}davies_bouldin": float(davies_bouldin_score(matrix, labels)),
    }


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


def labels_for_splits(model, matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Assign every split with the model fitted on train.

    Models are assigned with a predict method, for Ward that is the nearest centroid,
    so the validation and December stations are assigned to the nearest train cluster mean; that
    approximation is logged as a parameter and it makes the comparison consistent, because K-Means
    uses exactly the same rule.
    """
    labels = {"train": model.fit_predict(matrices["train"])}

    for split in ("validation", "test"):
        labels[split] = model.predict(matrices[split])
    return labels


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


def save_artifacts(
    folder: Path,
    names: list[str],
    matrices: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    keys_train: pd.DataFrame,
    stations: pd.DataFrame,
    title: str,
) -> None:
    """Write the cluster sizes, the hourly centroids, the profile summary and the station map."""
    sizes = (
        pd.DataFrame(
            {split: pd.Series(labels[split]).value_counts().sort_index() for split in SPLITS}
        )
        .fillna(0)
        .astype(int)
    )
    sizes.index.name = "cluster"
    sizes.to_csv(folder / "cluster_sizes.csv")

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
    figure.savefig(folder / "cluster_centroids.png", bbox_inches="tight")
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
        folder / "cluster_profile_summary.csv", index=False
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
    figure.savefig(folder / "cluster_map.png", bbox_inches="tight")
    plt.close(figure)


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
) -> dict:
    """Fit one model on train, evaluate every split and log the run in MLflow."""
    with mlflow.start_run(run_name=name) as run:
        start = perf_counter()
        labels = labels_for_splits(model, data["matrices"])
        # Includes the validation and test assignment; fit alone is not separable inside labels_for_splits
        fit_seconds = perf_counter() - start
        mlflow.log_metric("fit_seconds", fit_seconds)
        metrics = {}
        for split in SPLITS:
            prefix = "val_" if split == "validation" else f"{split}_"
            metrics.update(cluster_metrics(data["matrices"][split], labels[split], prefix))
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
            "feature_data_version": data["manifest"]["data_version"],
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
            save_artifacts(
                Path(folder),
                data["names"],
                data["matrices"],
                labels,
                data["keys"]["train"],
                data["stations"],
                figure_title,
            )
            data["k_table"].to_csv(Path(folder) / "k_selection.csv", index=False)
            mlflow.log_artifacts(folder)
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
        data["keys"][split], data["matrices"][split] = load_split(base_dir, split)
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
            stage="candidate",
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
