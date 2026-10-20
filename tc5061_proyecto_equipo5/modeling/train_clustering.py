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

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import typer

from tc5061_proyecto_equipo5.config import CLUSTERING_FEATURES_DIR, EXTERNAL_DATA_DIR, STUDY_YEAR
from tc5061_proyecto_equipo5.modeling.common import base_tags, data_version, setup_experiment

app = typer.Typer()

# P3 artifacts, written by `features build --question p3`
DEFAULT_BASE_DIR = CLUSTERING_FEATURES_DIR
SPLITS = ("train", "validation", "test")
# The ML Canvas expects between 3 and 6 functional groups
K_VALUES = (3, 4, 5, 6)
# Fixed K of the ablation already published in features_manifest.json, kept for comparability
REFERENCE_K = 4


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


def labels_for_splits(model, matrices: dict[str, np.ndarray], kind: str) -> dict[str, np.ndarray]:
    """Assign every split with the model fitted on train.

    K-Means predicts directly.  Ward has no predict method, so the validation and December stations
    are assigned to the nearest train cluster mean; that approximation is logged as a parameter and
    it makes the comparison consistent, because K-Means uses exactly the same rule.
    """
    labels = {"train": model.fit_predict(matrices["train"])}
    if kind == "kmeans":
        for split in ("validation", "test"):
            labels[split] = model.predict(matrices[split])
        return labels
    means = np.vstack(
        [
            matrices["train"][labels["train"] == group].mean(axis=0).ravel()
            for group in sorted(set(labels["train"].tolist()))
        ]
    )
    for split in ("validation", "test"):
        labels[split] = euclidean_distances(matrices[split], means).argmin(axis=1)
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
) -> dict:
    """Fit one model on train, evaluate every split and log the run in MLflow."""
    labels = labels_for_splits(model, data["matrices"], kind)
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
        "assignment": "predict" if kind == "kmeans" else "nearest train cluster mean",
        **model_params,
    }
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        mlflow.set_tags(base_tags("clustering"))
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
            mlflow.sklearn.log_model(model, name="model")
        run_id = run.info.run_id
    logger.success(f"{name}: run {run_id} logged")
    return {"run_id": run_id, "run_name": name, **metrics}


@app.callback()
def main() -> None:
    """Train the MiBici clustering models (keeps ``clustering`` a subcommand)."""


@app.command()
def clustering(
    base_dir: Path = DEFAULT_BASE_DIR,
    seed: int = 42,
):
    """Train the fixed-K reference, the best-K candidate and the Ward comparison."""
    # One shared experiment for the team, from modeling/common.py
    setup_experiment()
    manifest = load_manifest(base_dir)
    data = {
        "manifest": manifest,
        "names": column_names(manifest),
        "matrices": {},
        "keys": {},
        "stations": load_stations(),
    }
    for split in SPLITS:
        data["keys"][split], data["matrices"][split] = load_split(base_dir, split)
    data["windows"] = {split: window_labels(keys) for split, keys in data["keys"].items()}

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
        )
    )
    # Same K as the candidate so the comparison is like for like; Ward has no predict method,
    # which is why its model is not logged and its validation/December labels come from the means.
    records.append(
        train_and_log(
            f"clustering-candidate-ward-k{best_k}",
            AgglomerativeClustering(n_clusters=best_k, linkage="ward"),
            "ward",
            data,
            seed,
            model_params={"linkage": "ward"},
            figure_title=f"Perfiles horarios medios por clúster — Ward, K={best_k} (ventana de ajuste)",
            log_model=False,
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
