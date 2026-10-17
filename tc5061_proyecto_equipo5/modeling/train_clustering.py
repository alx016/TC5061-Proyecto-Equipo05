"""Train the clustering models for the MiBici station profile question.

P3 asks which groups of stations share usage patterns and where they are.  The features prepared by
``features.py`` are the 24 hourly departure proportions, the 24 hourly arrival proportions and the
global asymmetry of each station.  A fixed-K K-Means is the reference, the number of groups is then
chosen on the training window only, and an agglomerative Ward model is trained for comparison.

Clustering has no labels, so validation and test silhouettes are descriptive: the model is fitted on
train, the validation and test stations are assigned with the train model, and nothing is selected on
test.  Every run goes to the shared MLflow experiment with the helpers in modeling/common.py.
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
import typer

from tc5061_proyecto_equipo5.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, STUDY_YEAR
from tc5061_proyecto_equipo5.modeling.common import base_tags, data_version, setup_experiment

app = typer.Typer()

# Artifacts produced by: python -m tc5061_proyecto_equipo5.features main --problem p3 --prepare-ready
DEFAULT_BASE_DIR = PROCESSED_DATA_DIR / "fe_2024_v4" / "p3"
SPLITS = ("train", "validation", "test")
# The ML Canvas expects between 3 and 6 functional groups
K_VALUES = (3, 4, 5, 6)
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
    used as-is and never scaled again.  It is stored as CSR batches and loaded dense here because
    the profile table is small (a few hundred stations) and scikit-learn K-Means needs dense input.
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


def cluster_metrics(matrix: np.ndarray, labels: np.ndarray, prefix: str) -> dict[str, float]:
    """Return the silhouette and Davies-Bouldin scores of one split."""
    groups = len(set(labels.tolist()))
    if groups < 2 or groups >= matrix.shape[0]:
        return {f"{prefix}silhouette": float("nan"), f"{prefix}davies_bouldin": float("nan")}
    return {
        f"{prefix}silhouette": float(silhouette_score(matrix, labels)),
        f"{prefix}davies_bouldin": float(davies_bouldin_score(matrix, labels)),
    }


def choose_k(train_matrix: np.ndarray, k_values: tuple[int, ...], seed: int) -> tuple[int, dict]:
    """Pick the number of groups with the best train silhouette, never with validation or test."""
    scores = {}
    for k in k_values:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(train_matrix)
        scores[k] = float(silhouette_score(train_matrix, labels))
        logger.info(f"K={k}: train silhouette {scores[k]:.4f}")
    best = max(scores, key=scores.get)
    return best, scores


def labels_for_splits(model, matrices: dict[str, np.ndarray], kind: str) -> dict[str, np.ndarray]:
    """Assign every split with the model fitted on train.

    K-Means predicts directly.  Ward has no predict method, so the validation and test stations are
    assigned to the nearest train cluster mean; that approximation is logged as a parameter.
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
        distances = ((matrices[split][:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
        labels[split] = distances.argmin(axis=1)
    return labels


def centroid_frame(matrix: np.ndarray, labels: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Return the mean value of every column per cluster."""
    frame = pd.DataFrame(matrix, columns=names)
    return frame.groupby(labels).mean().sort_index()


def save_artifacts(
    folder: Path,
    names: list[str],
    matrices: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    keys_train: pd.DataFrame,
    stations: pd.DataFrame,
    title: str,
) -> None:
    """Write the cluster sizes, the hourly centroids and the station map."""
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
            columns = [name for name in centroids.columns if name.startswith(prefix)]
            hours = [int(name.rsplit("_", 1)[-1]) for name in columns]
            axis.plot(hours, row[columns].to_numpy(), marker="o", label=f"cluster {cluster}")
    for axis, label in zip(axes, ("Salidas", "Llegadas")):
        axis.set_xlabel("Hora del día (sin las horas de varianza cero en train)")
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
    figure, axis = plt.subplots(figsize=(7, 7))
    for cluster, group in located.groupby("cluster"):
        axis.scatter(
            group["longitude"], group["latitude"], s=18, alpha=0.8, label=f"cluster {cluster}"
        )
    axis.set_xlabel("Longitud")
    axis.set_ylabel("Latitud")
    axis.set_title("Estaciones de entrenamiento por clúster")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "cluster_map.png", bbox_inches="tight")
    plt.close(figure)


def load_stations() -> pd.DataFrame:
    """Read the newest station catalog for the map."""
    catalogs = sorted((EXTERNAL_DATA_DIR / "estaciones").glob("nomenclatura_*.csv"))
    if not catalogs:
        raise FileNotFoundError("No station catalog found; run the dataset download first")
    stations = pd.read_csv(catalogs[-1]).rename(columns={"id": "station_id"})
    return stations[["station_id", "latitude", "longitude"]]


def train_and_log(
    name: str,
    model,
    kind: str,
    data: dict,
    seed: int,
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
        "feature_set": "station_profiles",
        "feature_version": data["manifest"]["feature_version"],
        "feature_data_version": data["manifest"]["data_version"],
        "data_version": data_version(),
        "k_selection": "train silhouette only"
        if kind == "kmeans"
        else "inherited from the K-Means choice",
        "assignment": "predict" if kind == "kmeans" else "nearest train cluster mean",
        **data["model_params"],
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
                name,
            )
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

    best_k, scores = choose_k(data["matrices"]["train"], K_VALUES, seed)
    logger.info(f"Chosen K on train: {best_k} (scores {scores})")

    data["model_params"] = {"k_scores": json.dumps(scores)}
    records = [
        train_and_log(
            f"clustering-baseline-kmeans-k{REFERENCE_K}",
            KMeans(n_clusters=REFERENCE_K, n_init=10, random_state=seed),
            "kmeans",
            data,
            seed,
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
            log_model=True,
        )
    )
    data["model_params"] = {"linkage": "ward"}
    records.append(
        train_and_log(
            f"clustering-candidate-ward-k{best_k}",
            AgglomerativeClustering(n_clusters=best_k, linkage="ward"),
            "ward",
            data,
            seed,
            log_model=False,
        )
    )
    table = pd.DataFrame(records).set_index("run_name")
    logger.success("Comparison of the clustering runs:")
    print(table[[column for column in table.columns if "silhouette" in column]].to_string())


if __name__ == "__main__":
    app()
