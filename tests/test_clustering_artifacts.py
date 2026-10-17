"""Unit tests for the P3 clustering script.

All tests use small synthetic matrices, so they do not need the real profile artifacts.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.cluster import KMeans

from tc5061_proyecto_equipo5.modeling.train_clustering import (
    choose_k,
    cluster_metrics,
    column_names,
    labels_for_splits,
    load_split,
    profile_summary,
    zone_from_name,
)


def test_column_names_strip_the_transformer_prefix():
    manifest = {
        "prepared": {
            "selected_columns": ["numeric__departures_hour_00", "numeric__asymmetry"],
        }
    }

    assert column_names(manifest) == ["departures_hour_00", "asymmetry"]


def test_choose_k_prefers_the_real_number_of_groups():
    rng = np.random.default_rng(0)
    blobs = np.vstack(
        [rng.normal(center, 0.1, size=(30, 2)) for center in ([0, 0], [5, 5], [10, 0])]
    )

    best, table = choose_k(blobs, blobs, (2, 3, 4), seed=42)

    assert best == 3
    assert set(table["k"]) == {2, 3, 4}
    assert set(table.columns) == {
        "k",
        "train_silhouette",
        "train_davies_bouldin",
        "train_inertia",
        "validation_silhouette",
    }
    scores = table.set_index("k")["validation_silhouette"]
    assert scores[3] > scores[2]


def test_zone_from_name_reads_the_catalog_prefix():
    assert zone_from_name("(GDL-001) C. Epigmenio") == "GDL"
    assert zone_from_name("(ZPN-102) C. Volcán") == "ZPN"
    assert zone_from_name("sin prefijo") == "OTRA"


def test_profile_summary_reports_peaks_shares_and_zone():
    centroids = pd.DataFrame(
        {
            "departures_hour_08": [0.2, 0.05],
            "departures_hour_18": [0.05, 0.3],
            "arrivals_hour_08": [0.3, 0.05],
            "arrivals_hour_18": [0.05, 0.25],
            "asymmetry": [0.1, -0.2],
        },
        index=[0, 1],
    )
    labels = np.array([0, 0, 1, 1])
    located = pd.DataFrame(
        {"station_id": [1, 2, 3, 4], "cluster": [0, 0, 1, 1], "zone": ["GDL", "GDL", "ZPN", "ZPN"]}
    )

    summary = profile_summary(centroids, labels, located)

    assert summary["cluster"].tolist() == [0, 1]
    assert summary["peak_departure_hour"].tolist() == [8, 18]
    assert summary["peak_arrival_hour"].tolist() == [8, 18]
    assert summary["top_zone"].tolist() == ["GDL", "ZPN"]
    assert summary["share_percent"].tolist() == [50.0, 50.0]


def test_cluster_metrics_returns_nan_for_a_single_group():
    metrics = cluster_metrics(np.zeros((5, 2)), np.zeros(5, dtype=int), "test_")

    assert np.isnan(metrics["test_silhouette"])
    assert np.isnan(metrics["test_davies_bouldin"])


def test_labels_for_splits_assigns_every_split_with_kmeans():
    train = np.vstack([np.zeros((10, 2)), np.full((10, 2), 10.0)])
    other = np.vstack([np.zeros((3, 2)), np.full((3, 2), 10.0)])
    matrices = {"train": train, "validation": other, "test": other}

    labels = labels_for_splits(KMeans(n_clusters=2, n_init=10, random_state=0), matrices, "kmeans")

    assert set(labels) == {"train", "validation", "test"}
    assert len(labels["validation"]) == 6
    # Both far-apart groups must receive different labels.
    assert len(set(labels["validation"].tolist())) == 2


def test_labels_for_splits_ward_falls_back_to_nearest_mean():
    train = np.vstack([np.zeros((10, 2)), np.full((10, 2), 10.0)])
    other = np.vstack([np.zeros((3, 2)), np.full((3, 2), 10.0)])
    matrices = {"train": train, "validation": other, "test": other}

    from sklearn.cluster import AgglomerativeClustering

    labels = labels_for_splits(
        AgglomerativeClustering(n_clusters=2, linkage="ward"), matrices, "ward"
    )

    assert len(labels["test"]) == 6
    assert len(set(labels["test"].tolist())) == 2


def test_load_split_rejects_a_matrix_with_the_wrong_row_count(tmp_path):
    folder = tmp_path / "ready" / "train"
    folder.mkdir(parents=True)
    pd.DataFrame({"station_id": [1, 2]}).to_parquet(folder / "keys.parquet", index=False)
    sparse.save_npz(folder / "X_00000000.npz", sparse.csr_matrix(np.zeros((3, 4))))

    with pytest.raises(ValueError, match="rows but"):
        load_split(tmp_path, "train")
