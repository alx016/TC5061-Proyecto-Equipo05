"""Unit tests for the P3 clustering script.

All tests use small synthetic matrices, so they do not need the real profile artifacts.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import euclidean_distances

from tc5061_proyecto_equipo5.modeling.train_clustering import (
    ASSIGNMENTS,
    MODELS,
    WardClusterer,
    build_estimator,
    choose_k,
    cluster_metrics,
    column_names,
    labels_for_splits,
    load_split,
    profile_summary,
    save_silhouette_plot,
    temporal_ari,
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

    labels = labels_for_splits(KMeans(n_clusters=2, n_init=10, random_state=0), matrices)

    assert set(labels) == {"train", "validation", "test"}
    assert len(labels["validation"]) == 6
    # Both far-apart groups must receive different labels.
    assert len(set(labels["validation"].tolist())) == 2


def test_labels_for_splits_assigns_every_split_with_ward():
    train = np.vstack([np.zeros((10, 2)), np.full((10, 2), 10.0)])
    other = np.vstack([np.zeros((3, 2)), np.full((3, 2), 10.0)])
    matrices = {"train": train, "validation": other, "test": other}

    labels = labels_for_splits(WardClusterer(n_clusters=2), matrices)

    assert len(labels["test"]) == 6
    assert len(set(labels["test"].tolist())) == 2


def test_load_split_rejects_a_matrix_with_the_wrong_row_count(tmp_path):
    folder = tmp_path / "ready" / "train"
    folder.mkdir(parents=True)
    pd.DataFrame({"station_id": [1, 2]}).to_parquet(folder / "keys.parquet", index=False)
    sparse.save_npz(folder / "X_00000000.npz", sparse.csr_matrix(np.zeros((3, 4))))

    with pytest.raises(ValueError, match="rows but"):
        load_split(tmp_path, "train")


def three_blobs() -> np.ndarray:
    """Return 3 tight groups of 30 points around (0, 0), (5, 5) and (10, 0)."""
    rng = np.random.default_rng(0)
    return np.vstack(
        [rng.normal(center, 0.1, size=(30, 2)) for center in ([0, 0], [5, 5], [10, 0])]
    )


def test_ward_clusterer_keeps_the_labels_of_ward():
    X = three_blobs()

    model = WardClusterer(n_clusters=3).fit(X)

    expected = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
    np.testing.assert_array_equal(model.labels_, expected)
    assert model.n_features_in_ == 2


def test_ward_clusterer_centroids_are_the_train_cluster_means():
    X = three_blobs()

    model = WardClusterer(n_clusters=3).fit(X)

    assert model.cluster_centers_.shape == (3, 2)
    for group in range(3):
        np.testing.assert_allclose(
            model.cluster_centers_[group], X[model.labels_ == group].mean(axis=0)
        )


def test_ward_clusterer_predicts_the_label_of_the_nearest_blob():
    X = three_blobs()
    model = WardClusterer(n_clusters=3).fit(X)
    centers = np.array([[0, 0], [5, 5], [10, 0]])

    labels = model.predict(centers)

    # Ward does not promise which number each blob gets, so compare with the train labels
    np.testing.assert_array_equal(labels, model.labels_[[0, 30, 60]])


def test_ward_clusterer_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        WardClusterer(n_clusters=3).predict(three_blobs())


def test_ward_clusterer_supports_clone_and_get_params():
    model = clone(WardClusterer(n_clusters=5))

    assert model.get_params() == {"n_clusters": 5}


def test_ward_clusterer_predict_matches_the_nearest_mean_rule():
    X = three_blobs()
    model = WardClusterer(n_clusters=3).fit(X)
    new = np.random.default_rng(1).normal(5, 4, size=(50, 2))

    labels = model.predict(new)

    # The rule labels_for_splits used for Ward before the wrapper existed
    ward_labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
    means = np.vstack([X[ward_labels == group].mean(axis=0) for group in range(3)])
    np.testing.assert_array_equal(labels, euclidean_distances(new, means).argmin(axis=1))


def test_every_model_has_an_assignment_rule():
    assert set(ASSIGNMENTS) == set(MODELS)


@pytest.mark.parametrize("model", MODELS)
def test_build_estimator_logs_where_k_came_from(model):
    _, params = build_estimator(model, k=3, seed=0, k_selection="given on CLI")

    assert params["k_selection"] == "given on CLI"


def stations(ids) -> pd.DataFrame:
    return pd.DataFrame({"station_id": ids})


def test_save_silhouette_plot_writes_the_figure(tmp_path):
    X = three_blobs()
    labels = np.repeat([0, 1, 2], 30)

    save_silhouette_plot(tmp_path, X, labels, "test")

    assert (tmp_path / "silhouette_validation.png").is_file()


def test_save_silhouette_plot_skips_a_single_group(tmp_path):
    save_silhouette_plot(tmp_path, three_blobs(), np.zeros(90, dtype=int), "test")

    assert not (tmp_path / "silhouette_validation.png").exists()


def test_temporal_ari_ignores_renamed_labels():
    keys = stations([1, 2, 3, 4])

    ari = temporal_ari(keys, np.array([0, 0, 1, 1]), keys, np.array([1, 1, 0, 0]))

    assert ari == pytest.approx(1.0)


def test_temporal_ari_matches_rows_by_station_not_by_position():
    labels_a = np.array([0, 0, 1, 1])
    # Same assignment as labels_a, with the rows in another order
    keys_b, labels_b = stations([4, 1, 3, 2]), np.array([1, 0, 1, 0])

    ari = temporal_ari(stations([1, 2, 3, 4]), labels_a, keys_b, labels_b)

    assert ari == pytest.approx(1.0)


def test_temporal_ari_ignores_stations_missing_from_one_window():
    keys_a, labels_a = stations([1, 2, 3, 4]), np.array([0, 0, 1, 1])
    # Stations 5 and 6 only exist in the later window
    keys_b, labels_b = stations([1, 2, 3, 4, 5, 6]), np.array([0, 0, 1, 1, 0, 1])

    ari = temporal_ari(keys_a, labels_a, keys_b, labels_b)

    assert ari == pytest.approx(1.0)


def test_temporal_ari_is_nan_with_fewer_than_two_shared_stations():
    ari = temporal_ari(stations([1, 2]), np.array([0, 1]), stations([2, 3]), np.array([0, 1]))

    assert np.isnan(ari)
