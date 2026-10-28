import numpy as np
import pandas as pd
import pytest
from test_regression_training import make_prepared

from tc5061_proyecto_equipo5.modeling.train_classification import build_model
from tc5061_proyecto_equipo5.modeling.train_clustering import build_estimator
from tc5061_proyecto_equipo5.modeling.train_regression import candidates

SEED = 42


@pytest.mark.parametrize(
    "model_name",
    ["regression-candidate-lightgbm-tweedie", "regression-candidate-hist-gradient-boosting"],
)
def test_p1_model_with_the_same_seed_gives_identical_predictions(model_name):

    X, _, y = make_prepared(pd.date_range("2024-03-01", periods=24 * 30, freq="h"))

    builder, _ = candidates(SEED)[model_name]
    first = builder(X)

    builder, _ = candidates(SEED)[model_name]
    second = builder(X)

    first.fit(X, y)
    second.fit(X, y)

    np.testing.assert_array_equal(
        first.predict(X),
        second.predict(X),
    )


@pytest.mark.parametrize("model_name", ["random_forest", "lightgbm"])
def test_p2_model_with_the_same_seed_gives_identical_probabilities(model_name):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 5))
    score = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=len(X))
    y = np.zeros(len(X), dtype=int)
    y[score < -1] = 1
    y[score > 1] = 2

    first, _ = build_model(
        name=model_name,
        seed=SEED,
        hyperparams={},
    )
    second, _ = build_model(
        name=model_name,
        seed=SEED,
        hyperparams={},
    )

    first.fit(X, y)
    second.fit(X, y)

    # n_jobs=-1 sums the trees in parallel, so the order of the floating-point additions
    # changes between runs: identical trees still differ around 1e-16
    np.testing.assert_allclose(
        first.predict_proba(X),
        second.predict_proba(X),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("model_name", ["kmeans", "kmeans_scaled", "minibatch_kmeans", "gmm"])
def test_p3_model_with_the_same_seed_gives_identical_labels(model_name):
    rng = np.random.default_rng(0)
    blobs = np.vstack(
        [rng.normal(center, 1.0, size=(30, 2)) for center in ([0, 0], [5, 5], [10, 0])]
    )

    k = 3

    first, _ = build_estimator(
        model=model_name,
        k=k,
        seed=SEED,
        k_selection="test",
    )
    second, _ = build_estimator(
        model=model_name,
        k=k,
        seed=SEED,
        k_selection="test",
    )

    first.fit(blobs)
    second.fit(blobs)

    np.testing.assert_array_equal(
        first.predict(blobs),
        second.predict(blobs),
    )


def test_p2_random_forest_with_a_different_seed_gives_different_probabilities():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 5))
    score = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=len(X))
    y = np.zeros(len(X), dtype=int)
    y[score < -1] = 1
    y[score > 1] = 2

    model_name = "random_forest"

    first, _ = build_model(
        name=model_name,
        seed=42,
        hyperparams={},
    )
    second, _ = build_model(
        name=model_name,
        seed=43,
        hyperparams={},
    )

    first.fit(X, y)
    second.fit(X, y)

    probabilities_42 = first.predict_proba(X)
    probabilities_43 = second.predict_proba(X)

    assert np.abs(probabilities_42 - probabilities_43).max() > 1e-6
