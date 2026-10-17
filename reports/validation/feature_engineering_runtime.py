"""Deterministic synthetic scale and estimator-persistence checks (not real MiBici data).

Run as a module from the repository root, once per problem to isolate peak RSS:
python -m reports.validation.feature_engineering_runtime p1
"""

import json
import pickle
import resource
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from tc5061_proyecto_equipo5.features import (
    build_demand_features,
    build_risk_features,
    build_station_profiles,
    make_preprocessor,
    training_mask,
)


def run(problem):
    start = perf_counter()
    ids = np.arange(1, 361)
    times = pd.date_range("2024-01-01", "2025-01-01", freq="h", inclusive="left")
    panel = pd.MultiIndex.from_product([ids, times], names=["station_id", "target_time"]).to_frame(
        index=False
    )
    # Known deterministic counts, not a fitted distribution or a real-data sample.
    panel["departures"] = (np.arange(len(panel)) % 11).astype("int32")
    panel["arrivals"] = (np.arange(len(panel)) % 7).astype("int32")
    generated = perf_counter()
    if problem == "p1":
        result = build_demand_features(panel)
    elif problem == "p2":
        stations = pd.DataFrame({"station_id": ids, "capacity": 20})
        result = build_risk_features(panel, stations, catalog_known_at="2023-01-01")
    elif problem == "p3":
        result = build_station_profiles(panel)
    else:
        raise ValueError("Expected p1, p2 or p3")
    built = perf_counter()
    assert len(result.X) == len(result.keys)
    assert result.y is None or len(result.y) == len(result.X)
    if problem == "p3":
        assert result.X.shape == (360, 49)
        assert np.isfinite(result.X.to_numpy()).all()
        assert result.X.asymmetry.between(-1, 1).all()
        for direction in ("departures", "arrivals"):
            assert np.allclose(result.X.filter(like=f"{direction}_hour_").sum(axis=1), 1)
    else:
        assert len(result.X) == 360 * 8784
        assert result.keys["split"].value_counts().to_dict() == {
            "train": 360 * 6576,
            "validation": 360 * 1464,
            "test": 360 * 744,
        }
        assert not result.keys["split"].isna().any()
    persistence = "not exercised for this problem"
    if problem == "p1":
        # Small three-station subset: checks a real estimator, not a preprocessing stub.
        subset = panel.loc[panel.station_id.le(3)].copy()
        small = build_demand_features(subset)
        first_val_issue = small.keys.loc[small.keys["split"].eq("validation"), "issue_time"].min()
        train = training_mask(small.keys, fit_as_of=first_val_issue)
        val = small.keys["split"].eq("validation")
        model = make_pipeline(make_preprocessor(small.X), Ridge(solver="lsqr"))
        model.fit(small.X.loc[train], small.y.loc[train])
        predictions = model.predict(small.X.loc[val])
        restored = pickle.loads(pickle.dumps(model))
        assert np.array_equal(predictions, restored.predict(small.X.loc[val]))
        changed = subset.copy()
        cut = pd.Timestamp("2024-10-02 10:00")
        changed.loc[changed.target_time.ge(cut - pd.Timedelta(hours=1)), "departures"] = 999
        future = build_demand_features(changed)
        rows = small.keys.target_time.le(cut)
        assert_frame_equal(small.X.loc[rows], future.X.loc[rows])
        persistence = (
            "Ridge fit on purged train; pickle prediction parity; future invariance passed"
        )
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    return {
        "problem": problem,
        "input": "synthetic, 360 stations, 8784 hours (leap-year 2024)",
        "input_rows": len(panel),
        "output_shape": list(result.X.shape),
        "panel_generation_seconds": round(generated - start, 3),
        "recipe_seconds": round(built - generated, 3),
        "process_peak_rss_mib": round(rss / 1024**2, 1),
        "persistence": persistence,
        "status": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), indent=2))
