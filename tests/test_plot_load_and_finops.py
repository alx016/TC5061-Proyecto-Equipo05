"""The Entrega 6 figures render from synthetic outputs with the shape of the real ones."""

from pathlib import Path

import pandas as pd
import pytest
from scripts import plot_load_and_finops as plots
from tests.fixtures.load_fixtures import write_load_fixtures

FIGURES = [
    "01_rps_y_usuarios_vs_tiempo.png",
    "02_latencia_percentiles.png",
    "03_recursos_contenedores.png",
    "04_finops_comparativa_optimizaciones.png",
]


@pytest.fixture
def load_dir(tmp_path: Path) -> Path:
    out = tmp_path / "load"
    write_load_fixtures(out)
    return out


def run_main(monkeypatch, load_dir: Path, output_dir: Path) -> None:
    argv = ["plot", "--load-dir", str(load_dir), "--output-dir", str(output_dir)]
    monkeypatch.setattr("sys.argv", argv)
    plots.main()


def test_main_writes_the_four_figures(monkeypatch, load_dir, tmp_path):
    output_dir = tmp_path / "figs"
    run_main(monkeypatch, load_dir, output_dir)

    assert sorted(p.name for p in output_dir.glob("*.png")) == FIGURES


def test_history_with_only_aggregated_rows(monkeypatch, load_dir, tmp_path):
    # collect_load_metrics.py runs Locust without --csv-full-history.
    for path in load_dir.glob("locust_*_stats_history.csv"):
        history = pd.read_csv(path)
        history[history["Name"] == "Aggregated"].to_csv(path, index=False)

    output_dir = tmp_path / "figs"
    run_main(monkeypatch, load_dir, output_dir)

    assert (output_dir / FIGURES[0]).exists()


def test_scenarios_follow_the_run_order():
    found = ["batch_365x1", "extra", "baseline", "stress_120u", "nominal_10u"]

    assert plots.ordered_scenarios(found) == [
        "baseline",
        "nominal_10u",
        "stress_120u",
        "batch_365x1",
        "extra",
    ]


def test_each_optimization_is_compared_with_its_own_baseline(load_dir, tmp_path, monkeypatch):
    finops = plots.load_finops(load_dir)
    drawn = {}

    def fake_barh(self, labels, costs, **kwargs):
        drawn[labels[0]] = list(zip(labels, costs))
        return original_barh(self, labels, costs, **kwargs)

    original_barh = plots.plt.Axes.barh
    monkeypatch.setattr(plots.plt.Axes, "barh", fake_barh)
    plots.plot_finops(finops, tmp_path)

    peak = drawn["Línea base (pico)"]
    real = drawn["Línea base (24/7, demanda real)"]
    assert peak[0][1] == finops["baseline"]["cost_per_1m_pred_peak_usd"]
    assert real[0][1] == finops["baseline"]["cost_per_1m_pred_real_24_7_usd"]
    by_reference = {"peak": peak[1:], "real_24_7": real[1:]}
    for reference, bars in by_reference.items():
        expected = [o["name"] for o in finops["optimizations"] if o["reference"] == reference]
        assert [name for name, _ in bars] == expected


def test_usd_keeps_small_costs_readable():
    assert plots.usd(0.00276) == "$0.0028"
    assert plots.usd(63.871) == "$63.87"
