"""Tests of the FinOps formulas against small Locust and docker stats fixtures."""

import csv
import json
from pathlib import Path

import pytest
from scripts import finops_calc

LOCUST_COLUMNS = [
    "Type",
    "Name",
    "Request Count",
    "Failure Count",
    "Median Response Time",
    "Requests/s",
]
STATS_COLUMNS = [
    "timestamp",
    "scenario",
    "container",
    "cpu_pct",
    "mem_usage_mib",
    "mem_limit_mib",
    "mem_pct",
    "net_rx_mb",
    "net_tx_mb",
    "pids",
]


def write_locust(load_dir: Path, scenario: str, rows: list[tuple[str, int, int, float, float]]):
    """Write locust_<slug>_stats.csv; each row is (name, requests, failures, median ms, req/s)."""
    path = load_dir / f"locust_{scenario}_stats.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LOCUST_COLUMNS)
        for name, requests, failures, median, rps in rows:
            kind = "" if name == "Aggregated" else "POST"
            writer.writerow([kind, name, requests, failures, median, rps])


def write_stats(load_dir: Path, scenario: str, cpu: dict[str, float], mem: dict[str, float]):
    """Write docker_stats_<slug>.csv with two identical samples per container."""
    path = load_dir / f"docker_stats_{scenario}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(STATS_COLUMNS)
        for second in range(2):
            for name in ("api", "mlflow"):
                writer.writerow(
                    [
                        f"2026-11-20T00:00:0{second}Z",
                        scenario,
                        name,
                        cpu[name],
                        mem[name],
                        1024,
                        0,
                        0,
                        0,
                        10,
                    ]
                )


@pytest.fixture
def load_dir(tmp_path: Path) -> Path:
    """Six scenarios: 50u is the sustainable peak, 120u fails too often."""
    steps = {
        # scenario: (predict batch=1 req/s, batch=20 req/s, errors, api cpu)
        "nominal_10u": (5.0, 3.0, 0, 30.0),
        "high_50u": (10.0, 6.0, 0, 70.0),
        "stress_120u": (12.0, 7.0, 50, 99.0),
    }
    for scenario, (rps_1, rps_20, failures, api_cpu) in steps.items():
        write_locust(
            tmp_path,
            scenario,
            [
                ("/predict?batch=1", 100, 0, 20.0, rps_1),
                ("/predict?batch=20", 60, 0, 40.0, rps_20),
                ("Aggregated", 1000, failures, 25.0, rps_1 + rps_20),
            ],
        )
        write_stats(
            tmp_path, scenario, {"api": api_cpu, "mlflow": 10.0}, {"api": 300.0, "mlflow": 300.0}
        )

    write_locust(
        tmp_path,
        "batch_1x365",
        [("/predict?batch=365", 50, 0, 500.0, 2.0), ("Aggregated", 50, 0, 500.0, 2.0)],
    )
    write_stats(
        tmp_path, "batch_1x365", {"api": 90.0, "mlflow": 10.0}, {"api": 400.0, "mlflow": 300.0}
    )
    write_locust(
        tmp_path,
        "batch_365x1",
        [("/predict?batch=1", 900, 0, 10.0, 73.0), ("Aggregated", 900, 0, 10.0, 73.0)],
    )
    write_stats(
        tmp_path, "batch_365x1", {"api": 90.0, "mlflow": 10.0}, {"api": 400.0, "mlflow": 300.0}
    )
    return tmp_path


def test_predictions_per_second_weight_each_request_by_its_batch(load_dir: Path):
    throughput = finops_calc.read_throughput(load_dir / "locust_high_50u_stats.csv", "high_50u")
    assert throughput.req_per_s == pytest.approx(16.0)
    assert throughput.pred_per_s == pytest.approx(10.0 * 1 + 6.0 * 20)


def test_error_rate_comes_from_the_aggregated_row(load_dir: Path):
    throughput = finops_calc.read_throughput(
        load_dir / "locust_stress_120u_stats.csv", "stress_120u"
    )
    assert throughput.error_pct == pytest.approx(5.0)


def test_peak_skips_steps_with_errors_or_a_saturated_api(load_dir: Path):
    steps = [finops_calc.load_scenario(load_dir, name) for name in finops_calc.LOAD_STEPS]
    throughput, _ = finops_calc.pick_peak(steps)
    assert throughput.scenario == "high_50u"


def test_peak_fails_when_no_step_is_sustainable(load_dir: Path):
    steps = [finops_calc.load_scenario(load_dir, "stress_120u")]
    with pytest.raises(ValueError, match="No load step"):
        finops_calc.pick_peak(steps)


def test_cpu_credits_only_charge_above_the_t3_baseline():
    idle = finops_calc.Resources("idle", {"api": 20.0, "mlflow": 10.0}, {"api": 1, "mlflow": 1})
    busy = finops_calc.Resources("busy", {"api": 90.0, "mlflow": 10.0}, {"api": 1, "mlflow": 1})
    assert finops_calc.cpu_credit_usd_per_hour(idle) == 0.0
    # 1.0 vCPU used - 0.4 vCPU baseline = 0.6 vCPU-hour at $0.05
    assert finops_calc.cpu_credit_usd_per_hour(busy) == pytest.approx(0.03)


def test_cost_per_unit_divides_the_hourly_price_by_the_units_per_hour():
    assert finops_calc.cost_per_unit(0.0208, 1.0) == pytest.approx(0.0208 / 3600)
    with pytest.raises(ValueError):
        finops_calc.cost_per_unit(0.0208, 0.0)


def test_attribution_averages_cpu_and_memory_shares():
    resources = finops_calc.Resources(
        "peak", {"api": 75.0, "mlflow": 25.0}, {"api": 500.0, "mlflow": 500.0}
    )
    shares = finops_calc.attribution(resources)
    assert shares["api"] == pytest.approx(62.5)
    assert shares["api"] + shares["mlflow"] == pytest.approx(100.0)


def test_real_demand_uses_the_same_hours_per_month_as_the_price(load_dir: Path):
    summary = finops_calc.build_summary(load_dir, 20.0, None, None)
    baseline = summary["baseline"]
    assert baseline["real_pred_per_month"] == 365 * 730
    expected_monthly = 0.0208 * 730 + 20 * 0.08
    assert baseline["monthly_usd"] == pytest.approx(expected_monthly)
    assert baseline["cost_per_1m_pred_real_24_7_usd"] == pytest.approx(
        expected_monthly / (365 * 730) * 1e6
    )


def test_summary_keeps_the_contract_keys(load_dir: Path):
    summary = finops_calc.build_summary(load_dir, 20.0, None, None)
    contract = {
        "instance_type",
        "hourly_usd",
        "storage_monthly_usd",
        "monthly_usd",
        "peak_req_per_s",
        "peak_pred_per_s",
        "cost_per_req_peak_usd",
        "cost_per_pred_peak_usd",
        "cost_per_1m_pred_peak_usd",
        "cost_per_1m_pred_real_24_7_usd",
        "attribution_pct",
    }
    assert contract <= summary["baseline"].keys()
    for opt in summary["optimizations"]:
        assert {"id", "name", "cost_per_1m_pred_usd", "monthly_usd", "savings_pct"} <= opt.keys()


def test_batching_is_cheaper_than_one_station_per_request(load_dir: Path):
    summary = finops_calc.build_summary(load_dir, 20.0, None, None)
    opt1 = next(opt for opt in summary["optimizations"] if opt["id"] == "opt1")
    # 2 req/s x 365 = 730 pred/s against 73 pred/s with the same resources
    assert opt1["detail"]["throughput_ratio"] == pytest.approx(10.0)
    assert opt1["detail"]["savings_vs_unit_pct"] == pytest.approx(90.0)


def test_optional_optimizations_need_their_parameters(load_dir: Path):
    without = finops_calc.build_summary(load_dir, 20.0, None, None)
    assert {opt["id"] for opt in without["optimizations"]} == {"opt1", "opt2"}

    with_all = finops_calc.build_summary(load_dir, 20.0, 2.0, 0.00002)
    by_id = {opt["id"]: opt for opt in with_all["optimizations"]}
    assert by_id["opt3"]["savings_pct"] == pytest.approx(50.0)
    # 730 hourly runs of a 0.5 s batch
    assert by_id["opt4"]["detail"]["active_seconds_per_month"] == pytest.approx(365.0)
    assert by_id["opt4"]["monthly_usd"] == pytest.approx(365.0 * 0.00002)


def test_main_writes_the_json_and_the_annex(load_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "sys.argv", ["finops_calc.py", "--load-dir", str(load_dir), "--opt3-speedup", "2"]
    )
    finops_calc.main()
    summary = json.loads((load_dir / "finops_summary.json").read_text())
    annex = (load_dir / "anexo_c.md").read_text()
    assert summary["baseline"]["peak_scenario"] == "high_50u"
    assert "### C.3 Cálculo" in annex
    assert "Pendientes por falta de parámetro: opt4." in annex
