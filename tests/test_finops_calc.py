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
    "50%",
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


def write_locust(load_dir: Path, scenario: str, rows: list[tuple]) -> None:
    """Write locust_<slug>_stats.csv: (name, requests, failures, median, req/s[, p50])."""
    path = load_dir / f"locust_{scenario}_stats.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LOCUST_COLUMNS)
        for name, requests, failures, median, rps, *p50 in rows:
            kind = "" if name == "Aggregated" else "POST"
            writer.writerow(
                [kind, name, requests, failures, median, rps, p50[0] if p50 else median]
            )


def write_stats(
    load_dir: Path, scenario: str, cpu: dict[str, float], mem: dict[str, float]
) -> None:
    """Write docker_stats_<slug>.csv with two identical samples per container in ``cpu``."""
    path = load_dir / f"docker_stats_{scenario}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(STATS_COLUMNS)
        for second in range(2):
            for name, cpu_pct in cpu.items():
                writer.writerow(
                    [
                        f"2026-11-20T00:00:0{second}Z",
                        scenario,
                        name,
                        cpu_pct,
                        mem[name],
                        1024,
                        0,
                        0,
                        0,
                        10,
                    ]
                )


def write_benchmark(load_dir: Path, rows: list[tuple[str, int, float]]) -> Path:
    """Write opt3_benchmark.json; each row is (path, batch size, speedup)."""
    path = load_dir / "opt3_benchmark.json"
    results = [
        {"path": name, "batch_size": batch, "speedup_vs_pyfunc": speedup}
        for name, batch, speedup in rows
    ]
    path.write_text(json.dumps({"results": results}))
    return path


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

    # 0.5 s per batch of 365, the assumption of section 4 of the report
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


def summary_of(load_dir: Path, **params) -> dict:
    """build_summary with default parameters unless overridden."""
    return finops_calc.build_summary(load_dir, finops_calc.Params(**params))


def opt(summary: dict, opt_id: str) -> dict:
    """One optimization of a summary by id."""
    return next(item for item in summary["optimizations"] if item["id"] == opt_id)


# Reading the load outputs


def test_predictions_per_second_weight_each_request_by_its_batch(load_dir: Path):
    throughput = finops_calc.read_throughput(load_dir / "locust_high_50u_stats.csv", "high_50u")
    assert throughput.req_per_s == pytest.approx(16.0)
    assert throughput.pred_per_s == pytest.approx(10.0 * 1 + 6.0 * 20)


def test_error_rate_comes_from_the_aggregated_row(load_dir: Path):
    throughput = finops_calc.read_throughput(
        load_dir / "locust_stress_120u_stats.csv", "stress_120u"
    )
    assert throughput.error_pct == pytest.approx(5.0)


@pytest.mark.parametrize("missing", ["N/A", "--", ""])
def test_missing_median_falls_back_to_the_p50_column(tmp_path: Path, missing: str):
    write_locust(
        tmp_path,
        "batch_1x365",
        [("/predict?batch=365", 5, 0, missing, 1.0, "480"), ("Aggregated", 5, 0, 1, 1.0)],
    )
    path = tmp_path / "locust_batch_1x365_stats.csv"
    throughput = finops_calc.read_throughput(path, "batch_1x365")
    assert throughput.median_ms_by_batch[365] == pytest.approx(480.0)


def test_a_median_missing_everywhere_is_skipped(tmp_path: Path):
    write_locust(
        tmp_path,
        "batch_1x365",
        [("/predict?batch=365", 5, 5, "N/A", 0.0, "N/A"), ("Aggregated", 5, 5, "N/A", 0.0)],
    )
    path = tmp_path / "locust_batch_1x365_stats.csv"
    throughput = finops_calc.read_throughput(path, "batch_1x365")
    assert 365 not in throughput.median_ms_by_batch


def test_resources_without_mlflow_count_it_as_zero(tmp_path: Path):
    write_stats(tmp_path, "high_50u", {"api": 60.0}, {"api": 300.0})
    resources = finops_calc.read_resources(tmp_path / "docker_stats_high_50u.csv", "high_50u")
    assert resources.present == ("api",)
    assert resources.cpu_pct["mlflow"] == 0.0
    assert finops_calc.attribution(resources) == {"api": 100.0, "mlflow": 0.0}


def test_resources_without_api_fail(tmp_path: Path):
    write_stats(tmp_path, "high_50u", {"mlflow": 5.0}, {"mlflow": 300.0})
    with pytest.raises(ValueError, match="api container"):
        finops_calc.read_resources(tmp_path / "docker_stats_high_50u.csv", "high_50u")


# Peak, credits and attribution


def test_peak_skips_steps_with_errors_or_a_saturated_api(load_dir: Path):
    steps = [finops_calc.load_scenario(load_dir, name) for name in finops_calc.LOAD_STEPS]
    throughput, _ = finops_calc.pick_peak(steps)
    assert throughput.scenario == "high_50u"


def test_peak_fails_when_no_step_is_sustainable(load_dir: Path):
    steps = [finops_calc.load_scenario(load_dir, "stress_120u")]
    with pytest.raises(ValueError, match="No load step"):
        finops_calc.pick_peak(steps)


def test_cpu_credits_only_charge_above_the_t3_baseline():
    both = ("api", "mlflow")
    idle = finops_calc.Resources(
        "idle", {"api": 20.0, "mlflow": 10.0}, {"api": 1, "mlflow": 1}, both
    )
    busy = finops_calc.Resources(
        "busy", {"api": 90.0, "mlflow": 10.0}, {"api": 1, "mlflow": 1}, both
    )
    assert finops_calc.cpu_credit_usd_per_hour(idle) == 0.0
    # 1.0 vCPU used - 0.4 vCPU baseline = 0.6 vCPU-hour at $0.05
    assert finops_calc.cpu_credit_usd_per_hour(busy) == pytest.approx(0.03)


def test_cost_per_unit_divides_the_hourly_price_by_the_units_per_hour():
    assert finops_calc.cost_per_unit(0.0208, 1.0) == pytest.approx(0.0208 / 3600)
    with pytest.raises(ValueError):
        finops_calc.cost_per_unit(0.0208, 0.0)


def test_attribution_averages_cpu_and_memory_shares():
    resources = finops_calc.Resources(
        "peak", {"api": 75.0, "mlflow": 25.0}, {"api": 500.0, "mlflow": 500.0}, ("api", "mlflow")
    )
    shares = finops_calc.attribution(resources)
    assert shares["api"] == pytest.approx(62.5)
    assert shares["api"] + shares["mlflow"] == pytest.approx(100.0)


# Monthly figures and contract


def test_defaults_match_section_4_of_the_report(load_dir: Path):
    baseline = summary_of(load_dir)["baseline"]
    # t3.small 730 h + 10 GB gp3 = $15.98 per month
    assert baseline["monthly_usd"] == pytest.approx(0.0208 * 730 + 10 * 0.08)
    assert baseline["monthly_usd"] == pytest.approx(15.98, abs=0.01)


@pytest.mark.parametrize("hours", [720.0, 730.0])
def test_price_and_demand_use_the_same_hours_per_month(load_dir: Path, hours: float):
    baseline = summary_of(load_dir, storage_gb=20.0, hours_per_month=hours)["baseline"]
    expected_monthly = 0.0208 * hours + 20 * 0.08
    assert baseline["real_pred_per_month"] == pytest.approx(365 * hours)
    assert baseline["monthly_usd"] == pytest.approx(expected_monthly)
    assert baseline["cost_per_1m_pred_real_24_7_usd"] == pytest.approx(
        expected_monthly / (365 * hours) * 1e6
    )


def test_summary_keeps_the_contract_keys(load_dir: Path):
    summary = summary_of(load_dir)
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
    for item in summary["optimizations"]:
        assert {
            "id",
            "name",
            "reference",
            "cost_per_1m_pred_usd",
            "monthly_usd",
            "savings_pct",
        } <= item.keys()
        assert item["reference"] in {"peak", "real_24_7"}


# Optimizations


def test_batching_is_cheaper_than_one_station_per_request(load_dir: Path):
    opt1 = opt(summary_of(load_dir), "opt1")
    # 2 req/s x 365 = 730 pred/s against 73 pred/s with the same resources
    assert opt1["detail"]["throughput_ratio"] == pytest.approx(10.0)
    assert opt1["detail"]["savings_vs_unit_pct"] == pytest.approx(90.0)


def test_opt3_is_pending_without_a_speedup(load_dir: Path):
    summary = summary_of(load_dir)
    assert "opt3" not in {item["id"] for item in summary["optimizations"]}
    assert "opt3" in summary["pending"]


def test_opt3_divides_the_peak_cost_by_the_speedup(load_dir: Path):
    assert opt(summary_of(load_dir, opt3_speedup=2.0), "opt3")["savings_pct"] == pytest.approx(
        50.0
    )


def test_opt4_reproduces_the_lambda_bill_of_section_4(load_dir: Path):
    opt4 = opt(summary_of(load_dir), "opt4")
    batch = opt4["detail"]["batch"]
    # 730 hourly runs of 0.5 s = 365 s of Lambda at 1 GB, plus 730 invocations
    assert batch["active_seconds_per_month"] == pytest.approx(365.0)
    assert batch["compute_usd"] == pytest.approx(365 * 0.0000166667)
    assert batch["requests_usd"] == pytest.approx(730 * 0.2e-6)
    assert opt4["monthly_usd"] == pytest.approx(0.0062, abs=0.0001)
    assert opt4["reference"] == "real_24_7"


def test_opt4_also_prices_one_station_per_request(load_dir: Path):
    unit = opt(summary_of(load_dir), "opt4")["detail"]["unit"]
    # 365 x 730 calls of 10 ms each
    assert unit["requests_per_month"] == pytest.approx(365 * 730)
    assert unit["active_seconds_per_month"] == pytest.approx(365 * 730 * 0.010)


def test_opt4_is_pending_without_a_valid_batch_latency(load_dir: Path):
    write_locust(
        load_dir,
        "batch_1x365",
        [("/predict?batch=365", 50, 0, 0.0, 2.0, 0.0), ("Aggregated", 50, 0, 0.0, 2.0)],
    )
    summary = summary_of(load_dir)
    assert "opt4" not in {item["id"] for item in summary["optimizations"]}
    assert "opt4" in summary["pending"]


# Inputs


@pytest.mark.parametrize(
    "params",
    [
        {"storage_gb": -1.0},
        {"hours_per_month": 0.0},
        {"opt3_speedup": 0.0},
        {"opt3_speedup": -2.0},
        {"serverless_usd_per_second": 0.0},
        {"serverless_usd_per_request": -0.1},
    ],
)
def test_invalid_parameters_are_rejected(params: dict):
    with pytest.raises(ValueError):
        finops_calc.Params(**params)


@pytest.mark.parametrize(
    "flag", [["--opt3-speedup", "0"], ["--storage-gb", "-5"], ["--serverless-usd-per-second", "0"]]
)
def test_the_cli_rejects_invalid_values(
    load_dir: Path, monkeypatch: pytest.MonkeyPatch, flag: list[str]
):
    monkeypatch.setattr("sys.argv", ["finops_calc.py", "--load-dir", str(load_dir), *flag])
    with pytest.raises(SystemExit):
        finops_calc.main()


def test_opt3_speedup_keeps_the_worst_batch_of_the_best_path(tmp_path: Path):
    path = write_benchmark(
        tmp_path,
        [
            ("booster_numpy", 1, 4.0),
            ("booster_numpy", 365, 1.5),
            ("pyfunc_dataframe", 1, 1.0),
            ("booster_numpy_1thread", 1, 2.0),
            ("booster_numpy_1thread", 365, 1.8),
        ],
    )
    # The pyfunc reference is skipped by name, wherever it appears in the file
    assert finops_calc.read_opt3_speedup(path) == pytest.approx(1.8)


def test_opt3_speedup_ignores_non_positive_values(tmp_path: Path):
    path = write_benchmark(tmp_path, [("pyfunc_dataframe", 1, 1.0), ("booster_numpy", 1, 0.0)])
    assert finops_calc.read_opt3_speedup(path) is None


def test_opt3_speedup_is_none_without_a_benchmark(tmp_path: Path):
    assert finops_calc.read_opt3_speedup(tmp_path / "opt3_benchmark.json") is None


# Command line and annex


def test_main_reads_the_opt3_speedup_from_the_benchmark(
    load_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    write_benchmark(load_dir, [("pyfunc_dataframe", 1, 1.0), ("booster_numpy", 1, 4.0)])
    monkeypatch.setattr("sys.argv", ["finops_calc.py", "--load-dir", str(load_dir)])
    finops_calc.main()
    summary = json.loads((load_dir / "finops_summary.json").read_text())
    assert opt(summary, "opt3")["savings_pct"] == pytest.approx(75.0)


def test_an_explicit_opt3_speedup_wins_over_the_benchmark(
    load_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    write_benchmark(load_dir, [("pyfunc_dataframe", 1, 1.0), ("booster_numpy", 1, 4.0)])
    monkeypatch.setattr(
        "sys.argv", ["finops_calc.py", "--load-dir", str(load_dir), "--opt3-speedup", "2"]
    )
    finops_calc.main()
    summary = json.loads((load_dir / "finops_summary.json").read_text())
    assert opt(summary, "opt3")["savings_pct"] == pytest.approx(50.0)


def test_main_writes_the_json_and_an_annex_with_both_references(
    load_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("sys.argv", ["finops_calc.py", "--load-dir", str(load_dir)])
    finops_calc.main()
    summary = json.loads((load_dir / "finops_summary.json").read_text())
    annex = (load_dir / "anexo_c.md").read_text()
    assert summary["baseline"]["peak_scenario"] == "high_50u"
    assert "### C.4 Optimizaciones contra el pico sostenible" in annex
    assert "### C.5 Optimizaciones contra la operación real 24/7" in annex
    assert "Pendiente opt3" in annex
