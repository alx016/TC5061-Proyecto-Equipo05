"""Write synthetic load-test outputs with the exact shape of the real ones.

Lets ``scripts/plot_load_and_finops.py`` be developed and tested before the official
Locust run.  The numbers are invented but behave like a saturating API: latency grows
with load and ``stress_120u`` has some failures.  Files written per scenario:

- ``locust_<slug>_stats.csv``, ``_stats_history.csv``, ``_failures.csv`` and
  ``_exceptions.csv``, with the columns of Locust 2.47.  The history has one row per
  endpoint, as ``--csv-full-history`` writes it; without that flag Locust only writes
  the ``Aggregated`` rows.
- ``docker_stats_<slug>.csv`` with the columns of the plan's contract.

Plus ``finops_summary.json`` with the plan's schema.  ``baseline`` has no Locust files,
it is the run without traffic.

Usage:
    uv run python tests/fixtures/load_fixtures.py --output-dir /tmp/load-fixtures
"""

import argparse
import csv
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import random

PERCENTILES = ["50%", "66%", "75%", "80%", "90%", "95%", "98%", "99%", "99.9%", "99.99%", "100%"]
# How much slower each percentile is than the median.
PERCENTILE_FACTORS = [1.0, 1.15, 1.25, 1.3, 1.5, 1.8, 2.2, 2.6, 3.5, 4.0, 4.5]

STATS_COLUMNS = [
    "Type",
    "Name",
    "Request Count",
    "Failure Count",
    "Median Response Time",
    "Average Response Time",
    "Min Response Time",
    "Max Response Time",
    "Average Content Size",
    "Requests/s",
    "Failures/s",
    *PERCENTILES,
]
HISTORY_COLUMNS = [
    "Timestamp",
    "User Count",
    "Type",
    "Name",
    "Requests/s",
    "Failures/s",
    *PERCENTILES,
    "Total Request Count",
    "Total Failure Count",
    "Total Median Response Time",
    "Total Average Response Time",
    "Total Min Response Time",
    "Total Max Response Time",
    "Total Average Content Size",
]
FAILURES_COLUMNS = ["Method", "Name", "Error", "Occurrences", "First Seen", "Last Seen"]
EXCEPTIONS_COLUMNS = ["Count", "Message", "Traceback", "Nodes"]
DOCKER_COLUMNS = [
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

# Endpoint: (method, median latency at low load in ms, response size in bytes).
ENDPOINTS = {
    "/predict?batch=1": ("POST", 25, 180),
    "/predict?batch=20": ("POST", 45, 2_900),
    "/predict?batch=365": ("POST", 190, 52_000),
    "/health": ("GET", 3, 60),
    "/model-info": ("GET", 5, 420),
}
MIXED_WEIGHTS = {
    "/predict?batch=1": 5,
    "/predict?batch=20": 3,
    "/predict?batch=365": 1,
    "/health": 1,
    "/model-info": 1,
}

# slug: (users, spawn rate per s, duration in s, endpoint weights, API utilization, failure rate)
SCENARIOS = {
    "baseline": (0, 0, 60, {}, 0.0, 0.0),
    "nominal_10u": (10, 2, 120, MIXED_WEIGHTS, 0.30, 0.0),
    "high_50u": (50, 5, 120, MIXED_WEIGHTS, 0.80, 0.0),
    "stress_120u": (120, 10, 120, MIXED_WEIGHTS, 1.40, 0.02),
    "batch_1x365": (10, 10, 60, {"/predict?batch=365": 1}, 0.70, 0.0),
    "batch_365x1": (10, 10, 60, {"/predict?batch=1": 1}, 0.90, 0.0),
}
WAIT_TIME_S = 0.6  # mean of Locust's between(0.2, 1.0)
START = datetime(2026, 11, 20, 18, 0, tzinfo=UTC)
GAP_S = 30  # pause between scenarios


def latency_factor(utilization: float) -> float:
    """Queueing slowdown: latency grows quickly as the API nears saturation."""
    return 1 / (1 - min(utilization, 0.93))


def percentile_values(median_ms: float) -> list[int]:
    return [round(median_ms * factor) for factor in PERCENTILE_FACTORS]


def write_csv(path: Path, columns: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def write_locust(out_dir: Path, slug: str, start: datetime, rng: random.Random) -> None:
    users, spawn_rate, duration, weights, utilization, failure_rate = SCENARIOS[slug]
    total_weight = sum(weights.values())
    slowdown = latency_factor(utilization)
    medians = {name: ENDPOINTS[name][1] * slowdown for name in weights}
    mean_latency_s = sum(medians[n] * w for n, w in weights.items()) / total_weight / 1000
    # Past saturation, extra users only add waiting time, not throughput.
    capacity_rps = users / (WAIT_TIME_S + mean_latency_s) / max(utilization, 1.0)

    totals = {name: [0, 0] for name in weights}  # requests, failures
    history = []
    for second in range(duration + 1):
        current_users = min(users, spawn_rate * second)
        rps = capacity_rps * current_users / users * rng.uniform(0.92, 1.08)
        timestamp = int((start + timedelta(seconds=second)).timestamp())
        rows = []
        for name, weight in weights.items():
            requests = rps * weight / total_weight
            failures = requests * failure_rate
            totals[name][0] += round(requests)
            totals[name][1] += round(failures)
            if totals[name][0] == 0:
                current = ["N/A"] * len(PERCENTILES)
            else:
                current = percentile_values(medians[name] * rng.uniform(0.9, 1.1))
            rows.append(
                [timestamp, current_users, ENDPOINTS[name][0], name]
                + [round(requests, 2), round(failures, 2), *current]
                + [*totals[name], round(medians[name]), round(medians[name] * 1.2)]
                + [round(medians[name] * 0.4), round(medians[name] * 4.5), ENDPOINTS[name][2]]
            )
        requests_so_far = sum(t[0] for t in totals.values())
        aggregated_median = mean_latency_s * 1000
        if requests_so_far == 0:
            current = ["N/A"] * len(PERCENTILES)
        else:
            current = percentile_values(aggregated_median * rng.uniform(0.9, 1.1))
        history.extend(rows)
        history.append(
            [timestamp, current_users, "", "Aggregated", round(rps, 2)]
            + [round(rps * failure_rate, 2), *current]
            + [requests_so_far, sum(t[1] for t in totals.values())]
            + [round(aggregated_median), round(aggregated_median * 1.2)]
            + [round(aggregated_median * 0.1), round(aggregated_median * 6), 0]
        )
    write_csv(out_dir / f"locust_{slug}_stats_history.csv", HISTORY_COLUMNS, history)

    stats, failures = [], []
    for name, (requests, failed) in totals.items():
        median = medians[name]
        stats.append(
            [ENDPOINTS[name][0], name, requests, failed, round(median), round(median * 1.2)]
            + [round(median * 0.4), round(median * 4.5), ENDPOINTS[name][2]]
            + [round(requests / duration, 2), round(failed / duration, 2)]
            + percentile_values(median)
        )
        if failed:
            failures.append(
                [ENDPOINTS[name][0], name, "HTTPError('503 Server Error')", failed, "", ""]
            )
    requests_total = sum(t[0] for t in totals.values())
    failed_total = sum(t[1] for t in totals.values())
    aggregated_median = mean_latency_s * 1000
    stats.append(
        ["", "Aggregated", requests_total, failed_total, round(aggregated_median)]
        + [round(aggregated_median * 1.2), round(aggregated_median * 0.1)]
        + [round(aggregated_median * 6), 0, round(requests_total / duration, 2)]
        + [round(failed_total / duration, 2), *percentile_values(aggregated_median)]
    )
    write_csv(out_dir / f"locust_{slug}_stats.csv", STATS_COLUMNS, stats)
    write_csv(out_dir / f"locust_{slug}_failures.csv", FAILURES_COLUMNS, failures)
    write_csv(out_dir / f"locust_{slug}_exceptions.csv", EXCEPTIONS_COLUMNS, [])


def write_docker_stats(out_dir: Path, slug: str, start: datetime, rng: random.Random) -> None:
    _, _, duration, _, utilization, _ = SCENARIOS[slug]
    # container: (limit MiB, idle CPU %, idle MiB, extra MiB at full load, pids)
    containers = {"api": (1024, 0.4, 310, 140, 45), "mlflow": (512, 0.8, 175, 5, 12)}
    rows = []
    net = {name: [0.0, 0.0] for name in containers}
    for second in range(0, duration + 1, 2):
        timestamp = (start + timedelta(seconds=second)).isoformat().replace("+00:00", "Z")
        for name, (limit, idle_cpu, idle_mem, extra_mem, pids) in containers.items():
            load = min(utilization, 1.0) if name == "api" else 0.0
            cpu = idle_cpu + (99 - idle_cpu) * load
            mem = idle_mem + extra_mem * load
            net[name][0] += (2.5 if name == "api" else 0.01) * load * 2
            net[name][1] += (6.0 if name == "api" else 0.01) * load * 2
            cpu *= rng.uniform(0.9, 1.1)
            mem *= rng.uniform(0.99, 1.01)
            rows.append(
                [timestamp, slug, name, round(min(cpu, 100.0), 2), round(mem, 1), limit]
                + [round(100 * mem / limit, 2), round(net[name][0], 2), round(net[name][1], 2)]
                + [pids]
            )
    write_csv(out_dir / f"docker_stats_{slug}.csv", DOCKER_COLUMNS, rows)


def finops_summary() -> dict:
    hourly, storage_monthly, hours = 0.0208, 20 * 0.08, 730
    monthly = hourly * hours + storage_monthly
    peak_req, peak_pred = 45.0, 2_100.0
    real_pred_month = 365 * 24 * 30
    real_per_1m = monthly / real_pred_month * 1_000_000
    optimizations = [
        ("opt1", "Batching de 365 estaciones por petición", 0.92),
        ("opt2", "MLflow fuera del camino de inferencia", 0.75),
        ("opt3", "Booster + numpy sin DataFrame por petición", 0.85),
        ("opt4", "Scale-to-zero (serverless)", 0.04),
    ]
    return {
        "baseline": {
            "instance_type": "t3.small",
            "hourly_usd": hourly,
            "storage_monthly_usd": storage_monthly,
            "monthly_usd": round(monthly, 4),
            "peak_req_per_s": peak_req,
            "peak_pred_per_s": peak_pred,
            "cost_per_req_peak_usd": hourly / (peak_req * 3600),
            "cost_per_pred_peak_usd": hourly / (peak_pred * 3600),
            "cost_per_1m_pred_peak_usd": round(hourly / (peak_pred * 3600) * 1_000_000, 6),
            "cost_per_1m_pred_real_24_7_usd": round(real_per_1m, 4),
            "attribution_pct": {"api": 82.0, "mlflow": 18.0},
        },
        "optimizations": [
            {
                "id": opt_id,
                "name": name,
                "cost_per_1m_pred_usd": round(real_per_1m * ratio, 4),
                "monthly_usd": round(monthly * ratio, 4),
                "savings_pct": round(100 * (1 - ratio), 1),
            }
            for opt_id, name, ratio in optimizations
        ],
    }


def write_load_fixtures(out_dir: Path, seed: int = 42) -> None:
    """Write every fixture file into ``out_dir``; same seed, same files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    start = START
    for slug, (_, _, duration, _, _, _) in SCENARIOS.items():
        if slug != "baseline":
            write_locust(out_dir, slug, start, rng)
        write_docker_stats(out_dir, slug, start, rng)
        start += timedelta(seconds=duration + GAP_S)
    (out_dir / "finops_summary.json").write_text(
        json.dumps(finops_summary(), indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_load_fixtures(args.output_dir, args.seed)
    print(f"Fixtures written to {args.output_dir}")


if __name__ == "__main__":
    main()
