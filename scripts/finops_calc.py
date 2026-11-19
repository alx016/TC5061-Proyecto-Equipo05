"""Estimate the cost per inference of the P2 stack from the load-test results.

Reads the Locust and ``docker stats`` CSVs that ``scripts/collect_load_metrics.py`` writes in
``reports/load/`` and applies the formulas of contract 4 of the Entrega 6 plan: cost per
request and per prediction at the sustainable peak, cost per prediction with the real 24/7
demand of MiBici, cost attribution between ``api`` and ``mlflow``, and four optimizations.
Writes ``finops_summary.json`` and ``anexo_c.md`` next to the CSVs.

Usage:
    uv run python scripts/finops_calc.py
    uv run python scripts/finops_calc.py --storage-gb 20 --opt3-speedup 2.5
    uv run python scripts/finops_calc.py --serverless-usd-per-second 0.000024

Prices default to the base parameters agreed in WP-0 (section 4 of
reports/TC5061_Operacion_FinOps_Equipo05.md).
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

PROJ_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOAD_DIR = PROJ_ROOT / "reports" / "load"

# Base parameters agreed in WP-0 (AWS us-east-1, consulted on 2026-11-18)
INSTANCE_TYPE = "t3.small"
INSTANCE_VCPUS = 2
HOURLY_USD = 0.0208
EBS_USD_PER_GB_MONTH = 0.08
HOURS_PER_MONTH = 730
# t3 instances only guarantee 20% of each vCPU; above that, unlimited mode bills credits
CPU_BASELINE_FRACTION = 0.20
CPU_CREDIT_USD_PER_VCPU_HOUR = 0.05

# Real demand: every active station is scored once per hour
MIBICI_ACTIVE_STATIONS = 365

# Mixed-traffic steps of the official run; the sustainable peak is chosen among them
LOAD_STEPS = ("nominal_10u", "high_50u", "stress_120u")
BATCH_SCENARIO = "batch_1x365"
UNIT_SCENARIO = "batch_365x1"
MAX_ERROR_PCT = 1.0
MAX_API_CPU_PCT = 80.0

PREDICT_NAME = re.compile(r"^/predict\?batch=(\d+)$")
AGGREGATED = "Aggregated"
CONTAINERS = ("api", "mlflow")


@dataclass(frozen=True)
class Throughput:
    """Throughput and errors of one Locust scenario."""

    scenario: str
    req_per_s: float
    pred_per_s: float
    error_pct: float
    median_batch_ms: float | None


@dataclass(frozen=True)
class Resources:
    """Mean ``docker stats`` readings of one scenario, per container."""

    scenario: str
    cpu_pct: dict[str, float]
    mem_mib: dict[str, float]

    @property
    def vcpus_used(self) -> float:
        """vCPUs in use by the whole stack (``docker stats`` reports 100% per core)."""
        return sum(self.cpu_pct.values()) / 100


def read_throughput(path: Path, scenario: str) -> Throughput:
    """Read ``locust_<slug>_stats.csv``: requests/s, predictions/s and error rate."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    aggregated = next((row for row in rows if row["Name"] == AGGREGATED), None)
    if aggregated is None:
        raise ValueError(f"{path} has no {AGGREGATED!r} row")

    pred_per_s = 0.0
    median_batch_ms = None
    for row in rows:
        match = PREDICT_NAME.match(row["Name"])
        if match is None:
            continue
        batch = int(match.group(1))
        pred_per_s += float(row["Requests/s"]) * batch
        if batch == MIBICI_ACTIVE_STATIONS:
            median_batch_ms = float(row["Median Response Time"])

    requests = float(aggregated["Request Count"])
    failures = float(aggregated["Failure Count"])
    return Throughput(
        scenario=scenario,
        req_per_s=float(aggregated["Requests/s"]),
        pred_per_s=pred_per_s,
        error_pct=100 * failures / requests if requests else 0.0,
        median_batch_ms=median_batch_ms,
    )


def read_resources(path: Path, scenario: str) -> Resources:
    """Read ``docker_stats_<slug>.csv`` and average CPU and memory per container."""
    cpu: dict[str, list[float]] = {name: [] for name in CONTAINERS}
    mem: dict[str, list[float]] = {name: [] for name in CONTAINERS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["container"] in cpu:
                cpu[row["container"]].append(float(row["cpu_pct"]))
                mem[row["container"]].append(float(row["mem_usage_mib"]))

    missing = [name for name in CONTAINERS if not cpu[name]]
    if missing:
        raise ValueError(f"{path} has no samples for {missing}")

    return Resources(
        scenario=scenario,
        cpu_pct={name: sum(values) / len(values) for name, values in cpu.items()},
        mem_mib={name: sum(values) / len(values) for name, values in mem.items()},
    )


def load_scenario(load_dir: Path, scenario: str) -> tuple[Throughput, Resources]:
    """Read the Locust and ``docker stats`` files of one scenario."""
    throughput = read_throughput(load_dir / f"locust_{scenario}_stats.csv", scenario)
    resources = read_resources(load_dir / f"docker_stats_{scenario}.csv", scenario)
    return throughput, resources


def cpu_credit_usd_per_hour(resources: Resources) -> float:
    """Unlimited-mode charge for the vCPU used above the t3 baseline."""
    surplus = resources.vcpus_used - CPU_BASELINE_FRACTION * INSTANCE_VCPUS
    return max(0.0, surplus) * CPU_CREDIT_USD_PER_VCPU_HOUR


def cost_per_unit(hourly_usd: float, units_per_s: float) -> float:
    """Cost of one request or prediction when the instance serves ``units_per_s``."""
    if units_per_s <= 0:
        raise ValueError("Throughput must be positive to price one unit")
    return hourly_usd / (units_per_s * 3600)


def pick_peak(
    steps: list[tuple[Throughput, Resources]],
    max_error_pct: float = MAX_ERROR_PCT,
    max_api_cpu_pct: float = MAX_API_CPU_PCT,
) -> tuple[Throughput, Resources]:
    """Sustainable peak: most predictions/s with few errors and the API below its CPU cap."""
    sustainable = [
        (throughput, resources)
        for throughput, resources in steps
        if throughput.error_pct <= max_error_pct and resources.cpu_pct["api"] <= max_api_cpu_pct
    ]
    if not sustainable:
        raise ValueError(
            f"No load step has errors <= {max_error_pct}% and api CPU <= {max_api_cpu_pct}%"
        )
    return max(sustainable, key=lambda step: step[0].pred_per_s)


def attribution(resources: Resources) -> dict[str, float]:
    """Share of the cost of each container: mean of its CPU share and its memory share."""
    cpu_total = sum(resources.cpu_pct.values())
    mem_total = sum(resources.mem_mib.values())
    shares = {}
    for name in CONTAINERS:
        cpu_share = resources.cpu_pct[name] / cpu_total if cpu_total else 0.5
        mem_share = resources.mem_mib[name] / mem_total if mem_total else 0.5
        shares[name] = 100 * (cpu_share + mem_share) / 2
    return shares


def savings_pct(reference: float, value: float) -> float:
    """Percent saved by ``value`` against ``reference``."""
    return 100 * (reference - value) / reference


def build_summary(
    load_dir: Path,
    storage_gb: float,
    opt3_speedup: float | None,
    serverless_usd_per_second: float | None,
) -> dict[str, Any]:
    """Apply the contract-4 formulas to the scenarios found in ``load_dir``."""
    steps = [load_scenario(load_dir, scenario) for scenario in LOAD_STEPS]
    peak_throughput, peak_resources = pick_peak(steps)

    storage_monthly = storage_gb * EBS_USD_PER_GB_MONTH
    monthly = HOURLY_USD * HOURS_PER_MONTH + storage_monthly
    real_pred_per_month = MIBICI_ACTIVE_STATIONS * HOURS_PER_MONTH

    peak_hourly = HOURLY_USD + cpu_credit_usd_per_hour(peak_resources)
    cost_per_pred_peak = cost_per_unit(peak_hourly, peak_throughput.pred_per_s)
    cost_per_1m_peak = cost_per_pred_peak * 1e6
    cost_per_1m_real = monthly / real_pred_per_month * 1e6
    shares = attribution(peak_resources)

    baseline = {
        "instance_type": INSTANCE_TYPE,
        "hourly_usd": HOURLY_USD,
        "storage_monthly_usd": storage_monthly,
        "monthly_usd": monthly,
        "peak_scenario": peak_throughput.scenario,
        "peak_req_per_s": peak_throughput.req_per_s,
        "peak_pred_per_s": peak_throughput.pred_per_s,
        "peak_cpu_credit_usd_per_h": peak_hourly - HOURLY_USD,
        "cost_per_req_peak_usd": cost_per_unit(peak_hourly, peak_throughput.req_per_s),
        "cost_per_pred_peak_usd": cost_per_pred_peak,
        "cost_per_1m_pred_peak_usd": cost_per_1m_peak,
        "real_pred_per_month": real_pred_per_month,
        "cost_per_1m_pred_real_24_7_usd": cost_per_1m_real,
        "attribution_pct": shares,
    }

    # Opt 1: one request of 365 stations instead of 365 requests of one station
    batch_throughput, batch_resources = load_scenario(load_dir, BATCH_SCENARIO)
    unit_throughput, unit_resources = load_scenario(load_dir, UNIT_SCENARIO)
    batch_1m = 1e6 * cost_per_unit(
        HOURLY_USD + cpu_credit_usd_per_hour(batch_resources), batch_throughput.pred_per_s
    )
    unit_1m = 1e6 * cost_per_unit(
        HOURLY_USD + cpu_credit_usd_per_hour(unit_resources), unit_throughput.pred_per_s
    )
    optimizations = [
        {
            "id": "opt1",
            "name": "Batching de 365 estaciones por petición",
            "reference": "peak",
            "cost_per_1m_pred_usd": batch_1m,
            "monthly_usd": monthly,
            "savings_pct": savings_pct(cost_per_1m_peak, batch_1m),
            "detail": {
                "unit_cost_per_1m_pred_usd": unit_1m,
                "savings_vs_unit_pct": savings_pct(unit_1m, batch_1m),
                "throughput_ratio": batch_throughput.pred_per_s / unit_throughput.pred_per_s,
            },
        },
    ]

    # Opt 2: serve only the API bundle; the MLflow share of the stack is no longer paid
    api_fraction = shares["api"] / 100
    optimizations.append(
        {
            "id": "opt2",
            "name": "MLflow fuera del camino de inferencia",
            "reference": "peak",
            "cost_per_1m_pred_usd": cost_per_1m_peak * api_fraction,
            "monthly_usd": monthly * api_fraction,
            "savings_pct": 100 - shares["api"],
        }
    )

    # Opt 3: less overhead per call (measured by the WP-4b microbenchmark)
    if opt3_speedup:
        opt3_1m = cost_per_1m_peak / opt3_speedup
        optimizations.append(
            {
                "id": "opt3",
                "name": "Inferencia directa con el Booster y menos sobrecarga por petición",
                "reference": "peak",
                "cost_per_1m_pred_usd": opt3_1m,
                "monthly_usd": monthly,
                "savings_pct": savings_pct(cost_per_1m_peak, opt3_1m),
            }
        )

    # Opt 4: pay only the seconds the hourly batch of 365 stations is running
    if serverless_usd_per_second and batch_throughput.median_batch_ms:
        active_s = HOURS_PER_MONTH * batch_throughput.median_batch_ms / 1000
        opt4_monthly = active_s * serverless_usd_per_second
        opt4_1m = opt4_monthly / real_pred_per_month * 1e6
        optimizations.append(
            {
                "id": "opt4",
                "name": "Scale-to-zero con una corrida por hora",
                "reference": "real_24_7",
                "cost_per_1m_pred_usd": opt4_1m,
                "monthly_usd": opt4_monthly,
                "savings_pct": savings_pct(cost_per_1m_real, opt4_1m),
                "detail": {"active_seconds_per_month": active_s},
            }
        )

    return {
        "parameters": {
            "hours_per_month": HOURS_PER_MONTH,
            "storage_gb": storage_gb,
            "ebs_usd_per_gb_month": EBS_USD_PER_GB_MONTH,
            "instance_vcpus": INSTANCE_VCPUS,
            "cpu_baseline_fraction": CPU_BASELINE_FRACTION,
            "cpu_credit_usd_per_vcpu_hour": CPU_CREDIT_USD_PER_VCPU_HOUR,
            "max_error_pct": MAX_ERROR_PCT,
            "max_api_cpu_pct": MAX_API_CPU_PCT,
            "opt3_speedup": opt3_speedup,
            "serverless_usd_per_second": serverless_usd_per_second,
        },
        "scenarios": [
            {**asdict(throughput), **asdict(resources), "vcpus_used": resources.vcpus_used}
            for throughput, resources in [
                *steps,
                (batch_throughput, batch_resources),
                (unit_throughput, unit_resources),
            ]
        ],
        "baseline": baseline,
        "optimizations": optimizations,
    }


def _usd(value: float) -> str:
    """Format small USD amounts without losing the significant digits."""
    return f"${value:,.2f}" if value >= 1 else f"${value:.4g}"


def render_annex(summary: dict[str, Any]) -> str:
    """Annex C of the report: inputs, formulas and results, ready to paste."""
    base = summary["baseline"]
    params = summary["parameters"]
    shares = base["attribution_pct"]
    hourly = _usd(base["hourly_usd"])
    credits = _usd(base["peak_cpu_credit_usd_per_h"])
    storage = _usd(base["storage_monthly_usd"])
    credit_price = _usd(params["cpu_credit_usd_per_vcpu_hour"])
    real_pred = f"{base['real_pred_per_month']:,}"
    peak_rule = (
        f"error ≤ {params['max_error_pct']:g}% y CPU de `api` ≤ {params['max_api_cpu_pct']:g}%"
    )

    lines = [
        "### C.1 Parámetros",
        "",
        "| Parámetro | Valor |",
        "|:---|:---|",
        f"| Instancia | AWS EC2 `{base['instance_type']}`, {hourly}/h |",
        f"| Disco | {params['storage_gb']:g} GB EBS gp3, {storage}/mes |",
        f"| Horas por mes | {params['hours_per_month']} |",
        (
            f"| Créditos de CPU | {credit_price} por vCPU-hora sobre "
            f"{params['cpu_baseline_fraction']:.0%} de cada vCPU |"
        ),
        f"| Demanda real | {real_pred} predicciones/mes |",
        "",
        "### C.2 Escenarios medidos",
        "",
        "| Escenario | Peticiones/s | Predicciones/s | Error (%) | CPU api (%) | CPU mlflow (%) |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["scenarios"]:
        lines.append(
            f"| `{row['scenario']}` | {row['req_per_s']:.1f} | {row['pred_per_s']:.1f} | "
            f"{row['error_pct']:.2f} | {row['cpu_pct']['api']:.1f} | "
            f"{row['cpu_pct']['mlflow']:.1f} |"
        )
    lines += [
        "",
        f"Pico sostenible: `{base['peak_scenario']}` (más predicciones/s con {peak_rule}).",
        "",
        "### C.3 Cálculo",
        "",
        f"- Costo/hora en el pico = tarifa + créditos = {hourly} + {credits}",
        (
            "- Costo/petición = costo/hora / (peticiones/s × 3600) = "
            f"{_usd(base['cost_per_req_peak_usd'])}"
        ),
        (
            "- Costo/predicción = costo/hora / (predicciones/s × 3600) = "
            f"{_usd(base['cost_per_pred_peak_usd'])}"
        ),
        (f"- Costo por 1M de predicciones en el pico = {_usd(base['cost_per_1m_pred_peak_usd'])}"),
        (
            f"- Costo mensual 24/7 = tarifa × {params['hours_per_month']} + disco = "
            f"{_usd(base['monthly_usd'])}"
        ),
        (
            f"- Costo por 1M de predicciones con la demanda real = costo mensual / {real_pred} "
            f"× 10⁶ = {_usd(base['cost_per_1m_pred_real_24_7_usd'])}"
        ),
        (
            f"- Atribución: `api` {shares['api']:.1f}%, `mlflow` {shares['mlflow']:.1f}% "
            "(promedio de su parte de CPU y de memoria)"
        ),
        "",
        "### C.4 Optimizaciones",
        "",
        "| Optimización | Referencia | Costo por 1M pred. | Costo mensual | Ahorro |",
        "|:---|:---|---:|---:|---:|",
    ]
    for opt in summary["optimizations"]:
        lines.append(
            f"| {opt['name']} | {opt['reference']} | {_usd(opt['cost_per_1m_pred_usd'])} | "
            f"{_usd(opt['monthly_usd'])} | {opt['savings_pct']:.1f}% |"
        )
    pending = {"opt3", "opt4"} - {opt["id"] for opt in summary["optimizations"]}
    if pending:
        lines += ["", f"Pendientes por falta de parámetro: {', '.join(sorted(pending))}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Read the load results, write finops_summary.json and anexo_c.md."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--load-dir", type=Path, default=DEFAULT_LOAD_DIR)
    parser.add_argument("--storage-gb", type=float, default=20.0)
    parser.add_argument(
        "--opt3-speedup", type=float, default=None, help="Speedup of the WP-4b microbenchmark"
    )
    parser.add_argument(
        "--serverless-usd-per-second",
        type=float,
        default=None,
        help="Serverless price per active second (WP-3b)",
    )
    args = parser.parse_args()

    summary = build_summary(
        args.load_dir, args.storage_gb, args.opt3_speedup, args.serverless_usd_per_second
    )
    (args.load_dir / "finops_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.load_dir / "anexo_c.md").write_text(render_annex(summary))
    print(f"Wrote finops_summary.json and anexo_c.md to {args.load_dir}")


if __name__ == "__main__":
    main()
