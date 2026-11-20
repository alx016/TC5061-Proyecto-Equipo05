"""Estimate the cost per inference of the P2 stack from the load-test results.

Reads the Locust and ``docker stats`` CSVs that ``scripts/collect_load_metrics.py`` writes in
``reports/load/`` and applies the formulas of contract 4 of the Entrega 6 plan: cost per
request and per prediction at the sustainable peak, cost per prediction with the real 24/7
demand of MiBici, cost attribution between ``api`` and ``mlflow``, and four optimizations.
Writes ``finops_summary.json`` and ``anexo_c.md`` next to the CSVs.

Usage:
    uv run python scripts/finops_calc.py
    uv run python scripts/finops_calc.py --storage-gb 20 --opt3-speedup 2.5
    uv run python scripts/finops_calc.py --serverless-usd-per-second 0.0000265 \
        --serverless-usd-per-request 0.0000004

Prices default to section 4 of reports/TC5061_Operacion_FinOps_Equipo05.md: the t3.small
baseline of WP-0 and the AWS Lambda prices of WP-3b. Opt 3 reads its speedup from
``opt3_benchmark.json`` when that file exists.

Each optimization carries a ``reference``: Opt 1 to 3 are priced against the sustainable peak
(``peak``) and Opt 4 against the real 24/7 demand (``real_24_7``). The two scales differ by
orders of magnitude, so charts must not put them on the same axis.
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
DEFAULT_STORAGE_GB = 10.0
# One convention for both the price and the demand, so they cannot drift apart
DEFAULT_HOURS_PER_MONTH = 730.0
# t3 instances only guarantee 20% of each vCPU; above that, unlimited mode bills credits
CPU_BASELINE_FRACTION = 0.20
CPU_CREDIT_USD_PER_VCPU_HOUR = 0.05

# Opt 4 defaults: AWS Lambda us-east-1, x86, 1 GB (WP-3b, consulted on 2026-11-18)
LAMBDA_USD_PER_SECOND = 0.0000166667
LAMBDA_USD_PER_REQUEST = 0.20 / 1_000_000

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
# Reference path of the WP-4b microbenchmark (the API as it runs today)
PYFUNC_PATH = "pyfunc_dataframe"
MISSING_VALUES = {"", "N/A", "--"}


@dataclass(frozen=True)
class Params:
    """Prices and assumptions of one FinOps run; invalid values are rejected."""

    storage_gb: float = DEFAULT_STORAGE_GB
    hours_per_month: float = DEFAULT_HOURS_PER_MONTH
    opt3_speedup: float | None = None
    serverless_usd_per_second: float = LAMBDA_USD_PER_SECOND
    serverless_usd_per_request: float = LAMBDA_USD_PER_REQUEST

    def __post_init__(self) -> None:
        if self.storage_gb < 0:
            raise ValueError("storage_gb must be >= 0")
        positive = {
            "hours_per_month": self.hours_per_month,
            "serverless_usd_per_second": self.serverless_usd_per_second,
        }
        if self.opt3_speedup is not None:
            positive["opt3_speedup"] = self.opt3_speedup
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.serverless_usd_per_request < 0:
            raise ValueError("serverless_usd_per_request must be >= 0")


@dataclass(frozen=True)
class Throughput:
    """Throughput and errors of one Locust scenario."""

    scenario: str
    req_per_s: float
    pred_per_s: float
    error_pct: float
    median_ms_by_batch: dict[int, float]


@dataclass(frozen=True)
class Resources:
    """Mean ``docker stats`` readings of one scenario, per container."""

    scenario: str
    cpu_pct: dict[str, float]
    mem_mib: dict[str, float]
    present: tuple[str, ...]

    @property
    def vcpus_used(self) -> float:
        """vCPUs in use by the whole stack (``docker stats`` reports 100% per core)."""
        return sum(self.cpu_pct.values()) / 100


def _safe_float(value: str | None) -> float | None:
    """Parse a Locust cell; ``N/A``, ``--`` and empty cells become None."""
    if value is None or value.strip() in MISSING_VALUES:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_throughput(path: Path, scenario: str) -> Throughput:
    """Read ``locust_<slug>_stats.csv``: requests/s, predictions/s, errors and medians."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    aggregated = next((row for row in rows if row["Name"] == AGGREGATED), None)
    if aggregated is None:
        raise ValueError(f"{path} has no {AGGREGATED!r} row")

    pred_per_s = 0.0
    medians: dict[int, float] = {}
    for row in rows:
        match = PREDICT_NAME.match(row["Name"])
        if match is None:
            continue
        batch = int(match.group(1))
        pred_per_s += (_safe_float(row.get("Requests/s")) or 0.0) * batch
        median = _safe_float(row.get("Median Response Time"))
        if median is None:
            median = _safe_float(row.get("50%"))
        if median is not None:
            medians[batch] = median

    requests = _safe_float(aggregated.get("Request Count")) or 0.0
    failures = _safe_float(aggregated.get("Failure Count")) or 0.0
    return Throughput(
        scenario=scenario,
        req_per_s=_safe_float(aggregated.get("Requests/s")) or 0.0,
        pred_per_s=pred_per_s,
        error_pct=100 * failures / requests if requests else 0.0,
        median_ms_by_batch=medians,
    )


def read_resources(path: Path, scenario: str) -> Resources:
    """Read ``docker_stats_<slug>.csv`` and average CPU and memory per container.

    ``api`` is required; ``mlflow`` may be absent (``docker compose up --no-deps api``), in
    which case it counts as zero.
    """
    cpu: dict[str, list[float]] = {name: [] for name in CONTAINERS}
    mem: dict[str, list[float]] = {name: [] for name in CONTAINERS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["container"] in cpu:
                cpu[row["container"]].append(float(row["cpu_pct"]))
                mem[row["container"]].append(float(row["mem_usage_mib"]))

    if not cpu["api"]:
        raise ValueError(f"{path} has no samples for the api container")

    present = tuple(name for name in CONTAINERS if cpu[name])
    return Resources(
        scenario=scenario,
        cpu_pct={name: sum(v) / len(v) if v else 0.0 for name, v in cpu.items()},
        mem_mib={name: sum(v) / len(v) if v else 0.0 for name, v in mem.items()},
        present=present,
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
        if throughput.error_pct <= max_error_pct
        and resources.cpu_pct["api"] <= max_api_cpu_pct
        and throughput.pred_per_s > 0
    ]
    if not sustainable:
        raise ValueError(
            f"No load step has errors <= {max_error_pct}% and api CPU <= {max_api_cpu_pct}%"
        )
    return max(sustainable, key=lambda step: step[0].pred_per_s)


def attribution(resources: Resources) -> dict[str, float]:
    """Share of the cost of each container: mean of its CPU share and its memory share.

    A container without samples gets 0%; the containers present split evenly any resource
    whose total is zero.
    """
    present = resources.present
    cpu_total = sum(resources.cpu_pct[name] for name in present)
    mem_total = sum(resources.mem_mib[name] for name in present)
    shares = {}
    for name in CONTAINERS:
        if name not in present:
            shares[name] = 0.0
            continue
        cpu_share = resources.cpu_pct[name] / cpu_total if cpu_total else 1 / len(present)
        mem_share = resources.mem_mib[name] / mem_total if mem_total else 1 / len(present)
        shares[name] = 100 * (cpu_share + mem_share) / 2
    return shares


def read_opt3_speedup(path: Path) -> float | None:
    """Conservative speedup of the WP-4b microbenchmark, or None when it was not run.

    Each alternative path keeps its worst speedup across batch sizes; the best of those is
    used, so the saving holds for every batch size the API serves.
    """
    if not path.exists():
        return None
    worst_by_path: dict[str, float] = {}
    for row in json.loads(path.read_text())["results"]:
        if row["path"] == PYFUNC_PATH:
            continue
        speedup = float(row["speedup_vs_pyfunc"])
        worst_by_path[row["path"]] = min(speedup, worst_by_path.get(row["path"], speedup))
    valid = [speedup for speedup in worst_by_path.values() if speedup > 0]
    return max(valid) if valid else None


def savings_pct(reference: float, value: float) -> float:
    """Percent saved by ``value`` against ``reference``."""
    return 100 * (reference - value) / reference


def serverless_month(
    requests: float, median_ms: float, params: Params, real_pred: float
) -> dict[str, float]:
    """Monthly serverless bill for ``requests`` calls of ``median_ms`` each."""
    active_s = requests * median_ms / 1000
    compute = active_s * params.serverless_usd_per_second
    invocations = requests * params.serverless_usd_per_request
    monthly = compute + invocations
    return {
        "requests_per_month": requests,
        "active_seconds_per_month": active_s,
        "compute_usd": compute,
        "requests_usd": invocations,
        "monthly_usd": monthly,
        "cost_per_1m_pred_usd": monthly / real_pred * 1e6,
    }


def build_summary(load_dir: Path, params: Params) -> dict[str, Any]:
    """Apply the contract-4 formulas to the scenarios found in ``load_dir``."""
    steps = [load_scenario(load_dir, scenario) for scenario in LOAD_STEPS]
    peak_throughput, peak_resources = pick_peak(steps)

    storage_monthly = params.storage_gb * EBS_USD_PER_GB_MONTH
    monthly = HOURLY_USD * params.hours_per_month + storage_monthly
    real_pred_per_month = MIBICI_ACTIVE_STATIONS * params.hours_per_month

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
    optimizations: list[dict[str, Any]] = [
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

    pending: dict[str, str] = {}

    # Opt 3: less overhead per call (measured by the WP-4b microbenchmark)
    if params.opt3_speedup is not None:
        opt3_1m = cost_per_1m_peak / params.opt3_speedup
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
    else:
        pending["opt3"] = "falta opt3_benchmark.json o --opt3-speedup"

    # Opt 4: pay only the seconds the hourly batch of 365 stations is running
    batch_ms = batch_throughput.median_ms_by_batch.get(MIBICI_ACTIVE_STATIONS)
    if batch_ms is not None and batch_ms > 0:
        batch_bill = serverless_month(
            params.hours_per_month, batch_ms, params, real_pred_per_month
        )
        detail: dict[str, Any] = {"batch": batch_bill}
        unit_ms = unit_throughput.median_ms_by_batch.get(1)
        if unit_ms is not None and unit_ms > 0:
            unit_bill = serverless_month(real_pred_per_month, unit_ms, params, real_pred_per_month)
            unit_bill["savings_pct"] = savings_pct(
                cost_per_1m_real, unit_bill["cost_per_1m_pred_usd"]
            )
            detail["unit"] = unit_bill
        optimizations.append(
            {
                "id": "opt4",
                "name": "Scale-to-zero con una corrida por hora",
                "reference": "real_24_7",
                "cost_per_1m_pred_usd": batch_bill["cost_per_1m_pred_usd"],
                "monthly_usd": batch_bill["monthly_usd"],
                "savings_pct": savings_pct(cost_per_1m_real, batch_bill["cost_per_1m_pred_usd"]),
                "detail": detail,
            }
        )
    else:
        pending["opt4"] = f"sin latencia mediana válida de /predict?batch=365 en {BATCH_SCENARIO}"

    return {
        "parameters": {
            **asdict(params),
            "ebs_usd_per_gb_month": EBS_USD_PER_GB_MONTH,
            "instance_vcpus": INSTANCE_VCPUS,
            "cpu_baseline_fraction": CPU_BASELINE_FRACTION,
            "cpu_credit_usd_per_vcpu_hour": CPU_CREDIT_USD_PER_VCPU_HOUR,
            "max_error_pct": MAX_ERROR_PCT,
            "max_api_cpu_pct": MAX_API_CPU_PCT,
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
        "pending": pending,
    }


def _usd(value: float) -> str:
    """Format small USD amounts without losing the significant digits."""
    return f"${value:,.2f}" if value >= 1 else f"${value:.4g}"


def _optimization_rows(summary: dict[str, Any], reference: str) -> list[str]:
    """Rows of the optimizations priced against one reference."""
    return [
        f"| {opt['name']} | {_usd(opt['cost_per_1m_pred_usd'])} | "
        f"{_usd(opt['monthly_usd'])} | {opt['savings_pct']:.2f}% |"
        for opt in summary["optimizations"]
        if opt["reference"] == reference
    ]


def render_annex(summary: dict[str, Any]) -> str:
    """Annex C of the report: inputs, formulas and results, ready to paste."""
    base = summary["baseline"]
    params = summary["parameters"]
    shares = base["attribution_pct"]
    hours = f"{params['hours_per_month']:g}"
    hourly = _usd(base["hourly_usd"])
    credits = _usd(base["peak_cpu_credit_usd_per_h"])
    storage = _usd(base["storage_monthly_usd"])
    credit_price = _usd(params["cpu_credit_usd_per_vcpu_hour"])
    real_pred = f"{base['real_pred_per_month']:,.0f}"
    peak_rule = (
        f"error ≤ {params['max_error_pct']:g}% y CPU de `api` ≤ {params['max_api_cpu_pct']:g}%"
    )
    table_head = [
        "| Optimización | Costo por 1M pred. | Costo mensual | Ahorro |",
        "|:---|---:|---:|---:|",
    ]

    lines = [
        "### C.1 Parámetros",
        "",
        "| Parámetro | Valor |",
        "|:---|:---|",
        f"| Instancia | AWS EC2 `{base['instance_type']}`, {hourly}/h |",
        f"| Disco | {params['storage_gb']:g} GB EBS gp3, {storage}/mes |",
        f"| Horas por mes (precio y demanda) | {hours} |",
        (
            f"| Créditos de CPU | {credit_price} por vCPU-hora sobre "
            f"{params['cpu_baseline_fraction']:.0%} de cada vCPU |"
        ),
        (
            f"| Demanda real | {MIBICI_ACTIVE_STATIONS} estaciones × {hours} h = {real_pred} "
            "predicciones/mes |"
        ),
        (
            f"| Serverless (Opt 4) | {_usd(params['serverless_usd_per_second'])}/s activo + "
            f"{_usd(params['serverless_usd_per_request'])}/petición |"
        ),
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
        f"- Costo por 1M de predicciones en el pico = {_usd(base['cost_per_1m_pred_peak_usd'])}",
        f"- Costo mensual 24/7 = tarifa × {hours} + disco = {_usd(base['monthly_usd'])}",
        (
            f"- Costo por 1M de predicciones con la demanda real = costo mensual / {real_pred} "
            f"× 10⁶ = {_usd(base['cost_per_1m_pred_real_24_7_usd'])}"
        ),
        (
            f"- Atribución: `api` {shares['api']:.1f}%, `mlflow` {shares['mlflow']:.1f}% "
            "(promedio de su parte de CPU y de memoria)"
        ),
        "",
        "### C.4 Optimizaciones contra el pico sostenible",
        "",
        f"Línea base: {_usd(base['cost_per_1m_pred_peak_usd'])} por 1M de predicciones.",
        "",
        *table_head,
        *_optimization_rows(summary, "peak"),
        "",
        "### C.5 Optimizaciones contra la operación real 24/7",
        "",
        (
            f"Línea base: {_usd(base['monthly_usd'])}/mes, "
            f"{_usd(base['cost_per_1m_pred_real_24_7_usd'])} por 1M de predicciones. Las dos "
            "escalas no son comparables entre sí."
        ),
        "",
        *table_head,
        *_optimization_rows(summary, "real_24_7"),
    ]
    opt4 = next((opt for opt in summary["optimizations"] if opt["id"] == "opt4"), None)
    if opt4 is not None and "unit" in opt4["detail"]:
        unit = opt4["detail"]["unit"]
        lines += [
            "",
            (
                f"Sin agrupar (365 peticiones de 1 estación por hora): {_usd(unit['monthly_usd'])}"
                f"/mes, ahorro de {unit['savings_pct']:.2f}%."
            ),
        ]
    for opt_id, reason in sorted(summary["pending"].items()):
        lines += ["", f"Pendiente {opt_id}: {reason}."]
    return "\n".join(lines) + "\n"


def _positive(value: str) -> float:
    """argparse type for prices and factors that must be > 0."""
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return number


def _non_negative(value: str) -> float:
    """argparse type for quantities that may be 0 but not negative."""
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return number


def main() -> None:
    """Read the load results, write finops_summary.json and anexo_c.md."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--load-dir", type=Path, default=DEFAULT_LOAD_DIR)
    parser.add_argument("--storage-gb", type=_non_negative, default=DEFAULT_STORAGE_GB)
    parser.add_argument(
        "--hours-per-month",
        type=_positive,
        default=DEFAULT_HOURS_PER_MONTH,
        help="Used for both the instance price and the real demand",
    )
    parser.add_argument(
        "--opt3-speedup",
        type=_positive,
        default=None,
        help="Speedup for Opt 3; by default it is read from opt3_benchmark.json (WP-4b)",
    )
    parser.add_argument(
        "--serverless-usd-per-second",
        type=_positive,
        default=LAMBDA_USD_PER_SECOND,
        help="Serverless price per active second (default: AWS Lambda, 1 GB)",
    )
    parser.add_argument(
        "--serverless-usd-per-request",
        type=_non_negative,
        default=LAMBDA_USD_PER_REQUEST,
        help="Serverless price per request (default: AWS Lambda)",
    )
    args = parser.parse_args()

    opt3_speedup = args.opt3_speedup
    if opt3_speedup is None:
        opt3_speedup = read_opt3_speedup(args.load_dir / "opt3_benchmark.json")
    params = Params(
        storage_gb=args.storage_gb,
        hours_per_month=args.hours_per_month,
        opt3_speedup=opt3_speedup,
        serverless_usd_per_second=args.serverless_usd_per_second,
        serverless_usd_per_request=args.serverless_usd_per_request,
    )
    summary = build_summary(args.load_dir, params)
    (args.load_dir / "finops_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.load_dir / "anexo_c.md").write_text(render_annex(summary))
    print(f"Wrote finops_summary.json and anexo_c.md to {args.load_dir}")


if __name__ == "__main__":
    main()
