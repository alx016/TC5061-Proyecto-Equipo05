"""Orchestrate Locust load tests, container resource sampling, and inference microbenchmarks.

Produces:
    - ``reports/load/locust_<slug>_stats.csv`` (and history/failures CSVs; HTML is gitignored)
    - ``reports/load/docker_stats_<slug>.csv`` with normalized container names and numeric units
    - ``reports/load/opt3_benchmark.csv`` and ``reports/load/opt3_benchmark.json``
    - ``reports/load/anexo_b.md`` with Markdown tables ready for the technical report

Usage:
    uv run python scripts/collect_load_metrics.py run --host http://127.0.0.1:8000
    uv run python scripts/collect_load_metrics.py run --scenarios baseline,nominal_10u
    uv run python scripts/collect_load_metrics.py benchmark --model-uri models/p2-riesgo
    uv run python scripts/collect_load_metrics.py report --load-dir reports/load
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess  # nosec B404
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse
import urllib.request

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from load_tests.payloads import build_predict_payload

from tc5061_proyecto_equipo5.api.model import load_model, predict
from tc5061_proyecto_equipo5.modeling.train_classification import (
    CLASS_NAMES_ES,
    FEATURE_COLUMNS,
    apply_thresholds,
)

DEFAULT_HOST = "http://127.0.0.1:8000"
DEFAULT_OUTPUT_DIR = PROJ_ROOT / "reports" / "load"
DEFAULT_LOCUSTFILE = PROJ_ROOT / "load_tests" / "locustfile.py"
DEFAULT_MODEL_URI = PROJ_ROOT / "models" / "p2-riesgo"

DOCKER_STATS_COLUMNS = (
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
)

COMPOSE_CONTAINERS = ("mibici-api", "mibici-mlflow")
SINGLE_CONTAINER_FALLBACK = ("mibici-p2-api",)

CONTAINER_ALIASES = {
    "mibici-api": "api",
    "mibici-p2-api": "api",
    "mibici-mlflow": "mlflow",
}

_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)?\s*$")

_UNIT_BYTES = {
    "b": 1.0,
    "kb": 1_000.0,
    "mb": 1_000_000.0,
    "gb": 1_000_000_000.0,
    "tb": 1_000_000_000_000.0,
    "kib": 1024.0,
    "mib": 1024.0**2,
    "gib": 1024.0**3,
    "tib": 1024.0**4,
}


@dataclass(frozen=True)
class ScenarioSpec:
    """Definition of a single headless load or idle baseline scenario."""

    slug: str
    duration_s: int
    users: int = 0
    spawn_rate: int = 0
    locust_scenario: str | None = None


DEFAULT_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(slug="baseline", duration_s=60, users=0, spawn_rate=0, locust_scenario=None),
    ScenarioSpec(
        slug="nominal_10u", duration_s=120, users=10, spawn_rate=2, locust_scenario="mixed"
    ),
    ScenarioSpec(slug="high_50u", duration_s=120, users=50, spawn_rate=5, locust_scenario="mixed"),
    ScenarioSpec(
        slug="stress_120u", duration_s=120, users=120, spawn_rate=10, locust_scenario="mixed"
    ),
    ScenarioSpec(
        slug="batch_1x365",
        duration_s=60,
        users=1,
        spawn_rate=1,
        locust_scenario="batch_1x365",
    ),
    ScenarioSpec(
        slug="batch_365x1",
        duration_s=60,
        users=1,
        spawn_rate=1,
        locust_scenario="batch_365x1",
    ),
)

SCENARIO_MAP: dict[str, ScenarioSpec] = {spec.slug: spec for spec in DEFAULT_SCENARIOS}


def utc_timestamp() -> str:
    """Return the current UTC timestamp formatted in ISO-8601."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_pct(raw: str) -> float:
    """Parse a percentage string such as ``'12.34%'`` or ``'--'`` into a float."""
    cleaned = raw.strip().rstrip("%").strip()
    if not cleaned or cleaned in {"--", "N/A", "nan", "NaN"}:
        return 0.0
    return round(float(cleaned), 4)


def parse_pids(raw: str) -> int:
    """Parse a PID count string such as ``'17'`` or ``'--'`` into an integer."""
    cleaned = raw.strip()
    if not cleaned or cleaned in {"--", "N/A"}:
        return 0
    return int(float(cleaned))


def parse_size_to_bytes(raw: str) -> float:
    """Convert a Docker/Podman size string (IEC or SI) into bytes."""
    cleaned = raw.strip()
    if not cleaned or cleaned in {"--", "N/A"}:
        return 0.0
    match = _SIZE_RE.match(cleaned)
    if not match:
        raise ValueError(f"Unrecognized size string: {raw!r}")
    value = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    if unit not in _UNIT_BYTES:
        raise ValueError(f"Unsupported size unit {unit!r} in {raw!r}")
    return value * _UNIT_BYTES[unit]


def parse_size_to_mib(raw: str) -> float:
    """Convert a Docker/Podman memory string into MiB (base 1024)."""
    return round(parse_size_to_bytes(raw) / (1024.0**2), 4)


def parse_size_to_mb(raw: str) -> float:
    """Convert a Docker/Podman network I/O string into MB (base 1000)."""
    return round(parse_size_to_bytes(raw) / 1_000_000.0, 4)


def parse_pair(raw: str, converter: Any) -> tuple[float, float]:
    """Parse a ``'<left> / <right>'`` stats field using ``converter``."""
    if "/" not in raw:
        val = converter(raw)
        return val, 0.0
    left, right = raw.split("/", maxsplit=1)
    return converter(left), converter(right)


def normalize_container_name(name: str) -> str:
    """Map container names (e.g. ``mibici-api``, ``mibici-p2-api``) to ``api`` or ``mlflow``."""
    stripped = name.strip().lstrip("/")
    if stripped in CONTAINER_ALIASES:
        return CONTAINER_ALIASES[stripped]
    lowered = stripped.lower()
    if "mlflow" in lowered:
        return "mlflow"
    if "api" in lowered:
        return "api"
    return lowered.removeprefix("mibici-")


def parse_docker_stats_output(
    stdout: str,
    scenario: str,
    timestamp: str | None = None,
) -> list[dict[str, Any]]:
    """Parse tab-separated ``docker stats --no-stream`` output into contract rows."""
    ts = timestamp or utc_timestamp()
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("\t")
        if len(parts) < 6:
            raise ValueError(f"Expected 6 tab-separated fields in docker stats row, got: {line!r}")
        raw_name, raw_cpu, raw_mem_usage, raw_mem_pct, raw_net_io, raw_pids = parts[:6]
        mem_usage_mib, mem_limit_mib = parse_pair(raw_mem_usage, parse_size_to_mib)
        net_rx_mb, net_tx_mb = parse_pair(raw_net_io, parse_size_to_mb)
        rows.append(
            {
                "timestamp": ts,
                "scenario": scenario,
                "container": normalize_container_name(raw_name),
                "cpu_pct": parse_pct(raw_cpu),
                "mem_usage_mib": mem_usage_mib,
                "mem_limit_mib": mem_limit_mib,
                "mem_pct": parse_pct(raw_mem_pct),
                "net_rx_mb": net_rx_mb,
                "net_tx_mb": net_tx_mb,
                "pids": parse_pids(raw_pids),
            }
        )
    return rows


def detect_container_engine(preferred: str = "auto") -> str:
    """Locate ``docker`` or ``podman`` executable on PATH."""
    if preferred != "auto":
        resolved = shutil.which(preferred)
        if not resolved:
            raise RuntimeError(f"Requested container engine {preferred!r} not found on PATH")
        return resolved
    for candidate in ("docker", "podman"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Neither 'docker' nor 'podman' was found on PATH")


def list_running_containers(engine_bin: str) -> list[str]:
    """Return currently running container names."""
    result = subprocess.run(  # nosec B603
        [engine_bin, "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def resolve_target_containers(engine_bin: str, explicit: list[str] | None = None) -> list[str]:
    """Select active target containers (Compose pair first, single-container fallback)."""
    if explicit:
        return explicit
    running = set(list_running_containers(engine_bin))
    compose_present = [name for name in COMPOSE_CONTAINERS if name in running]
    if compose_present:
        return compose_present
    fallback_present = [name for name in SINGLE_CONTAINER_FALLBACK if name in running]
    if fallback_present:
        return fallback_present
    raise RuntimeError(
        "No active MiBici containers found. Start the stack (`docker compose up -d` or "
        "`make container`) or pass --containers explicitly."
    )


def sample_docker_stats_once(
    engine_bin: str,
    containers: list[str],
    scenario: str,
    timestamp: str | None = None,
) -> list[dict[str, Any]]:
    """Collect a single ``stats --no-stream`` snapshot across ``containers``."""
    cmd = [
        engine_bin,
        "stats",
        "--no-stream",
        "--format",
        "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.PIDs}}",
        *containers,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec B603
    return parse_docker_stats_output(result.stdout, scenario=scenario, timestamp=timestamp)


def write_docker_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write normalized container metric rows to CSV with exact header order."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DOCKER_STATS_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in DOCKER_STATS_COLUMNS})


class DockerStatsSampler:
    """Background sampler polling ``docker stats --no-stream`` at a fixed interval."""

    def __init__(
        self,
        engine_bin: str,
        containers: list[str],
        scenario: str,
        interval_s: float = 1.5,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")
        self.engine_bin = engine_bin
        self.containers = containers
        self.scenario = scenario
        self.interval_s = interval_s
        self.rows: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._first_sample_done = threading.Event()
        self._thread: threading.Thread | None = None
        self._errors: list[Exception] = []

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self.rows.extend(
                    sample_docker_stats_once(
                        engine_bin=self.engine_bin,
                        containers=self.containers,
                        scenario=self.scenario,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._errors.append(exc)
            finally:
                self._first_sample_done.set()
            elapsed = time.monotonic() - started
            wait_for = max(0.0, self.interval_s - elapsed)
            if self._stop_event.wait(wait_for):
                break

    def start(self) -> None:
        """Start sampling in a daemon thread."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        """Stop sampling, wait for the worker thread, and return collected rows."""
        if self._thread is not None:
            self._first_sample_done.wait(timeout=min(2.0, max(0.5, self.interval_s)))
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 3))
        if self._errors and not self.rows:
            raise RuntimeError(f"Failed to sample docker stats: {self._errors[0]}") from (
                self._errors[0]
            )
        return list(self.rows)


def check_api_health(host: str, timeout_s: float = 5.0) -> None:
    """Verify ``GET /health`` responds HTTP 200 before launching load traffic."""
    parsed = urlparse(host)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Host URL must use http or https scheme, got {host!r}")
    health_url = f"{host.rstrip('/')}/health"
    req = urllib.request.Request(health_url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # nosec B310
        if resp.status != 200:
            raise RuntimeError(f"API health check returned HTTP {resp.status} at {health_url}")


def build_locust_command(
    spec: ScenarioSpec,
    host: str,
    output_dir: Path,
    locustfile: Path = DEFAULT_LOCUSTFILE,
    use_uv: bool = True,
) -> list[str]:
    """Construct the headless Locust command for a scenario specification."""
    if not spec.locust_scenario:
        raise ValueError(f"Scenario {spec.slug!r} has no Locust profile")
    csv_prefix = output_dir / f"locust_{spec.slug}"
    html_path = output_dir / f"locust_{spec.slug}.html"

    locust_args = [
        "-f",
        str(locustfile),
        "--host",
        host,
        "--scenario",
        spec.locust_scenario,
        "--headless",
        "-u",
        str(spec.users),
        "-r",
        str(spec.spawn_rate),
        "-t",
        f"{spec.duration_s}s",
        "--csv",
        str(csv_prefix),
        "--html",
        str(html_path),
        "--only-summary",
    ]
    if use_uv:
        uv_bin = shutil.which("uv")
        if not uv_bin:
            raise RuntimeError("'uv' executable not found on PATH")
        return [uv_bin, "run", "--group", "perf", "locust", *locust_args]
    locust_bin = shutil.which("locust") or "locust"
    return [locust_bin, *locust_args]


def parse_scenario_selection(
    raw_names: str | None,
    stress_users: int = 120,
    batch_duration_s: int | None = None,
    duration_override_s: int | None = None,
) -> list[ScenarioSpec]:
    """Resolve requested scenario slugs with optional user/duration overrides."""
    if raw_names:
        requested = [item.strip() for item in raw_names.split(",") if item.strip()]
        unknown = [slug for slug in requested if slug not in SCENARIO_MAP]
        if unknown:
            raise ValueError(
                f"Unknown scenario(s): {unknown}. Valid choices: {list(SCENARIO_MAP)}"
            )
        selected = [SCENARIO_MAP[slug] for slug in requested]
    else:
        selected = list(DEFAULT_SCENARIOS)

    resolved: list[ScenarioSpec] = []
    for spec in selected:
        users = stress_users if spec.slug == "stress_120u" else spec.users
        duration = spec.duration_s
        if batch_duration_s is not None and spec.slug.startswith("batch_"):
            duration = batch_duration_s
        if duration_override_s is not None:
            duration = duration_override_s
        resolved.append(
            ScenarioSpec(
                slug=spec.slug,
                duration_s=duration,
                users=users,
                spawn_rate=spec.spawn_rate,
                locust_scenario=spec.locust_scenario,
            )
        )
    return resolved


def run_scenarios(
    scenarios: list[ScenarioSpec],
    host: str,
    output_dir: Path,
    interval_s: float = 1.5,
    engine: str = "auto",
    containers: list[str] | None = None,
    skip_docker_stats: bool = False,
    skip_healthcheck: bool = False,
    locustfile: Path = DEFAULT_LOCUSTFILE,
    use_uv: bool = True,
) -> dict[str, Any]:
    """Execute scenarios sequentially while recording Locust CSVs and container stats."""
    if interval_s <= 0:
        raise ValueError(f"interval_s must be > 0, got {interval_s}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not skip_healthcheck:
        check_api_health(host)

    engine_bin: str | None = None
    target_containers: list[str] = []
    if not skip_docker_stats:
        engine_bin = detect_container_engine(engine)
        target_containers = resolve_target_containers(engine_bin, containers)

    completed: list[str] = []
    for spec in scenarios:
        sampler: DockerStatsSampler | None = None
        if not skip_docker_stats and engine_bin is not None:
            sampler = DockerStatsSampler(
                engine_bin=engine_bin,
                containers=target_containers,
                scenario=spec.slug,
                interval_s=interval_s,
            )
            sampler.start()
        try:
            if spec.locust_scenario is None:
                time.sleep(spec.duration_s)
            else:
                cmd = build_locust_command(
                    spec=spec,
                    host=host,
                    output_dir=output_dir,
                    locustfile=locustfile,
                    use_uv=use_uv,
                )
                proc = subprocess.run(cmd, cwd=PROJ_ROOT, check=False)  # nosec B603
                stats_csv = output_dir / f"locust_{spec.slug}_stats.csv"
                if proc.returncode == 1 and stats_csv.is_file():
                    print(
                        f"[warn] Locust exited with status 1 for {spec.slug} "
                        f"(request failures recorded); continuing with {stats_csv.name}.",
                        file=sys.stderr,
                    )
                elif proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, cmd)
        finally:
            if sampler is not None:
                rows = sampler.stop()
                write_docker_stats_csv(output_dir / f"docker_stats_{spec.slug}.csv", rows)
        completed.append(spec.slug)

    return {
        "completed_scenarios": completed,
        "containers": target_containers,
        "output_dir": str(output_dir),
    }


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _predict_booster(
    booster: Any,
    classes: list[int],
    thresholds: dict[int, float],
    records: list[dict[str, Any]],
    num_threads: int | None = None,
) -> list[dict[str, Any]]:
    matrix = np.asarray(
        [[float(record[col]) for col in FEATURE_COLUMNS] for record in records],
        dtype=np.float64,
    )
    if num_threads is None:
        proba = booster.predict(matrix)
    else:
        proba = booster.predict(matrix, num_threads=num_threads)
    labels = apply_thresholds(proba, classes, thresholds)
    predictions: list[dict[str, Any]] = []
    for label_id, row in zip(labels, proba):
        predictions.append(
            {
                "label_id": int(label_id),
                "label": CLASS_NAMES_ES[label_id],
                "probabilities": {CLASS_NAMES_ES[cls]: float(p) for cls, p in zip(classes, row)},
            }
        )
    return predictions


def _assert_predictions_equivalent(
    baseline_out: list[dict[str, Any]],
    candidate_out: list[dict[str, Any]],
    classes: list[int],
    batch_size: int,
    label: str,
) -> float:
    class_names = [CLASS_NAMES_ES[cls] for cls in classes]
    baseline_labels = [int(item["label_id"]) for item in baseline_out]
    candidate_labels = [int(item["label_id"]) for item in candidate_out]
    baseline_proba = np.asarray(
        [[item["probabilities"][name] for name in class_names] for item in baseline_out],
        dtype=np.float64,
    )
    candidate_proba = np.asarray(
        [[item["probabilities"][name] for name in class_names] for item in candidate_out],
        dtype=np.float64,
    )
    max_prob_diff = float(np.max(np.abs(candidate_proba - baseline_proba)))
    if candidate_labels != baseline_labels or max_prob_diff > 1e-9:
        raise RuntimeError(
            f"{label} mismatch at batch_size={batch_size}: max_prob_diff={max_prob_diff}"
        )
    return max_prob_diff


def run_inference_benchmark(
    model_uri: Path | str = DEFAULT_MODEL_URI,
    batch_sizes: tuple[int, ...] = (1, 20, 365),
    iterations: int = 100,
    warmup: int = 10,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Benchmark ``pyfunc + pandas.DataFrame`` vs direct ``LightGBM Booster + numpy``."""
    import csv

    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError(f"batch_sizes must contain positive integers, got {batch_sizes!r}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    loaded = load_model(str(model_uri))
    raw_model = loaded.model.get_raw_model()
    booster = getattr(raw_model, "booster_", None)
    if booster is None:
        raise RuntimeError(f"Expected LightGBM model with booster_, got {type(raw_model)!r}")

    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        records = build_predict_payload(batch_size=batch_size, seed=100 + batch_size)["instances"]
        baseline_out = predict(loaded, records)

        paths = (
            ("pyfunc_dataframe", lambda recs=records: predict(loaded, recs)),
            (
                "booster_numpy",
                lambda recs=records: _predict_booster(
                    booster, loaded.classes, loaded.thresholds, recs, num_threads=None
                ),
            ),
            (
                "booster_numpy_1thread",
                lambda recs=records: _predict_booster(
                    booster, loaded.classes, loaded.thresholds, recs, num_threads=1
                ),
            ),
        )

        baseline_mean_ms: float | None = None
        for path_name, fn in paths:
            candidate_out = fn()
            max_prob_diff = _assert_predictions_equivalent(
                baseline_out=baseline_out,
                candidate_out=candidate_out,
                classes=loaded.classes,
                batch_size=batch_size,
                label=path_name,
            )
            for _ in range(warmup):
                fn()
            samples_ms: list[float] = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                fn()
                samples_ms.append((time.perf_counter() - t0) * 1000.0)
            mean_ms = statistics.fmean(samples_ms)
            if baseline_mean_ms is None:
                baseline_mean_ms = mean_ms
            p50_ms = _percentile(samples_ms, 50.0)
            p95_ms = _percentile(samples_ms, 95.0)
            pred_per_s = batch_size / (mean_ms / 1000.0)
            speedup = baseline_mean_ms / mean_ms if mean_ms > 0 else 0.0
            results.append(
                {
                    "batch_size": batch_size,
                    "path": path_name,
                    "iterations": iterations,
                    "mean_ms": round(mean_ms, 4),
                    "p50_ms": round(p50_ms, 4),
                    "p95_ms": round(p95_ms, 4),
                    "pred_per_s": round(pred_per_s, 2),
                    "speedup_vs_pyfunc": round(speedup, 2),
                    "max_prob_diff": round(max_prob_diff, 12),
                }
            )

    payload = {
        "generated_at": utc_timestamp(),
        "model_uri": str(model_uri),
        "iterations": iterations,
        "warmup": warmup,
        "results": results,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "opt3_benchmark.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        csv_path = output_dir / "opt3_benchmark.csv"
        fieldnames = list(results[0].keys()) if results else []
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    return payload


def _read_locust_stats(csv_path: Path) -> list[dict[str, str]]:
    import csv

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_docker_stats(csv_path: Path) -> list[dict[str, str]]:
    import csv

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(raw: str | None) -> float:
    cleaned = (raw or "").strip()
    if not cleaned or cleaned in {"--", "N/A", "nan", "NaN"}:
        return 0.0
    return float(cleaned)


def _safe_int(raw: str | None) -> int:
    return int(_safe_float(raw))


def _scenario_sort_key(slug: str) -> tuple[int, str]:
    order = {spec.slug: idx for idx, spec in enumerate(DEFAULT_SCENARIOS)}
    return (order.get(slug, len(order)), slug)


def generate_anexo_b(load_dir: Path, output_path: Path | None = None) -> Path:
    """Generate ``anexo_b.md`` summarizing Locust, ``docker stats``, and benchmark results."""
    target = output_path or (load_dir / "anexo_b.md")
    lines: list[str] = [
        "### Anexo B. Resultados completos de carga y monitoreo de recursos",
        "",
        f"*Generado automáticamente el {utc_timestamp()} desde `{load_dir.as_posix()}`.*",
        "",
    ]

    # Table B.1: Locust summary per scenario & endpoint (excluding Aggregated for error %)
    locust_files = sorted(
        load_dir.glob("locust_*_stats.csv"),
        key=lambda p: _scenario_sort_key(
            p.name.removeprefix("locust_").removesuffix("_stats.csv")
        ),
    )
    lines.append("#### B.1. Métricas de Locust por escenario y endpoint")
    lines.append("")
    if locust_files:
        lines.append(
            "| Escenario | Método | Endpoint | Peticiones | Fallos | Error (%) | "
            "p50 (ms) | p95 (ms) | p99 (ms) | RPS |"
        )
        lines.append("|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|")
        batch_rows: dict[str, dict[str, float]] = {}
        for stats_file in locust_files:
            slug = stats_file.name.removeprefix("locust_").removesuffix("_stats.csv")
            for row in _read_locust_stats(stats_file):
                name = row.get("Name", "")
                if name == "Aggregated":
                    continue
                req_count = _safe_int(row.get("Request Count"))
                fail_count = _safe_int(row.get("Failure Count"))
                err_pct = (100.0 * fail_count / req_count) if req_count > 0 else 0.0
                p50 = _safe_float(row.get("50%"))
                p95 = _safe_float(row.get("95%"))
                p99 = _safe_float(row.get("99%"))
                rps = _safe_float(row.get("Requests/s"))
                method = row.get("Type", "")
                lines.append(
                    f"| `{slug}` | `{method}` | `{name}` | {req_count} | {fail_count} | "
                    f"{err_pct:.2f} | {p50:.1f} | {p95:.1f} | {p99:.1f} | {rps:.2f} |"
                )
                if slug in {"batch_1x365", "batch_365x1"} and name.startswith("/predict"):
                    batch_size = 365 if slug == "batch_1x365" else 1
                    batch_rows[slug] = {
                        "rps": rps,
                        "pred_per_s": rps * batch_size,
                        "p50": p50,
                        "p95": p95,
                    }

        if "batch_1x365" in batch_rows and "batch_365x1" in batch_rows:
            b365 = batch_rows["batch_1x365"]
            b1 = batch_rows["batch_365x1"]
            ratio = b365["pred_per_s"] / b1["pred_per_s"] if b1["pred_per_s"] > 0 else 0.0
            row_b1 = (
                f"| `batch_365x1` | 1 | {b1['rps']:.2f} | {b1['pred_per_s']:.2f} | "
                f"{b1['p50']:.1f} | {b1['p95']:.1f} | 1.00x |"
            )
            row_b365 = (
                f"| `batch_1x365` | 365 | {b365['rps']:.2f} | {b365['pred_per_s']:.2f} | "
                f"{b365['p50']:.1f} | {b365['p95']:.1f} | {ratio:.2f}x |"
            )
            lines.extend(
                [
                    "",
                    "#### B.2. Comparativa de batching (`batch_1x365` vs `batch_365x1`)",
                    "",
                    (
                        "| Escenario | Tamaño lote | RPS HTTP | Predicciones/s | "
                        "p50 (ms) | p95 (ms) | Factor Pred/s |"
                    ),
                    "|:---|---:|---:|---:|---:|---:|---:|",
                    row_b1,
                    row_b365,
                ]
            )
    else:
        lines.append("*No se encontraron archivos `locust_*_stats.csv` en el directorio.*")

    # Table B.3: docker stats summary
    docker_files = sorted(
        load_dir.glob("docker_stats_*.csv"),
        key=lambda p: _scenario_sort_key(
            p.name.removeprefix("docker_stats_").removesuffix(".csv")
        ),
    )
    lines.extend(["", "#### B.3. Consumo de recursos de contenedores (`docker stats`)", ""])
    if docker_files:
        lines.append(
            "| Escenario | Contenedor | Muestras | CPU prom (%) | CPU máx (%) | "
            "RAM prom (MiB) | RAM máx (MiB) | RAM prom (%) | PIDs máx |"
        )
        lines.append("|:---|:---|---:|---:|---:|---:|---:|---:|---:|")
        for stats_file in docker_files:
            grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
            for row in _read_docker_stats(stats_file):
                grouped[(row["scenario"], row["container"])].append(row)
            for (scenario, container), group_rows in sorted(
                grouped.items(), key=lambda item: (_scenario_sort_key(item[0][0]), item[0][1])
            ):
                cpus = [_safe_float(r.get("cpu_pct")) for r in group_rows]
                mems = [_safe_float(r.get("mem_usage_mib")) for r in group_rows]
                mem_pcts = [_safe_float(r.get("mem_pct")) for r in group_rows]
                pids = [_safe_int(r.get("pids")) for r in group_rows]
                lines.append(
                    f"| `{scenario}` | `{container}` | {len(group_rows)} | "
                    f"{statistics.fmean(cpus):.2f} | {max(cpus):.2f} | "
                    f"{statistics.fmean(mems):.2f} | {max(mems):.2f} | "
                    f"{statistics.fmean(mem_pcts):.2f} | {max(pids)} |"
                )
    else:
        lines.append("*No se encontraron archivos `docker_stats_*.csv` en el directorio.*")

    # Table B.4: Opt 3 benchmark summary
    bench_json = load_dir / "opt3_benchmark.json"
    if bench_json.is_file():
        bench_data = json.loads(bench_json.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "#### B.4. Microbenchmark de sobrecarga por petición (Opt 3)",
                "",
                (
                    "| Lote | Ruta de inferencia | Media (ms) | p50 (ms) | "
                    "p95 (ms) | Predicciones/s | Aceleración |"
                ),
                "|---:|:---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in bench_data.get("results", []):
            lines.append(
                f"| {row['batch_size']} | `{row['path']}` | {row['mean_ms']:.4f} | "
                f"{row['p50_ms']:.4f} | {row['p95_ms']:.4f} | {row['pred_per_s']:.2f} | "
                f"{row['speedup_vs_pyfunc']:.2f}x |"
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``collect_load_metrics.py``."""
    parser = argparse.ArgumentParser(
        description="Run Locust scenarios with docker stats capture and inference benchmarks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run load scenarios and capture docker stats.")
    run_p.add_argument("--host", default=DEFAULT_HOST, help="Target API base URL.")
    run_p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV/MD outputs."
    )
    run_p.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated scenario slugs (default: all 6 frozen slugs).",
    )
    run_p.add_argument(
        "--interval", type=float, default=1.5, help="docker stats polling interval (s)."
    )
    run_p.add_argument(
        "--engine",
        choices=("auto", "docker", "podman"),
        default="auto",
        help="Container CLI engine.",
    )
    run_p.add_argument(
        "--containers",
        default=None,
        help="Comma-separated container names (default: auto-detect mibici-api/mlflow).",
    )
    run_p.add_argument(
        "--stress-users",
        type=int,
        default=120,
        help="Virtual users for stress_120u (default: 120).",
    )
    run_p.add_argument(
        "--batch-duration",
        type=int,
        default=None,
        help="Override duration in seconds for batch_1x365 and batch_365x1.",
    )
    run_p.add_argument(
        "--duration-override",
        type=int,
        default=None,
        help="Override duration in seconds for all selected scenarios (useful for smoke tests).",
    )
    run_p.add_argument(
        "--skip-docker-stats", action="store_true", help="Skip docker stats polling."
    )
    run_p.add_argument(
        "--skip-healthcheck", action="store_true", help="Skip initial /health check."
    )
    run_p.add_argument(
        "--no-uv", action="store_true", help="Invoke locust directly instead of uv run."
    )
    run_p.add_argument(
        "--with-benchmark",
        action="store_true",
        help="Also run the Opt 3 inference microbenchmark and include it in anexo_b.md.",
    )
    run_p.add_argument(
        "--model-uri",
        type=Path,
        default=DEFAULT_MODEL_URI,
        help="Model path used when --with-benchmark is passed.",
    )

    bench_p = subparsers.add_parser(
        "benchmark", help="Benchmark pyfunc+DataFrame vs direct LightGBM Booster+numpy."
    )
    bench_p.add_argument("--model-uri", type=Path, default=DEFAULT_MODEL_URI)
    bench_p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    bench_p.add_argument("--iterations", type=int, default=100)
    bench_p.add_argument("--warmup", type=int, default=10)
    bench_p.add_argument(
        "--batch-sizes",
        default="1,20,365",
        help="Comma-separated batch sizes to benchmark.",
    )
    bench_p.add_argument(
        "--update-anexo",
        action="store_true",
        help="Regenerate anexo_b.md after writing benchmark outputs.",
    )

    rep_p = subparsers.add_parser("report", help="Render anexo_b.md from existing load outputs.")
    rep_p.add_argument("--load-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    rep_p.add_argument("--output", type=Path, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        scenarios = parse_scenario_selection(
            args.scenarios,
            stress_users=args.stress_users,
            batch_duration_s=args.batch_duration,
            duration_override_s=args.duration_override,
        )
        containers = (
            [c.strip() for c in args.containers.split(",") if c.strip()]
            if args.containers
            else None
        )
        run_scenarios(
            scenarios=scenarios,
            host=args.host,
            output_dir=args.output_dir,
            interval_s=args.interval,
            engine=args.engine,
            containers=containers,
            skip_docker_stats=args.skip_docker_stats,
            skip_healthcheck=args.skip_healthcheck,
            use_uv=not args.no_uv,
        )
        if args.with_benchmark:
            run_inference_benchmark(model_uri=args.model_uri, output_dir=args.output_dir)
        generate_anexo_b(args.output_dir)
        return 0

    if args.command == "benchmark":
        batch_sizes = tuple(
            int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()
        )
        summary = run_inference_benchmark(
            model_uri=args.model_uri,
            batch_sizes=batch_sizes,
            iterations=args.iterations,
            warmup=args.warmup,
            output_dir=args.output_dir,
        )
        if args.update_anexo:
            generate_anexo_b(args.output_dir)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "report":
        output_path = generate_anexo_b(args.load_dir, args.output)
        print(f"Wrote {output_path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
