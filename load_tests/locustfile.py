"""Locust load-test scenarios for the MiBici P2 station risk API.

Domain justification:
    Guadalajara's MiBici network operates 365 active stations, and ``PredictRequest``
    accepts batches of 1 to 500 station instances per ``POST /predict`` call.
    The default ``mixed`` scenario exercises single-station lookups (``batch=1``),
    zone queries (``batch=20``), full-network scoring (``batch=365``), and metadata/health
    endpoints. The ``batch_1x365`` and ``batch_365x1`` scenarios support the FinOps
    batching comparison (1 request of 365 stations vs. 365 single-station requests).

Usage:
    uv run --group perf locust -f load_tests/locustfile.py --host http://127.0.0.1:8000
    uv run --group perf locust -f load_tests/locustfile.py --host http://127.0.0.1:8000 \\
        --scenario batch_1x365 --headless -u 1 -r 1 -t 60s
"""

from __future__ import annotations

import itertools
import os
import random
from typing import Any

from locust import HttpUser, between, events, task

from load_tests.payloads import MIBICI_ACTIVE_STATIONS, build_payload_pool

VALID_SCENARIOS = ("mixed", "batch_1x365", "batch_365x1")
_JSON_HEADERS = {"Content-Type": "application/json"}
_MIXED_WAIT = between(0.2, 1.0)

# Pre-serialized payload rings avoid client-side JSON overhead at 120+ concurrent users.
_PAYLOAD_POOLS: dict[int, tuple[str, ...]] = {
    1: build_payload_pool(batch_size=1, pool_size=32, base_seed=1000),
    20: build_payload_pool(batch_size=20, pool_size=16, base_seed=2000),
    MIBICI_ACTIVE_STATIONS: build_payload_pool(
        batch_size=MIBICI_ACTIVE_STATIONS, pool_size=8, base_seed=3000
    ),
}
_PAYLOAD_CYCLES = {
    batch_size: itertools.cycle(pool) for batch_size, pool in _PAYLOAD_POOLS.items()
}


@events.init_command_line_parser.add_listener
def add_mibici_cli_arguments(parser: Any) -> None:
    """Register the ``--scenario`` option for headless FinOps / batching runs."""
    parser.add_argument(
        "--scenario",
        type=str,
        choices=VALID_SCENARIOS,
        default=os.environ.get("MIBICI_LOAD_SCENARIO", "mixed"),
        help="MiBici traffic profile: mixed (default), batch_1x365, or batch_365x1.",
    )


def _resolve_scenario(environment: Any) -> str:
    parsed = getattr(environment, "parsed_options", None)
    scenario = getattr(parsed, "scenario", None) or os.environ.get(
        "MIBICI_LOAD_SCENARIO", "mixed"
    )
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"Unsupported scenario {scenario!r}; expected one of {VALID_SCENARIOS}")
    return scenario


@events.init.add_listener
def configure_scenario_tasks(environment: Any, **_kwargs: Any) -> None:
    """Restrict user tasks when running a dedicated FinOps batching comparison."""
    scenario = _resolve_scenario(environment)
    if scenario == "batch_1x365":
        MiBiciApiUser.tasks = [MiBiciApiUser.predict_network]
    elif scenario == "batch_365x1":
        MiBiciApiUser.tasks = [MiBiciApiUser.predict_single]
    else:
        MiBiciApiUser.tasks = [
            MiBiciApiUser.predict_single,
            MiBiciApiUser.predict_single,
            MiBiciApiUser.predict_single,
            MiBiciApiUser.predict_single,
            MiBiciApiUser.predict_single,
            MiBiciApiUser.predict_zone,
            MiBiciApiUser.predict_zone,
            MiBiciApiUser.predict_zone,
            MiBiciApiUser.predict_network,
            MiBiciApiUser.get_model_info,
            MiBiciApiUser.get_health,
        ]


class MiBiciApiUser(HttpUser):
    """Simulated client hitting ``/predict``, ``/model-info``, and ``/health``."""

    def wait_time(self) -> float:
        """Use realistic think time in ``mixed`` mode and zero wait in batching benchmarks."""
        if _resolve_scenario(self.environment) == "mixed":
            return float(_MIXED_WAIT(self))
        return 0.0

    def on_start(self) -> None:
        """Advance RNG slightly per virtual user so users rotate across payload rings."""
        for _ in range(random.randrange(4)):  # nosec B311
            next(_PAYLOAD_CYCLES[1])

    def _post_predict(self, batch_size: int) -> None:
        payload_json = next(_PAYLOAD_CYCLES[batch_size])
        with self.client.post(
            "/predict",
            data=payload_json,
            headers=_JSON_HEADERS,
            name=f"/predict?batch={batch_size}",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return
            try:
                body = resp.json()
            except ValueError as exc:
                resp.failure(f"Invalid JSON response: {exc}")
                return
            predictions = body.get("predictions")
            if not isinstance(predictions, list) or len(predictions) != batch_size:
                actual = len(predictions) if isinstance(predictions, list) else type(predictions)
                resp.failure(f"Expected {batch_size} predictions, got {actual}")
                return
            resp.success()

    @task(5)
    def predict_single(self) -> None:
        """Query risk for a single station (interactive lookup)."""
        self._post_predict(1)

    @task(3)
    def predict_zone(self) -> None:
        """Query risk for a 20-station corridor/zone."""
        self._post_predict(20)

    @task(1)
    def predict_network(self) -> None:
        """Query risk for all 365 active MiBici stations in one call."""
        self._post_predict(MIBICI_ACTIVE_STATIONS)

    @task(1)
    def get_model_info(self) -> None:
        """Read served model version and lineage metadata."""
        with self.client.get("/model-info", name="/model-info", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return
            try:
                body = resp.json()
            except ValueError as exc:
                resp.failure(f"Invalid JSON response: {exc}")
                return
            if not body.get("model_name") and not body.get("source_run_id"):
                resp.failure("Missing model metadata in /model-info")
                return
            resp.success()

    @task(1)
    def get_health(self) -> None:
        """Verify liveness and model readiness."""
        with self.client.get("/health", name="/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return
            try:
                body = resp.json()
            except ValueError as exc:
                resp.failure(f"Invalid JSON response: {exc}")
                return
            if body.get("status") != "ok":
                resp.failure(f"Unexpected /health body: {body!r}")
                return
            resp.success()
