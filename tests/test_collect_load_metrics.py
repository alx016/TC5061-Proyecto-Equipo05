"""Unit and integration tests for scripts/collect_load_metrics.py."""

import csv
import json
from pathlib import Path
import re
import subprocess
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scripts import collect_load_metrics as clm

ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def test_parse_units_and_docker_stats_normalization():
    stdout = (
        "mibici-api\t45.67%\t150.5MiB / 1GiB\t14.70%\t1.25MB / 500kB\t19\n"
        "mibici-mlflow\t2.5%\t96MiB / 512MiB\t18.75%\t200KiB / 100B\t6\n"
        "mibici-p2-api\t--\t512KiB / 2GB\t--\t-- / --\t--\n"
    )
    rows = clm.parse_docker_stats_output(
        stdout=stdout, scenario="nominal_10u", timestamp="2026-11-18T21:00:00Z"
    )
    assert len(rows) == 3

    api_row, mlflow_row, fallback_row = rows
    assert api_row["container"] == "api"
    assert api_row["cpu_pct"] == pytest.approx(45.67)
    assert api_row["mem_usage_mib"] == pytest.approx(150.5)
    assert api_row["mem_limit_mib"] == pytest.approx(1024.0)
    assert api_row["mem_pct"] == pytest.approx(14.70)
    assert api_row["net_rx_mb"] == pytest.approx(1.25)
    assert api_row["net_tx_mb"] == pytest.approx(0.5)
    assert api_row["pids"] == 19

    assert mlflow_row["container"] == "mlflow"
    assert mlflow_row["mem_limit_mib"] == pytest.approx(512.0)
    assert mlflow_row["net_rx_mb"] == pytest.approx(0.2048)
    assert mlflow_row["net_tx_mb"] == pytest.approx(0.0001)
    assert mlflow_row["pids"] == 6

    assert fallback_row["container"] == "api"
    assert fallback_row["cpu_pct"] == 0.0
    assert fallback_row["mem_usage_mib"] == pytest.approx(0.5)
    assert fallback_row["mem_limit_mib"] == pytest.approx(2_000_000_000 / (1024.0**2), rel=1e-4)
    assert fallback_row["pids"] == 0


def test_utc_timestamp_and_default_stats_timestamp_format():
    ts = clm.utc_timestamp()
    assert ISO_UTC_RE.match(ts)

    rows = clm.parse_docker_stats_output(
        stdout="mibici-api\t5.0%\t100MiB / 1GiB\t9.77%\t10kB / 20kB\t10\n",
        scenario="baseline",
        timestamp=None,
    )
    assert len(rows) == 1
    assert ISO_UTC_RE.match(rows[0]["timestamp"])


def test_write_docker_stats_csv_matches_contract_columns(tmp_path):
    csv_path = tmp_path / "docker_stats_baseline.csv"
    rows = clm.parse_docker_stats_output(
        "mibici-api\t1.0%\t100MiB / 1024MiB\t9.77%\t10kB / 20kB\t11\n",
        scenario="baseline",
        timestamp="2026-11-18T21:00:00Z",
    )
    clm.write_docker_stats_csv(csv_path, rows)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == clm.DOCKER_STATS_COLUMNS
        parsed = list(reader)
    assert len(parsed) == 1
    assert parsed[0]["container"] == "api"
    assert parsed[0]["scenario"] == "baseline"
    assert "%" not in parsed[0]["cpu_pct"]
    assert "MiB" not in parsed[0]["mem_usage_mib"]


def test_resolve_target_containers_prefers_compose_then_fallback(monkeypatch):
    monkeypatch.setattr(
        clm, "list_running_containers", lambda _bin: ["other", "mibici-mlflow", "mibici-api"]
    )
    assert clm.resolve_target_containers("/usr/bin/docker") == ["mibici-api", "mibici-mlflow"]

    monkeypatch.setattr(clm, "list_running_containers", lambda _bin: ["mibici-p2-api"])
    assert clm.resolve_target_containers("/usr/bin/docker") == ["mibici-p2-api"]

    monkeypatch.setattr(clm, "list_running_containers", lambda _bin: [])
    with pytest.raises(RuntimeError, match="No active MiBici containers"):
        clm.resolve_target_containers("/usr/bin/docker")


def test_parse_scenario_selection_defaults_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(clm.shutil, "which", lambda name: f"/usr/bin/{name}")

    all_specs = clm.parse_scenario_selection(None)
    assert [s.slug for s in all_specs] == [
        "baseline",
        "nominal_10u",
        "high_50u",
        "stress_120u",
        "batch_1x365",
        "batch_365x1",
    ]
    assert [s.duration_s for s in all_specs] == [60, 120, 120, 120, 60, 60]
    assert [s.users for s in all_specs] == [0, 10, 50, 120, 1, 1]
    assert [s.spawn_rate for s in all_specs] == [0, 2, 5, 10, 1, 1]

    with pytest.raises(ValueError, match="Unknown scenario"):
        clm.parse_scenario_selection("baseline,does_not_exist")

    specs = clm.parse_scenario_selection(
        "baseline,stress_120u,batch_1x365",
        stress_users=90,
        batch_duration_s=30,
    )
    assert [s.slug for s in specs] == ["baseline", "stress_120u", "batch_1x365"]
    assert specs[1].users == 90
    assert specs[2].duration_s == 30

    cmd = clm.build_locust_command(
        spec=specs[1], host="http://127.0.0.1:8000", output_dir=tmp_path
    )
    assert cmd[:5] == ["/usr/bin/uv", "run", "--group", "perf", "locust"]
    assert "--scenario" in cmd and "mixed" in cmd
    assert str(tmp_path / "locust_stress_120u") in cmd
    assert str(tmp_path / "locust_stress_120u.html") in cmd


def test_check_api_health_validates_scheme_and_status(monkeypatch):
    with pytest.raises(ValueError, match="http or https scheme"):
        clm.check_api_health("ftp://127.0.0.1:8000")

    class FakeResp:
        def __init__(self, status: int) -> None:
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(clm.urllib.request, "urlopen", lambda _req, timeout=5.0: FakeResp(200))
    clm.check_api_health("http://127.0.0.1:8000")

    monkeypatch.setattr(clm.urllib.request, "urlopen", lambda _req, timeout=5.0: FakeResp(503))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        clm.check_api_health("http://127.0.0.1:8000")


def test_run_scenarios_orchestrates_baseline_and_locust_with_sampler(tmp_path, monkeypatch):
    monkeypatch.setattr(clm, "check_api_health", lambda _host: None)
    monkeypatch.setattr(clm, "detect_container_engine", lambda _e: "/usr/bin/docker")
    monkeypatch.setattr(
        clm, "resolve_target_containers", lambda _bin, _c: ["mibici-api", "mibici-mlflow"]
    )
    monkeypatch.setattr(clm.shutil, "which", lambda name: f"/usr/bin/{name}")

    sampled_by_scenario: dict[str, threading.Event] = {
        "baseline": threading.Event(),
        "nominal_10u": threading.Event(),
    }

    def fake_sleep(_seconds: float) -> None:
        assert sampled_by_scenario["baseline"].wait(timeout=2.0)

    monkeypatch.setattr(clm.time, "sleep", fake_sleep)

    def fake_sample(engine_bin, containers, scenario, timestamp=None):
        sampled_by_scenario[scenario].set()
        return clm.parse_docker_stats_output(
            "mibici-api\t10.0%\t128MiB / 1024MiB\t12.5%\t1MB / 1MB\t12\n",
            scenario=scenario,
            timestamp="2026-11-18T21:05:00Z",
        )

    locust_calls = []

    def fake_run(cmd, cwd=None, check=True):
        locust_calls.append(cmd)
        assert sampled_by_scenario["nominal_10u"].wait(timeout=2.0)
        csv_idx = cmd.index("--csv")
        prefix = Path(cmd[csv_idx + 1])
        (prefix.parent / f"{prefix.name}_stats.csv").write_text(
            "Type,Name,Request Count,Failure Count,50%,95%,99%,Requests/s\n"
            "POST,/predict?batch=1,10,0,8,12,15,5.0\n"
            ",Aggregated,10,0,8,12,15,5.0\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(clm, "sample_docker_stats_once", fake_sample)
    monkeypatch.setattr(clm.subprocess, "run", fake_run)

    specs = clm.parse_scenario_selection("baseline,nominal_10u", duration_override_s=0)
    summary = clm.run_scenarios(
        scenarios=specs, host="http://127.0.0.1:8000", output_dir=tmp_path, interval_s=0.05
    )

    assert summary["completed_scenarios"] == ["baseline", "nominal_10u"]
    assert len(locust_calls) == 1

    baseline_csv = tmp_path / "docker_stats_baseline.csv"
    nominal_csv = tmp_path / "docker_stats_nominal_10u.csv"
    assert baseline_csv.is_file()
    assert nominal_csv.is_file()
    with baseline_csv.open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) >= 1
    with nominal_csv.open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) >= 1

    anexo = clm.generate_anexo_b(tmp_path)
    anexo_text = anexo.read_text(encoding="utf-8")
    assert "nominal_10u" in anexo_text
    assert "/predict?batch=1" in anexo_text
    assert "baseline" in anexo_text


def test_generate_anexo_b_renders_ordering_errors_na_and_benchmark(tmp_path):
    # Write scenarios out of canonical order to verify deterministic ordering.
    (tmp_path / "locust_stress_120u_stats.csv").write_text(
        "Type,Name,Request Count,Failure Count,50%,95%,99%,Requests/s\n"
        "POST,/predict?batch=1,200,10,N/A,--,120,25.0\n"
        ",Aggregated,999,99,777,888,999,999.0\n",
        encoding="utf-8",
    )
    (tmp_path / "locust_nominal_10u_stats.csv").write_text(
        "Type,Name,Request Count,Failure Count,50%,95%,99%,Requests/s\n"
        "GET,/health,50,0,2,4,5,5.0\n"
        ",Aggregated,50,0,2,4,5,5.0\n",
        encoding="utf-8",
    )
    (tmp_path / "locust_high_50u_stats.csv").write_text(
        "Type,Name,Request Count,Failure Count,50%,95%,99%,Requests/s\n"
        "GET,/health,80,0,3,6,8,10.0\n"
        ",Aggregated,80,0,3,6,8,10.0\n",
        encoding="utf-8",
    )
    (tmp_path / "locust_batch_365x1_stats.csv").write_text(
        "Type,Name,Request Count,Failure Count,50%,95%,99%,Requests/s\n"
        "POST,/predict?batch=1,200,0,8,10,12,100.0\n"
        ",Aggregated,200,0,8,10,12,100.0\n",
        encoding="utf-8",
    )
    (tmp_path / "locust_batch_1x365_stats.csv").write_text(
        "Type,Name,Request Count,Failure Count,50%,95%,99%,Requests/s\n"
        "POST,/predict?batch=365,100,0,22,26,30,40.0\n"
        ",Aggregated,100,0,22,26,30,40.0\n",
        encoding="utf-8",
    )

    for slug in ("batch_1x365", "baseline"):
        (tmp_path / f"docker_stats_{slug}.csv").write_text(
            "timestamp,scenario,container,cpu_pct,mem_usage_mib,mem_limit_mib,"
            "mem_pct,net_rx_mb,net_tx_mb,pids\n"
            f"2026-11-18T21:00:00Z,{slug},api,85.0,210.0,1024.0,20.5,5.0,2.0,18\n"
            f"2026-11-18T21:00:00Z,{slug},mlflow,1.2,98.0,512.0,19.1,0.1,0.1,6\n",
            encoding="utf-8",
        )

    (tmp_path / "opt3_benchmark.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "batch_size": 1,
                        "path": "pyfunc_dataframe",
                        "mean_ms": 2.0,
                        "p50_ms": 1.9,
                        "p95_ms": 2.4,
                        "pred_per_s": 500.0,
                        "speedup_vs_pyfunc": 1.0,
                    },
                    {
                        "batch_size": 1,
                        "path": "booster_numpy",
                        "mean_ms": 0.1,
                        "p50_ms": 0.09,
                        "p95_ms": 0.12,
                        "pred_per_s": 10000.0,
                        "speedup_vs_pyfunc": 20.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    out = clm.generate_anexo_b(tmp_path)
    text = out.read_text(encoding="utf-8")

    # Aggregated row must be excluded and N/A percentiles safely rendered as 0.0.
    assert "999.00" not in text
    assert "Aggregated" not in text
    assert "| `stress_120u` | `POST` | `/predict?batch=1` | 200 | 10 | 5.00 | 0.0 | 0.0 |" in text

    # Canonical B.1 scenario ordering.
    b1_section = text.split("#### B.2.")[0]
    assert (
        b1_section.index("`nominal_10u`")
        < b1_section.index("`high_50u`")
        < b1_section.index("`stress_120u`")
        < b1_section.index("`batch_1x365`")
        < b1_section.index("`batch_365x1`")
    )

    # Canonical B.3 scenario ordering: baseline before batch_1x365.
    b3_section = text.split("#### B.3.")[1].split("#### B.4.")[0]
    assert b3_section.index("`baseline`") < b3_section.index("`batch_1x365`")

    assert "Comparativa de batching" in text
    assert "14600.00" in text  # 40.0 * 365 predictions/s
    assert "146.00x" in text  # 14600 / 100
    assert "`mlflow`" in text
    assert "booster_numpy" in text


def test_predict_booster_equivalence_num_threads_and_mismatch_detection(tmp_path, monkeypatch):
    seen_num_threads = []

    class DummyBooster:
        def __init__(self, shift: float = 0.0) -> None:
            self.shift = shift

        def predict(self, X, num_threads=None):
            seen_num_threads.append(num_threads)
            row = np.array([0.7 - self.shift, 0.2 + self.shift, 0.1], dtype=np.float64)
            return np.tile(row, (len(X), 1))

    class DummyPyFunc:
        def __init__(self, shift: float = 0.0) -> None:
            self.shift = shift

        def get_raw_model(self):
            return SimpleNamespace(booster_=DummyBooster(shift=self.shift))

        def predict(self, df):
            return np.tile([0.7, 0.2, 0.1], (len(df), 1))

    fake_loaded = SimpleNamespace(
        info={"thresholds": {"shortage": 0.2, "saturation": 0.95}},
        model=DummyPyFunc(),
        classes=[0, 1, 2],
        thresholds={1: 0.2, 2: 0.95},
    )

    records = clm.build_predict_payload(batch_size=2, seed=42)["instances"]
    booster = fake_loaded.model.get_raw_model().booster_
    out = clm._predict_booster(booster, [0, 1, 2], {1: 0.2, 2: 0.95}, records, num_threads=1)
    assert len(out) == 2
    assert set(out[0].keys()) == {"label_id", "label", "probabilities"}
    assert set(out[0]["probabilities"].keys()) == {
        "Balanceado",
        "Riesgo_Desabasto",
        "Riesgo_Saturacion",
    }
    assert seen_num_threads == [1]

    seen_num_threads.clear()
    monkeypatch.setattr(clm, "load_model", lambda _uri: fake_loaded)
    summary = clm.run_inference_benchmark(
        model_uri=tmp_path / "dummy",
        batch_sizes=(1,),
        iterations=2,
        warmup=1,
        output_dir=tmp_path,
    )
    assert len(summary["results"]) == 3
    assert None in seen_num_threads and 1 in seen_num_threads

    mismatched_loaded = SimpleNamespace(
        info={"thresholds": {"shortage": 0.2, "saturation": 0.95}},
        model=DummyPyFunc(shift=0.05),
        classes=[0, 1, 2],
        thresholds={1: 0.2, 2: 0.95},
    )
    monkeypatch.setattr(clm, "load_model", lambda _uri: mismatched_loaded)
    with pytest.raises(RuntimeError, match="mismatch"):
        clm.run_inference_benchmark(
            model_uri=tmp_path / "mismatch",
            batch_sizes=(1,),
            iterations=1,
            warmup=0,
            output_dir=None,
        )


def test_run_inference_benchmark_with_real_export_if_present(tmp_path):
    bundle = clm.DEFAULT_MODEL_URI
    if not (bundle / "MLmodel").is_file():
        pytest.skip("Local model export not present")

    summary = clm.run_inference_benchmark(
        model_uri=bundle,
        batch_sizes=(1, 20),
        iterations=5,
        warmup=1,
        output_dir=tmp_path,
    )
    assert len(summary["results"]) == 6
    assert all(row["max_prob_diff"] == pytest.approx(0.0) for row in summary["results"])
    assert (tmp_path / "opt3_benchmark.json").is_file()
    assert (tmp_path / "opt3_benchmark.csv").is_file()


def test_cli_main_subcommands(tmp_path, monkeypatch, capsys):
    # report subcommand
    rc = clm.main(["report", "--load-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "anexo_b.md").is_file()

    # benchmark subcommand
    monkeypatch.setattr(
        clm,
        "run_inference_benchmark",
        lambda **_kwargs: {
            "generated_at": "2026-11-18T21:00:00Z",
            "model_uri": "models/p2-riesgo",
            "iterations": 2,
            "warmup": 1,
            "results": [],
        },
    )
    rc = clm.main(
        [
            "benchmark",
            "--output-dir",
            str(tmp_path),
            "--batch-sizes",
            "1,20",
            "--iterations",
            "2",
            "--warmup",
            "1",
            "--update-anexo",
        ]
    )
    assert rc == 0
    assert '"iterations": 2' in capsys.readouterr().out

    # run subcommand with --with-benchmark
    called = {"run": False, "bench": False}

    def fake_run_scenarios(**_kwargs):
        called["run"] = True
        return {"completed_scenarios": ["baseline"]}

    def fake_bench(**_kwargs):
        called["bench"] = True
        return {"results": []}

    monkeypatch.setattr(clm, "run_scenarios", fake_run_scenarios)
    monkeypatch.setattr(clm, "run_inference_benchmark", fake_bench)

    rc = clm.main(
        [
            "run",
            "--scenarios",
            "baseline",
            "--duration-override",
            "0",
            "--output-dir",
            str(tmp_path),
            "--skip-docker-stats",
            "--skip-healthcheck",
            "--with-benchmark",
        ]
    )
    assert rc == 0
    assert called == {"run": True, "bench": True}


def test_detect_engine_list_containers_and_sample_docker_stats_once(monkeypatch):
    monkeypatch.setattr(
        clm.shutil,
        "which",
        lambda name: "/usr/bin/podman" if name == "podman" else None,
    )
    assert clm.detect_container_engine("auto") == "/usr/bin/podman"
    with pytest.raises(RuntimeError, match="Requested container engine"):
        clm.detect_container_engine("docker")

    monkeypatch.setattr(clm.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="Neither 'docker' nor 'podman'"):
        clm.detect_container_engine("auto")

    seen_cmds = []

    def fake_run(cmd, capture_output=True, text=True, check=True):
        seen_cmds.append(cmd)
        if cmd[1] == "ps":
            return SimpleNamespace(stdout="mibici-api\nmibici-mlflow\n")
        return SimpleNamespace(stdout="mibici-api\t12.0%\t128MiB / 1GiB\t12.5%\t1MB / 500kB\t14\n")

    monkeypatch.setattr(clm.subprocess, "run", fake_run)
    assert clm.list_running_containers("/usr/bin/docker") == ["mibici-api", "mibici-mlflow"]
    rows = clm.sample_docker_stats_once(
        "/usr/bin/docker",
        ["mibici-api"],
        scenario="baseline",
        timestamp="2026-11-18T21:00:00Z",
    )
    assert len(rows) == 1
    assert rows[0]["container"] == "api"
    assert seen_cmds[1][:4] == ["/usr/bin/docker", "stats", "--no-stream", "--format"]


def test_parse_errors_and_build_locust_command_edge_cases(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Expected 6 tab-separated fields"):
        clm.parse_docker_stats_output("mibici-api\t10%\t100MiB\n", scenario="baseline")
    with pytest.raises(ValueError, match="Unrecognized size string"):
        clm.parse_size_to_bytes("not-a-size")
    with pytest.raises(ValueError, match="Unsupported size unit"):
        clm.parse_size_to_bytes("10XB")

    baseline_spec = clm.SCENARIO_MAP["baseline"]
    with pytest.raises(ValueError, match="has no Locust profile"):
        clm.build_locust_command(baseline_spec, host="http://127.0.0.1:8000", output_dir=tmp_path)

    nominal_spec = clm.SCENARIO_MAP["nominal_10u"]
    monkeypatch.setattr(clm.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="'uv' executable not found"):
        clm.build_locust_command(
            nominal_spec, host="http://127.0.0.1:8000", output_dir=tmp_path, use_uv=True
        )
    direct_cmd = clm.build_locust_command(
        nominal_spec, host="http://127.0.0.1:8000", output_dir=tmp_path, use_uv=False
    )
    assert direct_cmd[0] == "locust"


def test_docker_stats_sampler_errors_and_zero_duration_guarantee(monkeypatch):
    with pytest.raises(ValueError, match="interval_s must be > 0"):
        clm.DockerStatsSampler("/usr/bin/docker", ["mibici-api"], "baseline", interval_s=0)

    monkeypatch.setattr(
        clm,
        "sample_docker_stats_once",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("daemon down")),
    )
    failing_sampler = clm.DockerStatsSampler(
        "/usr/bin/docker", ["mibici-api"], "baseline", interval_s=0.05
    )
    failing_sampler.start()
    with pytest.raises(RuntimeError, match="Failed to sample docker stats"):
        failing_sampler.stop()

    monkeypatch.setattr(
        clm,
        "sample_docker_stats_once",
        lambda **_kw: clm.parse_docker_stats_output(
            "mibici-api\t2.0%\t100MiB / 1GiB\t9.77%\t10kB / 10kB\t8\n",
            scenario="baseline",
            timestamp="2026-11-18T21:00:00Z",
        ),
    )
    fast_sampler = clm.DockerStatsSampler(
        "/usr/bin/docker", ["mibici-api"], "baseline", interval_s=1.0
    )
    fast_sampler.start()
    rows = fast_sampler.stop()
    assert len(rows) >= 1


def test_run_scenarios_handles_locust_exit_code_1_and_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(clm, "check_api_health", lambda _host: None)
    monkeypatch.setattr(clm.shutil, "which", lambda name: f"/usr/bin/{name}")

    with pytest.raises(ValueError, match="interval_s must be > 0"):
        clm.run_scenarios([], host="http://127.0.0.1:8000", output_dir=tmp_path, interval_s=0)

    def fake_run_exit1_with_csv(cmd, cwd=None, check=False):
        prefix = Path(cmd[cmd.index("--csv") + 1])
        (prefix.parent / f"{prefix.name}_stats.csv").write_text(
            "Type,Name,Request Count,Failure Count,50%,95%,99%,Requests/s\n"
            "POST,/predict?batch=1,100,5,10,20,30,10.0\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(clm.subprocess, "run", fake_run_exit1_with_csv)
    specs = clm.parse_scenario_selection("stress_120u,batch_1x365", duration_override_s=1)
    summary = clm.run_scenarios(
        scenarios=specs,
        host="http://127.0.0.1:8000",
        output_dir=tmp_path,
        skip_docker_stats=True,
    )
    assert summary["completed_scenarios"] == ["stress_120u", "batch_1x365"]

    # Without a stats CSV (e.g. Locust CLI failure), CalledProcessError must be raised.
    empty_dir = tmp_path / "fail_case"
    monkeypatch.setattr(
        clm.subprocess, "run", lambda _cmd, cwd=None, check=False: SimpleNamespace(returncode=2)
    )
    with pytest.raises(subprocess.CalledProcessError):
        clm.run_scenarios(
            scenarios=clm.parse_scenario_selection("nominal_10u"),
            host="http://127.0.0.1:8000",
            output_dir=empty_dir,
            skip_docker_stats=True,
        )


def test_generate_anexo_b_empty_dir_and_table_b3_na_values(tmp_path):
    empty_out = clm.generate_anexo_b(tmp_path)
    empty_text = empty_out.read_text(encoding="utf-8")
    assert "No se encontraron archivos `locust_*_stats.csv`" in empty_text
    assert "No se encontraron archivos `docker_stats_*.csv`" in empty_text

    (tmp_path / "docker_stats_nominal_10u.csv").write_text(
        "timestamp,scenario,container,cpu_pct,mem_usage_mib,mem_limit_mib,"
        "mem_pct,net_rx_mb,net_tx_mb,pids\n"
        "2026-11-18T21:00:00Z,nominal_10u,api,N/A,--,1024.0,N/A,1.0,1.0,--\n",
        encoding="utf-8",
    )
    b3_text = clm.generate_anexo_b(tmp_path).read_text(encoding="utf-8")
    assert "| `nominal_10u` | `api` | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |" in b3_text


def test_benchmark_validation_and_missing_booster_and_prob_diff_errors(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="batch_sizes"):
        clm.run_inference_benchmark(batch_sizes=())
    with pytest.raises(ValueError, match="iterations"):
        clm.run_inference_benchmark(iterations=0)
    with pytest.raises(ValueError, match="warmup"):
        clm.run_inference_benchmark(warmup=-1)

    no_booster_loaded = SimpleNamespace(
        model=SimpleNamespace(get_raw_model=lambda: SimpleNamespace()),
        classes=[0, 1, 2],
        thresholds={1: 0.2, 2: 0.95},
    )
    monkeypatch.setattr(clm, "load_model", lambda _uri: no_booster_loaded)
    with pytest.raises(RuntimeError, match="booster_"):
        clm.run_inference_benchmark(model_uri=tmp_path / "no-booster")

    # Small probability drift (1e-4 > 1e-9) keeping identical labels must still fail.
    class DriftingBooster:
        def predict(self, X, num_threads=None):
            return np.tile([0.7 - 1e-4, 0.2 + 1e-4, 0.1], (len(X), 1))

    class BaselinePyFunc:
        def get_raw_model(self):
            return SimpleNamespace(booster_=DriftingBooster())

        def predict(self, df):
            return np.tile([0.7, 0.2, 0.1], (len(df), 1))

    drift_loaded = SimpleNamespace(
        model=BaselinePyFunc(),
        classes=[0, 1, 2],
        thresholds={1: 0.2, 2: 0.95},
    )
    monkeypatch.setattr(clm, "load_model", lambda _uri: drift_loaded)
    with pytest.raises(RuntimeError, match="mismatch"):
        clm.run_inference_benchmark(
            model_uri=tmp_path / "drift",
            batch_sizes=(1,),
            iterations=1,
            warmup=0,
        )
