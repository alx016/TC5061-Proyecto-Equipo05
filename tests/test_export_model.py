"""Unit tests for the P2 champion export; no registry or real model is touched."""

from datetime import datetime
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import export_model

MODEL_NAME = "mibici-equipo05-p2-riesgo"
ALIAS = "champion"


def fake_version(version: str = "3", run_id: str | None = "run-123") -> SimpleNamespace:
    return SimpleNamespace(version=version, run_id=run_id)


def fake_run(*, tags: dict | None = None, params: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            tags=tags
            if tags is not None
            else {
                "git_commit": "abc123",
                "git_dirty": "false",
            },
            params=params
            if params is not None
            else {
                "data_version": "data456",
                "threshold_shortage": "0.2",
                "threshold_saturation": "0.95",
            },
        )
    )


class FakeClient:
    def __init__(
        self,
        version: SimpleNamespace | None = None,
        run: SimpleNamespace | None = None,
        download_uri: str = "/tmp/model",
    ):
        self.version = version or fake_version()
        self.run = run or fake_run()
        self.download_uri = download_uri
        self.alias_calls: list[tuple[str, str]] = []
        self.run_calls: list[str] = []
        self.download_uri_calls: list[tuple[str, str]] = []

    def get_model_version_by_alias(self, model_name: str, alias: str) -> SimpleNamespace:
        self.alias_calls.append((model_name, alias))
        return self.version

    def get_run(self, run_id: str) -> SimpleNamespace:
        self.run_calls.append(run_id)
        return self.run

    def get_model_version_download_uri(self, model_name: str, version: str) -> str:
        self.download_uri_calls.append((model_name, version))
        return self.download_uri


def fake_download(
    calls: list[str],
    *,
    complete: bool = True,
    tracking_uris: list[str | None] | None = None,
):
    """Write the files a real MLflow download would leave in ``dst_path``."""

    def download(artifact_uri: str, dst_path: str, tracking_uri: str | None = None) -> None:
        calls.append(artifact_uri)
        if tracking_uris is not None:
            tracking_uris.append(tracking_uri)
        path = Path(dst_path)
        path.mkdir(parents=True, exist_ok=True)
        if complete:
            (path / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
        (path / "model.pkl").write_bytes(b"fake model")

    return download


def patch_download(
    monkeypatch,
    calls: list[str],
    *,
    complete: bool = True,
    tracking_uris: list[str | None] | None = None,
) -> None:
    """Resolve MLflow's lazy module once, then replace its downloader."""
    artifacts = importlib.import_module("mlflow.artifacts")
    monkeypatch.setattr(
        artifacts,
        "download_artifacts",
        fake_download(calls, complete=complete, tracking_uris=tracking_uris),
    )


def patch_load(monkeypatch, loader) -> None:
    """Resolve MLflow's lazy module once, then replace its loader."""
    pyfunc = importlib.import_module("mlflow.pyfunc")
    monkeypatch.setattr(pyfunc, "load_model", loader)


def test_build_metadata_follows_the_api_contract():
    client = FakeClient()

    metadata = export_model.build_metadata(client, MODEL_NAME, ALIAS)

    assert client.alias_calls == [(MODEL_NAME, ALIAS)]
    assert client.run_calls == ["run-123"]
    assert {key: value for key, value in metadata.items() if key != "exported_at"} == {
        "model_name": MODEL_NAME,
        "model_version": "3",
        "model_alias": ALIAS,
        "source_run_id": "run-123",
        "git_commit": "abc123",
        "data_version": "data456",
        "thresholds": {"shortage": 0.2, "saturation": 0.95},
    }
    assert metadata["exported_at"].endswith("Z")
    datetime.strptime(metadata["exported_at"], "%Y-%m-%dT%H:%M:%SZ")


def test_build_metadata_requires_a_source_run():
    client = FakeClient(version=fake_version(run_id=None))

    with pytest.raises(ValueError, match="source run"):
        export_model.build_metadata(client, MODEL_NAME, ALIAS)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("tags", "git_commit"),
        ("params", "data_version"),
        ("params", "threshold_shortage"),
        ("params", "threshold_saturation"),
    ],
)
def test_build_metadata_names_missing_lineage(section: str, key: str):
    run = fake_run()
    getattr(run.data, section).pop(key)

    with pytest.raises(ValueError, match=key):
        export_model.build_metadata(FakeClient(run=run), MODEL_NAME, ALIAS)


def test_build_metadata_rejects_a_non_numeric_threshold():
    run = fake_run(
        params={
            "data_version": "data456",
            "threshold_shortage": "high",
            "threshold_saturation": "0.95",
        }
    )

    with pytest.raises(ValueError, match="threshold_shortage"):
        export_model.build_metadata(FakeClient(run=run), MODEL_NAME, ALIAS)


def test_export_model_downloads_the_concrete_version_and_writes_metadata(tmp_path, monkeypatch):
    output_dir = tmp_path / "nested" / "p2-riesgo"
    downloads: list[str] = []
    loaded: list[str] = []
    patch_download(monkeypatch, downloads)
    patch_load(monkeypatch, lambda uri: loaded.append(uri) or object())

    metadata = export_model.export_model(MODEL_NAME, ALIAS, output_dir, client=FakeClient())

    assert downloads == [f"models:/{MODEL_NAME}/3"]
    assert len(loaded) == 1
    assert (output_dir / "MLmodel").is_file()
    assert (output_dir / "model.pkl").is_file()
    assert json.loads((output_dir / export_model.METADATA_FILENAME).read_text()) == metadata
    assert not (output_dir.parent / ".p2-riesgo.backup").exists()


def test_export_model_downloads_proxied_artifacts_through_the_server(tmp_path, monkeypatch):
    monkeypatch.setattr(export_model.mlflow, "get_tracking_uri", lambda: "http://127.0.0.1:5001")
    output_dir = tmp_path / "p2-riesgo"
    downloads: list[str] = []
    tracking_uris: list[str | None] = []
    patch_download(monkeypatch, downloads, tracking_uris=tracking_uris)
    patch_load(monkeypatch, lambda uri: object())
    client = FakeClient(download_uri="mlflow-artifacts:/1/models/m-x/artifacts")

    export_model.export_model(MODEL_NAME, ALIAS, output_dir, client=client)

    assert client.download_uri_calls == [(MODEL_NAME, "3")]
    assert downloads == [f"models:/{MODEL_NAME}/3"]
    assert tracking_uris == ["http://127.0.0.1:5001"]
    assert (output_dir / export_model.METADATA_FILENAME).is_file()


def test_export_model_uses_the_dotenv_server_for_proxied_artifacts(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MLFLOW_TRACKING_URI=http://127.0.0.1:5001\n", encoding="utf-8")
    monkeypatch.setattr(export_model, "PROJ_ROOT", tmp_path)
    monkeypatch.setattr(export_model.mlflow, "get_tracking_uri", lambda: "sqlite:///mlflow.db")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    output_dir = tmp_path / "p2-riesgo"
    downloads: list[str] = []
    tracking_uris: list[str | None] = []
    patch_download(monkeypatch, downloads, tracking_uris=tracking_uris)
    patch_load(monkeypatch, lambda uri: object())
    client = FakeClient(download_uri="mlflow-artifacts:/1/models/m-x/artifacts")

    export_model.export_model(MODEL_NAME, ALIAS, output_dir, client=client)

    assert tracking_uris == ["http://127.0.0.1:5001"]
    assert (output_dir / export_model.METADATA_FILENAME).is_file()


def test_export_model_rejects_proxied_artifacts_without_a_server_uri(tmp_path, monkeypatch):
    monkeypatch.setattr(export_model, "PROJ_ROOT", tmp_path)  # no .env in this project
    monkeypatch.setattr(export_model.mlflow, "get_tracking_uri", lambda: "sqlite:///mlflow.db")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    output_dir = tmp_path / "p2-riesgo"
    downloads: list[str] = []
    patch_download(monkeypatch, downloads)
    patch_load(monkeypatch, lambda uri: object())
    client = FakeClient(download_uri="mlflow-artifacts:/1/models/m-x/artifacts")

    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        export_model.export_model(MODEL_NAME, ALIAS, output_dir, client=client)

    assert downloads == []
    assert not output_dir.exists()


def test_an_exported_sqlite_uri_does_not_fall_back_to_the_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MLFLOW_TRACKING_URI=http://127.0.0.1:5001\n", encoding="utf-8")
    monkeypatch.setattr(export_model, "PROJ_ROOT", tmp_path)
    monkeypatch.setattr(export_model.mlflow, "get_tracking_uri", lambda: "sqlite:///mlflow.db")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    output_dir = tmp_path / "p2-riesgo"
    patch_download(monkeypatch, [])
    patch_load(monkeypatch, lambda uri: object())
    client = FakeClient(download_uri="mlflow-artifacts:/1/models/m-x/artifacts")

    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        export_model.export_model(MODEL_NAME, ALIAS, output_dir, client=client)


def test_export_model_replaces_an_existing_export(tmp_path, monkeypatch):
    output_dir = tmp_path / "p2-riesgo"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old", encoding="utf-8")
    patch_download(monkeypatch, [])
    patch_load(monkeypatch, lambda uri: object())

    export_model.export_model(MODEL_NAME, ALIAS, output_dir, client=FakeClient())

    assert not (output_dir / "old.txt").exists()
    assert (output_dir / export_model.METADATA_FILENAME).is_file()
    assert not (output_dir.parent / ".p2-riesgo.backup").exists()


def test_incomplete_download_preserves_the_previous_export(tmp_path, monkeypatch):
    output_dir = tmp_path / "p2-riesgo"
    output_dir.mkdir()
    sentinel = output_dir / "old.txt"
    sentinel.write_text("keep", encoding="utf-8")
    patch_download(monkeypatch, [], complete=False)
    patch_load(monkeypatch, lambda uri: object())

    with pytest.raises(ValueError, match="MLmodel"):
        export_model.export_model(MODEL_NAME, ALIAS, output_dir, client=FakeClient())

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (output_dir / export_model.METADATA_FILENAME).exists()


def test_unloadable_download_preserves_the_previous_export(tmp_path, monkeypatch):
    output_dir = tmp_path / "p2-riesgo"
    output_dir.mkdir()
    sentinel = output_dir / "old.txt"
    sentinel.write_text("keep", encoding="utf-8")
    patch_download(monkeypatch, [])

    def corrupt_model(uri: str):
        raise RuntimeError(f"cannot load {uri}")

    patch_load(monkeypatch, corrupt_model)

    with pytest.raises(RuntimeError, match="cannot load"):
        export_model.export_model(MODEL_NAME, ALIAS, output_dir, client=FakeClient())

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (output_dir / export_model.METADATA_FILENAME).exists()


def test_failed_swap_restores_the_previous_export(tmp_path, monkeypatch):
    output_dir = tmp_path / "p2-riesgo"
    output_dir.mkdir()
    sentinel = output_dir / "old.txt"
    sentinel.write_text("keep", encoding="utf-8")
    patch_download(monkeypatch, [])
    patch_load(monkeypatch, lambda uri: object())

    real_move = export_model.shutil.move
    moves: list[tuple[str, str]] = []

    def fail_on_second_move(source, destination):
        moves.append((str(source), str(destination)))
        if len(moves) == 2:
            raise OSError("disk full")
        return real_move(source, destination)

    monkeypatch.setattr(export_model.shutil, "move", fail_on_second_move)

    with pytest.raises(OSError, match="disk full"):
        export_model.export_model(MODEL_NAME, ALIAS, output_dir, client=FakeClient())

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (output_dir.parent / ".p2-riesgo.backup").exists()
