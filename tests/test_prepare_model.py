"""The delivery bundle can be restored safely in a clean checkout."""

from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_prepare_model_clean_checkout_and_refuse_overwrite(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "models").mkdir()
    shutil.copy(ROOT / "scripts/prepare_model.py", tmp_path / "scripts/prepare_model.py")
    shutil.copy(ROOT / "models/p2-champion-v3.zip", tmp_path / "models/p2-champion-v3.zip")
    cmd = [sys.executable, str(tmp_path / "scripts/prepare_model.py")]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    destination = tmp_path / "models/p2-riesgo"
    assert all(
        (destination / name).is_file()
        for name in ("MLmodel", "model.pkl", "registry_metadata.json")
    )
    marker = destination / "keep"
    marker.write_text("do not overwrite")
    refused = subprocess.run(cmd, capture_output=True, text=True)
    assert refused.returncode != 0
    assert marker.read_text() == "do not overwrite"
    subprocess.run([*cmd, "--replace"], check=True, capture_output=True, text=True)
    assert not marker.exists()


def test_prepare_model_rejects_changed_archive(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "models").mkdir()
    shutil.copy(ROOT / "scripts/prepare_model.py", tmp_path / "scripts/prepare_model.py")
    (tmp_path / "models/p2-champion-v3.zip").write_bytes(b"corrupt")
    result = subprocess.run(
        [sys.executable, str(tmp_path / "scripts/prepare_model.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "checksum mismatch" in result.stderr
    assert not (tmp_path / "models/p2-riesgo").exists()
