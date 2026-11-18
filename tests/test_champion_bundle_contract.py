"""Contract test for packaged champion model bundle archive integrity."""

from pathlib import Path
import time
import zipfile


def test_champion_bundle_archive_exists_and_uncorrupted():
    root = Path(__file__).resolve().parents[1]
    archive_path = root / "models" / "p2-champion-v3.zip"

    assert archive_path.is_file(), f"Expected archive to exist at {archive_path}"

    start_time = time.perf_counter()
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        corrupted_file = archive.testzip()
        assert corrupted_file is None, f"Corrupted file found in archive: {corrupted_file}"

        namelist = set(archive.namelist())
        assert "models/p2-riesgo/registry_metadata.json" in namelist
        assert "models/p2-riesgo/MLmodel" in namelist

    elapsed = time.perf_counter() - start_time
    assert elapsed < 0.100, f"Archive inspection took {elapsed:.4f}s, expected < 0.100s"
