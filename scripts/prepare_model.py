"""Restore the checked-in registry export without requiring a running MLflow server."""

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "models/p2-champion-v3.zip"
SHA256 = "73cab932d0eae7dbf2c4f78fe927610353a9184e86588d0eddf4cfcef23b305b"
DESTINATION = ROOT / "models/p2-riesgo"
PREFIX = "models/p2-riesgo/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="replace an existing export")
    args = parser.parse_args()
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != SHA256:
        raise SystemExit("Model archive checksum mismatch; refusing to extract")
    if DESTINATION.exists() and not args.replace:
        raise SystemExit("models/p2-riesgo already exists; use --replace to replace it")
    with tempfile.TemporaryDirectory(dir=DESTINATION.parent) as tmp:
        staged = Path(tmp) / "model"
        staged.mkdir()
        with ZipFile(ARCHIVE) as archive:
            for member in archive.infolist():
                if not member.filename.startswith(PREFIX) or member.is_dir():
                    raise SystemExit(f"Unexpected archive member: {member.filename}")
                relative = Path(member.filename.removeprefix(PREFIX))
                if relative.is_absolute() or ".." in relative.parts:
                    raise SystemExit(f"Unsafe archive member: {member.filename}")
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        if not all(
            (staged / name).is_file()
            for name in ("MLmodel", "model.pkl", "registry_metadata.json")
        ):
            raise SystemExit("Archive is missing required model files")
        backup = Path(tmp) / "previous"
        if DESTINATION.exists():
            DESTINATION.rename(backup)
        try:
            staged.rename(DESTINATION)
        except OSError:
            if backup.exists():
                backup.rename(DESTINATION)
            raise
    print(f"Prepared {DESTINATION} from verified champion v3 archive")


if __name__ == "__main__":
    main()
