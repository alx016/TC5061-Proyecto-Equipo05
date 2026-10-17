from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from loguru import logger
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import typer

from tc5061_proyecto_equipo5.config import (
    DATA_MANIFEST_PATH,
    END_YEAR,
    EXTERNAL_DATA_DIR,
    INITIAL_YEAR,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()

BASE_URL = "https://www.mibici.net"
PAGE_URL = f"{BASE_URL}/es/datos-abiertos/"
CSV_PATTERN = re.compile(
    r"/site/assets/files/\d+/datos_abiertos_(?P<year>\d{4})_(?P<month>\d{2})\.csv"
)
# Station catalog (id, name, coordinates, capacity); the site only lists the latest version
STATIONS_PATTERN = re.compile(
    r"/site/assets/files/\d+/nomenclatura_(?P<year>\d{4})_(?P<month>\d{2})\.csv"
)
TRIPS_HEADER_FIELDS = (b"Viaje_Id", b"Inicio_del_viaje")
STATIONS_HEADER_FIELDS = (b'"id"', b'"latitude"', b'"longitude"')
# The site rejects requests that do not look like they come from a browser
CURL_ARGS = [
    "curl", "--fail", "--silent", "--show-error", "--location",
    "--retry", "3", "--retry-delay", "1",
    "-A", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
    "-H", "Accept: */*",
]  # fmt: skip

RAW_NAME_PATTERN = re.compile(r"datos_abiertos_(?P<year>\d{4})_(?P<month>\d{2})")
# The column order is the same in every file, but the header spelling is not
# (UTF-8 or Latin-1, quoted or not, corrupt in some months), so we impose our own names
COLUMNS = [
    "viaje_id",
    "usuario_id",
    "genero",
    "anio_nacimiento",
    "inicio_viaje",
    "fin_viaje",
    "origen_id",
    "destino_id",
]
DTYPES = {
    "viaje_id": "int32",
    "usuario_id": "int32",
    "genero": pd.CategoricalDtype(categories=["F", "M"]),
    "anio_nacimiento": "Int16",
    "inicio_viaje": "string",
    "fin_viaje": "string",
    "origen_id": "int32",
    "destino_id": "int32",
}
DATE_COLUMNS = ["inicio_viaje", "fin_viaje"]
ISO_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# A few months were exported day-first and without seconds, e.g. "31/01/2021 8:15"
SLASH_DATE_FORMAT = "%d/%m/%Y %H:%M"


def fetch_page() -> str:
    """Return the HTML of the open data page."""
    proc = subprocess.run(
        [*CURL_ARGS, "--max-time", "60", PAGE_URL], capture_output=True, text=True, check=True
    )
    return proc.stdout


def find_csv_urls(html: str, initial_year: int, end_year: int) -> dict[tuple[int, int], str]:
    """Return {(year, month): url} for the trip CSVs of the given years."""
    urls = {}
    for match in CSV_PATTERN.finditer(html):
        year, month = int(match["year"]), int(match["month"])
        if initial_year <= year <= end_year:
            urls[(year, month)] = BASE_URL + match[0]
    return urls


def find_stations_url(html: str) -> str | None:
    """Return the url of the most recent station catalog, or None if the page has none."""
    matches = sorted(STATIONS_PATTERN.finditer(html), key=lambda m: (m["year"], m["month"]))
    return BASE_URL + matches[-1][0] if matches else None


def fetch_file(url: str, target: Path, header_fields: tuple[bytes, ...]) -> tuple[str, int, str]:
    """Download url into target unless it exists. Returns (status, size in bytes, error detail)."""
    temp = target.with_suffix(target.suffix + ".part")
    if target.exists():
        return ("EXISTS", target.stat().st_size, "")
    target.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [*CURL_ARGS, "--max-time", "600", "--output", str(temp), url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        temp.unlink(missing_ok=True)
        return ("ERROR", 0, proc.stderr.strip())
    try:
        size = temp.stat().st_size
        with temp.open("rb") as f:
            first = f.readline(2048)
        if size == 0 or not all(field in first for field in header_fields):
            temp.unlink(missing_ok=True)
            return ("INVALID", size, "CSV header missing or empty body")
        os.replace(temp, target)
        return ("OK", size, "")
    except OSError as exc:
        temp.unlink(missing_ok=True)
        return ("ERROR", 0, str(exc))


def download_file(item: tuple[tuple[int, int], str], output_dir: Path) -> tuple[str, int, str]:
    """Download one monthly trip CSV. Returns (status, size in bytes, error detail)."""
    (year, month), url = item
    target = output_dir / str(year) / f"datos_abiertos_{year}_{month:02}.csv"
    return fetch_file(url, target, TRIPS_HEADER_FIELDS)


def check_header(path: Path) -> None:
    """Raise if the CSV does not have the expected number of columns."""
    with open(path, "rb") as f:
        first_line = f.readline()
    n_columns = first_line.count(b",") + 1
    if n_columns != len(COLUMNS):
        raise ValueError(f"{path}: expected {len(COLUMNS)} columns, got {n_columns}")


def clean_file(path: Path) -> pd.DataFrame:
    """Read one raw monthly CSV with normalized column names, dtypes and dates."""
    check_header(path)
    # The body of every file is ASCII, so latin-1 only affects the header we discard
    df = pd.read_csv(path, header=0, names=COLUMNS, encoding="latin-1", dtype=DTYPES)
    is_slash = df["inicio_viaje"].str.contains("/", regex=False).any()
    date_format = SLASH_DATE_FORMAT if is_slash else ISO_DATE_FORMAT
    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column], format=date_format)
    return df


def describe_file(path: Path, root: Path) -> dict:
    """Return the path relative to root, row count, size and SHA-256 hash of a data file."""
    with open(path, "rb") as f:
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    if path.suffix == ".parquet":
        rows = pq.read_metadata(path).num_rows
    else:
        with open(path, "rb") as f:
            rows = sum(1 for _ in f) - 1  # minus the header
    return {
        "path": path.relative_to(root).as_posix(),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256,
    }


def build_manifest(data_dir: Path, stations_dir: Path) -> dict:
    """Describe every trip Parquet file under data_dir and every station catalog in stations_dir."""
    files = [describe_file(path, data_dir) for path in sorted(data_dir.glob("*/*.parquet"))]
    stations = [describe_file(path, stations_dir) for path in sorted(stations_dir.glob("*.csv"))]
    # One hash that summarizes all the files; this is the value to log in MLflow.
    # The manifest has no timestamp on purpose: unchanged data gives an identical manifest
    lines = "".join(f"{entry['path']}:{entry['sha256']}\n" for entry in files)
    lines += "".join(f"estaciones/{entry['path']}:{entry['sha256']}\n" for entry in stations)
    return {
        "data_version": hashlib.sha256(lines.encode()).hexdigest(),
        "total_files": len(files),
        "total_rows": sum(entry["rows"] for entry in files),
        "total_bytes": sum(entry["bytes"] for entry in files),
        "files": files,
        "estaciones": stations,
    }


@app.command()
def download(
    output_dir: Path = RAW_DATA_DIR,
    stations_dir: Path = EXTERNAL_DATA_DIR / "estaciones",
    initial_year: int = INITIAL_YEAR,
    end_year: int = END_YEAR,
    max_workers: int = 6,
):
    """Download the MiBici monthly trip CSVs into output_dir/<year>/ and the station catalog."""
    logger.info(f"Looking for CSV files from {initial_year} to {end_year} in {PAGE_URL}")
    html = fetch_page()
    urls = find_csv_urls(html, initial_year, end_year)
    if not urls:
        logger.error("No CSV links found; the page layout may have changed.")
        raise typer.Exit(1)

    items = sorted(urls.items())
    logger.info(f"Downloading {len(items)} files into {output_dir}")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(
            tqdm(pool.map(lambda item: download_file(item, output_dir), items), total=len(items))
        )

    available = [size for status, size, _ in results if status in ("OK", "EXISTS")]
    failed = [
        (key, status, detail)
        for (key, _), (status, _, detail) in zip(items, results)
        if status not in ("OK", "EXISTS")
    ]
    logger.info(
        f"{len(available)}/{len(items)} files available; {sum(available) / 1e9:.2f} GB in total."
    )
    for (year, month), status, detail in failed:
        logger.error(f"{year}-{month:02}: {status} {detail}")

    stations_url = find_stations_url(html)
    if stations_url is None:
        logger.error("No station catalog link found; the page layout may have changed.")
        raise typer.Exit(1)
    stations_target = stations_dir / stations_url.rsplit("/", 1)[-1]
    status, _, detail = fetch_file(stations_url, stations_target, STATIONS_HEADER_FIELDS)
    if status not in ("OK", "EXISTS"):
        logger.error(f"{stations_target.name}: {status} {detail}")
        raise typer.Exit(1)
    logger.info(f"Station catalog available at {stations_target}")
    older = [path.name for path in stations_dir.glob("*.csv") if path != stations_target]
    if older:
        logger.warning(f"Older station catalogs are also in {stations_dir}: {older}")

    if failed:
        raise typer.Exit(1)
    logger.success("Download complete.")


@app.command()
def clean(
    input_dir: Path = RAW_DATA_DIR,
    output_dir: Path = INTERIM_DATA_DIR / "viajes",
    overwrite: bool = False,
):
    """Clean the raw monthly CSVs into output_dir/year=<year>/viajes_<year>_<month>.parquet."""
    paths = sorted(input_dir.glob("*/*.csv"))
    if not paths:
        logger.error(f"No CSV files found in {input_dir}; run the download command first.")
        raise typer.Exit(1)

    logger.info(f"Cleaning {len(paths)} files into {output_dir}")
    rows = 0
    written = 0
    for path in tqdm(paths): # tqdm show a progress bar
        match = RAW_NAME_PATTERN.fullmatch(path.stem)
        if match is None:
            raise ValueError(f"{path}: unexpected file name")
        year, month = match["year"], match["month"]
        target = output_dir / f"year={year}" / f"viajes_{year}_{month}.parquet"
        if target.exists() and not overwrite:
            continue
        df = clean_file(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".part")
        df.to_parquet(temp, engine="pyarrow", compression="zstd", index=False)
        os.replace(temp, target)
        rows += len(df)
        written += 1

    logger.info(
        f"{written} files written ({rows:,} rows); {len(paths) - written} already existed."
    )
    logger.success("Cleaning complete.")


@app.command()
def manifest(
    data_dir: Path = INTERIM_DATA_DIR / "viajes",
    stations_dir: Path = EXTERNAL_DATA_DIR / "estaciones",
    manifest_path: Path = DATA_MANIFEST_PATH,
    check: bool = False,
):
    """Write the data manifest, or with --check compare the data against the saved one."""
    current = build_manifest(data_dir, stations_dir)
    if not current["files"]:
        logger.error(f"No Parquet files found in {data_dir}; run the clean command first.")
        raise typer.Exit(1)
    if not current["estaciones"]:
        logger.error(f"No station catalog found in {stations_dir}; run the download command first.")
        raise typer.Exit(1)

    if check:
        saved = json.loads(manifest_path.read_text())
        if saved == current:
            logger.success(f"Data matches the manifest (version {current['data_version'][:12]}).")
            return
        for section, section_dir in (("files", data_dir), ("estaciones", stations_dir)):
            saved_files = {entry["path"]: entry for entry in saved.get(section, [])}
            current_files = {entry["path"]: entry for entry in current[section]}
            for path in sorted(saved_files.keys() | current_files.keys()):
                if path not in current_files:
                    logger.error(f"{path}: missing from {section_dir}")
                elif path not in saved_files:
                    logger.error(f"{path}: not in the manifest")
                elif saved_files[path] != current_files[path]:
                    logger.error(
                        f"{path}: changed ({saved_files[path]['rows']:,} rows in the manifest, "
                        f"{current_files[path]['rows']:,} now)"
                    )
        raise typer.Exit(1)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(current, indent=2) + "\n")
    logger.info(
        f"{current['total_files']} files, {current['total_rows']:,} rows, "
        f"{len(current['estaciones'])} station catalog(s); "
        f"data version {current['data_version'][:12]}"
    )
    logger.success(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    app()
