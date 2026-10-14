import pandas as pd
import pytest

from tc5061_proyecto_equipo5.dataset import COLUMNS, build_manifest, check_header, clean_file

UNQUOTED = "Viaje_Id,Usuario_Id,Genero,Año_de_nacimiento,Inicio_del_viaje,Fin_del_viaje,Origen_Id,Destino_Id"
QUOTED = ",".join(f'"{name}"' for name in UNQUOTED.split(","))
ISO_ROWS = [
    "1,10,M,1990,2021-01-31 08:15:02,2021-01-31 08:40:10,5,7",
    "2,11,F,NA,2021-01-31 23:59:59,2021-02-01 00:10:00,7,5",
    "3,12,NULL,1985,2021-01-05 00:00:00,2021-01-05 00:05:00,5,5",
]
SLASH_ROWS = [
    "1,10,M,1990,31/01/2021 8:15,31/01/2021 8:40,5,7",
    "2,11,F,NA,05/03/2021 23:59,06/03/2021 0:10,7,5",
]

HEADERS = {
    "utf8_unquoted": (UNQUOTED.encode("utf-8"), "\n"),
    "latin1_quoted": (QUOTED.encode("latin-1"), "\n"),
    "utf8_quoted": (QUOTED.encode("utf-8"), "\n"),
    "latin1_unquoted_crlf": (UNQUOTED.encode("latin-1"), "\r\n"),
    "corrupt_name": (QUOTED.encode("latin-1").replace(b"A\xf1o_d", b"A}\xe4"), "\n"),
}


def write_csv(path, header: bytes, rows: list[str], eol: str = "\n"):
    body = eol.join(rows) + eol
    path.write_bytes(header + eol.encode() + body.encode("ascii"))
    return path


@pytest.mark.parametrize("variant", HEADERS)
def test_clean_file_normalizes_every_header_variant(tmp_path, variant):
    header, eol = HEADERS[variant]
    df = clean_file(write_csv(tmp_path / "sample.csv", header, ISO_ROWS, eol))

    assert list(df.columns) == COLUMNS
    assert len(df) == len(ISO_ROWS)
    assert df["viaje_id"].dtype == "int32"
    assert df["anio_nacimiento"].dtype == "Int16"
    assert list(df["genero"].cat.categories) == ["F", "M"]
    assert pd.api.types.is_datetime64_any_dtype(df["inicio_viaje"])
    assert pd.api.types.is_datetime64_any_dtype(df["fin_viaje"])


def test_clean_file_parses_iso_dates(tmp_path):
    df = clean_file(write_csv(tmp_path / "sample.csv", UNQUOTED.encode(), ISO_ROWS))

    assert df["inicio_viaje"].iloc[0] == pd.Timestamp(2021, 1, 31, 8, 15, 2)
    assert df["fin_viaje"].iloc[1] == pd.Timestamp(2021, 2, 1, 0, 10, 0)


def test_clean_file_parses_slash_dates_day_first(tmp_path):
    df = clean_file(write_csv(tmp_path / "sample.csv", UNQUOTED.encode(), SLASH_ROWS, "\r\n"))

    assert df["inicio_viaje"].iloc[0] == pd.Timestamp(2021, 1, 31, 8, 15)
    # 05/03 is March 5th, not May 3rd
    assert df["inicio_viaje"].iloc[1] == pd.Timestamp(2021, 3, 5, 23, 59)
    assert df["fin_viaje"].iloc[1] == pd.Timestamp(2021, 3, 6, 0, 10)


def test_clean_file_raises_on_unparseable_date(tmp_path):
    rows = ["1,10,M,1990,2021-13-45 08:15:02,2021-01-31 08:40:10,5,7"]

    with pytest.raises(ValueError):
        clean_file(write_csv(tmp_path / "sample.csv", UNQUOTED.encode(), rows))


def test_clean_file_handles_missing_values(tmp_path):
    rows = [
        "1,10,NULL,NA,2021-01-01 00:00:00,2021-01-01 00:05:00,5,7",
        "2,11,,,2021-01-01 00:00:00,2021-01-01 00:05:00,5,7",
        "3,12,M,1990.0,2021-01-01 00:00:00,2021-01-01 00:05:00,5,7",
    ]
    df = clean_file(write_csv(tmp_path / "sample.csv", UNQUOTED.encode(), rows))

    assert df["genero"].isna().tolist() == [True, True, False]
    assert df["anio_nacimiento"].isna().tolist() == [True, True, False]
    assert df["anio_nacimiento"].iloc[2] == 1990


@pytest.mark.parametrize(
    "header",
    [b"a,b,c,d,e,f,g", b"a,b,c,d,e,f,g,h,i", b"a;b;c;d;e;f;g;h", b""],
    ids=["7_columns", "9_columns", "semicolons", "empty"],
)
def test_check_header_rejects_wrong_column_count(tmp_path, header):
    path = tmp_path / "bad.csv"
    path.write_bytes(header + b"\n")

    with pytest.raises(ValueError, match="bad.csv: expected 8 columns"):
        check_header(path)


def test_parquet_round_trip_preserves_data(tmp_path):
    df = clean_file(write_csv(tmp_path / "sample.csv", UNQUOTED.encode(), ISO_ROWS))
    target = tmp_path / "sample.parquet"
    df.to_parquet(target, engine="pyarrow", compression="zstd", index=False)

    pd.testing.assert_frame_equal(pd.read_parquet(target), df)


def write_parquet(data_dir, name: str, rows: list[str]):
    csv = write_csv(data_dir / "sample.csv", UNQUOTED.encode(), rows)
    target = data_dir / "year=2021" / name
    target.parent.mkdir(exist_ok=True)
    clean_file(csv).to_parquet(target, engine="pyarrow", compression="zstd", index=False)
    csv.unlink()
    return target


def test_build_manifest_describes_each_file(tmp_path):
    write_parquet(tmp_path, "viajes_2021_02.parquet", ISO_ROWS[:1])
    first = write_parquet(tmp_path, "viajes_2021_01.parquet", ISO_ROWS)

    manifest = build_manifest(tmp_path)

    assert [entry["path"] for entry in manifest["files"]] == [
        "year=2021/viajes_2021_01.parquet",
        "year=2021/viajes_2021_02.parquet",
    ]
    assert [entry["rows"] for entry in manifest["files"]] == [3, 1]
    assert manifest["files"][0]["bytes"] == first.stat().st_size
    assert manifest["total_files"] == 2
    assert manifest["total_rows"] == 4


def test_build_manifest_is_stable_for_unchanged_data(tmp_path):
    write_parquet(tmp_path, "viajes_2021_01.parquet", ISO_ROWS)

    assert build_manifest(tmp_path) == build_manifest(tmp_path)


def test_build_manifest_version_changes_when_data_changes(tmp_path):
    write_parquet(tmp_path, "viajes_2021_01.parquet", ISO_ROWS)
    before = build_manifest(tmp_path)

    changed = [ISO_ROWS[0].replace(",1990,", ",1991,"), *ISO_ROWS[1:]]
    write_parquet(tmp_path, "viajes_2021_01.parquet", changed)
    after = build_manifest(tmp_path)

    # Same number of rows, so only the hash can tell the difference
    assert after["total_rows"] == before["total_rows"]
    assert after["files"][0]["sha256"] != before["files"][0]["sha256"]
    assert after["data_version"] != before["data_version"]
