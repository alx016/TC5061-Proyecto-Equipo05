import json
from pathlib import Path

import pandas as pd


def load_locust_stats(load_dir: Path) -> pd.DataFrame:
    """
    Concatena todos los locust_*_stats.csv en un solo DataFrame, con una sola columna scenario sacada del nombre del archivo.
    """
    pattern = "locust_*_stats.csv"
    paths = sorted(load_dir.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No hay locust_*_stats.csv en {load_dir}")

    frames = []

    for path in paths:
        df = pd.read_csv(path)
        slug = path.name.removeprefix("locust_").removesuffix("_stats.csv")
        df["scenario"] = slug
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_locust_history(load_dir: Path) -> pd.DataFrame:
    """
    Concatena todos los locust_*_stats_history.csv en un solo DataFrame, con una sola columna scenario sacada del nombre del archivo.
    """
    pattern = "locust_*_stats_history.csv"
    paths = sorted(load_dir.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No hay locust_*_stats_history.csv en {load_dir}")

    frames = []

    for path in paths:
        df = pd.read_csv(path)
        slug = path.name.removeprefix("locust_").removesuffix("_stats_history.csv")
        df["scenario"] = slug
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_docker_stats(load_dir: Path) -> pd.DataFrame:
    """
    concatena los docker_stats_*.csv, que ya traen la columna scenario.
    """
    pattern = "docker_stats_*.csv"
    paths = sorted(load_dir.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No hay docker_stats_*.csv en {load_dir}")

    frames = [pd.read_csv(path) for path in paths]

    return pd.concat(frames, ignore_index=True)


def load_finops(load_dir: Path) -> dict:
    """
    lee el JSON y regresa un dict.
    """
    path = load_dir / "finops_summary.json"
    if not path.exists(): 
        raise FileNotFoundError(f"No hay finops_summary.json en {load_dir}")
    return json.loads(path.read_text())
