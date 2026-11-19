import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    """Guarda la figura en output_dir a 300 DPI y la cierra."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def plot_finops(finops: dict, output_dir: Path) -> Path:
    """Figura 4: costo por 1M de predicciones de la línea base y de cada optimización.

    La línea base es el costo con la demanda real de MiBici y la instancia encendida 24/7.
    """
    base_cost = finops["baseline"]["cost_per_1m_pred_real_24_7_usd"]
    labels = ["Línea base (24/7)"]
    costs = [base_cost]
    texts = [f"${base_cost:.2f}"]
    for opt in finops["optimizations"]:
        labels.append(opt["name"])
        costs.append(opt["cost_per_1m_pred_usd"])
        texts.append(f"${opt['cost_per_1m_pred_usd']:.2f} (−{opt['savings_pct']:.0f}%)")
    colors = ["tab:gray"] + ["tab:blue"] * len(finops["optimizations"])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, costs, color=colors)
    ax.invert_yaxis()  # barh draws the first bar at the bottom
    ax.bar_label(bars, labels=texts, padding=3)
    ax.set_xlim(0, max(costs) * 1.25)
    ax.set_xlabel("USD por 1M de predicciones")
    ax.set_title("Costo por 1M de predicciones: línea base vs. optimizaciones")
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, output_dir, "04_finops_comparativa_optimizaciones.png")
