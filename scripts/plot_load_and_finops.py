"""Genera las 4 figuras de la Entrega 6 a partir de los CSV de Locust, docker stats y FinOps.

Lee lo que dejan ``scripts/collect_load_metrics.py`` y ``scripts/finops_calc.py`` en
``--load-dir`` y escribe los PNG a 300 DPI en ``--output-dir``:

- ``01_rps_y_usuarios_vs_tiempo.png``
- ``02_latencia_percentiles.png``
- ``03_recursos_contenedores.png``
- ``04_finops_comparativa_optimizaciones.png``

Uso:
    uv run python scripts/plot_load_and_finops.py
    uv run python scripts/plot_load_and_finops.py --load-dir /tmp/load-fixtures --output-dir /tmp/figs
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Orden en que corren los escalones; un slug desconocido va al final.
SCENARIO_ORDER = [
    "baseline",
    "nominal_10u",
    "high_50u",
    "stress_120u",
    "batch_1x365",
    "batch_365x1",
]
# Paleta categórica validada (daltonismo y contraste), siempre en este orden.
# Cada entidad conserva su color en todas las figuras.
ENDPOINT_COLORS = {
    "/predict?batch=1": "#2a78d6",
    "/predict?batch=20": "#eb6834",
    "/predict?batch=365": "#1baf7a",
    "/health": "#eda100",
    "/model-info": "#e87ba4",
}
CONTAINER_COLORS = {"api": "#2a78d6", "mlflow": "#eb6834"}
# Los percentiles son ordinales: un gris de claro a oscuro, sin competir con los endpoints.
PERCENTILE_COLORS = {"50%": "#c4c3bc", "95%": "#8c8b85", "99%": "#4d4c48"}
NEUTRAL = "#8c8b85"
TOTAL_COLOR = "#2a78d6"
BATCH_ENDPOINTS = ["/predict?batch=1", "/predict?batch=20", "/predict?batch=365"]


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


def ordered_scenarios(scenarios) -> list[str]:
    """Slugs en el orden de SCENARIO_ORDER; los desconocidos van al final, alfabéticos."""
    present = set(scenarios)
    known = [slug for slug in SCENARIO_ORDER if slug in present]
    return known + sorted(present - set(SCENARIO_ORDER))


def style_axis(ax: plt.Axes) -> None:
    """Ejes discretos: sin marco superior/derecho y rejilla horizontal tenue."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(ax: plt.Axes, table: pd.DataFrame, colors: dict, labels: dict | None = None):
    """Barras agrupadas: una fila de ``table`` por grupo y una columna por serie."""
    n_series = len(table.columns)
    width = 0.8 / n_series
    positions = range(len(table.index))
    for i, column in enumerate(table.columns):
        offsets = [p - 0.4 + width * (i + 0.5) for p in positions]
        label = labels.get(column, column) if labels else column
        ax.bar(offsets, table[column], width * 0.9, color=colors[column], label=label)
    ax.set_xticks(list(positions), table.index)
    style_axis(ax)


def usd(value: float) -> str:
    """Costos muy chicos (pico) necesitan más decimales que los de la demanda real."""
    return f"${value:.4f}" if value < 0.01 else f"${value:.2f}"


def plot_rps_users(history: pd.DataFrame, stats: pd.DataFrame, output_dir: Path) -> Path:
    """Figura 1: RPS total y usuarios en el tiempo, y RPS por endpoint en cada escalón.

    Sin ``--csv-full-history`` el historial de Locust solo trae filas ``Aggregated``, así
    que el desglose por endpoint sale de ``_stats.csv``.
    """
    history = history[history["Name"] == "Aggregated"].copy()
    t0 = history["Timestamp"].min()
    history["minute"] = (history["Timestamp"] - t0) / 60
    scenarios = ordered_scenarios(history["scenario"])

    fig = plt.figure(figsize=(10, 9))
    grid = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2])
    ax_rps = fig.add_subplot(grid[0])
    ax_users = fig.add_subplot(grid[1], sharex=ax_rps)
    ax_endpoints = fig.add_subplot(grid[2])

    # Una línea por escalón para no unir las pausas entre corridas.
    for slug in scenarios:
        run = history[history["scenario"] == slug]
        ax_rps.plot(run["minute"], run["Requests/s"], color=TOTAL_COLOR, linewidth=1.5)
        ax_users.step(run["minute"], run["User Count"], where="post", color=NEUTRAL, linewidth=1.5)
        middle = run["minute"].median()
        ax_rps.annotate(
            slug,
            (middle, 1.0),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#5f5e58",
        )
    ax_rps.set_ylabel("Peticiones/s (total)")
    ax_rps.set_ylim(bottom=0)
    ax_users.set_ylabel("Usuarios")
    ax_users.set_xlabel("Minutos desde el inicio de la corrida")
    ax_users.set_ylim(bottom=0)
    plt.setp(ax_rps.get_xticklabels(), visible=False)
    for ax in (ax_rps, ax_users):
        style_axis(ax)

    by_endpoint = stats[stats["Name"] != "Aggregated"].pivot_table(
        index="scenario", columns="Name", values="Requests/s", aggfunc="sum"
    )
    by_endpoint = by_endpoint.reindex(
        index=ordered_scenarios(by_endpoint.index),
        columns=[name for name in ENDPOINT_COLORS if name in by_endpoint.columns],
    ).fillna(0)
    grouped_bars(ax_endpoints, by_endpoint, ENDPOINT_COLORS)
    ax_endpoints.set_ylabel("Peticiones/s (promedio)")
    ax_endpoints.legend(frameon=False, ncols=3, fontsize=8, loc="upper left")

    fig.suptitle("Throughput y usuarios concurrentes por escalón de carga")
    return save_figure(fig, output_dir, "01_rps_y_usuarios_vs_tiempo.png")


def plot_latency(stats: pd.DataFrame, output_dir: Path) -> Path:
    """Figura 2: p50/p95/p99 por escalón, p95 por tamaño de lote y tasa de error."""
    aggregated = stats[stats["Name"] == "Aggregated"].set_index("scenario")
    aggregated = aggregated.reindex(ordered_scenarios(aggregated.index))
    endpoints = stats[stats["Name"] != "Aggregated"]

    fig, (ax_pct, ax_batch, ax_err) = plt.subplots(3, 1, figsize=(10, 10))

    percentiles = aggregated[list(PERCENTILE_COLORS)]
    grouped_bars(
        ax_pct, percentiles, PERCENTILE_COLORS, labels={p: f"p{p[:-1]}" for p in PERCENTILE_COLORS}
    )
    ax_pct.set_ylabel("Latencia (ms)")
    ax_pct.set_title("Percentiles de latencia (todas las peticiones)", fontsize=10)
    ax_pct.legend(frameon=False, fontsize=8)

    batches = endpoints[endpoints["Name"].isin(BATCH_ENDPOINTS)].pivot_table(
        index="scenario", columns="Name", values="95%"
    )
    batches = batches.reindex(
        index=ordered_scenarios(batches.index),
        columns=[name for name in BATCH_ENDPOINTS if name in batches.columns],
    )
    grouped_bars(ax_batch, batches, ENDPOINT_COLORS)
    ax_batch.set_ylabel("p95 (ms)")
    ax_batch.set_title("p95 por tamaño de lote de /predict", fontsize=10)
    ax_batch.legend(frameon=False, fontsize=8)

    # Tasa de error desde las filas por endpoint (sin la fila Aggregated), como pide el contrato.
    totals = endpoints.groupby("scenario")[["Request Count", "Failure Count"]].sum()
    totals = totals.reindex(ordered_scenarios(totals.index))
    error_pct = 100 * totals["Failure Count"] / totals["Request Count"]
    bars = ax_err.bar(error_pct.index, error_pct, color=NEUTRAL, width=0.5)
    ax_err.bar_label(bars, labels=[f"{v:.1f}%" for v in error_pct], padding=2, fontsize=8)
    ax_err.set_ylabel("Error (%)")
    ax_err.set_ylim(0, max(error_pct.max() * 1.3, 1))
    ax_err.set_title("Tasa de error (fallas / peticiones)", fontsize=10)
    style_axis(ax_err)

    fig.suptitle("Latencia y errores por escalón de carga")
    return save_figure(fig, output_dir, "02_latencia_percentiles.png")


def plot_resources(docker: pd.DataFrame, output_dir: Path) -> Path:
    """Figura 3: CPU % y memoria (MiB) de api y mlflow en reposo y en cada escalón.

    La barra es el promedio del escalón y el bigote llega al máximo observado.
    """
    containers = [name for name in CONTAINER_COLORS if name in set(docker["container"])]
    scenarios = ordered_scenarios(docker["scenario"])
    summary = docker.groupby(["scenario", "container"]).agg(
        cpu_mean=("cpu_pct", "mean"),
        cpu_max=("cpu_pct", "max"),
        mem_mean=("mem_usage_mib", "mean"),
        mem_max=("mem_usage_mib", "max"),
        mem_limit=("mem_limit_mib", "max"),
    )

    fig, (ax_cpu, ax_mem) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    width = 0.8 / len(containers)
    for i, container in enumerate(containers):
        rows = summary.xs(container, level="container").reindex(scenarios)
        offsets = [p - 0.4 + width * (i + 0.5) for p in range(len(scenarios))]
        color = CONTAINER_COLORS[container]
        for ax, metric in ((ax_cpu, "cpu"), (ax_mem, "mem")):
            mean, peak = rows[f"{metric}_mean"], rows[f"{metric}_max"]
            ax.bar(
                offsets,
                mean,
                width * 0.9,
                color=color,
                label=container,
                yerr=[[0] * len(mean), peak - mean],
                error_kw={"ecolor": "#5f5e58", "lw": 1},
            )
        limit = rows["mem_limit"].max()
        ax_mem.axhline(limit, color=color, linestyle="--", linewidth=1)
        ax_mem.annotate(
            f"límite {container} ({limit:.0f} MiB)",
            (1.0, limit),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=8,
            color="#5f5e58",
        )

    ax_cpu.set_ylabel("CPU (%)")
    ax_cpu.legend(frameon=False, fontsize=8)
    ax_mem.set_ylabel("Memoria (MiB)")
    ax_mem.set_xticks(range(len(scenarios)), scenarios)
    for ax in (ax_cpu, ax_mem):
        style_axis(ax)

    fig.suptitle("Recursos de los contenedores (docker stats): promedio y máximo")
    return save_figure(fig, output_dir, "03_recursos_contenedores.png")


def plot_finops(finops: dict, output_dir: Path) -> Path:
    """Figura 4: costo por 1M de predicciones de la línea base y de cada optimización.

    Cada optimización se compara contra su propia línea base (``reference``): las de
    capacidad contra el costo en el pico y scale-to-zero contra la demanda real 24/7.
    """
    baselines = {
        "peak": ("Línea base (pico)", "cost_per_1m_pred_peak_usd"),
        "real_24_7": ("Línea base (24/7, demanda real)", "cost_per_1m_pred_real_24_7_usd"),
    }

    fig, axes = plt.subplots(len(baselines), 1, figsize=(8, 6))
    for ax, (reference, (base_label, base_key)) in zip(axes, baselines.items()):
        base_cost = finops["baseline"][base_key]
        opts = [o for o in finops["optimizations"] if o["reference"] == reference]
        labels = [base_label] + [o["name"] for o in opts]
        costs = [base_cost] + [o["cost_per_1m_pred_usd"] for o in opts]
        texts = [usd(base_cost)] + [
            f"{usd(o['cost_per_1m_pred_usd'])} (−{o['savings_pct']:.0f}%)" for o in opts
        ]
        colors = [NEUTRAL] + [TOTAL_COLOR] * len(opts)

        bars = ax.barh(labels, costs, color=colors, height=0.6)
        ax.invert_yaxis()  # barh draws the first bar at the bottom
        ax.bar_label(bars, labels=texts, padding=3, fontsize=8)
        ax.set_xlim(0, max(costs) * 1.35)
        ax.set_xlabel("USD por 1M de predicciones")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Costo por 1M de predicciones: línea base vs. optimizaciones")
    return save_figure(fig, output_dir, "04_finops_comparativa_optimizaciones.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--load-dir", type=Path, default=Path("reports/load"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures/entrega6"))
    args = parser.parse_args()

    stats = load_locust_stats(args.load_dir)
    plot_rps_users(load_locust_history(args.load_dir), stats, args.output_dir)
    plot_latency(stats, args.output_dir)
    plot_resources(load_docker_stats(args.load_dir), args.output_dir)
    plot_finops(load_finops(args.load_dir), args.output_dir)


if __name__ == "__main__":
    main()
