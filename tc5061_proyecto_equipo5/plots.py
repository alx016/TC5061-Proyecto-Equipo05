"""Figures that justify the classification feature engineering.

The classification table mixes the raw station-hour flow, the engineered
features (calendar, neighbours, lags) and the label.  These six figures check
that the decisions behind that table are reasonable: the class balance, the
risk threshold, the warm-up of the lags, the daily demand cycle, the separation
of the classes by the lag and the coverage of the stations.

Every figure is written to ``reports/figures/classification/`` together with a
small ``summary.csv`` table.  Run it with:

    python -m tc5061_proyecto_equipo5.plots classification
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import typer

from tc5061_proyecto_equipo5.config import (
    CLASSIFICATION_FEATURES_PATH,
    FIGURES_DIR,
    RISK_THRESHOLD,
)

app = typer.Typer()


@app.callback()
def main() -> None:
    """Figures that justify the classification feature engineering."""


OUTPUT_DIR = FIGURES_DIR / "classification"

# Columns checked by the missing-values figure.  The model itself uses only the
# lagged and static columns (see modeling/train.py): the current-hour
# departures/arrivals/net_flow are not known at the minute-50 decision point.
FEATURE_COLUMNS = [
    "departures",
    "arrivals",
    "net_flow",
    "capacity",
    "latitude",
    "longitude",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "neighbor_net_flow",
    "net_flow_lag1",
    "net_flow_lag2",
    "net_flow_lag24",
    "arrivals_lag1",
    "arrivals_lag2",
    "arrivals_lag3",
    "departures_lag1",
    "departures_lag2",
    "departures_lag3",
]
# 0 = balanced, 1 = risk_shortage, 2 = risk_saturation
CLASS_LABELS = {0: "balanced", 1: "risk_shortage", 2: "risk_saturation"}
CLASS_COLORS = {0: "tab:blue", 1: "tab:red", 2: "tab:orange"}
# Spanish names for the figure that goes into the Spanish study-period report
PROXY_CLASS_LABELS = {0: "Balanceado", 1: "Riesgo_Desabasto", 2: "Riesgo_Saturacion"}


def load_features(path: Path) -> pd.DataFrame:
    """Read the classification feature table."""
    frame = pd.read_parquet(path)
    logger.info(f"Loaded {len(frame):,} rows and {frame.shape[1]} columns from {path}")
    return frame


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    """Save one figure into output_dir and close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {path}")
    return path


def plot_class_distribution(frame: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of the class counts with the share of labeled rows on each bar.

    The y-axis is on a log scale: the two risk classes are less than 1% of the
    rows, so on a linear scale they would be invisible next to ``balanced``.
    """
    counts = frame.loc[frame["target"].notna(), "target"].value_counts().sort_index()
    total = int(counts.sum())
    labels = [CLASS_LABELS[value] for value in counts.index]
    colors = [CLASS_COLORS[value] for value in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.to_numpy(), color=colors)
    for bar, count in zip(bars, counts.to_numpy()):
        share = 100 * count / total
        ax.annotate(
            f"{count:,}\n({share:.2f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of rows (log scale)")
    ax.set_title(
        f"Class distribution of target\npercentages over the {total:,} labeled rows "
        "(370 rows without label are excluded)"
    )
    ax.margins(y=0.25)
    save_figure(fig, output_dir, "class_distribution.png")


def plot_net_flow_histogram(frame: pd.DataFrame, output_dir: Path) -> None:
    """Histogram of net_flow with the two risk thresholds marked."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(frame["net_flow"], bins=200, color="tab:blue")
    ax.axvline(
        -RISK_THRESHOLD,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"shortage threshold ({-RISK_THRESHOLD:+.2f})",
    )
    ax.axvline(
        RISK_THRESHOLD,
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label=f"saturation threshold ({RISK_THRESHOLD:+.2f})",
    )
    ax.set_yscale("log")
    ax.set_xlim(-2, 2)
    ax.set_xlabel("net_flow = (arrivals - departures) / capacity")
    ax.set_ylabel("Number of station-hours (log scale)")
    ax.set_title("Distribution of net_flow (x-axis clipped to [-2, 2])")
    ax.legend(title="RISK_THRESHOLD = 0.40")
    save_figure(fig, output_dir, "net_flow_histogram.png")


def plot_missing_values(frame: pd.DataFrame, output_dir: Path) -> None:
    """Horizontal bar chart of the NaN count per feature column.

    Only the lag and neighbour columns have missing values: the first hours of
    January cannot look back one day, so they warm up.  Everything else is
    complete, which is why most bars are zero.
    """
    missing = frame[FEATURE_COLUMNS].isna().sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(missing.index, missing.to_numpy(), color="tab:purple")
    for position, count in enumerate(missing.to_numpy()):
        if count:
            ax.annotate(
                f"{count:,}",
                (count, position),
                ha="left",
                va="center",
                fontsize=9,
                xytext=(3, 0),
                textcoords="offset points",
            )
    ax.invert_yaxis()
    ax.set_xlabel("Number of NaN rows")
    ax.set_ylabel("Feature column")
    ax.set_title("Missing values per feature column (warm-up of the lagged features)")
    save_figure(fig, output_dir, "missing_values.png")


def plot_hourly_net_flow(frame: pd.DataFrame, output_dir: Path) -> None:
    """Average net_flow by hour of day (0-23)."""
    hourly = frame.groupby(frame["hour"].dt.hour)["net_flow"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(hourly.index, hourly.to_numpy(), color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average net_flow")
    ax.set_title("Average net_flow by hour of day (0-23)")
    save_figure(fig, output_dir, "hourly_net_flow.png")


def plot_class_vs_lag(frame: pd.DataFrame, output_dir: Path) -> None:
    """Average net_flow_lag1 per class, to show the lag separates the classes."""
    labeled = frame.dropna(subset=["net_flow_lag1", "target"])
    means = labeled.groupby("target")["net_flow_lag1"].mean().sort_index()
    labels = [CLASS_LABELS[value] for value in means.index]
    colors = [CLASS_COLORS[value] for value in means.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means.to_numpy(), color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Class (target)")
    ax.set_ylabel("Average net_flow_lag1 (previous hour)")
    ax.set_title("Average net_flow_lag1 per class: the lag separates the classes")
    save_figure(fig, output_dir, "class_vs_lag.png")


def plot_station_coverage(frame: pd.DataFrame, output_dir: Path) -> None:
    """Histogram of the number of active hours per station.

    An hour is active when the station moved at least one bike
    (departures + arrivals > 0).  The complete grid gives every station all the
    hours of the year, so this figure shows how many of them are real movement.
    """
    total_hours = frame["hour"].nunique()
    active = (
        frame.assign(active=(frame["departures"] + frame["arrivals"]) > 0)
        .groupby("station_id")["active"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(active, bins=40, color="tab:green", edgecolor="white")
    ax.axvline(
        total_hours,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"full year ({total_hours:,} hours)",
    )
    ax.set_xlabel("Active hours per station (departures + arrivals > 0)")
    ax.set_ylabel("Number of stations")
    ax.set_title(f"Station coverage: active hours per station (out of {total_hours:,})")
    ax.legend()
    save_figure(fig, output_dir, "station_coverage.png")


def plot_proxy_distribution(frame: pd.DataFrame, output_dir: Path) -> None:
    """Rebuild figure 04 of the study-period report with the full year.

    The old figure was built from January only and dropped the station-hours
    without trips, which inflated the two risk classes.  This version uses the
    complete 2024 grid, so its numbers match the feature table and the baselines.
    """
    labeled = frame.loc[frame["target"].notna(), "target"]
    total_labeled = len(labeled)
    counts = labeled.value_counts().sort_index()
    shares = 100 * counts / total_labeled
    labels = [PROXY_CLASS_LABELS[value] for value in counts.index]
    colors = [CLASS_COLORS[value] for value in counts.index]

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: continuous distribution of the proxy with the two stress thresholds.
    left.hist(frame["net_flow"], bins=160, color="tab:blue")
    left.axvspan(-1.6, -RISK_THRESHOLD, color="tab:red", alpha=0.12)
    left.axvspan(RISK_THRESHOLD, 1.6, color="tab:orange", alpha=0.12)
    left.axvline(
        -RISK_THRESHOLD,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"Umbral Desabasto (tau = -{RISK_THRESHOLD:.2f})",
    )
    left.axvline(
        RISK_THRESHOLD,
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        label=f"Umbral Saturación (tau = +{RISK_THRESHOLD:.2f})",
    )
    left.set_yscale("log")
    left.set_xlim(-1.6, 1.6)
    left.set_xlabel("Flujo Neto Normalizado (Llegadas - Salidas) / Capacidad")
    left.set_ylabel("Número de estaciones-hora (escala log)")
    left.set_title("Distribución del proxy de estrés operativo (2024 completo, eje x recortado)")
    left.legend()

    # Right: class shares; the log scale keeps the two rare classes visible.
    bars = right.bar(labels, shares.to_numpy(), color=colors)
    for bar, share, count in zip(bars, shares.to_numpy(), counts.to_numpy()):
        right.annotate(
            f"{share:.2f}%\n({count:,})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    right.set_yscale("log")
    right.set_ylim(0.1, 300)
    right.set_ylabel("Porcentaje de estaciones-hora (escala log)")
    right.set_title("Desbalance de clases de la variable objetivo")

    fig.suptitle(
        f"Viabilidad empírica del proxy de riesgo: {len(frame):,} estaciones-hora de 2024 "
        f"({frame['station_id'].nunique()} estaciones con actividad; {total_labeled:,} con etiqueta)"
    )
    save_figure(fig, output_dir, "04_p2_proxy_riesgo_distribucion.png")


def write_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    """Write a small summary table with the main numbers of the feature table."""
    labeled = frame.loc[frame["target"].notna(), "target"]
    total_labeled = len(labeled)
    splits = frame["split"].value_counts()
    counts = labeled.value_counts().sort_index()

    rows = [("total_rows", f"{len(frame):,}")]
    for split_name in ("train", "validation", "test"):
        rows.append((f"rows_{split_name}", f"{int(splits.get(split_name, 0)):,}"))
    for value, count in counts.items():
        label = CLASS_LABELS[value]
        share = 100 * count / total_labeled
        rows.append((f"class_{value}_{label}_count", f"{count:,}"))
        rows.append((f"class_{value}_{label}_percent", f"{share:.4f}"))
    rows.append(("labeled_rows", f"{total_labeled:,}"))
    rows.append(("rows_without_label", f"{int(frame['target'].isna().sum()):,}"))
    rows.append(("number_of_stations", f"{frame['station_id'].nunique():,}"))
    rows.append(("number_of_features", f"{len(FEATURE_COLUMNS)}"))

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(path, index=False)
    logger.info(f"Wrote {path}")


@app.command()
def classification(
    features_path: Path = CLASSIFICATION_FEATURES_PATH,
    output_dir: Path = OUTPUT_DIR,
):
    """Read the classification features and write the six figures and summary.csv."""
    logger.info(f"Building classification figures into {output_dir}")
    frame = load_features(features_path)
    plot_class_distribution(frame, output_dir)
    plot_net_flow_histogram(frame, output_dir)
    plot_missing_values(frame, output_dir)
    plot_hourly_net_flow(frame, output_dir)
    plot_class_vs_lag(frame, output_dir)
    plot_station_coverage(frame, output_dir)
    write_summary(frame, output_dir)
    logger.success("Classification figures complete.")


@app.command()
def proxy(
    features_path: Path = CLASSIFICATION_FEATURES_PATH,
    output_dir: Path = FIGURES_DIR,
):
    """Rebuild figure 04 (risk proxy) from the full-year classification table."""
    logger.info("Building the risk proxy figure (figure 04)")
    frame = load_features(features_path)
    plot_proxy_distribution(frame, output_dir)
    logger.success("Risk proxy figure complete.")


if __name__ == "__main__":
    app()
