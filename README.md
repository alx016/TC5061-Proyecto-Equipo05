# MiBici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Diseñar, como equipo, una propuesta de solución de Machine Learning para un conjunto de preguntas sobre el sistema de bicicleta pública MiBici, que combine modelos de regresión, clasificación y clustering. Y realizar tareas de manipulación, preparación, exploración y preprocesamiento de datos, gestionando grandes volúmenes de datos distribuidos y aplicando técnicas de versionado de datos para asegurar la reproducibilidad y trazabilidad en proyectos de Machine Learning.

## Getting the data

The data is not stored in Git. To download and prepare it, run:

```bash
make data
```

This downloads the MiBici monthly trip CSVs into `data/raw/`, cleans them into
Parquet files in `data/interim/viajes/`, downloads the latest station catalog
(id, name, coordinates and capacity) into `data/external/estaciones/`, and writes
`references/viajes_manifest.json`.
The years it covers are set by `INITIAL_YEAR` and `END_YEAR` in
`tc5061_proyecto_equipo5/config.py` (currently 2024–2026).

To check that your local data matches the committed manifest:

```bash
python tc5061_proyecto_equipo5/dataset.py manifest --check
```

## Feature engineering

Un único `features.py` ofrece recetas independientes para demanda, riesgo y clustering,
con historia causal y preprocesadores sin ajustar para cada estimador. Consulta los
[contratos, comandos y límites](docs/docs/feature-engineering.md) y la
[investigación con fuentes primarias](docs/docs/feature-engineering-research.md).

## Modeling scripts

Each project question has its own training script so the team can work in parallel without editing the same file:

| Question | Task | Script | Command |
|:---|:---|:---|:---|
| P1 | Regression | `modeling/train_regression.py` *(pending)* | `python -m tc5061_proyecto_equipo5.modeling.train_regression` |
| P2 | Classification | `modeling/train_classification.py` | `python -m tc5061_proyecto_equipo5.modeling.train_classification classification` |
| P3 | Clustering | `modeling/train_clustering.py` *(pending)* | `python -m tc5061_proyecto_equipo5.modeling.train_clustering` |

Rules that keep the three scripts comparable:

- Shared reproducibility helpers live in `modeling/common.py`: `data_version()`, `setup_experiment()` and `base_tags(question)`. Do not duplicate them.
- All runs go to the `mibici-equipo05` experiment, tagged with `team` and `question`. MLflow adds the Git commit, the branch and the repo URL by itself (`mlflow.source.git.*`), so there is nothing to read by hand.
- Metric names carry the split as a prefix: `test_macro_f1`, `val_mae`, etc.
- Every run logs the data version from the manifest as a parameter.
- Code, comments, docstrings and file names are written in English; figures and text artifacts are written in Spanish.

The classification artifacts are materialized with `python -m tc5061_proyecto_equipo5.features classification` and the justification figures with `python -m tc5061_proyecto_equipo5.plots classification`.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         tc5061_proyecto_equipo5 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── tc5061_proyecto_equipo5   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes tc5061_proyecto_equipo5 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── common.py                   <- Helpers shared by the three training scripts
    │   ├── predict.py                  <- Code to run model inference with trained models          
    │   ├── train_classification.py     <- P2: station risk classification baselines
    │   ├── train_clustering.py         <- P3: station clustering (pending)
    │   └── train_regression.py         <- P1: demand regression (pending)
    │
    └── plots.py                <- Code to create visualizations
```

--------

