# MiBici · MLOps (TC5061 · Equipo 05)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Propósito

MiBici, el sistema de bicicleta pública del Área Metropolitana de Guadalajara, necesita
anticipar dónde y cuándo se usarán sus bicicletas para planear la flota y el rebalanceo de
estaciones. Este proyecto propone una solución de Machine Learning reproducible que combina
regresión, clasificación y clustering sobre los datos abiertos de viajes, con ingesta
versionada, feature engineering sin fuga de información y experimentos trazables en MLflow.

## Preguntas de negocio

| | Pregunta | Tarea | Métrica primaria | Canvas |
|:---|:---|:---|:---|:---|
| **P1** | ¿Cuántos viajes se iniciarán en una estación durante la siguiente hora? | Regresión por estación-hora | WAPE (también MAE y RMSE) | [p1-demanda](docs/docs/canvases/p1-demanda.md) |
| **P2** | ¿Una estación se quedará sin bicicletas o sin anclajes durante la siguiente hora? | Clasificación de tres clases (desabasto, balanceada, saturación) | Macro-F1 y PR-AUC | [p2-riesgo](docs/docs/canvases/p2-riesgo.md) |
| **P3** | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering de perfiles horarios | Silhouette y Davies-Bouldin | [p3-clustering](docs/docs/canvases/p3-clustering.md) |

P1 y P2 predicen con una hora de anticipación: a las `h-00:10` se predice la hora `[h, h+1)`,
usando solo información disponible en ese momento.

## Datos

- **Fuente:** [datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/), con un
  CSV de viajes por mes (id de viaje, inicio, fin, estación de origen y de destino) y el
  catálogo de estaciones (id, nombre, coordenadas y capacidad). El portal solo publica la
  versión más reciente del catálogo.
- **Periodo descargado:** de enero de 2024 a septiembre de 2026 (33 archivos, 12.5 M viajes),
  definido por `INITIAL_YEAR` y `END_YEAR` en `tc5061_proyecto_equipo5/config.py`.
- **Año de estudio:** 2024 (4.6 M viajes, unas 370 estaciones activas), con partición temporal:
  train de enero a septiembre, validación de octubre a noviembre y test de diciembre.
  2025 y 2026 se reservan para monitoreo. La elección está justificada en
  [justificación del periodo de estudio](docs/docs/justificacion-periodo-estudio.md).
- **Versionado:** los datos no se guardan en Git. `references/viajes_manifest.json` registra el
  hash de cada archivo y un `data_version` global, que cada run de MLflow guarda como parámetro.
- **Privacidad:** los Parquet conservan las columnas personales del portal (usuario, género,
  año de nacimiento), pero ninguna receta de features ni modelo las usa.

## Instalación

Requisitos: Python 3.13, [uv](https://docs.astral.sh/uv/) y Git. En macOS, LightGBM necesita
además OpenMP (`brew install libomp`).

```bash
git clone https://github.com/alx016/TC5061-Proyecto-Equipo05.git
cd TC5061-Proyecto-Equipo05
uv sync                 # crea .venv con las versiones fijadas en uv.lock
uv run pytest           # verifica la instalación
```

Los comandos de este README usan `uv run`; si se activa el entorno (`source .venv/bin/activate`)
basta con `python`.

## Ingesta

```bash
make data
```

Descarga los CSV mensuales a `data/raw/`, los limpia y convierte a Parquet en
`data/interim/viajes/year=YYYY/`, descarga el catálogo de estaciones a
`data/external/estaciones/` y escribe `references/viajes_manifest.json`.

Para verificar que los datos locales coinciden con el manifest del repositorio:

```bash
uv run python tc5061_proyecto_equipo5/dataset.py manifest --check
```

## Experimento base

Cada pregunta tiene su propio script de entrenamiento y todos registran sus runs en el
experimento `mibici-equipo05` de MLflow (tracking local en `mlflow.db` y `mlruns/`). El
pipeline completo, después de la ingesta:

```bash
# 1. Features (los artefactos se escriben en data/processed/)
uv run python -m tc5061_proyecto_equipo5.features main --problem p1 --prepare-ready \
  --input-dir data/interim/viajes --output-dir data/processed/fe_2024_v4
uv run python -m tc5061_proyecto_equipo5.features classification
uv run python -m tc5061_proyecto_equipo5.features main --problem p3 --prepare-ready \
  --input-dir data/interim/viajes --output-dir data/processed/fe_2024_v4

# 2. Entrenamiento (tres runs por pregunta)
uv run python -m tc5061_proyecto_equipo5.modeling.train_regression regression
uv run python -m tc5061_proyecto_equipo5.modeling.train_classification classification
uv run python -m tc5061_proyecto_equipo5.modeling.train_clustering clustering

# 3. Resultados
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

`features main --prepare-ready` no sobrescribe una carpeta existente; para regenerar los
artefactos hay que mover o borrar antes `data/processed/fe_2024_v4/p1` o `p3`. En macOS el
puerto 5000 lo ocupa AirPlay, por eso se usa el 5001.

| Pregunta | Script | Candidatos | Selección |
|:---|:---|:---|:---|
| P1 | `modeling/train_regression.py` | Promedio estacional, Ridge, LightGBM Poisson | Menor WAPE de validación; solo el run `selected=true` se evalúa en test |
| P2 | `modeling/train_classification.py` | Clase mayoritaria, regresión logística, LightGBM | Comparación de macro-F1 y PR-AUC |
| P3 | `modeling/train_clustering.py` | K-Means con K=4 de referencia, K-Means con el mejor K, Ward | K con la mayor silhouette de validación |

Reglas comunes a los tres scripts:

- Las utilidades de reproducibilidad viven en `modeling/common.py`: `data_version()`,
  `setup_experiment()` y `base_tags(question)`. No se duplican.
- Cada run lleva los tags `team`, `question`, `git_commit` y `git_dirty`, y el parámetro
  `data_version`. Un run con `git_dirty=true` se entrenó con cambios sin commit y no es
  reproducible.
- Los nombres de las métricas llevan el split como prefijo: `val_wape`, `test_macro_f1`, etc.
- El código, los comentarios y los nombres de archivo van en inglés; las figuras y los
  artefactos de texto, en español.

Los contratos de features y el protocolo contra la fuga de información están en
[feature engineering](docs/docs/feature-engineering.md). P2 usa el catálogo de estaciones más
reciente y supone que la capacidad no cambió desde 2024; ese supuesto debe reportarse junto con
los resultados.

## Flujo de trabajo con Git

- `main` siempre debe funcionar y no recibe commits directos.
- Cada cambio vive en su propia rama, nombrada según su tipo: `feat/` (funcionalidad),
  `fix/` (corrección) o `docs/` (documentación). Por ejemplo, `feat/regression-training`.
- Los commits siguen [Conventional Commits](https://www.conventionalcommits.org/):
  `tipo(alcance): descripción`, por ejemplo `feat(modeling): add the P1 regression training`.
- Todo cambio entra a `main` mediante un pull request revisado por otro integrante. Antes de
  abrirlo: `make lint` y `uv run pytest`.
- Los datos (`data/`), los modelos y los runs de MLflow (`mlflow.db`, `mlruns/`) no se suben a
  Git. Los datos se reconstruyen con `make data` y se verifican con el manifest.
- Antes de un entrenamiento que sirva como evidencia, el commit debe estar hecho para que el
  run quede con `git_dirty=false`.

## Integrantes

Contribuciones de cada integrante según el historial de Git de `main`.

| Integrante | Matrícula | Usuario | Contribuciones |
|:---|:---|:---|:---|
| Manuel Alejandro Preciado Morán | A01639643 | `alx016` | Estructura inicial del proyecto (plantilla Cookiecutter Data Science). Ingesta: descarga, limpieza a Parquet y manifest de los viajes (`dataset.py`), descarga del catálogo de estaciones y rango de datos 2024–2026. P1: script de regresión con MLflow (`train_regression.py`), registro de las fechas de cada split y de los hiperparámetros con `get_params()`. Tags `git_commit` y `git_dirty` para todos los runs (`common.py`). Validación de capacidad y festivos en `features.py`. Instrucciones de datos en el README. |
| Oscar Ulises López Vázquez | A01840039 | `oscarlv` | P2: feature engineering de clasificación, figuras de justificación del proxy de riesgo (`plots.py`) y baselines de clasificación con MLflow, incluido el candidato LightGBM (`train_classification.py`). P3: baselines de clustering con MLflow (`train_clustering.py`). Separación en un script de entrenamiento por pregunta con utilidades compartidas (`common.py`). Incorporación de MLflow al proyecto, catálogo de estaciones en el manifest y sus pruebas, y regeneración de `uv.lock`. |
| Emilio Alejandro Martínez López | A01796936 | `Emilio` | Feature engineering causal por modelo (`features.py`) con sus pruebas y su validación. Sitio de documentación MkDocs: los ML Canvas de P1, P2 y P3, el contrato de feature engineering, la investigación de fuentes y la justificación técnica del periodo de estudio 2024 y de la partición temporal. Notebook de exploración inicial (`0.01-eaml`) y versiones exactas de las dependencias en `pyproject.toml`. |
| Fernando Gutiérrez Escárrega | A01840067 | `Fernando cuenta TEC` | Configuración del entorno con uv, Python 3.13 y dependencias fijas. Análisis exploratorio (`1.0-fge-eda-mibici.ipynb`): nomenclatura de estaciones, histórico 2014–2026, análisis de 2024 para feature engineering, auditoría de limpieza de 2024 y calendario de festivos, con sus figuras en `reports/figures/`. |

Los merges de los pull requests hechos desde la web de GitHub aparecen con el nombre de la cuenta
de GitHub (por ejemplo, `Oscar` o `Manuel Alejandro Preciado Morán`), no con el usuario de Git.

## Estructura del proyecto

```
├── Makefile                   <- Comandos: make data, make lint, make format, make test
├── README.md
├── pyproject.toml             <- Metadatos del paquete, dependencias y configuración de ruff
├── uv.lock                    <- Versiones exactas de las dependencias
├── data/                      <- Fuera de Git; se genera con make data
│   ├── raw/                   <- CSV mensuales descargados
│   ├── interim/viajes/        <- Viajes limpios en Parquet, particionados por año
│   ├── external/estaciones/   <- Catálogo de estaciones
│   └── processed/             <- Panel estación-hora y artefactos de features por pregunta
├── docs/                      <- Sitio MkDocs: canvases, feature engineering y justificaciones
├── notebooks/                 <- Exploración inicial y EDA (número-iniciales-descripción)
├── references/                <- ML Canvas de referencia y viajes_manifest.json
├── reports/                   <- Reportes y figuras generadas
├── tests/                     <- Pruebas unitarias con datos sintéticos (pytest)
├── mlflow.db, mlruns/         <- Tracking local de MLflow; fuera de Git
└── tc5061_proyecto_equipo5/   <- Código fuente
    ├── config.py              <- Rutas, años y constantes del estudio
    ├── dataset.py             <- Descarga, limpieza y manifest de los datos
    ├── features.py            <- Panel estación-hora y recetas de features de P1, P2 y P3
    ├── plots.py               <- Figuras de justificación
    └── modeling/
        ├── common.py                <- Utilidades compartidas de MLflow y reproducibilidad
        ├── train_regression.py      <- P1: regresión de demanda
        ├── train_classification.py  <- P2: clasificación de riesgo
        ├── train_clustering.py      <- P3: clustering de estaciones
        └── predict.py               <- Inferencia (plantilla, pendiente)
```
