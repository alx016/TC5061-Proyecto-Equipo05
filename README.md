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

Requisitos: Python 3.13, [uv](https://docs.astral.sh/uv/) y Git. LightGBM necesita además
OpenMP: en macOS `brew install libomp`; en Linux o WSL `sudo apt install libgomp1`.

```bash
git clone https://github.com/alx016/TC5061-Proyecto-Equipo05.git
cd TC5061-Proyecto-Equipo05
uv sync                 # crea .venv con las versiones fijadas en uv.lock
uv run pytest           # verifica la instalación
```

Los comandos de este README usan `uv run`; si se activa el entorno (`source .venv/bin/activate`)
basta con `python`. En un clon nuevo, usar una terminal sin otro entorno activado (o `deactivate`)
para que `uv` instale en el `.venv` del clon.

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

## MLflow Project (Entrega 2)

Una sola vez, copiar las variables de MLflow:

```bash
cp .env.example .env
```

En una terminal aparte, levantar el servidor de tracking y dejarlo abierto:

```bash
make mlflow-server      # http://127.0.0.1:5001
```

En cada terminal nueva, cargar las variables y correr los entry points:

```bash
set -a; source .env; set +a
uv run mlflow run . -e ingest --env-manager local
uv run mlflow run . -e features -P question=p1 --env-manager local
uv run mlflow run . -e train -P question=p1 -P model=seasonal_mean -P stage=baseline --env-manager local
uv run mlflow run . -e search -P question=p1 --env-manager local
```

- `Connection refused` en el puerto 5001: el servidor no está corriendo.
- `active experiment ID does not match`: faltó cargar `.env` (`MLFLOW_EXPERIMENT_NAME`).
- `--env-manager local` usa el `.venv` del proyecto; sin esa opción MLflow construye el
  entorno de `python_env.yaml` con pyenv y virtualenv.

## Experimento base (Entrega 1)

Cada pregunta tiene su propio script de entrenamiento y todos registran sus runs en el
experimento `mibici-equipo05` de MLflow. Las corridas de evidencia usan el servidor local
(`make mlflow-server`, SQLite en `mlflow.db` y artefactos en `mlartifacts/`) y
`MLFLOW_TRACKING_URI=http://127.0.0.1:5001`. El pipeline completo, después de la ingesta:

```bash
# 0. Servidor de tracking (terminal aparte)
make mlflow-server
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
export MLFLOW_EXPERIMENT_NAME=mibici-equipo05

# 1. Features (los artefactos se escriben en data/processed/)
uv run python -m tc5061_proyecto_equipo5.features main --problem p1 --prepare-ready \
  --input-dir data/interim/viajes --output-dir data/processed/fe_2024_v4
uv run python -m tc5061_proyecto_equipo5.features classification
uv run python -m tc5061_proyecto_equipo5.features main --problem p3 --prepare-ready \
  --input-dir data/interim/viajes --output-dir data/processed/fe_2024_v4

# 2. Entrenamiento
uv run python -m tc5061_proyecto_equipo5.modeling.train_regression regression

# P2: un modelo por run; train no evalúa test
uv run mlflow run . -e train -P question=p2 -P model=lightgbm \
  -P stage=baseline --env-manager local
uv run mlflow run . -e train -P question=p2 -P model=lightgbm_unweighted \
  -P stage=candidate --env-manager local
uv run mlflow run . -e search -P question=p2 -P model=lightgbm --env-manager local
uv run mlflow run . -e select -P question=p2 --env-manager local

uv run python -m tc5061_proyecto_equipo5.modeling.train_clustering clustering

# 3. Resultados
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

`features main --prepare-ready` no sobrescribe una carpeta existente; para regenerar los
artefactos hay que mover o borrar antes `data/processed/fe_2024_v4/p1` o `p3`. En macOS el
puerto 5000 lo ocupa AirPlay, por eso se usa el 5001.

| Pregunta | Script | Candidatos | Selección |
|:---|:---|:---|:---|
| P1 | `modeling/train_regression.py` | Promedio estacional, Ridge, LightGBM Poisson, LightGBM Tweedie, HistGradientBoosting Poisson | `regression` elige por WAPE de validación y evalúa solo su finalista en test |
| P2 | `modeling/train_classification.py` | Persistencia, clase mayoritaria, regresión logística, LightGBM balanceado y sin balancear, HistGradientBoosting y Random Forest | Mayor macro-F1 de validación con guardrails de recall de desabasto (`>= 0.85`) y precisión mínima; si ninguno los cumple, se elige el mayor macro-F1 y se marca `selection_fallback=true`. Solo el run `selected=true` se evalúa en test |
| P3 | `modeling/train_clustering.py` | K-Means con K=4 de referencia, K-Means con el mejor K, Ward | K con la mayor silhouette de validación |

P2 no usa `predict` directamente: entrena con `predict_proba`, calibra un umbral para
desabasto y otro para saturación usando **solo validación**, y aplica esos umbrales antes de
calcular las métricas. `train` y `search` nunca tocan diciembre; `select` carga el modelo
elegido, evalúa test una sola vez y guarda las matrices de confusión, las curvas PR y el
`classification_report.txt` en `plots/test/`. `select` se ejecuta como entry point para que la
corrida quede registrada como proyecto MLflow; rechaza runs con `git_dirty=true`, así que las
pruebas locales sobre un árbol sucio deben usar el módulo directo con `--allow-dirty`.

### Etapas de experimentación P1

El catálogo de `regression` conserva promedio estacional, Ridge y LightGBM Poisson como
referencias de la Entrega 1, con `stage_of_work=baseline`. La referencia LightGBM se llama
`regression-baseline-lightgbm`.

El entrypoint `train` ejecuta una configuración por corrida y usa `stage=candidate` por
defecto. Para repetir la referencia LightGBM sin ajustar hiperparámetros:

```bash
uv run mlflow run . -e train -P question=p1 -P model=lightgbm -P stage=baseline \
  --env-manager local
```

Esa corrida se llama `p1-baseline-lightgbm`. Para una variante ajustada, usar
`-P stage=candidate` junto con los parámetros `hp_*`; no etiquetarla como baseline.

En `search`, todas las Trials conservan `stage_of_work=search`. Solo la ganadora recibe
`selected=true`, en correspondencia con `best_run_id` de la Study y la fila seleccionada
del CSV. Para localizarla, filtrar por `question=p1`, `stage_of_work=search` y la Study
(`mlflow.parentRunId`), además de `selected=true`. Este tag indica el mínimo WAPE del grid:
no equivale a seleccionar el finalista entre familias, evaluar diciembre o promover un champion.

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

## Model Registry

Cada pregunta tiene un modelo registrado: `mibici-equipo05-p1-demanda`,
`mibici-equipo05-p2-riesgo` y `mibici-equipo05-p3-grupos`. El entry point `register` toma
el run que ganó el `search` o `select` de su pregunta (tag `selected=true`) y lo registra
en el modelo que corresponde a su tag `question`:

```bash
uv run mlflow run . -e register -P run_id=<run ganador> --env-manager local                     # entra como challenger
uv run mlflow run . -e register -P run_id=<run ganador> -P alias=champion --env-manager local    # lo promueve
```

- **Aliases en lugar de stages.** Los stages están deprecados desde MLflow 2.9. `champion`
  equivale a Production y `challenger` a un candidato en comparación. El código que consume el
  modelo siempre carga `models:/<nombre>@champion`; promover o regresar a otra versión es
  mover el alias, sin cambiar ese código.
- Un registro nuevo entra como `challenger`, para no reemplazar al `champion` por accidente.
  Promover reusa la misma versión y le quita `challenger`.
- Cada versión guarda el linaje de su run: `source_run_id`, `git_commit`, `git_dirty`,
  `data_version`, `features_data_version`, `selected` y los tags `selection_*` de su `select`.
- De P1 se registra el Trial con `selected=true`, no la Study: la Study no guarda modelo.

Para usar el modelo desde cualquier carpeta:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5001")
model = mlflow.pyfunc.load_model("models:/mibici-equipo05-p1-demanda@champion")
```

El modelo carga fuera del repositorio porque se guarda con el código del paquete
(`code_paths`). En MLflow 3, `runs:/<id>/model` apunta al logged model del run; el aviso
"has no artifacts at artifact path 'model'" al registrar es normal.

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
├── MLproject                  <- Entry points de MLflow: ingest, features, train, search, select, register
├── python_env.yaml            <- Entorno de Python del MLflow Project
├── .env.example               <- Variables de MLflow (tracking URI y experimento)
├── mlflow.db, mlartifacts/    <- Servidor de tracking local; fuera de Git
└── tc5061_proyecto_equipo5/   <- Código fuente
    ├── config.py              <- Rutas, años y constantes del estudio
    ├── dataset.py             <- Descarga, limpieza y manifest de los datos
    ├── features.py            <- Panel estación-hora y recetas de features de P1, P2 y P3
    ├── plots.py               <- Figuras de justificación
    └── modeling/
        ├── common.py                <- Utilidades compartidas de MLflow y reproducibilidad
        ├── train.py                 <- Entry point train: un modelo de una pregunta por run
        ├── search.py                <- Entry point search: Study con Trials anidados
        ├── registry.py              <- Entry point register: Model Registry y aliases
        ├── train_regression.py      <- P1: regresión de demanda
        ├── train_classification.py  <- P2: clasificación de riesgo
        ├── train_clustering.py      <- P3: clustering de estaciones
        └── predict.py               <- Inferencia (plantilla, pendiente)
```
