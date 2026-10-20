# Plan de acción · Entrega 2: MLflow Project y Tracking Avanzado

- **Entrega:** domingo 25 de octubre de 2026.
- **Ventana de trabajo:** lunes 19 a domingo 25 (7 días).
- **Capacidad real:** unas 4 h/día por integrante de lunes a viernes, compartidas con otras
  materias. Se planea con **~2.5 h efectivas por persona entre semana** y ~5 h el sábado. El
  domingo es colchón: revisión final y envío, sin trabajo nuevo.
- **Requisitos:** [entrega2-requerimientos.md](entrega2-requerimientos.md).

## Registro de decisiones

Reunión de arranque del lunes 19 de octubre
([agenda](entrega2-agenda-kickoff.md), sección 2).

| # | Decisión | Detalle |
|:---|:---|:---|
| 2.1 | ✅ **Un host de evidencia** | Cada quien desarrolla y prueba en su servidor local. Las corridas que cuentan como evidencia se lanzan en una sola máquina desde `main` limpio, para que todo quede en un solo `mlflow.db`. Persona: *por confirmar* |
| 2.2 | ✅ **Los baselines se vuelven a correr** | Se relanzan con `mlflow run` y `stage_of_work=baseline` (mismo código, partición y `data_version`). Los 9 runs de la entrega 1 se quedan con el tag `stage_of_work=legacy_e1`, y el reporte muestra que las métricas coinciden |
| — | ✅ **Reasignación** | WP-P1 (demanda) pasa a **Emilio**; WP-P3 (grupos) pasa a **Manuel** |
| 2.3 | ⏳ Pendiente | Stages o aliases en el Registry (propuesta: aliases) |
| 2.4 | ⏳ Pendiente | Uso de validación y test para elegir en P1 y P2 (propuesta: elegir solo con validación y registrar test en todas las candidatas) |
| 2.5 | ⏳ Pendiente | Ward en P3 (propuesta: envolverlo con `predict`) |
| 2.6 | ⏳ Pendiente | Formato del reporte |
| 2.7 | ⏳ Pendiente | Tag `question`, entry points y borrado de ramas |

Carga resultante: Manuel tiene WP-A y WP-P3 (~9 h), además de ser el host de evidencia si se
confirma la propuesta original. Si el host es otra persona, conviene que sea alguien con el
viernes y el sábado libres.

---

## 1. Diagnóstico: qué tenemos y qué falta

Revisión del repositorio al 19 de octubre (rama `docs/readme`, último merge #11):

| Requisito | Estado actual | Brecha |
|:---|:---|:---|
| Ingesta y manifest (`dataset.py`, `references/viajes_manifest.json`) | ✅ Hecho | Ninguna |
| `features.py` y `config.py` | ✅ Hecho | `config.py` no tiene los defaults de entrenamiento (semilla, carpetas de features por pregunta), que el `MLproject` debe tomar de ahí |
| Baseline por pregunta en `mibici-equipo05` | ✅ 9 runs en `mlflow.db` (3 por pregunta) | Se lanzaron con `python -m`, no con `mlflow run`, y no tienen los tags nuevos ni los artefactos nuevos |
| Servidor de tracking con base de datos | ❌ Tracking local sin `MLFLOW_TRACKING_URI` | Levantar `mlflow server`, apuntar los scripts a él y documentarlo. `mlflow.db` ya está en `.gitignore` ✅ |
| `MLproject` y `python_env.yaml` | ❌ No existen | Crear ambos. El entorno sale de `uv.lock` |
| Entry point `train` con `question`, `model` e hiperparámetros | ❌ Hay un script por pregunta, con los modelos fijos en el código | Crear un dispatcher `modeling/train.py` y parametrizar los tres scripts |
| ≥3 candidatas adicionales al baseline por pregunta | ⚠️ Solo hay 1–2 candidatas por pregunta | Agregar algoritmos o variantes de hiperparámetros |
| Búsqueda con nested runs | ❌ No existe | Corrida padre y una hija por combinación, al menos una por pregunta (recomendado) |
| Tags `question`, `model_family`, `stage_of_work` | ⚠️ `question` vale `regression`, `classification` o `clustering` | Agregar `model_family` y `stage_of_work`. Unificar `question` en `p1`, `p2` y `p3` |
| `log_model` con firma y `input_example` | ❌ Ningún `log_model` pasa firma ni ejemplo | Agregar `infer_signature` e `input_example` |
| Manifest como artefacto y `mlflow.log_input` | ❌ Solo se guarda `data_version` como parámetro | Guardar el manifest en `data/` y el dataset de train con `log_input` |
| Gráficas por tarea | ⚠️ P1: real vs. predicho ✅, sin residuales. P2: matriz de confusión ✅, sin curva PR. P3: centroides y mapa ✅ | P1 residuales, P2 curva precisión-recall |
| Configuración de la corrida como artefacto | ❌ | Guardar en `config/` los parámetros resueltos, `MLproject` y `python_env.yaml` |
| Subcarpetas `plots/`, `data/`, `config/` | ❌ Todo va a la raíz de artefactos | `log_artifact(..., artifact_path="plots")` |
| Model Registry, aliases y carga por `models:/…@champion` | ❌ | Script `modeling/registry.py` |
| Reporte de experimentos con capturas y APA 7 | ❌ | Redactarlo |
| PR por integrante (cada quien abre ≥1 y revisa ≥1) | — | Planearlo desde hoy (sección 5) |

### Problemas técnicos detectados que hay que resolver

1. **P2 evalúa test en todas las corridas** y guarda la matriz de confusión de *test*. Para que
   la selección sea defendible (rúbrica: justificación), P2 debe elegir con validación, igual que
   P1, y evaluar test solo en el modelo elegido o en las candidatas finales. Documenten el cambio
   como corrección de protocolo, no como cambio de métrica.
2. **Ward (P3) no tiene `predict`** y por eso no registra modelo. No se puede llevar al Registry
   así. Opciones: (a) el champion de P3 es K-Means, o (b) envolver Ward en un estimador que asigna
   al centroide de train más cercano (la lógica ya existe en `labels_for_splits`).
3. **P1 se serializa con cloudpickle e incluye clases propias** (`SeasonalMeanRegressor`,
   `NonNegativeRegressor`, `make_preprocessor`). Cargar `models:/…@champion` fuera del repo falla
   si el paquete no se puede importar. Hay que pasar `code_paths=["tc5061_proyecto_equipo5"]` en
   `log_model` (o instalar el paquete en el entorno) y **verificar la carga desde otra carpeta**.
4. **Un solo tracking para el equipo.** Con un servidor local en cada laptop, las corridas
   quedan repartidas en cuatro `mlflow.db`, y la comparación, el Registry y las capturas necesitan
   una sola base. **Decidido (2.1): un integrante es el "host de evidencia".** Cada quien
   desarrolla y prueba en su servidor local, y las corridas finales se lanzan en la máquina host
   desde `main` limpio (`git_dirty=false`).
5. **Baselines.** **Decidido (2.2).** Para que *todas* las corridas de esta entrega salgan de
   `mlflow run` y tengan los mismos tags y artefactos, se vuelven a lanzar los baselines con `stage_of_work=baseline`
   (mismo código, misma partición, mismo `data_version`). Los 9 runs viejos se quedan como
   referencia con el tag `stage_of_work=legacy_e1`. El reporte muestra que las métricas
   coinciden.
6. **Puerto 5000 en macOS** lo ocupa AirPlay. Se usa el 5001, como ya dice el README.
7. **`mlflow run` y el entorno.** El `MLproject` declara `python_env: python_env.yaml` (lo pide
   la rúbrica). Para iterar rápido se usa `--env-manager local` dentro del `.venv` de uv. Antes
   del viernes se valida **una vez** que el entorno declarado se construye desde un clon limpio
   (`virtualenv` necesita `pyenv` con Python 3.13).
8. ~~**`gh` no está instalado**~~ Resuelto el 19 de octubre: `gh` está instalado y autenticado
   en la máquina de Manuel.
9. **Tiempo de entrenamiento de P1.** Los runs de la entrega 1 no lo registran, porque el `fit`
   corre antes de `mlflow.start_run`. Estimado con los huecos entre runs consecutivos en
   `mlflow.db`, el LightGBM de P1 (2.3 M filas de train) entrena en unos **12 s**. Es una
   estimación imprecisa (un hueco salió negativo porque el run seleccionado se reabre para
   evaluar test), pero indica segundos o pocos minutos por modelo. Unas 15 corridas de P1
   caben con holgura, y repetirlas si aparece un error no amenaza el calendario.

---

## 2. Decisiones de diseño (cerrarlas el lunes)

| Tema | Propuesta | Por qué |
|:---|:---|:---|
| Servidor | `mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5001` y `MLFLOW_TRACKING_URI=http://127.0.0.1:5001` en `.env` (con un `.env.example` versionado) | `config.py` ya llama `load_dotenv()`. `mlartifacts/` ya está en `.gitignore` |
| Entry points | `ingest`, `features` (`question`), `train` (`question`, `model`, hiperparámetros), `search` (`question`, `model`, grid), `register` (opcional) | `features` evita el "paso manual" que la rúbrica penaliza al reproducir desde un clon limpio |
| Parámetros de `train` | `question`, `model`, `stage` y un conjunto genérico: `n_estimators`, `learning_rate`, `num_leaves`, `alpha`, `C`, `n_clusters`, `seed`. Con default `-1` o `none`, el script usa el valor de `config.py` | Cumple "parámetros = pregunta, modelo e hiperparámetros" y no repite valores en el `MLproject` |
| Defaults | Nuevos en `config.py`: `SEED = 42`, `CLUSTERING_FEATURES_DIR` y los grids de búsqueda | "Valores por defecto tomados de `config.py`" |
| Tags | `team`, `question` (`p1`/`p2`/`p3`), `task` (`regression`/…), `model_family`, `stage_of_work` (`baseline`/`search`/`candidate`), `git_commit`, `git_dirty`, `selected` | Filtrar en la UI: `tags.question = 'p1' and tags.stage_of_work = 'candidate'` |
| Nombres de runs | `p1-candidate-lightgbm-lr0.03`, `p2-search-lightgbm` (padre), `p2-search-lightgbm-03` (hija) | Se leen bien en las capturas |
| Ciclo de vida | **Aliases**: `champion` (producción) y `challenger` (segunda mejor). Tags de versión: `question`, `data_version`, `git_commit`, `validation_status` | Las *stages* están deprecadas desde MLflow 2.9. Los aliases son el esquema vigente en MLflow 3.17 y permiten promover o hacer rollback sin cambiar el código que consume |
| Nombres en el Registry | `mibici-equipo05-p1-demanda`, `mibici-equipo05-p2-riesgo`, `mibici-equipo05-p3-grupos` | Los que sugiere el enunciado |
| Reporte | `reports/entrega2_reporte_experimentos.md` y su PDF; capturas en `reports/figures/entrega2/` | Mismo lugar que los reportes de la entrega 1 |

---

## 3. Paquetes de trabajo

Cada paquete es un PR. Estimaciones en horas efectivas.

### WP-A · Infraestructura MLflow Project (Manuel · ~5 h)
Rama: `feat/mlflow-project`.
- [ ] `python_env.yaml`: `python: "3.13"`, `build_dependencies: [pip]` y `dependencies` con
      las mismas versiones fijas de `pyproject.toml`, más `-e .` para que el paquete se importe.
      Un test en `tests/test_dependencies.py` compara ambos archivos.
- [ ] `MLproject` en la raíz con los entry points de la sección 2. `ingest` ejecuta
      `python tc5061_proyecto_equipo5/dataset.py download && … clean && … manifest`.
- [ ] `modeling/train.py`: dispatcher de typer que recibe `--question` y llama a la función de
      la pregunta con `model` e hiperparámetros.
- [ ] `common.py`:
  - `setup_experiment()` respeta `MLFLOW_TRACKING_URI` (falla con un mensaje claro si el
    servidor no responde).
  - `base_tags(question, model_family, stage)` con los tags nuevos.
  - `log_run_context(params)`: guarda `references/viajes_manifest.json` en `data/` y la
    configuración resuelta (`run_config.json`, `MLproject`, `python_env.yaml`) en `config/`.
  - `log_training_input(df, source, name)`: `mlflow.data.from_pandas(...)` con una muestra o
    solo el esquema y el digest, nunca columnas personales, más `mlflow.log_input(..., context="train")`.
  - `log_model_with_signature(model, X_example, flavor)`: `infer_signature`, `input_example` de
    5 filas y `code_paths` para los estimadores propios.
- [ ] `.env.example` con `MLFLOW_TRACKING_URI=http://127.0.0.1:5001`. Target `make mlflow-server`.
- [ ] Tests unitarios de los helpers nuevos (mock de MLflow, como en `test_common.py`).
- **Hecho cuando:** `mlflow run . -e train -P question=p1 -P model=seasonal_mean --env-manager local`
  aparece en la UI con tags, `data/`, `config/` y un modelo con firma.

### WP-P1 · Experimentación P1 demanda (Emilio · ~5 h)
Rama: `feat/p1-experiments`.
- [ ] Parametrizar `train_regression.py`: una corrida = un modelo (`seasonal_mean`, `ridge`,
      `lightgbm`, `hist_gb` o `random_forest`) con sus hiperparámetros desde la CLI.
- [ ] Gráficas en `plots/`: real vs. predicho (ya existe), **residuales vs. predicho** e
      **histograma de residuales**, y error por hora del día (opcional, útil para el análisis).
- [ ] Búsqueda anidada `search`: LightGBM Poisson con `learning_rate` × `num_leaves` ×
      `n_estimators` (6–9 combinaciones). El padre guarda el mejor `val_wape` y la tabla de la
      búsqueda.
- [ ] ≥3 candidatas además de los baselines: p. ej. LightGBM Poisson ajustado (mejor de la
      búsqueda), LightGBM con `objective=tweedie`, `HistGradientBoostingRegressor(loss="poisson")`
      y Ridge con otro `alpha`.
- [ ] Mover `model.fit` **dentro** de `mlflow.start_run` (hoy corre antes, así que la duración
      del run no incluye el entrenamiento) y registrar `fit_seconds` y `predict_ms_per_1k` como
      métricas (para el criterio de costo). Aplica igual a P2 y P3.
- [ ] Confirmar con `fit_seconds` de la primera corrida que el grid elegido cabe. La estimación
      de la entrega 1 es de segundos por modelo (ver problema técnico 9).

### WP-P2 · Experimentación P2 riesgo (Oscar · ~5 h)
Rama: `feat/p2-experiments`.
- [ ] Parametrizar `train_classification.py` igual que P1.
- [ ] Corregir el protocolo: la selección se hace con validación, y la matriz de confusión y la
      curva PR se guardan para validación (y para test solo en las finalistas).
- [ ] Gráficas en `plots/`: matriz de confusión (normalizada y absoluta) y **curva
      precisión-recall one-vs-rest** para desabasto y saturación.
- [ ] Búsqueda anidada: LightGBM (`num_leaves`, `learning_rate`, `min_child_samples`) o
      regresión logística con `C`.
- [ ] ≥3 candidatas: LightGBM ajustado, LightGBM sin `class_weight` con umbral calibrado en
      validación, `HistGradientBoostingClassifier(class_weight="balanced")` y Random Forest
      balanceado.
- [ ] Métrica de negocio explícita: recall de desabasto y de saturación con la precisión
      mínima aceptable. Esto alimenta la justificación.

### WP-P3 · Experimentación P3 grupos (Manuel · ~4 h)
Rama: `feat/p3-experiments`.
- [ ] Parametrizar `train_clustering.py` (`model` = `kmeans`, `ward`, `gmm` o `minibatch_kmeans`,
      con `n_clusters`).
- [ ] Búsqueda anidada sobre K (3–8) para K-Means. `choose_k` ya calcula esto, solo hay que
      convertirlo en hijas.
- [ ] ≥3 candidatas: K-Means con el mejor K, Ward con el mejor K (envuelto para poder hacer
      `predict`), GaussianMixture y K-Means con features estandarizadas o sin estandarizar.
- [ ] Gráficas en `plots/`: centroides y mapa (ya existen), y silhouette por muestra.
- [ ] Métrica de estabilidad: ARI entre semillas o entre las asignaciones de validación y train.
      Sin etiquetas, es el argumento principal de "estabilidad".
- [ ] Revisar que `cluster_sizes.csv` y los perfiles no expongan datos personales (solo
      estaciones).

### WP-R · Registry, reproducibilidad y README (Fernando · ~5 h)
Rama: `feat/model-registry`, y después `docs/readme-entrega2`.
- [ ] `modeling/registry.py` (entry point `register`): recibe `question` y `run_id`, registra
      `runs:/<run_id>/model` con el nombre de la sección 2, pone la descripción del modelo y de la
      versión (métrica, baseline, `data_version`, commit), los tags de versión y los aliases
      `champion` y `challenger`.
- [ ] Verificación de carga: `mlflow.pyfunc.load_model("models:/mibici-equipo05-p1-demanda@champion")`
      y una predicción sobre el `input_example`, **desde una carpeta fuera del repo**. Agregar un
      test o un comando `verify`.
- [ ] Prueba de clon limpio (miércoles o jueves): `git clone` en `/tmp`, seguir el README al pie
      de la letra y anotar cada paso que falle.
- [ ] README: comando del servidor, variable `MLFLOW_TRACKING_URI`, comandos `mlflow run` para
      ingesta, features y el modelo elegido de cada pregunta, esquema de aliases y por qué, y
      actualizar la tabla de integrantes.

### WP-D · Reporte de experimentos (Emilio y Oscar en redacción, todos aportan · ~6 h)
Rama: `docs/experiment-report`.
- [ ] Esqueleto del reporte desde el martes: secciones del punto 10 del enunciado.
- [ ] Tabla por pregunta: run, modelo, hiperparámetros clave, métricas de val y test, Δ contra
      el baseline (%), tiempo de entrenamiento. Se genera con `mlflow.search_runs`, que da un
      script reproducible y evita copiar a mano.
- [ ] Análisis y justificación por pregunta (rúbrica 15%): (1) mejora sobre el baseline,
      (2) métrica ligada a la decisión de negocio (P1 WAPE para dimensionar el rebalanceo, P2
      recall de desabasto y saturación, P3 interpretabilidad y estabilidad de los grupos),
      (3) estabilidad (varianza entre semillas o entre val y test), (4) costo de entrenamiento e
      inferencia.
- [ ] Capturas: (a) comparación de corridas con nested runs expandidas, (b) artefactos de una
      candidata con las carpetas `plots/`, `data/` y `config/` y la firma visible, (c) los tres
      modelos registrados con el alias `champion` y su linaje (run de origen, tags `git_commit` y
      `data_version`).
- [ ] Referencias APA 7: documentación de MLflow (Projects, Tracking, Model Registry), Ke et al.
      (2017) para LightGBM, Pedregosa et al. (2011) para scikit-learn, datos abiertos de MiBici y
      Rousseeuw (1987) para silhouette.
- [ ] Lenguaje profesional e inclusivo; revisión cruzada por otro integrante.

---

## 4. Calendario

Las horas son por persona. **Congelamiento de código: viernes 23 a las 21:00.**

### Lunes 19 · Arranque (~1.5 h por persona)
- 30 min de kickoff (llamada): leer este plan, cerrar las decisiones de la sección 2, confirmar
  responsables y quién es el host de evidencia.
- Manuel: abre `feat/mlflow-project`, levanta el servidor y crea el esqueleto del `MLproject` y
  de `python_env.yaml`.
- Fernando: instala `pyenv` con 3.13 y prepara la prueba del entorno declarado.
- Oscar y Emilio: leen sus scripts (`train_classification.py` y `train_regression.py`) y anotan los
  cambios de parámetros que necesitan. Manuel hace lo mismo con `train_clustering.py`.
- Todos: `git pull`, `uv sync`, verificar que tienen datos (`manifest --check`) y features.

### Martes 20 · Infraestructura lista (~2.5 h)
- Manuel: termina WP-A y abre el **PR #1** a mediodía. **Revisa Fernando.** Es el bloqueo
  crítico: sin `common.py` nuevo, los demás no pueden empezar.
- Oscar, Emilio y Manuel: mientras tanto, parametrizan sus scripts en ramas basadas en
  `feat/mlflow-project`.
- Emilio: esqueleto del reporte.
- **Hito:** PR #1 mergeado antes de terminar el día.

### Miércoles 21 · Experimentación en ramas (~2.5 h)
- Owners de P1, P2 y P3: gráficas nuevas, búsqueda anidada y candidatas. Prueban en su servidor
  local con un grid reducido.
- Fernando: WP-R (`registry.py`) probado contra los runs locales. Prueba de clon limpio de
  `ingest`, `features` y `train`.
- **Hito:** las tres ramas de experimentos corren de punta a punta en local.

### Jueves 22 · PRs y revisiones (~2.5 h)
- Abren PRs: `feat/p1-experiments` (revisa Oscar), `feat/p2-experiments` (revisa Emilio),
  `feat/p3-experiments` (revisa Emilio) y `feat/model-registry` (revisa Oscar).
- Correcciones de revisión. `make lint` y `uv run pytest` en verde antes de pedir revisión.
- Fernando: README con los comandos (rama `docs/readme-entrega2`, todavía sin resultados).
- **Hito:** cada integrante abrió ≥1 PR y revisó ≥1.

### Viernes 23 · Merge y corridas de evidencia (~3 h)
- Mañana o tarde: mergear los PRs pendientes. **21:00 congelamiento de código.**
- Host de evidencia, desde `main` limpio (`git status` vacío):
  1. Etiquetar los runs viejos como `legacy_e1`.
  2. `mlflow run` de los baselines (P1, P2, P3) con `stage=baseline`.
  3. Búsquedas anidadas (una por pregunta).
  4. Candidatas (≥3 por pregunta).
- Si `fit_seconds` resulta mucho mayor que la estimación, lanzar la búsqueda de P1 en segundo
  plano durante la noche.
- Script `reports/entrega2_tabla.py` (o notebook) que genera las tablas comparativas.

### Sábado 24 · Selección, Registry y reporte (~5 h)
- Mañana (todos, 1 h en llamada): revisar la UI juntos y elegir champion y challenger de cada
  pregunta con los cuatro criterios.
- Fernando y host: `mlflow run . -e register` para las tres preguntas, verificar la carga por
  alias y tomar capturas.
- Emilio y Oscar: análisis y justificación en el reporte.
- Manuel: README final con los comandos para reproducir el modelo elegido de cada pregunta.
- Fernando: tabla de integrantes y contribuciones.
- **Hito:** borrador completo del reporte a las 20:00.

### Domingo 25 · Revisión y envío (~2 h)
- Revisión cruzada del reporte (ortografía, APA 7, todas las capturas).
- PR final `docs/experiment-report` revisado y mergeado.
- Prueba de humo final: clon limpio y `mlflow run . -e train` de un modelo elegido.
- Exportar a PDF y entregar **antes de las 20:00** (dejar margen hasta el cierre).

---

## 5. Matriz de Pull Requests

| PR | Rama | Autor | Revisor | Día |
|:---|:---|:---|:---|:---|
| 1 | `feat/mlflow-project` | Manuel | Fernando | Mar 20 |
| 2 | `feat/p1-experiments` | Emilio | Oscar | Jue 22 |
| 3 | `feat/p2-experiments` | Oscar | Emilio | Jue 22 |
| 4 | `feat/p3-experiments` | Manuel | Emilio | Jue 22 |
| 5 | `feat/model-registry` | Fernando | Oscar | Jue 22 |
| 6 | `docs/readme-entrega2` | Fernando | Manuel | Sáb 24 |
| 7 | `docs/experiment-report` | Emilio / Oscar | Fernando | Dom 25 |

Historial de la entrega 1 (PRs #1–#13): todo entró por PR y `main` exige 1 aprobación.

Cada integrante abre ≥1 PR y revisa ≥1 (5% de la rúbrica). Las revisiones deben dejar
comentarios reales en GitHub, no solo "Approve": eso es la evidencia.

---

## 6. Comandos objetivo (para el README y el reporte)

```bash
# Servidor (terminal aparte)
uv run mlflow server --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5001
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001

# Desde un clon limpio
uv run mlflow run . -e ingest
uv run mlflow run . -e features -P question=p1      # p2, p3
uv run mlflow run . -e train -P question=p1 -P model=lightgbm -P n_estimators=600 -P learning_rate=0.03
uv run mlflow run . -e search -P question=p2 -P model=lightgbm
uv run mlflow run . -e register -P question=p1 -P run_id=<RUN_ID>

# Verificar producción
uv run python -c "import mlflow; m = mlflow.pyfunc.load_model('models:/mibici-equipo05-p1-demanda@champion'); print(m.metadata.signature)"
```

(`--env-manager local` cuando se ejecuta dentro del `.venv` de uv. Sin esa opción, MLflow
construye el entorno de `python_env.yaml`.)

---

## 7. Checklist final contra la rúbrica

**MLflow Project y reproducibilidad (15%)**
- [ ] `MLproject` con entorno y entry points `ingest` y `train` parametrizados (más `features`, `search` y `register`).
- [ ] Defaults leídos de `config.py`, sin rutas, fechas ni semillas repetidas en el `MLproject`.
- [ ] Todas las corridas de evidencia tienen el tag `mlflow.project.entryPoint` (prueba de que salieron de `mlflow run`).
- [ ] Clon limpio reproducido con el README, sin pasos manuales.
- [ ] `git log --all -- '*.db' '*.parquet' 'mlruns' 'mlartifacts' '*.pkl'` vacío.

**Comparación de experimentos (20%)**
- [ ] ≥3 candidatas además del baseline en P1, P2 y P3.
- [ ] ≥1 búsqueda con nested runs (idealmente una por pregunta).
- [ ] Tags `question`, `model_family`, `stage_of_work`, `team`, `git_commit` y `git_dirty=false`.
- [ ] Misma partición temporal y mismas métricas que la entrega 1. El cambio de protocolo de P2 está justificado.

**Artefactos (15%)**
- [ ] Modelo con firma e `input_example` en cada candidata.
- [ ] `data/viajes_manifest.json` y `log_input` del set de train.
- [ ] `plots/` con las gráficas de cada tarea (P1 residuales, P2 matriz de confusión y curva PR, P3 mapa y perfiles).
- [ ] `config/` con la configuración resuelta.
- [ ] Sin columnas personales (usuario, género, año de nacimiento) en ningún artefacto ni input.

**Model Registry (20%)**
- [ ] Tres modelos con nombre claro y descripción.
- [ ] Linaje: run de origen, experimento, `git_commit` y `data_version` en la versión.
- [ ] Alias `champion` en cada uno (y `challenger` opcional).
- [ ] Carga verificada por `models:/<nombre>@champion`.
- [ ] El esquema de aliases está justificado en el README y en el reporte.

**Justificación (15%):** mejora contra el baseline, métrica de negocio, estabilidad y costo, para cada pregunta.

**Git/GitHub (5%):** matriz de PRs cumplida y nada directo a `main`.

**Reporte (10%):** enlace al repo, comandos, tablas, análisis, esquema de ciclo de vida, las 3 capturas y APA 7.

---

## 8. Riesgos y mitigación

| Riesgo | Prob. | Mitigación |
|:---|:---|:---|
| PR #1 (infraestructura) se atrasa y bloquea a todos | Media | Es lo primero que se hace. Los demás trabajan en ramas basadas en la de Manuel desde el martes |
| La búsqueda de P1 tarda demasiado | Baja | La entrega 1 sugiere ~12 s por LightGBM (problema técnico 9). Confirmar con `fit_seconds` en la primera corrida. Si no cabe: grid ≤9 combinaciones, `n_jobs=-1` y lanzarla de noche |
| Corridas en 4 servidores distintos | Baja | Decidido: un host de evidencia (decisión 2.1) |
| El modelo de P1 no carga desde el Registry (clases propias) | Media | `code_paths` y prueba de carga fuera del repo el miércoles |
| El entorno `python_env.yaml` no se construye (pyenv o LightGBM con libomp) | Media | Probarlo el miércoles. README con el requisito `brew install libomp`. Fallback documentado con `--env-manager local` |
| Corridas con `git_dirty=true` | Media | Congelamiento a las 21:00 del viernes y `git status` antes de lanzar |
| Otra materia absorbe a un integrante | Alta | Cada WP tiene un respaldo: WP-A ↔ Fernando, P1 ↔ Manuel (autor del script original), P2 ↔ Emilio, P3 ↔ Oscar (autor del script original), Registry ↔ Oscar |
| Capturas tomadas antes de tener los aliases finales | Baja | Tomarlas al final del sábado, después del registro |

---

## 9. Definición de "terminado"

La entrega está lista cuando alguien que no participó puede clonar el repo, seguir el README,
reproducir la ingesta y el modelo elegido de cada pregunta con `mlflow run`, abrir la UI, ver las
corridas comparables contra el baseline y cargar los tres modelos por
`models:/mibici-equipo05-pX-…@champion`, y el reporte explica con evidencia por qué se eligió
cada uno.
