# Plan de acción · Entrega 4: API con FastAPI y contenedor

- **Entrega:** domingo 8 de noviembre de 2026.
- **Ventana de trabajo:** lunes 2 a domingo 8 (7 días).
- **Capacidad:** ~2.5 h efectivas por persona entre semana y ~5 h el sábado. El domingo es
  colchón: revisión final y envío, sin trabajo nuevo.
- **Requisitos:** [deliverable4-requirements.md](deliverable4-requirements.md).

## Registro de decisiones

Cerradas por Manuel el lunes 2 y pendientes de confirmar con el equipo en la reunión de arranque.

| # | Decisión | Propuesta |
|:---|:---|:---|
| 4.1 | ✅ Modelo que se despliega | **P2 riesgo** (`mibici-equipo05-p2-riesgo@champion`). Recibe 18 columnas numéricas planas que se validan fácil con Pydantic, y la respuesta es una clase con probabilidades, que es fácil de explicar. P1 depende de clases propias y del histórico de la estación. P3 asigna grupos a estaciones, algo que no se le pide a una API una vez por petición |
| 4.2 | ✅ Cómo llega el modelo al contenedor | **Opción B, camino principal: el modelo va dentro de la imagen.** `make export-model` descarga `models:/mibici-equipo05-p2-riesgo/<versión>` del Registry a `models/p2-riesgo/`, junto con un archivo de metadatos (nombre, versión, `source_run_id`, `git_commit`, `data_version`), y el Containerfile lo copia. El contenedor corre sin MLflow. `models/` sigue fuera de Git: quien construye la imagen exporta antes. Se descartó la opción A como camino principal porque no hay MLflow compartido: solo funcionaría en la máquina que registró el champion |
| 4.3 | ✅ Carga en vivo desde el Registry | **Opción A, documentada como alternativa.** Si `MODEL_URI` empieza con `models:/`, la API lo carga del Registry al arrancar. El Makefile recibe `MLFLOW_HOST ?= 127.0.0.1` y para la demo se corre `make mlflow-server MLFLOW_HOST=0.0.0.0` (sin contraseña: solo mientras dura la demo). El contenedor usa `http://host.containers.internal:5001` en Podman o `http://host.docker.internal:5001` en Docker. Hay que probar si MLflow pide `--allowed-hosts` por el header `Host`. Se descartó un MLflow hospedado (DagsHub) para esta entrega |
| 4.4 | ✅ Motor de contenedores | **Podman** es el motor oficial: los comandos del README y de `make image`, `make container` y `make smoke` usan `podman`. El README también da el comando equivalente de Docker, y el mismo Containerfile sirve para los dos. Si un integrante solo tiene Docker, hace con Docker la prueba en la segunda máquina (WP-5), y eso cuenta como evidencia de que funciona en otro entorno |
| 4.5 | ✅ Estrategia de despliegue | **Inferencia en tiempo real** (REST, petición–respuesta). Las predicciones dependen de los flujos de la última hora, así que conviene calcularlas cuando se piden y no por adelantado. `POST /predict` acepta lotes, así que un proceso horario puede evaluar las ~300 estaciones en una sola llamada (micro-lote) y cubrir el escenario batch de la redistribución sin otro sistema. El modelo predice en milisegundos, así que tener el servicio siempre encendido cuesta poco. El README reconoce que el batch existe y explica por qué se eligió tiempo real |
| 4.6 | ✅ Dependencias de la imagen | **Separar las dependencias en grupos.** `dependencies` se queda con lo que el código usa para entrenar y predecir (mlflow, lightgbm, scikit-learn, pandas, numpy, pyarrow, matplotlib, loguru, typer, python-dotenv, tqdm). El resto pasa a grupos: `notebooks` (jupyterlab, notebook, ipython, folium, seaborn), `docs` (mkdocs), `dev` (pytest, ruff, virtualenv, bandit…) y `api` (fastapi, uvicorn). Con `[tool.uv] default-groups` las laptops siguen instalando todo. El contenedor corre `uv sync --frozen --no-dev --group api`, con las versiones exactas de `uv.lock`, y `make security` audita la API. Se descartó el `requirements.txt` del modelo porque MLflow lo infiere (el `log_model` no pasa `pip_requirements`) y puede quedar incompleto, y se descartó instalar el proyecto completo porque mete Jupyter y mkdocs en la imagen |
| 4.7 | ✅ Host del modelo | **Oscar** (dueño de P2) entrena el champion de P2, lo registra en su `mlflow.db` y corre `make export-model`. Como `models/` no está en Git: **(b)** para el trabajo de la semana, Oscar comparte la carpeta `models/p2-riesgo/` por la carpeta del equipo, con su archivo de metadatos (versión, run, commit), y **(a)** en WP-5, Emilio reproduce el entrenamiento desde `main` con las semillas fijas, registra, exporta y construye la imagen. Así se prueba la reproducibilidad completa, del entrenamiento al contenedor |

---

## 1. Diagnóstico: qué tenemos y qué falta

Revisión del repositorio al 2 de noviembre (`main` en `71b6c42`, merge del PR #28):

| Requisito | Estado | Brecha |
|:---|:---|:---|
| Modelo en el Model Registry | ⚠️ `registry.py` registra con aliases y linaje, y hace el pre-flight con `input_example` | Cada quien tiene su propio `mlflow.db`, y en el de Manuel no hay modelos registrados. Alguien tiene que entrenar y registrar el champion de P2 esta semana |
| Modelo que carga fuera del repo | ✅ `log_model_with_signature` guarda `code_paths` del paquete, firma e `input_example` | Comprobar que carga dentro de un contenedor limpio (sin el repo) |
| Script de inferencia | ❌ `modeling/predict.py` sigue siendo la plantilla de Cookiecutter | Reemplazarlo por el loader que usa la API, o borrarlo |
| API FastAPI | ❌ No existe; `fastapi` y `uvicorn` no están en las dependencias | Crear `tc5061_proyecto_equipo5/api/` |
| Esquemas Pydantic | ❌ | Entrada con las 18 `FEATURE_COLUMNS` de `train_classification.py`, salida con clase y probabilidades |
| Containerfile y `.containerignore` | ❌ | Crearlos |
| Make targets | ❌ | `export-model`, `api`, `image`, `container`, `smoke` |
| Pruebas de la API | ❌ | `TestClient` con un modelo falso (sin MLflow); hoy pasan 275 pruebas y deben seguir pasando |
| README | ⚠️ Cubre las entregas 1 a 3 | Nueva sección: decisión 4.2, build, run, `curl` con su respuesta real y la estrategia de despliegue |
| Git colaborativo (10%) | ⚠️ Pesa más Manuel (75 commits) que Emilio (16) | Cada integrante tiene un paquete con commits propios, abre ≥1 PR y revisa ≥1 |

### Problemas técnicos a resolver

1. **El servidor de MLflow escucha en `127.0.0.1`** (`make mlflow-server`). Desde un contenedor
   no se llega a esa dirección, ni siquiera con `host.containers.internal`. Por eso se propone
   empaquetar el modelo (4.2). Si además se muestra la carga en vivo, hay que levantar el
   servidor con `--host 0.0.0.0` solo para esa demostración.
2. **Puerto 5001, no 5000.** El enunciado usa 5000 como ejemplo, pero en macOS ese puerto lo
   ocupa AirPlay. El README debe decirlo.
3. **Versiones iguales a las del entrenamiento.** El modelo se serializó con scikit-learn 1.9.1,
   LightGBM 4.7.0 y pandas 3.0.6. Si en la imagen hay otras versiones, el modelo puede no cargar
   o predecir distinto. Por eso se instala el `requirements.txt` del modelo (4.6) con Python
   3.13, igual que `.python-version`.
4. **LightGBM necesita `libgomp1`**, que `python:3.13-slim` no trae. Hay que instalarla con
   `apt-get` en el Containerfile.
5. **Arquitectura.** En una Mac M se construye una imagen arm64. Para que funcione "en cualquier
   entorno" hay que probarla también con `--platform linux/amd64`, o al menos documentar el
   comando.
6. **La lógica de clases de P2.** El champion decide la clase con umbrales propios
   (`_labels_from_masks`). La API no debe repetir esa lógica: debe llamar a
   `pyfunc.predict` y traducir 0/1/2 con `CLASS_NAMES_ES`. Si el champion fuera
   `PersistenceRiskClassifier`, solo usa `net_flow_lag2`, pero la API pide igual las 18
   columnas, que son las de la firma.
7. **Podman en macOS** necesita `podman machine init` y `podman machine start`. Va en el README
   como prerrequisito.
8. **Seguridad (entrega 3).** La imagen corre con un usuario sin privilegios y sin secretos. Las
   dependencias nuevas pasan `make security`. `bandit` revisa `api/`.

---

## 2. Diseño propuesto

```
tc5061_proyecto_equipo5/api/
├── __init__.py
├── schemas.py   <- StationFeatures, PredictRequest, Prediction, PredictResponse, ModelInfo
├── model.py     <- carga MODEL_URI una vez al arrancar (ruta local o models:/), expone metadatos
└── main.py      <- app FastAPI: GET /health, GET /model-info, POST /predict
scripts/export_model.py  <- descarga models:/<nombre>/<versión> a models/p2-riesgo/
Containerfile
.containerignore
```

**Contrato de la API**

- `GET /health` responde `{"status": "ok"}` cuando el modelo está cargado. Si no, responde 503.
- `GET /model-info` devuelve nombre, versión, alias, `source_run_id`, `git_commit` y
  `data_version` (del `MLmodel` o de los tags de la versión). Es la evidencia del criterio de 20%.
- `POST /predict`: el cuerpo es `{"instances": [StationFeatures, ...]}` con 1 a 500
  elementos. La respuesta trae una predicción por instancia (`label`, `label_id`,
  `probabilities` si el modelo las da) y la versión del modelo que respondió.

**Validación (`StationFeatures`)**: `extra="forbid"`, para que una columna mal escrita falle.
`capacity > 0`. Latitud y longitud dentro del área de Guadalajara. `hour_sin` y `hour_cos` en
[-1, 1]. `day_of_week` en 0–6. `is_weekend` e `is_holiday` en {0, 1}. Un validador que
rechace `is_weekend` inconsistente con `day_of_week`. Llegadas y salidas ≥ 0. Si algo no
cumple, FastAPI responde 422 con el campo que falló. Los rangos salen de los datos de 2024
(`classification_features.parquet`) y no se inventan.

---

## 3. Paquetes de trabajo

| WP | Responsable | Revisa | Entregable | Horas |
|:---|:---|:---|:---|:---:|
| WP-D Dependencias | Fernando | Manuel | PR propio, lunes o martes, antes del Containerfile: grupos en `pyproject.toml`, `default-groups`, `python_env.yaml` recortado igual que `dependencies` (lo exige `tests/test_dependencies.py`), `uv.lock` regenerado y `make test` y `make security` en verde. Avisar al equipo que corra `uv sync` | 1.5 |
| WP-0 Champion de P2 | Oscar | Manuel | Entrenar y seleccionar P2 desde `main` limpio, registrarlo con `-P alias=champion` y anotar el run ID, la versión y el commit en el documento del equipo. `scripts/export_model.py` y `make export-model`. Compartir `models/p2-riesgo/` con el equipo (4.7b) | 3 |
| WP-1 API | Manuel | Oscar | `api/model.py` y `api/main.py`, dependencias `fastapi` y `uvicorn` en `pyproject.toml` y `uv.lock`, `make api` (uvicorn local). Reemplazar o borrar `modeling/predict.py` | 4 |
| WP-2 Esquemas y pruebas | Emilio | Manuel | `api/schemas.py` con los rangos tomados de los datos, y `tests/test_api.py` (casos válidos, 422 por campo faltante, extra o fuera de rango, lote vacío o demasiado grande, 503 sin modelo) con un modelo falso | 4 |
| WP-3 Contenedor | Fernando | Emilio | `Containerfile` (slim, `libgomp1`, usuario sin root, `HEALTHCHECK`, `EXPOSE 8000`), `.containerignore`, `make image`, `make container` y `make smoke` (`curl` a `/health` y a `/predict`). `MLFLOW_HOST` en `make mlflow-server` y prueba de la opción A (4.3). Probar con Podman y con Docker | 5 |
| WP-4 README y evidencia | Manuel | Fernando | Sección "API y contenedor (Entrega 4)": prerrequisitos, decisiones 4.2 (modelo en la imagen) y 4.3 (alternativa con Registry en vivo y sus pasos), comandos, petición y respuesta reales, estrategia de despliegue, capturas en `reports/figures/entrega4/` y tabla de integrantes actualizada | 3 |
| WP-5 Prueba en otra máquina | Emilio | Oscar | En una máquina distinta al host, clonar `main`, generar las features de P2, entrenar y registrar con `mlflow run` (semillas fijas), correr `make export-model`, construir la imagen y correr `make smoke`. Comparar las métricas y una predicción contra las del champion de Oscar. Esta es la evidencia de reproducibilidad del criterio de 30% | 2.5 |

Así cada integrante tiene al menos un PR propio y una revisión, que es lo que pide el criterio de
Git. A Emilio le toca WP-2 y WP-5 para equilibrar sus commits.

**Dependencias**

- WP-1 y WP-2 empiezan el lunes en paralelo. El contrato de `schemas.py` se acuerda en la
  reunión.
- WP-1 puede probar con cualquier run local de P2 mientras WP-0 termina.
- WP-D va primero porque cambia `uv.lock`: WP-1 agrega FastAPI sobre el grupo `api` y WP-3 lo
  instala en la imagen.
- WP-3 necesita `models/p2-riesgo/` (WP-0) y la app (WP-1).
- WP-4 necesita la imagen funcionando (WP-3) para copiar la respuesta real.

---

## 4. Calendario

| Día | Trabajo | Hito |
|:---|:---|:---|
| Lun 2 | Reunión de 30 min: cerrar 4.1 a 4.7 y el contrato de `schemas.py`. Oscar empieza WP-0, Manuel WP-1 y Emilio WP-2 | Decisiones registradas |
| Mar 3 | WP-0 registra el champion. Termina el primer borrador de WP-1 y WP-2. Fernando arma el Containerfile con un modelo local | Champion de P2 en el Registry y su versión anotada |
| Mié 4 | `make export-model`. WP-1 y WP-2 se integran y `uv run uvicorn` responde | PR de WP-1 y PR de WP-2 abiertos |
| Jue 5 | Merge de WP-1 y WP-2. WP-3 construye la imagen con el modelo exportado y `make smoke` pasa | Imagen funcionando en el host |
| Vie 6 | Merge de WP-3. WP-5 en la máquina de Emilio. Manuel empieza el README | Reproducible en una segunda máquina |
| Sáb 7 | WP-4: README, capturas, `make test`, `make security`, prueba en amd64. Merge final a `main` | `main` listo para entregar |
| Dom 8 | Colchón. Recorrer el README desde cero en un clon limpio y enviar | Entregado |

---

## 5. Checklist de entrega (contra la rúbrica)

- [ ] **Registry (20%)**: la imagen contiene `models:/mibici-equipo05-p2-riesgo/<versión>` y
      `/model-info` muestra el nombre, la versión y el linaje. El README explica la opción
      elegida (4.2) y cómo usar el Registry en vivo (4.3).
- [ ] **API (30%)**: `/predict` funciona con una o varias instancias, Pydantic valida entrada y
      salida (`response_model`), los errores responden 422 o 503, `/docs` muestra los ejemplos y
      `tests/test_api.py` está en verde.
- [ ] **Contenedor (30%)**: `podman build` y `podman run` sin errores, `make smoke` en verde en
      dos máquinas y el comando equivalente de Docker documentado.
- [ ] **README (10%)**: prerrequisitos, build, run, petición `curl` y respuesta real, y la
      justificación de tiempo real frente a lotes.
- [ ] **Git (10%)**: un PR por integrante con su revisión y la tabla de integrantes actualizada.
- [ ] `make test`, `make lint` y `make security` en verde sobre `main`.

## 6. Pensando en la semana siguiente (microservicios)

Toda la configuración entra por variables de entorno (`MODEL_URI`, `MLFLOW_TRACKING_URI`,
`PORT`). La API no guarda estado. `/health` sirve como sonda para el orquestador. Con esto, la
siguiente entrega puede agregar P1 o P3 como otro servicio con la misma estructura.
