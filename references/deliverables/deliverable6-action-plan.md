# Plan de acción · Entrega 6 (Actividad 9 / Entregable 7): Microservicios, CI/CD y FinOps

- **Entrega:** domingo 22 de noviembre de 2026, 23:59 (hora de Ciudad de México).
- **Ventana de trabajo:** martes 17 (noche) a domingo 22 de noviembre de 2026 (~5.5 días). El domingo es colchón: verificación en clon limpio, revisión del PDF y envío en Canvas, sin código nuevo.
- **Esfuerzo estimado:** ~27.5 h en total (~6.5–7.5 h por integrante).
- **Base en `main`:** `a3dfafb` (merge del PR #49, reporte de Entrega 5).
- **Requisitos:** especificación de Actividad 9 / Entregable 7 (desglosada localmente en `local/d6-code.md`, `local/d6-doc.md`, `local/d6-evaluation.md` y `local/d6-references.md`).

## Registro de decisiones

Propuesta de arranque al martes 17 de noviembre; consenso validado entre los 4 frentes antes de abrir los primeros PR:

| # | Decisión | Detalle |
|:---|:---|:---|
| 6.1 | ✅ **Topología en Docker Compose (sin Redis/RQ)** | Dos servicios obligatorios: **`mlflow`** (Tracking Server + Model Registry) y **`api`** (FastAPI P2), conectados por una red bridge dedicada (`mibici-net`), volumen persistente (`mlflow_data`) y `depends_on` con `condition: service_healthy`. **Redis + worker de RQ** se descarta para la implementación porque el enunciado lo declara opcional y P2 ya justificó inferencia HTTP síncrona de baja latencia (1–500 estaciones/llamada) en la Entrega 4; se analiza como alternativa de escalamiento en la sección FinOps. |
| 6.2 | ✅ **Modo de modelo por defecto (`MODEL_URI`)** | Por defecto en `docker-compose.yml`, la API carga el bundle inmutable `/app/models/p2-riesgo` (champion v3) para que `docker compose up -d --build` responda `200 OK` en `/health` y `/predict` desde un clon limpio aunque `mlflow_data` arranque vacío. Al mismo tiempo se inyecta `MLFLOW_TRACKING_URI=http://mlflow:5001` fijo dentro de `api`, permitiendo apuntar `MODEL_URI=models:/mibici-equipo05-p2-riesgo@champion` al servicio `mlflow` en vivo y validar ambos modos. |
| 6.3 | ✅ **Construcción en un solo comando sin error por `models/p2-riesgo/`** | Hoy `Containerfile` ejecuta `COPY models/p2-riesgo/ ./models/p2-riesgo/`, pero esa carpeta está en `.gitignore` y falta en un clon limpio (lo versionado es `models/p2-champion-v3.zip`, que extrae a `models/p2-riesgo/`). Se actualizan `.containerignore`, `.dockerignore` y `Containerfile` para aceptar ambos y ejecutar `python -m zipfile -e models/p2-champion-v3.zip .` durante el build si no existe `models/p2-riesgo/registry_metadata.json`, eliminando el `.zip` de la imagen final. |
| 6.4 | ✅ **Puertos, variables y compatibilidad Docker / Podman** | `mlflow` expone `${MLFLOW_PORT:-5001}:5001` (evitando el puerto `5000` usado por AirPlay en macOS) y `api` expone `${API_PORT:-8000}:8000`. El servidor de MLflow 3.17.2 se lanza con `--host 0.0.0.0 --port 5001 --workers 1 --allowed-hosts 'mlflow:5001,mlflow,localhost:5001,127.0.0.1:5001'`. `docker-compose.yml` fija `container_name: mibici-api` (`mem_limit: 1024m`, `cpus: 1.0`) y `container_name: mibici-mlflow` (`mem_limit: 512m`, `cpus: 0.5`) con claves de nivel superior (compatibles con Docker y Podman), **hardcodea `MLFLOW_TRACKING_URI=http://mlflow:5001`** (para que un `.env` local con `127.0.0.1:5001` no rompa el DNS interno) y **no** usa `env_file: .env`. En `.env.example` se mantiene `127.0.0.1:5001` activo para CLI local y `# MODEL_URI=/app/models/p2-riesgo` comentado. |
| 6.5 | ✅ **Pipeline CI/CD en GitHub Actions (100% rúbrica)** | Workflow `.github/workflows/ci.yml` bajo propiedad de **Manuel (`alx016`, dueño del repo en GitHub)** para gestionar permisos de Actions, secretos y GHCR (`packages: write`). Disparado en `push` a `main`, `pull_request` hacia `main` y `workflow_dispatch`. Incluye: (1) `lint-and-security` (`make lint` + `make security`); (2) **`test-matrix`** (`fast` / `slow`, caché `uv` + Python 3.12 para DeepChecks tras `make prepare-model`); (3) `compose-and-smoke` (`if: hashFiles('docker-compose.yml') != ''`, caché Buildx `type=gha` para `Containerfile` y `Containerfile.mlflow`, levantamiento `docker compose up -d --build` en clon limpio y verificación HTTP); (4) `publish-image` (`if: needs.compose-and-smoke.result == 'success' && github.event_name == 'push' && github.ref == 'refs/heads/main'`) a `ghcr.io/alx016/mibici-p2-api`, resumen en `$GITHUB_STEP_SUMMARY` y badge en `README.md`. |
| 6.6 | ✅ **Pruebas de carga con Locust (`load_tests/`), grupo `perf` y contrato fijo** | El directorio se nombra **`load_tests/`** (no `locust/`, para no sombrear la librería `locust` en `sys.path`). `locust` va en `[dependency-groups] perf` **fuera** de `[tool.uv] default-groups` para no contaminar `make security` (`pip-audit`) ni `python_env.yaml`; `load_tests/payloads.py` usa solo stdlib/Pydantic y las corridas usan `uv run --group perf ...`. Se agrega `/reports/load/*.html` a `.gitignore` para evitar falsos positivos en `detect-secrets-hook`. Se congelan 6 slugs de escenario, tags `name="/predict?batch=1\|20\|365"` y columnas CSV/JSON exactas entre `WP-4b`/`WP-4c`, `WP-1d` y `WP-2c`. |
| 6.7 | ✅ **Sub-WPs lineales desbloqueables, precios pre-acordados y host Linux** | Se fijan de arranque los parámetros base FinOps (`AWS EC2 t3.small`, `us-east-1`, `\$0.0208/h` + EBS gp3 `\$0.08/GB-mes`) y referencias APA 7 base en `WP-0`, eliminando todo bloqueo desde `WP-3b` hacia `WP-1d`/`WP-4c`/`WP-4d`/`WP-D`. **Fernando** asume Compose (`WP-1a`/`1b`/`1c`) más `scripts/finops_calc.py` (`WP-1d`); **Oscar** asume Locust + `collect_load_metrics.py` + corrida oficial única en Linux Docker nativo (`WP-4a..4d`); **Manuel** asume CI/CD + `plot_load_and_finops.py` (regenerando PNGs el viernes tras `WP-4c`) + PDF final + entrega en Canvas (`WP-0`, `WP-2a..2c`, `WP-D`); **Emilio** trabaja en tareas de baja complejidad y cero bloqueo crítico (`WP-3a..3d`). |
| 6.8 | ✅ **Formato del reporte PDF y nombre del entregable** | Fuente editable en `reports/TC5061_Operacion_FinOps_Equipo05.md` y PDF final en `reports/TC5061_Operacion_FinOps_Equipo05.pdf`. Tipografía **Arial 11** (o Liberation Sans / Helvetica 11pt equivalente), interlineado **1.15**, **3 a 5 cuartillas exactas de cuerpo principal sin contar anexos** (salto de página explícito antes de `## Anexos` verificado con `pdfinfo` / revisión visual). `Anexo A` se divide en `Anexo A.1` (Compose, Fernando) y `Anexo A.2` (CI/CD, Manuel); los scripts de captura y FinOps auto-emiten tablas para `Anexo B` y `Anexo C`. |

---

## 1. Diagnóstico: qué tenemos y qué falta

Revisión del repositorio al 17 de noviembre de 2026 (`main` en `a3dfafb`, último merge #49):

| Criterio (peso) | Estado actual | Qué existe ya | Brecha por cubrir |
|:---|:---|:---|:---|
| Orquestación con Docker Compose (25%) | 🟡 Base de API lista | `Containerfile` (Python 3.13-slim, `uv`, usuario no-root `app`, `HEALTHCHECK` en `/health`), `.containerignore`, `.dockerignore`, `models/p2-champion-v3.zip`, `api/` (`MODEL_URI` soporta carpeta local o `models:/`), `Makefile` (`prepare-model`, `export-model`, `image`, `container`, `smoke`, `mlflow-server`) | No existe `docker-compose.yml`; falta servicio `mlflow` contenerizado; `Containerfile` falla en clon limpio si no se corrió `make prepare-model`; faltan targets de Compose en `Makefile` y documentación de red/volúmenes/env |
| Pipeline de CI/CD con GitHub Actions (25%) | 🔴 Falta | No existe `.github/workflows/`; localmente pasan 449 pruebas en `tests/` (`445` rápidas + `4` `@pytest.mark.slow`), `make lint` y `make security` | Crear `.github/workflows/ci.yml` con caché `uv` + Python 3.12, matriz de pruebas, seguridad, build de imagen, levantamiento y prueba de humo con `docker compose`, publicación a GHCR en `main` y resumen del run; verificar workflow en verde en GitHub |
| Pruebas de carga con Locust (20%) | 🔴 Falta | `scripts/smoke_request.json` y `api/schemas.py` con validaciones exactas de dominio (1–500 instancias por petición, rangos de Guadalajara, coherencia `day_of_week`/`is_weekend`) | Agregar `locust` en `pyproject.toml`, diseñar `load_tests/locustfile.py` realista para MiBici, script de captura simultánea de `docker stats`, correr escalones de carga + experimento de batching + microbenchmark Opt 3, generar CSVs y gráficas PNG interpretadas |
| Análisis FinOps (20%) | 🔴 Falta | Evidencia previa de demanda de MiBici (365 estaciones activas en 2025, inferencia horaria) y tiempos/arquitectura de Entrega 4 | Medir CPU/RAM en reposo y bajo carga (`docker stats`), derivar fórmula explícita de costo por inferencia (capacidad pico vs. demanda real de MiBici), comparar instancia cloud de referencia con tarifa y fecha citadas, cuantificar optimizaciones basadas en datos |
| Documentación y Git colaborativo (10%) | 🟡 Al día hasta E5 | `README.md` cubre Entregas 1–5; historial de Git equilibrado entre los 4 integrantes; `reports/TC5061_Drift_ABTest_Equipo05.md` marca el estándar editorial | Agregar sección Entrega 6 y actualizar tabla de integrantes en `README.md`; redactar `reports/TC5061_Operacion_FinOps_Equipo05.md` y exportar PDF (3–5 cuartillas + anexos); mantener ≥2 PR y ≥2 revisiones por integrante |

### Problemas técnicos a resolver desde el arranque

1. **Clon limpio con un solo comando (`docker compose up -d --build`).**
   `Containerfile` hoy ejecuta `COPY models/p2-riesgo/ ./models/p2-riesgo/`, pero `models/p2-riesgo/` está en `.gitignore` y se genera solo tras correr `make prepare-model` o `make export-model`. Si el evaluador o un compañero clona el repo y corre `docker compose up --build` directamente, la construcción falla.
   - **Solución:** en `.containerignore` y `.dockerignore`, agregar `!models/p2-champion-v3.zip`. En `Containerfile`, copiar `models/` antes de `USER app` y ejecutar un `RUN` que, si no existe `models/p2-riesgo/registry_metadata.json`, ejecute `python -m zipfile -e models/p2-champion-v3.zip .` y elimine `models/p2-champion-v3.zip` en la misma capa. Así funciona en clon limpio y respeta `models/p2-riesgo/` si alguien corrió `make export-model`.

2. **MLflow 3.17.2 dentro de Docker Compose: `--allowed-hosts`, `--workers` y permisos de volumen.**
   Al consultar `uv run mlflow server --help` en el entorno del repo (`mlflow==3.17.2`):
   - El servidor bloquea cabeceras `Host` que no sean `localhost`, `127.0.0.1` o IP privada literal. Pasar `--allowed-hosts 'mlflow:5001,mlflow,localhost:5001,127.0.0.1:5001'` para permitir `http://mlflow:5001` en la red interna.
   - Usar `--workers 1`, `--backend-store-uri sqlite:////mlflow/mlflow.db` y `--artifacts-destination /mlflow/mlartifacts`, con volumen nombrado `mlflow_data:/mlflow`.
   - **Permisos de volumen con usuario no-root:** en `Containerfile.mlflow`, crear `mkdir -p /mlflow && chown -R app:app /mlflow` **antes** de `USER app` (UID 1000) para evitar `PermissionError` al inicializar `mlflow_data`.

3. **Healthcheck de `mlflow`, DNS interno vs `.env` local y límites de RAM.**
   - `modeling/common.py:63` ya confirma el endpoint `/health` de MLflow 3.17.2 en `:5001`. Al partir de `python:3.13-slim`, el `healthcheck` usa `python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5001/health', timeout=4)"`.
   - Docker Compose lee automáticamente `.env` en la raíz del repo y `config.py:7` llama `load_dotenv()`. Por eso `docker-compose.yml` debe **hardcodear `MLFLOW_TRACKING_URI=http://mlflow:5001`** en `api`, mientras `.env.example` conserva `127.0.0.1:5001` activo para CLI local y `# MODEL_URI=/app/models/p2-riesgo` **comentado**.
   - Bajo 120 usuarios concurrentes con lotes de 365 estaciones, `512m` en `api` arriesga OOM-kill; fijar `mem_limit: 1024m` / `cpus: 1.0` en `mibici-api` y `mem_limit: 512m` / `cpus: 0.5` en `mibici-mlflow`.

4. **Contrato de `tests/test_dependencies.py`, `locust` shadowing y `make security`.**
   `tests/test_dependencies.py` exige dependencias fijadas con `==` en `[project.dependencies]` y coincidencia exacta con `python_env.yaml`. Además:
   - **Evitar shadowing:** nunca nombrar una carpeta `locust/` en la raíz (rompe `from locust import HttpUser`); usar **`load_tests/`**.
   - **Aislar `pip-audit`:** `Makefile:84-85` ejecuta `uv export --quiet --no-emit-project` **sin `--no-default-groups`**. Por ello `perf = ["locust==..."]` debe quedar **fuera** de `[tool.uv] default-groups` (`pyproject.toml:84`), `load_tests/payloads.py` debe usar únicamente stdlib/Pydantic (sin `import locust`) para que `tests/test_locust_payloads.py` pase en `uv sync --frozen`, y las pruebas de carga se ejecutan con `uv run --group perf ...`.
   - **Evitar ruido en `detect-secrets-hook` (`Makefile:87`):** añadir `/reports/load/*.html` a `.gitignore` (los reportes HTML de Locust incrustan JS/Base64 de alta entropía) y versionar únicamente CSVs y `finops_summary.json`. Además, considerar que `make security` sobrescribe `reports/security/detect-secrets.json`.

5. **Concurrencia de FastAPI + LightGBM + pandas bajo carga.**
   En `tc5061_proyecto_equipo5/api/main.py`, `predict_risk` es una función síncrona (`def`), por lo que FastAPI la despacha en el threadpool de AnyIO (40 hilos por defecto). Dentro de `api/model.py`, cada llamada arma un `pd.DataFrame`, castea tipos de firma (`astype(SIGNATURE_DTYPES)`) y llama `loaded.model.predict(df)` (LightGBM `predict_proba`).
   - Bajo carga concurrente en Locust con 1 solo worker de Uvicorn (`CMD ["uvicorn", ..., "--port", "8000"]`), varios hilos compiten por el GIL durante la creación de DataFrames y por los hilos OpenMP de LightGBM.
   - **Oportunidad para el análisis:** medir el stack tal cual (1 worker) en la corrida oficial y ejecutar un microbenchmark reproducible en `WP-4b`/`WP-4c` (`pd.DataFrame` + `mlflow.pyfunc` frente a `Booster` + `numpy`) para sustentar la Opt 3 de FinOps con datos medidos.

6. **Pruebas `fast` vs `slow` (`tests/test_data_checks.py`) y `make prepare-model` en CI.**
   - `tests/test_api_model.py::test_real_export_serves_smoke_request` hace `pytest.skip` si no existe `models/p2-riesgo/MLmodel`. En CI, correr `make prepare-model` antes de `pytest` garantiza que el modelo real v3 se pruebe de verdad.
   - En `tests/test_data_checks.py`, además de las 4 pruebas con `@pytest.mark.slow`, las pruebas de las líneas `252-350` también invocan `uv run scripts/data_checks.py` (`python == 3.12`, DeepChecks) sin marca `slow`. Por ello `WP-2a` debe cachear `~/.local/share/uv` (o pre-instalar Python 3.12) en ambos slices de `test-matrix` para mantener el pipeline ágil.

7. **Diferencias de `docker stats` entre Linux y macOS.**
   En macOS (Docker Desktop / Podman Machine), los contenedores corren dentro de una VM Linux con recursos compartidos, y algunas versiones de `podman stats` reportan CPU% relativo a todos los núcleos de la VM. Para que la estimación de FinOps sea consistente, los límites de recursos (`cpus`, `mem_limit` / `deploy.resources.limits`) deben fijarse explícitamente en `docker-compose.yml` y la corrida oficial de Locust + `docker stats` debe ejecutarse en Linux Docker nativo (host de Oscar).

8. **Extensión estricta del PDF (3 a 5 cuartillas sin anexos).**
   La rúbrica/formato especifica: **Tipografía Arial 11 o equivalente, interlineado 1.15, de 3 a 5 cuartillas sin contar anexos**. El cuerpo del documento (Secciones 1 a 6 + Referencias o Referencias al final del cuerpo antes de Anexos — verificar conteo de páginas tras exportar) debe ser denso, visual y conciso; las tablas de configuración completas, el detalle del `docker-compose.yml`, el workflow YAML y logs completos pasan a **Anexos**.

---

## 2. Diseño propuesto y contratos técnicos

```
tc5061-proyecto-equipo05/
├── .github/
│   └── workflows/
│       └── ci.yml                       <- Pipeline CI/CD: lint, pytest, security, compose smoke, GHCR
├── Containerfile                        <- API P2 (fallback a models/p2-champion-v3.zip si falta p2-riesgo/)
├── Containerfile.mlflow                 <- Imagen ligera y reproducible de MLflow 3.17.2 (o stage en Containerfile)
├── .containerignore / .dockerignore     <- Permiten models/p2-champion-v3.zip además de models/p2-riesgo/
├── docker-compose.yml                   <- Servicios mlflow + api, red mibici-net, volumen mlflow_data, healthchecks
├── .env.example                         <- Mantiene 127.0.0.1:5001 para CLI local; overrides Compose comentados
├── .gitignore                           <- Incluye /reports/load/*.html para no disparar detect-secrets-hook
├── Makefile                             <- Targets: up, down, logs, compose-smoke, load-test, load-report (detec. docker/podman)
├── pyproject.toml / uv.lock             <- Grupo `perf` con locust fuera de default-groups (sin tocar project.dependencies ni python_env.yaml)
├── load_tests/                          <- Carpeta nombrada load_tests/ (evita shadowing de `import locust`)
│   ├── __init__.py
│   ├── locustfile.py                    <- Escenario MiBici: /health, /model-info, /predict?batch=1|20|365
│   └── payloads.py                      <- Generador determinista stdlib/Pydantic (sin importar locust)
├── scripts/
│   ├── collect_load_metrics.py          <- Orquesta Locust headless + docker stats -> CSVs + reports/load/anexo_b.md
│   ├── finops_calc.py                   <- Cálculo FinOps parametrizado -> finops_summary.json + reports/load/anexo_c.md
│   └── plot_load_and_finops.py          <- Genera 4 PNGs en reports/figures/entrega6/ desde contrato CSV/JSON
├── tests/
│   ├── test_batch_365_contract.py       <- TestClient (DummyPyFuncModel) lotes 1/20/365 + ZipFile read-only p2-champion-v3.zip
│   ├── test_finops_calc.py              <- Pruebas unitarias de las fórmulas FinOps y salida finops_summary.json
│   └── test_locust_payloads.py          <- Valida payloads de load_tests/payloads.py contra StationFeatures / PredictRequest
├── reports/
│   ├── load/                            <- CSVs de Locust y docker stats + finops_summary.json + anexo_b.md/anexo_c.md (*.html en .gitignore)
│   ├── figures/entrega6/                <- Gráficas PNG a 300 DPI (RPS, latencia p50/p95/p99, error, CPU/RAM, FinOps)
│   ├── TC5061_Operacion_FinOps_Equipo05.md
│   └── TC5061_Operacion_FinOps_Equipo05.pdf   <- Entregable oficial (3–5 cuartillas cuerpo + anexos)
└── README.md                            <- Sección Entrega 6 + tabla de integrantes actualizada
```

### Contrato 1: Orquestación con Docker Compose (`docker-compose.yml`)

- **Servicios:**
  1. `mlflow` (`container_name: mibici-mlflow`):
     - Build desde `Containerfile.mlflow` (Python `3.13-slim` + `uv` + `mlflow==3.17.2`, sin `curl`, creando `mkdir -p /mlflow && chown -R app:app /mlflow` antes de `USER app` UID 1000 para que el volumen nombrado sea escribible).
     - Comando:
       ```text
       mlflow server --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/mlartifacts --host 0.0.0.0 --port 5001 --workers 1 --allowed-hosts 'mlflow:5001,mlflow,localhost:5001,127.0.0.1:5001'
       ```
     - Puertos: `"${MLFLOW_PORT:-5001}:5001"`.
     - Volumen nombrado: `mlflow_data:/mlflow`.
     - Healthcheck con Python `urllib.request.urlopen("http://127.0.0.1:5001/health", timeout=4)` (confirmado en `modeling/common.py:63`).
     - Límites explícitos de nivel superior: `mem_limit: 512m`, `cpus: 0.5` (compatible con `docker compose` y `podman-compose`).
  2. `api` (`container_name: mibici-api`):
     - Build desde `Containerfile` (contexto `.`), imagen `mibici-p2-api:local`.
     - `depends_on`: `mlflow` con `condition: service_healthy`.
     - Puertos: `"${API_PORT:-8000}:8000"`.
     - Variables de entorno (**sin** `env_file: .env`):
       - `MODEL_URI=${MODEL_URI:-/app/models/p2-riesgo}`
       - `MLFLOW_TRACKING_URI=http://mlflow:5001` (**hardcodeado** dentro de Compose para que un `.env` local con `127.0.0.1:5001` no sobrescriba el DNS interno).
       - `MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-mibici-equipo05}`
     - Healthcheck: `python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"` (`interval: 15s`, `timeout: 5s`, `start_period: 20s`, `retries: 5`).
     - Límites explícitos de nivel superior: **`cpus: 1.0`, `mem_limit: 1024m`** (evita OOM-kill en el escalón de 120 usuarios × lotes de 365 estaciones).
- **Red y volúmenes:** red dedicada `mibici-net` (`driver: bridge`), volumen nombrado `mlflow_data`.
- **Comando único:** `docker compose up -d --build` (o `make up`), compatible también con `podman-compose up -d --build`.
- **Prueba de integración con Registry en vivo y Opt 2 (`WP-1b`):** registrar una versión directamente contra el servidor `mlflow` de Compose (`http://127.0.0.1:5001`, sin montar un `./mlflow.db` del host) y levantar `MODEL_URI=models:/mibici-equipo05-p2-riesgo@champion docker compose up -d api` para confirmar que `/model-info` refleja los metadatos servidos por `http://mlflow:5001`. Verificar también `docker compose up -d --no-deps api` con `/app/models/p2-riesgo` para validar Opt 2 (sin MLflow en runtime).

### Contrato 2: Pipeline CI/CD en GitHub Actions (`.github/workflows/ci.yml`)

- **Propiedad del repo:** **Manuel (`alx016`)** configura permisos de Actions (`packages: write`), visibilidad de GHCR y reglas de protección (sin exigir `compose-and-smoke` hasta integrarse `WP-1a`).
- **Disparadores:** `push` a `main`, `pull_request` hacia `main`, y `workflow_dispatch`.
- **Permisos mínimos:** `contents: read` (y `packages: write` solo en el job de publicación a GHCR).
- **Jobs:**
  1. `lint-and-security`:
     - `actions/checkout@v4`, `astral-sh/setup-uv@v5` (`enable-cache: true`, `python-version-file: .python-version`), `uv sync --frozen`
     - `make lint` (`ruff format --check` y `ruff check`)
     - `make security` (`bandit`, `pip-audit`, `detect-secrets-hook` contra `.secrets.baseline`)
  2. `test-matrix` (`strategy.matrix.suite: ["fast", "slow"]`):
     - `actions/checkout@v4`, `astral-sh/setup-uv@v5` (`enable-cache: true` + caché de `~/.local/share/uv` para Python 3.12 usado por `tests/test_data_checks.py`), `uv sync --frozen`
     - `make prepare-model` (para habilitar `test_real_export_serves_smoke_request`)
     - Si `suite == fast`: `uv run pytest tests -m "not slow"` (445+ pruebas)
     - Si `suite == slow`: `uv run pytest tests -m slow` (4 pruebas de DeepChecks)
     - Publicar conteos en `$GITHUB_STEP_SUMMARY`
  3. `compose-and-smoke` (`needs: [lint-and-security, test-matrix]`, `if: hashFiles('docker-compose.yml') != ''`):
     - Se activa automáticamente en cuanto exista `docker-compose.yml` (`WP-1a`); antes queda en `skipped` sin bloquear CI.
     - `actions/checkout@v4` + `docker/setup-buildx-action@v3` (caché GHA cubriendo tanto `Containerfile` como `Containerfile.mlflow`)
     - Construcción y arranque real de la arquitectura completa **sin** ejecutar `make prepare-model` previo (validando regla 6.3 de clon limpio): `docker compose up -d --build`
     - Espera activa con timeout acotado a que `mibici-mlflow` y `mibici-api` pasen a `healthy`
     - Verificación HTTP real contra `http://127.0.0.1:8000`: `GET /health`, `GET /model-info` (`model_name == mibici-equipo05-p2-riesgo`, `model_version == "3"`), `POST /predict` con `scripts/smoke_request.json`
     - En `if: always()`: `docker compose ps` y últimos logs si falla, seguido de `docker compose down -v`
  4. `publish-image` (`needs: compose-and-smoke`, `if: needs.compose-and-smoke.result == 'success' && github.event_name == 'push' && github.ref == 'refs/heads/main'`):
     - Login en `ghcr.io` con `secrets.GITHUB_TOKEN`
     - `docker/metadata-action@v5` + `docker/build-push-action@v6` con caché GHA (`cache-from: type=gha`, `cache-to: type=gha,mode=max`), publicando `ghcr.io/alx016/mibici-p2-api:latest` y `sha-<short>`
     - Escribir resumen estructurado con tag, digest y estado del despliegue en `$GITHUB_STEP_SUMMARY`, más badge de estado en `README.md`.

### Contrato 3: Escenario de carga con Locust (`load_tests/`), monitoreo y esquemas fijos

- **Pertinencia con MiBici (justificación explícita en código y reporte):**
  - El sistema MiBici cuenta con **365 estaciones activas** (dato auditado en Entrega 5) y la API `/predict` acepta lotes de **1 a 500 instancias** (`PredictRequest`).
  - Se modelan dos patrones de tráfico que reflejan la operación real y las pruebas de capacidad:
    1. **Escenario mixto realista (`MiBiciApiUser`, `wait_time = between(0.2, 1.0)`):**
       - `@task(5)` `POST /predict` (`name="/predict?batch=1"`) con **1 estación**.
       - `@task(3)` `POST /predict` (`name="/predict?batch=20"`) con **20 estaciones**.
       - `@task(1)` `POST /predict` (`name="/predict?batch=365"`) con **365 estaciones**.
       - `@task(1)` `GET /model-info` (`name="/model-info"`) y `@task(1)` `GET /health` (`name="/health"`).
    2. **Escenario de comparación de batching (experimento FinOps):**
       - Comparar latencia, RPS efectivos (predicciones/s) y CPU al enviar **1 petición de 365 estaciones** (`batch_1x365`) frente a **365 peticiones de 1 estación** (`batch_365x1`).
- **Validación de respuestas en Locust:** cada tarea valida `status_code == 200` y `len(resp.json()["predictions"]) == N` usando `catch_response=True`, sin payloads malformados que inflen errores 422. Los payloads se generan con variación controlada dentro de `StationFeatures` (`load_tests/payloads.py`, probado con `pytest` en `tests/test_locust_payloads.py`).
- **Slugs congelados y escalones headless (`uv run --group perf ...`):**
  1. `baseline` (**60 s**): contenedores levantados sin tráfico (`api` y `mlflow` en reposo).
  2. `nominal_10u` (**10 usuarios, spawn `2/s`, 2 min**): carga nominal holgada.
  3. `high_50u` (**50 usuarios, spawn `5/s`, 2 min**): carga alta.
  4. `stress_120u` (**120 usuarios, spawn `10/s`, 2 min**, ajustable tras piloto): saturación / rodilla de latencia.
  5. `batch_1x365` y 6. `batch_365x1`: comparativa de batching.
  - Además, **microbenchmark Opt 3** (`pd.DataFrame` + `mlflow.pyfunc` vs `Booster` + `numpy`) ejecutado en `WP-4b`/`WP-4c` para cuantificar la sobrecarga por llamada.
- **Contrato fijo de salidas (`scripts/collect_load_metrics.py` ↔ `scripts/finops_calc.py` ↔ `scripts/plot_load_and_finops.py`):**
  - `reports/load/docker_stats_<slug>.csv`: muestreo cada `1–2 s` normalizando nombres a `api` / `mlflow`, `timestamp` ISO-8601 UTC y flotantes puros (sin `%` ni sufijos `MiB`/`GiB`):
    `timestamp,scenario,container,cpu_pct,mem_usage_mib,mem_limit_mib,mem_pct,net_rx_mb,net_tx_mb,pids`
  - Locust exporta `--csv reports/load/locust_<slug>` (`*_stats.csv`, `*_stats_history.csv`, `*_failures.csv`) y `--html reports/load/locust_<slug>.html` (ignorado en `.gitignore`). En las gráficas, la tasa de error se calcula como `100 * Failure Count / Request Count` excluyendo la fila `Aggregated` cuando aplique desglose por endpoint/lote.
  - `scripts/finops_calc.py` exporta `reports/load/finops_summary.json` con esquema fijo:
    - `baseline`: `{instance_type, hourly_usd, storage_monthly_usd, monthly_usd, peak_req_per_s, peak_pred_per_s, cost_per_req_peak_usd, cost_per_pred_peak_usd, cost_per_1m_pred_peak_usd, cost_per_1m_pred_real_24_7_usd, attribution_pct: {api, mlflow}}`
    - `optimizations[]`: lista de `{id, name, cost_per_1m_pred_usd, monthly_usd, savings_pct}`
    - Emite además `reports/load/anexo_b.md` (desde `collect_load_metrics.py`) y `reports/load/anexo_c.md` (desde `finops_calc.py`) para evitar transcripción manual.
  - CLI fija de graficación: `uv run python scripts/plot_load_and_finops.py --load-dir reports/load --output-dir reports/figures/entrega6` (valores por defecto).
- **Gráficas obligatorias en `reports/figures/entrega6/` (300 DPI desde los CSVs/JSON del contrato):**
  1. `01_rps_y_usuarios_vs_tiempo.png`: RPS totales y por endpoint + usuarios concurrentes a lo largo de los escalones.
  2. `02_latencia_percentiles.png`: p50, p95 y p99 por escalón y desglosadas por tamaño de lote (`?batch=1|20|365`), más tasa de error (%).
  3. `03_recursos_contenedores.png`: CPU % y Memoria (MiB) de `api` y `mlflow` en reposo y cada escalón (`docker stats`).
  4. `04_finops_comparativa_optimizaciones.png`: costo por 1M de inferencias comparando línea base vs escenarios optimizados (batching 365x, sin MLflow en runtime, Opt 3 y scale-to-zero).

### Contrato 4: Metodología FinOps (fórmulas explícitas y trazables)

Para evitar estimaciones vagas ("cuesta poco"), todos los cálculos del reporte deben seguir estas definiciones explícitas, quedar implementados y probados en **`scripts/finops_calc.py` (`WP-1d`)** y guardados en **`reports/load/finops_summary.json` (`WP-4c`)**:

1. **Instancia cloud de referencia (pre-acordada en `WP-0` / `WP-1d` y ampliada en `WP-3b`):**
   - Para eliminar bloqueos de arranque, `WP-0` y `WP-1d` fijan como línea base **`AWS EC2 t3.small`** (2 vCPU, 2 GiB RAM, `\$0.0208/h` bajo demanda en `us-east-1`, coherente con los `1.5` CPUs y `1.5 GiB` asignados en Compose) más almacenamiento EBS `gp3` (`\$0.08/GB-mes`). `WP-3b` amplía la tabla con comparativa multi-proveedor y tarifas públicas exactas para Opt 4 (AWS Lambda / Google Cloud Run), registrando proveedor, región, URL pública y fecha exacta de consulta en APA 7.
2. **Dos unidades de análisis (distinción crítica):**
   - **Costo por petición HTTP (`$/req`):** útil para tráfico interactivo.
   - **Costo por predicción estación-hora (`$/pred` y `$/1M pred`):** la unidad real de negocio en MiBici (una llamada de 365 estaciones produce 365 inferencias).
3. **Tres regímenes de utilización (demuestra consideración de tiempo, recursos y escalabilidad):**
   - **Cota teórica a capacidad sostenible máxima (80% CPU sin errores):**
     $$\text{Costo/req}_{\text{pico}} = \frac{\text{Tarifa horaria instancia (USD/h)}}{\text{RPS}_{\text{sostenible}} \times 3600}$$
     $$\text{Costo/pred}_{\text{pico}} = \frac{\text{Tarifa horaria instancia (USD/h)}}{\text{Predicciones/s}_{\text{sostenibles}} \times 3600}$$
   - **Costo operativo real (siempre encendido 24/7 con demanda real de MiBici):**
     MiBici tiene 365 estaciones evaluadas cada hora $\Rightarrow$ `365 × 24 = 8,760` predicciones/día = `262,800` predicciones/mes (`~730 h/mes`).
     $$\text{Costo/pred}_{\text{real-24/7}} = \frac{\text{Costo mensual fijo (cómputo + disco + red)}}{\text{Predicciones reales/mes}}$$
     Esta comparación expone el **desperdicio por capacidad ociosa** (>99% del tiempo sin uso si solo se ejecuta una corrida horaria) y justifica automáticamente las optimizaciones FinOps.
   - **Costo atribuido por componente (`api` vs `mlflow`):** usando el porcentaje de CPU y RAM observados en `docker stats` entre el contenedor `api` y el contenedor `mlflow`.
4. **Optimizaciones concretas y cuantificadas con datos:**
   - **Opt 1 — Batching por red (`365 estaciones/req` vs `1 estación/req`):** usar la razón medida en Locust entre `Predicciones/s` en lote de 365 y `Predicciones/s` unitario.
   - **Opt 2 — Desacoplar MLflow del camino de inferencia en producción:** servir únicamente el contenedor `api` con el bundle exportado (`/app/models/p2-riesgo`, validado en `WP-1b` con `docker compose up -d --no-deps api`), reduciendo la RAM/CPU fija al eliminar el contenedor `mlflow` en producción.
   - **Opt 3 — Ajuste de concurrencia y eliminación de sobrecarga por petición:** cuantificar mediante microbenchmark reproducible (`WP-4b`/`WP-4c`) el costo de `pd.DataFrame` + `mlflow.pyfunc` por llamada frente a invocación directa del `Booster` con `numpy` o ajuste de `--workers` / `OMP_NUM_THREADS=1`.
   - **Opt 4 — Cómputo bajo demanda / scale-to-zero (Cloud Run / AWS Lambda / ejecución programada):** comparar instancia 24/7 frente a pago por segundo activo (`8,760` inferencias/día ejecutadas en lotes horarios de <1 s), calculando el ahorro porcentual mensual.

### Contrato 5: Reporte oficial (`TC5061_Operacion_FinOps_Equipo05.md` y `.pdf`)

- **Especificaciones obligatorias:**
  - Nombre exacto en Canvas: **`TC5061_Operacion_FinOps_Equipo05.pdf`**
  - Tipografía: **Arial 11** (o Liberation Sans / Helvetica 11pt en Linux/macOS para equivalencia métrica exacta)
  - Interlineado: **1.15**
  - Extensión: **3 a 5 cuartillas exactas sin contar anexos** (apuntar a **4.5–5.0 cuartillas** en el cuerpo principal, insertando un salto de página explícito justo antes de `## Anexos` y verificando la paginación final con `pdfinfo` / revisión visual).
- **Estructura propuesta del documento (con presupuesto de líneas por sección en `WP-0`):**
  - **Encabezado + Resumen ejecutivo** (~0.4 pág.): contexto MiBici P2, enlace al repo y resultado clave (CI en verde, RPS sostenibles, costo por inferencia real y optimizado). *(Manuel, `WP-D`)*
  - **1. Arquitectura operable y orquestación con Docker Compose** (~0.8 pág.): diagrama/topología (`api` + `mlflow`), red, volúmenes, variables de entorno, decisión sobre Redis/RQ y validación en un solo comando. *(Fernando, `WP-1c`)*
  - **2. Automatización CI/CD con GitHub Actions** (~0.6 pág.): etapas del pipeline (`lint`, matriz `pytest` con modelo real, `security`, `compose-smoke`, `GHCR`), cachés aplicadas y evidencia del run en verde. *(Manuel, `WP-2b`)*
  - **3. Evaluación bajo carga con Locust y monitoreo de recursos** (~1.2 pág.): diseño del escenario MiBici (365 estaciones), resultados por escalón, **gráficas de RPS/latencia/errores y CPU/RAM (`docker stats`)**, e interpretación técnica de cuellos de botella (GIL/DataFrame, hilos LightGBM, tamaño de lote). *(Oscar, `WP-4d`)*
  - **4. Análisis FinOps: costo por inferencia y optimizaciones** (~1.3 pág.): instancia cloud de referencia y precios públicos citados *(base `t3.small` pre-acordada en `WP-0`; comparativa ampliada con Opt 4 serverless investigada por Emilio en `WP-3b`)*, cálculo paso a paso (pico vs demanda real 24/7 vs atribución por servicio) y tabla/gráfica comparativa de las 4 optimizaciones cuantificadas con datos. *(Oscar, `WP-4d`)*
  - **5. Conclusiones y referencias (APA 7)** (~0.5 pág.): borrador de síntesis operativa a partir de los bullets del equipo + bibliografía APA 7 (`Gift & Deza, 2021`, `Treveil et al., 2020`, `Wilson, 2022`, `FinOps Foundation`, precios cloud) *(Emilio, `WP-3b`; cierre final Manuel en `WP-D`)*.
  - **Anexos (no cuentan en las 3–5 cuartillas):**
    - **Anexo A.1:** Extractos clave de `docker-compose.yml`, `Containerfile*` y validación Live Registry / Opt 2 *(Fernando, `WP-1c`)*.
    - **Anexo A.2:** Extractos clave de `.github/workflows/ci.yml` + evidencia del pipeline en verde y GHCR *(Manuel, `WP-2b`)*.
    - **Anexo B:** Tablas completas de Locust por endpoint y escalón + lecturas crudas de `docker stats` *(auto-generado en `reports/load/anexo_b.md` por `collect_load_metrics.py`, integrado por Oscar en `WP-4d`)*.
    - **Anexo C:** Memoria de cálculo FinOps y comandos de reproducción *(auto-generado en `reports/load/anexo_c.md` por `finops_calc.py` de Fernando `WP-1d`, integrado tras `WP-4c` por Oscar en `WP-4d`)*.

---

## 3. Paquetes de trabajo

**Principio de diseño — sub-WPs granulares y desbloqueo lineal:**
Cada frente se divide en sub-paquetes pequeños (`WP-0`, `WP-1a..1d`, `WP-2a..2c`, `WP-3a..3d`, `WP-4a..4d`, `WP-D`) para que **entregar los primeros sub-WPs desbloquee de inmediato a los demás sin esperar al cierre del frente completo**. Todos arrancan en paralelo desde `main` (`a3dfafb`), con tareas de baja complejidad y cero bloqueo crítico para Emilio, propiedad de GitHub Actions/GHCR/Canvas en Manuel (repo owner) y corrida oficial única en el Linux Docker de Oscar:

```text
Día 1 (0 bloqueos — todos arrancan contra main a3dfafb):
  ├─ Manuel (owner):  WP-0 (esqueleto MD + precios t3.small/APA base) ───► desbloquea WP-1c, WP-3b, WP-4d
  ├─ Manuel (owner):  WP-2a (CI base + cache Py3.12 + permisos GHCR) ───► CI verde desde el día 1
  ├─ Manuel (owner):  WP-2c (plot_load_and_finops.py con fixtures) ──────► listo antes de la corrida oficial
  ├─ Fernando:        WP-1a (Compose MVP + .zip fallback) ──► WP-1d (finops_calc.py) ──► WP-1b (Makefile + Registry)
  ├─ Oscar:           WP-4a (load_tests/ + perf) ───────────► WP-4b (collect + Opt 3 vs make container)
  └─ Emilio:          WP-3a (.env.example + test 365) ──────► WP-3b (precios/Opt4/APA7/Concl./CSS) [no bloquea]

Integración lineal:
  WP-1a ────────────────────────► WP-2b (CI compose-smoke + GHCR + Sec. 2 + Anexo A.2)
  WP-1a + WP-1d + WP-4a + WP-4b ──► WP-4c (corrida Linux vie.) ──► WP-2c (PNGs reales vie.) ──► WP-4d (Sec. 3 y 4)
  WP-1b + WP-2b ──────────────────► WP-3c (README.md sábado)
  WP-1c + WP-2b + WP-3b + WP-4d ──► WP-D (congelamiento sáb. 18:00 -> PDF 3–5p -> Canvas dom.) ──► WP-3d (QA no bloq.)
```

| WP | Responsable | Revisa | Entregable | Desbloquea a | Horas |
|:---|:---|:---|:---|:---|:---:|
| **WP-0** Esqueleto reporte + base FinOps | Manuel | Fernando | Esqueleto de `reports/TC5061_Operacion_FinOps_Equipo05.md` con presupuesto de líneas por sección, `Anexo A.1`/`A.2` y parámetros base pre-acordados (`t3.small` `\$0.0208/h`, EBS `\$0.08/GB-mes`, refs APA 7 base) | WP-1c, WP-3b, WP-4d | 0.75 |
| **WP-1a** Compose MVP + `.zip` fallback | Fernando | Manuel, Oscar | `Containerfile`, `.containerignore`, `.dockerignore` (fallback `p2-champion-v3.zip`), `Containerfile.mlflow` (`--workers 1`, `--allowed-hosts`, `/mlflow` UID 1000) y `docker-compose.yml` (`mibici-api` `1024m`/`1.0`, `mibici-mlflow` `512m`/`0.5`, sin `env_file`, `MLFLOW_TRACKING_URI=http://mlflow:5001` fijo) | **WP-2b**, **WP-4c** | 2.5 |
| **WP-1d** `finops_calc.py` + tests | Fernando | Oscar, Manuel | `scripts/finops_calc.py` + `tests/test_finops_calc.py` usando baseline `t3.small` pre-acordado; genera `reports/load/finops_summary.json` y auto-formatea `reports/load/anexo_c.md` | **WP-4c**, **WP-2c** (real), **WP-4d**, **WP-D** | 1.5 |
| **WP-1b** Targets `Makefile` + Live Registry | Fernando | Emilio, Oscar | Targets `up`, `down`, `logs`, `compose-smoke`, `load-test`, `load-report` en `Makefile` (detección `docker`/`podman`, `make image` con `.zip` fallback) + prueba Live Registry `:5001` y Opt 2 (`--no-deps api`) | WP-1c, WP-3c | 2.0 |
| **WP-1c** Sección 1 + Anexo A.1 | Fernando | Manuel, Emilio | Redacción de **Sección 1** (topología, red, volúmenes, variables, descarte Redis/RQ) + extractos Compose/Live Registry en **Anexo A.1** (antes del congelamiento sáb. 18:00) | WP-D | 1.5 |
| **WP-2a** CI/CD base + GHCR *(repo owner)* | Manuel | Fernando, Emilio | `.github/workflows/ci.yml` (`lint-and-security`, `test-matrix` `fast`/`slow` con `make prepare-model` y caché `~/.local/share/uv` para Python 3.12 en `test_data_checks.py`, guarda `hashFiles('docker-compose.yml') != ''`, configuración de permisos/secretos GHCR) | — (verde día 1) | 1.75 |
| **WP-2b** Activación `compose-smoke` + GHCR | Manuel | Fernando, Oscar | Caché Buildx (`Containerfile` + `Containerfile.mlflow`), verificación en verde de `compose-and-smoke` y `publish-image` (`ghcr.io/alx016/mibici-p2-api`) tras `WP-1a` + **Sección 2** + **Anexo A.2** | WP-3c, WP-D | 1.5 |
| **WP-2c** Gráficas (fixtures + corrida real vie.) | Manuel | Oscar, Emilio | `scripts/plot_load_and_finops.py` probado días 1–2 con fixtures del contrato CSV/JSON y **re-ejecutado el viernes tras `WP-4c`** para commitear `reports/figures/entrega6/01..04.png` reales a 300 DPI | **WP-4d**, **WP-D** | 1.5 |
| **WP-3a** `.env.example` + test contrato 365 | Emilio | Fernando, Manuel | `.env.example` (mantiene `127.0.0.1:5001` activo y `# MODEL_URI` comentado) + `tests/test_batch_365_contract.py` (`DummyPyFuncModel` con `smoke_request.json` para lotes 1/20/365 + validación `ZipFile` read-only de `p2-champion-v3.zip`) | — (no bloquea) | 1.5 |
| **WP-3b** Precios ampliados, APA 7, Concl. y estilo | Emilio | Oscar, Manuel | Tabla ampliada de precios cloud + tarifas serverless Opt 4 (URL/fecha exactas), prosa comparativa Opt 4, bibliografía APA 7 completa, **borrador Sección 5 (Conclusiones)** sobre bullets del equipo y snippet CSS/exportación PDF (Arial 11, 1.15, salto antes de Anexos) | — (no bloquea; `WP-0` cubre base) | 2.0 |
| **WP-3c** Actualización de `README.md` | Emilio | Fernando, Manuel | Sección Entrega 6 en `README.md` (badge CI, guía `docker compose`, variables `.env`, CI/CD, `load_tests/`, estructura y tabla de integrantes) tras `WP-1b` y `WP-2b` | WP-3d | 1.5 |
| **WP-3d** QA y auditoría rúbrica *(no bloqueante)* | Emilio | Manuel | Auditoría sábado noche / domingo: `make lint`, `uv run pytest tests -m "not slow"`, validación opcional en Docker Desktop macOS y revisión `pdfinfo` / visual de `TC5061_Operacion_FinOps_Equipo05.pdf` | Apoya revisión final | 1.5 |
| **WP-4a** Locust `load_tests/` + seguridad `perf` | Oscar | Manuel, Fernando | Grupo `perf` en `pyproject.toml` + `uv.lock` fuera de `default-groups`, `/reports/load/*.html` en `.gitignore`, `make security` limpio, `load_tests/payloads.py` (stdlib/Pydantic), `load_tests/locustfile.py` (`?batch=1\|20\|365`) y `tests/test_locust_payloads.py` | WP-4c | 2.0 |
| **WP-4b** Captura `docker stats` + microbenchmark Opt 3 | Oscar | Manuel, Fernando | `scripts/collect_load_metrics.py` (CSVs normalizados + `reports/load/anexo_b.md`) + microbenchmark Opt 3 (`pd.DataFrame`/`pyfunc` vs `Booster`/`numpy`), probados contra `make container` en `main` | WP-4c | 1.5 |
| **WP-4c** Corrida oficial en Linux Docker *(viernes)* | Oscar | Fernando, Emilio | Ejecución oficial en Linux Docker tras `WP-1a` + `WP-1d`: 6 slugs (`baseline`..`batch_365x1`) + Opt 3 + invocación `finops_calc.py`; commit de CSVs, `finops_summary.json`, `anexo_b.md` y `anexo_c.md` | **WP-2c** (real), **WP-4d** | 1.5 |
| **WP-4d** Secciones 3 y 4 + Anexos B y C | Oscar | Manuel, Emilio | Redacción de **Sección 3** (latencia, RPS, errores, cuellos GIL/LightGBM) + **Sección 4** (FinOps pico vs 24/7 vs 4 optimizaciones) citando PNGs reales del viernes + integración de **Anexos B y C** | WP-D | 1.5 |
| **WP-D** Integración PDF (3–5p) + entrega Canvas | Manuel | Fernando, Emilio, Oscar | Resumen ejecutivo (~0.4 pág.), integración tras congelamiento sáb. 18:00, ajuste estricto a **3–5 cuartillas (Arial 11, 1.15)** con salto antes de `## Anexos`, generación de `reports/TC5061_Operacion_FinOps_Equipo05.pdf`, smoke final en Linux y **subida a Canvas el domingo** | **Entregado** | 1.5 |

### Carga y balance de Git por integrante (criterio del 10%)

| Integrante | Sub-WPs asignados | Horas totales | PRs sugeridos (2–4 PRs limpios) | Revisa WPs de |
|:---|:---|:---:|:---|:---|
| Fernando (`Fernando cuenta TEC`) | `WP-1a`, `WP-1d`, `WP-1b`, `WP-1c` | **7.5 h** | `feat/docker-compose-core` (`WP-1a`), `feat/finops-calculator` (`WP-1d`), `feat/compose-makefile-registry` (`WP-1b`), `docs/report-compose` (`WP-1c`) | `WP-0`, `WP-2*`, `WP-3*`, `WP-4*` |
| Manuel (`alx016`, *repo owner*) | `WP-0`, `WP-2a`, `WP-2b`, `WP-2c`, `WP-D` | **7.0 h** | `docs/report-skeleton-e6` (`WP-0`), `ci/github-actions` (`WP-2a`/`2b`), `feat/load-finops-plots` (`WP-2c`), `docs/entrega6-pdf` (`WP-D`) | `WP-1*`, `WP-3*`, `WP-4*` |
| Emilio (`Emilio`) | `WP-3a`, `WP-3b`, `WP-3c`, `WP-3d` | **6.5 h** | `feat/env-and-batch-contract-test` (`WP-3a`), `docs/pricing-apa7-conclusions` (`WP-3b`), `docs/readme-entrega6` (`WP-3c`/`3d`) | `WP-1*`, `WP-2*`, `WP-4*` |
| Oscar (`oscarlv`) | `WP-4a`, `WP-4b`, `WP-4c`, `WP-4d` | **6.5 h** | `feat/locust-suite` (`WP-4a`), `feat/capture-and-opt3-bench` (`WP-4b`), `feat/official-linux-load-run` (`WP-4c`), `docs/report-load-finops` (`WP-4d`) | `WP-1*`, `WP-2*`, `WP-3*` |

*(Nota: mantener ≥3 commits significativos y ≥2 PRs propios revisados por integrante asegura evidencia equilibrada en `git shortlog -sn main`.)*

### Un dueño por archivo (evitar conflictos entre PRs)

| Archivo(s) | Dueño exclusivo |
|:---|:---|
| `Containerfile`, `Containerfile.mlflow`, `.containerignore`, `.dockerignore`, `docker-compose.yml`, `Makefile`, `scripts/finops_calc.py`, `tests/test_finops_calc.py` | `WP-1a` / `WP-1d` / `WP-1b` (Fernando) |
| `.github/workflows/ci.yml`, `scripts/plot_load_and_finops.py`, `reports/figures/entrega6/*`, `reports/TC5061_Operacion_FinOps_Equipo05.pdf` | `WP-2a..2c` / `WP-D` (Manuel) |
| `.env.example`, `tests/test_batch_365_contract.py`, `README.md` | `WP-3a` / `WP-3c` (Emilio) |
| `pyproject.toml`, `uv.lock`, `.gitignore` (`/reports/load/*.html`), `.secrets.baseline`, `load_tests/*`, `tests/test_locust_payloads.py`, `scripts/collect_load_metrics.py`, `reports/load/*` | `WP-4a..4c` (Oscar) |
| `reports/TC5061_Operacion_FinOps_Equipo05.md` | Esqueleto inicial `WP-0` (Manuel); luego cada quien edita **únicamente su bloque** (Sec. 1 + Anexo A.1 Fernando; Resumen/Sec. 2/Anexo A.2/cierre Manuel; Precios Opt 4 + APA 7 + borrador Sec. 5 Emilio; Sec. 3/4 + Anexos B/C Oscar) |

---

## 4. Calendario

| Fase | Día | Trabajo | Hito |
|:---|:---|:---|:---|
| **1. Arranque paralelo (cero bloqueos)** | Mar 17 (noche) / Mié 18 | Manuel entrega `WP-0` + `WP-2a` (CI base verde + permisos GHCR) + inicia `WP-2c`. Fernando entrega **`WP-1a`** (`docker-compose.yml` + `.zip` fallback) e inicia `WP-1d`. Oscar entrega `WP-4a` (`load_tests/` + `perf`) y avanza `WP-4b` contra `make container`. Emilio entrega `WP-3a` (`.env.example` + test 365) e inicia `WP-3b`. | `WP-0` y `WP-1a` desbloquean todos los frentes posteriores |
| **2. Desbloqueo CI & calculadora FinOps** | Jue 19 | Tras `WP-1a`: Manuel cierra `WP-2b` (`compose-smoke` + GHCR en verde + Sec. 2/Anexo A.2) y `WP-2c` (fixtures). Fernando cierra **`WP-1d`** (`finops_calc.py`) e inicia `WP-1b`. Oscar cierra `WP-4b` (`collect_load_metrics.py` + Opt 3). Emilio cierra `WP-3b` (precios Opt 4, APA 7, borrador Concl. y estilo PDF). | CI/CD completo en verde + scripts de carga/FinOps/gráficas listos |
| **3. Corrida oficial Linux & PNGs reales** | Vie 20 | Fernando cierra `WP-1b` (`Makefile` + Live Registry + Opt 2 `--no-deps`). **Oscar ejecuta `WP-4c`** en Linux Docker sobre `main` (6 slugs + Opt 3 + `finops_calc.py`), publicando CSVs, `finops_summary.json` y `anexo_b/c.md`. **Ese mismo viernes, Manuel re-ejecuta `WP-2c`** y publica `reports/figures/entrega6/01..04.png` reales. | Evidencia empírica completa (CSVs, JSON, Anexos B/C y 4 PNGs reales) |
| **4. Redacción, congelamiento e integración PDF** | Sáb 21 | Fernando cierra `WP-1c` (Sec. 1 + Anexo A.1) y Oscar cierra `WP-4d` (Sec. 3 y 4 citando los PNGs reales) antes del **congelamiento a las 18:00**. Emilio cierra `WP-3c` (`README.md`). **Manuel cierra `WP-D`** (Resumen ejecutivo, ajuste estricto **3–5 cuartillas, Arial 11, interlineado 1.15**, salto antes de `## Anexos` y PDF final). | Reporte PDF y `README.md` integrados en `main` con CI en verde |
| **5. QA y entrega en Canvas** | Dom 22 | **Emilio ejecuta `WP-3d`** (auditoría QA no bloqueante: `make lint`, `pytest -m "not slow"`, validación opcional en Docker Desktop macOS y revisión `pdfinfo`/visual del PDF). **Manuel** realiza la verificación final en clon limpio Linux y **sube `TC5061_Operacion_FinOps_Equipo05.pdf` a Canvas antes de las 23:59**. | **Entregado** |

---

## 5. Checklist de entrega (contra la rúbrica de 100 puntos)

- [ ] **Orquestación con Docker Compose (25%):**
  - [ ] `docker-compose.yml` levanta `mlflow` y `api` con **un solo comando** (`docker compose up -d --build`) desde un **clon limpio**, sin pasos manuales previos ni errores por falta de `models/p2-riesgo/`.
  - [ ] Incluye configuraciones avanzadas: red dedicada (`mibici-net`), volumen persistente (`mlflow_data`), variables de entorno parametrizadas (`.env.example`), healthchecks en ambos servicios, `depends_on: condition: service_healthy` y límites de recursos.
  - [ ] Validado compatibilidad con Docker Compose (y documentado/validado `podman-compose` / macOS).
  - [ ] Verificado tanto el bundle inmutable por defecto como la carga desde `models:/mibici-equipo05-p2-riesgo@champion` vía `http://mlflow:5001`.
- [ ] **Pipeline de CI/CD con GitHub Actions (25%):**
  - [ ] `.github/workflows/ci.yml` corre automáticamente en `push` y `pull_request` hacia `main`.
  - [ ] Ejecuta `make lint`, `make prepare-model` + matriz `pytest` (`fast` y `slow`), y `make security` (`bandit`, `pip-audit`, `detect-secrets-hook`).
  - [ ] Construye la imagen, levanta el stack con `docker compose`, valida `/health`, `/model-info` y `/predict`, y publica la imagen en GHCR (`main`).
  - [ ] Incluye optimizaciones: caché de `uv`, caché de capas Buildx (`type=gha`), matriz de pruebas, trabajos dependientes y resumen/notificación en `$GITHUB_STEP_SUMMARY`.
  - [ ] El último run sobre `main` en GitHub Actions se muestra **en verde** (incluir captura/enlace en reporte y badge o enlace en README).
- [ ] **Pruebas de carga con Locust + `docker stats` (20%):**
  - [ ] Escenario pertinente a MiBici (365 estaciones activas, consultas unitarias y por lote de hasta 365 estaciones, validando respuesta HTTP 200 y número de predicciones).
  - [ ] Corrida reproducible (`--headless` + CSVs) en múltiples escalones hasta identificar saturación, acompañada de `docker stats` simultáneo (CPU% y RAM de `api` y `mlflow`).
  - [ ] Gráficas claras en el reporte (RPS, latencia p50/p95/p99, tasa de error, CPU/RAM) con interpretación detallada y áreas de mejora identificadas.
- [ ] **Análisis FinOps (20%):**
  - [ ] Cálculo explícito y trazable del costo por petición HTTP y por inferencia estación-hora (y por 1M inferencias), basado en precio público citado de una instancia cloud comparable.
  - [ ] Considera múltiples factores: capacidad pico sostenible vs demanda real 24/7 de MiBici (capacidad ociosa), tiempo/latencia, memoria/CPU por servicio (`api` vs `mlflow`), disco/registro y escalabilidad.
  - [ ] Propone optimizaciones claras, viables y **cuantificadas con datos** (batching 365x, desacople de MLflow en runtime, right-sizing/concurrencia y scale-to-zero/serverless).
- [ ] **Documentación y Git colaborativo (10%):**
  - [ ] `README.md` actualizado con instrucciones claras paso a paso (Compose, variables, CI/CD, Locust, `docker stats`) y tabla de integrantes actualizada hasta Entrega 6.
  - [ ] Reporte PDF nombrado exactamente **`TC5061_Operacion_FinOps_Equipo05.pdf`**, en **Arial 11** (o equivalente), **interlineado 1.15**, **3 a 5 cuartillas** de cuerpo sin contar anexos, con gráficas de Locust, análisis FinOps y bibliografía en APA 7.
  - [ ] Historial de Git muestra PRs revisados y contribuciones significativas y equilibradas de los **4 integrantes**.
  - [ ] `make lint`, `make test` y `make security` en verde sobre `main` final.

---

## 6. Pensando en el cierre del proyecto

Con la Entrega 6, la arquitectura de MiBici P2 queda cerrada de punta a punta: datos versionados por manifest y validación DeepChecks (E1/E3), entrenamiento y Registry trazable en MLflow (E2), API contenerizada con Pydantic (E4), observabilidad con Evidently y A/B testing (E5), y ahora orquestación en un comando, CI/CD automatizado, pruebas de carga y economía operativa FinOps (E6). Si se requiere una presentación o entrega final integradora (E7/E8), el repositorio ya cuenta con artefactos reproducibles, pipeline verde e imagen publicada lista para desplegarse.
