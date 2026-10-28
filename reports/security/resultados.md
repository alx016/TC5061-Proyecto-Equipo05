# Entrega 3 — Chequeos de seguridad

- **Fecha de la corrida:** 28 de octubre de 2026.
- **Rama:** `feat/security-checks`.
- **Alcance:** código del paquete `tc5061_proyecto_equipo5` (bandit), dependencias
  declaradas en `pyproject.toml`/`uv.lock` (pip-audit) y archivos versionados
  (detect-secrets).
- **Fuera de alcance de bandit:** tests, notebooks y `reports/validation/`. El script de
  validación contiene un `pickle.loads(pickle.dumps(...))` en memoria que bandit marcaría
  como B301; no lee datos externos ni una ruta de entrada. También quedan fuera la
  infraestructura del servidor de MLflow y el análisis de contenedores o de nube.

## Resultado general

| Control | Antes | Después |
|:---|:---:|:---:|
| bandit (código del paquete) | 11 hallazgos (10 baja, 1 media) | 0 |
| pip-audit (dependencias de ejecución) | 42 avisos en 3 paquetes | 0 |
| pip-audit (entorno completo, con dev) | — | 0 |
| detect-secrets (archivos versionados) | 53 coincidencias, todas falsos positivos | 0 secretos reales |
| gitleaks (historial completo) | — | 0 fugas |
| Credenciales en el historial de Git | — | Ninguna |

## Verificación funcional

- `uv run pytest -q`: **250 pruebas pasan** (12 min 56 s) en la rama, sin fallos.
- `uv run ruff check` y `uv run ruff format --check`: sin errores.
- `uv lock --check`: lockfile consistente con `pyproject.toml`.
- Prueba directa del guard de descargas: `fetch_file("http://…")` lanza `ValueError` y no
  invoca a `curl`.

## Herramientas y comandos

Versiones usadas: Python 3.13.15, bandit 1.9.4, pip-audit 2.10.1, detect-secrets 1.5.0 y
gitleaks 8.30.1. bandit y pip-audit quedaron en el grupo `dev` de `pyproject.toml`, así que
se ejecutan con `uv run` y no forman parte de las dependencias de ejecución.

```bash
# Código
uv run bandit -r tc5061_proyecto_equipo5

# Dependencias de ejecución
uv export --no-dev --format requirements-txt --no-hashes -o /tmp/requirements.txt
uv run pip-audit -r /tmp/requirements.txt

# Entorno completo, incluido el grupo dev
uv export --format requirements-txt --no-hashes -o /tmp/requirements-full.txt
uv run pip-audit -r /tmp/requirements-full.txt

# Secretos en archivos versionados
git ls-files -z | xargs -0 uvx detect-secrets scan > /tmp/detect-secrets.json

# Secretos en todo el historial de Git
gitleaks git --log-opts="--all" --report-format json --report-path /tmp/gitleaks.json .
```

pip-audit omite el proyecto local `tc5061-proyecto-equipo5` porque no está publicado en
PyPI; no es una vulnerabilidad, solo una limitación del auditor.

## Hallazgos de bandit y resolución

Los 11 hallazgos iniciales se revisaron uno por uno. No había ejecución de comandos con
entrada del usuario ni `shell=True`: las llamadas a `subprocess` usan listas de argumentos
fijas. Las supresiones se documentan aquí y en el código con `# nosec` y su regla.

| Archivo | Regla | Severidad | Hallazgo | Resolución |
|:---|:---|:---|:---|:---|
| `dataset.py` | B404 | Baja | Import de `subprocess` | `# nosec B404`: solo se usan listas de argumentos fijas |
| `dataset.py` | B603 (2) | Baja | Llamadas a `curl` para la página y los CSV | `# nosec B603`: argumentos fijos de `CURL_ARGS`, sin shell; la URL viene del sitio oficial o se construye con `BASE_URL` |
| `features.py` | B403 | Baja | Import de `pickle` | `# nosec B403`: solo se escribe el preprocesador local en `preprocessor.pkl`; el módulo nunca lee un pickle externo |
| `features.py` | B101 | Baja | `assert` en código de producción | Se reemplazó por un `ValueError` explícito; el chequeo ya no desaparece con `python -O` |
| `modeling/common.py` | B404 | Baja | Import de `subprocess` | `# nosec B404`: argumentos fijos, sin shell |
| `modeling/common.py` | B310 | Media | `urlopen` con una URL dinámica | Se valida el esquema con `urlsplit(uri).scheme in {"http", "https"}` antes de abrir; luego `# nosec B310` |
| `modeling/common.py` | B603 (2) | Baja | `git rev-parse` y `git status` | `# nosec B603`: argumentos fijos, sin shell |
| `modeling/common.py` | B607 (2) | Baja | `git` se resuelve por `PATH` | `# nosec B607`: se acepta la resolución por `PATH` para `git` y `curl`; el `PATH` lo define el entorno de ejecución, no datos externos |

Estado final: `uv run bandit -r tc5061_proyecto_equipo5` termina con código 0 y no reporta
hallazgos.

Las supresiones son locales al código. Si alguna de esas llamadas empieza a aceptar rutas,
URLs o argumentos externos, hay que quitarlas y volver a revisar el flujo.

## Hallazgos de dependencias y resolución

pip-audit encontró 42 avisos repartidos en tres paquetes. Todos tenían versión corregida
disponible.

| Paquete | Antes | Avisos | Después | Acción |
|:---|:---:|:---:|:---:|:---|
| `pillow` | 12.3.0 | 39 | 13.0.0 | `uv lock --upgrade-package pillow` (dependencia transitiva) |
| `certifi` | 2026.7.22 | 2 | 2026.10.23 | `uv lock --upgrade-package certifi` (transitiva) |
| `mlflow` | 3.17.1 | 1 | 3.17.2 | Versión fijada a `3.17.2` en `pyproject.toml` y `uv.lock` |

Después de los cambios, pip-audit no reporta vulnerabilidades conocidas ni en las
dependencias de ejecución ni en el entorno completo. El cambio de `mlflow` es un parche de
la misma serie; además de `pyproject.toml` y `uv.lock`, se actualizó el pin de
`python_env.yaml` para que el entorno del MLflow Project coincida con el proyecto (lo
verifica `tests/test_dependencies.py`). El cambio de `pillow` es una dependencia transitiva
(no se declara directamente en `pyproject.toml`) y se validó con la suite de pruebas.

## Revisión de secretos

`detect-secrets` marcó 53 coincidencias en archivos versionados:

| Archivo | Coincidencias | Clasificación |
|:---|:---:|:---|
| `notebooks/0.01-eaml-exploracion-inicial.ipynb` | 5 | Falsos positivos: imágenes PNG embebidas en Base64 en las salidas |
| `notebooks/1.0-fge-eda-mibici.ipynb` | 13 | Falsos positivos: imágenes PNG embebidas en Base64; una coincidencia del detector `AWS Access Key` cae dentro de los datos de una imagen y no es una credencial |
| `references/viajes_manifest.json` | 35 | Falsos positivos: hashes SHA-256 de integridad del manifiesto de datos |

No se encontró ninguna credencial, token, contraseña ni clave privada. La revisión del
historial de Git confirma que `.env` nunca se versionó; el único archivo de entorno en el
repositorio es `.env.example`, que solo contiene la URI local de MLflow y el nombre del
experimento, sin valores sensibles. La búsqueda de nombres sensibles (`*.pem`, `*.key`,
`id_rsa`, credenciales, secretos) en todo el historial tampoco encontró archivos.

También se ejecutó `gitleaks 8.30.1` sobre todo el historial
(`gitleaks git --log-opts="--all"`): **148 commits y ~6.8 MB escaneados, 0 fugas**. La
búsqueda por patrones de claves de AWS, GitHub, OpenAI, Slack y encabezados de clave privada
en `git log --all -p` también dio 0 coincidencias.

## Prácticas que ya existían

- `.gitignore` excluye `/data/`, `.env`, `.venv`, `mlflow.db`, `mlruns/`, `mlartifacts/` y
  `local/`.
- `.env.example` documenta las variables necesarias con valores locales; la configuración
  se carga con `python-dotenv` desde `config.py`, no con credenciales escritas en el código.
- Los datos personales de los viajes (`usuario_id`, `genero`, `anio_nacimiento`) están en
  `PERSONAL_COLUMNS` y se excluyen de las entradas y artefactos de MLflow.
- No hay contraseñas ni tokens en el código, los notebooks ni el historial.

## Endurecimiento aplicado

Se agregó `.env.*` a `.gitignore`, con la excepción `!.env.example`, para que variantes como
`.env.local` o `.env.production` no puedan subirse por accidente sin dejar de versionar la
plantilla. También se ignoran `.envrc` y `.envrc.*`, que direnv usa para exportar variables
y suelen contener secretos.

Además, `fetch_file` rechaza cualquier URL que no sea `https` antes de invocar a `curl`, y
la comprobación de salud de MLflow cierra la respuesta HTTP en lugar de dejarla al recolector
de basura.

## Riesgos residuales y limitaciones

- bandit y pip-audit son revisiones puntuales: hay que repetirlas cuando cambien el código o
  las dependencias. pip-audit solo detecta vulnerabilidades conocidas a la fecha de la
  corrida.
- Las supresiones `# nosec` dependen de que las llamadas sigan usando argumentos fijos y de
  que el pickle siga siendo de escritura local. Cualquier cambio en esos flujos invalida la
  justificación.
- `detect-secrets` y gitleaks son detectores por reglas y entropía; las salidas de notebooks
  producen falsos positivos en detect-secrets. El escaneo de historial con gitleaks no
  encontró fugas, pero ninguno sustituye una auditoría manual.
- No se agregó automatización en CI; este frente deja los comandos listos para que el
  `Makefile` y el README de la entrega los invoquen.
- `python_env.yaml` lo resuelve `pip` al construir el entorno del MLflow Project, fuera del
  corpus que audita pip-audit. Se mantuvo sincronizado con `pyproject.toml` y
  `tests/test_dependencies.py` verifica esa igualdad.
