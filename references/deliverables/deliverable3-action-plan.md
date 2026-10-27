# Plan de acción · Entrega 3: Tests y chequeos de seguridad

- **Entrega:** domingo 1 de noviembre de 2026.
- **Ventana de trabajo:** lunes 26 de octubre a domingo 1 de noviembre (7 días). El domingo es
  colchón: revisión final y envío, sin trabajo nuevo.
- **Esfuerzo estimado:** ~17.5 h en total (detalle en la sección 3).
- **Requisitos:** [deliverable3-requirements.md](deliverable3-requirements.md).

---

## 1. Diagnóstico: qué tenemos y qué falta

Revisión del repositorio al 26 de octubre (`main` en `5e5a9f2`, último merge #22). La suite
completa corre con `uv run pytest`: **250 pruebas, todas pasan** (~3 min 45 s).

| Criterio (peso) | Estado | Qué hay | Brecha |
|:---|:---|:---|:---|
| Pytest (30%) | 🟡 Casi completo | 19 archivos en `tests/` con datos sintéticos: datos, features, entrenamiento de P1, P2 y P3, búsqueda, selección y registro. Ya incluyen casos límite (por ejemplo, WAPE indefinido) | No hay una prueba de reproducibilidad real: `test_tweedie_has_explicit_loss_and_reproducible_seed` solo revisa que `random_state == 7`, nunca entrena dos veces para comparar |
| DeepChecks (25%) | 🔴 Falta | Nada. `deepchecks` no funciona con las versiones del proyecto (ver WP-T2); se probó y funciona en un entorno aislado | Script de chequeos de integridad y de drift train/test, reporte y documentación de hallazgos |
| Seguridad (25%) | 🟡 Parcial | `.gitignore` cubre `.env`, `/data/`, `mlflow.db`, `mlruns/` y `mlartifacts/`; existe `.env.example`; la configuración se lee con `python-dotenv`. `.env` y `mlflow.db` nunca se subieron al historial, y no hay credenciales escritas en el código | No se ha corrido ninguna herramienta (bandit, pip-audit) ni hay evidencia de resultados |
| Organización y documentación (10%) | 🟡 Parcial | Las pruebas están en `tests/` y corren con un solo comando; el README menciona `uv run pytest` | Falta una sección de pruebas y chequeos en el README y las capturas. `make test` usa `python -m pytest` sin `uv run` y falla si no está activado el `.venv` |
| Colaboración (10%) | 🟢 Bien | Los cuatro integrantes tienen commits significativos en ramas `feat/` y `fix/` con PRs | Que cada quien haga un frente de esta entrega en su propia rama |

---

## 2. Frentes de trabajo

Un frente por integrante, cada uno en su rama y con su PR.

### WP-T1 · `test/reproducibility` · Prueba de reproducibilidad (Pytest, 30%)

**Responsable:** *por asignar*

Crear `tests/test_reproducibility.py` con datos sintéticos:

- **P2** (~1 h): construir `build_model("lightgbm", seed=42, …)`
  (`modeling/train_classification.py`), entrenarlo dos veces y verificar con
  `np.testing.assert_array_equal` que las predicciones son iguales.
- **P3** (~0.5 h): hacer lo mismo con `build_estimator("kmeans", k, seed=42, …)`
  (`modeling/train_clustering.py`) y comparar las etiquetas.
- **P1** (~0.5 h): entrenar dos veces el LightGBM Tweedie de `candidates(seed)` y verificar que las
  predicciones son idénticas.
- **Caso de contraste** (~0.5 h): con otra semilla, KMeans con `n_init=1` debe dar centroides distintos.
  Así la prueba demuestra que la semilla sí tiene efecto.
- **PR y revisión** (~0.5 h).

**Total WP-T1:** ~3 h.

### WP-T2 · `feat/deepchecks` · Validación de datos (25%)

**Responsable:** *por asignar*

**Compatibilidad (verificada el 26 de octubre):** `deepchecks` **no funciona en el `.venv` del
proyecto**. La versión más reciente en PyPI es la 0.19.1 y, como no declara límites superiores,
`uv add` la resuelve sin quejarse, pero falla al importarse:

- pide `pkg_resources`, que venía con `setuptools` y no está instalado;
- con `setuptools<81` instalado, falla porque pide el scorer `max_error`, que scikit-learn 1.9
  ya no tiene. Bajar scikit-learn no es opción: los modelos de las entregas anteriores se
  entrenaron con 1.9.

**Decisión:** deepchecks corre en un **entorno aislado**, separado del `.venv`. Las validaciones
solo leen los parquet y no necesitan el código de modelos, así que no pierden nada. Esta
combinación se probó con datos sintéticos: las dos suites corren, detectan el drift y generan el
reporte HTML.

| Paquete | Versión |
|:---|:---|
| Python | 3.12 |
| `deepchecks` | 0.19.1 |
| `scikit-learn` | 1.5.2 |
| `pandas` | 2.2.3 |
| `numpy` | 1.26.4 |
| `setuptools` | `<81` (aporta `pkg_resources`) |
| `plotly` | `<6` (con la 6, el reporte HTML pide `anywidget`) |
| `pyarrow` | cualquiera (para leer los parquet) |

1. (~0.25 h) Si alguien ya agregó deepchecks al proyecto, quitarlo:
   `uv remove --group dev deepchecks`. No debe quedar ni en `pyproject.toml` ni en `uv.lock`.
2. (~3 h) Crear `scripts/data_checks.py` como **script independiente**:
   - Declara sus dependencias en su propio encabezado (metadatos en línea, PEP 723) con las
     versiones de la tabla y `requires-python = "==3.12.*"`. Se ejecuta con
     `uv run scripts/data_checks.py` y uv arma el entorno aislado solo.
   - **No importa nada de `tc5061_proyecto_equipo5`**, porque ese paquete no está en el entorno
     aislado. Las rutas y las fechas de la partición se pasan como argumentos o se duplican en
     constantes con un comentario que diga de dónde vienen (`config.py`).
   - Corre la suite `data_integrity` sobre las features procesadas: tipos de datos, valores
     faltantes, duplicados, columnas constantes y etiquetas en conflicto.
   - Corre `train_test_validation` sobre la partición temporal (train de enero a septiembre,
     test de diciembre) para medir el drift de las features y de la etiqueta.
   - Guarda los reportes HTML en `reports/deepchecks/` y un resumen en JSON (qué chequeo pasó,
     falló o dio error), que es lo que lee la prueba del paso 4.
   - **Pone condiciones explícitas.** Las de por defecto son permisivas: en la prueba con ~5% de
     duplicados y faltantes, `data_integrity` marcó 0 fallas. Por ejemplo,
     `DataDuplicates().add_condition_ratio_less_or_equal(0)`.
   - Construye cada `Dataset` con `datetime_name` e `index_name`. Sin ellos, los chequeos de
     fuga entre train y test (*Date/Index Train Test Leakage*) dan error en vez de correr.
3. (~1.5 h) Escribir `reports/deepchecks/hallazgos.md` con lo que se detectó y cómo se interpreta. Como
   la partición es temporal, algo de drift es esperado; documentarlo es lo que da el nivel
   "Excelente".
4. (~1 h) Agregar `tests/test_data_checks.py`. Como deepchecks no se puede importar en el `.venv`,
   la prueba no lo importa: escribe un parquet sintético en `tmp_path` con faltantes y duplicados
   metidos a propósito, ejecuta el script con `subprocess` (`uv run scripts/data_checks.py …`) y
   verifica en el JSON que esos problemas se detectaron. Así corre dentro de `uv run pytest`
   como las demás. La primera vez tarda más mientras uv arma el entorno; conviene marcarla como
   `slow`.
5. PR y revisión (~0.5 h).

**Total WP-T2:** ~6.5 h.

### WP-T3 · `feat/security-checks` · Seguridad (25%)

**Responsable:** *por asignar*

1. (~0.25 h) Agregar al grupo dev: `uv add --group dev bandit pip-audit`.
2. (~0.5 h) Correr `bandit -r tc5061_proyecto_equipo5` para el código y `pip-audit` (o
   `uv export | pip-audit -r /dev/stdin`) para las dependencias.
3. (~0.5 h) Opcional, pero suma mucho: revisar el historial de git en busca de secretos con
   `gitleaks detect` o `uvx detect-secrets scan`.
4. (~2 h) Documentar en `reports/security/resultados.md` cada hallazgo y su resolución: corregirlo, o
   justificarlo con `# nosec` o como falso positivo. Documentar también las prácticas que ya
   existen: `.gitignore`, `.env.example` y que `.env` nunca se subió.
5. PR y revisión (~0.5 h).

**Total WP-T3:** ~4 h.

### WP-T4 · `docs/readme-entrega3` · Organización, comandos y documentación (10%)

**Responsable:** *por asignar*

1. (~0.5 h) `Makefile`:
   - corregir `test` para que use `uv run pytest`;
   - nuevos targets `data-checks` (`uv run scripts/data_checks.py`), `security` y un `check`
     que corra todo.
2. (~0.25 h) Configurar pytest en `pyproject.toml` (`[tool.pytest.ini_options]`) con
   `testpaths = ["tests"]`. Opcional: un marker `slow`.
3. (~1.5 h) README: nueva sección "Pruebas y chequeos" que explique qué se valida, cómo correr cada cosa
   y cómo leer los resultados. Incluir por qué deepchecks corre en un entorno aislado.
4. (~1 h) Capturas de pytest, DeepChecks, bandit y pip-audit en `reports/figures/entrega3/`.
5. PR y revisión (~0.5 h).

**Total WP-T4:** ~4 h.

---

## 3. Esfuerzo y calendario

| Frente | Esfuerzo | Peso que cubre |
|:---|:---|:---|
| WP-T1 · Reproducibilidad | ~3 h | 30% |
| WP-T2 · DeepChecks | ~6.5 h | 25% |
| WP-T3 · Seguridad | ~4 h | 25% |
| WP-T4 · Organización y README | ~4 h | 10% |
| **Total** | **~17.5 h** | |

WP-T2 pesa casi el doble que los demás. Como WP-T1 es el más corto, conviene que su responsable
tome también el paso 4 de WP-T2 (la prueba en Pytest de los chequeos, ~1 h).

| Día | Trabajo |
|:---|:---|
| Lunes 26 | Asignar responsables. ✅ Compatibilidad de deepchecks verificada: se usa un entorno aislado |
| Martes 27 a miércoles 28 | WP-T1 y WP-T3 completos, en paralelo. WP-T2: script `scripts/data_checks.py`. WP-T4: `Makefile` y configuración de pytest |
| Jueves 29 | WP-T2: reportes y `hallazgos.md`. PRs de WP-T1 y WP-T3 |
| Viernes 30 | WP-T2: prueba en Pytest y PR. Merge de todo a `main` |
| Sábado 31 | WP-T4: README y capturas, tomadas desde `main` ya integrado |
| Domingo 1 | Colchón: revisión final y envío |
