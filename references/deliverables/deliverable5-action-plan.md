# Plan de acción · Entrega 5: Monitoreo con Evidently y A/B testing

- **Entrega:** domingo 15 de noviembre de 2026.
- **Ventana de trabajo:** lunes 9 a domingo 15 (7 días).
- **Capacidad:** ~2.5 h efectivas por persona entre semana y ~5 h el sábado. El domingo es
  colchón: revisión final y envío, sin trabajo nuevo.
- **Requisitos:** [deliverable5-requirements.md](deliverable5-requirements.md).
- **Planes anteriores:** [Entrega 2](deliverable2-action-plan.md) ·
  [Entrega 4](deliverable4-action-plan.md).

## Registro de decisiones

| # | Decisión | Propuesta |
|:---|:---|:---|
| 5.1 | ✅ Modelo que se monitorea | **P2 riesgo** (`mibici-equipo05-p2-riesgo@champion`), el mismo que sirve la API de la entrega 4. Así la capa de observabilidad vigila al modelo que está "en producción". P2 tiene otra ventaja: su etiqueta se conoce una hora después, así que además del drift se puede medir el desempeño real en el periodo actual, cosa que con P3 no se puede |
| 5.2 | ✅ Conjunto de referencia | **Las filas de `train` de P2 (enero a septiembre de 2024)**, con los mismos filtros temporales que usó el entrenamiento (`split_masks`). Es literalmente lo que vio el modelo. Se descarta usar todo 2024 porque octubre a diciembre se usaron para elegir umbrales y para test, no para ajustar |
| 5.3 | ✅ Conjunto actual (drift) | **Enero a septiembre de 2025**: los mismos meses un año después. Al comparar meses iguales, el drift que aparezca no se explica por la temporada (lluvias, vacaciones, fin de año) sino por cambios reales en la demanda o en la red. `reports/justificacion_periodo_estudio.md` ya reservaba 2025 para drift e inferencia. Como complemento se puede correr un segundo reporte con octubre a diciembre de 2025 |
| 5.4 | ✅ Versión alternativa (challenger) | **Mismo algoritmo e hiperparámetros del champion, reentrenado con datos recientes**: ajuste con enero 2024 a julio 2025, umbrales calibrados con agosto y septiembre de 2025 (misma regla de `calibrate_thresholds`, con el piso de recall de 0.85). Así la única diferencia entre versiones son los datos, y el A/B responde la pregunta de Continuous Training: ¿reentrenar sirve? Se registra como nueva versión de `mibici-equipo05-p2-riesgo` con alias `challenger` usando `registry.py`, que ya soporta ese alias |
| 5.5 | ✅ Conjunto de evaluación del A/B | **Octubre a diciembre de 2025**, que ninguna de las dos versiones vio (ni para ajustar ni para umbrales). Son los mismos meses que el champion usó para validación y test en 2024, así que sus métricas originales sirven de punto de comparación. Opcional, si sobra tiempo: repetir el A/B con enero a septiembre de 2026 como prueba de robustez |
| 5.6 | ✅ Regla de decisión del A/B | **Se fija antes de ver resultados** (sección 2, punto 5) y se escribe en el reporte tal cual. Cada versión predice con sus propios umbrales, porque son parte del modelo |
| 5.7 | ✅ Dónde corre Evidently | **En el `.venv` del proyecto, grupo `monitoring`.** Se verificó el 9 de noviembre: `evidently==0.7.23` resuelve e importa con Python 3.13, pandas 3.0.6, numpy 2.5.4 y scikit-learn 1.9.1, y genera un reporte con `DataDriftPreset` y `DataSummaryPreset`. No se repite el problema de DeepChecks ([notas](deliverable3-deepchecks-notes.md)). Si al regenerar `uv.lock` se mueve alguna versión que usa el modelo (mlflow, pandas, numpy, scikit-learn, lightgbm), se cae al plan B: script con encabezado PEP 723, como en la entrega 3 |
| 5.8 | ✅ Host de evidencia | **Oscar.** El champion de P2 vive en su `mlflow.db`, y el challenger tiene que quedar en el mismo Registry para que sean dos versiones del mismo modelo. Él registra el challenger y corre el A/B final desde `main`. Los demás desarrollan contra cualquier run local de P2 (no hay MLflow compartido) |
| 5.9 | ✅ Formato del reporte | **`reports/entrega5_monitoreo.md`**, enlazado desde una sección corta del README, con las gráficas de Evidently exportadas como PNG en `reports/figures/entrega5/` y los HTML interactivos en `reports/monitoring/`. Se exporta a PDF para la entrega si el profesor lo pide |

---

## 1. Diagnóstico: qué tenemos y qué falta

Revisión del repositorio al 9 de noviembre (`docs/plan-entrega5`, sobre el merge del PR #36):

| Requisito | Estado | Brecha |
|:---|:---|:---|
| Datos del periodo actual | ⚠️ `data/interim/viajes/` ya tiene 2025 (12 meses) y 2026 (enero a septiembre), y están en el manifest (33 archivos) | No existe la tabla de features de P2 para 2025: `classification_features.parquet` solo cubre 2024 (3.1 M filas, 358 estaciones) |
| Features de P2 para otro año | ❌ `features.py` tiene el año fijo: `HOLIDAYS_2024`, `add_split` usa `STUDY_YEAR` y los meses de train/val/test, y el panel es de un solo año | Parametrizar por año y agregar feriados 2025 y 2026 (sección 1, problemas 1 a 3) |
| Evidently | ❌ No está en las dependencias | Grupo `monitoring` (5.7) |
| Reporte de drift | ❌ | Módulo `monitoring/drift.py`, HTML + JSON + PNG |
| Entrenar con otra ventana temporal | ❌ `train_classification.py` toma las ventanas de `config.py` (`TRAIN_MONTHS`, `P2_FIT_AS_OF`, `P2_SELECTION_AS_OF`) | Permitir otra tabla de entrada y otras fechas de corte sin romper el champion |
| Registry con challenger | ✅ `registry.py` registra con `--alias challenger`, hace pre-flight y no degrada al champion | Solo usarlo con el run del challenger |
| A/B testing | ❌ | Módulo `monitoring/ab_test.py` con métricas, intervalos y la regla de decisión |
| Recomendación de Continuous Training | ❌ | Sección del reporte |
| Git colaborativo (10%) | - | Reparto parejo (sección 3): cada integrante tiene ~6 h, un paquete de código con pruebas, una sección del reporte y dos revisiones |

### Problemas técnicos a resolver

1. **Feriados.** `is_holiday` usa la lista de 2024. Si se reutiliza con 2025, todos los feriados
   de 2025 saldrían como día normal, y Evidently reportaría un drift falso en `is_holiday`. Hay
   que agregar `HOLIDAYS_2025` (1 ene, 3 feb, 17 mar, 1 may, 16 sep, 17 nov, 25 dic) y
   `HOLIDAYS_2026` (1 ene, 2 feb, 16 mar, 1 may, 16 sep, 16 nov, 25 dic), o elegir la lista
   según el año. En 2024 hubo un feriado extra (1 de octubre, cambio de Ejecutivo) que no se
   repite: es un cambio real y el reporte lo debe mencionar.
2. **Estaciones nuevas o retiradas.** El panel solo conserva estaciones que operaron los 12
   meses del año (`full_year_station_ids`). En 2025 puede haber estaciones que no existían en
   2024. Para el drift se usan todas las de 2025, porque eso es lo que vería producción, y se
   reporta cuántas son nuevas. Para el A/B se reportan las métricas sobre todas las estaciones
   y sobre las comunes a los dos años, para separar el efecto de la red del de la demanda.
3. **Capacidad estática.** El catálogo es una sola foto (`nomenclatura_2026_09.csv`) y
   `net_flow` se normaliza por capacidad. Si una estación cambió de tamaño entre 2024 y 2025,
   no se ve. Se documenta como limitación.
4. **Primer día del año.** Los lags (`net_flow_lag24`, etc.) del 1 de enero de 2025 necesitan
   las últimas horas de 2024. Si el panel se arma por año, las primeras 24 horas quedan sin lag.
   Para el challenger, que junta 2024 y 2025, conviene armar un solo panel continuo.
5. **Tamaño y prueba estadística.** Referencia ≈ 2.35 M filas. Evidently con millones de filas
   tarda y produce un HTML enorme. Además, con muestras tan grandes una prueba con p-value (KS,
   chi²) marca drift en cualquier diferencia mínima. Propuesta: muestra aleatoria con semilla 42
   (por ejemplo 100 k filas por conjunto, estratificada por mes) y métricas de distancia
   (Wasserstein normalizada para numéricas, PSI o Jensen-Shannon para categóricas), que son el
   default de Evidently para muestras grandes. El reporte explica la prueba y el umbral usados.
6. **Variables de calendario.** `hour_sin`, `hour_cos`, `day_of_week` e `is_weekend` no deberían
   tener drift entre enero-septiembre de 2024 y de 2025. Sirven de control: si aparecen con
   drift, hay un error en la construcción de features.
7. **Drift del objetivo y de la predicción.** Además de las 18 variables, el reporte incluye la
   columna `target` (proporción real de desabasto y saturación) y la predicción del champion en
   los dos conjuntos. Con eso se distingue drift de covariables, drift de la etiqueta y
   degradación del modelo.
8. **No hay MLflow compartido.** Champion y challenger deben estar en el mismo `mlflow.db` (5.8).
   Para que el A/B sea reproducible en otra máquina, `ab_test.py` también acepta rutas locales
   de los dos modelos.
9. **Fuga en el A/B.** El challenger no puede ver octubre a diciembre de 2025, ni siquiera para
   calibrar umbrales. Las fechas de corte se escriben en `config.py` y se guardan como params
   del run.

---

## 2. Diseño propuesto

```
tc5061_proyecto_equipo5/monitoring/
├── __init__.py
├── datasets.py   <- referencia, actual y evaluación con las fechas de la sección 2, punto 1
├── drift.py      <- reporte de Evidently: HTML, JSON con el resumen y PNG para el reporte
└── ab_test.py    <- carga dos versiones, evalúa en el mismo conjunto, intervalos y decisión
reports/monitoring/                <- HTML y JSON de Evidently, resumen del A/B
reports/figures/entrega5/          <- PNG que usa el reporte
reports/entrega5_monitoreo.md      <- documento final
```

**Contrato**

Es el acuerdo para trabajar en paralelo. Igual que en la entrega 4, si alguien propone un
cambio se platica antes y se actualiza esta sección.

1. **Periodos** en `config.py`, para que nadie los escriba a mano:

   ```python
   # Entrega 5: monitoring windows (target hour, inclusive)
   DRIFT_REFERENCE = ("2024-01-01", "2024-09-30")   # P2 train split
   DRIFT_CURRENT = ("2025-01-01", "2025-09-30")
   AB_EVALUATION = ("2025-10-01", "2025-12-31")
   CHALLENGER_FIT_AS_OF = "2025-07-31 23:40"
   CHALLENGER_SELECTION_AS_OF = "2025-09-30 23:40"
   CURRENT_FEATURES_PATH = PROCESSED_DATA_DIR / "classification_features_2025.parquet"
   MONITORING_SAMPLE_SIZE = 100_000
   ```

2. **Tabla de features del periodo actual.** Mismas columnas y tipos que
   `classification_features.parquet` (las 18 `FEATURE_COLUMNS`, `station_id`, `hour`,
   `target`), para que el champion la pueda calificar sin cambios. La columna `split` vale
   `current` en enero a septiembre y `evaluation` en octubre a diciembre de 2025.
   `tests/test_features.py` comprueba que las columnas y tipos son iguales a los de 2024.

3. **Salida de `drift.py`.**
   - `reports/monitoring/drift_2024_vs_2025.html` (el reporte interactivo).
   - `reports/monitoring/drift_summary.json`: por variable, la prueba, la distancia, el umbral
     y si hay drift; el porcentaje de variables con drift; tamaños de muestra, semilla y
     periodos. Es lo que lee el reporte y lo que usaría un disparador de reentrenamiento.
   - PNG en `reports/figures/entrega5/` de las variables con más drift (Evidently no exporta
     PNG directamente: se hacen con matplotlib desde el JSON o con capturas del HTML).
   - Columnas extra del reporte: `target` y `prediction` (la clase del champion con sus
     umbrales), punto 7 de los problemas.

4. **Salida de `ab_test.py`.** Un run de MLflow en el experimento del proyecto con
   `stage_of_work=ab_test`, los dos `model_uri` y versiones como params, y estas métricas por
   versión sobre `AB_EVALUATION`:
   `pr_auc_minority_mean`, `macro_f1`, `recall_shortage`, `recall_saturation`,
   `precision_shortage`, `alerts_per_day`. Se calculan con las funciones que ya existen en
   `train_classification.py` (`metrics`, `apply_thresholds`, `alerts_per_day`), sin
   reescribirlas, para que sean las mismas que en la entrega 2. Además:
   - **Diferencia con intervalo de confianza del 95%** (challenger − champion) por **bootstrap
     pareado por día**: se remuestrean días completos, no filas, porque las filas de una misma
     hora y estación están correlacionadas. 1 000 réplicas, semilla 42.
   - **McNemar** sobre los aciertos por fila en las clases de riesgo, como prueba
     complementaria.
   - Desglose por mes y por estaciones nuevas vs. comunes.
   - `reports/monitoring/ab_summary.json` con todo lo anterior y la decisión.

5. **Regla de decisión (decisión 5.6).** El challenger reemplaza al champion solo si cumple las
   tres condiciones:
   1. **Guardarraíl operativo:** `recall_shortage ≥ 0.85` en la evaluación (el mismo piso de
      `P2_MIN_RECALL_SHORTAGE`).
   2. **Mejora real:** el intervalo del 95% de la diferencia en `pr_auc_minority_mean` queda
      por encima de 0.
   3. **No inferioridad:** `macro_f1` no baja más de 0.01 y `alerts_per_day` no sube más de
      20%, para no saturar al equipo de redistribución.

   Si no se cumplen, el champion se queda. Si gana el challenger, se promueve con
   `registry.py --alias champion`, se reexporta con `make export-model` y la imagen de la
   entrega 4 se reconstruye sin cambiar código: esa es la evidencia de que el flujo de
   Continuous Training funciona de punta a punta.

6. **Targets del Makefile** (dueño: WP-D): `features-current`, `drift-report`,
   `train-challenger`, `ab-test`. Mientras no estén, cada quien usa `uv run python -m ...`.

---

## 3. Paquetes de trabajo

**Nadie espera a WP-D para empezar.** Evidently se puede probar desde hoy con
`uv run --with evidently==0.7.23 ...` sin tocar `pyproject.toml`. WP-1 y WP-3 pueden programar
contra la tabla de 2024 (por ejemplo, referencia = enero a junio y actual = julio a septiembre)
hasta que WP-0 publique la de 2025.

**Criterio de reparto.** Cada integrante tiene **~6 a 6.5 h**, **un paquete de código con
pruebas**, **una sección del reporte** firmada con sus propios commits y **dos revisiones**. El
trabajo de host (entrenar, registrar y correr el A/B final en su `mlflow.db`) cuenta como horas
de Oscar. Así ninguna parte de la rúbrica depende de una sola persona y el historial de Git
muestra contribuciones parejas.

| WP | Responsable | Revisa | Entregable | Horas |
|:---|:---|:---|:---|:---:|
| WP-D Dependencias y Makefile | Fernando | Manuel | `evidently==0.7.23` en el grupo `monitoring` de `pyproject.toml`, `uv.lock` regenerado sin mover las versiones del modelo (5.7), `python_env.yaml` y `tests/test_dependencies.py` al día, `make test` y `make security` en verde. Targets de la sección 2, punto 6. Avisar al equipo que corra `uv sync` | 1.5 |
| WP-0 Datos del periodo actual | Oscar | Emilio | Constantes de la sección 2, punto 1, en `config.py`. `features.py` parametrizado por año: feriados 2025 y 2026, `add_split` con los periodos nuevos, panel continuo 2024–2025 para el challenger (problema 4). `classification_features_2025.parquet` con el contrato del punto 2. Pruebas: columnas y tipos iguales, feriados de 2025 bien marcados, sin filas de octubre a diciembre en `current`. Reportar en el chat cuántas estaciones nuevas y retiradas hay | 4 |
| WP-1 Reporte de drift | Emilio | Oscar | `monitoring/datasets.py` y `monitoring/drift.py`: muestra con semilla, predicción del champion, Evidently con `DataDriftPreset` y `DataSummaryPreset` (+ `target` y `prediction`), salidas del punto 3. Pruebas con DataFrames sintéticos (drift inducido sí se detecta, mismo conjunto no) | 4 |
| WP-2a Entrenamiento con otra ventana | Manuel | Oscar | `train_classification.py` acepta otra tabla y otras fechas de corte por parámetro (entry point `train` del `MLproject`), sin cambiar el comportamiento por defecto: las pruebas actuales deben seguir pasando y se agrega una que prueba que el challenger no ve filas de `AB_EVALUATION`. Revisa Oscar porque conoce el protocolo temporal de P2 | 2 |
| WP-2b Challenger en el Registry (host) | Oscar | Manuel | Entrenar el challenger (5.4) desde `main` limpio con WP-2a, registrarlo con `-P alias=challenger` y anotar run ID, versión y commit en el documento del equipo. El viernes, correr el A/B final (`make ab-test`) en el mismo `mlflow.db` | 1 |
| WP-3 A/B testing | Fernando | Emilio | `monitoring/ab_test.py` con las salidas del punto 4 y la regla del punto 5. Pruebas: bootstrap reproducible con semilla, regla de decisión con casos que pasan y que fallan cada condición, y que una versión no pueda usar los umbrales de la otra | 4 |
| WP-4a Sección: drift por variable | Emilio | Manuel | Sección del reporte (parte del 20% de interpretación): qué variables tienen drift, cuánto y por qué (demanda, red, calendario); control con las variables de calendario; drift de `target` y de la predicción; gráficas de Evidently exportadas a `reports/figures/entrega5/` | 2 |
| WP-4b Sección: implicaciones para el modelo | Oscar | Emilio | Sección del reporte (resto del 20%): qué variables pesan más en LightGBM (importancias del champion) y cuáles de ellas tienen drift; qué pasa con los umbrales calibrados en 2024 si cambia la prevalencia; efecto de las estaciones nuevas | 1 |
| WP-3b Sección: A/B testing | Fernando | Manuel | Sección del reporte (30% de la rúbrica): diseño (versiones, conjunto de evaluación, regla fijada antes), tabla de métricas con intervalos, desglose por mes y por estaciones nuevas vs. comunes, y la decisión | 1 |
| WP-5 Reporte, CT e integración | Manuel | Fernando | **Lunes:** esqueleto de `reports/entrega5_monitoreo.md` con un encabezado por sección, para que cada quien escriba la suya sin conflictos. **Sábado:** introducción, periodos y su justificación (5.2–5.5), recomendación de Continuous Training (contenido mínimo abajo), conclusiones, sección corta en el README y tabla de integrantes. Si gana el challenger: promoción, reexportación y `make smoke` | 3 |
| WP-6 Reproducción | Manuel | Fernando | En su máquina (distinta a la de Oscar), desde `main`: `make features-current` y `make drift-report`, y comparar `drift_summary.json` con el del host (deben coincidir porque la muestra tiene semilla). El resultado va en el reporte como evidencia de reproducibilidad | 1.5 |

**Carga por integrante**

| Integrante | Paquetes | Horas | PR propios | Revisa |
|:---|:---|:---:|:---|:---|
| Oscar | WP-0, WP-2b, WP-4b | 6 | Features de 2025; sección de implicaciones | WP-1, WP-2a |
| Emilio | WP-1, WP-4a | 6 | Reporte de drift; sección de drift | WP-0, WP-3, WP-4b |
| Fernando | WP-D, WP-3, WP-3b | 6.5 | Dependencias; A/B; sección de A/B | WP-5, WP-6 |
| Manuel | WP-2a, WP-5, WP-6 | 6.5 | Entrenamiento con otra ventana; reporte y CT | WP-D, WP-2b, WP-4a, WP-3b |

Manuel revisa un PR más que los demás, pero las secciones del reporte son cortas y revisarlas es
parte de integrarlas. Emilio revisa la sección de implicaciones (WP-4b) porque continúa la suya
de drift por variable (WP-4a), así las dos partes de la interpretación quedan coherentes. Si alguien se atrasa, el primero en liberar horas toma la revisión que falte, no el
paquete de otro.

**Recomendación de Continuous Training (contenido mínimo de WP-5).** Responde: ¿conviene
reentrenar ahora? (sale del A/B), ¿cada cuánto? (sale del drift por mes), y ¿con qué disparador?
Propuesta para discutir con los resultados:

- **Disparador por desempeño**, que es el principal porque la etiqueta de P2 llega una hora
  después: si `pr_auc_minority_mean` de la última semana cae más de X puntos frente al test de
  2024, o `recall_shortage` baja de 0.85.
- **Disparador por datos**, como alerta temprana: si más del 30% de las variables tiene drift, o
  si alguna de las 3 más importantes del modelo tiene drift.
- **Cadencia base**: reentrenar con ventana móvil (por ejemplo, últimos 12 a 18 meses) y
  promover solo si pasa la regla del punto 5. Así el A/B de esta entrega se vuelve el paso de
  validación automática del pipeline de CT.

**Un dueño por archivo**, para que los PR en paralelo no choquen:

| Archivo | Dueño |
|:---|:---|
| `pyproject.toml`, `uv.lock`, `python_env.yaml`, `Makefile` | WP-D (Fernando) |
| `config.py`, `features.py`, `tests/test_features.py` | WP-0 (Oscar) |
| `train_classification.py`, `MLproject` | WP-2a (Manuel) |
| `monitoring/datasets.py`, `monitoring/drift.py`, `tests/test_drift.py` | WP-1 (Emilio) |
| `monitoring/ab_test.py`, `tests/test_ab_test.py` | WP-3 (Fernando) |
| `reports/entrega5_monitoreo.md` | Una sección por dueño, bajo el encabezado que deja el esqueleto de WP-5: drift por variable (Emilio), implicaciones (Oscar), A/B (Fernando), periodos, CT y conclusiones (Manuel). Cada quien edita solo su sección |
| `reports/figures/entrega5/` | WP-4a (Emilio) |
| `README.md` | WP-5 (Manuel) |

**Dependencias reales**

- WP-1 y WP-3 necesitan la tabla de 2025 (WP-0) para los resultados finales, pero no para
  programar.
- WP-2a puede programarse con la tabla de 2024; WP-2b necesita WP-2a y el panel continuo
  2024–2025 de WP-0.
- WP-3 necesita el challenger registrado (WP-2b) para el A/B final; antes prueba con dos runs
  cualesquiera de P2.
- WP-4a y WP-4b necesitan el reporte final de drift (WP-1). WP-3b necesita el A/B final.
  WP-5 (sábado) necesita todas las secciones. WP-6 necesita WP-0 y WP-1 en `main`.

---

## 4. Calendario

| Fase | Día | Trabajo | Hito |
|:---|:---|:---|:---|
| 1. Arranque en paralelo | Lun 9 | Manuel cierra las decisiones, comparte el plan y sube el esqueleto del reporte (WP-5). Empiezan WP-D, WP-0, WP-1 (con datos de 2024), WP-2a (con la tabla de 2024) y WP-3 (con dos runs locales) | Plan y esqueleto del reporte en `main` |
| | Mar 10 | **Merge de WP-D.** WP-0 abre PR con feriados y features parametrizadas. PR de WP-2a | Evidently en `main` |
| | Mié 11 | **Merge de WP-0 y WP-2a**: `classification_features_2025.parquet` reproducible con `make features-current`. Primer reporte de drift con datos reales | Datos de 2025 listos |
| 2. Resultados | Jue 12 | Merge de WP-1. Oscar entrena y registra el challenger (WP-2b). Emilio escribe WP-4a y Oscar WP-4b | Reporte de drift final; challenger `@challenger` en el Registry |
| | Vie 13 | Merge de WP-3. Oscar corre el A/B final desde `main`. Manuel reproduce el drift en su máquina (WP-6). Fernando escribe WP-3b | Resultado del A/B y decisión |
| 3. Cierre | Sáb 14 | Merge de WP-4a, WP-4b y WP-3b. WP-5: periodos, CT, conclusiones y README. Si gana el challenger: promoción, `make export-model`, `make smoke`. `make test`, `make lint`, `make security` | `main` listo para entregar |
| | Dom 15 | Colchón. Leer el reporte completo, revisar enlaces y figuras en un clon limpio y enviar | Entregado |

---

## 5. Checklist de entrega (contra la rúbrica)

- [ ] **Reporte de drift (30%)**: periodos de referencia y actual definidos con fechas exactas y
      justificados; HTML de Evidently con drift por variable y calidad de datos; tamaño de
      muestra, semilla, prueba y umbral documentados.
- [ ] **Interpretación (20%)**: variables con drift identificadas y cuantificadas; control con
      las de calendario; drift de `target` y de la predicción; implicaciones concretas para P2
      (umbrales, importancias, estaciones nuevas).
- [ ] **A/B testing (30%)**: dos versiones del mismo modelo en el Registry; mismo conjunto de
      evaluación que ninguna vio; regla fijada antes de ver resultados; métricas con intervalos
      por bootstrap pareado por día; decisión justificada con la regla.
- [ ] **Continuous Training (10%)**: recomendación clara (reentrenar o no, cada cuánto y con
      qué disparadores) apoyada en el drift y en el A/B.
- [ ] **Claridad y Git (10%)**: reporte estructurado, figuras legibles; un PR por integrante con
      su revisión y la tabla de integrantes actualizada.
- [ ] `make test`, `make lint` y `make security` en verde sobre `main`.

## 6. Pensando en la semana siguiente

`drift_summary.json` y `ab_summary.json` ya son las entradas de un pipeline de Continuous
Training: un job programado que construya las features del último periodo, corra el drift, y si
se dispara, reentrene, corra el A/B y promueva. Si la siguiente entrega lo pide, la API puede
registrar las peticiones que recibe (con el `station_id` y la hora) para usarlas como conjunto
actual en lugar de los datos abiertos.
