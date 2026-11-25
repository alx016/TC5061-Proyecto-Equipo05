# Model Card — P2: Riesgo Operativo por Estación (`mibici-equipo05-p2-riesgo`)

> **Marco de referencia:** Adaptado de *Model Cards for Model Reporting* (Mitchell et al., 2019) para el proyecto integrador TC5061 (MLOps, Equipo 05).  
> **Alcance en la arquitectura del proyecto:** El sistema analítico de MiBici comprende tres modelos complementarios: **P1** (regresión de demanda horaria, `mibici-equipo05-p1-demanda@champion`), **P2** (clasificación de riesgo operativo, `mibici-equipo05-p2-riesgo@champion`) y **P3** (clustering funcional de estaciones, `mibici-equipo05-p3-grupos@champion`). Este documento describe exclusivamente el modelo **P2** (`champion` v3), que es el modelo desplegado en la API productiva de FastAPI, monitoreado con Evidently y evaluado en A/B testing.

---

## 1. Detalles del modelo

| Campo | Especificación |
|:---|:---|
| **Nombre en el Model Registry** | `mibici-equipo05-p2-riesgo` |
| **Versión y estado de producción** | Versión **3**, alias `@champion` (equivalente a *Production*) |
| **Familia y algoritmo** | `LGBMClassifier` (`lightgbm==4.7.0`, `scikit-learn==1.9.1`, `mlflow==3.17.2`) |
| **Tarea** | Clasificación supervisada multiclase (3 clases mutuamente excluyentes: `0: Balanceado`, `1: Riesgo_Desabasto`, `2: Riesgo_Saturacion`) |
| **Hiperparámetros entrenados** | `objective="multiclass"`, `num_class=3`, `n_estimators=200`, `learning_rate=0.1`, `num_leaves=255`, `min_child_samples=100`, `class_weight="balanced"`, `random_state=42` |
| **Umbrales de decisión calibrados** | `threshold_shortage = 0.20` (desabasto) · `threshold_saturation = 0.95` (saturación), aplicados sobre `predict_proba` (en empate gana la clase de mayor probabilidad) |
| **Fechas clave** | Run de entrenamiento: 2 de noviembre de 2026 (`run_id`: `7bf71250a91947ba8465d5db48d4bce5`) · Exportación del bundle: 5 de noviembre de 2026 (`2026-11-05T01:06:39Z`) · Model Card: 24 de noviembre de 2026 |
| **Responsables (Equipo 05 · TC5061)** | **Oscar Ulises López Vázquez** (`oscarlv`, A01840039 — features P2, entrenamiento, calibración de umbrales, guardrail de recall, exportación, pruebas de carga Locust y Model Card) · **Manuel Alejandro Preciado Morán** (`alx016`, A01639643) · **Emilio Alejandro Martínez López** (`Emilio`, A01796936) · **Fernando Gutiérrez Escárrega** (`Fernando cuenta TEC`, A01840067) |
| **Licencia y condiciones de los datos** | Datos abiertos publicados por el Gobierno del Estado de Jalisco y la Agencia Metropolitana de Infraestructura para la Movilidad (AMIM) bajo la política de Datos Abiertos de [MiBici](https://www.mibici.net/es/datos-abiertos/). Uso autorizado para investigación académica, análisis cívico y desarrollo de modelos de Machine Learning con atribución al sistema MiBici. Proyecto desarrollado con fines académicos en [TC5061-Proyecto-Equipo05](https://github.com/alx016/TC5061-Proyecto-Equipo05). |

---

## 2. Uso previsto

- **Propósito principal:** Anticipar con **1 hora de anticipación** si una estación de MiBici entrará en riesgo crítico de **desabasto** (quedarse sin bicicletas disponibles para retirar) o **saturación** (quedarse sin anclajes libres para devolver), a fin de apoyar la planeación operativa y el despacho oportuno de cuadrillas de rebalanceo.
- **Ventana causal de emisión:** La predicción para la ventana objetivo `[h, h+1)` se emite al minuto **`h-00:10`** utilizando únicamente información histórica cerrada disponible en ese instante.
- **Usuarios previstos:** Personal de operación, despacho y logística de campo de MiBici en el Área Metropolitana de Guadalajara (así como el equipo de MLOps responsable de operar y monitorear el servicio).
- **Integración operativa:** El modelo se consume vía API REST (`POST /predict` en FastAPI), validando entradas con Pydantic v2 (`StationFeatures`, lotes de `1` a `500` estaciones por petición). Evaluar las 365 estaciones activas de la red en una sola petición por lotes toma una mediana de **78 ms** (frente al objetivo de `< 20 s` del ML Canvas) y devuelve la clase operativa junto con las probabilidades calibradas para priorizar las rutas de camiones de carga.

---

## 3. Fuera de alcance

Este modelo **no debe utilizarse** para los siguientes fines:

1. **Decisiones estructurales autónomas:** Retirar, cerrar, reubicar estaciones o modificar la capacidad física de anclajes tomando únicamente las alertas del modelo sin análisis de planeación urbana ni validación humana.
2. **Horizontes superiores a 1 hora:** Pronósticos multi-hora o de planeación semanal/mensual emitidos sin actualizar la historia hora por hora.
3. **Transferencia directa a otras ciudades o sistemas:** El modelo utiliza coordenadas geográficas explícitas del Área Metropolitana de Guadalajara (`latitude` $\in [20.4, 21.0]$, `longitude` $\in [-103.6, -103.1]$) y capacidades de MiBici; no es válido en otra red sin reentrenamiento ni recalibración.
4. **Perfilado o decisiones sobre personas:** La granularidad es exclusivamente estación-hora; no procesa, almacena ni infiere atributos de usuarios individuales.
5. **Lectura de $\Delta_{s,t}$ como inventario absoluto:** El modelo estima un proxy de presión de flujo neto normalizado por capacidad, no el número físico exacto de bicicletas presentes en la estación.
6. **Inferencia con umbrales por defecto (`argmax`):** Utilizar el modelo sin los umbrales calibrados (`0.20` desabasto, `0.95` saturación) colapsa el recall de las clases minoritarias debido al desbalance extremo (`98.7%` balanceado).

---

## 4. Datos y construcción de características

### 4.1 Periodo de estudio y universo de estaciones

- **Año canónico de entrenamiento (2024):** Tras auditar la serie histórica 2014–2026 ([`reports/justificacion_periodo_estudio.md`](justificacion_periodo_estudio.md)), se seleccionó **2024 completo** (`4,645,921` viajes limpios; `367.38 MB` en CSV crudos) porque excluye la deriva espacial de la etapa piloto (2014–2018, desde 86 estaciones) y el shock no estacionario de COVID-19 (2020–2021, caída de `-60.3%` en demanda). Los datos de **2025 y 2026** (hasta septiembre de 2026, sumando `12,540,018` viajes en `references/viajes_manifest.json`) se reservan para monitoreo de drift y evaluación A/B.
- **Estaciones activas todo el año (`full_year_station_ids`):** De las `484` estaciones nominales en el catálogo oficial (`nomenclatura_2026_09.csv`) y `370` con al menos un viaje en 2024, el panel de P2 conserva las **358 estaciones que operaron en los 12 meses de 2024** ([`reports/feature_engineering_2024.md`](feature_engineering_2024.md)). Esto evita rellenar con ceros las horas de estaciones que aún no existían (lo que inflaría artificialmente la clase `Balanceado`), aunque sus viajes sí se contabilizan en los arribos y salidas reales de las 358 estaciones continuas.

### 4.2 Partición temporal estricta (*Zero Data Leakage*)

La partición se asigna por la hora objetivo (`target_time = hour + 1h`) y se filtra con `training_mask` para purgar cualquier etiqueta cuyo cierre (`label_available_at = target_time + 1h`) sea posterior al corte operativo de entrenamiento o selección:

| Partición | Periodo objetivo (2024) | Filas elegibles | Corte causal (`as_of`) | Rol en el pipeline |
|:---|:---|---:|:---|:---|
| **`train`** | 1 ene – 30 sep 2024 | `2,344,900` | `2024-09-30 23:40` | Ajuste de pesos del `LGBMClassifier` |
| **`validation`** | 1 oct – 30 nov 2024 | `523,754` | `2024-11-30 23:40` | Calibración conjunta de umbrales y selección de candidato |
| **`test` (OOT)** | 1 dic – 31 dic 2024 | `266,352` | Evaluación única | Prueba ciega *Out-of-Time* ejecutada una sola vez en `select` |

### 4.3 Definición del proxy de riesgo y variables de entrada

Dado que los datos abiertos de MiBici no publican el inventario en tiempo real de bicicletas por estación, el riesgo en la hora siguiente $[t, t+1)$ se formula mediante el **flujo neto normalizado por capacidad física** ($C_s$):

$$\Delta_{s,t+1} = \frac{\text{Llegadas}_{s,t+1} - \text{Salidas}_{s,t+1}}{C_s}$$

Con umbral crítico de estrés operativo $\tau = 0.40$:

- **`0: Balanceado`** si $-0.40 < \Delta_{s,t+1} < +0.40$ (`98.716%` en las filas de ajuste 2024; `99.073%` en dic 2024).
- **`1: Riesgo_Desabasto`** si $\Delta_{s,t+1} \le -0.40$ (`0.674%` en ajuste 2024; `0.492%`, `1,311` eventos en dic 2024).
- **`2: Riesgo_Saturacion`** si $\Delta_{s,t+1} \ge +0.40$ (`0.610%` en ajuste 2024; `0.434%`, `1,157` eventos en dic 2024).

El modelo recibe **18 variables predictoras (`FEATURE_COLUMNS`)**, todas conocidas en el minuto `h-00:10`:

- **Infraestructura y ubicación (3):** `capacity` (capacidad de anclajes de `nomenclatura_2026_09.csv`), `latitude`, `longitude`.
- **Calendario cíclico y feriados (5):** `hour_sin`, `hour_cos`, `day_of_week` (`0–6`), `is_weekend`, `is_holiday` (días festivos oficiales por año).
- **Contexto espacial rezagado (1):** `neighbor_net_flow` (promedio del flujo neto en las 3 estaciones vecinas más cercanas por distancia euclidiana, rezagado 1 hora a $t-1$).
- **Rezagos temporales de demanda y flujo (9):** `net_flow_lag1`, `net_flow_lag2`, `net_flow_lag24`, `arrivals_lag1`, `arrivals_lag2`, `arrivals_lag3`, `departures_lag1`, `departures_lag2`, `departures_lag3`.

### 4.4 Privacidad y validación de integridad (DeepChecks)

- **Cero PII:** Aunque los Parquet interinos conservan las columnas seudonimizadas del portal abierto (`usuario_id`, `genero`, `anio_nacimiento`), `features.py` y el modelo las excluyen por completo. Ningún dato personal entra a entrenamiento, al Registry ni a la API.
- **Validación DeepChecks (`make data-checks`):** `0.0%` de fuga temporal (`Date Train-Test Leakage Overlap = 0.0%`) y `0.0%` de duplicados. Se documentaron dos advertencias aceptadas bajo política operativa ([`reports/deepchecks/check_policy.md`](deepchecks/check_policy.md), [`hallazgos.md`](deepchecks/hallazgos.md)): `0.30%` de etiquetas conflictivas (límite `FAIL`: `0.50%`, concentradas en horas de madrugada con flujo cero) y `6.75%` de mezcla de filas idénticas entre `train` y `test` (límite `FAIL`: `10.0%`).

---

## 5. Métricas y evaluación

### 5.1 Criterio de evaluación y calibración

Con ~`98.7%` de observaciones `Balanceado`, la exactitud (*Accuracy*) no refleja el valor operativo. Siguiendo el ML Canvas y `train_classification.py`:

- **Métrica primaria reportada:** **Macro-F1** (pondera por igual las tres clases).
- **Criterio de ranking de candidatos:** **`val_pr_auc_minority_mean`** (promedio del *Average Precision* uno-contra-resto de `Riesgo_Desabasto` y `Riesgo_Saturacion`, libre de umbral), con desempate por `val_macro_f1`.
- **Guardrail operativo obligatorio:** **`recall_shortage >= 0.85`**. Los umbrales se buscan conjuntamente en una grilla `19 × 19` (`0.05` a `0.95` en pasos de `0.05`) sobre **solo validación**, maximizando `val_macro_f1` sujeto a `val_recall_shortage >= 0.85` (`threshold_fallback = false`, `selection_fallback = false`).

### 5.2 Desempeño cuantitativo del Champion v3

| Métrica | Validación (oct–nov 2024, `523,754` filas) | Test OOT (dic 2024, `266,352` filas) | Evaluación Q4 2025 (`ab_summary.json`, `805,920` filas) |
|:---|---:|---:|---:|
| **`macro_f1`** | `0.4801` | `0.4578` | `0.4563` |
| **`pr_auc_minority_mean`** | `0.2884` | `0.2462` | `0.1970` |
| **`pr_auc_shortage`** | `0.2460` | `0.2009` | — |
| **`pr_auc_saturation`** | `0.3309` | `0.2915` | — |
| **`pr_auc_macro`** | `0.5253` | `0.4972` | — |
| **`recall_shortage`** (piso `>= 0.85`) | **`0.8605`** (✓) | `0.8452` | **`0.7854` (`0.785`)** (**✗ bajo piso**) |
| **`precision_shortage`** | `0.0662` | `0.0476` | `0.0425` |
| **`recall_saturation`** | `0.3306` | `0.3423` | `0.3194` |
| **`precision_saturation`** | `0.4103` | `0.3188` | — |
| **`alerts_per_day`** | `812.1` | `765.6` | `802.6` |
| **`accuracy` / `balanced_accuracy`** | `0.9073` / `0.7008` | `0.9105` / `0.7002` | — |

> [!WARNING]
> **Degradación explícita por debajo del piso operativo (`0.85`):**  
> En la evaluación fuera de muestra de octubre–diciembre de 2025 (`ab_summary.json`), el **recall de desabasto del champion fue `0.785` (`0.7854`)**, **explícitamente por debajo del piso mínimo requerido de `0.85`** (y `0.060` puntos debajo del test de diciembre de 2024, `0.8452`). Esta caída ocurre sin drift estadístico detectable en las variables de entrada y se asocia al descenso en la prevalencia de desabasto (`0.674%` en ajuste 2024 vs. `0.475%` en Q4 2025) frente a umbrales calibrados de manera fija en 2024.  
> Asimismo, las metas aspiracionales originales del ML Canvas (`Macro F1 >= 0.72`, `PR-AUC >= 0.82`) no son alcanzables con la prevalencia real (~`0.5%–0.7%` por clase de riesgo) sin inventario en tiempo real; el modelo opera como un detector de alta cobertura con baja precisión (`4.2%–4.8%`).

### 5.3 Comparativa frente a baselines (Validación oct–nov 2024)

| Modelo en MLflow | `val_pr_auc_minority_mean` | `val_macro_f1` | `val_recall_shortage` | Observación |
|:---|---:|---:|---:|:---|
| **Champion v3 (`lightgbm`, `num_leaves=255`, `balanced`)** | **`0.2884`** | `0.4801` | **`0.8605`** | Único finalista limpio que cumple el piso `>= 0.85` con mayor AP |
| `lightgbm` baseline (`num_leaves=31`, `balanced`) | `0.2576` | `0.4653` | `0.8553` | Cumple guardrail pero con menor separación PR-AUC |
| `lightgbm_unweighted` (sin `class_weight`) | `0.2235` | **`0.5366`** | `0.4181` | Macro-F1 mayor por precisión en saturación, pero pierde más de la mitad de desabastos |
| `logistic_regression` (`StandardScaler` + `balanced`) | `0.0774` | `0.3001` | `0.8842` | Capacidad de ordenamiento muy baja; exceso de falsas alarmas |
| `persistence` / `dummy_most_frequent` | `0.0067` | `0.3311` | `0.0000` | Nula detección de riesgo |

Las evidencias gráficas del conjunto de prueba del champion están disponibles en [`reports/figures/entrega2/08-p2-test-confusion-matrix.png`](figures/entrega2/08-p2-test-confusion-matrix.png), [`09-p2-test-pr-curve-shortage.png`](figures/entrega2/09-p2-test-pr-curve-shortage.png) y [`10-p2-test-pr-curve-saturation.png`](figures/entrega2/10-p2-test-pr-curve-saturation.png).

---

## 6. Limitaciones y riesgos

1. **Estaciones nuevas o retiradas:** En 2025 operaron todo el año `365` estaciones frente a `358` en 2024: **10 estaciones nuevas** que no estaban en el panel de 2024 y **3 estaciones** de 2024 ausentes del panel continuo de 2025 (`reports/TC5061_Drift_ABTest_Equipo05.md`, Anexo B). Aunque las coordenadas continuas permiten inferir sin errores de esquema, el modelo nunca vio esas 10 ubicaciones en entrenamiento: en el desglose A/B de Q4 2025 sobre las 10 estaciones nuevas (`22,080` filas, solo `11` eventos de desabasto y `12` de saturación), el `pr_auc_minority_mean` cayó a **`0.0170`** frente a **`0.1984`** en las 355 estaciones comunes. Además, la apertura o retiro de estaciones altera la identidad de los 3 vecinos más cercanos en `neighbor_net_flow`.
2. **Capacidad estática del catálogo:** El pipeline utiliza una única fotografía del catálogo oficial (`nomenclatura_2026_09.csv`, con capacidades entre `11` y `39` anclajes en el panel activo) y asume que $C_s$ no cambió entre 2024, 2025 y 2026. Si una estación amplió o redujo puertos, tanto el predictor `capacity` como la etiqueta normalizada $\Delta_{s,t}$ quedan sesgados sin reflejarse en los análisis de drift.
3. **Etiqueta proxy y retraso de 1 hora en la retroalimentación:** Al no existir inventario real ($B_{s,t}$), un flujo neto negativo elevado puede no provocar desabasto si la estación inició la hora completamente llena, y viceversa. Adicionalmente, la etiqueta real `target` solo se consolida al cierre de la hora objetivo (`label_available_at = target_time + 1h`), por lo que la evaluación de aciertos en producción tiene un **retraso mínimo de 1 hora**.
4. **Asimetría de costos (desabasto no detectado vs. falsa alarma):** No detectar un desabasto real deja a los usuarios sin servicio y erosiona la confianza en el sistema (costo operativo alto, razón del piso `recall_shortage >= 0.85`). En contrapartida, con una prevalencia real de `0.48%–0.67%`, sostener alta sensibilidad mediante `threshold_shortage = 0.20` produce una **precisión de `4.25%–4.76%`** (`~766–803` alertas diarias sobre `365` estaciones, es decir, `~20` alertas preventivas por cada evento real). Despachar cuadrillas sin priorización probabilística saturaría la flota operativa; en campo debe ordenarse por $P(\text{Desabasto} \mid X)$ y capacidad $C_s$.
5. **Sensibilidad del drift agregado:** El reporte de Evidently compara muestras estratificadas de `100,000` filas a nivel de toda la red con umbral `0.10`, lo que puede diluir cambios focalizados en un subconjunto de estaciones o franjas horarias específicas.
6. **Concentración en variables espaciales:** `longitude` (`18.91%`), `latitude` (`14.49%`), `net_flow_lag24` (`13.97%`), `hour_cos` (`13.48%`) y `hour_sin` (`8.89%`) concentran ~`69.7%` del *gain* total de LightGBM; el error puede variar significativamente entre corredores céntricos y estaciones periféricas de menor rotación.

---

## 7. Monitoreo, A/B testing y Continuous Training

El reporte completo de observabilidad se encuentra en [`reports/TC5061_Drift_ABTest_Equipo05.md`](TC5061_Drift_ABTest_Equipo05.md), [`reports/monitoring/drift_2024_vs_2025.html`](monitoring/drift_2024_vs_2025.html) y [`reports/monitoring/ab_summary.json`](monitoring/ab_summary.json).

### 7.1 Drift de datos, objetivo y predicciones (2024 vs. 2025)

- **Configuración (Evidently 0.7.23):** Comparación de ventanas equivalentes de 9 meses —referencia `2 ene – 30 sep 2024` (`train`) contra actual `2 ene – 30 sep 2025`— sobre muestras estratificadas por mes de `100,000` filas (`seed=42`). Se usaron distancias robustas al tamaño muestral (Wasserstein normalizada para continuas y divergencia de Jensen-Shannon para categóricas/booleanas, umbral `0.10`).
- **Resultado:** **`0 / 20` columnas con drift (`dataset_drift = false`, `trigger_data_drift = false`, `trigger_label_drift = false`)**. Las mayores distancias correspondieron a rezagos recientes de viajes (`departures_lag3 = 0.0294`, `departures_lag1 = 0.0286`, `arrivals_lag1 = 0.0268`), muy debajo de `0.10`. Las tres variables más importantes (`longitude = 0.0117`, `latitude = 0.0150`, `net_flow_lag24 = 0.0162`), `target` (`0.0033`) y `prediction` (`0.0044`) se mantuvieron estables; las 5 variables de calendario usadas como control negativo registraron distancias `< 0.01` (`0.0004–0.0085`). La reproducibilidad del reporte se verificó en una segunda máquina con diferencia máxima de punto flotante de $3.5 \times 10^{-18}$.

### 7.2 A/B testing entre versiones (octubre–diciembre 2025)

Se comparó al **Champion v3** (entrenado ene–sep 2024, calibrado oct–nov 2024, umbrales `0.20 / 0.95`) contra un **Challenger v2** (mismos hiperparámetros de LightGBM, reentrenado ene 2024 – jul 2025 sobre las 355 estaciones comunes y calibrado ago–sep 2025 con umbrales `0.25 / 0.95`) sobre **`805,920` filas** (`93` días de entrada, `365` estaciones) de **octubre–diciembre de 2025**, periodo que ningún modelo vio.

| Métrica | Champion v3 | Challenger v2 | Diferencia (IC 95% bootstrap por día) | Regla de promoción (fijada el 9 nov) |
|:---|---:|---:|---:|:---|
| **`recall_shortage`** | `0.7854` | **`0.8358`** | `+0.0504` `[+0.0409, +0.0610]` | **✗ Falla** (`>= 0.85` requerido) |
| **`pr_auc_minority_mean`** | `0.1970` | **`0.2263`** | `+0.0292` `[+0.0257, +0.0324]` | ✓ Cumple (IC > 0) |
| **`macro_f1`** | **`0.4563`** | `0.4540` | `-0.0023` `[-0.0051, +0.0004]` | ✓ Cumple (caída $\le 0.01$) |
| **`alerts_per_day`** | **`802.6`** | `943.9` | `+141.3` (`1.18×`) | ✓ Cumple ($\le 1.20\times$) |

- **Decisión: se mantiene el Champion v3.** Aunque el challenger mejora de forma significativa el PR-AUC (`+0.0292`) y el recall de desabasto (`+0.0504`, McNemar: `504` aciertos exclusivos vs. `145` del champion), no alcanza el piso operativo de `0.85` (`0.8358`). En apego estricto a la regla preacordada, no se promueve el challenger y se conserva el champion v3 en producción, dejando constancia de su degradación (`0.7854`).

### 7.3 Propuesta de Continuous Training (CT)

Por decisión de alcance de la entrega final (Decisión F.8 del plan de acción), el ciclo automatizado de CT **no se implementa en código**, pero queda especificado a partir de los hallazgos de la Entrega 5:

- **Disparador principal (por desempeño):** Dado que el drift de entradas no anticipó la caída de recall, el disparador primario evalúa métricas reales en ventana móvil de 7 días (factible porque la etiqueta llega 1 hora después): `recall_shortage < 0.85` o caída de `pr_auc_minority_mean > 0.03` respecto al test de 2024 (`< 0.216`). Con los datos de Q4 2025, este disparador ya estaría activo.
- **Alerta temprana (por drift):** `trigger_data_drift` (`> 30%` de variables con drift o drift en `longitude`, `latitude` o `net_flow_lag24`) o `trigger_label_drift` (drift en `target` o `prediction`).
- **Cadencia y calibración con margen:** Recalibración **semanal** de umbrales (sin reentrenar, buscando un recall $\ge 0.88$ en calibración para absorber caídas de prevalencia), reentrenamiento **mensual** con ventana móvil de ~18 meses y compuerta automática con `make ab-test` bajo la regla vigente antes de promover en el Registry.

---

## 8. Linaje y reproducción

### 8.1 Trazabilidad exacta del artefacto (`models/p2-riesgo/registry_metadata.json`)

| Campo de linaje | Valor exacto |
|:---|:---|
| `model_name` | `mibici-equipo05-p2-riesgo` |
| `model_version` / `model_alias` | `3` / `champion` |
| `source_run_id` | `7bf71250a91947ba8465d5db48d4bce5` |
| `selection_run_id` | `88490b76821842ba8c3393fad38e5739` |
| `git_commit` | `71b6c42f47040bd5ada00078982a01fe6a8e9dd4` (`git_dirty=false`) |
| `data_version` | `4c54003cb6d8c15f49dd6016daf984dbacc464700d268f9691465ffdaead4246` |
| `features_data_version` (`feature_table_sha256`) | `c92233fe5134482c2895869bbaa47081ea3df8c4f87f1f95e2e17b598cbe6df0` |
| `model.pkl` (SHA-256) | `6dbe66145e1bfabd74fd1ab1ccc4fdefd60aa18a35bbaaf9b0d62d5d3b9df7c7` |
| `models/p2-champion-v3.zip` (SHA-256) | `73cab932d0eae7dbf2c4f78fe927610353a9184e86588d0eddf4cfcef23b305b` |

La **reproducibilidad técnica cruzada** fue validada en una segunda máquina (macOS Apple Silicon sobre commit `8afe775`, `git_dirty=false`), obteniendo los mismos umbrales (`0.20 / 0.95`) y una variación relativa de `0.59%` en `test_macro_f1` (`0.4551` vs. `0.4578`), muy por debajo del umbral de tolerancia del `30%`.

### 8.2 Comandos para reentrenar, registrar y operar

```bash
# 1. Entorno, datos y verificación de hashes
uv sync
make data
uv run python tc5061_proyecto_equipo5/dataset.py manifest --check
uv run python -m tc5061_proyecto_equipo5.features classification

# 2. Servidor MLflow local (en terminal aparte) y variables de entorno
make mlflow-server                          # http://127.0.0.1:5001
set -a; source .env; set +a

# 3. Entrenamiento, selección y registro en Model Registry (MLproject)
uv run mlflow run . -e train -P question=p2 -P model=lightgbm -P stage=candidate \
  -P hp_num_leaves=255 -P hp_min_child_samples=100 -P hp_n_estimators=200 --env-manager local
uv run mlflow run . -e select -P question=p2 --env-manager local
uv run mlflow run . -e register -P run_id=<run_id_ganador> -P alias=champion --env-manager local

# 4. Exportación (o restauración desde bundle versionado) y levantamiento en Compose
make export-model                           # o `make prepare-model` desde p2-champion-v3.zip
make up && make compose-smoke

# 5. Monitoreo de drift y A/B testing (Entrega 5)
make features-current
make drift-report
make features-challenger && make train-challenger
make ab-test
```

---

## 9. Costo operativo (FinOps)

- **Estimación mensual de la opción elegida:** En la arquitectura recomendada (**Opt 4: ejecución *scale-to-zero* con una corrida horaria por lotes de 365 estaciones y sin MLflow en inferencia**), el costo operativo estimado es de **\$0.0011 USD/mes en AWS Lambda** (**\$0.0018 USD/mes en Google Cloud Run**; menos de \$0.01 USD/mes aun con margen conservador de latencia), frente a **\$15.98 USD/mes** (**\$59.99 USD por millón de predicciones** para la demanda real de `266,450` predicciones/mes) en una instancia EC2 `t3.small` encendida 24/7 —una reducción de costo del **99.99%** ([`reports/load/finops_summary.json`](load/finops_summary.json), [`reports/TC5061_Operacion_FinOps_Equipo05.md`](TC5061_Operacion_FinOps_Equipo05.md), figura comparativa en [`04_finops_comparativa_optimizaciones.png`](figures/entrega6/04_finops_comparativa_optimizaciones.png)).

| Opción evaluada (`finops_summary.json`) | Costo por 1M predicciones | Costo mensual | Ahorro |
|:---|---:|---:|---:|
| Línea base en pico sostenible (`nominal_10u`, EC2 `t3.small`, `583.8 pred/s`) | `$0.01258` | `$15.98` (+ burst) | — |
| Línea base operación real 24/7 (`t3.small`, `266,450 pred/mes`) | `$59.99` | `$15.98` | — |
| **Opt 1:** Batching de 365 estaciones por petición (vs. pico) | `$0.00315` | `$15.98` | `74.95%` |
| **Opt 2:** Retirar MLflow del camino de inferencia (`--no-deps api`, vs. pico) | `$0.00836` | `$10.63` | `33.49%` |
| **Opt 3:** Inferencia directa con `Booster` + `numpy` (`1.61×` en `batch=365`, vs. pico) | `$0.00781` | `$15.98` | `37.89%` |
| **Opt 4 (recomendada):** *Scale-to-zero* horario en AWS Lambda (vs. 24/7 real) | `$0.00411` | **`$0.0011`** | **`99.99%`** |

---

## 10. Referencias

- Amazon Web Services. (2026). *Amazon EC2 On-Demand pricing* & *AWS Lambda pricing*. https://aws.amazon.com/lambda/pricing/
- Dorard, L. (2018). *From data to AI with the Machine Learning Canvas (Part I)*. Own Machine Learning.
- Evidently AI. (2026). *Evidently documentation* (versión 0.7.23). https://docs.evidentlyai.com/
- FinOps Foundation. (2026). *What is FinOps?* https://www.finops.org/introduction/what-is-finops/
- Gift, N., & Deza, A. (2021). *Practical MLOps: Operationalizing machine learning models*. O'Reilly Media.
- Huyen, C. (2022). *Designing machine learning systems: An iterative process for production-ready applications*. O'Reilly Media.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. En *Advances in Neural Information Processing Systems 30* (pp. 3146–3154). Curran Associates.
- MiBici. (2026). *Datos abiertos* [Conjunto de datos]. Sistema de Bicicleta Pública del Área Metropolitana de Guadalajara / AMIM. https://www.mibici.net/es/datos-abiertos/
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\* '19)* (pp. 220–229). ACM. https://doi.org/10.1145/3287560.3287596
- Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A., Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps: How to scale machine learning in the enterprise*. O'Reilly Media.
- Wilson, B. (2022). *Machine learning engineering in action*. Manning Publications.
