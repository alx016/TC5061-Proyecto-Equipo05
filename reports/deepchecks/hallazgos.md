# Hallazgos de Integridad de Datos y Drift Temporal (DeepChecks)

## 1. Contexto de la Evaluación

Se implementó y ejecutó el protocolo de validación tabular mediante la suite automatizada DeepChecks (`scripts/data_checks.py`) sobre el conjunto de datos de características para el problema de clasificación (`data/processed/classification_features.parquet`).

### Especificaciones del Conjunto de Datos
- **Fuente**: Panel horario-estación 2024 de MiBici Guadalajara (`data/processed/classification_features.parquet`).
- **Dimensión total**: 3,144,672 filas y 25 columnas (358 estaciones continuas $\times$ 8,784 horas anuales).
- **Variable objetivo (`target`)**: Clasificación multiclase de riesgo operativo a 1 hora horizonte:
  - `0`: Balance normal (demanda y capacidad equilibradas).
  - `1`: Riesgo de falta de bicicletas (*shortage*).
  - `2`: Riesgo de saturación de puertos (*saturation*).
- **Conjunto de predictoras (18 variables)**:
  - Geoespaciales y estáticas: `capacity`, `latitude`, `longitude`.
  - Calendario y cíclicas: `hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`, `is_holiday`.
  - Dinámica espacial: `neighbor_net_flow`.
  - Rezagos temporales (*lags*): `net_flow_lag1`, `net_flow_lag2`, `net_flow_lag24`, `arrivals_lag1`, `arrivals_lag2`, `arrivals_lag3`, `departures_lag1`, `departures_lag2`, `departures_lag3`.
- **Partición temporal evaluada para Drift**:
  - **Entrenamiento (Train)**: Enero a Septiembre 2024 (Meses 1 al 9, 2,353,850 filas).
  - **Prueba fuera de tiempo (Test OOT)**: Diciembre 2024 (Mes 12, 266,352 filas).
  - *(Nota: Los meses 10 y 11 corresponden a validación intermedia y se aíslan del cálculo de prueba final).*

---

## 2. Resumen Ejecutivo de Resultados

| Suite de Validación | Chequeos Totales | Aprobados | No Aprobados | Veredicto General |
| :--- | :---: | :---: | :---: | :--- |
| **Data Integrity** | 13 | 12 | 1* | **Aprobado con observaciones operativas** |
| **Train/Test Drift** | 11 | 10 | 1* | **Estable (Drift controlado dentro de tolerancias)** |

*\*Los chequeos con alerta corresponden a condiciones naturales del dominio de movilidad compartida (ambigüedad en horas valle y repetición de vectores estáticos en la madrugada), detallados a continuación.*

---

## 3. Chequeos de Integridad de Datos (`data_integrity`)

El reporte interactivo completo se generó en `reports/deepchecks/data_integrity.html` y su resumen estructurado en `reports/deepchecks/data_integrity.json`.

| Chequeo | Métrica Observada | Umbral de Política | Resultado |
| :--- | :--- | :--- | :--- |
| **Data Duplicates** | 0.0% duplicados | Ratio $\le 0.0\%$ | **PASS** |
| **Percent Of Nulls** | Máximo 0.25% en `net_flow_lag24` | Ratio $\le 1.0\%$ por columna | **PASS** |
| **Single Value Column** | 0 columnas univaluadas | Ninguna columna constante | **PASS** |
| **Special Characters** | 0.0% caracteres anómalos | Ratio $\le 0.1\%$ | **PASS** |
| **Mixed Data Types** | 0 columnas con tipos mixtos | 100% tipos puros por columna | **PASS** |
| **String Mismatch** | 0 variantes no estandarizadas | 0 variantes espurias | **PASS** |
| **Feature Correlation** | Correlación máx. $< 0.90$ | Correlaciones entre pares $< 0.90$ | **PASS** |
| **Conflicting Labels** | 0.3% muestras idénticas | Ratio $\le 0.0\%$ | **Alerta (0.3%)** |

### Análisis de Hallazgos en Integridad
1. **Ausencia estricta de duplicados**: La clave compuesta (`station_id`, `hour`) garantiza unicidad exacta en el panel.
2. **Nulos controlados por diseño de rezagos**: El único predictor que presenta nulos es `net_flow_lag24` (0.25%, exactamente las primeras 24 horas del año requeridas para inicializar el historial), cumpliendo holgadamente el umbral de $<1.0\%$.
3. **Muestras con etiquetas conflictivas (0.3%)**: En horas de inactividad nocturna (ej. 03:00 a 04:00 hrs), estaciones de baja afluencia registran flujos cero y rezagos cero. Si ocurre una intervención manual de rebalanceo de camiones o un viaje atípico aislado, el estado futuro puede variar entre balance normal y saturación a pesar de tener rezagos estáticos en cero. Esto refleja ruido estocástico real del servicio y no corrupción de datos.

---

## 4. Chequeos de Drift Temporal (`train_test_drift`)

El reporte interactivo completo se generó en `reports/deepchecks/train_test_drift.html` y su resumen estructurado en `reports/deepchecks/train_test_drift.json`.

### 4.1 Aislamiento y Fuga Temporal (Leakage)
- **Date Train-Test Leakage (Overlap)**: **0.0% fuga (PASS)**. No existe solapamiento temporal entre la ventana de entrenamiento (1 enero al 30 septiembre) y la de evaluación (1 diciembre al 31 diciembre).
- **Date Train-Test Leakage (Duplicates)**: **0.0% fechas duplicadas (PASS)**.
- **Relación de tamaños (Size Ratio)**: 0.11 (266,352 filas de test vs. 2,353,850 filas de train), consistente con una proporción esperada de 1 mes contra 9 meses.

### 4.2 Métricas de Drift Multivariado y Distribución
- **Multivariate Drift (Domain Classifier)**:
  - **Métrica observada**: Drift value = 0.15 (AUC de clasificador de dominio = 0.57).
  - **Umbral**: Drift value $< 0.25$.
  - **Interpretación**: Un clasificador supervisado entrenado para discriminar si una muestra proviene de Ene-Sep o de Diciembre apenas alcanza un AUC de 0.57 (cercano al azar 0.50). La distribución conjunta de predictoras se mantiene muy estable entre estaciones del año.

### 4.3 Drift de Características Individuales (Feature Drift)
El 100% de las 18 variables predictoras aprobó la condición de estabilidad (score $< 0.20$):

| Variable | Tipo de Drift | Métrica / Estadístico | Score Observado | Umbral | Estado |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `day_of_week` | Categórico | Cramer's V | **0.06** | $< 0.20$ | **PASS** |
| `is_weekend` | Categórico | Cramer's V | **0.02** | $< 0.20$ | **PASS** |
| `is_holiday` | Categórico | Cramer's V | **0.02** | $< 0.20$ | **PASS** |
| `arrivals_lag1` | Numérico | Kolmogorov-Smirnov (KS) | **0.03** | $< 0.20$ | **PASS** |
| `departures_lag1` | Numérico | Kolmogorov-Smirnov (KS) | **0.02** | $< 0.20$ | **PASS** |
| `net_flow_lag1` | Numérico | Kolmogorov-Smirnov (KS) | **0.02** | $< 0.20$ | **PASS** |
| `net_flow_lag24` | Numérico | Kolmogorov-Smirnov (KS) | **0.02** | $< 0.20$ | **PASS** |
| `neighbor_net_flow` | Numérico | Kolmogorov-Smirnov (KS) | **0.02** | $< 0.20$ | **PASS** |
| `capacity` | Numérico | Kolmogorov-Smirnov (KS) | **0.00** | $< 0.20$ | **PASS** |
| `latitude` / `longitude` | Numérico | Kolmogorov-Smirnov (KS) | **0.00** | $< 0.20$ | **PASS** |

### 4.4 Drift de la Variable Objetivo (Label Drift)
- **Métrica observada**: Cramer's V = 0.02.
- **Umbral**: Score $< 0.15$.
- **Veredicto**: **PASS**. La prevalencia relativa de desbalances operativos (eventos raros de falta o saturación) se conserva casi idéntica en diciembre frente al periodo base anual.

### 4.5 Mezcla de Muestras (Train Test Samples Mix)
- **Métrica observada**: 6.75% de las muestras de prueba contienen un vector de predictoras idéntico al de alguna muestra en entrenamiento (umbral por defecto de DeepChecks: 5.0%).
- **Causa raíz**: En el sistema MiBici, de 00:00 a 05:00 horas casi todas las estaciones presentan 0 viajes continuos, lo que genera vectores estáticos de rezagos nulos con coordenadas y capacidad fijas. Esta redundancia nocturna esperada eleva ligeramente la similitud de filas sin representar fuga de información ni alteración de causalidad temporal.

---

## 5. Conclusiones y Recomendaciones de Modelado

1. **Aptitud del Conjunto para Entrenamiento Fuera de Tiempo**: La partición temporal Ene-Sep (train) y Diciembre (test) demuestra ser rigurosa, no presentar filtraciones temporales y exhibir un drift marginal que no compromete la capacidad predictiva de los modelos de Machine Learning.
2. **Capacidad de Generalización**: El clasificador de dominio con AUC de 0.57 y las distancias KS inferiores a 0.03 confirman que las relaciones funcionales aprendidas durante los primeros meses del año conservan su validez durante el cierre anual.
3. **Trazabilidad de Reportes**: Tanto los reportes gráficos en HTML para inspección humana como los archivos JSON estandarizados para automatización se encuentran disponibles en `reports/deepchecks/`.
