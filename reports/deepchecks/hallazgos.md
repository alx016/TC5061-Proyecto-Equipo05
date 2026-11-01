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

La tabla muestra la corrida con los umbrales originales de 0% y 5%. La política actual conserva las alertas y usa límites de 0.5% y 10% para detener la ejecución.

| Suite de Validación | Chequeos Totales | Aprobados | No Aprobados | Informativos (Sin Condición) | Veredicto General |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Data Integrity** | 13 | 11 | 1* | 1 | **Aprobado con observaciones operativas** |
| **Train/Test Drift** | 12 | 11 | 1* | 0 | **Estable (Drift controlado dentro de tolerancias)** |

*\*Los dos resultados excedieron los umbrales de diagnóstico originales. Las posibles causas descritas abajo son hipótesis, no una causa raíz demostrada. La política documenta límites operativos acotados y las alertas siguen visibles y requieren revisión.*

---

## 3. Chequeos de Integridad de Datos (`data_integrity`)

El reporte interactivo completo se regenera con `make data-checks` en `reports/deepchecks/data_integrity.html` y su resumen estructurado en `reports/deepchecks/data_integrity.json` (no se versionan).

| Chequeo | Métrica Observada | Umbral de Política | Resultado |
| :--- | :--- | :--- | :--- |
| **Data Duplicates** | 0.0% duplicados | Ratio $\le 0.0\%$ | **PASS** |
| **Percent Of Nulls** | Máximo 0.25% en `net_flow_lag24` | Ratio $\le 1.0\%$ por columna | **PASS** |
| **Single Value Column** | 0 columnas univaluadas | Ninguna columna constante | **PASS** |
| **Special Characters** | 0.0% caracteres anómalos | Ratio $\le 0.1\%$ | **PASS** |
| **Mixed Data Types** | 0 columnas con tipos mixtos | 100% tipos puros por columna | **PASS** |
| **Mixed Nulls** | 0 tipos de nulos mezclados | Coherencia estricta de nulos | **PASS** |
| **String Mismatch** | 0 variantes no estandarizadas | 0 variantes espurias | **PASS** |
| **String Length Out of Bounds** | 0 cadenas atípicas | 0 longitudes anómalas | **PASS** |
| **Feature Correlation** | Correlación máx. $< 0.90$ | Correlaciones entre pares $< 0.90$ | **PASS** |
| **Feature Label Correlation** | PPS máx. $< 0.90$ | PPS predictivo $< 0.90$ | **PASS** |
| **Identifier Label Correlation** | PPS máx. $< 0.90$ | Claves sin fuga predictiva | **PASS** |
| **Conflicting Labels** | 0.3% muestras idénticas | Ratio $\le 0.0\%$ | **Alerta (0.3%)** |
| **Outlier Sample Detection** | Diagnóstico LoOP (muestras atípicas) | Diagnóstico informativo sin condición | **INFO** |

### Análisis de Hallazgos en Integridad
1. **Duplicados**: El chequeo de filas no detectó duplicados en la corrida registrada. Esto no demuestra por sí solo la unicidad de la clave compuesta (`station_id`, `hour`).
2. **Nulos controlados por diseño de rezagos**: El único predictor que presenta nulos es `net_flow_lag24` (0.25%, exactamente las primeras 24 horas del año requeridas para inicializar el historial), cumpliendo holgadamente el umbral de $<1.0\%$.
3. **Muestras con etiquetas conflictivas (0.3%)**: Una hipótesis es que vectores iguales de flujos y rezagos puedan preceder resultados distintos. También podrían existir errores en datos o etiquetas. Los resultados disponibles no identifican la causa. Deben inspeccionarse los grupos conflictivos y contrastarse con la fuente antes de descartarlos como ruido.
4. **Chequeo informativo de anomalías (`Outlier Sample Detection`)**: Se evalúa mediante el algoritmo LoOP (Local Outlier Probabilities). Al ser un análisis puramente exploratorio en DeepChecks, se ejecuta sin condiciones de aprobación o fallo binario, por lo que no altera la tasa de éxito técnico.

Al revisar los resultados históricos encontramos 1,215 grupos con características iguales. De ellos, 1,145 tienen más de una etiqueta no nula y 70 incluyen una etiqueta ausente. La mayoría de las filas corresponde a las 05:00 o 06:00 y el 91.6% tiene cero llegadas y salidas en el primer rezago. Esto apunta a periodos de baja actividad, aunque no explica por sí solo las diferencias en las etiquetas.

---

## 4. Chequeos de Drift Temporal (`train_test_drift`)

El reporte interactivo completo se regenera con `make data-checks` en `reports/deepchecks/train_test_drift.html` y su resumen estructurado en `reports/deepchecks/train_test_drift.json` (no se versionan).

| Chequeo | Métrica Observada | Umbral de Política | Resultado |
| :--- | :--- | :--- | :--- |
| **Datasets Size Comparison** | Ratio 0.11 (266K test vs 2.35M train) | Proporción representativa | **PASS** |
| **New Label Train Test** | 0 etiquetas nuevas | Mismas 3 clases operativas | **PASS** |
| **New Category Train Test** | 0 categorías nuevas | Categorías válidas en calendario | **PASS** |
| **String Mismatch Comparison** | 0 variantes no estandarizadas | Coherencia de cadenas | **PASS** |
| **Date Train-Test Leakage (Duplicates)** | 0.0% fechas duplicadas | Ratio $\le 0.0\%$ | **PASS** |
| **Date Train-Test Leakage (Overlap)** | 0.0% solapamiento temporal | Ratio $\le 0.0\%$ | **PASS** |
| **Index Train-Test Leakage** | 0.0% identificadores solapados | Ratio $\le 0.0\%$ | **PASS** |
| **Feature Label Correlation Change** | Correlaciones estables con target | Diferencia de PPS acotada | **PASS** |
| **Multivariate Drift** | Drift value = 0.15 (AUC = 0.57) | Drift value $< 0.25$ | **PASS** |
| **Feature Drift** | Max KS 0.03, Max Cramer V 0.06 | Scores $< 0.20$ | **PASS** |
| **Label Drift** | Cramer's V = 0.02 | Score $< 0.15$ | **PASS** |
| **Train Test Samples Mix** | 6.75% filas idénticas de predictores y etiqueta | Umbral original $\le 5.0\%$ | **Alerta (6.75%)** |

### 4.1 Aislamiento y Fuga Temporal (Leakage)
- **Date Train-Test Leakage (Overlap)**: **0.0% solapamiento detectado (PASS)**. Las particiones se asignan por la fecha de la etiqueta (`hour + 1h`). Las horas de entrada de prueba empiezan el 30 de noviembre a las 23:00.
- **Date Train-Test Leakage (Duplicates)**: **0.0% fechas duplicadas (PASS)**.
- **Relación de tamaños (Size Ratio)**: 0.11 (266,352 filas de test vs. 2,353,850 filas de train), consistente con una proporción esperada de 1 mes contra 9 meses.

### 4.2 Métricas de Drift Multivariado y Distribución
- **Multivariate Drift (Domain Classifier)**:
  - **Métrica observada**: Drift value = 0.15 (AUC de clasificador de dominio = 0.57).
  - **Umbral**: Drift value $< 0.25$.
  - **Interpretación**: En la muestra evaluada, el clasificador que distingue entrenamiento y prueba obtuvo un AUC cercano a 0.57, próximo al azar (0.50). Este resultado por sí solo no garantiza estabilidad para todos los periodos o segmentos.

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
- **Métrica observada**: 6.75% de las muestras de prueba evaluadas tienen predictores y etiqueta idénticos a alguna muestra de entrenamiento evaluada (umbral original: 5.0%). DeepChecks incluye la etiqueta en esta comparación, así que no cuantifica todas las coincidencias de predictores con etiquetas distintas.
- **Hipótesis pendiente de comprobar**: La repetición podría deberse a periodos de baja actividad y características estáticas de estación. La coincidencia de predictores por sí sola no demuestra fuga, pero tampoco la descarta. Hay que examinar filas coincidentes y la construcción temporal de cada variable.
- **Revisión adicional**: Encontramos 2,586 combinaciones distintas que aparecen en ambas muestras. Todas tienen etiqueta 0 y el 99.6% tiene cero llegadas y salidas en el primer rezago. Esto apoya la hipótesis de baja actividad, pero no explica la causa de todos los casos.

---

## 5. Conclusiones y Recomendaciones de Modelado

1. **Partición temporal**: Los chequeos de fechas no detectaron solapamiento en la corrida documentada. Esto no descarta otros tipos de fuga en las variables o etiquetas.
2. **Generalización**: Las métricas de drift describen diferencias de distribución en la muestra evaluada. No prueban que el modelo mantenga su desempeño. La capacidad predictiva debe medirse con métricas de clasificación sobre datos de prueba reservados y por segmentos relevantes.
3. **Trazabilidad de Reportes**: Tanto los reportes gráficos en HTML para inspección humana como los archivos JSON estandarizados para automatización se regeneran con `make data-checks` en `reports/deepchecks/`; este documento y `check_policy.md` sí se versionan.
