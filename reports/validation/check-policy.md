# Política de Validación de Datos y Drift Temporal (Check Policy)

## 1. Alcance y Contexto de Gobernanza

Esta política formaliza las reglas, métricas, umbrales y criterios de severidad para la validación automática de datos en el proyecto MiBici MLOps (Equipo 05). Se ejecuta mediante el script independiente `scripts/data_checks.py` utilizando DeepChecks 0.19.1 en un entorno aislado bajo especificación PEP 723.

### Conjunto de Datos Evaluado
- **Ruta**: `data/processed/classification_features.parquet`
- **Estructura**: Panel regular estación-hora 2024 (358 estaciones $\times$ 8,784 horas = 3,144,672 filas, 25 columnas).
- **Partición temporal**:
  - **Entrenamiento (Train)**: Enero a Septiembre de 2024 (Meses 1–9, 2,353,850 filas).
  - **Prueba OOT (Test)**: Diciembre de 2024 (Mes 12, 266,352 filas).
  - **Validación intermedia**: Octubre y Noviembre de 2024 (reservados, aislados de la prueba ciega).

---

## 2. Inventario de Variables y Exclusiones

### 2.1 Predictoras Incluidas (18 variables)
| Nombre de Variable | Tipo de Dato | Rol / Dominio |
| :--- | :--- | :--- |
| `capacity` | Numérico continuo | Capacidad física de anclajes de la estación |
| `latitude`, `longitude` | Numérico continuo | Coordenadas geográficas estáticas de la estación |
| `hour_sin`, `hour_cos` | Numérico continuo | Codificación trigonométrica del ciclo horario diario |
| `day_of_week` | Categórico ordinal | Día de la semana (0 = Lunes, 6 = Domingo) |
| `is_weekend` | Categórico binario | Indicador de sábado o domingo (0 o 1) |
| `is_holiday` | Categórico binario | Indicador de día festivo oficial en Jalisco |
| `neighbor_net_flow` | Numérico continuo | Flujo neto agregado de estaciones vecinas |
| `net_flow_lag1`, `net_flow_lag2`, `net_flow_lag24` | Numérico continuo | Rezagos causales de flujo neto normalizado |
| `arrivals_lag1`, `arrivals_lag2`, `arrivals_lag3` | Numérico discreto | Rezagos causales de viajes concluidos (llegadas) |
| `departures_lag1`, `departures_lag2`, `departures_lag3` | Numérico discreto | Rezagos causales de viajes iniciados (salidas) |

### 2.2 Variable Objetivo
- `target`: Clasificación multiclase a 1 hora horizonte (0 = Balance normal, 1 = Falta de bicis, 2 = Saturación de puertos).

### 2.3 Variables Excluidas del Cálculo Estadístico de Drift
- `station_id`: Identificador categórico de alta cardinalidad; su distribución estática se valida por ausencia de nulos y cardinalidad fija.
- `record_id`: Clave primaria sintética única (`station_id_hour`), empleada como índice.
- `hour`: Marca temporal (`datetime`), evaluada exclusivamente para aserciones de cero solapamiento y secuencia cronológica.

---

## 3. Especificación de Reglas de Integridad (`data_integrity`)

| Regla / Chequeo | API DeepChecks | Métrica y Umbral | Tolerancia / Excepción Justificada | Severidad | Acción de Falla |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Unicidad de Registros** | `DataDuplicates` | Ratio duplicados $\le 0.0\%$ | Cero tolerancia en la clave `station_id_hour`. | **CRÍTICA** | Bloquear pipeline de modelado. |
| **Completitud de Atributos** | `PercentOfNulls` | Nulos por columna $\le 1.0\%$ | Se permite hasta $0.25\%$ únicamente en `net_flow_lag24` por las 24 hrs iniciales de inicialización temporal. | **CRÍTICA** | Bloquear si excede $1.0\%$ o si ocurre en `capacity`, `target` o coordenadas. |
| **Columnas Univaluadas** | `SingleValueColumn` | 0 columnas constantes | Ninguna predictora puede ser invariable a lo largo del año. | **ALTA** | Bloquear pipeline. |
| **Pureza de Tipos** | `MixedDataTypes` | 0 columnas con tipos mixtos | Cada columna debe tener un tipo primitivo estricto (float, int o datetime). | **CRÍTICA** | Bloquear pipeline. |
| **Caracteres Especiales** | `SpecialCharacters` | Ratio anómalo $\le 0.1\%$ | Cero cadenas corruptas o caracteres invisibles. | **ALTA** | Bloquear pipeline. |
| **Etiquetas Conflictivas** | `ConflictingLabels` | Ratio en conflicto $\le 0.0\%$ | Se permite alerta informativa ($\le 0.5\%$) por redundancia estática en madrugadas con intervención manual. | **MEDIA** | Registrar en bitácora de auditoría sin detener ejecución. |

---

## 4. Especificación de Reglas de Drift Temporal (`train_test_drift`)

### 4.1 Política de Muestreo
Para garantizar tiempos de ejecución acotados en suites de alta dimensionalidad sin pérdida de significancia estadística, se utiliza una muestra representativa estratificada de hasta $50,000$ registros (`n_samples=50000`). Los chequeos de fuga temporal y nulos se computan sobre el conjunto completo.

### 4.2 Reglas y Umbrales
| Dimensión de Drift | Chequeo DeepChecks | Métrica Estadística | Umbral de Alerta | Disposición |
| :--- | :--- | :--- | :---: | :--- |
| **Fuga Temporal (Leakage)** | `IndexTrainTestLeakage` | Solapamiento de fechas / registros | Exactamente **0.0%** | **CRÍTICA**: Si hay $>0.0\%$, bloquea de inmediato. |
| **Drift Multivariado** | `MultivariateDrift` | Drift value (Clasificador de dominio) | $< 0.25$ (AUC equivalente $< 0.625$) | Alerta de revisión: requiere inspección de estabilidad conjunta. |
| **Drift de Predictoras Numéricas** | `FeatureDrift` | Prueba Kolmogorov-Smirnov (KS) | Estadístico KS $< 0.20$ | Alerta individual: evalúa si una feature específica requiere calibración. |
| **Drift de Predictoras Categóricas** | `FeatureDrift` | V de Cramér | V de Cramér $< 0.20$ | Alerta individual de distribución temporal. |
| **Drift de Etiqueta Objetivo** | `LabelDrift` | V de Cramér en `target` | V de Cramér $< 0.15$ | Alerta prioritaria: evalúa si cambió la prevalencia del riesgo operativo. |
| **Mezcla de Vectores (Samples Mix)** | `TrainTestSamplesMix` | Vectores idénticos entre train y test | $\le 5.0\%$ (por defecto) | Alerta operativa tolerada hasta $10.0\%$ por vectores estáticos nocturnos (0 viajes continuos de madrugada). |

---

## 5. Política de Respuestas y Salidas de Error

1. **Falla de Integridad (`data_integrity['passed'] == False`)**:
   - Genera los reportes visuales `data_integrity.html` y estructurados `data_integrity.json`.
   - Se marca como condición no cumplida en la suite.
   - En ejecución de pipeline productivo o pruebas Pytest (`tests/test_data_checks.py`), la violación de integridad genera un código de salida distinto de cero o falla de aserción.
2. **Advertencias de Drift Temporal**:
   - El drift temporal natural (cambio estacional entre invierno y el resto del año) no invalida la integridad técnica del dataset.
   - Genera `train_test_drift.html` y `train_test_drift.json` para inspección en el PRR (Production Readiness Review).
   - Los hallazgos se documentan en `reports/deepchecks/hallazgos.md`.
