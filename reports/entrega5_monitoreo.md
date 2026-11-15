# Monitoreo del modelo P2 con Evidently y A/B testing (Entrega 5)

<!--
Esqueleto del reporte (WP-5). Cada sección tiene un dueño: edita solo la tuya, en tu propio PR,
para que los cambios no choquen. Los comentarios como este dicen qué va en cada sección y se
borran cuando la sección queda escrita.
-->

- **Proyecto:** TC5061 MLOps, Equipo 05. Datos abiertos de MiBici.
- **Modelo monitoreado:** P2 riesgo de desabasto o saturación por estación
  (`mibici-equipo05-p2-riesgo`), el mismo que sirve la API de la entrega 4.
- **Fecha:** 15 de noviembre de 2026.

## Resumen

<!--
Dueño: Manuel (WP-5). Se escribe el sábado 14, al final.
Un párrafo: qué drift se encontró, quién ganó el A/B y qué se recomienda sobre Continuous
Training. Tres o cuatro cifras clave como máximo.
-->

*Pendiente.*

## 1. Conjuntos de datos

<!--
Dueño: Manuel (WP-5). Decisiones 5.2, 5.3 y 5.5 del plan.
Falta completar: filas y estaciones de cada conjunto (las reporta WP-0), cuántas estaciones son
nuevas o se retiraron en 2025, y el tamaño de la muestra que usa Evidently.
-->

| Conjunto | Periodo (hora objetivo) | Uso | Filas | Estaciones |
|:---|:---|:---|---:|---:|
| Referencia | 1 ene – 30 sep 2024 | Filas de `train` de P2: lo que vio el modelo al ajustarse | | |
| Actual | 1 ene – 30 sep 2025 | Comparación de drift contra la referencia | | |
| Evaluación del A/B | 1 oct – 31 dic 2025 | Evaluación de las dos versiones; ninguna lo vio | | |

*Pendiente:* justificar por qué se comparan los mismos meses de dos años distintos (el drift
no se explica por la temporada), por qué la referencia es solo `train` y no todo 2024, y por
qué la evaluación queda fuera de las ventanas del challenger.

## 2. Reporte de drift

<!--
Dueño: Emilio (WP-4a). Parte del 20% de interpretación de la rúbrica.
Gráficas en reports/figures/entrega5/; HTML interactivo en reports/monitoring/.
-->

### 2.1 Configuración

<!--
Presets de Evidently usados (DataDriftPreset, DataSummaryPreset), tamaño de muestra y semilla,
prueba estadística por tipo de variable y su umbral, y por qué distancias en vez de p-values
con muestras grandes (problema 5 del plan).
-->

*Pendiente.*

### 2.2 Calidad de datos

<!--
Lo que reporta DataSummaryPreset: nulos, rangos, valores nuevos. Comparar referencia y actual.
-->

*Pendiente.*

### 2.3 Drift por variable

<!--
Tabla con las 18 variables: prueba, distancia, umbral, ¿drift? Ordenada de mayor a menor.
Explicar las variables con drift agrupadas por causa: demanda (lags, flujos), red (capacity,
latitude, longitude) y calendario (is_holiday por el 1 de octubre de 2024).
Incluir 2 o 3 gráficas de las variables con más drift.
-->

*Pendiente.*

### 2.4 Control con variables de calendario

<!--
hour_sin, hour_cos, day_of_week e is_weekend no deberían tener drift (problema 6 del plan).
Decir si se cumple; si no, explicar.
-->

*Pendiente.*

### 2.5 Drift del objetivo y de la predicción

<!--
Proporción de Balanceado, Riesgo_Desabasto y Riesgo_Saturacion en la referencia y en el
periodo actual (columna target), y la distribución de la predicción del champion en ambos.
-->

*Pendiente.*

## 3. Implicaciones para el modelo

El análisis se realizó con el bundle local del champion P2 v3, run `7bf71250a91947ba8465d5db48d4bce5`. La ausencia de drift detectado en la muestra de entradas no garantiza que el modelo conserve su desempeño: en la evaluación posterior el recall de desabasto quedó por debajo del piso operativo de 0.85.

### Importancia de las variables y drift

La importancia se mide como gain de LightGBM normalizado a 100%; indica la contribución a las particiones del modelo, no una relación causal. Se compararon las filas elegibles de ajuste de 2024 contra enero–septiembre de 2025 mediante muestras estratificadas de 100,000 filas por periodo, semilla 42.

**Tabla 3.1. Importancia del champion y drift de las cinco variables principales.**

Fuente: gain del bundle champion v3 y comparación de drift descrita arriba.

| Variable | Gain | Distancia Wasserstein | Umbral | ¿Drift? |
|:---|---:|---:|---:|:---:|
| `longitude` | 18.91% | 0.0117 | 0.10 | No |
| `latitude` | 14.49% | 0.0150 | 0.10 | No |
| `net_flow_lag24` | 13.97% | 0.0162 | 0.10 | No |
| `hour_cos` | 13.48% | 0.0020 | 0.10 | No |
| `hour_sin` | 8.89% | 0.0012 | 0.10 | No |

Ninguna de las 18 variables de entrada superó su umbral en esta muestra. Para `target` y `prediction`, Jensen–Shannon produjo 0.0033 y 0.0044, respectivamente, ambos por debajo de 0.10. Estos resultados dependen de los conjuntos, muestra y pruebas elegidas y no descartan cambios que las métricas no detecten.

### Prevalencia, umbrales y desempeño

El champion usa umbrales de 0.20 para desabasto y 0.95 para saturación. Las clases de riesgo son poco frecuentes:

**Tabla 3.2. Prevalencia observada por cohorte.**

Fuente: tablas de features de 2024 y 2025; referencia filtrada con `load_rows` y `split_masks`.

| Cohorte | Filas con etiqueta | Desabasto | Saturación |
|:---|---:|---:|---:|
| Referencia: filas elegibles de ajuste, enero–septiembre 2024 | 2,344,900 | 0.674% | 0.610% |
| Actual: split `current`, enero–septiembre 2025 | 2,391,115 | 0.596% | 0.571% |

La referencia aplica las máscaras temporales del ajuste; la cohorte actual cuenta el split bruto con etiqueta conocida. El cambio de prevalencia no demuestra por sí solo un cambio en la precisión ni en el volumen de alertas. En el periodo reservado de octubre–diciembre 2025, sobre 805,920 filas, el champion local obtuvo recall de desabasto de **0.7854**, precisión de desabasto de **0.0425** y PR-AUC medio de clases minoritarias de **0.1970**. El recall quedó debajo del piso de 0.85; debe revisarse el equilibrio entre cobertura y carga operativa con etiquetas observadas.

**Tabla 3.3. Desempeño del champion original en dos periodos de evaluación.**

Fuente: métricas del run original en la tabla de reproducibilidad del README y evaluación local sobre el split `evaluation` de 2025 con `monitoring.ab_test`.

| Métrica | Test de diciembre 2024 | Evaluación octubre–diciembre 2025 |
|:---|---:|---:|
| Recall de desabasto | 0.8452 | 0.7854 |
| PR-AUC medio de clases minoritarias | 0.2462 | 0.1970 |
| Alertas por fecha de entrada | 765.6 | 802.6 |

Para 2025, el código divide el total de alertas entre **93 fechas distintas de entrada**: la primera hora objetivo de octubre utiliza una entrada del 30 de septiembre, una fecha parcial. El periodo de horas objetivo abarca 92 fechas de octubre–diciembre; la tasa de 802.6 sigue la convención del código y no representa alertas por cada una de esas 92 fechas objetivo. La comparación con diciembre de 2024 usa periodos de distinta duración y composición; no identifica por sí sola la causa de las diferencias.

### Estaciones y lectura para P2

Las tablas de estaciones de año completo contienen 358 estaciones en 2024 y 365 en 2025: 355 comunes, 10 presentes solo en el panel de 2025 y tres ausentes de este. Una ausencia del panel no prueba que la estación haya sido retirada. El constructor de la tabla continua del challenger conserva la intersección de estaciones de año completo en ambos años, por lo que excluye las 10 estaciones nuevas de ese panel de entrenamiento. Las estaciones nuevas deben evaluarse por separado; estos conteos no cuantifican su efecto en el desempeño ni prueban la causa de una diferencia.

El drift de entrada es una alerta complementaria y no sustituye la evaluación de desempeño cuando llega la etiqueta. La comparación con una versión alternativa y cualquier decisión de reentrenamiento requieren la evidencia y regla de la sección 4; las métricas anteriores corresponden únicamente al champion local.

## 4. A/B testing entre versiones

<!--
Dueño: Fernando (WP-3b). 30% de la rúbrica. Revisa Manuel.
-->

### 4.1 Diseño

| | Champion | Challenger |
|:---|:---|:---|
| Versión en el Registry | *pendiente* (`@champion`) | *pendiente* (`@challenger`) |
| Algoritmo e hiperparámetros | Los del champion | Los mismos que el champion |
| Ajuste | 1 ene – 30 sep 2024 | 1 ene 2024 – 31 jul 2025 |
| Calibración de umbrales | 1 oct – 30 nov 2024 | 1 ago – 30 sep 2025 |
| Evaluación | 1 oct – 31 dic 2025 | 1 oct – 31 dic 2025 |
| Run ID | *pendiente* | *pendiente* |

La única diferencia entre las dos versiones son los datos de entrenamiento, así que el A/B mide
si reentrenar con datos recientes mejora el modelo. Cada versión predice con sus propios
umbrales, porque son parte del modelo.

### 4.2 Regla de decisión

Esta regla se fijó el 9 de noviembre de 2026, antes de entrenar el challenger y de ver
cualquier resultado (decisión 5.6 del plan). El challenger reemplaza al champion solo si
cumple las tres condiciones sobre el conjunto de evaluación:

1. **Requisito operativo mínimo:** `recall_shortage ≥ 0.85`.
2. **Mejora real:** el intervalo de confianza del 95% de la diferencia (challenger − champion) en `pr_auc_minority_mean` queda por encima de 0. El intervalo se obtiene por bootstrap pareado por día (1 000 réplicas, semilla 42), remuestreando días completos porque las filas de un mismo día están correlacionadas.
3. **No inferioridad:** `macro_f1` no baja más de 0.01 y `alerts_per_day` no sube más de 20%.

Si alguna falla, el champion se queda.

### 4.3 Resultados

<!--
Tabla por versión: pr_auc_minority_mean, macro_f1, recall_shortage, recall_saturation, precision_shortage, alerts_per_day; diferencia con su intervalo del 95%; McNemar.
Desglose por mes y por estaciones nuevas vs. comunes. Datos: reports/monitoring/ab_summary.json y el run de MLflow con stage_of_work=ab_test.
-->

*Pendiente.*

### 4.4 Decisión

<!--
Aplicar la regla de 4.2 condición por condición y decir qué versión queda vigente.
Si gana el challenger, la evidencia de la promoción la agrega Manuel en la sección 6.
-->

*Pendiente.*

## 5. Reproducibilidad

<!--
Dueño: Manuel (WP-6). drift_summary.json generado en una segunda máquina desde main, comparado
con el del host (Oscar). Commit usado y resultado de la comparación.
-->

*Pendiente.*

## 6. Recomendación de Continuous Training

<!--
Dueño: Manuel (WP-5). 10% de la rúbrica.
Responder con la evidencia de las secciones 2 a 4:
- ¿Conviene reentrenar ahora? (sale del A/B)
- ¿Cada cuánto? (sale del drift y del desempeño por mes)
- ¿Con qué disparadores? Desempeño como principal (la etiqueta llega en una hora) y drift como
  alerta temprana. Umbrales concretos.
- Cómo se promueve una versión: la regla de 4.2 como validación automática, promoción con
  registry.py, make export-model y reconstrucción de la imagen de la entrega 4.
-->

*Pendiente.*

## 7. Conclusiones y limitaciones

<!--
Dueño: Manuel (WP-5). Limitaciones conocidas: capacidad estática del catálogo (problema 3),
lags del 1 de enero (problema 4), muestreo para Evidently, A/B offline con datos históricos
y no con tráfico real de la API.
-->

*Pendiente.*

## Cómo reproducir

<!--
Dueño: Manuel (WP-5). Confirmar los nombres finales de los targets con WP-D.
-->

```bash
make features-current   # tabla de features de P2 para 2025
make drift-report       # reporte de Evidently en reports/monitoring/
make train-challenger   # entrena y registra el challenger (host de evidencia)
make ab-test            # A/B entre @champion y @challenger
```

## Referencias

<!--
APA 7. Al menos Evidently (documentación) y los datos abiertos de MiBici.
-->

*Pendiente.*
