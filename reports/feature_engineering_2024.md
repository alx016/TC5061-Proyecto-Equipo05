# Feature Engineering 2024: ejecución y decisiones

## Resultado y alcance

Se ejecutaron P1 y P3 con viajes reales de 2024 y diciembre de 2023 como historia.
Los artefactos locales están en `data/processed/fe_2024_v4/`, excluidos de Git.
P2 tiene receta y pruebas, **no un dataset etiquetado válido**: falta capacidad
histórica. El ML Canvas define la presión de flujo/capacidad, pero ni él ni el
EDA acreditan capacidades de 2024. El catálogo de 2026 no se retrofecha.
No se usa ningún calendario externo, archivos del DOF ni indicadores de días
especiales. Hora, día de semana y mes se calculan desde los timestamps.

## Evidencia de partida

Fuente principal: `notebooks/1.0-fge-eda-mibici.ipynb`, contrastada con los doce
Parquets mensuales y `references/viajes_manifest.json`. La auditoría de 2024 dio:

| Comprobación | Resultado |
|---|---:|
| Viajes / días con salidas | 4,645,921 / 366 |
| Estaciones origen distintas / destino distintas | 370 / 370 |
| Nulos en los cinco campos requeridos / IDs duplicados | 0 / 0 |
| Duraciones negativas / iguales a cero | 0 / 20,850 |
| Viajes mayores de 180 minutos | 537 |

Las cifras narrativas antiguas (~3.825 millones y proporciones P2) no son una
medición de esta ejecución. Las salidas ejecutadas del EDA y esta auditoría tienen
precedencia; no se modifican retrospectivamente los notebooks.

## Variables originales frente a representación

Solo se consumen ID de viaje, inicio, fin, estación origen y estación destino.
Edad, género e identificadores personales no son predictores ni claves exportadas.

| Origen | Transformación | Motivo técnico y de negocio |
|---|---|---|
| Inicio/fin, estación | Conteos horarios separados de salidas/llegadas | Demanda observada y presión para redistribución; cada evento en su propio reloj |
| Hora y día | Seno/coseno horario/semanal, ordinales, interacción fin de semana, ciclo mensual | Continuidad cíclica y diferencias de movilidad laboral/recreativa |
| Historia de conteos P1 | Lags 2, 3, 24, 168; medias/desviaciones de 3/6 horas tras shift(2) | Persistencia, rutina diaria/semanal y volatilidad; información disponible a h−10 min |
| Historia numérica | log1p de conteos; signed-log de ratios en P2 | Reducir asimetría para modelos lineales, conservar valores originales y signos |
| Estación P1/P2 | One-hot con categorías desconocidas ignoradas | Efectos locales sin distancia ordinal ficticia |
| Flujo/capacidad P2 | Ratios históricos y presión objetivo, umbral ±0.4 | Proxy de desabasto/saturación, no inventario observado; bloqueado sin capacidad histórica |
| Metadatos acreditados, opcionales | Capacidad, coordenadas, municipio, vecinos a 500m y presión lag2 de tres vecinos próximos | Contexto territorial y sustitución entre estaciones; no se activaron sin fuente histórica |
| Horarios P3 | 24 proporciones de salidas + 24 de llegadas + asimetría | Comparar función urbana y forma de uso, no solo volumen |

No se incorpora clasificación de festivos, días especiales o días hábiles regulares.
La receta v4 y sus métricas se generan sin ese supuesto ni ese archivo.

## Calidad, selección y causalidad

- Los buckets faltantes se completan a cero bajo cobertura mensual completa. La
  primera actividad observada limita elegibilidad; no demuestra fecha de apertura,
  cierre, fallas de telemetría ni demanda insatisfecha. No hay bfill.
- No se eliminan salidas por duración cero o larga: su resultado futuro no debe
  decidir si una salida ya ocurrida cuenta como demanda. Se auditan anomalías;
  no se recortan picos válidos ni el target.
- Mediana, indicadores de ausencia, RobustScaler y eliminación de varianza cero
  se ajustan solo en train. P3 conserva escala de proporciones. La selección es
  conservadora, no una afirmación de que todas las variables sean predictivas.
- Train enero–septiembre; validación octubre–noviembre; test diciembre. P1 purga
  calentamiento de 168h y etiquetas no disponibles al corte de fit 30-sep 23:40;
  validación se purga al 30-nov 23:40. Historia se actualiza hora a hora, no es un
  pronóstico multimes emitido de una vez. Latencia real de publicación no observada.
- P3 construye perfiles independientes por ventana y exige ≥100 salidas por mes
  y actividad en ambas direcciones. Esto excluye estaciones pequeñas; reportar
  cobertura, no extrapolar conclusiones al conjunto de la ciudad.
- Sin atributos sensibles no se certifica equidad: estación/geografía pueden ser
  proxies; evaluar errores por estación/volumen y cobertura de estaciones nuevas.

## Dataset entregado

| Caso | X original | Columnas transformadas | Train | Validación | Test |
|---|---|---:|---:|---:|---:|
| P1 | 3,174,744 filas × 28 columnas | 389 | 2,371,776 | 529,606 | 270,792 |
| P3 | 350 perfiles train × 49 columnas | 46 | 350 | 359 | 365 |
| P2 | No generado: capacidad histórica pendiente | — | — | — | — |

Cada carpeta P1/P3 contiene `X.parquet`, `keys.parquet`, `y.parquet` cuando aplica,
`preprocessor.pkl`, `features_manifest.json` y `ready/{train,validation,test}/`.
Ready contiene lotes CSR float32 `.npz`, claves y etiquetas alineadas; el manifiesto
registra columnas en orden, rangos de lotes, SHA-256 de entradas/matrices, cortes,
calidad de train y métricas. En P3 el X raíz corresponde solo a train.
Se verificaron carga del preprocesador en otro proceso, dimensiones de todos los
lotes y transformación finita. Cargar pickle únicamente de fuentes confiables.

## Comparación controlada sin consultar test

P1: Ridge fijo (alpha=1), mismo conjunto elegible, muestreo determinista de
49,412 filas train y 19,616 de validación; pipelines independientes ajustados
solo en train. Predicciones negativas se truncan a cero; targets sin modificar.

| Representación | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Hora/día de semana + estación + lag24 | 1.31045 | 2.31999 | 84.87% |
| Receta ampliada | 1.12436 | 1.88153 | 72.82% |

Reducción de MAE aproximada de 14.2%, **no** cumplimiento del objetivo de negocio
ni evidencia de cada feature aislada. WAPE sigue alto frente a la aspiración del
Canvas; faltan validación temporal repetida, tuning y evaluación por segmentos.

P3: KMeans fijo K=4, seed=42, 10 inicializaciones; modelos ajustados en train y
asignaciones de validación congeladas. Ambos se evalúan en el mismo espacio de
49 dimensiones de perfiles sobre estaciones elegibles comunes.

| Representación para agrupar | Grupos en validación | Silhouette | Davies–Bouldin |
|---|---:|---:|---:|
| Volumen salidas/llegadas | 3 | −0.18395 | 2.54456 |
| Perfil horario | 4 | 0.11646 | 1.88465 |

Mejora relativa, cohesión todavía modesta. No se seleccionó K mirando test.
Analizar estabilidad e interpretación de centroides antes de decisiones operativas.

## Reproducción y siguiente etapa

1. Obtener CSV oficiales 2024 y diciembre 2023 con los helpers de `dataset.py`;
   ejecutar `python -m tc5061_proyecto_equipo5.dataset clean`.
2. Ejecutar los comandos de la guía de feature engineering con un directorio de
   salida nuevo. La publicación es atómica y rechaza sobrescribir entregas.
3. Usar ready directamente con su transformador congelado; no volver a escalar.
   Para otro candidato, CV o refit, partir del X crudo y ajustar un pipeline nuevo
   en el train correspondiente, nunca reutilizar un scaler ajustado con otros folds.
4. Conseguir capacidad histórica antes de P2. Entrenamiento/scoring productivo,
   integración MLflow y disponibilidad online siguen fuera de esta entrega.

## Verificación de implementación

- 61 pruebas completas; Ruff y mypy (`--ignore-missing-imports`, sin stubs de
  pandas: comprobación limitada de tipos externos).
- Revisión Standards: cero incumplimientos documentados; recomendación no
  bloqueante de centralizar los cortes temporales repetidos dentro de `features.py`.
- Revisión Spec: se corrigieron dos defectos opcionales con pruebas RED/GREEN:
  vecinos definidos por actividad futura y vigencia omitida en catálogos P2 sin
  coordenadas. La segunda revisión no encontró bloqueantes pendientes.
- Con catálogo histórico, el universo vecinal proviene de ese snapshot acreditado,
  no de futuras primeras observaciones. Disponibilidad y vigencia se validan por
  separado, también para capacidad sin geografía. Estas correcciones no cambian
  los datasets reales P1/P3 entregados, que no utilizan metadatos de estaciones.

## Ajuste de alcance: receta sin calendario externo

P1/P3 se regeneraron con `FEATURE_VERSION=4`. Las cifras anteriores ya
corresponden a esa ejecución, no a la receta descartada. P1 usa 28 columnas
crudas y 389 transformadas; sus manifiestos solo incluyen las 13 fuentes de viajes.
No se proporciona ni consume un archivo adicional de días especiales. Los
artefactos locales v3 están reemplazados y no deben usarse para este análisis.
