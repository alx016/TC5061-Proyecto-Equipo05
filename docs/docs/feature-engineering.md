# Features independientes, una base temporal compartida

`tc5061_proyecto_equipo5/features.py` contiene las recetas de P1, P2 y P3, no una
matriz universal. Comparte validación y agregación; cada estimador puede elegir
lags, ventanas, columnas y preprocesamiento. No incorpora un feature store ni
una jerarquía de plugins para tres casos locales. La [investigación con fuentes
primarias](feature-engineering-research.md) sustenta esta decisión, **no** la
existencia de un layout universal de proyectos «élite».

Para implementar el entrenamiento y scoring, consultar el
[plan de desarrollo de modelos](#plan-para-desarrollar-modelos-con-estas-features).

## Estado de la ejecución v4

P1 y P3 ya se generaron con los viajes reales de 2024. P2 tiene implementación
probada, pero su dataset sigue bloqueado por falta de capacidad histórica.
Consultar [auditoría, decisiones y métricas](../../reports/feature_engineering_2024.md).
Las variables temporales se derivan exclusivamente de las fechas de los viajes;
no se usa un calendario externo ni indicadores de festivos o días especiales.

## Contratos y salidas

| Interfaz | Resultado / regla |
|---|---|
| `build_hourly_panel(trips, station_ids, start=..., end=...)` | Grilla densa estación-hora; salidas por `inicio_viaje`, llegadas por `fin_viaje`; intervalo `[start,end)` |
| `build_demand_features(panel, lags=..., windows=...)` | P1: calendario cíclico, lags de salidas y media/desviación móvil; `y=demand` separado |
| `build_risk_features(panel, stations, catalog_known_at=..., lags=..., tau=.4)` | P2: historia de entradas/salidas/flujo, capacidad y ratios históricos; `y=risk` separado |
| `build_station_profiles(panel, start=..., end=..., min_monthly_trips=100)` | P3: 24 proporciones de salidas + 24 de llegadas + asimetría, por estación; sin target |
| `training_mask(keys, fit_as_of=..., splits=("train",))` | Filas cuyas etiquetas ya están disponibles al ajustar; nunca selecciona test |
| `make_preprocessor(X, scale_numeric=...)` | Transformer **sin ajustar**; imputación mediana, escalado opcional y one-hot de estación |

`FeatureSet` mantiene `keys`, `X` e `y` alineados por posición. Los identificadores,
instantes, cortes y targets nunca entran por accidente a la matriz predictora.
`station_id` sí entra a P1/P2 como categoría explícita, no como distancia numérica.
Las recetas no mutan entradas ni ajustan estadísticas globales.

### Tiempo y fuga: ejemplo verificable

Una fila con `target_time=2024-10-01 11:00` predice **[11:00,12:00)** y se emite
**10:50**. La última hora cerrada es **[09:00,10:00)**: `lag_2`, no `lag_1`.
Los lags se miden desde el inicio del objetivo. Por defecto:

- P1: lags 2, 3, 24, 168 horas; rolling de 3 y 6 horas después de `shift(2)`.
- P2: lags 2, 3, 24 de cada flujo; la etiqueta usa solo el flujo de la hora objetivo.
  Desabasto si ratio ≤ −0.40, saturación si ≥ +0.40. El ratio puede exceder ±1:
  representa presión, **no inventario real**.
- P3: perfil disponible al terminar `[2024-01-01,2024-10-01)`. Se excluyen estaciones
  con menos de 100 salidas en cualquier mes o sin actividad en alguna dirección.
  Cada vector direccional suma 1. No se usa diciembre para caracterizar enero.

Las particiones de P1/P2 siguen la **hora objetivo**: enero–septiembre / octubre–
noviembre / diciembre. Se permite historia anterior al corte entre particiones:
esto simula **pronósticos horarios sucesivos**, no un pronóstico de todo diciembre
emitido el 1 de diciembre. Para este último habría que reconstruir las features
congelando la historia en el origen del pronóstico.

**La partición no basta para entrenar sin fuga.** `keys.label_available_at` indica
el cierre del bucket objetivo (más la latencia real si el consumidor la incorpora).
La última etiqueta de septiembre cierra el 1 de octubre a las 00:00, después del
primer scoring de validación, emitido a las 23:50. Usar `training_mask` con un
`fit_as_of` no posterior a ese scoring y reservar tiempo para ajustar el modelo.
La máscara purga etiquetas no disponibles sin cambiar su partición. Para reajustar
antes de test puede usar `splits=("train", "validation")`, manteniendo el corte del
primer scoring de test; nunca incluir etiquetas test. El mismo requisito afecta
a las métricas de validación usadas para seleccionar el modelo antes de test.

Los NaN de calentamiento se conservan: no usar backfill. El consumidor puede
excluir filas sin historia antes de entrenar, o imputarlas con estadísticas de train.
La evaluación debe documentar la misma política y la ventana efectiva resultante.

## Receta distinta por estimador

```python
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from tc5061_proyecto_equipo5.features import (
    build_demand_features, make_preprocessor, training_mask,
)

# Configuración concreta de Ridge, no compartida obligatoriamente con un árbol.
linear = build_demand_features(panel, lags=(2, 3, 24, 168), windows=(3, 6))
val = linear.keys["split"].eq("validation")
first_issue = linear.keys.loc[val, "issue_time"].min()
fit_as_of = first_issue - pd.Timedelta(minutes=10)  # presupuesto ilustrativo de fit
train = training_mask(linear.keys, fit_as_of=fit_as_of)
model = make_pipeline(make_preprocessor(linear.X, scale_numeric=True), Ridge(solver="lsqr"))
model.fit(linear.X.loc[train], linear.y.loc[train])
predictions = model.predict(linear.X.loc[val])  # nunca fit sobre validación/test

# Otro estimador puede pedir otra representación, sin duplicar la agregación.
tree = build_demand_features(panel, lags=(2, 24, 168), windows=())
tree_preprocessing = make_preprocessor(tree.X, scale_numeric=False)
```

Este ejemplo ilustra composición, no entrena modelos al generar features.
El pipeline ajustado completo y su configuración deberán persistirse juntos en
el experimento de la siguiente etapa de modelado. Una codificación adicional (one-hot horario,
splines, etc.) debe añadirse como receta/preprocesador explícito cuando exista
un experimento que la requiera, no como nueva copia de `features.py`.

## Plan para desarrollar modelos con estas features

**Estado:** esta sección define el trabajo de modelado pendiente; no afirma que
esté implementado. `features.py` ya ofrece los builders y `training_mask`, pero
`modeling/train.py` y `modeling/predict.py` siguen siendo scaffolds. La CLI permite
preparar datasets y comparar representaciones con estimadores fijos; todavía no
selecciona modelos finales ni registra experimentos. No ejecutar esos scaffolds
como si entrenaran modelos.

### 1. Responsabilidades y orden de implementación

| Archivo | Responsabilidad que se implementará |
|---|---|
| `features.py` | Único módulo de feature engineering: agregados compartidos, recetas por problema, preprocesadores y futura extracción causal para scoring |
| `modeling/train.py` | Configuración explícita de candidatos, selección de receta/columnas, cortes, construcción de pipelines nuevos, fit, evaluación y registro |
| `modeling/predict.py` | Cargar el bundle seleccionado, validar su contrato, reconstruir las mismas features con historia disponible y predecir sin ajustar nada |
| `tests/` | Contratos de recetas, elegibilidad temporal, comparación justa y paridad entrenamiento/inferencia |

El recorrido será: **auditar fuentes → construir panel una vez → generar receta
por candidato → seleccionar filas/columnas → ajustar pipeline → validar → elegir
configuración → refit opcional → test final → exportar bundle**. No copiar fórmulas
de lags, ratios o perfiles a los scripts de modelado ni a notebooks. Una receta
idéntica puede reutilizar su resultado entre estimadores; eso no obliga a que
representaciones distintas compartan una matriz final universal.

### 2. Configuración por candidato, no por archivo

Se empezará con un catálogo explícito de configuraciones en `train.py`, sin una
jerarquía de plugins ni archivos `features_<modelo>.py`. Cada candidato declarará:
`model_id`, problema, builder, versión de receta, argumentos, columnas ordenadas,
preprocesamiento, hiperparámetros, semilla y política de calentamiento. La selección
de builder se hará mediante nombres permitidos, no con `eval` ni imports arbitrarios.

Ejemplo de configuración a implementar; las columnas son ilustrativas, no una
selección validada empíricamente:

```json
{
  "model_id": "p1_ridge_v1",
  "problem": "p1",
  "feature_version": "2",
  "recipe_version": "1",
  "builder": "build_demand_features",
  "lags": [2, 3, 24, 168],
  "windows": [3, 6],
  "columns": [
    "station_id", "hour_sin", "hour_cos",
    "departures_lag_2", "departures_lag_24", "departures_mean_6"
  ],
  "preprocessing": {"scale_numeric": true},
  "warmup_policy": "common_complete_history",
  "estimator": "Ridge",
  "hyperparameters": {"alpha": 1.0, "solver": "lsqr"},
  "seed": 42
}
```

`FEATURE_VERSION` identifica el contrato compartido; `recipe_version` y los
parámetros identifican la representación concreta. Cambiar columnas, lags o
escalado exige otra identidad de configuración, aunque no cambie el código del
builder. La identidad se obtendrá del JSON canónico completo, no solo del nombre.
Guardar también los defaults efectivos: reconstruir un run no debe depender de
los defaults de una versión futura del código.

### 3. Recetas iniciales por familia

Son puntos de partida para comparar, no ganadores ni hiperparámetros óptimos.
Las variables espaciales siguen pendientes de fuentes históricas válidas.
Los festivos y días especiales están fuera del alcance de esta entrega.

| Candidato | Representación | Transformaciones aprendidas |
|---|---|---|
| P1 · baseline estacional | `departures_lag_24`; alternativamente, promedio estación/día/hora ajustado solo en train | Ninguna para persistencia; el promedio histórico pertenece al estimador, no a una agregación sobre todo el año |
| P1 · Ridge / ElasticNet | Calendario cíclico, lags y rolling seleccionados explícitamente | Imputación, escala numérica y one-hot de estación, ajustados dentro del pipeline |
| P1 · árbol, etapa posterior | Receta propia de demanda; puede omitir rolling o elegir otros lags | Sin escalado numérico; encoding y tratamiento de ausencias compatibles con el estimador elegido |
| P2 · persistencia | Discretizar `net_ratio_lag_2` con el mismo `tau` del target | Ninguna; no usar el estado de la hora incompleta |
| P2 · logística | Calendario, capacidad e historia de entradas/salidas/ratios | Imputación, escalado y one-hot; pesos de clase calculados con etiquetas de entrenamiento elegibles |
| P2 · árbol, etapa posterior | Subconjunto propio de la receta de riesgo | Sin imponer el preprocesamiento de la logística; mismo target y mismo universo de evaluación |
| P3 · K-Means | 49 dimensiones: perfiles direccionales y asimetría, sin ID de estación | Baseline sin escalado adicional; una variante escalada ajustará su scaler únicamente sobre perfiles train |

`make_preprocessor` sirve como composición inicial para lineales y variantes
compatibles, no como obligación para todos los algoritmos. Nuevas codificaciones
(one-hot horario, splines, etc.) se añadirán en el mismo `features.py` cuando un
experimento las requiera. Los árboles avanzados no se consideran instalados ni
implementados; dependencias adicionales se decidirán en esa etapa.

### 4. Entrenamiento y selección temporal de P1/P2

1. **Congelar protocolo y datos.** Verificar hashes y procedencia del catálogo;
   fijar `tau=0.40`, horizonte, emisión, latencia asumida y semilla antes de comparar.
2. **Comparar las mismas observaciones.** Alinear candidatos por estación/target,
   nunca solo por índice. Para el baseline de comparación se excluirán las filas
   de calentamiento sin historia completa para cualquiera de los candidatos,
   usando solo sus columnas históricas seleccionadas. Registrar la máscara común
   y exclusiones por split; no quitar filas porque sus targets o errores incomoden.
   Un experimento de imputación del calentamiento será otro protocolo declarado.
3. **Ajustar para validación.** Construir un pipeline nuevo por candidato. Usar
   `training_mask(..., fit_as_of=...)` antes de todo fit; el corte debe preceder el
   primer scoring de octubre y dejar presupuesto para entrenar. Ajustar imputador,
   scaler, encoder, selección aprendida y estimador solo con esas filas train.
4. **Elegir sin mirar test.** Comparar en octubre–noviembre, pero para una decisión
   previa al primer scoring de diciembre usar solo etiquetas disponibles al corte
   de selección. `training_mask(keys, fit_as_of=selection_as_of,
   splits=("validation",))` permite construir esa máscara: no implica entrenar en
   validación. La última hora de noviembre todavía no está cerrada a las 23:50.
5. **Refit opcional.** Con receta e hiperparámetros elegidos, crear otro pipeline
   nuevo y usar `splits=("train", "validation")` con el corte previo al primer
   scoring de diciembre. Reservar tiempo para selección y fit; registrar ambos
   instantes y si hubo refit. No reutilizar transformadores ya ajustados al cambiar
   el conjunto de entrenamiento.
6. **Evaluar una sola vez.** Congelar el bundle y medir diciembre cuando sus
   etiquetas estén disponibles. No reajustar receta, umbral o modelo a partir de
   ese resultado. La historia observada puede avanzar cada hora; el modelo queda
   congelado durante este test salvo un protocolo de reentrenamiento separado.

Si se añade validación temporal con múltiples folds, dividir por **horas únicas**
y aplicar el corte simultáneamente a todas las estaciones. El panel está ordenado
por estación: dividir directamente sus filas puede mezclar pasado y futuro entre
estaciones. Repetir máscara de disponibilidad y pipeline nuevo en cada fold; no
usar partición aleatoria ni un scaler ajustado una vez sobre todos los folds.

Para P1 se reportarán WAPE, MAE y RMSE; si la suma real es cero, WAPE será no
definido y llevará motivo, no una división corregida con un epsilon oculto.
La política de no negatividad de la predicción se fijará y exportará como parte
del bundle, aplicándose antes de métricas tanto offline como en scoring.
Para P2 se reportarán Macro F1, recall de desabasto y PR-AUC multiclase con clases
y regla de agregación explícitas; una métrica no definida por ausencia de clases
se señalará, no se inventará. Mapear probabilidades según las clases del estimador,
no suponiendo un orden fijo de columnas. El target sigue siendo presión proxy.

### 5. Protocolo separado para P3

P3 no tiene etiquetas horarias ni usa `training_mask`. Sus instantes de
disponibilidad son los cierres de sus ventanas de observación:

- **Train:** perfiles enero–septiembre, disponibles el 1 de octubre a las 00:00.
  Ajustar ahí el eventual scaler y el clustering, sin incluir octubre–diciembre.
- **Validación:** perfiles octubre–noviembre, disponibles el 1 de diciembre a las
  00:00. Comparar configuraciones y K (rango inicial 3–6), usando transformaciones
  train congeladas y reportando cobertura de estaciones y métricas definidas.
- **Test:** perfiles de diciembre, disponibles el 1 de enero de 2025. Congelar la
  configuración tras seleccionar con validación y antes de consultar los perfiles
  de diciembre; no elegir K con ese mes.

El primer scoring de diciembre de P1/P2 ocurre antes del cierre de noviembre:
**no reutilizar ese reloj para P3**. Su evaluación es un estudio posterior al
cierre del periodo, no un pronóstico horario. No redefinir la cohorte train
mirando estaciones que sobreviven en validación/test; registrar por ventana las
estaciones elegibles, las excluidas por baja actividad y la cobertura resultante.

Para K-Means, asignar perfiles nuevos con el modelo train congelado; no ejecutar
`fit_predict` sobre cada split y presentar esos clusters como predicciones del
mismo modelo. Silhouette y Davies–Bouldin solo se reportarán cuando el número de
muestras/grupos permita definirlos, junto con estabilidad e interpretación urbana.
Alternativas transductivas, como Ward, necesitarán otro protocolo explícito: no
suponer que ofrecen el mismo scoring inductivo ni comparar sus etiquetas sin
considerar permutaciones. Los perfiles anuales serán un análisis descriptivo
adicional, nunca un predictor retroactivo ni sustituto del test temporal.

### 6. Registro y bundle reproducible

El siguiente módulo registrará cada candidato y su evaluación en el experimento
`mibici-equipo05`. Esta integración con MLflow está pendiente; no hay runs reales
creados por generar features. El contrato mínimo del run/bundle será:

| Elemento | Contenido obligatorio |
|---|---|
| Linaje | Commit, hash de datos verificado, hashes de entradas, catálogo y su procedencia histórica |
| Receta | Configuración efectiva completa, `FEATURE_VERSION`, versión de receta e identidad de configuración |
| Esquema | Columnas en orden, tipos, granularidad, IDs y significado del target/clases |
| Protocolo | Ventanas, cortes de fit/selección, disponibilidad, calentamiento, población/máscara, modo rolling y política de refit |
| Modelo | Pipeline completo ajustado, semilla efectiva, hiperparámetros, versiones de librerías y posprocesamiento |
| Resultados | Métricas con prefijos `train_`, `val_`, `test_`, tamaño de muestra, cobertura y motivos de métricas no definidas |
| Evidencia | P1: serie real/predicha; P2: curva PR y matriz de confusión; P3: perfiles/centroides y mapa de clusters |

La selección de columnas debe viajar con el modelo, dentro del pipeline o en la
receta exportada y validada por el consumidor. No exportar solo los coeficientes
ni reconstruir columnas desde un orden casual del DataFrame. Guardar artefactos
en almacenamiento excluido de Git, como `mlartifacts/`; conservar en Git código,
configuraciones y evidencia revisable sin datos personales. Los bundles serializados
solo se cargarán desde fuentes confiables. La evidencia de interfaz de MLflow
seguirá el requerimiento de cuatro capturas de la entrega.

### 7. Scoring y criterios para cerrar el desarrollo

`predict.py` cargará el bundle y comprobará esquema/versiones antes de reconstruir
su receta. No hará `fit`, no elegirá nuevos hiperparámetros ni cambiará la receta
por defaults actuales. Un contrato incompatible deberá fallar con diagnóstico,
no completar features ausentes con ceros para poder producir una predicción.

**Interfaz pendiente:** los builders actuales son offline y producen también `y`
a partir de un panel con la hora objetivo. Para scoring se añadirá en el mismo
`features.py` una interfaz de predictores que reciba historia disponible y
solicitudes estación/target, devuelva `X`/keys y reutilice los mismos helpers.
No requerirá conteos futuros, destinos aún no observados ni etiquetas simuladas.
La historia deberá cubrir el mayor lag/ventana requerido y acreditar disponibilidad;
una hora ausente de la fuente no equivale a demanda cero.

Checklist de aceptación del futuro desarrollo:

- [ ] Cada candidato declara y exporta su propia receta, selección y pipeline.
- [ ] Cambiar una etiqueta no disponible no altera el modelo/predicción elegible.
- [ ] Los folds y el refit respetan cortes globales, latencia y presupuesto de fit.
- [ ] Offline y scoring obtienen el mismo `X` para la misma historia y solicitud.
- [ ] El bundle conserva predicciones tras serializar, cargar y validar columnas.
- [ ] P1/P2 comparten filas de comparación; P3 documenta sus cohortes por ventana.
- [ ] Métricas/artefactos y linaje quedan registrados sin usar test para selección.
- [ ] Se ejecuta replay con viajes y catálogo históricos acreditados, no solo fixtures.

P2 no se entrenará mientras falte capacidad histórica; P1/P3 pueden continuar
con los supuestos de cobertura documentados. Este plan amplía la guía de uso,
no modifica la evidencia ya obtenida
ni convierte los scaffolds actuales en un sistema de modelado terminado.

## Generar datasets preparados

La ruta nueva `--prepare-ready` necesita doce Parquets de 2024 y permite diciembre
2023 como contexto. P1/P3 no requieren catálogo: usan la red observada y limitan
las filas por primera observación e historia completa, no por fecha de apertura.

```bash
python -m tc5061_proyecto_equipo5.features main --problem p1 --prepare-ready \
  --input-dir data/interim/viajes \
  --output-dir data/processed/fe_2024_v4
python -m tc5061_proyecto_equipo5.features main --problem p3 --prepare-ready \
  --input-dir data/interim/viajes \
  --output-dir data/processed/fe_2024_v4
```

Los artefactos de clasificación (P2) tienen sus propios comandos: `features panel`
construye el panel estación-hora compartido y `features classification` construye la
tabla que consumen `modeling/train_classification.py` y `plots.py`. Como el portal
publica únicamente el catálogo de estaciones vigente, esa ruta **asume capacidad
estática** entre 2024 y el snapshot del catálogo; el supuesto queda registrado y debe
reportarse junto con los resultados.

Usar otro directorio para repetir: se rechaza sobrescribir una entrega publicada.
La receta ampliada agrega interacciones, variables temporales y logaritmos. El
preprocesador ready usa mediana/indicadores, one-hot, escalado robusto (P1/P2) y
selección de varianza entrenada solo en train. P3 no añade escalado. Cada caso
exporta X crudo, claves, etiquetas cuando corresponda, preprocesador serializado,
manifiesto y lotes CSR float32 separados por split. Los datos permanecen fuera de
Git. `FEATURE_VERSION=4`; `selected_columns` fija el orden de las matrices.

**No hacer fit de nuevo sobre ready.** Para cambiar estimador/receta/fold, utilizar
X crudo y un preprocesador nuevo; `make_preprocessor` ofrece `numeric_scaling`
(`standard`, `robust`, `none`), `missing_indicators` y `select_variance`.

P2 exige `--stations-path` y `--catalog-known-at`, con capacidad históricamente
válida. Coordenadas/municipio y vecinos son opcionales y requieren igual respaldo.
`catalog-known-at` es declaración de procedencia, no certificación automática.
No basta renombrar el catálogo de 2026 ni inventarle una fecha antigua.
La CLI anterior sin `--prepare-ready` conserva su contrato baseline con catálogo;
las garantías de publicación atómica y matrices listas corresponden al modo nuevo.

## Garantías y límites explícitos

- Grilla completa, duplicados, fechas (también nulos después del parseo), conteos
  y capacidades inválidas se validan; los conteos se normalizan a `int64` con un
  límite conservador que evita overflow de las agregaciones de ambas direcciones;
  un ID desconocido genera error, no se elimina silenciosamente.
- Completar con cero presupone **fuentes completas y estación operativa**.
  Doce archivos presentes no certifican ausencia de huecos dentro de cada archivo.
  Falta disponibilidad histórica de estaciones: no se infiere apertura/cierre
  mirando su actividad futura.
- Los timestamps son hora local Guadalajara sin zona, acorde con la ingesta.
  Los timestamps con zona se rechazan: normalizarlos explícitamente antes de entrar.
- El CSV mensual solo permite replay offline. Se supone disponibilidad inmediata
  al cierre de cada hora; no demuestra latencia real, ausencia de correcciones
  tardías ni un servicio online. Una garantía productiva exige `available_at`,
  snapshots de metadatos y pruebas de paridad del pipeline desplegado.
- El modo preparado incorpora diciembre de 2023 si está presente y descarta
  calentamiento; no prueba completitud de publicación. El modo legacy solo lee 2024.
- No se incorpora ningún calendario externo. Capacidad/vecinos no se
  materializan sin metadatos históricos.
- No se instalan Feast, TFX, LightGBM ni dependencias nuevas para esta entrega.

## Verificación

```bash
python -m pytest tests -q
python -m ruff check tc5061_proyecto_equipo5/features.py
```

La suite incorpora los seis casos adversariales y comprueba la invariancia de la
primera predicción al cambiar una etiqueta aún no publicada. También perturba
futuro y hora incompleta, comprueba lags por estación,
normalización y filtrado de P3, separación de labels, cortes temporales,
catálogo futuro, preprocesado ajustado solo en train y los tres caminos CLI
con datos sintéticos. No equivale a validación de disponibilidad productiva.
