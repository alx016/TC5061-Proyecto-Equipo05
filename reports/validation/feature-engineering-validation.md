# Ingeniería de features: correcciones verificadas

**Fecha:** 17 de octubre de 2026 (America/Mexico_City).
**Base auditada:** `d7be27013acd850ae339560bbaf5f63fea2c54f2`.
**Estado histórico:** esta evidencia corresponde a la versión 2.
La ejecución actual de la versión 4 se documenta en
[Feature Engineering 2024](../feature_engineering_2024.md).

La validación inicial encontró seis fallos en tres causas raíz. Se corrigieron
manteniendo un único `features.py` y las recetas independientes de P1/P2/P3.
El contrato ahora usa `FEATURE_VERSION = "2"`. Los casos adversariales están
integrados en la suite normal, no ocultos mediante `xfail` ni umbrales relajados.

## Cambios y prueba de corrección

| Causa | Ajuste | Evidencia |
|---|---|---|
| Última etiqueta del split previo no disponible al primer scoring | `label_available_at` y `training_mask(keys, fit_as_of=...)`; las particiones por objetivo no cambian | Fronteras septiembre/octubre y noviembre/diciembre, latencia adicional y prueba con Ridge |
| Texto `"NaT"` se perdía después de parsear | Comprobación de nulos después de `pd.to_datetime`, antes de agregar | Rechazo explícito en ambos timestamps |
| Restas unsigned invertían flujo y asimetría | Normalización a `int64` con rango conservador para sumas de ambas direcciones | Desabasto −0,4 y asimetría −0,6 correctos; rechazo de conteos fuera de rango |

Una etiqueta de `[23:00,00:00)` no está completa al scoring de las 23:50. El
consumidor debe ajustar el modelo usando `training_mask` con un corte no posterior
al primer scoring y reservar tiempo de cómputo para el fit. Ajustar solo sobre
`split == train` sigue siendo incorrecto: el helper es explícito, no un interceptor
que pueda proteger un entrenamiento que lo omita. El ejemplo de uso fue corregido.

La prueba con Ridge perturba la última etiqueta aún no publicada: la primera
predicción permanece igual con la máscara y cambia con el ajuste original sin
purga. Ese control positivo demuestra que la prueba detecta el patrón de fuga.
`training_mask` permite retrasar disponibilidad por latencia, nunca anticiparla
al cierre del target, y no permite incluir etiquetas test en el entrenamiento.

## Resultados

- **48 pruebas aprobadas en Python 3.13.11**: 35 originales, seis casos migrados y
  siete casos adicionales de cortes, rango y predicción real.
- Ruff check/format y `make lint`: correctos.
- CLI sintética de los tres casos: Parquet y recibos deterministas; los keys
  supervisados incorporan disponibilidad de etiqueta y el manifiesto indica versión 2.
- Pipeline Ridge: fit con train elegible, serialización y paridad de predicción.
- Sin instalación ni cambios de dependencias del repositorio. Se usó un entorno
  temporal mínimo Python 3.13 con las versiones directas fijadas. La instalación
  completa desde cero mediante `uv sync --frozen` no forma parte de esta evidencia.

## Ejecución de escala sintética

360 estaciones × 8.784 horas = **3.162.240 filas**. Cada receta se ejecutó en un
proceso independiente; los valores son observaciones locales, no SLOs ni tiempos
sobre viajes reales. No incluyen carga de Parquet ni agregación de viajes crudos.
RSS es el máximo del proceso completo, no memoria exclusiva del builder.

| Receta | Salida | Segundos de receta | RSS máximo (MiB) |
|---|---:|---:|---:|
| P1 | 3.162.240 × 14 | 1,516 | 817,6 |
| P2 | 3.162.240 × 19 | 1,931 | 1.230,9 |
| P3 | 360 × 49 | 0,414 | 725,8 |

El [harness](feature_engineering_runtime.py) permite repetir estas comprobaciones. Sus JSON,
logs y descripción del entorno son evidencia local regenerable, excluida por
`.gitignore`; el script y este informe sí pertenecen a la entrega.

## Reproducir

Desde la raíz, con Python 3.13 y las dependencias declaradas:

```bash
python -m pytest tests -q
make lint
python -m ruff check tests/test_features.py reports/validation/feature_engineering_runtime.py
python -m ruff format --check tests/test_features.py reports/validation/feature_engineering_runtime.py
python -m reports.validation.feature_engineering_runtime p1
python -m reports.validation.feature_engineering_runtime p2
python -m reports.validation.feature_engineering_runtime p3
```

## Límites y siguiente gate

El clon contiene **0 de los 12 Parquets 2024 requeridos** y ningún catálogo local
que acredite capacidades/red históricas. No se certifican completitud real,
actividad de estaciones, censura de llegadas desde 2023 ni disponibilidad online.
El manifiesto versionado no reemplaza esos archivos ni su auditoría de calidad.
La corrección del código no se presenta como aprobación de ese replay ausente.

Unidad de trabajo: fixes + regresiones + contratos + harness/informe e ignores de
salidas locales. Rollback: retirar esta corrección como unidad restaura versión 1
y sus defectos; no afecta ingesta ni dependencias. Las exclusiones de contexto de
agentes son locales en `.git/info/exclude`, no parte del rollback compartido.
Preparación: archivos seleccionados para commit, sin crear commit ni hacer push.
