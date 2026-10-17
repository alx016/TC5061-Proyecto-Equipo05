# Feature engineering por caso de uso: evidencia y decisión de diseño

Consulta: 17 de octubre de 2026. Alcance: P1 demanda, P2 riesgo proxy y P3 perfiles
de estaciones de MiBici. Esta nota distingue evidencia externa de recomendaciones
para este repositorio; no presenta una estructura de archivos como estándar universal.

## Qué respaldan las fuentes primarias

1. **La representación puede depender del algoritmo.** El ejemplo oficial de
   demanda de bicicletas de scikit-learn compara árboles con regresión lineal y
   distintas codificaciones temporales. Los árboles modelan relaciones no monótonas
   con variables ordinales; los modelos lineales necesitan representaciones más
   expresivas. Seno/coseno aporta continuidad periódica, pero no garantiza superar
   one-hot, splines u otras alternativas. Esto justifica compartir los hechos
   históricos sin imponer idéntico preprocesamiento a todos los estimadores.
   [scikit-learn: Time-related feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html).

2. **Las transformaciones aprendidas pertenecen al entrenamiento.** scikit-learn
   indica separar train/test antes de ajustar escalado, imputación, selección u
   otras estadísticas aprendidas; aplicar después la transformación ajustada a
   evaluación. Sus pipelines ayudan a mantener esa separación en validación
   cruzada. Un builder determinista de calendario o de historia causal no equivale
   a un escalador ajustado sobre todo el año.
   [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).

3. **La propiedad operacional es la paridad, no el nombre del archivo.** Google
   documenta skew cuando entrenamiento y serving calculan entradas de distinta
   forma, recomienda medirlo y reutilizar señales realmente disponibles en
   producción. TFX ofrece un ejemplo concreto: exportar el grafo de transformación
   para que serving aplique el mismo procesamiento. Ninguna de estas fuentes exige
   un archivo por modelo ni un único `features.py`.
   [Google: Rules of Machine Learning, reglas 29 y 37](https://developers.google.com/machine-learning/guides/rules-of-ml),
   [TensorFlow: Transform component](https://www.tensorflow.org/tfx/guide/transform).

4. **La historia debe reconstruirse respecto al instante de decisión.** Feast
   describe joins point-in-time que recuperan valores anteriores al timestamp de
   cada entidad, dentro de su TTL. Además distingue tiempo del evento de tiempo de
   disponibilidad: un valor corregido después puede filtrarse retrospectivamente
   si solo se restringe el primero. El principio es útil aun sin instalar Feast.
   [Feast: Point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins).

## Recomendación para este proyecto (inferencia de ingeniería)

**Un solo `features.py` sí es compatible con estas prácticas**, siempre que sea
una frontera cohesiva y comprobable, no una función gigante con flags mezclados.
La recomendación no es afirmar que “todos los equipos de élite lo hacen así”, sino
adoptar las garantías observables que sustentan las fuentes anteriores:

- **Base compartida:** validación de contratos y agregación estación-hora de
  salidas y llegadas; completar horas sin viajes solo dentro de cobertura conocida.
  Un hueco de fuente no debe convertirse silenciosamente en demanda cero.
- **Constructores independientes por caso:** demanda consume lags/rolling y
  calendario; riesgo consume historia de entradas, salidas y flujo/capacidad;
  clustering consume perfiles de 24 horas por dirección más asimetría. P3 tiene
  otra granularidad y otro corte temporal: no forzarlo a la misma matriz de P1/P2.
- **Configuración explícita y contratos versionables:** parámetros pequeños para
  lags/ventanas, identificadores de entidad, corte y columnas de salida. Funciones
  públicas pequeñas y helpers privados compartidos permiten ampliar recetas sin
  duplicar agregaciones ni crear una jerarquía de clases prematura.
- **Una segunda frontera por estimador:** dentro de cada caso, Ridge, logística,
  árboles y clustering pueden seleccionar columnas/codificaciones distintas. Los
  escaladores, imputadores y encoders ajustables irán en el pipeline del modelo,
  no se ajustarán en el builder general. El entrenamiento no es alcance de esta nota.
- **Sin feature store por ahora:** tres casos offline y un repositorio local no
  justifican añadir Feast/TFX solo por reputación. Mantener contratos, cortes y
  pruebas permite una migración futura si aparecen consumidores distribuidos,
  serving online o necesidades reales de backfills y reutilización.

## Riesgos concretos que el diseño debe hacer visibles

Estas son aplicaciones del principio de disponibilidad de Google/Feast a los
canvases locales, no hechos sobre una infraestructura productiva ya existente:

- Si se predice a las **10:50**, no se puede usar el conteo final de 10:00–11:00.
  Para un objetivo 11:00–12:00, la última hora completa es 09:00–10:00. Definir
  explícitamente si cada lag se mide respecto al inicio objetivo o al corte.
- Para llegadas, usar `Fin_del_viaje`: conocer el origen de un viaje a las 10:50
  no hace disponible su destino observado a las 11:15. Ventanas cerradas a la
  izquierda y shift causal deben respetarse por estación, incluso entre meses.
- Los CSV mensuales permiten **simulación histórica offline**, pero no prueban
  disponibilidad operativa al minuto 50. Declarar esa limitación.
- Un catálogo descargado en **2026** no demuestra capacidad o coordenadas vigentes
  en **2024**. Permitir su uso solo como supuesto retrospectivo explícito, o exigir
  metadatos históricos para reclamar point-in-time correctness total.
- Flujo neto/capacidad no es inventario ni está acotado necesariamente a [-1, 1]:
  una estación puede rotar varias veces en una hora. No recortar el ratio.
- P3 puede describir todo 2024 al cierre del periodo, pero su perfil anual no debe
  incorporarse como predictor de enero de 2024. Un vector de dirección sin viajes
  requiere política explícita de cero/ausencia y no normalización por cero.

## Pruebas que evidencian la garantía

Cambiar o agregar viajes posteriores al corte no debe alterar features previas;
un viaje iniciado antes y terminado después solo debe afectar salidas conocidas;
la historia no debe cruzar estaciones; las horas faltantes no deben desplazar
lags; y entrenamiento/serving deben invocar la misma receta. Probar además
capacidad nula/no positiva, columnas exclusivas por caso y normalización de P3.
