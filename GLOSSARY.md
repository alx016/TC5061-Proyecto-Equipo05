# MiBici MLOps

Sistema de analítica predictiva, servicio y monitoreo continuo de la red de bicicletas públicas MiBici en el Área Metropolitana de Guadalajara.

## Monitoreo y Observabilidad

**Reference Dataset**:
Conjunto de datos histórico correspondiente a la partición de entrenamiento (enero a septiembre de 2024) con el que se ajustó el modelo champion.
_Avoid_: Baseline data, golden set

**Current Dataset**:
Conjunto de datos del periodo posterior (enero a septiembre de 2025) utilizado para auditar la vigencia de los supuestos del modelo en producción.
_Avoid_: Production data, streaming data

**Data Drift**:
Desplazamiento en la distribución estadística de las características de entrada ($P(X)$) entre el periodo de referencia y el actual.
_Avoid_: Covariate shift, feature drift

**Prediction Drift**:
Desplazamiento en la distribución de las probabilidades o clases operativas emitidas por el modelo ($P(\hat{Y})$) entre periodos.
_Avoid_: Model drift, output shift

**Concept Drift**:
Cambio en la relación funcional entre las características de entrada y la variable objetivo ($P(Y|X)$), evaluable cuando se observa la etiqueta real.
_Avoid_: Target drift, label shift

**Distance Metric**:
Medida cuantitativa de divergencia de probabilidad (ej. Wasserstein normalizada o Jensen-Shannon) que mide la magnitud práctica del desplazamiento independiente del tamaño muestral.
_Avoid_: P-value test, significance test

