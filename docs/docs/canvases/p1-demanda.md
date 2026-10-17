# Machine Learning Canvas — P1: Regresión de Demanda de Viajes

!!! info "Ficha Técnica del Caso de Uso"
    - **Marco metodológico:** OwnML (Louis Dorard, v1.2)  
    - **Proyecto:** TC5061 - MiBici (Equipo 05)  
    - **Pregunta de negocio (P1):** ¿Cuántos viajes se iniciarán en una estación en un periodo futuro?  
    - **Tipo de tarea:** Regresión supervisada  
    - **Usuario principal:** Operador del sistema MiBici (Planeación de flota y logística)

---

## 1. Propuesta de Valor (Value Proposition)

- **Objetivo:** Predecir con 1 hora de anticipación el número de viajes que se iniciarán en cada una de las 360 estaciones activas de MiBici durante la siguiente hora ($t+1$).
- **Valor para el usuario:** Permite al operador del centro de control anticipar cuellos de botella y picos de demanda pendulares (7:00–9:00 hrs y 18:00–20:00 hrs), programando con anticipación el suministro de bicicletas antes de que se agoten.
- **Ventaja sobre heurísticas:** Supera a los promedios estáticos al capturar interacciones dinámicas de temporalidad (día festivo, día hábil, tendencias de semanas recientes y estacionalidad horaria).

---

## 2. Tarea Central de Machine Learning (Central ML Task)

- **Tipo de tarea:** Regresión supervisada tabular.
- **Entidad / Granularidad:** Estación-hora ($s \times h$).
- **Horizonte de predicción:** 1 hora hacia adelante ($t+1$).
- **Variable objetivo ($Y$):** Cantidad de viajes iniciados en la estación $s$ durante la hora $t+1$ ($Y_{s, t+1} \in \mathbb{N}_0$).
- **Restricción de no-negatividad:** La predicción debe ser acotada $\hat{Y} \ge 0$.

---

## 3. Decisiones Operativas (Decisions)

- **Consumo de predicción:** El sistema de despacho operativo recibe cada hora la tabla de predicciones para todas las estaciones.
- **Acción derivada:**
    - Si $\hat{Y}_{s, t+1} > 0.7 \times \text{Capacidad}_s$, la estación se cataloga con **Alerta de alta demanda**, priorizando el envío preventivo de bicicletas si el stock remanente es bajo.
    - Ajuste dinámico de rutas de camionetas para concentrar inventario en nodos de salida crítica.

---

## 4. Inferencia en Producción (Making Predictions)

- **Modo:** Batch horario.
- **Momento de ejecución:** Minuto 50 de cada hora $t$ para generar las predicciones del intervalo $[t+1, t+2]$.
- **Latencia:** Menor a 30 segundos para scoring completo de las 360 estaciones.
- **Despliegue:** Módulo batch en Python (`mibici/modeling/predict.py`).

---

## 5. Recolección y Etiquetado (Data Collection)

- **Fuente de verdad:** Registros de viajes mensuales de datos abiertos de MiBici (`datos_abiertos_YYYY_MM.csv`).
- **Mecanismo de etiquetado:** Agregación determinista de filas filtradas por `Origen_Id` y truncadas a la hora de `Inicio_del_viaje`.
- **Latencia de etiqueta:** La etiqueta real $Y_{s, t+1}$ se consolida al concluir la hora $t+1$ (retroalimentación continua en producción).

---

## 6. Variables y Señales de Entrada (Features)

- **Lags históricos de demanda:** Salidas observadas en $t-1$, $t-2$, $t-24$ (mismo horario ayer) y $t-168$ (mismo horario semana previa).
- **Ventanas móviles:** Media móvil y desviación estándar de salidas de las últimas 3 y 6 horas.
- **Codificación temporal:** Seno y coseno de la hora ($\sin(2\pi h/24)$, $\cos(2\pi h/24)$), día de la semana (0 a 6), bandera binaria de fin de semana y festivo.
- **Atributos estáticos de estación:** Capacidad total de anclajes $C_s$, coordenadas espaciales (latitud, longitud), y conteo de estaciones vecinas en un radio de 500 metros.

---

## 7. Entrenamiento Offline (Building Models)

- **Esquema de aprendizaje:** Supervisado offline sobre histórico canónico 2024.
- **Partición temporal estricta:**
    - **Train:** Enero a Septiembre 2024 (75%).
    - **Validation:** Octubre a Noviembre 2024 (16.7%).
    - **Test Out-of-Time:** Diciembre 2024 (8.3%).
- **Modelos candidatos:**
    1. *Baseline Heurístico:* Media histórica por estación, día de semana y hora.
    2. *Modelo Lineal Regularizado:* Ridge / ElasticNet con penalización $L_2$.
    3. *Gradient Boosting:* LightGBM Regressor con función de pérdida Poisson / Tweedie para modelar conteos no negativos.

---

## 8. Métricas de Evaluación (Evaluation)

- **Métricas Offline:**
    - **WAPE (Weighted Absolute Percentage Error):** Métrica primaria de negocio; evita divisiones por cero en horas valle ($\text{WAPE} \le 18\%$).
    - **RMSE (Root Mean Squared Error):** Penaliza fuertemente errores grandes en horas pico.
    - **MAE (Mean Absolute Error):** Error medio interpretable en número absoluto de viajes.
- **Monitoreo en Producción:**
    - Seguimiento de WAPE rodante de 7 días.
    - Alerta de degradación de modelo si $\text{WAPE} > 25\%$ durante tres periodos consecutivos.
