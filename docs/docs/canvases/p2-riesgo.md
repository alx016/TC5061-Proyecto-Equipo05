# Machine Learning Canvas — P2: Clasificación de Riesgo Operativo

!!! info "Ficha Técnica del Caso de Uso"
    - **Marco metodológico:** OwnML (Louis Dorard, v1.2)  
    - **Proyecto:** TC5061 - MiBici (Equipo 05)  
    - **Pregunta de negocio (P2):** ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro?  
    - **Tipo de tarea:** Clasificación supervisada multiclase (3 clases)  
    - **Usuario principal:** Equipo de rebalanceo y logística de campo de MiBici

---

## 1. Propuesta de Valor (Value Proposition)

- **Objetivo:** Predecir con 1 hora de anticipación si una estación entrará en estado crítico de **Desabasto** (sin bicis para retirar) o **Saturación** (sin anclajes para devolver), para planificar el despacho de camionetas de rebalanceo antes del colapso.
- **Valor para el usuario:** Permite optimizar el ruteo de vehículos de carga asignando la acción correcta: llevar bicicletas a estaciones con desabasto inminente o recoger bicicletas en estaciones con saturación inminente.
- **Justificación del proxy:** Dado que el portal de datos abiertos no publica el inventario en tiempo real ($B_{s,t}$), el riesgo se aproxima formalmente a través del **Flujo Neto Normalizado por Capacidad** ($\Delta_{s,t}$), capturando la presión de vaciado o llenado relativo al tamaño físico de la estación.

---

## 2. Tarea Central de Machine Learning (Central ML Task)

- **Tipo de tarea:** Clasificación supervisada multiclase (3 clases mutuamente excluyentes).
- **Entidad / Granularidad:** Estación individual por intervalo de 1 hora ($s \times h$).
- **Horizonte de predicción:** 1 hora hacia adelante ($t+1$).
- **Formulación matemática del Target ($Y_{s, t+1}$):**
    Definiendo el flujo neto normalizado para la estación $s$ con capacidad física $C_s$:
    $$\Delta_{s, t+1} = \frac{\text{Llegadas}_{s, [t, t+1]} - \text{Salidas}_{s, [t, t+1]}}{C_s} \in [-1, 1]$$
    Con un umbral crítico de estrés operativo $\tau = 0.40$ (déficit o superávit $\ge 40\%$ de la capacidad de anclajes en 1 hora):
    $$Y_{s, t+1} = \begin{cases} 
    \text{Riesgo\_Desabasto} & \text{si } \Delta_{s, t+1} \le -0.40 \quad (\text{Salidas superan a entradas en } \ge 40\% \times C_s) \\
    \text{Riesgo\_Saturacion} & \text{si } \Delta_{s, t+1} \ge +0.40 \quad (\text{Entradas superan a salidas en } \ge 40\% \times C_s) \\
    \text{Balanceado} & \text{si } -0.40 < \Delta_{s, t+1} < +0.40 
    \end{cases}$$

---

## 3. Decisiones Operativas (Decisions)

- **Consumo:** El despachador de cuadrillas de rebalanceo recibe cada hora el mapa clasificado por color: Rojo (Desabasto), Azul (Saturación), Verde (Balanceado).
- **Acción concreta:**
    - **Alerta Desabasto:** Despachar camioneta con carga de bicicletas desde el taller o desde estaciones saturadas.
    - **Alerta Saturación:** Despachar camioneta para retirar bicicletas y liberar anclajes antes del arribo de viajes laborales.
    - **Priorización:** Se ordenan las alertas por el valor de probabilidad calibrada $P(Y = \text{Desabasto} \mid X)$ y por la capacidad $C_s$.

---

## 4. Inferencia en Producción (Making Predictions)

- **Modo:** Batch horario.
- **Momento de ejecución:** Minuto 50 de cada hora $t$ para programar el despacho de la hora $t+1$.
- **Latencia de respuesta:** Menor a 20 segundos para scoring probabilístico de la red completa.
- **Salida requerida:** Vector de probabilidades $[P(\text{Desabasto}), P(\text{Saturacion}), P(\text{Balanceado})]$ para permitir ajuste de umbral de decisión en campo.

---

## 5. Recolección y Etiquetado (Data Collection)

- **Fuentes:**
    - Viajes mensuales: `Origen_Id`, `Destino_Id`, `Inicio_del_viaje`, `Fin_del_viaje`.
    - Nomenclatura oficial: Identificador de estación y capacidad física total $C_s$.
- **Construcción de etiqueta:** Cálculo determinista al cierre de cada hora sumando viajes iniciados y finalizados por estación, normalizando por $C_s$ y aplicando la regla de discretización.

---

## 6. Variables y Señales de Entrada (Features)

- **Flujo neto reciente:** $\Delta_{s, t-1}$, $\Delta_{s, t-2}$, $\Delta_{s, t-24}$.
- **Demanda individualizada:** Llegadas y salidas separadas en las últimas 1, 2 y 3 horas.
- **Capacidad física y ubicación:** Capacidad total de anclajes $C_s$, municipio, latitud y longitud.
- **Contexto temporal:** Hora del día (seno/coseno), día de la semana, indicador festivo.
- **Estado de vecinos espaciales:** Promedio de flujo neto en las 3 estaciones contiguas más cercanas ($k\text{-NN}$ espacial).

---

## 7. Entrenamiento Offline (Building Models)

- **Esquema de aprendizaje:** Supervisado multiclase offline con balanceo de clases.
- **Partición temporal estricta:**
    - **Train:** Ene–Sep 2024 (75%).
    - **Validation:** Oct–Nov 2024 (16.7%).
    - **Test Out-of-Time:** Dic 2024 (8.3%).
- **Modelos candidatos:**
    1. *Baseline Heurístico:* Persistencia del estado previo ($\hat{Y}_{s, t+1} = Y_{s, t-1}$).
    2. *Regresión Logística Multinomial:* Regularización ElasticNet ($L_1/L_2$).
    3. *Gradient Boosting:* LightGBM Classifier con función de pérdida multiclase (`multiclass`) y pesos inversos de clase (`class_weight='balanced'`).

---

## 8. Métricas de Evaluación (Evaluation)

!!! important "Métricas Obligatorias ante Desbalance"
    Debido a que la clase mayoritaria `Balanceado` representa el 98.74% de las observaciones, se descarta *Accuracy* como métrica decisional:

    - **Métrica Primaria:** **Macro F1-Score** $\ge 0.72$ (pondera por igual las tres clases).
    - **Métrica de Detección:** **Recall en Desabasto** $\ge 85\%$ (minimiza usuarios varados sin bici).
    - **Métrica de Separación:** **PR-AUC Multiclase** $\ge 0.82$.
