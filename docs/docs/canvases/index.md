# Arquitectura de Machine Learning Canvases

Este módulo documenta la especificación formal de los tres casos de uso analíticos del proyecto, desarrollados bajo el marco metodológico del **Machine Learning Canvas v1.2** (Dorard, 2018).

---

## Matriz Comparativa de Casos de Uso

| Dimensión | P1 · Regresión de Demanda | P2 · Clasificación de Riesgo | P3 · Clustering Funcional |
|:---|:---|:---|:---|
| **Tipo de Tarea** | Regresión supervisada | Clasificación supervisada tri-clase | Aprendizaje no supervisado |
| **Entidad / Granularidad** | Estación-hora ($s \times h$) | Estación-hora ($s \times h$) | Estación individual ($s \in \{1,\dots,N\}$) |
| **Horizonte Temporal** | 1 hora hacia adelante ($t+1$) | 1 hora hacia adelante ($t+1$) | Perfil anual consolidado (24 horas) |
| **Variable Objetivo** | Viajes iniciados ($Y_{s,t+1} \in \mathbb{N}_0$) | Riesgo tri-clase ($\text{Desabasto}, \text{Balanceado}, \text{Saturacion}$) | Asignación a tipología urbana ($k \in \{1,\dots,K\}$) |
| **Señal Operativa Clave** | Lags de demanda ($t-1, t-2, t-24, t-168$) | Flujo neto normalizado ($\Delta_{s,t} = \frac{In - Out}{C_s}$) | Perfil horario medio normalizado ($L_1$) |
| **Métrica Primaria** | WAPE $\le 18\%$ / RMSE | PR-AUC $\ge 0.82$ / Macro F1 | Silhouette Score $\ge 0.45$ / Inercia |
| **Usuario Operativo** | Centro de control y planeación de flota | Cuadrillas de rebalanceo en campo | Dirección de planeación y expansión de red |

---

## Casos de Uso Detallados

<div class="grid cards" markdown>

- [**P1 — Regresión de Demanda de Viajes**](p1-demanda.md)
    
    Predicción horaria de viajes salientes por estación con 1 hora de anticipación. Modela la demanda esperada para programar el suministro preventivo de bicicletas.

- [**P2 — Clasificación de Riesgo Operativo**](p2-riesgo.md)
    
    Detección temprana de eventos de desabasto (estación vacía) o saturación (estación llena) a partir de la tasa de cambio de inventario vs. capacidad física.

- [**P3 — Clustering Funcional de Estaciones**](p3-clustering.md)
    
    Segmentación no supervisada de estaciones según su perfil diario de 24 horas, identificando nodos residenciales, corporativos y de tránsito mixto.

</div>
