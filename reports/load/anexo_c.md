### C.1 Parámetros

| Parámetro | Valor |
|:---|:---|
| Instancia | AWS EC2 `t3.small`, $0.0208/h |
| Disco | 10 GB EBS gp3, $0.8/mes |
| Horas por mes (precio y demanda) | 730 |
| Créditos de CPU | $0.05 por vCPU-hora sobre 20% de cada vCPU |
| Demanda real | 365 estaciones × 730 h = 266,450 predicciones/mes |
| Serverless (Opt 4) | $1.667e-05/s activo + $2e-07/petición |

### C.2 Escenarios medidos

| Escenario | Peticiones/s | Predicciones/s | Error (%) | CPU api (%) | CPU mlflow (%) |
|:---|---:|---:|---:|---:|---:|
| `nominal_10u` | 14.3 | 583.8 | 0.00 | 47.8 | 3.5 |
| `high_50u` | 23.8 | 917.4 | 0.00 | 99.2 | 3.4 |
| `stress_120u` | 20.9 | 830.0 | 0.00 | 100.5 | 3.2 |
| `batch_1x365` | 11.9 | 4358.1 | 0.00 | 94.1 | 3.1 |
| `batch_365x1` | 55.5 | 55.5 | 0.00 | 87.5 | 4.0 |

Pico sostenible: `nominal_10u` (más predicciones/s con error ≤ 1% y CPU de `api` ≤ 80%).

### C.3 Cálculo

- Costo/hora en el pico = tarifa + créditos = $0.0208 + $0.005627
- Costo/petición = costo/hora / (peticiones/s × 3600) = $5.148e-07
- Costo/predicción = costo/hora / (predicciones/s × 3600) = $1.258e-08
- Costo por 1M de predicciones en el pico = $0.01258
- Costo mensual 24/7 = tarifa × 730 + disco = $15.98
- Costo por 1M de predicciones con la demanda real = costo mensual / 266,450 × 10⁶ = $59.99
- Atribución: `api` 66.5%, `mlflow` 33.5% (promedio de su parte de CPU y de memoria)

### C.4 Optimizaciones contra el pico sostenible

Línea base: $0.01258 por 1M de predicciones.

| Optimización | Costo por 1M pred. | Costo mensual | Ahorro |
|:---|---:|---:|---:|
| Batching de 365 estaciones por petición | $0.00315 | $15.98 | 74.95% |
| MLflow fuera del camino de inferencia | $0.008364 | $10.63 | 33.49% |
| Inferencia directa con el Booster y menos sobrecarga por petición | $0.007811 | $15.98 | 37.89% |

### C.5 Optimizaciones contra la operación real 24/7

Línea base: $15.98/mes, $59.99 por 1M de predicciones. Las dos escalas no son comparables entre sí.

| Optimización | Costo por 1M pred. | Costo mensual | Ahorro |
|:---|---:|---:|---:|
| Scale-to-zero con una corrida por hora | $0.00411 | $0.001095 | 99.99% |

Sin agrupar (365 peticiones de 1 estación por hora): $0.1243/mes, ahorro de 99.22%.
