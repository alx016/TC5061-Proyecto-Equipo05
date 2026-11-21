### Anexo B. Resultados completos de carga y monitoreo de recursos

*Generado automáticamente el 2026-11-20T00:58:06Z desde `reports/load`.*

#### B.1. Métricas de Locust por escenario y endpoint

| Escenario | Método | Endpoint | Peticiones | Fallos | Error (%) | p50 (ms) | p95 (ms) | p99 (ms) | RPS |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| `nominal_10u` | `GET` | `/health` | 145 | 0 | 0.00 | 10.0 | 110.0 | 310.0 | 1.22 |
| `nominal_10u` | `GET` | `/model-info` | 161 | 0 | 0.00 | 11.0 | 180.0 | 410.0 | 1.35 |
| `nominal_10u` | `POST` | `/predict?batch=1` | 746 | 0 | 0.00 | 38.0 | 260.0 | 770.0 | 6.26 |
| `nominal_10u` | `POST` | `/predict?batch=20` | 486 | 0 | 0.00 | 42.0 | 340.0 | 630.0 | 4.08 |
| `nominal_10u` | `POST` | `/predict?batch=365` | 162 | 0 | 0.00 | 120.0 | 710.0 | 1200.0 | 1.36 |
| `high_50u` | `GET` | `/health` | 257 | 0 | 0.00 | 790.0 | 1800.0 | 2400.0 | 2.15 |
| `high_50u` | `GET` | `/model-info` | 250 | 0 | 0.00 | 1100.0 | 2500.0 | 2900.0 | 2.09 |
| `high_50u` | `POST` | `/predict?batch=1` | 1290 | 0 | 0.00 | 1300.0 | 2800.0 | 4200.0 | 10.81 |
| `high_50u` | `POST` | `/predict?batch=20` | 793 | 0 | 0.00 | 1400.0 | 2800.0 | 4300.0 | 6.64 |
| `high_50u` | `POST` | `/predict?batch=365` | 253 | 0 | 0.00 | 1800.0 | 3100.0 | 4900.0 | 2.12 |
| `stress_120u` | `GET` | `/health` | 228 | 0 | 0.00 | 3300.0 | 5800.0 | 6300.0 | 1.91 |
| `stress_120u` | `GET` | `/model-info` | 237 | 0 | 0.00 | 4500.0 | 7500.0 | 8000.0 | 1.99 |
| `stress_120u` | `POST` | `/predict?batch=1` | 1144 | 0 | 0.00 | 4700.0 | 7600.0 | 8200.0 | 9.60 |
| `stress_120u` | `POST` | `/predict?batch=20` | 656 | 0 | 0.00 | 4700.0 | 7700.0 | 8400.0 | 5.50 |
| `stress_120u` | `POST` | `/predict?batch=365` | 232 | 0 | 0.00 | 5500.0 | 8200.0 | 9200.0 | 1.95 |
| `batch_1x365` | `POST` | `/predict?batch=365` | 705 | 0 | 0.00 | 78.0 | 110.0 | 200.0 | 11.94 |
| `batch_365x1` | `POST` | `/predict?batch=1` | 3277 | 0 | 0.00 | 16.0 | 27.0 | 43.0 | 55.48 |

#### B.2. Comparativa de batching (`batch_1x365` vs `batch_365x1`)

| Escenario | Tamaño lote | RPS HTTP | Predicciones/s | p50 (ms) | p95 (ms) | Factor Pred/s |
|:---|---:|---:|---:|---:|---:|---:|
| `batch_365x1` | 1 | 55.48 | 55.48 | 16.0 | 27.0 | 1.00x |
| `batch_1x365` | 365 | 11.94 | 4358.06 | 78.0 | 110.0 | 78.56x |

#### B.3. Consumo de recursos de contenedores (`docker stats`)

| Escenario | Contenedor | Muestras | CPU prom (%) | CPU máx (%) | RAM prom (MiB) | RAM máx (MiB) | RAM prom (%) | PIDs máx |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | `api` | 30 | 1.75 | 21.87 | 241.25 | 241.50 | 23.56 | 24 |
| `baseline` | `mlflow` | 30 | 2.97 | 39.21 | 373.71 | 384.30 | 72.99 | 21 |
| `nominal_10u` | `api` | 60 | 47.79 | 101.85 | 247.31 | 259.20 | 24.15 | 33 |
| `nominal_10u` | `mlflow` | 60 | 3.47 | 41.99 | 374.27 | 387.50 | 73.10 | 21 |
| `high_50u` | `api` | 60 | 99.16 | 109.10 | 267.02 | 282.70 | 26.08 | 65 |
| `high_50u` | `mlflow` | 60 | 3.37 | 43.68 | 373.99 | 384.70 | 73.04 | 21 |
| `stress_120u` | `api` | 60 | 100.54 | 108.15 | 280.32 | 293.90 | 27.38 | 69 |
| `stress_120u` | `mlflow` | 60 | 3.20 | 48.20 | 374.64 | 386.00 | 73.17 | 26 |
| `batch_1x365` | `api` | 31 | 94.12 | 101.85 | 277.98 | 288.00 | 27.15 | 63 |
| `batch_1x365` | `mlflow` | 31 | 3.12 | 47.31 | 374.93 | 375.30 | 73.23 | 20 |
| `batch_365x1` | `api` | 31 | 87.54 | 99.90 | 277.46 | 288.40 | 27.10 | 26 |
| `batch_365x1` | `mlflow` | 31 | 4.00 | 48.79 | 377.05 | 389.20 | 73.64 | 21 |

#### B.4. Microbenchmark de sobrecarga por petición (Opt 3)

| Lote | Ruta de inferencia | Media (ms) | p50 (ms) | p95 (ms) | Predicciones/s | Aceleración |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | `pyfunc_dataframe` | 10.4810 | 9.0135 | 19.5737 | 95.41 | 1.00x |
| 1 | `booster_numpy` | 0.5145 | 0.1959 | 2.8341 | 1943.72 | 20.37x |
| 1 | `booster_numpy_1thread` | 0.1909 | 0.1886 | 0.2308 | 5238.91 | 54.91x |
| 20 | `pyfunc_dataframe` | 13.8600 | 10.2022 | 26.0605 | 1443.00 | 1.00x |
| 20 | `booster_numpy` | 3.2150 | 1.4230 | 8.1398 | 6220.92 | 4.31x |
| 20 | `booster_numpy_1thread` | 2.3379 | 2.2566 | 3.0486 | 8554.61 | 5.93x |
| 365 | `pyfunc_dataframe` | 17.3909 | 15.6613 | 26.8483 | 20987.95 | 1.00x |
| 365 | `booster_numpy` | 10.8354 | 8.0723 | 22.7326 | 33686.03 | 1.61x |
| 365 | `booster_numpy_1thread` | 38.5979 | 35.1957 | 54.9231 | 9456.46 | 0.45x |
