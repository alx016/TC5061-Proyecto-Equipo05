<style>
@page {
  size: letter;
  margin: 2.54cm;
}
body {
  font-family: Arial, "Liberation Sans", Helvetica, sans-serif;
  font-size: 11pt;
  line-height: 1.15;
}
</style>

# Operación, pruebas de carga y FinOps del modelo P2 (Entrega 6)

- **Proyecto:** TC5061 MLOps, Equipo 05. Datos abiertos de MiBici.
- **Repositorio:** <https://github.com/alx016/TC5061-Proyecto-Equipo05> (orquestación en
  `docker-compose.yml`, pipeline en `.github/workflows/ci.yml`, pruebas de carga en `load_tests/`).
- **Sistema evaluado:** API de P2, riesgo de desabasto o saturación por estación
  (`mibici-equipo05-p2-riesgo`, champion v3), con el servidor de MLflow de la Entrega 2.
- **Fecha:** 22 de noviembre de 2026.

<!--
Guía del esqueleto (WP-0). Borrar este comentario en WP-D.

- Formato del PDF: Arial 11 (o Liberation Sans / Helvetica), interlineado 1.15.
- Cuerpo (Resumen a Referencias): de 3 a 5 cuartillas. Los Anexos no cuentan.
- Presupuesto por sección, en páginas: Resumen 0.4, Sec. 1 0.8, Sec. 2 0.6, Sec. 3 1.2,
  Sec. 4 1.3, Sec. 5 0.5. Total: 4.8.
- Cada quien edita solo su bloque. El responsable está en el comentario de cada sección.
- Figuras en reports/figures/entrega6/ (las genera scripts/plot_load_and_finops.py).
- Tablas largas, YAML completo y logs van a los Anexos.
-->

## Resumen

<!-- Manuel (WP-D). ~0.4 pág. Qué se construyó, CI en verde, RPS sostenibles, costo por
inferencia actual y optimizado, y la recomendación principal. Se escribe al final. -->

## 1. Arquitectura y orquestación con Docker Compose

<!-- Fernando (WP-1c). ~0.8 pág.
- Topología: servicios `api` y `mlflow`, red `mibici-net`, volumen `mlflow_data`.
- Variables de entorno, healthchecks, `depends_on: service_healthy` y límites de CPU/RAM.
- Un solo comando desde un clon limpio (fallback a `models/p2-champion-v3.zip`).
- Modos de carga del modelo: bundle local por defecto y `models:/...@champion` vía MLflow.
- Por qué no se usa Redis + RQ.
- Detalle en el Anexo A.1. -->

## 2. CI/CD con GitHub Actions

<!-- Manuel (WP-2b). ~0.6 pág.
- Disparadores y jobs: lint y seguridad, matriz de pruebas, compose + smoke, publicación a GHCR.
- Cachés (uv, Buildx) y cómo se valida el levantamiento con Compose.
- Evidencia del run en verde (enlace y captura).
- Detalle en el Anexo A.2. -->

## 3. Pruebas de carga con Locust y uso de recursos

<!-- Oscar (WP-4d). ~1.2 pág.
- Escenario y por qué es pertinente para MiBici (365 estaciones, lotes de 1, 20 y 365).
- Escalones: baseline, 10, 50 y 120 usuarios; comparación de batching.
- Figura 1: RPS y usuarios. Figura 2: latencia p50/p95/p99 y tasa de error.
- Figura 3: CPU y memoria por contenedor (docker stats).
- Interpretación: punto de saturación, cuellos de botella y áreas de mejora.
- Tablas completas en el Anexo B. -->

## 4. Análisis FinOps

<!-- Oscar (WP-4d), con la comparativa de precios de Emilio (WP-3b). ~1.3 pág.
- Instancia de referencia y precios (parámetros de la tabla de abajo).
- Costo por petición y por predicción, en tres regímenes: capacidad sostenible, demanda real
  24/7 de MiBici y atribución por servicio (api vs mlflow).
- Figura 4: costo por 1M de predicciones, línea base vs optimizaciones.
- Optimizaciones cuantificadas: batching, MLflow fuera del camino de inferencia, concurrencia y
  sobrecarga por petición, scale-to-zero.
- Memoria de cálculo en el Anexo C. -->

**Parámetros base acordados (WP-0).** Los usan `scripts/finops_calc.py` y esta sección.

| Parámetro | Valor | Fuente |
|:---|:---|:---|
| Instancia de referencia | AWS EC2 `t3.small` (2 vCPU, 2 GiB, solo EBS, red de hasta 5 Gbps), Linux, `us-east-1` | Amazon Web Services (2026a) |
| Tarifa bajo demanda | $0.0208 USD/h | Amazon Web Services (2026a) |
| Almacenamiento | EBS `gp3`, $0.08 USD/GB-mes | Amazon Web Services (2026b) |
| Horas por mes | 730 h | Convención de AWS |
| Demanda real de MiBici | 365 estaciones × 24 h = 8,760 predicciones/día (262,800/mes) | Entrega 5 |
| Créditos de CPU (`t3` unlimited) | $0.05 USD por vCPU-hora sobre la línea base (20% por vCPU) | Amazon Web Services (2026c) |

<!-- Las instancias t3 son burstable: sostener 80% de CPU consume créditos. El costo a capacidad
sostenible debe sumar ese cargo o compararse contra una instancia sin burst. Las especificaciones
y tarifa de t3.small, la tarifa de EBS gp3 y el cargo por créditos de CPU ($0.05 USD por vCPU-hora
sobre la línea base del 20%, Amazon Web Services 2026c) se verificaron el 18 de noviembre de 2026. -->

**Comparativa de precios serverless y ejecución bajo demanda (Optimización 4: scale-to-zero).** Tarifas públicas sin nivel gratuito verificadas el 18 de noviembre de 2026.

| Plataforma y configuración | Tarifa pública de cómputo y peticiones | Lote horario consolidado (batch=365: 730 req/mes, ~0.5 s/req = 365 s/mes) | Peticiones unitarias (batch=1: 262,800 req/mes, ~50 ms/req = 13,140 s/mes) | Ahorro mensual vs. t3.small 24/7 ($15.98 USD/mes) | Fecha de consulta y fuente |
|:---|:---|:---|:---|:---|:---|
| AWS EC2 `t3.small` 24/7 (`us-east-1`, 2 vCPU, 2 GiB + 10 GB EBS `gp3`; 20 GB EBS = $16.78 USD/mes) | $0.0208 USD/h ($15.18 USD/mes en 730 h) + $0.08 USD/GB-mes EBS ($0.80 USD/mes por 10 GB; $1.60 USD/mes por 20 GB) + $0.05 USD/vCPU-h créditos `t3` sobre línea base del 20% | $15.98 USD/mes (99.99% tiempo ocioso: 365 s activos de 2,628,000 s) | $15.98 USD/mes (99.50% tiempo ocioso: 13,140 s activos de 2,628,000 s) | Línea base (0%) | 18 de noviembre de 2026; Amazon Web Services (2026a, 2026b, 2026c) |
| AWS Lambda (`us-east-1`, x86, 1024 MB / 1 GB RAM) | $0.0000166667 USD/GB-s + $0.20 USD por 1,000,000 de invocaciones | $0.0062 USD/mes ($0.00608 USD cómputo + $0.00015 USD invocaciones; < $0.02 USD/mes) | $0.2716 USD/mes ($0.2190 USD cómputo + $0.0526 USD invocaciones; < $0.45 USD/mes) | 99.96% (lote) / 98.30% (unitario) | 18 de noviembre de 2026; Amazon Web Services (2026d) |
| Google Cloud Run (`us-central1`, Tier 1, *scale-to-zero* facturado por petición, 1 vCPU, 1 GiB RAM) | $0.00002400 USD/vCPU-s + $0.00000250 USD/GiB-s ($0.00002650 USD/s activo) + $0.40 USD por 1,000,000 de peticiones | $0.0100 USD/mes ($0.00967 USD cómputo + $0.00029 USD peticiones; < $0.02 USD/mes) | $0.4494 USD/mes a 49.5 ms ($0.4533 USD/mes a 50 ms: $0.3482 USD cómputo + $0.1051 USD peticiones; ~$0.38–$0.45 USD/mes en el rango 40–50 ms) | 99.94% (lote) / 97.16%–97.60% (unitario) | 18 de noviembre de 2026; Google Cloud (2026) |

Una instancia dedicada AWS EC2 `t3.small` encendida las 24 horas del día durante 730 horas al mes factura 2,628,000 segundos continuos por un costo fijo de $15.98 USD/mes con 10 GB de almacenamiento EBS `gp3`, o $16.78 USD/mes con 20 GB de EBS `gp3` (Amazon Web Services, 2026a, 2026b). Sin embargo, la operación real de MiBici para el modelo P2 requiere predecir 365 estaciones activas cada hora, lo que equivale a 8,760 predicciones diarias y 262,800 predicciones mensuales. Bajo peticiones unitarias (`batch=1`, 262,800 invocaciones mensuales a ~50 ms por petición), el servicio `api` consume 13,140 segundos de cómputo efectivo al mes y deja ocioso el 99.50% del tiempo facturado. Al consolidar la inferencia en un lote horario por corrida (`batch=365`, 730 peticiones mensuales a ~0.5 s por lote), el tiempo activo cae a 365 segundos al mes y el desperdicio por capacidad ociosa alcanza el 99.99% (~99.9% de la ventana mensual).

La combinación del lote horario consolidado (Optimización 1, `batch=365`) con el desacoplamiento de MLflow del camino crítico de inferencia mediante `--no-deps api` y el paquete inmutable `models/p2-champion-v3.zip` (Optimización 2) convierte al contenedor `api` en un servicio sin estado apto para ejecución bajo demanda con reducción a cero instancias (*scale-to-zero*, Optimización 4). Con el patrón por lote horario (730 invocaciones y 365 segundos de cómputo al mes), el gasto mensual baja a $0.0062 USD/mes en AWS Lambda (`us-east-1`, 1 GB RAM) y a $0.0100 USD/mes en Google Cloud Run (`us-central1`, 1 vCPU y 1 GiB RAM facturados por petición), y mantiene el costo por debajo de $0.02 USD/mes con un ahorro superior al 99.9% frente a la instancia `t3.small` dedicada (Amazon Web Services, 2026d; Google Cloud, 2026). Incluso sin agrupar peticiones (`batch=1`, 262,800 peticiones y 13,140 segundos al mes), un despliegue *serverless* factura $0.2716 USD/mes en AWS Lambda y entre $0.4494 y $0.4533 USD/mes en Google Cloud Run (< $0.45 USD/mes), lo que representa una reducción del 97.16% al 98.30% respecto al esquema 24/7.

## 5. Conclusiones

<!-- Borrador de Emilio (WP-3b) sobre los bullets del equipo; cierre de Manuel (WP-D). ~0.5 pág.
Cuatro o cinco bullets: operación en un comando, CI, comportamiento bajo carga, costo y
siguiente paso. -->

## Referencias

<!-- Emilio (WP-3b) completa y ordena en APA 7. Las de precios llevan fecha de consulta. -->

Amazon Web Services. (2026a). *Amazon EC2 On-Demand pricing*. Recuperado el 18 de noviembre de 2026, de
https://aws.amazon.com/ec2/pricing/on-demand/

Amazon Web Services. (2026b). *Amazon EBS pricing*. Recuperado el 18 de noviembre de 2026, de
https://aws.amazon.com/ebs/pricing/

Amazon Web Services. (2026c). *Burstable performance instances: Unlimited mode*. Recuperado el
[fecha] de https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/burstable-performance-instances-unlimited-mode.html

Docker. (2026). *Docker Compose documentation* [Documentación de software].
https://docs.docker.com/compose/

FinOps Foundation. (2026). *What is FinOps?* Recuperado el [fecha] de
https://www.finops.org/introduction/what-is-finops/

Gift, N., & Deza, A. (2021). *Practical MLOps: Operationalizing machine learning models*.
O'Reilly Media.

GitHub. (2026). *GitHub Actions documentation* [Documentación de software].
https://docs.github.com/en/actions

Locust. (2026). *Locust documentation* [Documentación de software]. https://docs.locust.io/

MLflow Project. (2026). *MLflow documentation* [Documentación de software].
https://mlflow.org/docs/latest/

Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A.,
Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps: How to scale machine learning in the
enterprise*. O'Reilly Media.

Wilson, B. (2022). *Machine learning engineering in action*. Manning Publications.

<div style="page-break-before: always"></div>

# Anexos

Los anexos amplían el cuerpo del reporte y no cuentan en la extensión.

## Anexo A.1. Docker Compose

<!-- Fernando (WP-1c). Extractos de docker-compose.yml y Containerfile*, validación con el
Registry en vivo y del modo sin MLflow (`--no-deps api`). -->

## Anexo A.2. Pipeline de CI/CD

<!-- Manuel (WP-2b). Extractos de ci.yml, captura del run en verde e imagen en GHCR. -->

## Anexo B. Resultados completos de las pruebas de carga

<!-- Oscar (WP-4d). Pegar reports/load/anexo_b.md (lo genera collect_load_metrics.py). -->

## Anexo C. Memoria de cálculo FinOps

<!-- Oscar (WP-4d). Pegar reports/load/anexo_c.md (lo genera finops_calc.py) y los comandos
para reproducir. -->
