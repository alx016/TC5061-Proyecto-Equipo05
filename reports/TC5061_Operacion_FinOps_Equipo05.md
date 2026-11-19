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
sostenible debe sumar ese cargo o compararse contra una instancia sin burst. La tarifa y las
especificaciones de t3.small y la tarifa de EBS gp3 se confirmaron el 18 de noviembre de 2026.
Falta confirmar los créditos de CPU y poner su fecha de consulta en Referencias (WP-3b). -->

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
