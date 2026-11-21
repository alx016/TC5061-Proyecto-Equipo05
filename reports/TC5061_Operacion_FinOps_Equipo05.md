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

El workflow `.github/workflows/ci.yml` tiene tres disparadores: push a `main`, PR hacia `main` y ejecución manual (`workflow_dispatch`). En todos se corren las validaciones de lint y seguridad, pruebas y Compose (detalle en la tabla); `publish-image` solo corre en push a `main`. Con `concurrency` y `cancel-in-progress: true`, cada push nuevo a la misma rama cancela el run anterior para no gastar minutos en código viejo. Siguiendo el **principio de mínimo privilegio**, el workflow tiene `contents: read` por defecto y solo el job de publicación recibe `packages: write`; no se usan secretos propios, solo el `GITHUB_TOKEN` que GitHub genera en cada run. Así, ningún job puede modificar el repo ni publicar paquetes sin haber pasado todas las validaciones.

| Job | Qué valida y cómo | Qué falla atrapa |
|:---|:---|:---|
| `lint-and-security` | `make lint`: formato y errores estáticos (`ruff`); `make security`: `bandit` (código propio), `pip-audit` (dependencias) y `detect-secrets` (todo el repo). | Código mal formateado o con errores estáticos; código inseguro; librerías con vulnerabilidades conocidas (CVE); credenciales commiteadas. |
| `test-matrix` | La suite de pytest se divide en dos jobs paralelos con marcadores: `slow` (suites de DeepChecks) y `not slow` (todo lo demás); `fail-fast: false` permite que ambos terminen aunque uno falle; `make prepare-model` descomprime `models/p2-champion-v3.zip` en `models/p2-riesgo/`. | Regresiones en la lógica (contrato de la API, features, entrenamiento, cálculo FinOps); problemas de integridad o drift en los datos (DeepChecks); que la API deje de ser compatible con el champion real, lo que sin el bundle se reportaría como *skip* (falso verde) en lugar de fallo. |
| `compose-and-smoke` | Construye las dos imágenes de `docker-compose.yml` (`api` y `mlflow`); levanta el stack con `docker compose up --wait`, que espera a que pasen los healthchecks; `make compose-smoke` prueba `/health` de MLflow y `/model-info`, `/health` y `/predict` de la API; verifica que se sirva `mibici-equipo05-p2-riesgo` en su versión 3. | Errores de red, variables de entorno, healthchecks o un `Containerfile` roto; que la API sirva otro modelo o una versión equivocada. |
| `publish-image` | Depende (`needs`) de los tres jobs anteriores y solo corre en push a `main` (`if`); publica `ghcr.io/alx016/mibici-p2-api` con dos tags: `latest` y el SHA del commit. | No detecta errores, los previene: evita que una imagen que no pasó lint, pruebas o el smoke test llegue a GHCR y alguien la despliegue. El tag por SHA da trazabilidad (imagen ↔ commit) y permite hacer rollback si la versión nueva falla. |

La caché acelera los runs al no reconstruir lo que no cambió: dependencias de Python (incluido el entorno de DeepChecks) con uv y capas de Docker con Buildx. `compose-and-smoke` y `publish-image` comparten la caché de la imagen `api`, así que la publicación reutiliza sus capas: en el run [`44657398679`](https://github.com/alx016/TC5061-Proyecto-Equipo05/actions/runs/44657398679) de `main`, la imagen se construyó en 25 s con el 50% de las capas en caché.

Ese run pasó en verde en ~10 min (capturas en el Anexo A.2). La ruta crítica es `Tests (fast)` (~9.3 min) seguido de `publish-image` (38 s): para acelerar el pipeline habría que acelerar `fast`, porque optimizar los otros jobs no cambiaría el tiempo total.

El pipeline cubre CI y **entrega continua**, pero no despliegue continuo: nada se despliega automáticamente a un servidor. Las pruebas de Locust (Sección 3) tampoco corren en CI por su duración y porque en runners compartidos los resultados varían según la máquina virtual, lo que haría poco confiable el cálculo FinOps (Sección 4).

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

- **Orquestación reproducible en un comando y respaldo por paquete inmutable.** `docker-compose.yml` despliega `mibici-api` (`1.0 vCPU`, `1024 MiB`) y `mibici-mlflow` (`0.5 vCPU`, `512 MiB`, MLflow `3.17.2` con `--allowed-hosts` y volumen `mlflow_data`) en la red `mibici-net` sincronizados mediante `depends_on` con `condition: service_healthy`. El `Containerfile` desempaqueta automáticamente `models/p2-champion-v3.zip` cuando falta el directorio local, lo que garantiza el arranque con `docker compose up -d --build` desde un clon limpio y permite alternar sin cambios de código hacia la resolución en vivo del Registry (`models:/mibici-equipo05-p2-riesgo@champion`).
- **Gobernanza automatizada de CI/CD y seguridad en GitHub Actions.** El flujo `.github/workflows/ci.yml` bloquea regresiones mediante compuertas estáticas y de seguridad (`ruff`, `bandit`, `pip-audit` y `detect-secrets`), ejecuta en paralelo la matriz de pruebas `fast` y `slow` con caché de `uv` (incluyendo DeepChecks en Python 3.12 y el contrato del paquete `v3` con `make prepare-model`), valida la pila completa con pruebas de humo en Docker Compose (`/health`, `/model-info` y `/predict`) y publica imágenes verificadas en GHCR (`ghcr.io/alx016/mibici-p2-api`).
- **Resiliencia bajo carga de 365 estaciones y perfilado de concurrencia en Locust.** Las pruebas de carga (`load_tests/locustfile.py`) y la telemetría de `docker stats` (`scripts/collect_load_metrics.py`) sobre escalones de 10, 50 y 120 usuarios concurrentes demuestran que agrupar las 365 estaciones activas de MiBici en lotes (`batch=20` y `batch=365` frente a `batch=1`) multiplica el rendimiento efectivo en predicciones por segundo. El perfilado bajo saturación identifica como cuellos de botella la contención del GIL en un único proceso Uvicorn, la construcción repetida de `pd.DataFrame` con `astype(SIGNATURE_DTYPES)` en `mlflow.pyfunc` y la competencia de hilos OpenMP en LightGBM, lo que fundamenta la ejecución directa sobre `Booster` con arreglos `numpy` (Optimización 3).
- **Eficiencia FinOps mediante procesamiento por lotes, desacoplamiento y scale-to-zero.** La demanda real de MiBici (`365` estaciones por `24 h` = `262,800` predicciones/mes) deja ociosa una instancia dedicada AWS EC2 `t3.small` (`$15.98 USD/mes` con 10 GB EBS `gp3` o `$16.78 USD/mes` con 20 GB) entre el `99.50%` (`batch=1`) y el `99.99%` (`batch=365`) del mes. Combinar el lote horario consolidado (`730` peticiones/mes, Optimización 1) y el retiro de MLflow del camino de inferencia (`--no-deps api`, Optimización 2) con ejecución *serverless* bajo demanda (Optimización 4) reduce el gasto mensual a `$0.0062 USD/mes` en AWS Lambda y `$0.0100 USD/mes` en Google Cloud Run (`< $0.02 USD/mes`, ahorro `>99.9%`), o a `< $0.45 USD/mes` bajo peticiones unitarias (ahorro `>97%` a `98.3%`, FinOps Foundation, 2026).
- **Madurez MLOps de extremo a extremo para la operación de MiBici P2.** La integración de las Entregas 1 a 6 consolida un ciclo operativo trazable que enlaza datos versionados por manifiesto y validación DeepChecks (E1 y E3), gobierno de experimentos y Model Registry en MLflow (E2), servicio contenedorizado con contratos estrictos en FastAPI y Pydantic (E4), monitoreo de *data drift* con Evidently y pruebas A/B con disparadores de reentrenamiento continuo (E5), y operación reproducible con CI/CD, pruebas de estrés y disciplina FinOps lista para producción (E6; Gift & Deza, 2021; Treveil et al., 2020; Wilson, 2022).

## Referencias

<!-- Emilio (WP-3b) completa y ordena en APA 7. Las de precios llevan fecha de consulta. -->

Amazon Web Services. (2026a). *Amazon EC2 On-Demand pricing*. Recuperado el 18 de noviembre de 2026, de https://aws.amazon.com/ec2/pricing/on-demand/

Amazon Web Services. (2026b). *Amazon EBS pricing*. Recuperado el 18 de noviembre de 2026, de https://aws.amazon.com/ebs/pricing/

Amazon Web Services. (2026c). *Burstable performance instances: Unlimited mode*. Recuperado el 18 de noviembre de 2026, de https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/burstable-performance-instances-unlimited-mode.html

Amazon Web Services. (2026d). *AWS Lambda pricing*. Recuperado el 18 de noviembre de 2026, de https://aws.amazon.com/lambda/pricing/

Docker. (2026). *Docker Compose documentation* [Documentación de software]. https://docs.docker.com/compose/

FinOps Foundation. (2026). *What is FinOps?* Recuperado el 18 de noviembre de 2026, de https://www.finops.org/introduction/what-is-finops/

Gift, N., & Deza, A. (2021). *Practical MLOps: Operationalizing machine learning models*. O'Reilly Media.

GitHub. (2026). *GitHub Actions documentation* [Documentación de software]. https://docs.github.com/en/actions

Google Cloud. (2026). *Cloud Run pricing*. Recuperado el 18 de noviembre de 2026, de https://cloud.google.com/run/pricing

Locust. (2026). *Locust documentation* [Documentación de software]. https://docs.locust.io/

MLflow Project. (2026). *MLflow documentation* [Documentación de software]. https://mlflow.org/docs/latest/

Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A., Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps: How to scale machine learning in the enterprise*. O'Reilly Media.

Wilson, B. (2022). *Machine learning engineering in action*. Manning Publications.

<div style="page-break-before: always"></div>

# Anexos

Los anexos amplían el cuerpo del reporte y no cuentan en la extensión.

## Anexo A.1. Docker Compose

<!-- Fernando (WP-1c). Extractos de docker-compose.yml y Containerfile*, validación con el
Registry en vivo y del modo sin MLflow (`--no-deps api`). -->

## Anexo A.2. Pipeline de CI/CD

Las capturas corresponden al run [`44657398679`](https://github.com/alx016/TC5061-Proyecto-Equipo05/actions/runs/44657398679) del workflow `CI`, disparado por el push a `main` del merge del PR #60 (commit `1df0f86`), y al paquete [`ghcr.io/alx016/mibici-p2-api`](https://github.com/alx016/TC5061-Proyecto-Equipo05/pkgs/container/mibici-p2-api) que publicó.

**Disparadores, permisos por defecto y cancelación de runs** (`.github/workflows/ci.yml`, líneas 3–16):

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

**Levantamiento del stack y smoke test en `compose-and-smoke`** (líneas 114–132). `--wait` espera a que los healthchecks de ambos servicios pasen; el último paso comprueba que se sirva el champion v3:

```yaml
- name: Start the stack
  run: docker compose up -d --no-build --wait --wait-timeout 180

- name: Smoke test (/health, /model-info, /predict)
  run: make compose-smoke

- name: Check the served model
  run: |
    info=$(curl -fsS http://localhost:8000/model-info)
    echo "$info"
    test "$(echo "$info" | jq -r .model_name)" = "mibici-equipo05-p2-riesgo"
    test "$(echo "$info" | jq -r .model_version)" = "3"

- name: Show container logs
  if: failure()
  run: docker compose logs --tail=200

- name: Stop the stack
  if: always()
  run: docker compose down -v
```

**Publicación condicionada en `publish-image`** (líneas 134–172, extracto). Solo corre en push a `main` tras los otros tres jobs, es el único con `packages: write` y reutiliza la caché `api`:

```yaml
publish-image:
  needs: [lint-and-security, test-matrix, compose-and-smoke]
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  permissions:
    contents: read
    packages: write
  steps:
    # ... checkout, setup-buildx y login a ghcr.io con GITHUB_TOKEN ...
    - id: meta
      uses: docker/metadata-action@v6
      with:
        images: ghcr.io/${{ github.repository_owner }}/mibici-p2-api
        tags: |
          type=raw,value=latest
          type=sha

    - name: Build and push
      uses: docker/build-push-action@v7
      with:
        context: .
        file: Containerfile
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        cache-from: type=gha,scope=api
        cache-to: type=gha,scope=api,mode=max
```

![Figura A.1. Resumen del run del workflow CI en verde](figures/entrega6/ci_run_verde1.png)

**Figura A.1.** Resumen del run `44657398679` (10 min 5 s en total). Los cinco jobs terminaron en verde y el grafo muestra que `publish-image` espera a la matriz `test-matrix`, a `lint-and-security` (1 min 25 s) y a `compose-and-smoke` (2 min 53 s). Abajo, el resumen que escribe `Tests (fast)`: 516 pruebas pasadas y 5 excluidas (las `slow`).

![Figura A.2. Resumen de publicación de la imagen y del build de Docker](figures/entrega6/ci_run_verde2.png)

**Figura A.2.** Resumen del job `publish-image` del mismo run: la imagen se publicó con los tags `latest` y `sha-1df0f86`, y el build del `Containerfile` tardó 25 s con el 50% de las capas tomadas de la caché compartida `api`.

![Figura A.3. Paquete mibici-p2-api publicado en GitHub Container Registry](figures/entrega6/ghcr_paquete.png)

**Figura A.3.** Paquete público `mibici-p2-api` en GHCR, ligado al repositorio, con los tags `latest` y `sha-1df0f86` y el comando `docker pull` para descargar la versión exacta del commit.

## Anexo B. Resultados completos de las pruebas de carga

<!-- Oscar (WP-4d). Pegar reports/load/anexo_b.md (lo genera collect_load_metrics.py). -->

## Anexo C. Memoria de cálculo FinOps

<!-- Oscar (WP-4d). Pegar reports/load/anexo_c.md (lo genera finops_calc.py) y los comandos
para reproducir. -->
