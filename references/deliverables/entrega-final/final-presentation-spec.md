# Spec · Presentación y guion de la Entrega Final

Formato **aserción y evidencia**: el título de cada diapositiva es una oración con la idea
principal y el cuerpo es una sola evidencia visual. Todo lo demás va en el guion
(`references/deliverables/entrega-final/final-video-script.md`). Leyendo solo los títulos de corrido, el PDF
debe contar el argumento completo.

- **Duración objetivo:** 4:30 (plan, F.6). Si un ensayo pasa de 4:40, se recorta narración, no demo.
- **Teclado:** Fernando (F.5). Las demos son pantallas en vivo; su diapositiva existe para el PDF
  y como respaldo si algo falla.
- **Reparto de voz:** Manuel ~0:55 · Emilio ~1:15 · Fernando ~1:15 · Oscar ~1:05.

## Resumen

| # | Tiempo | Narra | Título (aserción) | Evidencia |
|:---:|:---|:---|:---|:---|
| 1 | 0:00–0:10 | Manuel | Anticipar con una hora dónde faltarán bicis o anclajes en MiBici | Portada |
| 2 | 0:10–0:35 | Manuel | Una estación vacía o llena se descubre cuando ya es tarde | Figura del proxy de riesgo |
| 3 | 0:35–1:00 | Emilio | Entrenamos con el pasado y evaluamos en el futuro: cero fuga temporal | Partición temporal |
| 4 | 1:00–1:25 | Emilio | Cada experimento es un run reproducible y trazable a su commit y sus datos | Runs de P2 en MLflow |
| 5 | 1:25–1:55 | Fernando | **DEMO** · El champion v3 vive en el Registry con su linaje completo | MLflow UI en vivo |
| 6 | 1:55–2:40 | Fernando | **DEMO** · Un comando levanta API y MLflow; la API responde en vivo | Terminal en vivo |
| 7 | 2:40–3:05 | Emilio | **DEMO** · Ningún cambio llega a `main` sin pasar tres checks; solo `main` publica la imagen | GitHub Actions en vivo |
| 8 | 3:05–3:35 | Oscar | Con la demanda real, el costo es la instancia ociosa: scale-to-zero lo baja casi a cero | Comparativa FinOps |
| 9 | 3:35–4:00 | Oscar | Los datos no cambiaron, pero el recall de desabasto ya cayó debajo del piso | Drift + A/B |
| 10 | 4:00–4:15 | Oscar | El Model Card dice lo que el modelo no hace bien | Captura del Model Card |
| 11 | 4:15–4:30 | Manuel | Cinco decisiones de MLOps definieron el sistema | Tabla de decisiones + enlaces |

---

## Diapositivas

### 1 · Portada (0:00–0:10, Manuel)

- **Título:** Anticipar con una hora dónde faltarán bicis o anclajes en MiBici
- **Evidencia:** nombre del curso, Equipo 05, los 4 integrantes. Sin figura o con una foto/mapa
  neutro (sin logotipos de MiBici).
- **La narración cubre:** quiénes son y qué van a mostrar (problema, solución, demo, decisiones).

### 2 · Problema (0:10–0:35, Manuel)

- **Título:** Una estación vacía o llena se descubre cuando ya es tarde
- **Evidencia:** `reports/figures/04_p2_proxy_riesgo_distribucion.png`
- **La narración cubre:**
  - MiBici (Área Metropolitana de Guadalajara) rebalancea estaciones con camiones; necesita saber
    antes dónde ir.
  - Tres preguntas: P1 demanda por estación-hora, **P2 riesgo de desabasto o saturación**, P3
    grupos de estaciones. P2 es el modelo que llevamos a producción y que verán en la demo.
  - El riesgo es raro: ~0.67% de las horas son desabasto y ~0.61% saturación en 2024
    (Model Card, PR #69). Por eso la métrica no es accuracy.
  - Predicción a las `h-00:10` para la hora `[h, h+1)` (README, Preguntas de negocio).

### 3 · Datos y features (0:35–1:00, Emilio)

- **Título:** Entrenamos con el pasado y evaluamos en el futuro: cero fuga temporal
- **Evidencia:** `reports/figures/05_particion_temporal_zero_leakage.png`
- **La narración cubre:**
  - Datos abiertos de viajes; 2024 para entrenar (justificación en
    `reports/justificacion_periodo_estudio.md`), 2025 reservado para monitoreo.
  - Partición temporal (train ene–sep, validación y test al final del año) y features solo con
    información disponible al momento de predecir (lags, calendario, festivos, capacidad).
  - DeepChecks: `0.0%` de fuga temporal entre train y test (`reports/deepchecks/hallazgos.md`).

### 4 · Experimentación (1:00–1:25, Emilio)

- **Título:** Cada experimento es un run reproducible y trazable a su commit y sus datos
- **Evidencia:** `reports/figures/entrega2/01-p2-runs.png` (alternativa:
  `07-p2-search-parent.png`)
- **La narración cubre:**
  - `MLproject` con 6 entry points: `ingest`, `features`, `train`, `search`, `select`, `register`.
  - Cada run guarda params, métricas, firma del modelo y tags `git_commit`, `git_dirty` y
    `data_version`.
  - P2 comparó 7 candidatos (de persistencia y clase mayoritaria a LightGBM, HistGradientBoosting
    y Random Forest; README, Etapas de experimentación); se elige por PR-AUC de validación con un
    guardrail: recall de desabasto `>= 0.85`. Ganó LightGBM.

### 5 · DEMO Registry (1:25–1:55, Fernando)

- **Título:** El champion v3 vive en el Registry con su linaje completo
- **Evidencia (PDF):** captura del modelo sembrado en el MLflow de Compose (tomarla el miércoles,
  después del merge del PR #70). Mientras: `reports/figures/entrega2/03-p2-selected-run-tags.png`
- **En pantalla:** `localhost:5001` → Models → `mibici-equipo05-p2-riesgo` → versión con alias
  `champion` → tags de linaje.
- **La narración cubre:**
  - Aliases en lugar de stages (deprecados desde MLflow 2.9): `champion` = producción,
    `challenger` = candidato. Una versión nueva entra como `challenger`; promover es mover el alias.
  - Linaje en los tags de la versión: run de origen, commit, versión de datos.
  - Aparece como versión 1 porque el MLflow de Compose arranca vacío; el tag
    `source_registry_version` dice que es la v3 de desarrollo (plan, problema 2).

### 6 · DEMO API en Compose (1:55–2:40, Fernando)

- **Título:** Un comando levanta API y MLflow; la API responde en vivo
- **Evidencia (PDF):** diagrama de arquitectura (el mismo del README, PR #68)
- **En pantalla:** `docker compose ps` (2 servicios `healthy`) →
  `curl -fsS http://localhost:8000/model-info` → `curl … /predict -d @scripts/smoke_request.json`
- **La narración cubre:**
  - Dos servicios en una red privada, con healthchecks y límites de CPU/memoria
    (`docker-compose.yml`).
  - `/model-info` confirma qué modelo y versión se sirven (cargado desde `models:/…@champion`) y
    sus umbrales: 0.20 desabasto, 0.95 saturación (son params del run, no tags de la versión, por
    eso se muestran aquí y no en la demo 5).
  - `/predict` devuelve la clase y las probabilidades por estación; leer en voz alta un resultado.
  - Latencia mediana con 10 usuarios: 38 ms por estación, 120 ms con 365 estaciones
    (`reports/load/finops_summary.json`, `nominal_10u`).

### 7 · DEMO CI/CD (2:40–3:05, Emilio)

- **Título:** Ningún cambio llega a `main` sin pasar tres checks; solo `main` publica la imagen
- **Evidencia (PDF):** `reports/figures/entrega6/ci_run_verde1.png`
- **En pantalla:** GitHub → Actions → última corrida verde en `main`.
- **La narración cubre:**
  - Tres jobs en paralelo: lint y seguridad (ruff, bandit, pip-audit, detect-secrets), pruebas
    fast/slow (536: 531 rápidas y 5 lentas, corrida del merge de #70) y Compose con smoke test.
    Los tres son required checks en la protección de `main` (desde el 2026-11-25).
  - La publicación de la imagen en GHCR (`ghcr.io/alx016/mibici-p2-api`) espera a los tres y
    solo corre en push a `main` (`ci.yml:137-138`).
  - La corrida tarda ~7 a 12 min (últimas seis en `main`); la de #70 tardó 10 min.

### 8 · Carga y FinOps (3:05–3:35, Oscar)

- **Título:** Con la demanda real, el costo es la instancia ociosa: scale-to-zero lo baja casi
  a cero
- **Evidencia:** `reports/figures/entrega6/04_finops_comparativa_optimizaciones.png`
- **La narración cubre** (fuente: `reports/TC5061_Operacion_FinOps_Equipo05.md`, resumen
  ejecutivo):
  - Locust: cero errores; 14.26 req/s con 10 usuarios y CPU bajo 80%; con 50 usuarios la vCPU
    satura cerca de 24 req/s.
  - Enviar las 365 estaciones en una petición da 78.56× más predicciones por segundo.
  - En una `t3.small`: $0.01258 por millón de predicciones en el pico, pero $59.99 con la demanda
    real (365 predicciones por hora), porque la instancia pasa ociosa.
  - Recomendación: lote horario con scale-to-zero y sin MLflow en inferencia, de $15.98 a menos
    de $0.01 al mes.

### 9 · Monitoreo (3:35–4:00, Oscar)

- **Título:** Los datos no cambiaron, pero el recall de desabasto ya cayó debajo del piso
- **Evidencia:** `reports/figures/entrega5/evidently_drift_overview.png` y, al lado, un número
  grande: recall de desabasto `0.785` vs piso `0.85`
- **La narración cubre** (fuentes: `reports/monitoring/drift_summary.json`,
  `reports/monitoring/ab_summary.json`):
  - Drift: ene–sep 2024 vs ene–sep 2025, muestra de 100,000 filas con semilla 42: 0 de 18
    variables con drift, ni en la etiqueta ni en la predicción.
  - A/B en oct–dic 2025 (805,920 filas, 93 días, que ninguna versión vio): el challenger mejora
    el PR-AUC en +0.029 (IC 95%: 0.026 a 0.032) y el recall de desabasto (0.836 vs 0.785), y
    cumple macro-F1 y el límite de alertas, pero su recall no llega al piso de 0.85, la única
    condición que falla (`decision.checks` en `ab_summary.json`). Por la regla fijada el 9 de
    noviembre, antes de entrenar el challenger, se queda el champion.
  - Continuous Training propuesto: disparador por desempeño (la etiqueta llega una hora
    después), drift como alerta temprana, reentrenamiento mensual.

### 10 · Gobernanza (4:00–4:15, Oscar)

- **Título:** El Model Card dice lo que el modelo no hace bien
- **Evidencia:** captura de la sección de limitaciones de `reports/model_card.md` (después del
  merge del PR #69)
- **La narración cubre:**
  - Uso previsto (apoyar el rebalanceo) y fuera de alcance (decidir sin una persona, horizontes
    mayores a una hora).
  - Limitación principal: recall de desabasto 0.785 en 2025; la prevalencia bajó de 0.674% a
    0.475%, con umbrales fijos de 2024. Precisión baja (~4%): detecta casi todo, con muchas falsas
    alarmas.

### 11 · Decisiones y cierre (4:15–4:30, Manuel)

- **Título:** Cinco decisiones de MLOps definieron el sistema
- **Evidencia:** tabla de 5 filas, máximo 6 palabras cada una, más los enlaces al repo y al video
- **Las cinco decisiones** (de los registros de decisiones de cada plan):
  1. Partición temporal sin fuga.
  2. Aliases `champion`/`challenger` en lugar de stages.
  3. Bundle inmutable por defecto, Registry en vivo como opción.
  4. Inferencia síncrona por lotes, sin cola (Redis + RQ descartado: 365 predicciones por
     hora caben en un lote, `reports/TC5061_Operacion_FinOps_Equipo05.md:66`).
  5. Reentrenar por desempeño, no solo por drift.
- **La narración cubre:** una frase por decisión como máximo, y el cierre.

---

## Pendientes antes de armar el deck

- [ ] **Run ID del champion.** El bundle que sirve la API dice
      `source_run_id = 7bf71250…`, pero `ab_summary.json` registra el champion como
      `7c0d9747…` (el A/B usó la reproducción del champion en un solo Registry, Entrega 5).
      En la diapositiva 5 se muestra el del bundle. Confirmar con Fernando y Oscar que el Model
      Card y el seed usen el mismo.
- [ ] **Número de escenarios de carga.** El reporte de FinOps dice 6 y `finops_summary.json`
      tiene 5. En la narración, decir "cero errores en todas las pruebas de carga" sin número, o
      confirmarlo con Oscar.
- [ ] Capturas de las diapositivas 5 y 10 cuando se integren los PR #70 y #69.
- [ ] Los porcentajes de prevalencia vienen del Model Card del PR #69, que todavía está en
      revisión.
