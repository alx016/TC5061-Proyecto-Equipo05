# TC5061 — Machine Learning Operations (MLOps)

<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-val">3,803,799</div>
    <div class="kpi-label">Viajes Analizados</div>
    <div class="kpi-sub">Año Canónico 2024 (100% auditado)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">360</div>
    <div class="kpi-label">Estaciones Activas</div>
    <div class="kpi-sub">Red Metropolitana de Guadalajara</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">0.00%</div>
    <div class="kpi-label">Fuga Temporal</div>
    <div class="kpi-sub">Split temporal estricto (OOT Dic 2024)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">3 Modelos</div>
    <div class="kpi-label">Machine Learning Canvases</div>
    <div class="kpi-sub">P1 Regresión · P2 Riesgo · P3 Clustering</div>
  </div>
</div>

---

## Arquitectura de Casos de Uso (ML Canvases)

<div class="grid cards" markdown>

- [**P1 — Regresión de Demanda Horaria**](canvases/p1-demanda.md)

    Predicción de viajes originados por estación y hora ($s \times h$) con 1 hora de anticipación.

  - **Algoritmo:** LightGBM Regressor con función de pérdida Poisson / Tweedie.
  - **Features:** Lags históricos ($t-1, t-2, t-24, t-168$), agregaciones móviles y calendario cíclico.
  - **Métrica Primaria:** WAPE $\le 18\%$ y RMSE en estaciones críticas.

- [**P2 — Clasificación de Riesgo Operativo**](canvases/p2-riesgo.md)

    Detección temprana de eventos de desabasto (estación vacía) o saturación (estación llena) a nivel horario.

  - **Proxy de Estado:** Balance de flujo neto normalizado vs. capacidad física ($\Delta = \frac{In - Out}{C_s}$).
  - **Umbral Operativo:** $|\Delta| \ge 0.40$ (riesgo inminente con $40\%$ de rotación en 1h).
  - **Métrica Primaria:** Macro F1 $\ge 0.72$; Recall $\ge 85\%$ en desabasto y PR-AUC $\ge 0.82$ como métricas de detección y separación.

- [**P3 — Clustering Funcional de Estaciones**](canvases/p3-clustering.md)

    Agrupación no supervisada de estaciones según su firma horaria de demanda (perfil 24h normalizado).

  - **Algoritmo:** K-Means sobre representaciones continuas normalizadas ($L_1$).
  - **Tipologías:** Residencial (origen matutino), Atractora (destino matutino) y Equilibrada.
  - **Métrica Primaria:** Silhouette Score $\ge 0.45$, estabilidad temporal mensual.

</div>

---

## Documentación Técnica del Sistema

- **[Justificación Técnica del Periodo de Estudio](justificacion-periodo-estudio.md):** Auditoría longitudinal de 11 años (2014–2026), muestreo testigo (*Anchor Sampling*), fundamentación del año canónico 2024, validación empírica del proxy P2 y protocolo de partición temporal sin fuga de datos (*zero data leakage*).
- **[Especificación Completa de ML Canvases](canvases/index.md):** Matriz comparativa, flujos de valor, granularidad, formulación matemática de variables objetivo y métricas de negocio para P1, P2 y P3.
- **[Guía de Inicio y Reproducibilidad](getting-started.md):** Instrucciones de clonación, entorno virtual con `uv`, ejecución de pruebas automatizadas y compilación del portal.

---

## Organización del Repositorio

La arquitectura sigue el estándar internacional de **Cookiecutter Data Science (CCDS)**:

```
├── data/              <- Datos locales blindados en .gitignore (raw, interim, processed)
├── docs/              <- Documentación construida con MkDocs Material y MathJax
├── mibici/            <- Paquete modular en Python (config, dataset, features, modeling)
├── models/            <- Modelos entrenados y serializados (excluidos de Git)
├── notebooks/         <- Notebooks reproducibles de exploración (1.0-exploracion-inicial.ipynb)
├── references/        <- Machine Learning Canvases v1.2 oficiales y manuales de datos
├── reports/           <- Reportes técnicos generados y figuras estáticas (reports/figures/)
└── tests/             <- Suite automatizada de pruebas unitarias con pytest
```
