# Plan de acción · Entrega Final: Proyecto integrador, presentación y video

- **Entrega:** viernes 27 de noviembre de 2026, 23:59 (hora de Ciudad de México). La fecha
  original era el domingo 29; el cambio se avisó el martes 24 y el calendario se comprimió
  (sección 4, decisión F.9).
- **Ventana de trabajo:** lunes 23 a viernes 27 de noviembre (5 días). El viernes es colchón:
  revisión final del PDF, del video y envío en Canvas antes de las 20:00, sin trabajo nuevo.
- **Esfuerzo estimado:** ~24 h de paquetes (~6 h por integrante) + ~2.5 h por integrante de
  ensayo y grabación en conjunto.
- **Base en `main`:** `3a4ebc2` (merge del PR #67, reporte de Entrega 6 en PDF).
- **Requisitos:** [finaldeliverable-requirements.md](finaldeliverable-requirements.md).
- **Planes anteriores:** Entrega 4 y Entrega 5 (rama `docs/plan-entrega5`) · Entrega 6 (rama
  `docs/plan-entrega6-proposal`).

## Registro de decisiones

| # | Decisión | Propuesta |
|:---|:---|:---|
| F.1 | Modelo que documenta el Model Card | **P2 riesgo, champion v3** (`mibici-equipo05-p2-riesgo@champion`), el mismo que sirve la API, monitorea Evidently y compara el A/B. P1 y P3 se mencionan en la presentación como parte del sistema, pero el Model Card es del modelo en producción |
| F.2 | Formato del Model Card | **`reports/model_card.md`**, con las secciones de Mitchell et al. (2019), *Model Cards for Model Reporting*, adaptadas al proyecto (sección 2, punto 1). Se enlaza desde el README y desde la presentación |
| F.3 | Registry en la demo | **Sembrar el champion en el MLflow de Compose** con un script (`make seed-registry`), para que el video muestre el modelo registrado, su linaje y el alias `champion` en `localhost:5001`, y la API cargando desde `models:/…@champion`. El default de `MODEL_URI` **no cambia** (sigue siendo el bundle), así el job `compose-and-smoke` del CI, que espera `model_version = "3"`, sigue igual |
| F.4 | Qué se muestra en vivo y qué no | **En vivo:** `docker compose ps` con los dos servicios saludables, UI de MLflow (Registry y linaje), `curl` a `/model-info` y `/predict` con respuesta real, última corrida verde de GitHub Actions y el HTML de Evidently. **No en vivo:** Locust (5 escenarios tardan minutos) y FinOps, que se muestran con las figuras de `reports/figures/entrega6/` en las diapositivas |
| F.5 | Quién maneja el teclado | **Fernando.** Es dueño de Compose y del Registry, y la demo corre en su máquina. Los otros tres narran por turnos (requisito: video en equipo, sin cortes) |
| F.6 | Duración objetivo | **4:00 a 4:30.** Deja ~30 s de margen bajo el máximo de 5:00 y queda muy por encima del mínimo de 2:00. Si un ensayo pasa de 4:40, se recorta guion, no demo |
| F.7 | Hospedaje del video | **Archivo `TC5061_Proyecto_Final_Equipo05.mp4` en Canvas** y, además, enlace en la última diapositiva (YouTube no listado o Drive con acceso "cualquiera con el enlace"). El enlace se prueba en una ventana de incógnito antes de enviar |
| F.8 | Continuous Training (opcional) | **No se implementa.** No modifica la rúbrica y la semana no da para hacerlo bien. Se presenta en una diapositiva la propuesta de la Entrega 5 (disparador por desempeño, drift como alerta temprana, reentrenamiento mensual) |
| F.9 | Calendario comprimido (entrega el viernes 27) | **Se graba el jueves 26 en la noche y el viernes queda de colchón.** Manuel escribe completos la presentación y el guion a partir del spec (`entrega-final/final-presentation-spec.md`), para que el resto del equipo se concentre en sus paquetes y revisiones; cada narrador solo lee y ensaya sus bloques. Los PR #69 (WP-A), #70 (WP-B) y #68 (WP-C) se integran el miércoles, en ese orden. Si la grabación del jueves falla, se repite el viernes antes de las 18:00 |

---

## 1. Diagnóstico: qué tenemos y qué falta

Revisión del repositorio al 23 de noviembre (`main` en `3a4ebc2`):

| Criterio (peso) | Estado | Evidencia | Brecha |
|:---|:---|:---|:---|
| Estructura y experimentación (20%) | ✅ | Plantilla Cookiecutter, `features.py`, `MLproject` con 6 entry points (`ingest`, `features`, `train`, `search`, `select`, `register`), MLflow Tracking con firma, linaje y tags `git_commit`/`git_dirty` | Ninguna; solo hacerla visible desde el README (WP-C) |
| Modelo y Registry (15%) | ✅ / ⚠️ | `registry.py`, aliases `champion` (= Production) y `challenger`, linaje por versión, documentado en el README | El MLflow de Compose arranca vacío y la API sirve el bundle: en el video no se ve el Registry (WP-B) |
| Despliegue y operación (25%) | ✅ | FastAPI en `Containerfile`, `docker-compose.yml` con healthchecks y límites, CI de 4 jobs en verde en `main` (lint/seguridad, pytest fast/slow, compose y smoke, publicación en GHCR), Locust con 5 escenarios en `reports/load/` | Ninguna |
| Monitoreo, gobernanza y FinOps (20%) | ⚠️ | Evidently (`reports/monitoring/drift_2024_vs_2025.html`), A/B (`ab_summary.json`), FinOps (`reports/TC5061_Operacion_FinOps_Equipo05.pdf`) | **No existe Model Card** (WP-A) |
| Presentación y video (20%) | ❌ | — | Presentación, guion, ensayo y grabación (WP-D y sección 4) |
| Git colaborativo (recomendado) | ✅ | Los 4 integrantes tienen commits y PR en `main` | Mantenerlo: un PR propio y una revisión por integrante |

### Problemas técnicos a resolver

1. **Linaje de un modelo sembrado.** Con `MODEL_URI=models:/…`, la API (`api/model.py`) toma
   el `run_id` del `MLmodel` y lee de ese run `params.threshold_shortage`,
   `params.threshold_saturation`, `params.data_version` y `tags.git_commit`. Si el script de
   WP-B registra el bundle en un run nuevo sin esos campos, `/predict` falla al cargar. El run
   sembrado debe copiar esos valores de `registry_metadata.json` y guardar el run original
   (`7bf71250a91947ba8465d5db48d4bce5`) en un tag, por ejemplo `source_run_id_original`, para no
   perder el linaje real.
2. **Número de versión.** En el MLflow de Compose el champion sembrado será la versión 1, no la
   3. Se guarda `source_registry_version=3` como tag de la versión y se explica en el video en
   una frase ("es la v3 de nuestro Registry de desarrollo, sembrada en el stack"). El CI no se
   toca porque sigue usando el bundle (F.3).
3. **El volumen `mlflow_data` persiste.** `make down` conserva el volumen; correr
   `make seed-registry` dos veces no debe crear versiones duplicadas ni mover el alias por
   accidente. El script tiene que ser idempotente: si ya existe `@champion` con el mismo
   `source_run_id_original`, no hace nada.
4. **Una sola toma.** Cualquier fallo (contenedor que no levanta, puerto ocupado, notificación,
   pestaña equivocada) obliga a repetir todo. La demo se prepara con el stack ya levantado y
   sembrado antes de grabar (sección 4).

---

## 2. Diseño propuesto

```
reports/model_card.md                 <- WP-A
scripts/seed_registry.py              <- WP-B: registra el bundle en el MLflow de Compose
tests/test_seed_registry.py           <- WP-B
Makefile                              <- WP-B: target seed-registry
README.md                             <- WP-C: sección "Proyecto final" al inicio
reports/TC5061_Proyecto_Final_Equipo05.pdf   <- WP-D
references/deliverables/entrega-final/final-video-script.md <- WP-D: guion con tiempos y narradores
```

**Contrato**

1. **Secciones del Model Card** y de dónde sale cada dato:

   | Sección | Contenido | Fuente |
   |:---|:---|:---|
   | Detalles del modelo | Nombre, versión 3, alias `champion`, LightGBM, fecha, responsables, licencia de los datos | `models/p2-riesgo/registry_metadata.json`, README (Integrantes) |
   | Uso previsto | Riesgo de desabasto o saturación por estación con una hora de anticipación, para apoyar el rebalanceo. Usuarios: operación de MiBici | `docs/docs/canvases/p2-riesgo.md` |
   | Fuera de alcance | Lo que el modelo no debe decidir solo (por ejemplo, retirar estaciones), horizontes mayores a una hora, otras ciudades | Canvas P2 |
   | Datos | Periodo 2024, partición temporal sin fuga, proxy de riesgo, estaciones que operaron todo el año | `justificacion-periodo-estudio.md`, `feature-engineering.md` |
   | Métricas | Macro-F1 y PR-AUC en test; umbrales calibrados (0.20 desabasto, 0.95 saturación); recall de desabasto 0.785, **por debajo del piso de 0.85**, dicho explícitamente | Run `7bf71250…`, `ab_summary.json` |
   | Limitaciones y riesgos | Estaciones nuevas o retiradas, capacidad estática del catálogo, etiqueta con una hora de retraso, costo de un desabasto no detectado frente a una falsa alarma | Reporte de Entrega 5 |
   | Monitoreo | Resultado del drift 2024 vs 2025, resultado del A/B y por qué se queda el champion, propuesta de CT | `reports/TC5061_Drift_ABTest_Equipo05.md` |
   | Linaje y reproducción | `source_run_id`, `git_commit`, `data_version`, comandos para reentrenar y registrar | `registry_metadata.json`, README (MLflow Project, Model Registry) |
   | Costo operativo | Costo mensual estimado de la opción elegida (una línea) | `reports/load/finops_summary.json` |

2. **`scripts/seed_registry.py`.** Recibe el tracking URI (default `http://127.0.0.1:5001`) y la
   carpeta del bundle (default `models/p2-riesgo`). Crea un run en el experimento
   `mibici-equipo05` con los params y tags del problema 1, registra el modelo como
   `mibici-equipo05-p2-riesgo`, pone los tags de linaje a la versión y le asigna `@champion`.
   Es idempotente (problema 3). Uso:

   ```bash
   make up                       # stack con el bundle (default)
   make seed-registry            # siembra el champion en el MLflow de Compose
   MODEL_URI=models:/mibici-equipo05-p2-riesgo@champion docker compose up -d api
   curl -fsS http://localhost:8000/model-info   # model_alias = champion, linaje del run
   ```

3. **Sección "Proyecto final" del README.** Una pantalla como máximo, antes de "Preguntas de
   negocio": diagrama de arquitectura (datos → features → MLflow Tracking → Registry → API en
   Compose → CI/CD → monitoreo), tabla rúbrica → archivo o comando que lo demuestra, y enlaces
   al Model Card, la presentación y el video.

4. **Estructura de la presentación** (8 a 10 diapositivas): problema (MiBici, P1/P2/P3) ·
   datos y feature engineering sin fuga · experimentación (MLflow Project y Tracking) ·
   Registry (champion/challenger, linaje) · arquitectura de despliegue (Compose, API) · CI/CD ·
   carga y FinOps · monitoreo (drift y A/B) · gobernanza (Model Card) y CT · decisiones clave y
   aprendizajes. Última diapositiva: enlace al video y al repositorio.

5. **Borrador del guion** (WP-D lo ajusta con los tiempos de los ensayos):

   | Tiempo | Bloque | Narra | En pantalla |
   |:---|:---|:---|:---|
   | 0:00–0:40 | Problema y preguntas de negocio | Manuel | Diapositivas 1–2 |
   | 0:40–1:20 | Datos, features y experimentación | Emilio | Diapositivas 3–4 |
   | 1:20–2:00 | Registry y linaje (demo) | Fernando | UI de MLflow: modelo, versión, alias, tags |
   | 2:00–2:50 | API en Compose (demo) | Fernando | `docker compose ps`, `curl` a `/model-info` y `/predict` |
   | 2:50–3:20 | CI/CD (demo) | Emilio | Última corrida verde en GitHub Actions |
   | 3:20–3:55 | Carga y FinOps | Oscar | Diapositiva de carga y FinOps |
   | 3:55–4:25 | Monitoreo y Model Card | Oscar | HTML de Evidently, diapositiva del Model Card |
   | 4:25–4:40 | Decisiones y cierre | Manuel | Última diapositiva |

---

## 3. Paquetes de trabajo

**Nadie espera a otro para empezar.** WP-D arranca con lo que ya está en `main`; las
diapositivas del Model Card y de la demo del Registry se completan el miércoles.

**Criterio de reparto.** Cada integrante tiene **~6 h**, **un entregable propio en su PR** y
**una revisión**. Cada paquete le toca a quien construyó esa parte en entregas anteriores, para
no gastar la semana aprendiendo código ajeno.

| WP | Responsable | Revisa | Entregable | Horas |
|:---|:---|:---|:---|:---:|
| WP-A Model Card | Oscar | Emilio | `reports/model_card.md` con el contrato de la sección 2, punto 1. Entrenó P2, calibró sus umbrales y diseñó el guardrail de recall | 4 |
| WP-B Registry en vivo | Fernando | Oscar | `scripts/seed_registry.py`, `tests/test_seed_registry.py` (con un tracking local en `tmp_path`: siembra, idempotencia, params que lee la API), target `make seed-registry` y una sección corta en el README (Entrega 6, "Modos de resolución del modelo"). Revisa Oscar porque escribió `export_model.py` y el bundle | 4 |
| WP-B2 Preparación de la demo | Fernando | — | Ensayo técnico de la demo en su máquina, lista de pestañas y comandos (va en el guion de WP-D), y respaldo: si algo falla al grabar, cómo dejar el stack listo en menos de 2 minutos | 1.5 |
| WP-C README final | Emilio | Manuel | Sección "Proyecto final" (sección 2, punto 3) con el diagrama de arquitectura y la tabla rúbrica → evidencia. Tabla de integrantes con lo de esta entrega | 2.5 |
| WP-C2 Verificación en clon limpio | Emilio | — | En su máquina, desde `main`: `make up`, `make compose-smoke` y, después del merge de WP-B, el flujo del punto 2. Reporta en el chat lo que falle. Decide con el equipo si `reports/TC5061_Drift_ABTest_Equipo05.pdf` entra al repo | 1.5 |
| WP-D Presentación y guion | Manuel | Fernando | `TC5061_Proyecto_Final_Equipo05.pdf` (sección 2, punto 4) y `references/deliverables/entrega-final/final-video-script.md` con tiempos, narradores, pantallas y lista de verificación antes de grabar | 5 |
| WP-D2 Integración y envío | Manuel | — | Enlace al video en la última diapositiva, PDF final en `reports/`, prueba del enlace en incógnito, envío en Canvas (repo, PDF, video) | 1 |

**Carga por integrante**

| Integrante | Paquetes | Horas | PR propios | Revisa |
|:---|:---|:---:|:---|:---|
| Oscar | WP-A | 4 + 1 de revisión | Model Card | WP-B |
| Fernando | WP-B, WP-B2 | 5.5 + 0.5 de revisión | Seed del Registry | WP-D |
| Emilio | WP-C, WP-C2 | 4 + 2 de revisión y pruebas | README final | WP-A |
| Manuel | WP-D, WP-D2 | 6 | Presentación y guion completos (F.9) | WP-C |

Además, **todos**: ~2.5 h de ensayo y grabación (sección 4). Si alguien se atrasa, el primero
en liberar horas toma la revisión que falte, no el paquete de otro.

**Un dueño por archivo**, para que los PR en paralelo no choquen:

| Archivo | Dueño |
|:---|:---|
| `reports/model_card.md` | WP-A (Oscar) |
| `scripts/seed_registry.py`, `tests/test_seed_registry.py`, `Makefile` | WP-B (Fernando) |
| `README.md` | WP-C (Emilio). Fernando le pasa el texto de su sección y Emilio lo integra |
| `reports/TC5061_Proyecto_Final_Equipo05.pdf`, `references/deliverables/entrega-final/final-video-script.md` | WP-D (Manuel) |

**Dependencias reales**

- La diapositiva del Model Card necesita WP-A; la de la demo del Registry necesita WP-B.
- WP-C enlaza el Model Card (WP-A) y la presentación (WP-D); puede dejar los enlaces listos y
  verificarlos el miércoles.
- WP-C2 necesita WP-B en `main` para la segunda parte.
- Los ensayos necesitan WP-B y el guion de WP-D. La grabación necesita todo en `main`.

---

## 4. Calendario

| Fase | Día | Trabajo | Hito |
|:---|:---|:---|:---|
| 1. Construir | Lun 23 | Manuel comparte el plan. Arrancan WP-A, WP-B, WP-C y WP-D en paralelo | Plan acordado |
| | Mar 24 | PR abiertos de WP-A (#69), WP-B (#70) y WP-C (#68), con sus revisiones. Manuel sube el spec de la presentación y la estructura del guion | Los 3 PR en revisión |
| 2. Integrar | Mié 25 | **Merge de #69 y #70 en la mañana, luego #68.** Emilio corre WP-C2 en clon limpio. Fernando hace el ensayo técnico de la demo (WP-B2) y toma la captura del Registry. Manuel escribe el guion completo y arma el PDF v1 (F.9) | `main` con todo el contenido; PDF v1 y guion completo |
| 3. Ensayar y grabar | Jue 26 | Manuel comparte el guion en la mañana para que cada narrador lea sus bloques. **Los 4 juntos en la noche:** ensayo 1 con cronómetro, ajustes, ensayo 2 y grabación (una toma continua, las que hagan falta). Exportar `TC5061_Proyecto_Final_Equipo05.mp4`, subirlo y poner el enlace en el PDF | Video grabado |
| 4. Entregar | Vie 27 | Colchón. Ver el video completo, revisar el PDF y los enlaces en incógnito, enviar en Canvas **antes de las 20:00**. Solo si la grabación del jueves falló: repetirla antes de las 18:00 | Entregado |

**Lista de verificación antes de grabar** (WP-D la completa con WP-B2):

- [ ] `main` actualizado en la máquina de Fernando; `make up` y `make seed-registry` corridos;
      API en modo `models:/…@champion`; `make compose-smoke` en verde.
- [ ] Pestañas abiertas en orden: diapositivas, MLflow (`localhost:5001`, modelo P2), terminal
      con los comandos listos, GitHub Actions (última corrida en `main`), HTML de Evidently.
- [ ] Notificaciones apagadas, tamaño de letra de la terminal legible, audio probado de los 4.
- [ ] Cronómetro visible para quien narra.
- [ ] Si algo falla a media toma, se repite desde el principio: el video no se edita.

---

## 5. Checklist de entrega (contra la rúbrica)

- [ ] **Estructura y experimentación (20%)**: el README lleva de cada elemento (Cookiecutter,
      features, MLflow Project, Tracking) al archivo o comando que lo demuestra.
- [ ] **Modelo y Registry (15%)**: champion registrado y versionado, con linaje en sus tags y
      alias `champion` como estado de producción, mostrado en vivo en el video.
- [ ] **Despliegue y operación (25%)**: `docker compose up` desde un clon limpio, CI en verde
      en `main`, resultados de Locust en `reports/load/`.
- [ ] **Monitoreo, gobernanza y FinOps (20%)**: drift, A/B, FinOps y **Model Card** enlazados
      desde el README.
- [ ] **Presentación y video (20%)**: video continuo de 2 a 5 minutos (objetivo 4:00–4:30),
      con la API respondiendo una petición real; la presentación cubre problema, solución y
      decisiones de MLOps.
- [ ] Nombres: `TC5061_Proyecto_Final_Equipo05.pdf` y `TC5061_Proyecto_Final_Equipo05.mp4`.
- [ ] Canvas: enlace al repo, PDF y video (o enlace al video dentro del PDF).
- [ ] `make test`, `make lint` y `make security` en verde sobre `main`.
- [ ] Un PR propio y una revisión por integrante en esta entrega.
