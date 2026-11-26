# Guion · Video de la Entrega Final

Texto narrado completo del video `TC5061_Proyecto_Final_Equipo05.mp4`. Las diapositivas siguen el
formato aserción y evidencia; los datos, fuentes y figuras de cada bloque están en
[final-presentation-spec.md](final-presentation-spec.md).

- **Duración objetivo:** 4:30. Si un ensayo pasa de 4:40, se recorta narración, no demo.
- **Teclado:** Fernando. Los demás narran por turnos.
- **Toma única, sin cortes.** Si algo falla, se repite desde el principio.
- **Presupuesto de palabras:** ~2.3 palabras por segundo (ritmo claro para video). Leer en voz
  alta con cronómetro y anotar el tiempo real en cada bloque.

## Resumen

| # | Tiempo | Narra | Bloque | Palabras máx. | Tiempo real |
|:---:|:---|:---|:---|:---:|:---:|
| 1 | 0:00–0:10 | Manuel | Portada | 23 | ~7 seg|
| 2 | 0:10–0:35 | Manuel | Problema | 58 | ~25 seg|
| 3 | 0:35–1:00 | Emilio | Datos y features | 58 | ~22 seg|
| 4 | 1:00–1:25 | Emilio | Experimentación | 58 | ~25 seg|
| 5 | 1:25–1:55 | Fernando | DEMO Registry | 70 | ~21 seg|
| 6 | 1:55–2:40 | Fernando | DEMO API en Compose | 105 | ~37 seg|
| 7 | 2:40–3:05 | Emilio | DEMO CI/CD | 58 | ~21 seg|
| 8 | 3:05–3:35 | Oscar | Carga y FinOps | 70 | ~31 seg|
| 9 | 3:35–4:00 | Oscar | Monitoreo | 58 | ~25 seg |
| 10 | 4:00–4:15 | Oscar | Gobernanza | 35 | ~11 seg|
| 11 | 4:15–4:30 | Manuel | Decisiones y cierre | 35 | ~15 seg|
| | | | **Total** | **628** | |

---

## 1 · Portada (0:00–0:10, Manuel)

- **Diapositiva:** Anticipar con una hora dónde faltarán bicis o anclajes en MiBici
- **Cubre:** quiénes somos y qué vamos a mostrar.

**Narración:**

> Somos el Equipo 05 de MLOps. Les mostraremos cómo un modelo de MiBici pasa de los datos a
> producción, con demo en vivo.

**Transición:** pasa a la diapositiva 2.

## 2 · Problema (0:10–0:35, Manuel)

- **Diapositiva:** Una estación vacía o llena se descubre cuando ya es tarde
- **Cubre:** rebalanceo con camiones · P1, P2 y P3 · P2 es el modelo en producción · el riesgo es
  raro (~0.67% y ~0.61% de las horas) · predicción a las `h-00:10` para `[h, h+1)`.

**Narración:**

> MiBici rebalancea estaciones con camiones y necesita saber a dónde ir antes de que una se
> vacíe o se llene. Planteamos tres preguntas: demanda, riesgo y grupos; llevamos a producción
> la de riesgo. Diez minutos antes de cada hora, predice si una estación se quedará sin bicis o
> sin anclajes. Para cada estación, pasa en poco más de una de cada cien horas.

**Transición:** cede la palabra a Emilio.

## 3 · Datos y features (0:35–1:00, Emilio)

- **Diapositiva:** Entrenamos con el pasado y evaluamos en el futuro: cero fuga temporal
- **Cubre:** 2024 para entrenar, 2025 para monitorear · partición temporal · features con
  información disponible al predecir · DeepChecks `0.0%` de fuga.

**Narración:**

> Usamos los viajes de 2024: entrenamos con enero a septiembre, ajustamos umbrales en
> octubre y noviembre y probamos en diciembre; 2025 queda para monitorear. Las dieciocho
> variables usan solo lo que se sabe diez minutos antes: flujos recientes, calendario, festivos,
> y estaciones vecinas. DeepChecks confirmó cero solapamiento de fechas entre
> entrenamiento y prueba.

## 4 · Experimentación (1:00–1:25, Emilio)

- **Diapositiva:** Cada experimento es un run reproducible y trazable a su commit y sus datos
- **Cubre:** 6 entry points del `MLproject` · params, métricas, firma y tags de linaje · 7
  candidatos, selección por PR-AUC con guardrail de recall `>= 0.85` · ganó LightGBM.

**Narración:**

> Todo corre como MLflow Project, con seis entry points. Cada run guarda parámetros, métricas,
> firma del modelo, commit y versión de datos. Comparamos siete candidatos, desde líneas base
> simples hasta LightGBM. Gana el mejor PR-AUC de validación que mantenga el recall de desabasto
> en al menos 85%, y solo compiten runs de código versionado. Ganó LightGBM.

**Transición:** cede la palabra a Fernando; Fernando cambia a la pestaña de MLflow.

## 5 · DEMO Registry (1:25–1:55, Fernando)

- **Diapositiva (PDF):** El champion v3 vive en el Registry con su linaje completo
- **Cubre:** aliases `champion`/`challenger` en lugar de stages · linaje (run, commit, datos) ·
  aparece como versión 1 porque el MLflow de Compose arranca vacío; el tag
  `source_registry_version` dice que es la v3 de desarrollo.

**En pantalla:**

1. `localhost:5001` → **Models** → `mibici-equipo05-p2-riesgo`.
2. Versión con alias `champion` → tags de linaje.

**Narración:**

> Este es nuestro Model Registry. Usamos aliases en lugar de stages: champion es lo que sirve
> producción y challenger, el candidato; promover un modelo es solo mover el alias. Aquí aparece
> como versión uno porque este servidor arrancó vacío, pero sus tags dicen que es la versión tres
> de desarrollo. También guardan el commit, la versión de datos y el run de origen.

**Transición:** Fernando cambia a la terminal.

## 6 · DEMO API en Compose (1:55–2:40, Fernando)

- **Diapositiva (PDF):** Un comando levanta API y MLflow; la API responde en vivo
- **Cubre:** dos servicios con healthchecks y límites · `/model-info` confirma modelo, versión y
  umbrales (0.20 desabasto, 0.95 saturación) ·
  leer en voz alta un resultado de `/predict` · latencia mediana 38 ms (1 estación) y 120 ms
  (365 estaciones).

**En pantalla:**

```bash
docker compose ps
curl -fsS http://localhost:8000/model-info
curl -fsS -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/smoke_request.json
```

**Narración:**

> Con make up levantamos dos servicios, MLflow y la API, en una red privada, cada uno con
> healthcheck y límites de CPU y memoria; los dos están sanos. Model-info confirma que la API
> sirve la versión uno con alias champion, cargada desde el Registry, y sus umbrales: 0.20 para
> desabasto y 0.95 para saturación. Ahora pedimos una predicción para la estación cien: la
> clasifica como balanceada, con más de 99% de probabilidad. Con diez usuarios, la mediana es de
> 38 milisegundos por estación y 120 con las 365.

**Transición:** cede la palabra a Emilio; Fernando cambia a GitHub Actions.

## 7 · DEMO CI/CD (2:40–3:05, Emilio)

- **Diapositiva (PDF):** Ningún cambio llega a `main` sin pasar tres checks; solo `main` publica la imagen
- **Cubre:** tres jobs en paralelo (lint y seguridad, pruebas fast/slow, Compose y smoke) ·
  publicación en GHCR solo si pasan los tres y solo en `main` · 536 pruebas · ~10 min por corrida.

**En pantalla:** GitHub → **Actions** → última corrida verde en `main` → vista de los 4 jobs.

**Narración:**

> Cada push o pull request a main dispara cuatro jobs. Tres corren en paralelo: lint y seguridad,
> 536 pruebas rápidas y lentas, y un smoke test que levanta el stack de Compose. Solo si los tres
> pasan, y solo en main, se publica la imagen en GitHub Container Registry. Esta corrida tardó
> diez minutos.

**Transición:** cede la palabra a Oscar; Fernando regresa a las diapositivas (8).

## 8 · Carga y FinOps (3:05–3:35, Oscar)

- **Diapositiva:** Con la demanda real, el costo es la instancia ociosa: scale-to-zero lo baja casi
  a cero
- **Cubre:** cero errores en carga · 14.26 req/s con 10 usuarios, saturación cerca de 24 req/s ·
  lote de 365 estaciones = 78.56× · $0.01258 vs $59.99 por millón · de $15.98 a menos de $0.01
  al mes.

**Narración:**

> Con Locust no hubo un solo error. Con diez usuarios la API sostiene catorce peticiones por
> segundo; con cincuenta se satura cerca de veinticuatro. Enviar las 365 estaciones juntas da 78
> veces más predicciones por segundo. Pero la demanda real es un lote por hora, así que una
> instancia siempre encendida pasa ociosa: sesenta dólares por millón de predicciones. Con
> scale-to-zero, el gasto mensual baja de dieciséis dólares a menos de un centavo.

## 9 · Monitoreo (3:35–4:00, Oscar)

- **Diapositiva:** Los datos no cambiaron, pero el recall de desabasto ya cayó debajo del piso
- **Cubre:** 0 de 18 variables con drift (2024 vs 2025) · A/B oct–dic 2025: challenger +0.029
  PR-AUC, recall 0.836 vs 0.785, cumple macro-F1 y alertas; solo falla el piso de 0.85 →
  se queda el champion · CT por desempeño.

**En pantalla (opcional, si el tiempo alcanza):** HTML de Evidently
(`reports/monitoring/drift_2024_vs_2025.html`).

**Narración:**

> Con Evidently, comparamos 2025 contra 2024: ninguna de las dieciocho variables tuvo drift, ni
> la etiqueta ni la predicción. Aun así, el recall de desabasto cayó a 79%. En el A/B, un
> challenger reentrenado subió a 84%, pero la regla, fijada antes, exige 85%: se queda el
> champion. Por eso proponemos reentrenar por desempeño, no solo por drift.

## 10 · Gobernanza (4:00–4:15, Oscar)

- **Diapositiva:** El Model Card dice lo que el modelo no hace bien
- **Cubre:** uso previsto y fuera de alcance · recall 0.785 con prevalencia de 0.674% a 0.475% ·
  precisión ~4%.

**Narración:**

> Y lo dejamos por escrito en el Model Card: por cada desabasto real emite unas veinte alertas,
> casi no acierta en las diez estaciones nuevas y no sirve para decidir cerrar o mover
> estaciones.

**Transición:** cede la palabra a Manuel.

## 11 · Decisiones y cierre (4:15–4:30, Manuel)

- **Diapositiva:** Cinco decisiones de MLOps definieron el sistema
- **Cubre:** partición sin fuga · aliases en lugar de stages · bundle inmutable con Registry en
  vivo · inferencia síncrona por lotes · reentrenar por desempeño · cierre.

**Narración:**

> Cinco decisiones definieron el sistema: separar los datos por tiempo, promover con aliases,
> servir un bundle inmutable o el Registry, predecir por lotes sin cola y reentrenar por
> desempeño, no solo por drift. Gracias.

---

## Antes de grabar

- [ ] `main` actualizado en la máquina de Fernando; `make up` y `make seed-registry` corridos;
      API en modo `models:/mibici-equipo05-p2-riesgo@champion`; `make compose-smoke` en verde.
- [ ] Pestañas abiertas en orden: diapositivas, MLflow (`localhost:5001`, modelo P2), terminal
      con los comandos del bloque 6 listos, GitHub Actions (última corrida en `main`), HTML de
      Evidently.
- [ ] Notificaciones apagadas, letra de la terminal legible, audio probado de los 4.
- [ ] Cronómetro visible para quien narra.

## Si algo falla a media toma

Se detiene la grabación y se repite desde el principio. Si el stack no responde, Fernando lo
deja listo con:

```bash
make down && make up && make seed-registry
MODEL_URI=models:/mibici-equipo05-p2-riesgo@champion docker compose up -d api
make compose-smoke
```

## Registro de ensayos

| Ensayo | Fecha | Duración | Ajustes |
|:---:|:---|:---:|:---|
| 1 | | | |
| 2 | | | |
