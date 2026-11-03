# Feature: Despliegue de Modelos MiBici con FastAPI y Contenedores OCI

- **Feature Name**: `model-serving-fastapi-container`
- **Objective**: Implementar el servicio REST de inferencia con FastAPI empaquetado en contenedor OCI (Docker/Podman), sirviendo los modelos registrados en MLflow (`p1`, `p2`, `p3`) bajo esquema Champion/Challenger con validación Pydantic robusta y fallback local.
- **Authorized Scope**: API FastAPI, esquemas Pydantic v2, cargador de registro híbrido, exportador de bundle, Containerfile/Dockerfile, Makefile, tests y documentación en README.
- **Delivery Strategy**: `ask-on-risk` (Forecast: ~500-600 LOC distribuidas en componentes modulares).
- **TDD Mode**: Disabled (functional checks con PyTest `TestClient` y smoke tests de contenedor).

## Checklist de Tareas

- [x] **TASK-01: Actualización de dependencias y archivo congelado** (WP-D)
- [x] **TASK-02: Cargador de Model Registry y exportador de bundle local** (WP-0)
- [x] **TASK-03: Esquemas Pydantic v2 y normalización de variables de dominio** (WP-2 / Ticket 04)
  - *Evidence*: `tc5061_proyecto_equipo5/api/schemas.py`, `api/schemas.py`, 70 tests pasados en `tests/test_schemas.py`.
- [x] **TASK-04: API FastAPI y rutas de inferencia y observabilidad** (WP-1 / Ticket 03)
- [x] **TASK-05: Suite de pruebas automatizadas de la API y esquemas** (WP-2 / Ticket 04)
  - *Evidence*: 93 tests pasados en `pytest tests/test_schemas.py tests/test_api.py`, 368 tests de regresión en verde.
- [ ] **TASK-06: Contenerización OCI y automatización con Makefile** (WP-3 / Ticket 05)
- [ ] **TASK-07: Ejecución en contenedor y generación de evidencia** (WP-3 / Ticket 05)
- [ ] **TASK-08: Documentación completa en README.md** (WP-4 / Ticket 06)
