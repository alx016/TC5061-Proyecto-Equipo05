#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TC5061-Proyecto-Equipo05
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = uv run python

IMAGE = mibici-p2-api
API_PORT ?= 8000
MLFLOW_HOST ?= 127.0.0.1
CONTAINER_ENGINE ?= $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null || echo podman)
BUILD_FLAGS ?= $(if $(findstring podman,$(CONTAINER_ENGINE)),--format docker,)
MLFLOW_PORT ?= 5001
# docker compose respects depends_on: service_healthy; podman-compose 1.0.x does not
COMPOSE ?= $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo podman-compose)

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format



## Run tests
.PHONY: test
test:
	uv run pytest tests


## Run data integrity and train-test drift checks (DeepChecks isolated environment)
.PHONY: data-checks
data-checks:
	@test -f data/processed/classification_features.parquet || { \
		echo ">>> Falta data/processed/classification_features.parquet."; \
		echo ">>> Genera los datos con 'make data' y las features de P2 con el entry point 'features' de MLproject."; \
		exit 1; \
	}
	uv run scripts/data_checks.py

## Run data integrity and train-test drift checks (alias)
.PHONY: validate-data
validate-data: data-checks


## Run Evidently data, prediction and concept drift monitoring pipeline
.PHONY: drift-report
drift-report:
	uv run python -m tc5061_proyecto_equipo5.monitoring.drift




## Run security checks: bandit, pip-audit (runtime + full) and detect-secrets
.PHONY: security
security:
	uv run bandit -q -r tc5061_proyecto_equipo5
	uv export --quiet --no-default-groups --group api --no-emit-project --format requirements-txt --no-hashes -o /tmp/requirements.txt
	uv run pip-audit --disable-pip --no-deps -r /tmp/requirements.txt
	uv export --quiet --no-emit-project --format requirements-txt --no-hashes -o /tmp/requirements-full.txt
	uv run pip-audit --disable-pip --no-deps -r /tmp/requirements-full.txt
	git ls-files -z | xargs -0 uv run detect-secrets scan --exclude-files '^\.secrets\.baseline$$' > reports/security/detect-secrets.json
	git ls-files -z | xargs -0 uv run detect-secrets-hook --baseline .secrets.baseline --exclude-files '^\.secrets\.baseline$$'

## Run every check: tests, security and data checks
.PHONY: check
check: test security data-checks

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) tc5061_proyecto_equipo5/dataset.py download
	$(PYTHON_INTERPRETER) tc5061_proyecto_equipo5/dataset.py clean
	$(PYTHON_INTERPRETER) tc5061_proyecto_equipo5/dataset.py manifest


## Start the MLflow tracking server (metadata in mlflow.db, artifacts in mlartifacts/)
.PHONY: mlflow-server
mlflow-server:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db \
		--artifacts-destination ./mlartifacts --host $(MLFLOW_HOST) --port 5001

## Export the P2 champion from the Model Registry to models/p2-riesgo/
.PHONY: export-model
export-model:
	uv run python scripts/export_model.py

## Restore the checked-in champion bundle (does not overwrite an existing export)
.PHONY: prepare-model
prepare-model:
	@test ! -e models/p2-riesgo || { echo ">>> models/p2-riesgo/ ya existe; consérvala o elimínala manualmente antes de preparar el bundle."; exit 1; }
	python3 -m zipfile -e models/p2-champion-v3.zip .

## Run the API locally, without a container
.PHONY: api
api:
	uv run uvicorn tc5061_proyecto_equipo5.api.main:app --reload --port $(API_PORT)

## Build the API image (uses models/p2-riesgo/ or unpacks models/p2-champion-v3.zip)
.PHONY: image
image:
	$(CONTAINER_ENGINE) build $(BUILD_FLAGS) -t $(IMAGE) -f Containerfile .

## Run the API container in the background
.PHONY: container
container:
	$(CONTAINER_ENGINE) run -d --rm --name $(IMAGE) -p $(API_PORT):8000 $(IMAGE)

## Check that the running container answers /health and /predict
.PHONY: smoke
smoke:
	curl -fsS --retry 15 --retry-delay 2 --retry-all-errors http://localhost:$(API_PORT)/health
	@echo
	curl -fsS -X POST http://localhost:$(API_PORT)/predict \
		-H "Content-Type: application/json" -d @scripts/smoke_request.json
	@echo

## Start MLflow and the API with Compose, building the images if needed
.PHONY: up
up:
	API_PORT=$(API_PORT) MLFLOW_PORT=$(MLFLOW_PORT) $(COMPOSE) up -d --build

## Stop the Compose stack (the mlflow_data volume is kept)
.PHONY: down
down:
	$(COMPOSE) down

## Follow the logs of the Compose stack
.PHONY: logs
logs:
	$(COMPOSE) logs -f --tail=100

## Check the Compose stack: MLflow /health, API /model-info, then the API smoke test
.PHONY: compose-smoke
compose-smoke:
	curl -fsS --retry 30 --retry-delay 2 --retry-all-errors http://localhost:$(MLFLOW_PORT)/health
	@echo
	curl -fsS --retry 15 --retry-delay 2 --retry-all-errors http://localhost:$(API_PORT)/model-info
	@echo
	@$(MAKE) --no-print-directory smoke API_PORT=$(API_PORT)

## Register the bundled P2 champion in the Compose MLflow (unpacks the bundle if missing; run make up first)
.PHONY: seed-registry
seed-registry:
	@test -f models/p2-riesgo/registry_metadata.json || $(MAKE) --no-print-directory prepare-model
	uv run python scripts/seed_registry.py --tracking-uri http://127.0.0.1:$(MLFLOW_PORT)

## Run the official load scenarios against the API, with docker stats and the Opt 3 benchmark
LOAD_ARGS ?= --with-benchmark
.PHONY: load-test
load-test:
	@test -f models/p2-riesgo/MLmodel || $(MAKE) --no-print-directory prepare-model
	uv run --group perf python scripts/collect_load_metrics.py run \
		--host http://127.0.0.1:$(API_PORT) $(LOAD_ARGS)

## Render anexo_b.md and the FinOps summary (finops_summary.json, anexo_c.md) from reports/load/
FINOPS_ARGS ?=
.PHONY: load-report
load-report:
	uv run python scripts/collect_load_metrics.py report
	uv run python scripts/finops_calc.py $(FINOPS_ARGS)

## Build the 2025 P2 feature table (current and evaluation splits)
.PHONY: features-current
features-current:
	uv run python -m tc5061_proyecto_equipo5.features current

## Build the continuous 2024-2025 P2 table that trains the challenger
.PHONY: features-challenger
features-challenger:
	uv run python -m tc5061_proyecto_equipo5.features challenger

## Train the P2 challenger on the recent window with the champion's hyperparameters
CHALLENGER_HP ?= -P hp_num_leaves=255 -P hp_min_child_samples=100 -P hp_n_estimators=200
.PHONY: train-challenger
train-challenger:
	uv run mlflow run . -e train -P question=p2 -P model=lightgbm -P window=challenger \
		$(CHALLENGER_HP) --env-manager local

## A/B test between the P2 @champion and @challenger
AB_ARGS ?=
.PHONY: ab-test
ab-test:
	uv run python -m tc5061_proyecto_equipo5.monitoring.ab_test $(AB_ARGS)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
