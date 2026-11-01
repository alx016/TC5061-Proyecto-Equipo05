#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TC5061-Proyecto-Equipo05
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = uv run python

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




## Run security checks: bandit, pip-audit (runtime + full) and detect-secrets
.PHONY: security
security:
	uv run bandit -q -r tc5061_proyecto_equipo5
	uv export --quiet --no-dev --no-emit-project --format requirements-txt --no-hashes -o /tmp/requirements.txt
	uv run pip-audit -r /tmp/requirements.txt
	uv export --quiet --no-emit-project --format requirements-txt --no-hashes -o /tmp/requirements-full.txt
	uv run pip-audit -r /tmp/requirements-full.txt
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
		--artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5001


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
