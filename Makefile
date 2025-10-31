SHELL := /bin/bash

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
PYTHON_BIN ?= python3.11

PROJECT := bodocache

.PHONY: setup bootstrap ensure-venv check fmt lint type security test llm-live deps-audit all clean

ensure-venv:
	@set -euo pipefail; \
	if [ ! -x "$(PYTHON)" ]; then \
		echo "Creating virtual environment in $(VENV)"; \
		$(PYTHON_BIN) -m venv $(VENV); \
	elif ! $(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)'; then \
		echo "Recreating virtual environment with $(PYTHON_BIN) (requires >=3.11)"; \
		rm -rf $(VENV); \
		$(PYTHON_BIN) -m venv $(VENV); \
	fi

setup bootstrap: ensure-venv
	@set -euo pipefail; \
	$(PYTHON) -m pip install --upgrade pip
	@set -euo pipefail; \
	if $(PIP) install -e .[dev]; then \
		echo "Installed project with dev extras"; \
	else \
		echo "Dev extras unavailable; falling back to base install"; \
		$(PIP) install -e .; \
	fi
	@if [ -f .pre-commit-config.yaml ]; then \
		echo "Installing pre-commit hooks"; \
		$(VENV)/bin/pre-commit install --install-hooks; \
		$(VENV)/bin/pre-commit install --hook-type commit-msg; \
	fi

fmt:
	@set -euo pipefail; \
	$(VENV)/bin/black --check .

lint:
	@set -euo pipefail; \
	$(VENV)/bin/ruff check .

type:
	@set -euo pipefail; \
	$(VENV)/bin/mypy

security:
	@set -euo pipefail; \
	$(VENV)/bin/bandit -q -r $(PROJECT)
	@set -euo pipefail; \
	$(VENV)/bin/detect-secrets scan --baseline .secrets.baseline .

check: setup fmt lint type security

test: setup
	@set -euo pipefail; \
	$(VENV)/bin/pytest --cov=$(PROJECT) --cov-report=term-missing --cov-report=xml

llm-live: setup
	@if [ -d tests_llm_live ]; then \
		echo "Running LLM live tests"; \
		$(VENV)/bin/pytest tests_llm_live; \
	else \
		echo "llm-live skipped: no tests_llm_live/ directory detected"; \
	fi

deps-audit: setup
	@set -euo pipefail; \
	$(VENV)/bin/pip-audit --skip-editable

all:
	@set -euo pipefail; \
	$(MAKE) check
	@set -euo pipefail; \
	$(MAKE) test
	@set -euo pipefail; \
	$(MAKE) llm-live
	@set -euo pipefail; \
	$(MAKE) deps-audit

clean:
	@rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml
