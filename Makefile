.PHONY: install install-precommit-hooks check lint format typecheck test int-test recovery

install:
	uv sync

install-precommit-hooks: install
	uv run pre-commit install

check: lint typecheck test int-test

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest src

int-test:
	uv run pytest tests

ITERS ?= 300

recovery:
	uv run scripts/synthetic_recovery.py --iters $(ITERS)
