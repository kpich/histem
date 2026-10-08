.PHONY: install install-precommit-hooks check lint format typecheck test int-test

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
