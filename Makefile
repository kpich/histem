.PHONY: install install-precommit-hooks check lint format typecheck test int-test norman tb tb-all

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

norman:
	uv run scripts/norman_fit.py

N ?= 1
tb:  # the N most recent runs
	uv run tensorboard --logdir_spec "$$(ls -dt ../runs/2*/ | head -n $(N) | xargs -n1 basename | sed 's|.*|&:../runs/&|' | paste -sd, -)"

tb-all:
	uv run tensorboard --logdir ../runs
