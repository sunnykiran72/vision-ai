.PHONY: setup install dev lint test

setup:
	uv sync --extra dev

install: setup

dev:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8011

lint:
	uv run ruff check .
	uv run mypy app

test:
	uv run pytest
