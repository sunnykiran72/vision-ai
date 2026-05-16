# Glamify AI

Single-container GPU API for the wardrobe and try-on pipeline.

## Principles

- One container
- One service boundary
- Focused on the 4 GPU-backed APIs only
- Simple app-first structure
- `uv`-first Python workflow
- Reproducible local setup

## Layout

```text
ai/
  app/
  scripts/
  tests/
  docs/
  pyproject.toml
  Makefile
  uv.lock
```

## Local development

```bash
./scripts/bootstrap.sh
uv run uvicorn app.main:app --reload
```

## Standard workflow

```bash
make setup
make dev
make test
make lint
```

## Tooling choice

- `uv` manages the virtual environment and dependency sync
- `uv run` executes project commands inside the managed environment
- `uvicorn` remains the ASGI server for FastAPI

## First implementation track

1. Define API contracts
2. Implement `/v1/user_validation`
3. Implement `/v1/wardrobe`
4. Implement `/v1/tryon`
5. Implement `/v1/upscale`
