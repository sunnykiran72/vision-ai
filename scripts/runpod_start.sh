#!/usr/bin/env bash
# Start glamify-image-ai on a RunPod GPU pod.
#
# Expects:
#   - uv-managed venv at /workspace/.venvs/glamify-image-ai
#   - app checkout at /workspace/glamify-image-ai
#   - .env in the app checkout
set -euo pipefail

APP_DIR="${APP_DIR:-/workspace/glamify-image-ai}"
VENV_DIR="${VENV_DIR:-/workspace/.venvs/glamify-image-ai}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Python not found at $VENV_DIR/bin/python" >&2
  exit 1
fi

if [[ ! -f "$APP_DIR/.env" ]]; then
  echo "Missing $APP_DIR/.env. Copy .env.runpod.example and fill required secrets/paths." >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export VLLM_USE_FLASHINFER="${VLLM_USE_FLASHINFER:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PORT

cd "$APP_DIR"
uvicorn_args=(
  app.main:app
  --host "$HOST"
  --port "$PORT"
)

if [[ "$WORKERS" != "1" ]]; then
  uvicorn_args+=(--workers "$WORKERS")
fi

exec "$VENV_DIR/bin/python" -m uvicorn "${uvicorn_args[@]}"
