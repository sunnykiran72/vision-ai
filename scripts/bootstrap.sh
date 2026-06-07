#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
WORKERS="${UVICORN_WORKERS:-1}"
SYNC_DEPS="${SYNC_DEPS:-1}"
RUN_SERVER="${RUN_SERVER:-1}"
RELOAD="${RELOAD:-0}"

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap.sh [options]

Options:
  --host HOST         Bind host. Default: 0.0.0.0
  --port PORT         Bind port. Default: 8000
  --reload            Start uvicorn with --reload. Intended for local dev only.
  --no-sync           Skip uv sync.
  --check-only        Sync and validate config, but do not start the API server.
  -h, --help          Show this help.

Environment:
  SYNC_DEPS=0                  Same as --no-sync.
  RUN_SERVER=0                 Same as --check-only.
  UVICORN_WORKERS=1            Must stay 1 for the shared in-process GPU queue.
  Startup always validates config and loads resident runtimes.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:?--host requires a value}"
      shift 2
      ;;
    --port)
      PORT="${2:?--port requires a value}"
      shift 2
      ;;
    --reload)
      RELOAD=1
      shift
      ;;
    --no-sync)
      SYNC_DEPS=0
      shift
      ;;
    --check-only)
      RUN_SERVER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ "$WORKERS" != "1" ]]; then
  echo "UVICORN_WORKERS must be 1 for GPU runtime isolation; got: $WORKERS" >&2
  echo "Multiple workers create separate model instances and independent queues." >&2
  exit 1
fi

if [[ "$SYNC_DEPS" == "1" ]]; then
  uv sync --extra dev
fi

uv run python - <<'PY'
from app.config import get_settings
from app.runtime.warmup import validate_required_service_config

settings = get_settings()
validate_required_service_config(settings)

print("Configuration validation passed.")
print(f"  app_env={settings.app_env}")
print(f"  qwen_image_edit_model_path={settings.qwen_image_edit_model_path}")
print(f"  ai_toolkit_root={settings.ai_toolkit_root}")
print(f"  wardrobe_lora_top_path={settings.wardrobe_lora_top_path}")
print(f"  wardrobe_lora_bottom_path={settings.wardrobe_lora_bottom_path}")
print(f"  wardrobe_lora_dress_path={settings.wardrobe_lora_dress_path}")
print(f"  minicpm_model_path={settings.minicpm_model_path}")
print(f"  azure_wardrobe_input_container={settings.azure_wardrobe_input_container}")
print(f"  azure_wardrobe_output_container={settings.azure_wardrobe_output_container}")
print(f"  system_queue_max_size={settings.system_queue_max_size}")
print(f"  glamify_api_base_url={settings.glamify_api_base_url}")
PY

if [[ "$RUN_SERVER" != "1" ]]; then
  exit 0
fi

export UVICORN_WORKERS=1

CMD=(
  uv run uvicorn app.main:app
  --host "$HOST"
  --port "$PORT"
  --workers 1
  --log-level "$LOG_LEVEL"
)

if [[ "$RELOAD" == "1" ]]; then
  CMD+=(--reload)
fi

echo "Starting Glamify AI API on ${HOST}:${PORT} with one worker."
echo "Wardrobe GPU execution is serialized in-process; do not add workers for this pod."
exec "${CMD[@]}"
