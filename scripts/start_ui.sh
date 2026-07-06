#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

UI_HOST="${UI_HOST:-0.0.0.0}"
UI_PORT="${UI_PORT:-8503}"
UI_BASE_PATH="${UI_BASE_PATH:-}"

STREAMLIT_ARGS=(
  --server.address="$UI_HOST"
  --server.port="$UI_PORT"
)
if [[ -n "$UI_BASE_PATH" ]]; then
  UI_BASE_PATH="/${UI_BASE_PATH#/}"
  UI_BASE_PATH="${UI_BASE_PATH%/}"
  STREAMLIT_ARGS+=(
    --server.baseUrlPath="$UI_BASE_PATH"
    --server.enableCORS=false
    --server.enableXsrfProtection=false
  )
fi

exec uv run --extra ui streamlit run ui/streamlit_app.py "${STREAMLIT_ARGS[@]}" "$@"
