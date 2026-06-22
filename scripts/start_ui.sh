#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

UI_HOST="${UI_HOST:-0.0.0.0}"
UI_PORT="${UI_PORT:-8501}"

exec uv run --extra ui streamlit run ui/streamlit_app.py \
  --server.address="$UI_HOST" \
  --server.port="$UI_PORT" \
  "$@"
