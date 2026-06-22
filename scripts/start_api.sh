#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8080}"

exec uv run uvicorn app.main:app --host "$API_HOST" --port "$API_PORT" "$@"
