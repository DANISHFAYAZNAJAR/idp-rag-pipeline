#!/usr/bin/env bash
set -euo pipefail

cd /app

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8080}"
UI_HOST="${UI_HOST:-0.0.0.0}"
UI_PORT="${UI_PORT:-8501}"

log() {
  echo "[entrypoint] $*"
}

wait_for_db() {
  log "waiting for postgres (${POSTGRES_HOST:-postgres})..."
  python scripts/wait_for_db.py
}

run_migrations() {
  log "running alembic migrations..."
  alembic upgrade head
}

case "${1:-api}" in
  api)
    wait_for_db
    run_migrations
    log "starting api on ${API_HOST}:${API_PORT}"
    exec uvicorn app.main:app --host "$API_HOST" --port "$API_PORT"
    ;;
  worker)
    wait_for_db
    log "starting celery worker..."
    exec python -m worker
    ;;
  ui)
    STREAMLIT_ARGS=(
      --server.address="$UI_HOST"
      --server.port="$UI_PORT"
      --browser.gatherUsageStats=false
    )
    UI_BASE_PATH="${UI_BASE_PATH:-}"
    if [[ -n "$UI_BASE_PATH" ]]; then
      # Streamlit expects a leading slash, no trailing slash.
      UI_BASE_PATH="/${UI_BASE_PATH#/}"
      UI_BASE_PATH="${UI_BASE_PATH%/}"
      STREAMLIT_ARGS+=(
        --server.baseUrlPath="$UI_BASE_PATH"
        --server.enableCORS=false
        --server.enableXsrfProtection=false
      )
      log "streamlit base path: ${UI_BASE_PATH}"
    fi
    log "starting streamlit on ${UI_HOST}:${UI_PORT}"
    exec streamlit run ui/streamlit_app.py "${STREAMLIT_ARGS[@]}"
    ;;
  migrate)
    wait_for_db
    run_migrations
    ;;
  debug)
    log "dns check:"
    getent hosts "${POSTGRES_HOST:-postgres}" || true
    log "env:"
    env | grep -E "DATABASE|POSTGRES|REDIS" || true
    wait_for_db
    run_migrations
    ;;
  *)
    exec "$@"
    ;;
esac
