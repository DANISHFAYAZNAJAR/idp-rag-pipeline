#!/usr/bin/env bash
# Build and smoke-test the Docker demo stack.
set -euo pipefail
cd "$(dirname "$0")/.."

API_PORT="${API_PORT:-8080}"
UI_PORT="${UI_PORT:-8501}"

echo "Building images..."
docker compose build

echo "Starting stack..."
docker compose up -d

echo "Waiting for API health (port ${API_PORT})..."
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${API_PORT}/health" >/dev/null; then
    echo "API is healthy."
    curl -s "http://localhost:${API_PORT}/health"
    echo
    echo "UI:  http://localhost:${UI_PORT}"
    echo "API: http://localhost:${API_PORT}/docs"
    exit 0
  fi
  sleep 5
done

echo "API did not become healthy in time. Logs:"
docker compose logs api --tail=80
exit 1
