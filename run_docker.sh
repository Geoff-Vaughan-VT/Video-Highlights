#!/usr/bin/env bash
# CPU stack: API + Studio UI + queue worker. Usage: ./run_docker.sh [--logs]
set -euo pipefail
cd "$(dirname "$0")"
[[ -f .env ]] || { cp .env.example .env; echo "created .env from .env.example (edit VH_MEDIA_DIR)"; }
mkdir -p media
docker compose up --build -d api worker
port="$(grep -E '^VH_API_PORT=' .env | tail -1 | cut -d= -f2)"; port="${port:-8000}"
echo "waiting for http://localhost:${port}/v1/health ..."
for _ in $(seq 1 120); do
    if curl -fsS "http://localhost:${port}/v1/health" >/dev/null 2>&1; then
        echo "Studio: http://localhost:${port}   API docs: http://localhost:${port}/docs"
        [[ "${1:-}" == "--logs" ]] && exec docker compose logs -f api worker
        exit 0
    fi
    sleep 2
done
echo "API did not become healthy; recent logs:" >&2
docker compose logs --tail 80 api >&2
exit 1
