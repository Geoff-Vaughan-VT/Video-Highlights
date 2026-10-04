#!/usr/bin/env bash
# Single-container launcher: API (+ Studio UI) and one queue worker.
#
# Default CMD of both images, so `docker run -p 8000:8000 <image>` is the
# whole product. Compose runs api/worker as separate services instead.
#
#   VH_START_WORKER=false   API only
#   VH_API_PORT=8000        listen port
#   VH_READY_TIMEOUT=120    seconds to wait for /v1/health before failing
set -uo pipefail

export VH_DB_URL="${VH_DB_URL:-sqlite:////data/db/video_highlights.db}"
export VH_OUTPUT_ROOT="${VH_OUTPUT_ROOT:-/data/outputs}"
export VH_LOCAL_STORAGE_ROOT="${VH_LOCAL_STORAGE_ROOT:-/data/storage}"
export VH_JOB_EXECUTION_MODE="${VH_JOB_EXECUTION_MODE:-queue}"
export VH_AUTH_REQUIRED="${VH_AUTH_REQUIRED:-false}"
export VH_SKIP_USER_MANAGEMENT="${VH_SKIP_USER_MANAGEMENT:-true}"
export VH_BASE_TENANT_SLUG="${VH_BASE_TENANT_SLUG:-sandbox}"
export VH_BASE_TENANT_NAME="${VH_BASE_TENANT_NAME:-Sandbox Tenant}"
PORT="${VH_API_PORT:-8000}"
READY_TIMEOUT="${VH_READY_TIMEOUT:-120}"
START_WORKER="${VH_START_WORKER:-true}"

# sqlite:////data/db/x.db -> make sure /data/db exists.
if [[ "$VH_DB_URL" == sqlite:///* ]]; then
    mkdir -p "$(dirname "${VH_DB_URL#sqlite:///}")"
fi
mkdir -p "$VH_OUTPUT_ROOT" "$VH_LOCAL_STORAGE_ROOT"

api_pid=""
worker_pid=""

log() { echo "[start-all] $*"; }

shutdown() {
    local code="${1:-0}"
    log "stopping (exit ${code})"
    for pid in $worker_pid $api_pid; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in $worker_pid $api_pid; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$code"
}
trap 'shutdown 143' TERM
trap 'shutdown 130' INT

log "API on :${PORT} (db ${VH_DB_URL}, outputs ${VH_OUTPUT_ROOT})"
python -m uvicorn backend.main:app --host 0.0.0.0 --port "$PORT" &
api_pid=$!

# Wait for the API (it creates the DB schema) before starting the worker.
deadline=$(( SECONDS + READY_TIMEOUT ))
until curl -fsS "http://127.0.0.1:${PORT}/v1/health" >/dev/null 2>&1; do
    if ! kill -0 "$api_pid" 2>/dev/null; then
        wait "$api_pid"; code=$?
        log "API exited during startup (code ${code})"
        exit "${code:-1}"
    fi
    if (( SECONDS >= deadline )); then
        log "API not healthy after ${READY_TIMEOUT}s"
        shutdown 1
    fi
    sleep 1
done
log "API ready: Studio http://localhost:${PORT}  docs http://localhost:${PORT}/docs"

if [[ "$START_WORKER" == "true" ]]; then
    python -m backend.worker &
    worker_pid=$!
    log "worker started (pid ${worker_pid})"
fi

# Exit as soon as either process dies, forwarding its exit code so a
# restart policy can recover the container.
wait -n $api_pid $worker_pid
code=$?
log "a process exited with code ${code}"
shutdown "$code"
