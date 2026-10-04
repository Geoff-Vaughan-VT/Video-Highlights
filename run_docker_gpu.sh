#!/usr/bin/env bash
# NVIDIA stack: API (CPU image) + GPU worker. Needs the NVIDIA Container Toolkit.
# On arm64 (DGX Spark) the GPU image is built with CUDA 13 / cu130 wheels.
# Usage: ./run_docker_gpu.sh [--logs]
set -euo pipefail
cd "$(dirname "$0")"
[[ -f .env ]] || { cp .env.example .env; echo "created .env from .env.example (edit VH_MEDIA_DIR)"; }
mkdir -p media
if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
    export VH_CUDA_IMAGE="${VH_CUDA_IMAGE:-nvidia/cuda:13.0.1-base-ubuntu24.04}"
    export VH_TORCH_INDEX="${VH_TORCH_INDEX:-cu130}"
fi
# The CPU worker must not compete for the same queue (claim is not atomic).
docker compose stop worker >/dev/null 2>&1 || true
docker compose --profile gpu up --build -d api worker-gpu
port="$(grep -E '^VH_API_PORT=' .env | tail -1 | cut -d= -f2)"; port="${port:-8000}"
for _ in $(seq 1 120); do
    if curl -fsS "http://localhost:${port}/v1/health" >/dev/null 2>&1; then
        echo "Studio: http://localhost:${port}   GPU status: http://localhost:${port}/v1/health/gpu"
        docker compose --profile gpu exec -T worker-gpu nvidia-smi -L || echo "WARNING: nvidia-smi failed inside worker-gpu" >&2
        [[ "${1:-}" == "--logs" ]] && exec docker compose --profile gpu logs -f api worker-gpu
        exit 0
    fi
    sleep 2
done
echo "API did not become healthy; recent logs:" >&2
docker compose logs --tail 80 api >&2
exit 1
