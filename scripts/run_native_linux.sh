#!/usr/bin/env bash
# Linux native runner (x86_64 NVIDIA, DGX Spark arm64, or CPU).
#
#   scripts/run_native_linux.sh            # API + queue worker
#   scripts/run_native_linux.sh --inline   # API runs jobs in-process
#   TORCH_INDEX=https://download.pytorch.org/whl/cu126 scripts/run_native_linux.sh
#
# Torch index: aarch64 + NVIDIA (DGX Spark, CUDA 13) -> cu130;
# x86_64 + NVIDIA -> cu128; no nvidia-smi -> cpu.
# Prereqs: python3.11 (or 3.12) with venv, ffmpeg (apt install ffmpeg).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/_native_common.sh
source "$REPO_ROOT/scripts/_native_common.sh"

MODE="queue"
[[ "${1:-}" == "--inline" ]] && MODE="inline"

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found: sudo apt install ffmpeg" >&2; exit 1; }

arch="$(uname -m)"
if [[ -z "${TORCH_INDEX:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        if [[ "$arch" == "aarch64" ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu130"   # DGX Spark / GB10
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        fi
        nvidia-smi -L
    else
        vh_log "no NVIDIA GPU detected; installing CPU torch"
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    fi
fi
if ffmpeg -hide_banner -hwaccels 2>/dev/null | grep -qx cuda && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
    vh_log "ffmpeg supports -hwaccel cuda and h264_nvenc"
else
    vh_log "ffmpeg without CUDA/NVENC support; decode/encode run on the CPU"
fi

vh_load_env
vh_pick_python
vh_setup_venv
vh_native_defaults
export VH_JOB_EXECUTION_MODE="${VH_JOB_EXECUTION_MODE:-$MODE}"
vh_download_models
vh_print_status

worker_pid=""
api_pid=""
cleanup() { kill $worker_pid $api_pid 2>/dev/null || true; }
trap cleanup EXIT INT TERM

python -m uvicorn backend.main:app --host 0.0.0.0 --port "$VH_API_PORT" &
api_pid=$!
for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:${VH_API_PORT}/v1/health" >/dev/null 2>&1 && break
    kill -0 "$api_pid" 2>/dev/null || { wait "$api_pid"; exit $?; }
    sleep 1
done
if [[ "$VH_JOB_EXECUTION_MODE" == "queue" ]]; then
    python -m backend.worker &
    worker_pid=$!
fi
vh_log "Studio: http://localhost:${VH_API_PORT}   (mode: ${VH_JOB_EXECUTION_MODE}, data: ${VH_DATA_DIR})"
wait -n $api_pid $worker_pid
