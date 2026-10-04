#!/usr/bin/env bash
# Apple Silicon native runner: torch MPS for detection, VideoToolbox for
# ffmpeg decode/encode. (Docker on macOS has no GPU, so use this instead.)
#
#   scripts/run_native_mac.sh            # install once, start API (inline jobs)
#   scripts/run_native_mac.sh --queue    # API + separate worker process
#   VH_API_PORT=8080 scripts/run_native_mac.sh
#
# Prereqs: Homebrew python@3.11 and ffmpeg (brew install python@3.11 ffmpeg).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/_native_common.sh
source "$REPO_ROOT/scripts/_native_common.sh"

MODE="inline"
[[ "${1:-}" == "--queue" ]] && MODE="queue"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This script is for macOS; use scripts/run_native_linux.sh" >&2
    exit 1
fi
[[ "$(uname -m)" == "arm64" ]] || vh_log "WARNING: Intel Mac - no MPS; expect CPU speeds"

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found: brew install ffmpeg" >&2; exit 1; }
if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_videotoolbox; then
    vh_log "ffmpeg has h264_videotoolbox"
else
    vh_log "WARNING: ffmpeg lacks h264_videotoolbox; renders fall back to libx264 (brew install ffmpeg)"
fi

vh_load_env
vh_pick_python
TORCH_INDEX=""   # PyPI macOS arm64 wheels include MPS
vh_setup_venv
vh_native_defaults
export VH_JOB_EXECUTION_MODE="${VH_JOB_EXECUTION_MODE:-$MODE}"
# Run ops MPS lacks on the CPU instead of failing.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
python -c 'import torch; import sys; ok = torch.backends.mps.is_available(); print(f"[vh] torch {torch.__version__} mps_available={ok}"); sys.exit(0)'
vh_download_models
vh_print_status

worker_pid=""
if [[ "$VH_JOB_EXECUTION_MODE" == "queue" ]]; then
    python -m backend.worker &
    worker_pid=$!
    trap '[[ -n "$worker_pid" ]] && kill "$worker_pid" 2>/dev/null || true' EXIT
fi
vh_log "Studio: http://localhost:${VH_API_PORT}   (mode: ${VH_JOB_EXECUTION_MODE}, data: ${VH_DATA_DIR})"
python -m uvicorn backend.main:app --host 0.0.0.0 --port "$VH_API_PORT"
