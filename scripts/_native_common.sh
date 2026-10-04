# Shared helpers for scripts/run_native_{mac,linux}.sh (sourced, not executed).
# Expects: REPO_ROOT, TORCH_INDEX ("" = default PyPI), PYTHON (interpreter).

vh_log() { echo "[vh] $*"; }

vh_load_env() {
    # Load .env (KEY=VALUE lines) without overriding variables already set.
    local file="$REPO_ROOT/.env" line key
    [[ -f "$file" ]] || return 0
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
        key="${line%%=*}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [[ -n "${!key+x}" ]] && continue
        export "$key=${line#*=}"
    done < "$file"
}

vh_pick_python() {
    local candidate
    for candidate in ${PYTHON:-} python3.11 python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && \
           "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            PYTHON="$candidate"; return 0
        fi
    done
    echo "Python 3.11+ not found (macOS: brew install python@3.11; Ubuntu: apt install python3.11-venv)" >&2
    return 1
}

vh_setup_venv() {
    # Creates .venv and installs torch (from TORCH_INDEX) + requirements once;
    # reinstalls only when the requirement files or the torch index change.
    local venv="$REPO_ROOT/.venv" stamp want
    [[ -x "$venv/bin/python" ]] || { vh_log "creating .venv with $PYTHON"; "$PYTHON" -m venv "$venv"; }
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    want="$(cat "$REPO_ROOT"/requirements-base.txt "$REPO_ROOT"/requirements-ml.txt | cksum | cut -d' ' -f1)-${TORCH_INDEX:-pypi}"
    stamp="$venv/.vh-install-stamp"
    if [[ "${VH_SKIP_INSTALL:-0}" == "1" || ( -f "$stamp" && "$(cat "$stamp")" == "$want" ) ]]; then
        vh_log "dependencies up to date"
        return 0
    fi
    python -m pip install --upgrade pip wheel
    if [[ -n "${TORCH_INDEX:-}" ]]; then
        vh_log "installing torch from $TORCH_INDEX"
        python -m pip install torch torchvision --index-url "$TORCH_INDEX"
    else
        vh_log "installing torch from PyPI"
        python -m pip install torch torchvision
    fi
    python -m pip install -r "$REPO_ROOT/requirements-ml.txt"
    echo "$want" > "$stamp"
}

vh_native_defaults() {
    export VH_DATA_DIR="${VH_DATA_DIR:-$REPO_ROOT/data}"
    export VH_DB_URL="${VH_DB_URL:-sqlite:///$VH_DATA_DIR/db/video_highlights.db}"
    export VH_OUTPUT_ROOT="${VH_OUTPUT_ROOT:-$VH_DATA_DIR/outputs}"
    export VH_LOCAL_STORAGE_ROOT="${VH_LOCAL_STORAGE_ROOT:-$VH_DATA_DIR/storage}"
    export VH_MODEL_DIR="${VH_MODEL_DIR:-$REPO_ROOT/models}"
    export VH_DEVICE="${VH_DEVICE:-auto}"
    export VH_SKIP_USER_MANAGEMENT="${VH_SKIP_USER_MANAGEMENT:-true}"
    export VH_API_PORT="${VH_API_PORT:-8000}"
    mkdir -p "$VH_DATA_DIR/db" "$VH_OUTPUT_ROOT" "$VH_LOCAL_STORAGE_ROOT" "$VH_MODEL_DIR"
}

vh_download_models() {
    python "$REPO_ROOT/scripts/download_models.py" --model-dir "$VH_MODEL_DIR" --allow-missing ${VH_MODELS:-yolov8n.pt yolov8s.pt}
}

vh_print_status() {
    python - <<'PY'
from backend.services.gpu_status import get_gpu_status
s = get_gpu_status()
enc = ", ".join(k for k, v in s["encoders"].items() if v) or "none"
print(f"[vh] device={s['recommended_device']} hardware_class={s['hardware_class']} "
      f"encoder={s['recommended_encoder']} hwaccel={s['recommended_hwaccel']} (ffmpeg encoders: {enc})")
PY
}
