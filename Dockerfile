# syntax=docker/dockerfile:1.7
# CPU image: API + queue worker for any x86_64/arm64 host (and the API half of
# a GPU deployment). Torch comes from the CPU wheel index, so the multi-GB
# CUDA wheel is never downloaded.
#
#   docker build -t video-highlights:cpu .
#   docker build --build-arg MODELS="yolov8n.pt yolov8s.pt yolov8m.pt" -t video-highlights:cpu .
#
# GPU (NVIDIA) image: Dockerfile.gpu. Apple Silicon: run natively
# (scripts/run_native_mac.sh) - Docker on macOS has no GPU.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# ---------------------------------------------------------------- builder --
FROM ${PYTHON_IMAGE} AS builder

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG MODELS="yolov8n.pt yolov8s.pt"

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Compilers only for the rare dependency without a wheel for this arch.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && python -m venv /opt/venv

WORKDIR /build
COPY requirements-base.txt requirements-ml.txt ./
# Torch first from the CPU index, then the rest (pip keeps the CPU torch).
RUN pip install --upgrade pip wheel \
 && pip install torch torchvision --index-url "${TORCH_INDEX}" \
 && pip install -r requirements-ml.txt

# Bake detector weights into /models (bundled yolov8n.pt is copied, others
# downloaded once at build time; no network needed at runtime).
COPY scripts/download_models.py scripts/download_models.py
COPY yolov8n.pt /build/seed/yolov8n.pt
RUN python scripts/download_models.py --model-dir /models --seed-dir /build/seed ${MODELS}

# ---------------------------------------------------------------- runtime --
FROM ${PYTHON_IMAGE} AS runtime

ARG VH_UID=10001
ARG VH_GID=10001

# ffmpeg: proxy pass, render, clips. libgl1/libglib2.0-0: opencv-python
# (pulled by ultralytics). libsndfile1: soundfile/librosa. curl: start_all.sh.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libsndfile1 curl tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid "${VH_GID}" vh \
 && useradd --uid "${VH_UID}" --gid vh --create-home --shell /usr/sbin/nologin vh \
 && mkdir -p /data/db /data/outputs /data/storage /media /models \
 && chown -R vh:vh /data /models

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=vh:vh /models /models

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VH_MODEL_DIR=/models \
    YOLO_CONFIG_DIR=/home/vh/.config/Ultralytics \
    VH_DB_URL=sqlite:////data/db/video_highlights.db \
    VH_OUTPUT_ROOT=/data/outputs \
    VH_LOCAL_STORAGE_ROOT=/data/storage \
    VH_JOB_EXECUTION_MODE=queue \
    VH_DEVICE=auto \
    VH_API_PORT=8000

WORKDIR /app
COPY --chown=root:root requirements*.txt ./
COPY --chown=root:root VideoHighlights.py ./
COPY --chown=root:root backend ./backend
COPY --chown=root:root frontend ./frontend
COPY --chown=root:root scripts ./scripts
COPY --chown=root:root docker ./docker
RUN chmod 0755 /app/docker/*.sh

USER vh
# Bare model names (YOLO("yolov8s.pt")) resolve to /models.
RUN python scripts/download_models.py --model-dir /models --configure-ultralytics --allow-missing

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["python", "/app/docker/healthcheck.py"]

ENTRYPOINT ["/usr/bin/tini", "--"]
# Default: API + worker in one container (docker run). Compose runs them as
# separate services with their own `command:`.
CMD ["/app/docker/start_all.sh"]
