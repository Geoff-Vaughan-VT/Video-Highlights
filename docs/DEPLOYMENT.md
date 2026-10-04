# Deployment

Targets, in priority order: (1) Windows 11 + NVIDIA RTX, (2) Apple Silicon
Mac Studio, (3) NVIDIA DGX Spark, (4) cloud (Kubernetes / any container
host, Postgres, object storage). Per-platform native setup is in
[`LOCAL_SETUP.md`](../LOCAL_SETUP.md); this file covers containers, GPU
plumbing, storage, configuration and known limitations.

## 1. Moving parts

| Process | Command | Needs |
|---|---|---|
| `api` | `uvicorn backend.main:app` | DB, `/data`, read access to `/media`. Serves the Studio UI, API, files, on-demand clips. |
| `worker` | `python -m backend.worker` | DB, `/data`, `/media`, models, GPU if available. Polls the queue every 2 s and runs one job at a time. |

`VH_JOB_EXECUTION_MODE=queue` (container default) makes the API enqueue and
the worker execute. `inline` makes the API run jobs in a thread pool
(`VH_JOB_MAX_WORKERS`) and is meant for single-user native installs
(`scripts/run_native_mac.sh` default).

Images:

| File | Base | Torch | For |
|---|---|---|---|
| `Dockerfile` | `python:3.11-slim-bookworm` | CPU wheel (`download.pytorch.org/whl/cpu`) | API anywhere; CPU worker; amd64 + arm64 |
| `Dockerfile.gpu` | `nvidia/cuda:12.8.1-base-ubuntu24.04` (`CUDA_IMAGE` arg) | `cu128` (`TORCH_INDEX` arg; `cu130` for Spark) | NVIDIA worker (or whole stack) |

Both are multi-stage (build tools and pip caches stay in the builder), run
as non-root `vh` (uid/gid 10001, `VH_UID`/`VH_GID` build args), use `tini` as
PID 1, carry `HEALTHCHECK` = `docker/healthcheck.py` (GET
`/v1/health`), bake weights into `/models` (`MODELS` build arg, default
`yolov8n.pt yolov8s.pt`; the bundled `yolov8n.pt` is copied, never
downloaded) and point Ultralytics' `weights_dir` there. No pytest in the
runtime image. Default `CMD` is `docker/start_all.sh`: API, wait for
`/v1/health` (curl loop, `VH_READY_TIMEOUT`, default 120 s), then one
worker (`VH_START_WORKER=false` to skip); signals are forwarded and the
first process to exit stops the container with its exit code.

The GPU image uses the CUDA *base* image because torch wheels bundle CUDA,
cuDNN and cuBLAS; a `cudnn-runtime` base would ship them twice (~3 GB).

## 2. Docker Compose

```bash
cp .env.example .env               # edit VH_MEDIA_DIR at least
docker compose up -d --build       # api + worker (CPU)
docker compose --profile gpu up -d --build api worker-gpu   # NVIDIA
docker compose logs -f api worker-gpu
docker compose --profile gpu --profile cloud down           # stop all (stop_docker.sh)
```

| Service | Image | Profile | Notes |
|---|---|---|---|
| `api` | `video-highlights:cpu` | - | Port `${VH_API_PORT:-8000}` -> 8000. Healthcheck gates dependents. |
| `worker` | `video-highlights:cpu` | - | Starts after `api` is healthy. |
| `worker-gpu` | `video-highlights:gpu` | `gpu` | `deploy.resources.reservations.devices` (nvidia, all GPUs, `video` capability), `ipc: host`, 8 GB shm. |
| `postgres` | `postgres:16-alpine` | `cloud` | Only used when `VH_DB_URL` points at it. |

Volumes:

| Mount | Kind | Content |
|---|---|---|
| `/data` | named volume `vh-data` | `db/video_highlights.db` (SQLite), `outputs/<job_id>/` (run dirs, `VH_OUTPUT_ROOT`), `storage/` (uploads, `VH_LOCAL_STORAGE_ROOT`) |
| `/models` | named volume `vh-models` | Detector weights + cached TensorRT engines. Seeded from the image on first use; writable so new models/engines persist. |
| `/media` | bind `${VH_MEDIA_DIR:-./media}`, read-only | Source matches. Register `/media/<file>` as a local-path source; nothing is copied. |

`.env` feeds both compose interpolation and the containers (`env_file`,
optional). Values set under `environment:` in compose use
`${VAR:-default}`, so `.env` wins over the defaults. The in-container port is
always 8000; `VH_API_PORT` only moves the host port.

Windows paths in `.env`: forward slashes (`VH_MEDIA_DIR=D:/Videos`). Docker
Desktop shares all fixed drives by default with the WSL2 backend. A folder
on a network share must be mapped inside WSL2 first.

Run only one of `worker` / `worker-gpu` against a database (see
Limitations). `run_docker_gpu.sh/.bat` stop the CPU worker before starting
the GPU one.

### Single container

```bash
docker run -d --name vh -p 8000:8000 -v vh-data:/data -v D:/Videos:/media:ro video-highlights:cpu
docker run -d --name vh --gpus all --ipc=host --shm-size=8g -p 8000:8000 \
  -v vh-data:/data -v vh-models:/models -v /srv/matches:/media:ro video-highlights:gpu
```

## 3. NVIDIA GPU (Windows Docker Desktop, Linux, DGX Spark)

Host requirements: current NVIDIA driver; Linux: NVIDIA Container Toolkit
(`nvidia-ctk runtime configure --runtime=docker`); Windows: Docker Desktop
with the WSL2 backend (GPU support is built in; do not install a Linux driver
inside WSL). Check: `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`.

`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` is set in the image and
compose: `compute` = CUDA, `utility` = nvidia-smi, `video` = NVDEC/NVENC
libraries mounted into the container. Without `video`, ffmpeg lists
`h264_nvenc` but fails at runtime.

ffmpeg: Ubuntu 24.04 ships ffmpeg 6.1 built with nv-codec-headers, so
`-hwaccel cuda`, `h264_nvenc`, `hevc_nvenc`, `scale_cuda` and
`hwupload_cuda` are present (verified: `ffmpeg -hwaccels` lists `cuda`,
`ffmpeg -encoders` lists both NVENC encoders, `ffmpeg -filters` lists
`scale_cuda`/`hwupload_cuda`). `scale_npp` (non-free) is not. Verify in the
running worker:

```bash
docker compose --profile gpu exec worker-gpu ffmpeg -hide_banner -hwaccels
docker compose --profile gpu exec worker-gpu ffmpeg -hide_banner -f lavfi -i testsrc2=size=1920x1080:duration=2 -c:v h264_nvenc -f null -
curl -s localhost:8000/v1/health/gpu   # from the api: ready, encoders, hardware_class
```

NVENC/NVDEC under WSL2 work with current drivers; if the NVENC test above
fails, renders fall back to `libx264` automatically and only the encode
stages slow down.

Session limits: GeForce cards allow a limited number of concurrent NVENC
sessions (8 on current drivers). One worker uses at most two at a time.

### DGX Spark (GB10, arm64, CUDA 13)

`run_docker_gpu.sh` detects `aarch64` and builds with
`VH_CUDA_IMAGE=nvidia/cuda:13.0.1-base-ubuntu24.04`, `VH_TORCH_INDEX=cu130`.
Manually:

```bash
docker build -f Dockerfile.gpu --build-arg CUDA_IMAGE=nvidia/cuda:13.0.1-base-ubuntu24.04 \
  --build-arg TORCH_INDEX=cu130 -t video-highlights:gpu .
```

The published `:gpu-arm64` tag is that build. Unified 128 GB memory: the
`auto` batch can go to 32 at imgsz 1280; `nvidia-smi` reports memory as
`[N/A]`, which `classify_hardware` treats as Spark. Ollama for match reports
can share the box (`VH_LLM_BASE_URL=http://host.docker.internal:11434`;
on Linux add `extra_hosts: ["host.docker.internal:host-gateway"]`).

### TensorRT

Build the GPU image with `--build-arg INSTALL_TENSORRT=1` (adds
`tensorrt-cu12`/`tensorrt-cu13`, `onnx`, `onnxslim`), set `VH_TENSORRT=1`.
The first job exports an engine per (model, imgsz, precision, batch) next to
the weights in `/models` and reuses it afterwards; keep `vh-models` as a
volume so the export survives container rebuilds. Engines are specific to
the GPU model and TensorRT version.

## 4. Apple Silicon

Docker Desktop on macOS runs Linux VMs without Metal access, so containers
are CPU-only there. Use `scripts/run_native_mac.sh`: Homebrew Python 3.11 +
ffmpeg, PyPI torch (MPS included), `PYTORCH_ENABLE_MPS_FALLBACK=1`,
`VH_DEVICE=auto` (resolves to `mps`), VideoToolbox decode/encode
(`h264_videotoolbox`). Data under `./data`, weights under `./models`.
`--queue` runs a separate worker process instead of inline jobs.

## 5. Cloud / multi-host

* **Database:** set `VH_DB_URL=postgresql+psycopg://user:pass@host:5432/db`
  (driver `psycopg[binary]` is in `requirements-base.txt`). Compose
  `--profile cloud` starts a local Postgres for testing. Tables are created
  on API startup; there are no migrations yet, so schema changes need a
  manual migration or a fresh DB.
* **Files:** `VH_STORAGE_BACKEND=s3` with `VH_S3_*` for uploads and signed
  download URLs. Run directories (`VH_OUTPUT_ROOT`) are still a filesystem
  path: workers and the API must share it (RWX volume such as EFS/Filestore/
  Azure Files, or run API + worker in one pod).
* **Kubernetes sketch:** one `Deployment` for `api` (CPU image, readiness/
  liveness `GET /v1/health`), one for `worker` on GPU nodes
  (`nvidia.com/gpu: 1`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`,
  CPU image or GPU image), a shared RWX PVC at `/data`, a PVC or init
  container for `/models`, Postgres from a managed service. Keep `replicas: 1`
  for workers until the queue claim is atomic.
* **Secrets:** `VH_JWT_SECRET`, `VH_API_TOKENS`, `VH_S3_SECRET_ACCESS_KEY`,
  `VH_SMTP_PASSWORD`, `VH_LLM_API_KEY`/`OPENAI_API_KEY` belong in a secret
  store, not in `.env` committed anywhere. Set `VH_AUTH_REQUIRED=true` and
  `VH_SKIP_USER_MANAGEMENT=false` for anything reachable from a network.

## 6. Configuration reference

`.env.example` lists every variable with defaults. Groups:

| Group | Variables |
|---|---|
| Wiring (compose only) | `VH_MEDIA_DIR`, `VH_API_PORT`, `VH_IMAGE_CPU`, `VH_IMAGE_GPU`, `VH_CUDA_IMAGE`, `VH_TORCH_INDEX`, `VH_INSTALL_TENSORRT`, `VH_MODELS`, `POSTGRES_*` |
| Storage / DB / queue | `VH_DB_URL`, `VH_OUTPUT_ROOT`, `VH_LOCAL_STORAGE_ROOT`, `VH_JOB_EXECUTION_MODE`, `VH_JOB_MAX_WORKERS`, `VH_STORAGE_BACKEND`, `VH_S3_ENDPOINT_URL`, `VH_S3_BUCKET`, `VH_S3_REGION`, `VH_S3_ACCESS_KEY_ID`, `VH_S3_SECRET_ACCESS_KEY`, `VH_S3_KEY_PREFIX` |
| Hardware | `VH_DEVICE` (auto/cuda/cuda:N/mps/cpu), `VH_MODEL_DIR`, `VH_MODEL_FAMILY` (yolov8/yolo11/yolo26), `VH_TENSORRT`, `VH_FFMPEG`, `VH_FFPROBE`, `VH_FFMPEG_BIN`, `VH_RNNOISE_MODEL_PATH` |
| Auth / tenancy | `VH_AUTH_REQUIRED`, `VH_API_TOKENS`, `VH_AUTH_BOOTSTRAP_KEY`, `VH_JWT_SECRET`, `VH_JWT_ALGORITHM`, `VH_JWT_ISSUER`, `VH_JWT_AUDIENCE`, `VH_JWT_DEFAULT_EXP_MINUTES`, `VH_SKIP_USER_MANAGEMENT`, `VH_BASE_TENANT_SLUG`, `VH_BASE_TENANT_NAME` |
| Uploads / SLA | `VH_UPLOAD_MAX_GB` (3), `VH_UPLOAD_EXTENDED_MAX_GB` (8, tenants with `entitlements.extended_uploads`), `VH_UPLOAD_MIN_DURATION_SECONDS` (0 = off), `VH_PROCESSING_SLA_HOURS_MIN` (4), `VH_PROCESSING_SLA_HOURS_MAX` (6) |
| Notifications | `VH_NOTIFY_BACKEND` (console/smtp/disabled), `VH_SMTP_HOST`, `VH_SMTP_PORT` (587), `VH_SMTP_USERNAME`, `VH_SMTP_PASSWORD`, `VH_SMTP_FROM`, `VH_SMTP_STARTTLS` (true) |
| Logging | `VH_TEST_MODE`, `VH_LOG_LEVEL` (INFO), `VH_JOB_LOG_DETAIL` (basic/detailed/extreme), `VH_PERSIST_JOB_LOGS` |
| LLM | `VH_LLM_PROVIDER`, `VH_LLM_MODEL`, `VH_LLM_BASE_URL`, `VH_LLM_API_KEY`, `VH_LLM_TIMEOUT_SECONDS`, `VH_LLM_KEEP_ALIVE`, `OPENAI_API_KEY` |
| Launcher | `VH_START_WORKER`, `VH_READY_TIMEOUT`, `VH_HEALTHCHECK_URL` (start_all.sh / healthcheck.py) |

Compose defaults: `VH_TEST_MODE=false`, `VH_LOG_LEVEL=INFO`,
`VH_JOB_LOG_DETAIL=basic`, `VH_AUTH_REQUIRED=false`,
`VH_SKIP_USER_MANAGEMENT=true` (local dev convenience; change for shared
hosts). Roster, sharing, notifications, player routing, source catalog and
stat catalog features need no extra infrastructure beyond SMTP for real
email.

## 7. Known limitations

* **Queue claim is not atomic.** `job_runner.run_next_queued_job` selects a
  queued job and then updates it in a separate step; two workers polling the
  same DB can claim the same job. Run exactly one worker per database until
  the claim becomes a single conditional `UPDATE ... WHERE status='queued'`
  (or `SELECT ... FOR UPDATE SKIP LOCKED` on Postgres).
* **SQLite across containers** works on one host with a local named volume.
  Do not put the SQLite file on NFS/SMB or share it across hosts; use
  Postgres for that.
* **No schema migrations.** Upgrading over an old DB may need a fresh DB.
* **Output directory is a filesystem path** even with S3 storage; API and
  workers must share it.
* **Apple GPU only natively.** macOS containers are CPU-only.
* **NVENC session caps** on GeForce cards limit parallel renders.
* **Runtime estimates are planning numbers.** Run `bench/bench_match.py` on
  the target box; `GET /v1/health/gpu` reports which `hardware_class`
  the estimate uses.
* **arm64 GPU image builds** need a native arm64 builder (the publish
  workflow uses `ubuntu-24.04-arm`) or slow QEMU emulation.
