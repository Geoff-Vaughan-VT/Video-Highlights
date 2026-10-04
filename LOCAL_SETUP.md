# Local Setup

Per-platform setup, native and Docker. Container internals, cloud and
limitations: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Every `VH_*`
variable: [`.env.example`](.env.example).

| Platform | Recommended | GPU path | Script |
|---|---|---|---|
| Windows 11 + NVIDIA RTX | Docker Desktop (WSL2) or native venv | CUDA torch `cu128`, NVDEC/NVENC | `run_docker_gpu.bat`, `run_native_windows.bat` |
| Apple Silicon (Mac Studio) | **native only** | torch MPS, VideoToolbox | `scripts/run_native_mac.sh` |
| DGX Spark (GB10, arm64) | Docker + NVIDIA toolkit, or native | CUDA 13 torch `cu130` | `run_docker_gpu.sh`, `scripts/run_native_linux.sh` |
| Linux x86_64 + NVIDIA | Docker or native | `cu128` | same |
| Anything else | Docker CPU | CPU | `run_docker.sh` / `.bat` |

Native runs keep state in `./data` (`db/`, `outputs/`, `storage/`) and
weights in `./models`. Containers use the `vh-data` and `vh-models` volumes
and mount `VH_MEDIA_DIR` at `/media` read-only.

## 1. Windows 11 + NVIDIA

### Docker Desktop

1. Install the current NVIDIA Game Ready/Studio driver and Docker Desktop
   (WSL2 backend; GPU support is automatic). Check in PowerShell:
   `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`.
2. Raise WSL2 limits if the box is big (`%UserProfile%\.wslconfig`:
   `[wsl2]` `memory=48GB` `processors=24`), then `wsl --shutdown`.
3. `copy .env.example .env`, set `VH_MEDIA_DIR=D:/Videos`.
4. `run_docker_gpu.bat` (API + GPU worker) or `run_docker.bat` (CPU).
   Stop with `stop_docker.bat`.
5. In the Studio choose **Local file on the server** and enter
   `/media/<file>.mp4`.

### Native

1. Python 3.11 from python.org (with the `py` launcher), ffmpeg
   (`winget install Gyan.FFmpeg`, reopen the terminal), NVIDIA driver.
2. `run_native_windows.bat` (wraps `scripts/run_native_windows.ps1`):
   creates `.venv`, installs torch from the `cu128` index and
   `requirements-ml.txt`, downloads weights to `.\models`, starts the API and,
   once `/v1/health` answers, the queue worker. Options: `-Inline` (no
   worker), `-TorchIndex cpu`, `-Port 8080`, `-SkipInstall`.
3. Register videos by their real path (`D:\Videos\match.mp4`).

## 2. Apple Silicon (Mac Studio, MacBook Pro)

Docker on macOS cannot use the Apple GPU, so run natively:

```bash
brew install python@3.11 ffmpeg
scripts/run_native_mac.sh            # API with inline jobs on :8000
scripts/run_native_mac.sh --queue    # API + separate worker process
```

The script creates `.venv`, installs PyPI torch (its macOS arm64 wheels
include MPS), checks `ffmpeg -encoders` for `h264_videotoolbox`, sets
`VH_DEVICE=auto` (resolves to `mps`) and `PYTORCH_ENABLE_MPS_FALLBACK=1`,
downloads weights and starts the API. `GET /v1/health/gpu` should show
`mps_available: true`, `recommended_encoder: h264_videotoolbox`,
`hardware_class: apple_m2_ultra` (or `apple_m1_max`).

## 3. NVIDIA DGX Spark and Linux

### Docker

```bash
# once: NVIDIA Container Toolkit (preinstalled on DGX OS)
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
cp .env.example .env                 # VH_MEDIA_DIR=/srv/matches
./run_docker_gpu.sh                  # aarch64 -> CUDA 13 base + cu130 torch
./stop_docker.sh
```

### Native

```bash
sudo apt install python3.12-venv ffmpeg      # or python3.11-venv
scripts/run_native_linux.sh                  # API + worker; --inline for in-process jobs
```

Index selection: `aarch64` + `nvidia-smi` -> `cu130`; `x86_64` +
`nvidia-smi` -> `cu128`; otherwise `cpu`. Override with
`TORCH_INDEX=https://download.pytorch.org/whl/<tag>`.

## 4. Manual install (any platform)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick your index
pip install -r requirements-ml.txt          # + requirements-dev.txt for tests
python scripts/download_models.py           # -> ./models (VH_MODEL_DIR)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Requirement files: `requirements-base.txt` (API/worker control plane, no ML),
`requirements-ml.txt` (+ torch, ultralytics, lap, librosa, moviepy),
`requirements-dev.txt` (+ pytest), `requirements.txt` (umbrella = dev).

Queue mode natively: set `VH_JOB_EXECUTION_MODE=queue`, run the API and
`python -m backend.worker` in a second terminal (`run_worker.sh/.bat`).

## 5. Verify the hardware path

```bash
curl -s localhost:8000/v1/health/gpu | python -m json.tool
python bench/bench_match.py --minutes 2 --height 2160      # measured fps + 90-min projection
```

Look for `recommended_device` (`cuda`/`mps`), `recommended_encoder`
(`h264_nvenc`/`h264_videotoolbox`), `recommended_hwaccel` and
`hardware_class`. The bench writes `bench/results/<timestamp>.md`.

## 6. Models

Profiles use `yolov8n.pt` (fast), `yolov8s.pt` (balanced), `yolov8m.pt`
(quality). `VH_MODEL_FAMILY=yolo26` switches them to `yolo26n/s/m.pt`.
`python scripts/download_models.py yolov8m.pt yolo26s.pt` fetches more;
existing files are never re-downloaded, so it also runs offline. Images bake
`MODELS` (build arg) into `/models`.

## 7. API examples

Optional auth lock-down:

```bash
set VH_AUTH_REQUIRED=true
set VH_API_TOKENS=admin-token:admin,coach-token:coach,analyst-token:analyst,tenant-admin-token:tenant_admin
```

JWT auth mode:

```bash
set VH_JWT_SECRET=replace-with-strong-secret
set VH_JWT_ISSUER=video-highlights
set VH_JWT_DEFAULT_EXP_MINUTES=120
```

Optional bootstrap token issuing:

```bash
set VH_AUTH_BOOTSTRAP_KEY=bootstrap-secret
```

Issue JWT token (once API is running):

```bash
curl -X POST http://localhost:8000/v1/auth/token ^
  -H "Authorization: Bearer admin-token" ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"coach_1\",\"role\":\"coach\",\"tenant_id\":\"default\",\"expires_in_minutes\":120}"
```

Tenant-scoped API requests:

```bash
curl -X GET http://localhost:8000/v1/matches ^
  -H "Authorization: Bearer coach-token" ^
  -H "X-Tenant-Id: default"
```

Skip user-management for core testing (auto-provision memberships):

```bash
set VH_SKIP_USER_MANAGEMENT=true
set VH_BASE_TENANT_SLUG=sandbox
set VH_BASE_TENANT_NAME=Sandbox Tenant
```

Enable deep test/debug logging (including extreme job logs):

```bash
set VH_TEST_MODE=true
set VH_LOG_LEVEL=DEBUG
set VH_JOB_LOG_DETAIL=extreme
```

Global admin API examples:

```bash
curl -X GET http://localhost:8000/v1/admin/global/summary ^
  -H "Authorization: Bearer admin-token"
```

```bash
curl -X POST http://localhost:8000/v1/admin/global/tenants ^
  -H "Authorization: Bearer admin-token" ^
  -H "Content-Type: application/json" ^
  -d "{\"slug\":\"club-a\",\"name\":\"Club A\",\"status\":\"active\",\"metadata\":{}}"
```

Tenant admin API example:

```bash
curl -X GET http://localhost:8000/v1/admin/tenant/summary ^
  -H "Authorization: Bearer tenant-admin-token" ^
  -H "X-Tenant-Id: club-a"
```

Job log and fast-kill examples:

```bash
curl -X GET http://localhost:8000/v1/jobs/{job_id}/logs?detail_level=extreme&limit=500 ^
  -H "X-Tenant-Id: sandbox"
```

```bash
curl -X POST http://localhost:8000/v1/jobs/{job_id}/kill-session ^
  -H "X-Tenant-Id: sandbox" ^
  -H "Content-Type: application/json" ^
  -d "{}"
```

Rerun job with updated model/version targets:

```bash
curl -X POST http://localhost:8000/v1/jobs/{job_id}/rerun ^
  -H "X-Tenant-Id: sandbox" ^
  -H "Content-Type: application/json" ^
  -d "{\"config_overrides\":{\"model_version\":\"event-v1\",\"focus_event_types\":[\"goal\",\"corner_kick\"]},\"reason\":\"model-upgrade\"}"
```

Create analysis-only job (bookmark table only, no clip rendering):

```bash
curl -X POST http://localhost:8000/v1/matches/{match_id}/jobs ^
  -H "X-Tenant-Id: sandbox" ^
  -H "Content-Type: application/json" ^
  -d "{\"config\":{\"analysis_only\":true,\"model_version\":\"event-v1\",\"focus_event_types\":[\"goal\",\"corner_kick\"]}}"
```

Fetch bookmark/event table for a specific run:

```bash
curl -X GET "http://localhost:8000/v1/matches/{match_id}/events?job_id={job_id}&limit=1000" ^
  -H "X-Tenant-Id: sandbox"
```

Fetch live bookmark table for a running/completed job:

```bash
curl -X GET "http://localhost:8000/v1/jobs/{job_id}/bookmarks?limit=5000" ^
  -H "X-Tenant-Id: sandbox"
```

Delete an old run (removes run + logs + run-linked events):

```bash
curl -X DELETE "http://localhost:8000/v1/jobs/{job_id}" ^
  -H "X-Tenant-Id: sandbox"
```

Render frame-accurate clip for an event bookmark:

```bash
curl -X POST http://localhost:8000/v1/matches/{match_id}/events/{event_id}/clip-on-demand ^
  -H "X-Tenant-Id: sandbox" ^
  -H "Content-Type: application/json" ^
  -d "{\"pre_seconds\":1.5,\"post_seconds\":5.0,\"anchor\":\"event_window\",\"include_audio\":true,\"prefer_gpu\":true,\"force_rebuild\":false}"
```

Export a final highlight reel from selected bookmark events:

```bash
curl -X POST "http://localhost:8000/v1/matches/{match_id}/exports/highlights" ^
  -H "X-Tenant-Id: sandbox" ^
  -H "Content-Type: application/json" ^
  -d "{\"event_ids\":[\"evt_1\",\"evt_2\"],\"pre_seconds\":1.0,\"post_seconds\":3.0,\"anchor\":\"event_window\",\"include_audio\":true,\"prefer_gpu\":true,\"title\":\"Selected Highlights\"}"
```

S3-compatible storage mode:

```bash
set VH_STORAGE_BACKEND=s3
set VH_S3_BUCKET=video-highlights
set VH_S3_ENDPOINT_URL=https://<s3-endpoint>
set VH_S3_ACCESS_KEY_ID=<key>
set VH_S3_SECRET_ACCESS_KEY=<secret>
set VH_S3_REGION=<region>
set VH_S3_KEY_PREFIX=video-highlights
```


## 8. Common issues

| Symptom | Fix |
|---|---|
| `/v1/health/gpu` shows `cuda_available: false` in Docker | Host `nvidia-smi` works? Toolkit installed? Service started with `--profile gpu` (`worker-gpu`)? `docker compose --profile gpu exec worker-gpu nvidia-smi`. |
| ffmpeg lists `h264_nvenc` but renders fail | Container lacks the `video` driver capability (set in the image/compose) or no NVIDIA GPU; renders fall back to `libx264`. |
| `torch.cuda.is_available()` false natively on Windows | CPU wheel installed: rerun `run_native_windows.ps1 -TorchIndex cu128` (it reinstalls when the index changes). |
| MPS op not implemented | `PYTORCH_ENABLE_MPS_FALLBACK=1` (set by the Mac script). |
| Permission denied writing `/data` or `/models` | Volumes created by an older root image: `docker compose down -v` (wipes data) or `docker run --rm -v vh-data:/data alpine chown -R 10001:10001 /data`. |
| Video not visible in the Studio local-path picker | Path must be under `/media` in Docker; check `VH_MEDIA_DIR` in `.env` and that Docker Desktop can access the drive. |
| Old schema errors after upgrading | Delete the SQLite file (`./data/db/video_highlights.db` natively, or `docker compose down -v`). |
