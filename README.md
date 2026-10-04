# Video Highlights

Video Highlights is a soccer video analysis platform that converts full-match recordings into events, player spotlights, team analytics, and shareable highlight packages.

## Project Status

This repository now provides a local Python pipeline plus a web/API foundation with multitenant controls:

1. CLI pipeline: `VideoHighlights.py`
2. Desktop UI: `VideoHighlightsGUI.py`
3. FastAPI backend + built-in Studio web UI: `backend/main.py` + `frontend/`
   (library & review with view toggles, upload & process, job tracking -
   one server, one port, no separate frontend build)

The planned direction is a web/cloud architecture with Dockerized workers and optional GPU acceleration.

## Target Architecture (Planned)

1. Web app for upload, review, annotation, and export
2. API for users, teams, projects, jobs, and outputs
3. Queue and scheduler for asynchronous processing
4. GPU analysis workers for tracking, event detection, and player analytics
5. Rendering workers for clips, overlays, and montages
6. Object storage, relational database, and observability stack

## Product Capability Targets

1. Follow-cam generation from panoramic match recordings
2. Automated soccer event timeline detection
3. Player spotlight reels and jersey-assisted identity workflows
4. Team momentum graph, heatmaps, and position summaries
5. Timeline editor with annotation and sharing tools
6. Live streaming and instant replay markers (phase-gated)
7. Data trust workflows (confidence calibration and human review queues)
8. Season intelligence and opponent scouting automation
9. Coaching action plans, recruiting workflows, and distribution tooling
10. Open APIs, integrations, and edge/fleet operational visibility
11. AI copilot workflows via LLM API integration for query, explainability, and review assistance
12. Feedback-driven continuous learning loop for improving event quality over time

## Quick Start (Current Local Pipeline)

### Prerequisites

- Python 3.10+
- FFmpeg in PATH
- Optional: NVIDIA GPU + CUDA drivers

### Install

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Run CLI

```bash
python VideoHighlights.py --video path/to/match.mp4 --out ./highlights_output
```

Follow-cam clip export is available for player-centric runs:

```bash
python VideoHighlights.py --video path/to/match.mp4 --out ./highlights_output --camera-mode follow_action --zoom-factor 1.6
```

Use `--camera-mode wide` to preserve the original frame. `follow_player` keeps the selected or auto-selected player centered; `follow_action` blends the player track with nearby ball detections; `follow_ball` is the game-centric camera that tracks the ball itself (see below).

### Game Camera (`follow_ball`), Ball Tracking, and Goal Detection

The `follow_ball` mode plans a virtual camera around "the center of the game":

```bash
python VideoHighlights.py --video match.mp4 --out ./out \
  --camera-mode follow_ball --zoom-factor 1.8 --render-full-follow-cam \
  --debug --log-file ./out/run_debug.log --debug-video --dump-training-data
```

What it does:

1. **Ball tracking**: raw YOLO ball detections are filtered into a clean track
   (teleporting outliers rejected, short gaps interpolated, re-acquisition
   after long occlusions). Stats are logged and written to the manifests.
2. **Field & goal geometry**: field bounds and both goal mouths are estimated
   from the distribution of player positions across the match. Override with
   `--goal-box-left x1,y1,x2,y2` / `--goal-box-right x1,y1,x2,y2`
   (normalized 0-1 or pixel coordinates).
3. **Game states**: the timeline is classified into `in_play`, `ball_lost`,
   `restart_left/right` (goal kick / corner wait), `restart_touchline`
   (throw-in wait), and `goal_left/right`. **During a restart wait the camera
   locks onto the goal and does not leave** until the ball is confirmed back
   in play.
4. **Goal flagging**: three independent signals flag goals - the ball observed
   inside a goal mouth (after entering from the field), the ball observed
   crossing the goal line between the posts, and the ball vanishing while
   heading into the goal mouth. Kickoff re-appearance at the center circle and
   crowd-noise overlap raise confidence. Goals become `goal` bookmarks with a
   guaranteed highlight clip.
5. **Camera planning**: one decision per frame (center, zoom, focus, state,
   confidence, and a human-readable *reason*). The camera leads the ball,
   blends toward the nearby-player centroid, zooms out when the ball is lost,
   and holds on goals/restarts.

Set pieces and cards:

1. **Set-piece detection**: a stationary ball followed by a kick is
   classified by location into corner kicks, free kicks, penalties, goal
   kicks, and kickoffs. During a **free kick near goal** the camera keeps
   the threatened goal in view; during a **corner** it frames the corner and
   the goal together, then tightens as the ball comes in. A general
   goal-threat mode keeps ball AND goal in frame whenever an attack closes
   in on a goal.
2. **Cinematic smoothing**: the camera path is planned offline for the whole
   video and smoothed with zero-phase (future-aware) filtering plus speed
   and acceleration limits - the camera glides and anticipates play instead
   of chasing it.
3. **Yellow/red card flagging** (on by default; disable with
   `--no-card-detection`): stopped-play windows are scanned for the raised
   card signature (small saturated yellow/red patch persisting across
   frames). Detections become `yellow_card` / `red_card` bookmarks with
   confidence, and review crops are saved to `card_crops/` for
   verification and training.

Broadcast reel (default; `--no-broadcast-reel` for a plain montage):

1. **Story-aware boundaries**: goal clips start where the move began (the
   dead ball or change of attacking direction that launched it) and every
   clip ends when the crowd noise decays back to baseline - not at fixed
   offsets.
2. **`highlights_reel.mp4`**: cold-open teaser of the best moment,
   chronological clips joined with crossfades and per-clip audio
   normalization, **slow-motion replays spliced in after goals**, fade-out
   ending.
3. **Operator deadband**: within a state the camera ignores sub-1.5%-of-frame
   aim changes, so it rests like a human operator instead of micro-hunting.

Debug & training outputs:

1. `--debug` prints every diagnostic; `--log-file` captures a full timestamped
   DEBUG log.
2. `--debug-video` renders `debug_camera_wide.mp4`: the wide frame annotated
   with the crop box, camera-center crosshair, ball + trail, field/goal boxes,
   and a banner stating the game state and **why** the camera is where it is.
3. `--dump-training-data` writes `camera_decisions.jsonl` (every per-frame
   camera decision with reasons) and `ball_track.csv` for tuning/training.
4. Every run writes `analysis_game_states.json` (state segments, goal events,
   field geometry, ball-track stats).

### Run the Studio Web UI

```bash
python -m uvicorn backend.main:app --port 8000
# open http://localhost:8000
```

The web UI is a componentized ES-module app served straight from `frontend/`
(no build step): `frontend/js/views/*` hold the views, `frontend/js/api.js`
the API client, `frontend/css/app.css` the responsive light/dark styles.
It signs in against the API (token or developer mode — no hardcoded tenant),
and provides:

1. **Matches** — per-match dashboard: the baseline 15-stat team catalog with
   per-stat availability flags and evidence drill-down, roster management
   (single entries or CSV template import via `GET /v1/matches/roster-template.csv`),
   manual highlight-to-player assignment, jersey routing, player cards, saved
   team rosters, and share links for the match or any single highlight.
2. **Create** — guided upload wizard: drag-and-drop with progress, pre-flight
   validation against `GET /v1/matches/upload-policy` (size caps, formats,
   minimum length), resolution warnings, a paste-a-link option that discloses
   how many of the 15 stats that provider can support, and an optional notify email.
3. **Jobs** — queue with stage progress, elapsed-vs-SLA turnaround messaging,
   live logs, and completion-notification delivery state.
4. **Runs** — the film review workspace (video views, bookmarks, team panel,
   AI match report).

Upload/notification behavior is configured by environment:

```bash
VH_UPLOAD_MAX_GB=3                  # standard upload cap
VH_UPLOAD_EXTENDED_MAX_GB=8         # cap for tenants with the extended_uploads entitlement
VH_UPLOAD_MIN_DURATION_SECONDS=0    # set 1800 to enforce the 30-minute match minimum
VH_PROCESSING_SLA_HOURS_MIN=4       # turnaround target shown in the UI
VH_PROCESSING_SLA_HOURS_MAX=6
VH_NOTIFY_BACKEND=console           # console | smtp | disabled
VH_SMTP_HOST=... VH_SMTP_PORT=587 VH_SMTP_USERNAME=... VH_SMTP_PASSWORD=... VH_SMTP_FROM=...
```

A tenant gets the extended upload cap when its metadata contains
`{"entitlements": {"extended_uploads": true}}`. Completion emails go to the
job config's `notify_email` (set by the Create wizard) or the match metadata's
`notify_email`, and every attempt is recorded and visible via
`GET /v1/jobs/{job_id}/notifications`.

### Sharing, player cards, and ingest sources

Share links are unguessable tokens that work without an account:

```bash
curl -X POST localhost:8000/v1/matches/<match_id>/shares -d '{"scope":"match"}' -H 'Content-Type: application/json'
# -> {"token": "...", "url_path": "/#share/<token>", ...}
curl localhost:8000/v1/public/shares/<token>     # no auth, no tenant header
```

`scope` is `match`, `highlight` (with `event_id`), or `player_card` (with
`roster_entry_id`). Public payloads are assembled field by field, so
filesystem paths, tenant ids, and reviewer metadata never appear in them.
Revoke with `DELETE /v1/shares/{share_id}`.

Roster and player workflows:

```bash
POST /v1/matches/{id}/roster/route          # attach highlights to players by jersey number
GET  /v1/matches/{id}/roster/{entry}/card   # a player's highlights and tallies
POST /v1/matches/{id}/roster/cards/send     # email every rostered player their card link
POST /v1/matches/{id}/roster/save-template  # save this roster as a reusable team
POST /v1/matches/{id}/roster/apply-template/{template_id}
```

Note on jersey routing: the analysis pipeline does not yet recognize jersey
numbers from video, so `Event.jersey_number` is only populated by manual
assignment or reviewer corrections today. Routing runs automatically after a
job completes and reports exactly what it could and could not place; the
recognition step is the remaining work for full `FR-ROSTER-02`.

`GET /v1/sources` returns the ingest capability matrix — which of the 15
statistics each source can produce. Pasted links are classified on match
creation (stored as `metadata.source_type`), the Create wizard discloses the
coverage before submission, and the stat catalog marks anything the source
cannot support as unavailable rather than reporting zero.

For large match files, including 10GB+ recordings, use the portal's **Local File Path (10GB+)** video source. Use **Browse** or paste a path to register a file that already exists on the API/worker machine and process it in place. The portal preflights the path through the API worker and reports file size, basic media metadata when `ffprobe` is available, and clear messages for missing, zero-byte, cloud-placeholder, or still-copying files. Browser upload is intended only for smaller files.

Leave **Limit to test window** enabled for the first smoke test. The default window processes only the first 2 minutes, which is much faster and safer than starting with a full 10GB match.

### GPU Acceleration

The API exposes GPU readiness at:

```bash
curl http://127.0.0.1:8000/v1/health/gpu
```

On NVIDIA systems, install CUDA-enabled PyTorch instead of the CPU wheel. For this machine's CUDA 13 driver path:

```bash
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install "pillow<12,>=9.2.0"
```

The portal sidebar shows whether PyTorch CUDA is ready and whether FFmpeg can see `h264_nvenc` for GPU clip rendering. When **Require GPU** is enabled, jobs fail early if CUDA is not available.

The main analysis controls include a **GPU Analysis** section. The current default is `yolo26s.pt`, `botsort.yaml`, image size `960`, confidence `0.18`, and frame stride `1`. For heavier GPU use and better small-player detection, try `yolo26m.pt` with image size `1280`; for quick smoke tests, lower image size or increase frame stride. You can also provide a custom `.pt` path from a fine-tuned detector.

### YOLO Detector Training

The Training Lab can launch real Ultralytics detector training from a YOLO dataset YAML. The training run writes a normal Ultralytics `best.pt`; paste that path into **GPU Analysis > Custom Detector Weights** for future processing runs.

API example:

```bash
curl -X POST http://127.0.0.1:8000/v1/training/runs ^
  -H "Content-Type: application/json" ^
  -H "X-Tenant-Id: sandbox" ^
  -d "{\"target_model\":\"yolo-detector\",\"training_config\":{\"kind\":\"ultralytics_yolo\",\"dataset_yaml\":\"C:\\\\datasets\\\\soccer\\\\data.yaml\",\"base_model\":\"yolo26s.pt\",\"epochs\":50,\"imgsz\":960,\"batch\":8,\"device\":\"0\"}}"
```

### Run API Client UI (Streamlit)

```bash
streamlit run app_api.py
```

### Run Global Admin Portal (Streamlit)

```bash
streamlit run app_admin_global.py
```

### Run Tenant Admin Portal (Streamlit)

```bash
streamlit run app_admin_tenant.py
```

### Run V1 API

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use launcher scripts:

```bash
run_api.bat
```

```bash
./run_api.sh
```

### Run Queue Worker (Optional Queue Mode)

Set API execution mode to queue:

```bash
set VH_JOB_EXECUTION_MODE=queue
```

Start API and worker separately:

```bash
run_api.bat
```

```bash
run_worker.bat
```

Validation:

```bash
python test_api_smoke.py
python test_api_auth_queue.py
python -m pytest --cov=backend --cov-report=term-missing
```

or:

```bash
run_tests.bat
```

### Auth and Roles (Configurable)

Default mode is open dev access (`VH_AUTH_REQUIRED=false`).

To require bearer tokens:

```bash
set VH_AUTH_REQUIRED=true
set VH_API_TOKENS=admin-token:admin,coach-token:coach,analyst-token:analyst,tenant-admin-token:tenant_admin
```

Then call APIs with:

```bash
Authorization: Bearer admin-token
```

JWT support is also available:

1. Set `VH_JWT_SECRET` (required for JWT issue/verify)
2. Optional: `VH_AUTH_BOOTSTRAP_KEY` for initial token bootstrap
3. Issue JWT via `POST /v1/auth/token`
4. Inspect current identity via `GET /v1/auth/me`
5. Send `X-Tenant-Id` header (tenant id or slug) for tenant-scoped endpoints

### LLM Analysis Provider

The match assistant endpoints (`/v1/matches/{match_id}/agent/*`) and the per-run AI match report (`match_report.md`, rendered on the run page) can run with fallback summaries, OpenAI, or a local LLM. The same `VH_LLM_*` settings drive both.

OpenAI cloud:

```bash
set VH_LLM_PROVIDER=openai
set VH_LLM_MODEL=gpt-4o-mini
set OPENAI_API_KEY=<your-key>
```

Ollama local:

```bash
set VH_LLM_PROVIDER=ollama
set VH_LLM_MODEL=gemma4:e2b
set VH_LLM_BASE_URL=http://127.0.0.1:11434
```

Check the active AI configuration:

```bash
curl -H "X-Tenant-Id: sandbox" http://127.0.0.1:8000/v1/agent/status
```

The Streamlit match assistant also shows provider, model, reachability, and local setup hints.

OpenAI-compatible local servers (LM Studio, vLLM, llama.cpp server, Ollama `/v1` mode):

```bash
set VH_LLM_PROVIDER=openai_compatible
set VH_LLM_MODEL=local-model-name
set VH_LLM_BASE_URL=http://127.0.0.1:1234/v1
set VH_LLM_API_KEY=local-dev-key
```

Optional timeout override:

```bash
set VH_LLM_TIMEOUT_SECONDS=20
```

By default, Ollama models are unloaded immediately after each assistant response so the GPU stays available for video analysis. Override only if you want faster back-to-back assistant chats:

```bash
set VH_LLM_KEEP_ALIVE=5m
```

### Multi-tenant and Admin API

Tenant-scoped APIs require tenant context. Use request header:

```bash
X-Tenant-Id: <tenant_id_or_slug>
```

Admin surfaces:

1. Global admin API: `/v1/admin/global/*` (tenant, user, membership, inventory controls)
2. Tenant admin API: `/v1/admin/tenant/*` (tenant-scoped users, memberships, summary)

Quick dev mode to test core flows without managing users:

```bash
set VH_SKIP_USER_MANAGEMENT=true
set VH_BASE_TENANT_SLUG=sandbox
set VH_BASE_TENANT_NAME=Sandbox Tenant
```

With this enabled, tenant membership is auto-provisioned on first use for the selected tenant.

Test/Debug logging mode:

```bash
set VH_TEST_MODE=true
set VH_LOG_LEVEL=DEBUG
set VH_JOB_LOG_DETAIL=extreme
```

Per-run logging profiles are also available from the Processing Portal:

1. `Standard`: core status and failure logs.
2. `Detailed`: process-language checkpoints plus technical context for normal testing.
3. `Diagnostic`: detailed logs plus raw config/pipeline invocation checkpoints for deep debugging.

Run Monitor's Log Inspector can show the same logs as a **Process Story**, a **Technical Table**, or raw rows.

Job-level debug endpoints:

1. `GET /v1/jobs/{job_id}/logs` (supports `level`, `stage`, `detail_level`, `limit`)
2. `GET /v1/jobs/{job_id}/diagnostics` (human-readable status summary, likely issue, and next action)
3. `GET /v1/jobs/{job_id}/bookmarks` (live bookmark table from events/job result/manifest)
4. `POST /v1/jobs/{job_id}/kill-session` (fast cancel path for testing)
5. `POST /v1/jobs/{job_id}/rerun` (rerun with optional config/model/event-target overrides)
6. `GET /v1/matches/{match_id}/events?job_id=<job_id>` (bookmark/event table for a specific processing run)
7. `POST /v1/matches/{match_id}/events/{event_id}/clip-on-demand` (frame-accurate bookmark clip rendering with cache reuse)
8. `DELETE /v1/jobs/{job_id}` (delete old run, including job logs and job-linked events)
9. `POST /v1/matches/{match_id}/exports/highlights` (export one highlight reel from selected bookmarks/events)

Codec fallback behavior:

1. Clip export attempts `h264_nvenc` first when available.
2. If NVENC fails at runtime, export auto-falls back to `libx264`, then `mpeg4`.

Analysis-only mode and bookmark outputs:

1. Set job config `"analysis_only": true` to skip clip rendering and generate fast event/bookmark analysis only.
2. Set job config `"trim_start"` and `"trim_end"` in seconds, or use the portal **Limit to test window** control, to process only a short slice of a match.
3. Every run writes `analysis_bookmarks.json` and `analysis_bookmarks.csv` to the job output directory.
4. Completed jobs persist bookmark data in job result payload and auto-create `Event` rows linked to the job.
5. Processing Portal Game Library includes full-match playback with bookmark jump controls.
6. Bookmark rows can render frame-accurate on-demand clips without reprocessing the full match.
7. Operations Console includes bulk queue controls to kill queued/active jobs for a selected match.
8. Game Library includes a Match Workspace for one-click reprocess (latest config) and custom reprocess (same uploaded source video).
9. Processing Portal supports an Experience toggle (`User Friendly` / `Technical`) for non-technical vs advanced workflows.
10. Match Studio supports deleting old runs per match and exporting selected bookmarks into a final highlight reel.

Portal launchers:

```bash
run_admin_global.bat
run_admin_tenant.bat
```

### Storage Backend

Default storage is local filesystem (`VH_STORAGE_BACKEND=local`).

S3-compatible mode:

```bash
set VH_STORAGE_BACKEND=s3
set VH_S3_BUCKET=video-highlights
set VH_S3_ENDPOINT_URL=https://<s3-compatible-endpoint>
set VH_S3_ACCESS_KEY_ID=<key>
set VH_S3_SECRET_ACCESS_KEY=<secret>
set VH_S3_REGION=<region>
set VH_S3_KEY_PREFIX=video-highlights
```

Use `GET /v1/matches/{match_id}/assets/{asset_id}/download-url` to resolve local paths or signed URLs.

### Run Desktop UI

```bash
python VideoHighlightsGUI.py
```

## Docker Hub Publishing (Automated)

Every push to `main` builds the image and publishes it to Docker Hub via
`.github/workflows/docker-publish.yml` (default image:
`geoffvaughan/video-highlights`, tags `latest` + `sha-<commit>`; git tags
like `v1.2.3` publish version tags too).

One-time setup in the GitHub repo (Settings > Secrets and variables > Actions):

1. Secret `DOCKERHUB_USERNAME`: your Docker Hub username.
2. Secret `DOCKERHUB_TOKEN`: a Docker Hub personal access token (Read/Write).
3. Optional variable `DOCKERHUB_IMAGE`: override the image name.

Manual publish from any machine:

```bash
docker login -u <username>
docker build -t <username>/video-highlights:latest .
docker push <username>/video-highlights:latest
```

Run the FULL product (API + worker + all web UIs) without cloning the repo:

```bash
docker pull geoffvaughan/video-highlights:latest
docker run -d --name video-highlights \
  -p 8000:8000 -p 8501:8501 -p 8502:8502 -p 8503:8503 -p 8504:8504 \
  -v C:\VideoHighlights\data:/app/data \
  geoffvaughan/video-highlights:latest
```

Then open:

- **Studio (web UI)**: http://localhost:8000 - Library & Review (toggle
  Original / Debug / Zoom / Reel / Clips), Create & Process, and Job
  tracking in one app served by the API itself.
- API docs: http://localhost:8000/docs

In Docker Desktop's "Run a new container" dialog this means: map every
listed port to the same host port, and add a volume from a Windows folder
(e.g. `C:\VideoHighlights\data`) to `/app/data` so the database, uploads,
and rendered outputs persist. No environment variables are required - the
image ships single-container defaults (override any `VH_*` value with `-e`
if needed).

One-off CLI processing of a single video (no UI):

```bash
docker run --rm -v C:\Path\To\Videos:/media geoffvaughan/video-highlights:latest \
  python VideoHighlights.py --video /media/match.mp4 --out /media/highlights_out \
  --camera-mode follow_ball --zoom-factor 1.8 --debug-video
```

## Maximize Your Hardware

Two container images ship with the repo:

| Image | File | For |
| --- | --- | --- |
| CPU (default) | `Dockerfile` | Any machine; published to Docker Hub on merge to `main` |
| GPU | `Dockerfile.gpu` | NVIDIA rigs (x86_64) and NVIDIA DGX Spark (arm64) |

### High-End PC ("hardcore rig")

Host setup: NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (bundled with Docker Desktop on Windows - just enable GPU support).

Build and run the GPU image with everything opened up:

```bash
docker build -f Dockerfile.gpu -t video-highlights:gpu .

docker run -d --name vh-gpu \
  --gpus all \
  --ipc=host --shm-size=8g \
  --cpus="0.000" --memory=0 \
  -p 8000:8000 \
  -v vh-outputs:/app/outputs -v vh-storage:/app/storage \
  video-highlights:gpu
```

- `--gpus all` exposes every GPU; `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` is baked into the image so ffmpeg can use **h264_nvenc** for GPU-encoded rendering, not just CUDA inference.
- `--cpus="0.000" --memory=0` means *no limit* - the container may use every core and all RAM. On Docker Desktop for Windows also raise the WSL2 limits in `%UserProfile%\.wslconfig` (e.g. `memory=48GB`, `processors=24`), because WSL2 caps Docker below your real hardware by default.
- `--ipc=host --shm-size=8g` prevents PyTorch shared-memory stalls on big videos.
- Mount your recordings folder read-only (e.g. `-v D:\matches:/videos:ro` on Windows) and use the Create page's **Local file on the server** source with `/videos/match.mp4` - no upload step at all.

Then push the quality knobs up in the Create page (they map to job config):

| Knob | Fast test | Max quality (big GPU) |
| --- | --- | --- |
| `inference_imgsz` | 736 | 1280-1536 |
| `vid_stride` | 2 | 1 |
| Detector | `yolo26s.pt` | `yolo26m.pt` / `yolo26l.pt` |
| Processing window | First 10 min | Full match |

Verify the GPU is live: `curl http://localhost:8000/v1/health/gpu` and check the job log for `Using device: cuda` and the GPU name.

### NVIDIA DGX Spark

The Spark is arm64 (Grace + Blackwell GB10) with a CUDA 13 driver stack, so build the GPU image natively on it with the cu130 wheel index:

```bash
# on the Spark
git clone https://github.com/Aleut-Geoff-Vaughan/Video-Highlights.git && cd Video-Highlights
docker build -f Dockerfile.gpu \
  --build-arg CUDA_IMAGE=nvidia/cuda:13.0.0-runtime-ubuntu24.04 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 \
  -t video-highlights:spark .

docker run -d --name vh-spark --gpus all --ipc=host --shm-size=16g \
  -p 8000:8000 -v vh-outputs:/app/outputs -v vh-storage:/app/storage \
  video-highlights:spark
```

Spark tuning: its unified 128 GB memory removes the usual VRAM ceiling - run `inference_imgsz` 1280+, `vid_stride` 1, and the larger detector weights without concern, and it comfortably co-hosts an Ollama model alongside the video worker for the AI match report (see below). To cross-build from an x86 machine instead: `docker buildx build --platform linux/arm64 ...` with the same build args.

### Local AI (Ollama) for Match Reports

Each processing run can end with an AI-written match report (`match_report.md`, shown on the run page). YOLO remains the analysis engine - the LLM only reads the structured results (goals, cards, possession, set pieces, data-quality stats) and writes the narrative plus tuning suggestions. Wire it to any local or remote OpenAI-compatible endpoint:

```bash
# Ollama running on the host (or on the Spark)
docker run ... \
  -e VH_LLM_PROVIDER=ollama \
  -e VH_LLM_BASE_URL=http://host.docker.internal:11434 \
  -e VH_LLM_MODEL=llama3.1:8b \
  video-highlights:gpu
```

Any OpenAI-compatible API works the same way (`VH_LLM_PROVIDER=openai_compatible`, `VH_LLM_BASE_URL=...`, `VH_LLM_API_KEY=...`). With no provider configured the report falls back to a deterministic template, and the **AI match report** checkbox on the Create page turns it off per run.

## Docker Desktop Launch (This Repo)

1. Open Docker Desktop and confirm the engine is running.
2. From repo root, launch the stack:

```bash
docker compose up --build api worker
```

3. Open the Studio portal at `http://localhost:8000` (API docs at `http://localhost:8000/docs`).

Windows convenience:

```bash
run_docker.bat
```

Stop all containers quickly:

```bash
stop_docker.bat
```

GPU worker profile (builds `Dockerfile.gpu`; requires NVIDIA toolkit and Docker GPU support):

```bash
docker compose --profile gpu up --build api worker-gpu
```

Windows convenience:

```bash
run_docker_gpu.bat
```

## Documentation

1. `PRD.md`: Product requirements and acceptance criteria
2. `ROADMAP.md`: Phase plan and milestone scope
3. `REQUIREMENTS_TRACEABILITY.md`: Mapping of context-window requirements to PRD and roadmap
4. `LOCAL_SETUP.md`: Local and Docker setup guidance
5. `PERFORMANCE_IMPROVEMENTS.md`: Implemented optimizations in current code
6. `PERFORMANCE_RECOMMENDATIONS.md`: Next optimization opportunities
7. `FEEDBACK_EVENT_API_SCHEMA.md`: Concrete event/feedback payload schema and API endpoints
8. `IMPLEMENTATION_STATUS.md`: Current build status and next autonomous implementation queue
9. `TESTING.md`: Automated testing framework, local commands, and CI
