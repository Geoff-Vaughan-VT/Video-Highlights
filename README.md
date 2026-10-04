# Video Highlights

Turns one panoramic match recording (xBotGo Falcon 4K, Veo, any wide static
camera) into:

1. a smooth 1080p game-camera movie (`full_follow_ball_zoom.mp4`),
2. corroborated highlights and a reel (`highlight_NN.mp4`, `highlights_reel.mp4`),
3. team and per-player stats (possession, distance, speed, heatmaps, shots),
4. follow-any-player re-renders without re-running detection.

FastAPI backend + queue worker + a no-build web Studio (`frontend/`), YOLO
(Ultralytics) detection and tracking, ffmpeg for every decode/encode. Runs
natively or in Docker on Windows/Linux NVIDIA rigs, natively on Apple Silicon
(MPS + VideoToolbox), on an NVIDIA DGX Spark (arm64, CUDA 13), and in any
container host with Postgres + S3 for cloud use.

The v2 rebuild plan, the reasons the old pipeline took 40 hours per match,
and the workstreams are in [`PLAN.md`](PLAN.md). The run-directory contract
(every file a run writes) is [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md).
Deployment details: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Quick start

All paths end at the Studio on http://localhost:8000 (API docs at `/docs`,
GPU/encoder probe at `/v1/health/gpu`).

**Windows 11 + NVIDIA, Docker Desktop** (WSL2 backend, GPU support on):

```powershell
git clone https://github.com/Aleut-Geoff-Vaughan/Video-Highlights.git; cd Video-Highlights
copy .env.example .env    # set VH_MEDIA_DIR=D:/Videos (mounted read-only at /media)
run_docker_gpu.bat        # api + worker-gpu; run_docker.bat for CPU only
```

**Windows 11 + NVIDIA, native** (Python 3.11, `winget install Gyan.FFmpeg`):

```powershell
git clone https://github.com/Aleut-Geoff-Vaughan/Video-Highlights.git; cd Video-Highlights
run_native_windows.bat    # .venv, torch cu128, API + worker
```

**Apple Silicon Mac Studio / MacBook** (native only; Docker on macOS has no GPU):

```bash
brew install python@3.11 ffmpeg
git clone https://github.com/Aleut-Geoff-Vaughan/Video-Highlights.git && cd Video-Highlights
scripts/run_native_mac.sh # .venv, torch with MPS, VideoToolbox ffmpeg, API (inline jobs)
```

**NVIDIA DGX Spark / Linux + NVIDIA** (Docker + NVIDIA Container Toolkit):

```bash
git clone https://github.com/Aleut-Geoff-Vaughan/Video-Highlights.git && cd Video-Highlights
cp .env.example .env      # set VH_MEDIA_DIR
./run_docker_gpu.sh       # arm64 builds Dockerfile.gpu with CUDA 13 / cu130 automatically
```

Native Linux / Spark without Docker: `scripts/run_native_linux.sh` (picks
cu130 on aarch64, cu128 on x86_64, CPU without `nvidia-smi`).

Source videos: put them in `VH_MEDIA_DIR` and register them in the Studio as
**Local file on the server** (`/media/<file>.mp4` in Docker, the real path
natively). No upload, no copy.

## Profiles and expected runtime

`profile` in the job config selects defaults; any key set explicitly in the
job config wins (`backend/services/perf_profiles.py::resolve_job_config`).

| key | fast | balanced (default) | quality |
|---|---|---|---|
| `proxy_height` | 720 | 1080 | 1080 |
| `inference_imgsz` | 960 | 1280 | 1536 |
| `vid_stride` | 2 | 1 | 1 |
| `yolo_model` | `yolov8n.pt` | `yolov8s.pt` | `yolov8m.pt` |
| `batch_size` | auto | auto | auto |
| `output_height` | 1080 | 1080 | 1440 |
| `debug_video` | off | off | off |
| `ball_tiles` | off | off | on |
| `tracker_config` | `bytetrack.yaml` | `botsort.yaml` | `botsort.yaml` |

`VH_MODEL_FAMILY=yolo26` (or `yolo11`) swaps the profile weights for the same
size in that family (`yolo26s.pt` for balanced); a `yolo_model` set in the
job (a path to a fine-tuned `best.pt`, say) is never swapped.

Estimated wall time for a **90-minute 4K30 match**
(`perf_profiles.estimate_runtime`; planning numbers, measure your machine
with `python bench/bench_match.py --minutes 2 --height 2160`):

| hardware class | fast | balanced | quality |
|---|---:|---:|---:|
| `rtx_4090` (4090/5090) | 18 min | 28 min | 54 min |
| `rtx_4080` (4080/5080) | 20 min | 32 min | 73 min |
| `rtx_3080` (3080/3090/4070) | 24 min | 42 min | 110 min |
| `dgx_spark` (GB10) | 26 min | 41 min | 88 min |
| `apple_m2_ultra` (M2/M3 Ultra, M4 Max) | 30 min | 57 min | 157 min |
| `apple_m1_max` (M1-M3 Max and smaller) | 46 min | 103 min | 334 min |
| `cpu_8core` | 4.2 h | 12.7 h | 52 h |

Balanced on an RTX 4080 breaks down as proxy 6.0 min (NVDEC + NVENC),
detect+track 16.1 min (imgsz 1280, batch 16, fp16), analysis 1.8 min, final
1080p render 7.7 min, clips + reel 0.5 min. Detection dominates; `fast`
halves it with stride 2 and a smaller model, TensorRT (`VH_TENSORRT=1`,
GPU image built with `INSTALL_TENSORRT=1`) cuts it by roughly 1.8x.
`GET /v1/health/gpu` reports this machine's `hardware_class`.

## Architecture (v2)

```
source.mp4 (4K)
   │  ffmpeg (hwaccel) ── once ──► proxy_1080p.mp4 + audio_analysis.wav + thumbs/
   ▼
tracking_engine (proxy) ──► TrackingResult (all players + ball, teams per track) ──► tracks.npz
   ├─► game_tracking (ball track, field geometry, states, goals, set pieces)
   ├─► match_stats (homography, per-player + team stats) ──► analysis_player_stats.json, analysis_team_stats.json
   ├─► event_engine (shots/saves/chances/goals/cards/sprints, excitement, reel plan) ──► analysis_events.json
   └─► camera_planner v2 ──► camera_decisions.jsonl + camera_crops.txt + camera_quality.json
                                   │
   source.mp4 ── ffmpeg crop(sendcmd)+scale+NVENC ── once ──► full_follow_ball_zoom.mp4 (1080p)
                                   │  stream-copy cuts
                                   ▼
                      highlight_NN.mp4 ──► highlights_reel.mp4 (ffmpeg xfade)
```

Track once, analyze many times: re-running stats, re-planning the camera for
another player, or building a different reel never re-detects. The 4K source
is decoded exactly twice per run (proxy + final render). Every output file is
listed in [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md).

Processes: `api` (FastAPI + Studio UI, also serves files and on-demand clips)
and `worker` (`python -m backend.worker`, claims queued jobs). With
`VH_JOB_EXECUTION_MODE=inline` the API runs jobs itself (single-user native
installs). State: SQLite (`VH_DB_URL`) or Postgres; files under
`VH_OUTPUT_ROOT` / `VH_LOCAL_STORAGE_ROOT` or S3 (`VH_STORAGE_BACKEND=s3`).

## Product capability targets

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

## Usage

### Install (manual)

Install the torch build for your hardware first, then the rest
(`requirements.txt` = ML stack + test tools; `requirements-base.txt` is the
API-only set):

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA
# pip install torch torchvision                                                    # Apple Silicon (MPS)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 # DGX Spark
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU
pip install -r requirements.txt
python scripts/download_models.py          # yolov8n.pt + yolov8s.pt into ./models
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
VH_UPLOAD_MAX_GB=0                  # upload size cap in GB; 0 = no limit (default, self-hosted)
VH_UPLOAD_EXTENDED_MAX_GB=0         # optional higher cap for tenants with the extended_uploads entitlement
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

For large match files, including 10GB+ recordings, use the **Local file on the server** video source (in Docker the host folder `VH_MEDIA_DIR` is mounted read-only at `/media`, so paths look like `/media/match.mp4`). Use **Browse** or paste a path to register a file that already exists on the API/worker machine and process it in place. The portal preflights the path through the API worker and reports file size, basic media metadata when `ffprobe` is available, and clear messages for missing, zero-byte, cloud-placeholder, or still-copying files. Browser upload is intended only for smaller files.

Leave **Limit to test window** enabled for the first smoke test. The default window processes only the first 2 minutes, which is much faster and safer than starting with a full 10GB match.

### GPU and hardware acceleration

`GET /v1/health/gpu` reports everything the pipeline uses to pick hardware:
`torch` (CUDA/MPS), `nvidia_smi`, `mps_available`, ffmpeg `hwaccels` and
`encoders` (`h264_nvenc`, `hevc_nvenc`, `h264_videotoolbox`, `libx264`),
`recommended_device` (`cuda`/`mps`/`cpu`, overridable with `VH_DEVICE`),
`recommended_encoder`, `recommended_hwaccel`, and the `hardware_class` used
for runtime estimates. When **Require GPU** is enabled, jobs fail early if
CUDA is not available.

Torch indexes: `cu128` for RTX 30/40/50 on Windows/Linux x86_64, `cu130` for
the DGX Spark (arm64, CUDA 13 driver), the default PyPI wheel on Apple
Silicon (MPS), `cpu` elsewhere. Ubuntu 24.04's ffmpeg (used by the GPU image)
already supports `-hwaccel cuda` and `h264_nvenc`; on Windows use the Gyan
build (`winget install Gyan.FFmpeg`); on macOS Homebrew's ffmpeg has
VideoToolbox.

Detector weights resolve from `VH_MODEL_DIR` (`/models` in the images,
`./models` natively) and are baked into the images at build time
(`MODELS` build arg). Larger models: `python scripts/download_models.py
yolov8m.pt yolo26s.pt`. A custom `.pt` path from a fine-tuned detector works
anywhere a model name does.

### YOLO Detector Training

The Training Lab can launch real Ultralytics detector training from a YOLO dataset YAML. The training run writes a normal Ultralytics `best.pt`; paste that path into **GPU Analysis > Custom Detector Weights** for future processing runs.

API example:

```bash
curl -X POST http://127.0.0.1:8000/v1/training/runs ^
  -H "Content-Type: application/json" ^
  -H "X-Tenant-Id: sandbox" ^
  -d "{\"target_model\":\"yolo-detector\",\"training_config\":{\"kind\":\"ultralytics_yolo\",\"dataset_yaml\":\"C:\\\\datasets\\\\soccer\\\\data.yaml\",\"base_model\":\"yolo26s.pt\",\"epochs\":50,\"imgsz\":960,\"batch\":8,\"device\":\"0\"}}"
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
python -m pytest -q -p no:cacheprovider
```

or `run_tests.bat` / `./run_tests.sh`. See [`TESTING.md`](TESTING.md).

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

## Deployment

Full guide: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Summary:

| Target | How | GPU path |
|---|---|---|
| Windows 11 + RTX | `run_docker_gpu.bat` (Docker Desktop, WSL2) or `run_native_windows.bat` | CUDA torch cu128, NVDEC/NVENC |
| Apple Silicon | `scripts/run_native_mac.sh` (native only) | torch MPS, VideoToolbox |
| DGX Spark (arm64) | `./run_docker_gpu.sh` (builds cu130 image) or `scripts/run_native_linux.sh` | CUDA 13 torch cu130, NVDEC/NVENC |
| Any Linux / cloud host | `docker compose up -d` (CPU) or `--profile gpu`, Postgres via `--profile cloud`, S3 via `VH_STORAGE_BACKEND=s3` | as above |

Images (`Dockerfile` CPU, `Dockerfile.gpu` NVIDIA) are multi-stage, run as the
non-root user `vh`, have a `HEALTHCHECK` on `/v1/health`, and bake detector
weights into `/models`. Compose services: `api`, `worker` (CPU), `worker-gpu`
(profile `gpu`), `postgres` (profile `cloud`); data lives in the named volume
`vh-data` (`/data`), weights in `vh-models` (`/models`), source media is the
read-only bind mount `${VH_MEDIA_DIR}:/media`. Configure with `.env`
(`cp .env.example .env`; every `VH_*` variable is documented there).

Single container, no clone (API + worker in one container):

```bash
docker run -d --name video-highlights -p 8000:8000 \
  -v vh-data:/data -v D:/Videos:/media:ro \
  geoffvaughan/video-highlights:latest            # :gpu (add --gpus all) / :gpu-arm64 for the Spark
```

Images are published on `v*` tags by `.github/workflows/docker-publish.yml`
(`latest` CPU amd64+arm64, `gpu` amd64 cu128, `gpu-arm64` arm64 cu130).

Local AI for match reports (Ollama on the host):
`VH_LLM_PROVIDER=ollama`, `VH_LLM_BASE_URL=http://host.docker.internal:11434`,
`VH_LLM_MODEL=llama3.1:8b` in `.env`.

## Documentation

1. `PLAN.md`: v2 rebuild plan, root causes, workstreams, acceptance criteria
2. `docs/ARTIFACTS.md`: run-directory contract (every output file and schema)
3. `docs/DEPLOYMENT.md`: Docker, GPU toolkit, Apple native, Spark, Postgres/S3, limitations
4. `LOCAL_SETUP.md`: per-platform native and Docker setup, troubleshooting
5. `PERFORMANCE_IMPROVEMENTS.md`: what v2 changes for speed and why
6. `PERFORMANCE_RECOMMENDATIONS.md`: tuning guide and next optimizations
7. `TESTING.md`: test suite, CI, bench
8. `PRD.md`, `ROADMAP.md`, `REQUIREMENTS_TRACEABILITY.md`: product requirements and phases
9. `FEEDBACK_EVENT_API_SCHEMA.md`: event/feedback payload schema and endpoints
10. `IMPLEMENTATION_STATUS.md`: build status
