# Video Highlights: Rebuild Plan (v2)

Goal: turn a single xBotGo Falcon 4K panoramic recording into (1) a smooth,
watchable game-camera movie, (2) legitimate highlights, (3) world-class team
and player stats, with the option to pick and follow any player. Runs locally
first (Windows NVIDIA rig, Apple Studio, DGX Spark), containerized for the
cloud later, and a 90-minute match must finish in well under an hour on a
modern GPU, not 40 hours.

## Where time went (why runs took 40 hours)

| Cause | Evidence | Fix |
|---|---|---|
| 3-4 full 4K decode passes per run (tracking, debug video, full movie, trim re-encode) plus seek-heavy team/card scans | `VideoHighlights.py:766, 2573, 2596, 327`; `team_classification.py:120`; `card_detection.py:249` | One ffmpeg pass makes a 1080p proxy + analysis audio. Every analysis stage reads the proxy. The 4K source is decoded exactly once more, by ffmpeg, for the final render. |
| Render loop is per-frame Python: CPU decode of 4K, crop, **upscale back to 4K**, 25 MB raw pipe write per frame | `camera_render.py:288-309`, `follow_cam.py:183-196` | ffmpeg-native render: `crop` driven by a per-frame `sendcmd` script, `scale` to 1080p, NVENC/VideoToolbox/x264, audio muxed in the same command. Zero Python per frame. |
| Inference batch=1 at imgsz 960 on CPU-decoded 4K frames; `vid_stride` still decodes skipped frames | `VideoHighlights.py:766-779` | Threaded proxy decode, batched inference (auto batch by VRAM), fp16, optional TensorRT engine cache on CUDA, MPS on Apple. |
| moviepy in hot paths (reel, wide clips, montage, overlays) | `broadcast.py:363`, `VideoHighlights.py:1245` | ffmpeg concat/xfade; clips cut from the finished movie with stream copy. |
| On-demand clips seek after `-i` (decode from 0:00) | `event_clip_renderer.py:60` | `-ss` before `-i`. |
| Debug video, full movie, reel, cards, auto colours and LLM all on by default in the UI | `frontend/index.html:235-267` | Profiles (`fast`/`balanced`/`quality`); debug video off by default. |

Target budget for a 90-minute 4K30 match on an RTX 4080-class GPU:
proxy pass ~6 min (NVDEC), detection+tracking ~12-20 min at 1280 batch 16,
analysis < 2 min, final 1080p render ~8 min (NVDEC+NVENC), clips+reel ~3 min.
Total ~30-40 min. DGX Spark similar; Apple Studio (MPS + VideoToolbox) ~2-3x
slower on detection. CPU-only is a fallback, not a target.

## Why the camera is bad today

* The hard "keep point in frame" constraint is applied **after** smoothing,
  per frame, as a clamp (`camera_planner.py:488-499`), so the camera snaps
  whenever smoothing eases the ball toward the crop edge.
* Zoom follows the ball speed and goal distance frame by frame with no
  hysteresis; the deadband creates step changes that the zero-phase filter
  turns into visible zoom pumping.
* Output is upscaled 4K from a 1/1.8 crop (soft image) and the crop size is
  rounded to integers per frame (sub-pixel jitter).
* Only the ball is used; the planner has no notion of the attacking team's
  shape, so the shot is often too tight to understand the play.

Fix (camera planner v2): plan on a smooth low-frequency target, enforce
constraints by *zooming out* rather than snapping pan, give zoom strong
hysteresis and a rate limit (max ~0.12x/s, min dwell 4 s), bound max zoom so
every output pixel is a real source pixel (4K->1080p: zoom <= 2.0), add
"play width" framing (fit the ball and the players engaged in the play),
and publish a smoothness report with hard thresholds that tests enforce.

## Why stats and highlights are not credible today

* Only one player's track is kept; everyone else is an ID-less 10 Hz point
  cloud (`VideoHighlights.py:815-877`). Per-player stats are impossible.
* Team labelling stops after ~2-4 minutes because of a row cap
  (`team_classification.py:67,119`), so possession and second-half goal
  attribution are wrong.
* No pitch model: everything is in pixels; no metres, speeds, distances.
* Highlights are driven by one arbitrary player's px/s speed plus audio; a
  speed+audio overlap is labelled "goal" at 0.85 confidence
  (`VideoHighlights.py:420-424`). Goal detection accepts a single sample in
  an estimated goal box (`game_tracking.py:758`).

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

Track once, analyze many times: re-running stats, re-planning the camera
for a different player, or building a different reel never re-detects.

## Workstreams (Wave 1, parallel, isolated file ownership)

| # | Workstream | Owns | Deliverables |
|---|---|---|---|
| A | Frame source + tracking engine | `backend/services/frame_source.py`, `tracking_engine.py`, `detectors.py`, `player_focus.py`, `device.py` | Proxy pass; batched inference; all tracks retained; ByteTrack/BoT-SORT; re-ID stitching with appearance; ground-truth detector for tests; device auto (cuda/mps/cpu); TensorRT cache; progress + cancel. |
| B | Camera planner v2 + ffmpeg-native render | `camera_planner.py`, `camera_render.py`, `follow_cam.py` | Constraint-aware smoothing, zoom hysteresis, 1080p output, sendcmd render, smoothness metrics + tests, scorebug via ffmpeg drawtext. |
| C | Stats engine | `match_stats.py`, `pitch_calibration.py`, `team_classification.py` | Homography, per-player and team stats, per-track team labels (fixes the cap bug), timelines, heatmaps. |
| D | Event engine + reel | `event_engine.py`, `game_tracking.py` (goal/set-piece quality), `broadcast.py`, `event_clip_renderer.py` | Shots/saves/chances/sprints, corroborated goals, excitement ranking, reel plan, ffmpeg reel. |
| E | Studio UI + studio API | `frontend/`, `backend/routers/studio.py` | Match-centric library, player page with timeline + markers, player picker ("track this player"), stats dashboards, reel builder, job progress with ETA, cancel/rerun/delete. |
| F | Containers, profiles, bench, docs | `Dockerfile*`, `docker-compose.yml`, `docker/`, `scripts/`, `bench/`, `.github/`, `*.md` | Fixed compose with media mount + env file, Apple native runner, model baked in, non-root + healthcheck, perf bench with ETA projection, updated docs. |

Wave 2 (integration, single agent): wire A-D into `VideoHighlights.py` /
`job_runner.py`, typed `JobConfig` with profiles, `focus_track_id` and
`reuse_tracking_from_job` (re-render without re-tracking), progress.json,
cancel hook, fix the pre-existing failing test.

Wave 3 (verification): code review, full pytest, synthetic end-to-end run,
real YOLO smoke, commit, PR.

## Acceptance criteria

1. One synthetic 20 s match runs end to end on CPU in CI in < 3 min and
   produces every artifact in `docs/ARTIFACTS.md`.
2. Camera quality: pan speed p95 < 0.9 crop-widths/s, zoom rate p95 <
   0.15x/s, zero hard snaps, ball in frame > 97% of in-play frames (on the
   synthetic ground truth).
3. Stats: on synthetic ground truth, possession within 5 points of truth,
   goal attribution 100%, per-player distance within 10% of truth.
4. Events: both scripted goals found, zero false goals on the synthetic match.
5. Full source decode passes per run: exactly 2 (proxy + final render), 1 in
   analysis-only mode.
6. Studio: a match can be uploaded, processed, watched with event markers,
   a player clicked and re-rendered as a follow-player movie, and stats
   viewed, without touching the API docs.
