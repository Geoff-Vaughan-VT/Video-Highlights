# Testing

## Commands

```bash
python -m pytest -q -p no:cacheprovider            # whole suite (CI command)
python -m pytest -q -p no:cacheprovider tests/test_perf_profiles.py tests/test_gpu_status.py
python -m pytest --cov=backend --cov-report=term-missing
./run_tests.sh   |   run_tests.bat                 # same as the first line, extra args pass through
python bench/bench_match.py --minutes 0.2 --height 360   # bench smoke (CI job bench-smoke)
```

`pytest.ini` sets `pythonpath = .`, `testpaths = tests`, `addopts = -q`.
`-p no:cacheprovider` keeps `.pytest_cache` out of the tree (and out of
read-only checkouts).

## Dependencies

| File | Use |
|---|---|
| `requirements-dev.txt` | Full suite + bench: ML stack + pytest. Install CPU torch first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`. |
| `requirements-backend-test.txt` | API-only subset (no torch/ultralytics). Tests needing the ML stack or ffmpeg skip or fail. |

ffmpeg on `PATH` is required for the media tests (frame source, render,
clips, synthetic match via ffmpeg).

## Layout

* `tests/conftest.py`: isolated SQLite DB per test (`isolated_db`),
  `client` / `auth_client` TestClients with settings reset per test.
* API and control plane: `test_health`, `test_matches_events`,
  `test_feedback_training_agent`, `test_auth_and_queue`, `test_validation_errors`,
  `test_jwt_auth`, `test_storage_backends`, `test_contract_api`,
  `test_multitenancy_admin`, `test_dev_skip_user_management`,
  `test_job_logging_and_kill`, `test_job_bookmarks_analysis`,
  `test_event_clip_on_demand`, `test_job_delete_and_live_bookmarks`,
  `test_highlight_export_selected`, `test_rbac_and_upload`, `test_studio_api`.
* Product features: `test_match_stats`, `test_roster_and_assignment`,
  `test_roster_templates`, `test_player_routing`, `test_sharing`,
  `test_source_catalog`, `test_upload_validation`, `test_notifications`,
  `test_llm_agent_local`, `test_match_report`, `test_yolo_training`.
* Pipeline (v2 workstreams): `test_frame_source`, `test_detectors`,
  `test_tracking_engine`, `test_player_focus`, `test_team_classification`,
  `test_game_tracking`, `test_set_pieces_and_cards`, `test_goal_bookmarks`,
  `test_camera_planner`, `test_camera_render`, `test_follow_cam`,
  `test_full_follow_cam_export`, `test_event_follow_cam_router`,
  `test_broadcast`, `test_highlight_fallback`, `test_media_timeline`,
  `test_audio_editor`. Most run on `backend/services/synthetic_match.py`
  footage with exact ground truth.
* Platform: `test_perf_profiles` (profile table, `resolve_job_config`
  precedence, model family swap, `resolve_model_path`, `estimate_runtime`
  shape and ordering, `classify_hardware` for every class) and
  `test_gpu_status` (mocked `subprocess.run` + fake `torch` module for an
  RTX rig, Apple Silicon, NVENC-listed-but-no-GPU, no ffmpeg/torch, and
  `VH_DEVICE` override).
* Root-level `test_api_smoke.py`, `test_api_auth_queue.py`,
  `test_performance.py` are manual scripts against a running API, not part
  of the pytest run.

## CI (`.github/workflows/ci.yml`)

* `tests`: Python 3.11, apt ffmpeg, CPU torch + `requirements-dev.txt`,
  `bash -n` on every shell script, compose YAML parse, then
  `python -m pytest -q -p no:cacheprovider`.
* `bench-smoke`: CPU torch + `requirements-ml.txt`, cached weights,
  `bench/bench_match.py --minutes 0.2 --height 360 --detect-frames 4`,
  uploads `bench/results/*` as an artifact. CPU fps are for harness checks
  only.

`docker-publish.yml` builds the images on `v*` tags (see `docs/DEPLOYMENT.md`).

## Conventions

1. Each test gets its own SQLite DB; queue mode avoids background threads.
2. Hardware is mocked, never required: torch via `sys.modules`, ffmpeg and
   `nvidia-smi` via `subprocess.run` monkeypatching.
3. Media tests use synthetic footage with ground truth, generated in
   `tmp_path`, seconds long.
4. `PLAN.md` acceptance criteria (camera smoothness, stats accuracy, both
   scripted goals found, decode-pass count) are being turned into tests on
   the synthetic match during the v2 integration.
