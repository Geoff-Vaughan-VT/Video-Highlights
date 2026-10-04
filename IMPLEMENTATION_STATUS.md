# Implementation Status

This file tracks concrete implementation progress against the documented V1 architecture and requirements.

## Completed in This Pass

## Backend foundation

1. Added FastAPI service at `backend/main.py`.
2. Added persistent SQLite data layer via SQLModel in `backend/models.py`.
3. Added routers for health, matches, jobs, events, feedback, training, and agent APIs.
4. Added cursor-based list pagination helpers.
5. Added request ID middleware and standardized error envelope responses.
6. Added role-based authorization scaffolding (`admin`, `analyst`, `coach`, `parent`, `system`).
7. Added JWT-based authentication support (`/v1/auth/token`, `/v1/auth/me`) with optional bootstrap issuance flow.

## Multi-tenant control plane

1. Added tenant/user/membership models and tenant-scoped foreign keys across core domain objects.
2. Added tenant context resolution and membership enforcement dependency layer (`backend/tenant.py`).
3. Applied tenant isolation across match, job, event, feedback, training, and agent endpoints.
4. Added global admin API router (`/v1/admin/global/*`) for tenant, user, and membership management.
5. Added tenant admin API router (`/v1/admin/tenant/*`) for tenant-scoped user and membership management.
6. Added tenant-aware auth response and JWT issuance fields (`tenant_id`, `is_global_admin`).

## Processing orchestration

1. Added background job runner in `backend/services/job_runner.py`.
2. Connected processing jobs to existing pipeline `VideoHighlights.process_video_highlights(...)`.
3. Added job result artifact persistence (`output_dir`, generated MP4 list).
4. Added queue execution mode (`VH_JOB_EXECUTION_MODE=queue`) plus dedicated worker loop (`backend/worker.py`).
5. Added worker trigger endpoint for controlled processing (`POST /v1/jobs/worker/run-once`).
6. Added job retry endpoint (`POST /v1/jobs/{job_id}/retry`).
7. Added job debug logging store (`job_logs`) with detail levels (`basic`, `detailed`, `extreme`).
8. Added job log endpoint (`GET /v1/jobs/{job_id}/logs`) and fast test cancel endpoint (`POST /v1/jobs/{job_id}/kill-session`).
9. Added `cancel_requested` lifecycle handling for easier test-session shutdown behavior.
10. Added job rerun endpoint with config override support (`POST /v1/jobs/{job_id}/rerun`).
11. Added match-level processing mechanics history persistence (`metadata.processing_history`).
12. Added runtime codec fallback for clip export (`h264_nvenc` -> `libx264` -> `mpeg4`) to avoid NVENC export failures.
13. Added analysis-only processing mode (`analysis_only`) for fast detection/bookmark-only runs.
14. Added analysis bookmark artifacts (`analysis_bookmarks.json`, `analysis_bookmarks.csv`) generated for every run.
15. Added job-run bookmark ingestion into `Event` rows for review/feedback workflows.
16. Added event list filter by processing job (`GET /v1/matches/{match_id}/events?job_id=...`).
17. Expanded processing portal game library with full-match playback and bookmark jump controls.
18. Added frame-accurate clip-on-demand endpoint for event bookmarks with storage-backed caching (`POST /v1/matches/{match_id}/events/{event_id}/clip-on-demand`).
19. Added portal UI controls to render/play bookmark clips without full job reprocessing.
20. Added job bookmark feed endpoint (`GET /v1/jobs/{job_id}/bookmarks`) for live UI updates while runs are active.
21. Added run deletion endpoint (`DELETE /v1/jobs/{job_id}`) with linked job-log and event cleanup.
22. Added selected-bookmark highlight export endpoint (`POST /v1/matches/{match_id}/exports/highlights`).
23. Added user-friendly experience mode with top-level workflow nav and match-centered run management.
24. Added player ROI selection plumbing from portal/API job config into the YOLO tracking pipeline.
25. Added follow-cam clip rendering modes (`wide`, `follow_player`, `follow_action`) with zoom controls.
26. Added `analysis_tracking.json` manifests and event evidence links so on-demand clips and exports can reuse tracking.
27. Fixed follow-cam timestamp alignment for trimmed processing runs.

## LLM + feedback loop scaffolding

1. Added LLM agent service abstraction in `backend/services/llm_agent.py`.
2. Added natural-language query and event explanation endpoints.
3. Added structured feedback submission/review endpoints.
4. Added feedback batch and training-run endpoints with retraining pipeline scaffolding.
5. Enforced review-role restrictions on approval flows.
6. Added OpenAI-compatible and Ollama/local provider paths for match-assistant responses.
7. Added portal match-assistant UI for match summaries, missed-moment review prompts, coaching notes, and selected-event explanations.

## Storage foundation

1. Added storage abstraction service in `backend/services/storage.py`.
2. Switched match asset upload endpoint to storage backend interface (local backend implemented).
3. Added S3-compatible backend support and signed download URL resolution endpoint.

## Tooling and docs

1. Added API run scripts: `run_api.sh`, `run_api.bat`.
2. Added concrete schema doc: `FEEDBACK_EVENT_API_SCHEMA.md`.
3. Added API smoke test: `test_api_smoke.py`.
4. Updated `README.md`, `LOCAL_SETUP.md`, and `requirements.txt` for API execution.
5. Added worker run scripts: `run_worker.sh`, `run_worker.bat`.
6. Added auth + queue smoke test: `test_api_auth_queue.py`.
7. Added API-driven Streamlit client UI: `app_api.py`.
8. Added pytest framework with isolated DB fixtures and API integration tests under `tests/`.
9. Added CI workflow `.github/workflows/ci.yml` for automated backend test execution.
10. Added testing docs and scripts: `TESTING.md`, `run_tests.sh`, `run_tests.bat`.
11. Added contract tests for OpenAPI + schema enum alignment.
12. Added dedicated global and tenant admin Streamlit portals (`app_admin_global.py`, `app_admin_tenant.py`).
13. Added launch scripts for admin portals (`run_admin_global.*`, `run_admin_tenant.*`).
14. Added multitenant isolation and admin API integration tests.
15. Added debug/test run scripts for quick Docker shutdown (`stop_docker.bat`, `stop_docker.sh`).
16. Rebuilt `app.py` into a SaaS-style processing portal with dashboards, announcements, game library, rerun flows, and operations console.
17. Polished the Streamlit portal visual system with a cleaner operations-studio header, flatter cards, tighter metrics, safer HTML escaping, and follow-cam mode visibility across run summaries.

## Customer FAQ batch (stats, roster, uploads, notifications, UI)

1. Added baseline 15-stat per-team catalog service (`backend/services/stat_catalog.py`) with per-stat availability flags, attribution buckets (home/away/unattributed), shot dedup, possession from `analysis_team_stats.json`, and evidence event ids (`FR-STATS-01/02/04`).
2. Added `GET /v1/matches/{match_id}/stats` (optionally scoped by `job_id`).
3. Added roster management (`backend/routers/roster.py`): CRUD, CSV template download, alias-tolerant CSV import with per-row errors, and per-side jersey uniqueness (`FR-ROSTER-01/06`).
4. Added manual highlight assignment `POST /v1/matches/{match_id}/events/{event_id}/assign` plus an `assigned` filter on the events list; deleting a roster entry unassigns its events (`FR-ROSTER-04`).
5. Added upload validation: extension checks, 3 GB standard / 8 GB entitlement-gated caps, optional minimum-duration gate, ffprobe metadata capture, and `GET /v1/matches/upload-policy` for client pre-flight (`FR-INGEST-05..07`).
6. Added completion notifications (`backend/services/notifications.py`): console/smtp/disabled backends, `notification_logs` table, job-runner terminal-state hooks, and `GET /v1/jobs/{job_id}/notifications` (`FR-NOTIFY-01`).
7. Rebuilt the web UI as a componentized ES-module app under `frontend/` (still no build step): sign-in (token or dev) with no hardcoded tenant, responsive layout with mobile nav, match dashboard (stat catalog, roster import, highlight assignment), guided 3-step upload wizard with policy pre-flight and progress, jobs view with SLA turnaround messaging and notification state, and the ported film-review workspace (`FR-UI-01..04`, `FR-UI-05/06` first pass, `FR-UI-12`).
8. Added test coverage: `tests/test_match_stats.py`, `tests/test_roster_and_assignment.py`, `tests/test_upload_validation.py`, `tests/test_notifications.py`; browser smoke test of the new UI ran green under Playwright/Chromium.

## Sharing, routing, templates, and source coverage batch

1. Added public share links (`backend/services/sharing.py`, `backend/routers/sharing.py`): tokenized `match`, `highlight`, and `player_card` scopes, reuse of an existing live link, revocation, view counting, and an unauthenticated `GET /v1/public/shares/{token}` whose payload is assembled field by field so paths and tenant internals cannot leak (`FR-SHARE-01/03`).
2. Added player routing (`backend/services/player_routing.py`): maps `Event.jersey_number` to roster entries per team side, tolerates leading zeros, refuses ambiguous numbers when the event has no team, is idempotent, and runs automatically after job completion (`FR-ROSTER-02` routing half).
3. Added player cards: `GET /v1/matches/{id}/roster/{entry}/card` (per-player highlights and tallies) and `POST /v1/matches/{id}/roster/cards/send`, which creates a player-card share link per rostered player and emails it (`FR-ROSTER-03`).
4. Added reusable roster templates (`FR-ROSTER-05`): save a match roster as a named team, list, apply to another match (skip or replace existing), and delete; saving under an existing name replaces it.
5. Added the ingest source capability matrix (`backend/services/source_catalog.py`): YouTube, Vimeo, VEO, Hudl, Pixellot, XbotGo, NBC Sports Engine plus raw upload and local path, each with the statistics it can support. `GET /v1/sources` serves the matrix and classifies a URL; match creation stores `metadata.source_type`; the stat catalog marks source-limited statistics `not_available_for_source` (`FR-SOURCE-01/02/03`, `FR-STATS-04`).
6. Extended the web UI: public share view (no sign-in, app nav hidden), share buttons for the match and each highlight, share-link management with revoke, player card panel with its own share link, roster routing and card-send actions, saved-team picker, and link-source coverage disclosure in the Create wizard.
7. Added test coverage: `tests/test_sharing.py`, `tests/test_player_routing.py`, `tests/test_roster_templates.py`, `tests/test_source_catalog.py`.

## Validated

1. `python -m compileall app.py VideoHighlights.py backend` succeeds.
2. `python test_api_smoke.py` passes end-to-end for core API flow.
3. `python test_api_auth_queue.py` passes for auth-required + queue-worker execution mode.
4. `python -m pytest --basetemp .pytest_tmp` passes locally.
5. Current automated test suite: 172 passing tests (3 skipped without ffmpeg/torch extras).
6. Multitenant + admin test paths pass in local pytest run.
7. `python -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false` starts the portal locally.

## Next Implementation Batch (Autonomous Queue)

1. Add token revocation/rotation support and service-account style credentials for non-interactive clients.
2. Add S3 bucket policy validation and startup health checks for remote storage dependencies.
3. Add asynchronous worker-process integration tests with real subprocess lifecycle.
4. Add CI stages for lint/type checks plus smoke tests (`test_api_smoke.py`, `test_api_auth_queue.py`).
5. Add OpenAPI contract snapshot checks to detect breaking API changes automatically.
6. Add tenant-aware usage quotas and billing guardrails.
7. Recognize jersey numbers from video so routing runs without manual assignment (completes `FR-ROSTER-02`); requires a jersey-number model and per-player crop extraction.
8. Fetch and analyze link-based sources end to end (`FR-SOURCE-01` retrieval half) — the capability matrix and disclosure exist, the download/ingest path does not.
9. Detect the remaining baseline stats (passes, pass accuracy, key passes, duels, offsides, assists) so their availability flags flip on (`FR-STATS-01` completion).
10. Add social posting targets for share links (`FR-SHARE-02`, Facebook/X/TikTok/Instagram).
11. Add plans, entitlements, and billing surfaces (`FR-PLAN-01` to `FR-PLAN-05`) — the upload-entitlement hook already reads tenant metadata.
