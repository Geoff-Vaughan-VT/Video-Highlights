# Feedback and Event API Schema

This document defines concrete payloads and endpoints for event storage, reviewer feedback, and feedback-to-training workflows.

## 1. API Conventions

- Base path: `/v1`
- Content type: `application/json`
- Auth: `Authorization: Bearer <token>`
- Tenant scope header: `X-Tenant-Id: <tenant_id_or_slug>`
- Idempotency for writes: `Idempotency-Key` header (recommended)
- Time format: UTC ISO 8601 for timestamps, integer milliseconds for video offsets

## 2. Common Enums

### 2.1 Event Types

- `goal`
- `shot`
- `corner_kick`
- `penalty_kick`
- `free_kick`
- `goal_kick`
- `kickoff`
- `foul`
- `save`
- `yellow_card`
- `red_card`
- `chance` (v2 event engine)
- `sprint` (v2 event engine)
- `dribble` (v2 event engine)
- `turnover` (v2 event engine)
- `foul_candidate` (v2 event engine)

### 2.2 Event Status

- `auto_detected`
- `confirmed`
- `corrected`
- `rejected`

### 2.3 Feedback Types

- `false_positive`
- `missed_event`
- `wrong_timestamp`
- `wrong_event_type`
- `wrong_player`
- `wrong_team`
- `duplicate_event`
- `confidence_miscalibrated`

### 2.4 Feedback Status

- `pending_review`
- `approved`
- `rejected`
- `needs_more_info`
- `merged`

### 2.5 Reviewer Role

- `coach`
- `analyst`
- `admin`
- `tenant_admin`
- `parent`
- `system`

## 3. Core Objects

### 3.1 Event Object

```json
{
  "event_id": "evt_01J2B8W6Q0M5J4V1A7F2K9N3P6",
  "tenant_id": "tenant_01J2A1...",
  "match_id": "match_01J2B8SH6V6Q9N0A5M3P7R1T8",
  "job_id": "job_01J2B8TRC9H3A4V5N6M7Q8P1K2",
  "event_type": "goal",
  "status": "auto_detected",
  "confidence": 0.94,
  "period": "2H",
  "occurred_at_ms": 3123456,
  "start_ms": 3120000,
  "end_ms": 3129000,
  "frame_index": 93689,
  "team_id": "team_home",
  "player_id": "player_123",
  "jersey_number": "10",
  "source": {
    "detector": "cv_event_model",
    "detector_version": "event-v0.18.3",
    "tracker_version": "tracker-v0.11.1",
    "follow_cam_version": "followcam-v0.6.4"
  },
  "location": {
    "x_norm": 0.63,
    "y_norm": 0.22,
    "zone": "right_final_third"
  },
  "participants": [
    {
      "team_id": "team_home",
      "player_id": "player_123",
      "jersey_number": "10",
      "role": "shooter"
    },
    {
      "team_id": "team_away",
      "player_id": "player_212",
      "jersey_number": "1",
      "role": "goalkeeper"
    }
  ],
  "evidence": {
    "source_asset_id": "asset_match_video_01",
    "follow_cam_asset_id": "asset_followcam_01",
    "evidence_clip_asset_id": "asset_clip_evt_01",
    "thumbnail_asset_id": "asset_thumb_evt_01"
  },
  "explanations": [
    {
      "signal": "ball_velocity_spike",
      "value": 0.82
    },
    {
      "signal": "net_motion_confidence",
      "value": 0.76
    }
  ],
  "created_at": "2026-02-20T14:12:44Z",
  "updated_at": "2026-02-20T14:12:44Z"
}
```

### 3.2 Feedback Object

```json
{
  "feedback_id": "fb_01J2B93FTVQ6H4R2N8P5K1M7A3",
  "tenant_id": "tenant_01J2A1...",
  "match_id": "match_01J2B8SH6V6Q9N0A5M3P7R1T8",
  "event_id": "evt_01J2B8W6Q0M5J4V1A7F2K9N3P6",
  "feedback_type": "wrong_timestamp",
  "status": "pending_review",
  "severity": "medium",
  "comment": "Goal detected a few seconds late.",
  "submitted_by": {
    "user_id": "user_01HXYZ",
    "role": "coach"
  },
  "correction": {
    "expected_event_type": "goal",
    "corrected_occurred_at_ms": 3119800,
    "corrected_start_ms": 3117000,
    "corrected_end_ms": 3125000,
    "corrected_team_id": "team_home",
    "corrected_player_id": "player_123",
    "corrected_jersey_number": "10"
  },
  "evidence": [
    {
      "asset_id": "asset_clip_evt_01",
      "start_ms": 3117000,
      "end_ms": 3125000,
      "note": "Ball crossed line earlier than detected timestamp."
    }
  ],
  "review": {
    "reviewed_by_user_id": null,
    "review_decision": null,
    "review_note": null,
    "reviewed_at": null
  },
  "created_at": "2026-02-20T14:20:10Z",
  "updated_at": "2026-02-20T14:20:10Z"
}
```

### 3.3 Missed Event Feedback (No Existing Event)

```json
{
  "feedback_type": "missed_event",
  "status": "pending_review",
  "severity": "high",
  "comment": "Missed penalty in second half.",
  "submitted_by": {
    "user_id": "user_01HXYZ",
    "role": "analyst"
  },
  "correction": {
    "expected_event_type": "penalty_kick",
    "corrected_occurred_at_ms": 3552100,
    "corrected_start_ms": 3550000,
    "corrected_end_ms": 3559000,
    "corrected_team_id": "team_away",
    "corrected_player_id": "player_290",
    "corrected_jersey_number": "9"
  },
  "evidence": [
    {
      "asset_id": "asset_match_video_01",
      "start_ms": 3550000,
      "end_ms": 3559000,
      "note": "Foul in box and referee points to spot."
    }
  ]
}
```

## 4. Field Validation Rules

### 4.1 Event Fields

- `event_id`, `match_id`, `job_id`: string, required, immutable IDs
- `event_type`: enum from section 2.1
- `status`: enum from section 2.2
- `confidence`: float in `[0.0, 1.0]`
- `occurred_at_ms`, `start_ms`, `end_ms`: integer, `0 <= start_ms <= occurred_at_ms <= end_ms`
- `frame_index`: integer, `>= 0`
- `period`: one of `1H`, `2H`, `ET1`, `ET2`, `PK`
- `location.x_norm`, `location.y_norm`: float in `[0.0, 1.0]`

### 4.2 Feedback Fields

- `feedback_id`, `match_id`: required
- `event_id`: optional only when `feedback_type = missed_event`
- `feedback_type`: enum from section 2.3
- `status`: enum from section 2.4
- `severity`: one of `low`, `medium`, `high`, `critical`
- `comment`: string, max 2000 characters
- `correction.corrected_occurred_at_ms`: required for `wrong_timestamp` and `missed_event`
- `correction.expected_event_type`: required for `wrong_event_type` and `missed_event`

## 5. Event Endpoints

### 5.1 List Events

`GET /v1/matches/{match_id}/events`

Query params:

- `event_type`
- `job_id`
- `status`
- `team_id`
- `player_id`
- `period`
- `min_confidence`
- `from_ms`
- `to_ms`
- `limit` (default 100, max 500)
- `cursor`

Response `200`:

```json
{
  "items": [],
  "next_cursor": "cur_01J..."
}
```

### 5.2 Get Event

`GET /v1/matches/{match_id}/events/{event_id}`

Response `200`: Event object

### 5.3 Upsert Event (Pipeline/Internal)

`PUT /v1/matches/{match_id}/events/{event_id}`

Request body: Event object (without immutable audit fields)

Response `200`:

```json
{
  "event_id": "evt_01J...",
  "status": "auto_detected",
  "updated_at": "2026-02-20T14:12:44Z"
}
```

### 5.4 Correct Event

`PATCH /v1/matches/{match_id}/events/{event_id}`

Request body example:

```json
{
  "status": "corrected",
  "event_type": "goal",
  "occurred_at_ms": 3119800,
  "team_id": "team_home",
  "player_id": "player_123",
  "jersey_number": "10"
}
```

### 5.5 Render Event Clip On Demand

`POST /v1/matches/{match_id}/events/{event_id}/clip-on-demand`

Request body:

```json
{
  "pre_seconds": 1.5,
  "post_seconds": 5.0,
  "anchor": "event_window",
  "include_audio": true,
  "prefer_gpu": true,
  "force_rebuild": false,
  "expires_seconds": 3600
}
```

### 5.6 Export Selected Highlights

`POST /v1/matches/{match_id}/exports/highlights`

Request body:

```json
{
  "event_ids": ["evt_01J...", "evt_01K..."],
  "pre_seconds": 1.0,
  "post_seconds": 3.0,
  "anchor": "event_window",
  "include_audio": true,
  "prefer_gpu": true,
  "title": "Selected Highlights",
  "expires_seconds": 3600
}
```

Response `200`:

```json
{
  "export_id": "export_01J...",
  "match_id": "match_01J...",
  "event_ids": ["evt_01J...", "evt_01K..."],
  "clip_count": 2,
  "asset_id": "asset_01J...",
  "path": "/app/storage/match_.../asset_..._export.mp4",
  "download_url": "/app/storage/match_.../asset_..._export.mp4",
  "duration_ms": 12600,
  "created_at": "2026-02-20T16:00:00Z"
}
```

Behavior:

1. Computes clip window from event timestamps.
2. Renders frame-accurate clip from source video.
3. Stores clip as a match asset and caches signature for reuse.
4. Returns playback path/download URL.

Response `200`:

```json
{
  "clip_id": "eclip_01J...",
  "match_id": "match_01J...",
  "event_id": "evt_01J...",
  "asset_id": "asset_01J...",
  "path": "/app/storage/match_.../asset_..._evt_...mp4",
  "download_url": "/app/storage/match_.../asset_..._evt_...mp4",
  "start_ms": 18900,
  "end_ms": 27500,
  "duration_ms": 8600,
  "include_audio": true,
  "anchor": "event_window",
  "reused_existing": false
}
```

## 6. Feedback Endpoints

### 6.1 Submit Feedback for Existing Event

`POST /v1/matches/{match_id}/events/{event_id}/feedback`

Request body: Feedback object (without server-managed fields)

Response `201`:

```json
{
  "feedback_id": "fb_01J...",
  "status": "pending_review"
}
```

### 6.2 Submit Missed Event Feedback

`POST /v1/matches/{match_id}/feedback`

Request body: Missed event feedback object from section 3.3

Response `201`: same as 6.1

### 6.3 List Feedback

`GET /v1/matches/{match_id}/feedback`

Query params:

- `feedback_type`
- `status`
- `severity`
- `submitted_by_user_id`
- `from_created_at`
- `to_created_at`
- `limit`
- `cursor`

Response `200`:

```json
{
  "items": [],
  "next_cursor": "cur_01J..."
}
```

### 6.4 Get Feedback

`GET /v1/matches/{match_id}/feedback/{feedback_id}`

Response `200`: Feedback object

### 6.5 Review Feedback

`POST /v1/matches/{match_id}/feedback/{feedback_id}/review`

Request body:

```json
{
  "review_decision": "approved",
  "review_note": "Timestamp correction validated by analyst."
}
```

Validation:

- `review_decision`: `approved | rejected | needs_more_info | merged`

Response `200`:

```json
{
  "feedback_id": "fb_01J...",
  "status": "approved",
  "reviewed_at": "2026-02-20T14:31:09Z"
}
```

## 7. Feedback-to-Training Endpoints

### 7.1 Create Training Candidate Batch

`POST /v1/training/feedback-batches`

Request body:

```json
{
  "match_ids": ["match_01J..."],
  "feedback_status": "approved",
  "feedback_types": ["false_positive", "missed_event", "wrong_timestamp"],
  "from_date": "2026-02-01",
  "to_date": "2026-02-20"
}
```

Response `201`:

```json
{
  "batch_id": "fbatch_01J...",
  "item_count": 482
}
```

### 7.2 Trigger Retraining Run

`POST /v1/training/runs`

Request body:

```json
{
  "batch_id": "fbatch_01J...",
  "target_model": "event-v0",
  "notes": "Weekly retrain using approved feedback."
}
```

Response `202`:

```json
{
  "run_id": "train_01J...",
  "status": "queued"
}
```

### 7.3 Get Training Run Status

`GET /v1/training/runs/{run_id}`

Response `200`:

```json
{
  "run_id": "train_01J...",
  "status": "evaluating",
  "candidate_model_version": "event-v0.19.0",
  "metrics": {
    "goal_precision": 0.92,
    "goal_recall": 0.88,
    "foul_precision": 0.71,
    "foul_recall": 0.64
  },
  "gates_passed": false
}
```

### 7.4 Promote or Reject Candidate Model

`POST /v1/training/runs/{run_id}/promote`

Request body:

```json
{
  "decision": "approved",
  "reason": "Metrics pass threshold.",
  "notes": "Promoted to production candidate.",
  "force": false
}
```

Response `200` (approved):

```json
{
  "model_id": "model_01J...",
  "target_model": "event-v0",
  "version": "event-v0.19.0",
  "run_id": "train_01J...",
  "promoted": true,
  "promoted_by_user_id": "user_admin_1",
  "promoted_at": "2026-02-20T15:12:00+00:00",
  "metrics": {
    "goal_precision": 0.92
  },
  "notes": "Promoted to production candidate.",
  "created_at": "2026-02-20T15:12:00+00:00"
}
```

### 7.5 List Model Versions

`GET /v1/training/models`

Optional query params:

- `target_model`

Response `200`: array of model version objects.

## 8. Error Model

Error response shape:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "correction.corrected_occurred_at_ms is required for feedback_type=wrong_timestamp",
    "details": [
      {
        "field": "correction.corrected_occurred_at_ms",
        "issue": "missing"
      }
    ],
    "request_id": "req_01J..."
  }
}
```

Common status codes:

- `400` validation failure
- `401` unauthorized
- `403` forbidden
- `404` not found
- `409` conflict
- `429` rate limited
- `500` internal error

## 9. Minimal Relational Schema (Reference)

### `events`

- `event_id` (PK)
- `tenant_id` (indexed)
- `match_id` (indexed)
- `job_id` (indexed)
- `event_type` (indexed)
- `status` (indexed)
- `confidence`
- `period`
- `occurred_at_ms` (indexed)
- `start_ms`
- `end_ms`
- `frame_index`
- `team_id` (indexed)
- `player_id` (indexed)
- `jersey_number`
- `payload_json`
- `created_at`
- `updated_at`

## 10. Auth Endpoints (JWT/API Access)

### 10.1 Current Auth Identity

`GET /v1/auth/me`

Response `200`:

```json
{
  "user_id": "user_123",
  "role": "coach",
  "tenant_id": "tenant_01J2A1...",
  "tenant_role": "coach",
  "is_global_admin": false,
  "auth_source": "jwt"
}
```

### 10.2 Issue JWT Token

`POST /v1/auth/token`

Request body:

```json
{
  "user_id": "user_123",
  "role": "coach",
  "tenant_id": "tenant_01J2A1...",
  "is_global_admin": false,
  "expires_in_minutes": 120
}
```

Auth behavior:

1. Requires `admin` or `system` requester token by default.
2. Optional bootstrap override using `X-Bootstrap-Key` when configured.

Response `200`:

```json
{
  "access_token": "<jwt>",
  "token_type": "bearer",
  "expires_at": "2026-02-20T18:00:00+00:00",
  "issued_for_user_id": "user_123",
  "issued_for_role": "coach",
  "issued_for_tenant_id": "tenant_01J2A1...",
  "issued_for_is_global_admin": false
}
```

## 11. Asset Download URL Endpoint

### 11.1 Resolve Asset Download URL

`GET /v1/matches/{match_id}/assets/{asset_id}/download-url`

Query params:

- `expires_seconds` (default 3600, range 60..86400)

Response `200`:

```json
{
  "match_id": "match_01J...",
  "asset_id": "asset_01J...",
  "storage_backend": "s3",
  "path": "s3://vh-bucket/video-highlights/match_01J/...mp4",
  "download_url": "https://...signed-url...",
  "expires_seconds": 3600
}
```

### `event_feedback`

- `feedback_id` (PK)
- `tenant_id` (indexed)
- `match_id` (indexed)
- `event_id` (nullable, indexed)
- `feedback_type` (indexed)
- `status` (indexed)
- `severity` (indexed)
- `submitted_by_user_id` (indexed)
- `submitted_by_role`
- `comment`
- `correction_json`
- `evidence_json`
- `review_json`
- `created_at`
- `updated_at`

### `training_feedback_batches`

- `batch_id` (PK)
- `tenant_id` (indexed)
- `criteria_json`
- `item_count`
- `created_by_user_id`
- `created_at`

### `training_runs`

- `run_id` (PK)
- `tenant_id` (indexed)
- `batch_id` (FK)
- `target_model`
- `status`
- `candidate_model_version`
- `metrics_json`
- `gates_passed`
- `created_at`
- `updated_at`

## 12. Multi-tenant Admin Endpoints

### 12.1 Global Admin

- `GET /v1/admin/global/summary`
- `GET /v1/admin/global/inventory`
- `GET /v1/admin/global/tenants`
- `POST /v1/admin/global/tenants`
- `PATCH /v1/admin/global/tenants/{tenant_id}`
- `GET /v1/admin/global/users`
- `POST /v1/admin/global/users`
- `PATCH /v1/admin/global/users/{user_id}`
- `GET /v1/admin/global/tenants/{tenant_id}/memberships`
- `POST /v1/admin/global/tenants/{tenant_id}/memberships`
- `PATCH /v1/admin/global/memberships/{membership_id}`

### 12.2 Tenant Admin

- `GET /v1/admin/tenant/summary`
- `GET /v1/admin/tenant/inventory`
- `GET /v1/admin/tenant/users`
- `POST /v1/admin/tenant/users`
- `PATCH /v1/admin/tenant/users/{user_id}`
- `PATCH /v1/admin/tenant/memberships/{membership_id}`
- `GET /v1/admin/tenant/matches`

## 13. Job Debug Logging Endpoints

### 13.1 List Job Logs

`GET /v1/jobs/{job_id}/logs`

Query params:

- `level` (`debug|info|warning|error`)
- `stage` (string)
- `detail_level` (`basic|detailed|extreme`)
- `limit` (default 200, max 5000)

### 13.2 List Job Bookmarks

`GET /v1/jobs/{job_id}/bookmarks`

Behavior:

1. Returns run bookmarks from `events` when events are already persisted.
2. Falls back to `job.result.bookmarks`.
3. Falls back to live `analysis_bookmarks.json` while processing is still running.

Query params:

- `limit` (default 2000, max 10000)

### 13.3 Kill Job Session (Testing Utility)

`POST /v1/jobs/{job_id}/kill-session`

Behavior:

1. Marks `cancel_requested=true`.
2. Immediately cancels queued/claimed jobs.
3. Marks running jobs as `cancel_requested` and finalizes cancel at safe checkpoints.

### 13.4 Rerun Job (Model/Config Refresh)

`POST /v1/jobs/{job_id}/rerun`

Request body:

```json
{
  "config_overrides": {
    "model_version": "event-v1",
    "focus_event_types": ["goal", "corner_kick"],
    "analysis_only": true
  },
  "reason": "model-upgrade"
}
```

Behavior:

1. Clones the source job configuration.
2. Applies `config_overrides`.
3. Queues a new processing job for the same match.

### 13.5 Delete Job (Run Cleanup)

`DELETE /v1/jobs/{job_id}`

Behavior:

1. Deletes the selected run record (non-running jobs only).
2. Deletes job logs linked to that run.
3. Deletes run-linked events.
4. Refreshes match processing metadata pointers.
