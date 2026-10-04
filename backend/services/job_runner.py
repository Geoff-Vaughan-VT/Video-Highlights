from __future__ import annotations

import json
import os
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlmodel import select

from ..config import settings
from ..database import session_scope
from ..models import Event, Match, ProcessingJob, TrainingFeedbackBatch, TrainingRun
from .gpu_status import get_gpu_status
from .job_logging import append_job_log
from .notifications import notify_job_terminal_state
from .player_routing import route_match_events
from .yolo_training import train_ultralytics_yolo
from ..utils import ensure_dir


#: Test hook: when set, ``process_video_highlights`` gets this as
#: ``detector_factory`` (e.g. a ground-truth detector on synthetic footage).
DETECTOR_FACTORY_OVERRIDE: Optional[Callable[[Dict[str, object]], object]] = None

#: Seconds between cancel polls of the job row while the pipeline runs.
CANCEL_POLL_S = 1.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _read_analysis_manifest(output_dir: str) -> Dict[str, object]:
    manifest_path = Path(output_dir) / "analysis_bookmarks.json"
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
    except Exception:
        return {}
    return {}


def _training_notes_payload(raw_notes: Optional[str]) -> Dict[str, Any]:
    if not raw_notes:
        return {}
    try:
        payload = json.loads(raw_notes)
    except Exception:
        return {"notes": raw_notes}
    return payload if isinstance(payload, dict) else {"notes": raw_notes}


_LOG_PROFILE_RANK = {"standard": 0, "detailed": 1, "diagnostic": 2}
_DETAIL_PROFILE_RANK = {"basic": 0, "detailed": 1, "extreme": 2}


def _log_profile(config: Dict[str, Any]) -> str:
    value = str(config.get("log_profile") or config.get("logging_profile") or "standard").strip().lower()
    if bool(config.get("detailed_logging", False)) and value == "standard":
        value = "detailed"
    if value in {"off", "none", "false"}:
        return "standard"
    if value not in _LOG_PROFILE_RANK:
        return "standard"
    return value


def _profile_allows(config: Dict[str, Any], detail_level: str) -> bool:
    detail_rank = _DETAIL_PROFILE_RANK.get((detail_level or "basic").strip().lower(), 0)
    return _LOG_PROFILE_RANK.get(_log_profile(config), 0) >= detail_rank


def _append_process_log(
    *,
    session,
    job: ProcessingJob,
    config: Dict[str, Any],
    level: str,
    stage: str,
    message: str,
    process_message: str,
    technical_message: str,
    detail_level: str = "detailed",
    data: Optional[Dict[str, Any]] = None,
) -> None:
    if detail_level != "basic" and not _profile_allows(config, detail_level):
        return
    payload = dict(data or {})
    payload.update(
        {
            "process_message": process_message,
            "technical_message": technical_message,
            "log_profile": _log_profile(config),
        }
    )
    append_job_log(
        session=session,
        job_id=job.id,
        tenant_id=job.tenant_id,
        level=level,
        stage=stage,
        message=message,
        detail_level=detail_level,
        data=payload,
        force_persist=True,
    )


def _primary_source_asset_id(match: Match) -> Optional[str]:
    metadata = dict(match.metadata_json or {})
    assets = list(metadata.get("assets", []) or [])
    if not assets:
        return None
    path = (match.source_video_path or "").strip()
    matched = next((asset for asset in assets if str(asset.get("path", "")) == path), None)
    source = matched or assets[0]
    value = source.get("asset_id")
    return str(value) if value else None


def recover_interrupted_inline_jobs(session) -> int:
    if settings.job_execution_mode != "inline":
        return 0

    interrupted = list(
        session.exec(
            select(ProcessingJob).where(
                ProcessingJob.status.in_(["claimed", "running", "cancel_requested"])
            )
        )
    )
    recovered = 0
    for job in interrupted:
        previous_status = str(job.status or "")
        job.status = "failed"
        job.stage = "failed"
        job.progress = min(float(job.progress or 0.0), 0.99)
        job.error_message = (
            "Processing was interrupted before completion, likely because the API process restarted. "
            "Create a new run from the same config."
        )
        job.completed_at = _utcnow()
        job.updated_at = _utcnow()
        session.add(job)
        append_job_log(
            session=session,
            job_id=job.id,
            tenant_id=job.tenant_id,
            level="warning",
            stage="failed",
            message="Recovered interrupted inline job after API startup",
            detail_level="basic",
            data={
                "previous_status": previous_status,
                "reason": "Inline processing jobs cannot survive an API process restart.",
            },
        )
        recovered += 1
    return recovered


def _normalize_event_type(value: object, fallback: str = "shot") -> str:
    allowed = {
        "goal",
        "shot",
        "corner_kick",
        "penalty_kick",
        "free_kick",
        "goal_kick",
        "kickoff",
        "foul",
        "save",
        "yellow_card",
        "red_card",
        "chance",
        "sprint",
        "dribble",
        "turnover",
        "foul_candidate",
    }
    event_type = str(value or "").strip().lower()
    if event_type in allowed:
        return event_type
    return fallback


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
    return {}


def _team_side(team: object) -> Optional[str]:
    """Analysis team index -> catalog bucket (0 = home, 1 = away)."""
    try:
        value = int(team)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return {0: "home", 1: "away"}.get(value)


def _explanations(signals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Signal rows; numbers stay numbers, strings/bools stay as they are."""
    rows: List[Dict[str, Any]] = []
    for key, value in (signals or {}).items():
        if value is None or isinstance(value, (dict, list)):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append({"signal": str(key), "value": float(value)})
        else:
            rows.append({"signal": str(key), "value": value})
    return rows


def _sync_job_events_from_manifest(
    session,
    job: ProcessingJob,
    match: Match,
    config: Dict[str, object],
    manifest: Dict[str, object],
    output_dir: Optional[str] = None,
) -> int:
    """Replace this job's Event rows with the run's analysis.

    v2 runs (``analysis_events.json`` present): one row per analysis event,
    ``source_json.analysis_event_id`` = event id, team_id ``home``/``away``,
    track ids in ``participants_json``; reel-selected events keep the
    bookmark window/ids. Legacy runs: one row per bookmark.
    """
    bookmarks = [b for b in list(manifest.get("bookmarks", []) or []) if isinstance(b, dict)]
    run_dir = Path(output_dir or config.get("output_dir") or os.path.join(settings.output_root, job.id))

    existing = list(
        session.exec(
            select(Event)
            .where(Event.job_id == job.id)
            .where(Event.match_id == job.match_id)
            .where(Event.tenant_id == job.tenant_id)
        )
    )
    for item in existing:
        session.delete(item)

    detector_version = str(config.get("model_version") or "event-v2")
    follow_cam_mode = str(config.get("camera_mode") or "wide").strip().lower()
    follow_cam_zoom = float(config.get("zoom_factor", 1.6) or 1.6)
    source_asset_id = _primary_source_asset_id(match)
    resolved_dir = run_dir.resolve()
    base_evidence = {
        "source_asset_id": source_asset_id,
        "analysis_manifest_path": str(resolved_dir / "analysis_bookmarks.json"),
        "tracking_manifest_path": str(resolved_dir / "analysis_tracking.json"),
    }
    base_source = {
        "detector_version": detector_version,
        "follow_cam_version": "camera-planner-v2" if follow_cam_mode != "wide" else None,
        "camera_mode": follow_cam_mode,
        "zoom_factor": follow_cam_zoom,
    }

    def _window(start_s: float, end_s: float, occurred_s: float) -> Tuple[int, int, int]:
        start_ms = max(0, int(round(start_s * 1000.0)))
        end_ms = max(start_ms, int(round(end_s * 1000.0)))
        occurred_ms = min(max(start_ms, int(round(occurred_s * 1000.0))), end_ms)
        return start_ms, end_ms, occurred_ms

    events_doc = _read_json_file(run_dir / "analysis_events.json")
    analysis_events = [e for e in list(events_doc.get("events") or []) if isinstance(e, dict)]
    created = 0
    if analysis_events:
        trim = float(events_doc.get("trim_offset_seconds", manifest.get("trim_offset_seconds", 0.0)) or 0.0)
        tracks_meta = _read_json_file(run_dir / "tracks_meta.json")
        jerseys = {
            int(p.get("track_id")): p.get("jersey_number")
            for p in list(tracks_meta.get("players") or [])
            if isinstance(p, dict) and p.get("track_id") is not None
        }
        frame_w = float(tracks_meta.get("frame_width") or 0.0)
        frame_h = float(tracks_meta.get("frame_height") or 0.0)
        bookmark_by_event = {str(b.get("event_id")): b for b in bookmarks if b.get("event_id")}
        for ev in analysis_events:
            ev_id = str(ev.get("id") or "")
            bookmark = bookmark_by_event.get(ev_id)
            t = float(ev.get("t", 0.0) or 0.0)
            if bookmark:
                start_s = float(bookmark.get("start_s", t + trim) or 0.0)
                end_s = float(bookmark.get("end_s", t + trim) or 0.0)
            else:
                t_start = ev.get("t_start")
                t_end = ev.get("t_end")
                start_s = (float(t_start) if t_start is not None else t - 4.0) + trim
                end_s = (float(t_end) if t_end is not None else t + 6.0) + trim
            start_ms, end_ms, occurred_ms = _window(start_s, end_s, t + trim)
            team_id = _team_side(ev.get("team"))
            participants: List[Dict[str, Any]] = []
            jersey: Optional[str] = None
            for role, key in (("primary", "player_track_id"), ("secondary", "secondary_track_id")):
                tid = ev.get(key)
                if tid is None:
                    continue
                number = jerseys.get(int(tid))
                if role == "primary" and number not in (None, ""):
                    jersey = str(number)
                participants.append({
                    "team_id": team_id if role == "primary" else None,
                    "player_id": None,
                    "jersey_number": str(number) if number not in (None, "") else None,
                    "role": role,
                    "track_id": int(tid),
                })
            evidence_src = dict(ev.get("evidence") or {})
            location: Dict[str, Any] = {}
            ball_xy = evidence_src.get("ball_xy")
            if isinstance(ball_xy, (list, tuple)) and len(ball_xy) >= 2 and frame_w > 0 and frame_h > 0:
                location = {
                    "x_norm": round(float(ball_xy[0]) / frame_w, 4),
                    "y_norm": round(float(ball_xy[1]) / frame_h, 4),
                }
            if isinstance(evidence_src.get("pitch_xy_m"), (list, tuple)):
                location["pitch_xy_m"] = list(evidence_src["pitch_xy_m"])[:2]
            signals = dict((bookmark or {}).get("signals") or {})
            signals.setdefault("excitement", ev.get("excitement"))
            signals.setdefault("reason", ev.get("reason"))
            if ev.get("side") is not None:
                signals.setdefault("side", ev.get("side"))
            event = Event(
                tenant_id=job.tenant_id,
                match_id=job.match_id,
                job_id=job.id,
                event_type=_normalize_event_type(ev.get("type"), fallback="shot"),
                status="auto_detected",
                confidence=min(1.0, max(0.0, float(ev.get("confidence", 0.0) or 0.0))),
                period=None,
                occurred_at_ms=occurred_ms,
                start_ms=start_ms,
                end_ms=end_ms,
                frame_index=0,
                team_id=team_id,
                player_id=None,
                jersey_number=jersey,
                source_json={
                    **base_source,
                    "detector": "videohighlights-event-engine-v2",
                    "analysis_event_id": ev_id,
                    "analysis_event_type": ev.get("type"),
                    "bookmark_id": (bookmark or {}).get("bookmark_id"),
                    "bookmark_label": (bookmark or {}).get("label"),
                    "reel_selected": bookmark is not None,
                    "clip_index": (bookmark or {}).get("clip_index"),
                    "team": ev.get("team"),
                    "team_name": ev.get("team_name"),
                    "side": ev.get("side"),
                    "player_track_id": ev.get("player_track_id"),
                    "excitement": ev.get("excitement"),
                    "sources": list(ev.get("sources") or []),
                },
                location_json=location,
                participants_json=participants,
                evidence_json={
                    **base_evidence,
                    "bookmark_id": (bookmark or {}).get("bookmark_id"),
                    "analysis_events_path": str(resolved_dir / "analysis_events.json"),
                    **{k: evidence_src[k] for k in ("ball_xy", "pitch_xy_m", "card_crop_path", "on_target")
                       if k in evidence_src},
                },
                explanations_json=_explanations(signals),
            )
            session.add(event)
            created += 1
        return created

    for bookmark in bookmarks:
        start_s = float(bookmark.get("start_s", 0.0) or 0.0)
        end_s = float(bookmark.get("end_s", start_s) or start_s)
        occurred_s = float(bookmark.get("occurred_at_s", (start_s + end_s) / 2.0) or 0.0)
        start_ms, end_ms, occurred_ms = _window(start_s, end_s, occurred_s)
        confidence = min(1.0, max(0.0, float(bookmark.get("confidence", 0.0) or 0.0)))
        signals = bookmark.get("signals", {}) if isinstance(bookmark.get("signals", {}), dict) else {}
        event = Event(
            tenant_id=job.tenant_id,
            match_id=job.match_id,
            job_id=job.id,
            event_type=_normalize_event_type(bookmark.get("event_type"), fallback="shot"),
            status="auto_detected",
            confidence=confidence,
            period=None,
            occurred_at_ms=occurred_ms,
            start_ms=start_ms,
            end_ms=end_ms,
            frame_index=0,
            team_id=_team_side(bookmark.get("team")),
            player_id=None,
            jersey_number=None,
            source_json={
                **base_source,
                "detector": "videohighlights-multi-factor",
                "analysis_event_id": bookmark.get("event_id"),
                "bookmark_id": bookmark.get("bookmark_id"),
                "bookmark_label": bookmark.get("label"),
                "sources": bookmark.get("sources", []),
            },
            location_json={},
            participants_json=[],
            evidence_json={**base_evidence, "bookmark_id": bookmark.get("bookmark_id")},
            explanations_json=_explanations(signals),
        )
        session.add(event)
        created += 1
    return created


def _job_output_dir(job: ProcessingJob, config: Dict[str, Any], reuse_dir: Optional[str] = None) -> str:
    """``config.output_dir`` or ``<output_root>/<job id>``; never the reused run's folder."""
    output_dir = str(config.get("output_dir") or os.path.join(settings.output_root, job.id))
    if reuse_dir and os.path.abspath(output_dir) == os.path.abspath(reuse_dir):
        output_dir = os.path.join(settings.output_root, job.id)
    return output_dir


def _resolve_reuse_dir(session, job: ProcessingJob, reuse_job_id: Optional[object]) -> Optional[str]:
    """Run directory of ``reuse_tracking_from_job`` (same tenant) when it holds tracks."""
    if not reuse_job_id:
        return None
    from .tracking_types import TrackingResult

    reuse_id = str(reuse_job_id)
    candidates: List[str] = []
    source = session.get(ProcessingJob, reuse_id)
    if source is not None and source.tenant_id == job.tenant_id:
        cfg = dict(source.config_json or {})
        res = dict(source.result_json or {})
        for value in (cfg.get("output_dir"), res.get("output_dir")):
            if value:
                candidates.append(str(value))
    if source is None or source.tenant_id == job.tenant_id:
        candidates.append(os.path.join(settings.output_root, reuse_id))
    for cand in candidates:
        if TrackingResult.exists(cand):
            return str(Path(cand).resolve())
    return None


class JobRunner:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = Lock()
        self._futures: Dict[str, Future] = {}

    def submit_processing_job(self, job_id: str) -> None:
        future = self._executor.submit(self._run_processing_job, job_id)
        with self._lock:
            self._futures[job_id] = future

    def submit_training_run(self, run_id: str) -> None:
        future = self._executor.submit(self._run_training_job, run_id)
        with self._lock:
            self._futures[run_id] = future

    def run_next_queued_job(self, tenant_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Claim and process one queued job synchronously in the current thread.
        Useful for dedicated worker processes.
        Returns (worked, job_id).
        """
        job_id: Optional[str] = None
        with session_scope() as session:
            stmt = select(ProcessingJob).where(ProcessingJob.status == "queued")
            if tenant_id:
                stmt = stmt.where(ProcessingJob.tenant_id == tenant_id)
            stmt = stmt.order_by(ProcessingJob.created_at.asc()).limit(1)
            job = session.exec(stmt).first()
            if not job:
                return False, None
            job.status = "claimed"
            job.stage = "claimed"
            job.updated_at = _utcnow()
            session.add(job)
            append_job_log(
                session=session,
                job_id=job.id,
                tenant_id=job.tenant_id,
                level="info",
                stage="claimed",
                message="Worker claimed queued job",
                detail_level="basic",
                data={"tenant_id": tenant_id},
            )
            job_id = job.id

        if not job_id:
            return False, None
        self._run_processing_job(job_id)
        return True, job_id

    def _run_processing_job(self, job_id: str) -> None:
        try:
            with session_scope() as session:
                job = session.get(ProcessingJob, job_id)
                if not job:
                    return
                if job.cancel_requested:
                    job.status = "canceled"
                    job.stage = "canceled"
                    job.progress = 1.0
                    job.completed_at = _utcnow()
                    job.updated_at = _utcnow()
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="warning",
                        stage="canceled",
                        message="Job canceled before initialization",
                        detail_level="basic",
                    )
                    return

                match = session.get(Match, job.match_id)
                if not match:
                    job.status = "failed"
                    job.error_message = f"Match not found for job {job_id}"
                    job.updated_at = _utcnow()
                    job.completed_at = _utcnow()
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="error",
                        stage="failed",
                        message="Match not found for job",
                        detail_level="basic",
                        data={"match_id": job.match_id},
                    )
                    return

                job.status = "running"
                job.stage = "initializing"
                job.progress = 0.01
                job.started_at = _utcnow()
                job.updated_at = _utcnow()
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="info",
                    stage="initializing",
                    message="Job initialization started",
                    detail_level="basic",
                )

                config = job.config_json or {}
                video_path = config.get("video_path") or match.source_video_path
                reuse_dir = _resolve_reuse_dir(session, job, config.get("reuse_tracking_from_job"))
                output_dir = _job_output_dir(job, config, reuse_dir)
                ensure_dir(output_dir)
                _append_process_log(
                    session=session,
                    job=job,
                    config=config,
                    level="info",
                    stage="initializing",
                    message="Run plan assembled",
                    process_message="The worker has the game, source video path, output folder, and run options it needs to start.",
                    technical_message="Resolved match.source_video_path/config.video_path, output_dir, execution mode, trim window, camera mode, and model config.",
                    detail_level="detailed",
                    data={
                        "video_path": video_path,
                        "output_dir": output_dir,
                        "execution_mode": settings.job_execution_mode,
                        "analysis_only": bool(config.get("analysis_only", False)),
                        "camera_mode": str(config.get("camera_mode") or "wide"),
                        "model_version": config.get("model_version"),
                        "focus_event_types": list(config.get("focus_event_types", []) or []),
                        "trim_start": config.get("trim_start"),
                        "trim_end": config.get("trim_end"),
                    },
                )
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="debug",
                    stage="initializing",
                    message="Resolved runtime configuration",
                    detail_level="detailed",
                    data={
                        "video_path": video_path,
                        "output_dir": output_dir,
                        "execution_mode": settings.job_execution_mode,
                    },
                )
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="debug",
                    stage="initializing",
                    message="Raw job configuration",
                    detail_level="extreme",
                    data={"config": config},
                )

                if not os.path.exists(video_path):
                    job.status = "failed"
                    job.error_message = f"Video path not found: {video_path}"
                    job.updated_at = _utcnow()
                    job.completed_at = _utcnow()
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="error",
                        stage="failed",
                        message="Video path does not exist",
                        detail_level="basic",
                        data={"video_path": video_path},
                    )
                    return
                try:
                    source_size = os.path.getsize(video_path)
                except OSError:
                    source_size = None
                _append_process_log(
                    session=session,
                    job=job,
                    config=config,
                    level="info",
                    stage="initializing",
                    message="Source video validated",
                    process_message="The worker can see the selected video file and will use it for this run.",
                    technical_message="os.path.exists passed for video_path; source file size was read before invoking the processing pipeline.",
                    detail_level="detailed",
                    data={"video_path": video_path, "size_bytes": source_size},
                )

                job.stage = "processing_video"
                job.progress = 0.05
                job.updated_at = _utcnow()
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="info",
                    stage="processing_video",
                    message="Video processing started",
                    detail_level="basic",
                )

            # Delay import to keep API startup lightweight in environments without CV deps.
            from VideoHighlights import parse_time, process_video_highlights

            def _parse_trim(value: Optional[object]) -> Optional[float]:
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str) and value.strip():
                    return float(parse_time(value))
                return None

            with session_scope() as session:
                job = session.get(ProcessingJob, job_id)
                if not job:
                    return
                if job.cancel_requested:
                    job.status = "canceled"
                    job.stage = "canceled"
                    job.progress = 1.0
                    job.completed_at = _utcnow()
                    job.updated_at = _utcnow()
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="warning",
                        stage="canceled",
                        message="Job canceled before pipeline invocation",
                        detail_level="basic",
                    )
                    return
                config = job.config_json or {}
                match = session.get(Match, job.match_id)
                if not match:
                    job.status = "failed"
                    job.error_message = f"Match not found for job {job_id}"
                    job.updated_at = _utcnow()
                    job.completed_at = _utcnow()
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="error",
                        stage="failed",
                        message="Match not found before pipeline run",
                        detail_level="basic",
                        data={"match_id": job.match_id},
                    )
                    return

                video_path = config.get("video_path") or match.source_video_path
                reuse_dir = _resolve_reuse_dir(session, job, config.get("reuse_tracking_from_job"))
                output_dir = _job_output_dir(job, config, reuse_dir)
                gpu_status = get_gpu_status()
                _append_process_log(
                    session=session,
                    job=job,
                    config=config,
                    level="info" if gpu_status.get("ready") else "warning",
                    stage="processing_video",
                    message="Acceleration check completed",
                    process_message=(
                        "GPU analysis and GPU clip rendering are ready."
                        if gpu_status.get("ready") and gpu_status.get("rendering_ready")
                        else "The worker checked acceleration before processing; review technical details if performance is lower than expected."
                    ),
                    technical_message="Checked PyTorch CUDA, nvidia-smi, and ffmpeg h264_nvenc availability.",
                    detail_level="detailed",
                    data=gpu_status,
                )
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="info" if gpu_status.get("ready") else "warning",
                    stage="processing_video",
                    message="GPU readiness checked",
                    detail_level="basic",
                    data={
                        "ready": gpu_status.get("ready"),
                        "rendering_ready": gpu_status.get("rendering_ready"),
                        "torch": gpu_status.get("torch", {}),
                        "nvidia_smi": gpu_status.get("nvidia_smi", {}),
                        "ffmpeg_nvenc": gpu_status.get("ffmpeg_nvenc", {}),
                        "require_gpu": bool(config.get("require_gpu", False)),
                    },
                )

            with session_scope() as session:
                job = session.get(ProcessingJob, job_id)
                if job:
                    config = job.config_json or {}
                    _append_process_log(
                        session=session,
                        job=job,
                        config=config,
                        level="info",
                        stage="processing_video",
                        message="Pipeline invocation prepared",
                        process_message="The worker is handing the selected time window and output options to the video analysis engine.",
                        technical_message="Calling VideoHighlights.process_video_highlights with resolved trim, GPU, sensitivity, target, camera, and ROI parameters.",
                        detail_level="extreme",
                        data={
                            "pre_seconds": config.get("pre_seconds", 2.0),
                            "post_seconds": config.get("post_seconds", 6.0),
                            "min_clip_duration": config.get("min_clip_duration", config.get("min_clip", 4.0)),
                            "trim_start": config.get("trim_start"),
                            "trim_end": config.get("trim_end"),
                            "threads": config.get("threads"),
                            "require_gpu": bool(config.get("require_gpu", False)),
                            "speed_sensitivity": config.get("speed_sensitivity", 2.0),
                            "audio_sensitivity": config.get("audio_sensitivity", 2.0),
                            "focus_event_types": list(config.get("focus_event_types", []) or []),
                            "analysis_only": bool(config.get("analysis_only", False)),
                            "camera_mode": str(config.get("camera_mode") or "wide"),
                            "zoom_factor": config.get("zoom_factor", 1.6),
                            "render_full_follow_cam": bool(config.get("render_full_follow_cam", False)),
                            "player_roi_enabled": isinstance(config.get("player_roi"), dict),
                            "profile": config.get("profile"),
                            "yolo_model": config.get("yolo_model"),
                            "tracker_config": config.get("tracker_config"),
                            "inference_imgsz": config.get("inference_imgsz"),
                            "detection_conf": config.get("detection_conf", 0.18),
                            "vid_stride": config.get("vid_stride"),
                            "proxy_height": config.get("proxy_height"),
                            "output_height": config.get("output_height"),
                            "focus_track_id": config.get("focus_track_id"),
                            "reuse_tracking_from_job": config.get("reuse_tracking_from_job"),
                        },
                    )

            cancel_event = threading.Event()
            cancel_state: Dict[str, Any] = {"last_poll": 0.0}

            def _poll_cancel(force: bool = False) -> None:
                """Read job.cancel_requested from the DB and set the pipeline's cancel event."""
                if cancel_event.is_set():
                    return
                now_mono = time.monotonic()
                if not force and now_mono - float(cancel_state["last_poll"]) < CANCEL_POLL_S:
                    return
                cancel_state["last_poll"] = now_mono
                try:
                    with session_scope() as poll_session:
                        polled = poll_session.get(ProcessingJob, job_id)
                        if polled is not None and (
                            polled.cancel_requested or str(polled.status or "").lower() in {"cancel_requested", "canceled"}
                        ):
                            cancel_event.set()
                except Exception:
                    pass

            watcher_stop = threading.Event()

            def _cancel_watcher() -> None:
                while not watcher_stop.wait(CANCEL_POLL_S):
                    _poll_cancel(force=True)
                    if cancel_event.is_set():
                        return

            progress_state: Dict[str, Any] = {
                "last_at": None,
                "last_progress": 0.0,
                "last_sub_stage": "",
                "last_message": "",
            }

            def _record_engine_progress(
                sub_stage: str,
                progress: float,
                message: str,
                data: Optional[Dict[str, object]] = None,
            ) -> None:
                _poll_cancel()
                stage_key = str(sub_stage or "processing").strip().lower()
                message_text = str(message or stage_key).strip()
                try:
                    progress_value = max(0.0, min(0.99, float(progress)))
                except Exception:
                    progress_value = float(progress_state.get("last_progress") or 0.0)
                now = _utcnow()
                last_at = progress_state.get("last_at")
                seconds_since_last = (
                    (now - last_at).total_seconds()
                    if isinstance(last_at, datetime)
                    else 999.0
                )
                stage_changed = stage_key != str(progress_state.get("last_sub_stage") or "")
                progress_moved = progress_value >= float(progress_state.get("last_progress") or 0.0) + 0.015
                message_changed = message_text != str(progress_state.get("last_message") or "")
                important = stage_changed or progress_moved or progress_value >= 0.98 or message_changed
                if not important and seconds_since_last < 2.0:
                    return

                with session_scope() as progress_session:
                    progress_job = progress_session.get(ProcessingJob, job_id)
                    if not progress_job:
                        return
                    if str(progress_job.status or "").lower() not in {"claimed", "running", "cancel_requested"}:
                        return
                    current_progress = float(progress_job.progress or 0.0)
                    progress_job.progress = max(current_progress, progress_value)
                    progress_job.stage = "processing_video"
                    progress_job.updated_at = now
                    progress_session.add(progress_job)
                    append_job_log(
                        session=progress_session,
                        job_id=progress_job.id,
                        tenant_id=progress_job.tenant_id,
                        level="info",
                        stage="processing_video",
                        message=message_text,
                        detail_level="detailed",
                        data={
                            "sub_stage": stage_key,
                            "progress": round(progress_job.progress, 4),
                            **dict(data or {}),
                        },
                        # Engine progress IS the workflow view: always
                        # persist it (the emitter above already rate-limits
                        # to stage changes / +1.5% progress / new messages).
                        force_persist=True,
                    )
                progress_state.update(
                    {
                        "last_at": now,
                        "last_progress": progress_value,
                        "last_sub_stage": stage_key,
                        "last_message": message_text,
                    }
                )

            def _cfg_int(key: str) -> Optional[int]:
                value = config.get(key)
                return int(value) if value not in (None, "") else None

            watcher = threading.Thread(target=_cancel_watcher, name=f"vh-cancel-{job_id}", daemon=True)
            watcher.start()
            pipeline_started = time.monotonic()
            try:
                success = process_video_highlights(
                    video_path=video_path,
                    output_dir=output_dir,
                    select_player=False,  # interactive selection is impossible on a worker
                    pre_seconds=float(config.get("pre_seconds", 2.0)),
                    post_seconds=float(config.get("post_seconds", 6.0)),
                    min_clip_duration=float(config.get("min_clip_duration", config.get("min_clip", 4.0))),
                    no_audio=bool(config.get("no_audio", False)),
                    overlay=bool(config.get("overlay", False)),
                    trim_start=_parse_trim(config.get("trim_start")),
                    trim_end=_parse_trim(config.get("trim_end")),
                    threads=int(config["threads"]) if config.get("threads") is not None else None,
                    require_gpu=bool(config.get("require_gpu", False)),
                    speed_sensitivity=float(config.get("speed_sensitivity", 2.0)),
                    audio_sensitivity=float(config.get("audio_sensitivity", 2.0)),
                    focus_event_types=list(config.get("focus_event_types", []) or []),
                    model_version=str(config.get("model_version")) if config.get("model_version") else None,
                    analysis_only=bool(config.get("analysis_only", False)),
                    camera_mode=str(config.get("camera_mode") or "wide"),
                    zoom_factor=float(config.get("zoom_factor", 1.6) or 1.6),
                    render_full_follow_cam=bool(config.get("render_full_follow_cam", False)),
                    player_roi=dict(config.get("player_roi") or {}) if isinstance(config.get("player_roi"), dict) else None,
                    yolo_model=str(config["yolo_model"]) if config.get("yolo_model") else None,
                    tracker_config=str(config["tracker_config"]) if config.get("tracker_config") else None,
                    inference_imgsz=_cfg_int("inference_imgsz"),
                    detection_conf=float(config.get("detection_conf", 0.18) or 0.18),
                    vid_stride=_cfg_int("vid_stride"),
                    progress_callback=_record_engine_progress,
                    debug=bool(config.get("debug", False)),
                    log_file=str(Path(output_dir) / "pipeline_debug.log") if config.get("debug") else None,
                    debug_video=bool(config["debug_video"]) if config.get("debug_video") is not None else None,
                    dump_training_data=bool(config.get("dump_training_data", False)),
                    goal_box_left=dict(config.get("goal_box_left") or {}) if isinstance(config.get("goal_box_left"), dict) else None,
                    goal_box_right=dict(config.get("goal_box_right") or {}) if isinstance(config.get("goal_box_right"), dict) else None,
                    detect_cards=bool(config.get("detect_cards", True)),
                    broadcast_reel=bool(config.get("broadcast_reel", True)),
                    scorebug=bool(config.get("scorebug", True)),
                    team_left=str(config.get("team_left") or "HOME"),
                    team_right=str(config.get("team_right") or "AWAY"),
                    team_left_color=str(config.get("team_left_color")) if config.get("team_left_color") else None,
                    team_right_color=str(config.get("team_right_color")) if config.get("team_right_color") else None,
                    auto_detect_team_colors=bool(config.get("auto_detect_team_colors", False)),
                    llm_report=bool(config.get("llm_report", True)),
                    profile=str(config["profile"]) if config.get("profile") else None,
                    proxy_height=_cfg_int("proxy_height"),
                    output_height=_cfg_int("output_height"),
                    batch_size=config.get("batch_size"),
                    ball_tiles=bool(config["ball_tiles"]) if config.get("ball_tiles") is not None else None,
                    use_tensorrt=bool(config["use_tensorrt"]) if config.get("use_tensorrt") is not None else None,
                    device=str(config["device"]) if config.get("device") else None,
                    camera_style=str(config["camera_style"]) if config.get("camera_style") else None,
                    focus_track_id=_cfg_int("focus_track_id"),
                    reuse_tracking_from=reuse_dir,
                    pitch_corners=config.get("pitch_corners") if isinstance(config.get("pitch_corners"), list) else None,
                    reel_minutes=float(config["reel_minutes"]) if config.get("reel_minutes") else None,
                    reel_preset=str(config["reel_preset"]) if config.get("reel_preset") else None,
                    player_spotlight_reel=bool(config.get("player_spotlight_reel", False)),
                    cancel_event=cancel_event,
                    detector_factory=DETECTOR_FACTORY_OVERRIDE,
                )
            finally:
                watcher_stop.set()
            pipeline_s = round(time.monotonic() - pipeline_started, 3)
            _poll_cancel(force=True)
            was_canceled = cancel_event.is_set()

            artifacts = sorted(str(path.resolve()) for path in Path(output_dir).glob("*.mp4"))
            analysis_manifest = _read_analysis_manifest(output_dir)
            bookmarks = list(analysis_manifest.get("bookmarks", []) or [])
            result_payload = {
                "output_dir": str(Path(output_dir).resolve()),
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
                "engine": "VideoHighlights.py",
                "model_version": config.get("model_version"),
                "focus_event_types": config.get("focus_event_types", []),
                "analysis_only": bool(config.get("analysis_only", False)),
                "render_full_follow_cam": bool(config.get("render_full_follow_cam", False)),
                "bookmarks_count": len(bookmarks),
                "bookmarks": bookmarks,
                "analysis_manifest_path": str((Path(output_dir) / "analysis_bookmarks.json").resolve()),
                "tracking_manifest_path": str((Path(output_dir) / "analysis_tracking.json").resolve()),
                "analysis_table_csv_path": str((Path(output_dir) / "analysis_bookmarks.csv").resolve()),
            }
            run_dir = Path(output_dir).resolve()
            run_summary = _read_json_file(run_dir / "run_summary.json")
            for key, filename in (
                ("events_path", "analysis_events.json"),
                ("player_stats_path", "analysis_player_stats.json"),
                ("team_stats_path", "analysis_team_stats.json"),
                ("game_states_path", "analysis_game_states.json"),
                ("tracks_path", "tracks.npz"),
                ("progress_path", "progress.json"),
            ):
                if (run_dir / filename).exists():
                    result_payload[key] = str(run_dir / filename)
            proxy_info = dict(run_summary.get("proxy") or {})
            if proxy_info.get("path"):
                result_payload["proxy_path"] = proxy_info.get("path")
                result_payload["proxy"] = proxy_info
            camera_quality = run_summary.get("camera_quality") or _read_json_file(run_dir / "camera_quality.json")
            if camera_quality:
                result_payload["camera_quality"] = camera_quality
            full_movie = run_dir / "full_follow_ball_zoom.mp4"
            if full_movie.exists():
                result_payload["full_follow_cam_path"] = str(full_movie)
            reel = run_dir / "highlights_reel.mp4"
            if reel.exists():
                result_payload["reel_path"] = str(reel)
            result_payload["timings"] = {**dict(run_summary.get("timings") or {}), "pipeline_s": pipeline_s}
            result_payload["profile"] = config.get("profile")
            if reuse_dir:
                result_payload["reused_tracking_from"] = reuse_dir

            with session_scope() as session:
                job = session.get(ProcessingJob, job_id)
                if not job:
                    return
                match = session.get(Match, job.match_id)
                if not match:
                    return
                config = job.config_json or {}
                _append_process_log(
                    session=session,
                    job=job,
                    config=config,
                    level="info" if success else "error",
                    stage="processing_video" if success else "failed",
                    message="Pipeline returned",
                    process_message=(
                        "The video engine finished and the worker is collecting bookmarks and artifacts."
                        if success
                        else "The video engine reported that processing did not complete successfully."
                    ),
                    technical_message="process_video_highlights returned; worker read analysis_bookmarks.json and scanned output directory for MP4 artifacts.",
                    detail_level="detailed",
                    data={
                        "success": bool(success),
                        "bookmarks_count": len(bookmarks),
                        "artifact_count": len(artifacts),
                        "analysis_manifest_path": str((Path(output_dir) / "analysis_bookmarks.json").resolve()),
                        "output_dir": str(Path(output_dir).resolve()),
                    },
                )

                if was_canceled or job.cancel_requested or str(job.status or "").lower() == "cancel_requested":
                    job.status = "canceled"
                    job.stage = "canceled"
                    job.progress = 1.0
                    job.result_json = {**result_payload, "canceled": True}
                    job.error_message = "Job canceled by request"
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="warning",
                        stage="canceled",
                        message="Job canceled while processing",
                        detail_level="basic",
                        data={"pipeline_returned": bool(success)},
                    )
                elif success:
                    created_events = _sync_job_events_from_manifest(
                        session=session,
                        job=job,
                        match=match,
                        config=config,
                        manifest=analysis_manifest,
                        output_dir=output_dir,
                    )
                    # Attach any highlight carrying a recognized jersey number
                    # to its roster entry. No-op until the CV layer populates
                    # jersey numbers or a roster has been uploaded.
                    try:
                        session.flush()
                        routing = route_match_events(session, job.match_id, job.tenant_id)
                        if routing.roster_size:
                            append_job_log(
                                session=session,
                                job_id=job.id,
                                tenant_id=job.tenant_id,
                                level="info",
                                stage="completed",
                                message="Routed highlights to rostered players",
                                detail_level="detailed",
                                data=routing.model_dump(),
                            )
                    except Exception as routing_error:  # routing must never fail the job
                        append_job_log(
                            session=session,
                            job_id=job.id,
                            tenant_id=job.tenant_id,
                            level="warning",
                            stage="completed",
                            message="Player routing skipped after an error",
                            detail_level="detailed",
                            data={"error": str(routing_error)},
                        )
                    if job.cancel_requested:
                        job.status = "canceled"
                        job.stage = "canceled"
                        job.progress = 1.0
                        job.result_json = result_payload
                        job.error_message = "Job canceled after pipeline completion"
                        append_job_log(
                            session=session,
                            job_id=job.id,
                            tenant_id=job.tenant_id,
                            level="warning",
                            stage="canceled",
                            message="Job was cancel_requested; marked canceled after pipeline return",
                            detail_level="detailed",
                        )
                    else:
                        job.status = "completed"
                        job.stage = "completed"
                        job.progress = 1.0
                        job.result_json = result_payload
                        job.error_message = None
                        append_job_log(
                            session=session,
                            job_id=job.id,
                            tenant_id=job.tenant_id,
                            level="info",
                            stage="completed",
                            message="Job completed successfully",
                            detail_level="basic",
                            data={"artifact_count": len(artifacts)},
                        )
                        append_job_log(
                            session=session,
                            job_id=job.id,
                            tenant_id=job.tenant_id,
                            level="info",
                            stage="completed",
                            message="Bookmark analysis persisted",
                            detail_level="detailed",
                            data={"bookmarks_count": len(bookmarks), "events_created": created_events},
                            force_persist=_profile_allows(config, "detailed"),
                        )
                        _append_process_log(
                            session=session,
                            job=job,
                            config=config,
                            level="info",
                            stage="completed",
                            message="Review data ready",
                            process_message="Bookmarks were saved to the run result and copied into the review table for this match.",
                            technical_message="Synced analysis manifest bookmark rows into Event records linked to the processing job.",
                            detail_level="detailed",
                            data={"bookmarks_count": len(bookmarks), "events_created": created_events},
                        )
                        append_job_log(
                            session=session,
                            job_id=job.id,
                            tenant_id=job.tenant_id,
                            level="debug",
                            stage="completed",
                            message="Job artifacts",
                            detail_level="extreme",
                            data={"artifacts": artifacts},
                            force_persist=_profile_allows(config, "extreme"),
                        )
                else:
                    job.status = "failed"
                    job.stage = "failed"
                    job.progress = 1.0
                    job.result_json = result_payload
                    job.error_message = "Processing pipeline reported failure"
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="error",
                        stage="failed",
                        message="Processing pipeline returned failure",
                        detail_level="basic",
                    )

                job.updated_at = _utcnow()
                job.completed_at = _utcnow()
                notify_job_terminal_state(session, job)

        except Exception as exc:
            error = f"{exc}\n{traceback.format_exc()}"
            with session_scope() as session:
                job = session.get(ProcessingJob, job_id)
                if not job:
                    return
                job.status = "failed"
                job.stage = "failed"
                job.progress = 1.0
                job.error_message = error
                job.updated_at = _utcnow()
                job.completed_at = _utcnow()
                notify_job_terminal_state(session, job)
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="error",
                    stage="failed",
                    message="Unhandled exception in processing job",
                    detail_level="basic",
                    data={"error": str(exc)},
                )
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="debug",
                    stage="failed",
                    message="Unhandled exception traceback",
                    detail_level="extreme",
                    data={"traceback": traceback.format_exc()},
                )

    def _run_training_job(self, run_id: str) -> None:
        try:
            with session_scope() as session:
                run = session.get(TrainingRun, run_id)
                if not run:
                    return
                run.status = "running"
                run.updated_at = _utcnow()

            with session_scope() as session:
                run = session.get(TrainingRun, run_id)
                if not run:
                    return
                run.status = "evaluating"
                run.updated_at = _utcnow()

                batch = session.get(TrainingFeedbackBatch, run.batch_id) if run.batch_id else None
                item_count = int(batch.item_count) if batch else 0
                notes_payload = _training_notes_payload(run.notes)
                training_config = dict(notes_payload.get("training_config") or {})
                training_kind = str(training_config.get("kind") or training_config.get("training_type") or "").strip().lower()

                if training_kind == "ultralytics_yolo":
                    result = train_ultralytics_yolo(training_config)
                    run.candidate_model_version = str(result["candidate_model_version"])
                    run.metrics_json = dict(result.get("metrics") or {})
                    run.gates_passed = bool(result.get("gates_passed", False))
                else:
                    candidate_version = f"{run.target_model}.{_utcnow().strftime('%Y%m%d%H%M%S')}"

                    # Deterministic metrics keep feedback-model promotion testable until event model training is added.
                    base = min(0.95, 0.60 + (item_count / 1000.0))
                    metrics = {
                        "goal_precision": round(base, 3),
                        "goal_recall": round(max(0.4, base - 0.04), 3),
                        "foul_precision": round(max(0.35, base - 0.20), 3),
                        "foul_recall": round(max(0.30, base - 0.25), 3),
                        "feedback_items_used": item_count,
                        "training_type": "feedback_event_model",
                    }

                    run.candidate_model_version = candidate_version
                    run.metrics_json = metrics
                    run.gates_passed = item_count >= 20
                run.status = "completed"
                run.updated_at = _utcnow()

        except Exception as exc:
            with session_scope() as session:
                run = session.get(TrainingRun, run_id)
                if not run:
                    return
                run.status = "failed"
                run.metrics_json = {"error": str(exc)}
                run.updated_at = _utcnow()


job_runner = JobRunner(max_workers=max(1, settings.job_max_workers))
