from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..config import settings
from ..database import get_session
from ..models import Event, JobLogEntry, Match, NotificationLog, ProcessingJob
from ..schemas import JobCreate, JobRead, JobRerunRequest
from ..serializers import job_log_to_read, job_to_read, notification_to_read
from ..services.job_logging import append_job_log
from ..services.job_runner import job_runner
from ..tenant import TenantContext, get_tenant_context
from ..utils import decode_cursor, encode_cursor, utcnow

router = APIRouter(tags=["jobs"])


def _get_tenant_job_or_404(session: Session, tenant_id: str, job_id: str) -> ProcessingJob:
    job = session.get(ProcessingJob, job_id)
    if not job or job.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


def _mark_job_cancel_requested(session: Session, job: ProcessingJob, reason: str) -> ProcessingJob:
    if job.status in {"completed", "failed", "canceled"}:
        append_job_log(
            session=session,
            job_id=job.id,
            tenant_id=job.tenant_id,
            level="info",
            stage=job.stage,
            message="Cancel requested but job already terminal",
            detail_level="detailed",
            data={"status": job.status, "reason": reason},
        )
        return job

    job.cancel_requested = True
    if job.status in {"queued", "claimed"}:
        job.status = "canceled"
        job.stage = "canceled"
        job.progress = 1.0
        job.completed_at = utcnow()
        message = "Job canceled before processing started"
    else:
        job.status = "cancel_requested"
        job.stage = "cancel_requested"
        message = "Cancel requested for running job"
    job.updated_at = utcnow()
    session.add(job)
    append_job_log(
        session=session,
        job_id=job.id,
        tenant_id=job.tenant_id,
        level="warning",
        stage=job.stage,
        message=message,
        detail_level="basic",
        data={"reason": reason},
    )
    return job


def _record_match_processing_mechanics(
    session: Session,
    match: Match,
    job: ProcessingJob,
    source_job_id: str | None = None,
    reason: str | None = None,
) -> None:
    metadata = dict(match.metadata_json or {})
    history = list(metadata.get("processing_history", []))
    config = dict(job.config_json or {})
    history.append(
        {
            "job_id": job.id,
            "source_job_id": source_job_id,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "model_version": config.get("model_version"),
            "focus_event_types": config.get("focus_event_types", []),
            "profile_name": config.get("profile_name"),
            "reason": reason,
            "config": config,
        }
    )
    metadata["processing_history"] = history[-200:]
    metadata["latest_job_id"] = job.id
    metadata["latest_model_version"] = config.get("model_version")
    metadata["latest_focus_event_types"] = config.get("focus_event_types", [])
    metadata["last_processing_updated_at"] = utcnow().isoformat()
    match.metadata_json = metadata
    match.updated_at = utcnow()
    session.add(match)


def _refresh_match_latest_processing_metadata(session: Session, match: Match) -> None:
    metadata = dict(match.metadata_json or {})
    jobs = list(
        session.exec(
            select(ProcessingJob)
            .where(ProcessingJob.match_id == match.id)
            .where(ProcessingJob.tenant_id == match.tenant_id)
            .order_by(ProcessingJob.created_at.desc())
        )
    )
    existing_ids = {job.id for job in jobs}
    history = list(metadata.get("processing_history", []))
    metadata["processing_history"] = [entry for entry in history if str(entry.get("job_id")) in existing_ids][-200:]
    if jobs:
        latest = jobs[0]
        config = dict(latest.config_json or {})
        metadata["latest_job_id"] = latest.id
        metadata["latest_model_version"] = config.get("model_version")
        metadata["latest_focus_event_types"] = config.get("focus_event_types", [])
    else:
        metadata.pop("latest_job_id", None)
        metadata.pop("latest_model_version", None)
        metadata.pop("latest_focus_event_types", None)
    metadata["last_processing_updated_at"] = utcnow().isoformat()
    match.metadata_json = metadata
    match.updated_at = utcnow()
    session.add(match)


def _read_live_manifest_bookmarks(job: ProcessingJob) -> list[Dict[str, object]]:
    config = dict(job.config_json or {})
    output_dir = str(config.get("output_dir") or os.path.join(settings.output_root, job.id))
    manifest_path = os.path.join(output_dir, "analysis_bookmarks.json")
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    bookmarks = list(payload.get("bookmarks", []) or [])
    return [entry for entry in bookmarks if isinstance(entry, dict)]


def _config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_version": config.get("model_version"),
        "camera_mode": config.get("camera_mode", "wide"),
        "analysis_only": bool(config.get("analysis_only", False)),
        "require_gpu": bool(config.get("require_gpu", False)),
        "focus_event_types": list(config.get("focus_event_types", []) or []),
        "trim_start": config.get("trim_start"),
        "trim_end": config.get("trim_end"),
        "log_profile": _log_profile(config),
    }


def _log_profile(config: Dict[str, Any]) -> str:
    value = str(config.get("log_profile") or config.get("logging_profile") or "standard").strip().lower()
    return value if value in {"standard", "detailed", "diagnostic"} else "standard"


def _next_action_for(text: str, status: str, bookmarks_count: int) -> str:
    lower = text.lower()
    if "video path" in lower or "file not found" in lower or "does not exist" in lower:
        return "Re-register the local video source on the match, then rerun the job."
    if "gpu" in lower or "cuda" in lower:
        return "Check /v1/health/gpu, then rerun with Require GPU enabled only after CUDA is ready."
    if "ffmpeg" in lower or "ffprobe" in lower or "encoder" in lower:
        return "Confirm FFmpeg/ffprobe are available to the API worker, then rerun."
    if "log" in lower and "profile" in lower:
        return "Use Run Monitor logs to follow the run story and technical details."
    if status in {"queued", "claimed"}:
        return "Run the worker once or keep the monitor open while the worker picks up the job."
    if status in {"running", "cancel_requested"}:
        return "Keep this monitor open. Cancel or kill only if progress is stuck."
    if status == "completed" and bookmarks_count <= 0:
        return "Try a longer test window or broader event targets, then rerun analysis."
    if status == "completed":
        return "Open the Game Library review view to inspect bookmarks and render/export clips."
    if status in {"canceled"}:
        return "Rerun from the latest config when ready."
    return "Open logs for details, fix the reported issue, then rerun."


def _diagnostic_summary(job: ProcessingJob, logs: List[JobLogEntry]) -> Dict[str, object]:
    status = str(job.status or "").lower()
    stage = str(job.stage or "")
    result = dict(job.result_json or {})
    config = dict(job.config_json or {})
    bookmarks_count = int(result.get("bookmarks_count", 0) or 0)
    error_logs = [item for item in logs if str(item.level).lower() == "error"]
    warning_logs = [item for item in logs if str(item.level).lower() == "warning"]

    if status == "failed":
        summary = str(job.error_message or (error_logs[0].message if error_logs else "") or "Run failed.")
        severity = "error"
    elif status == "completed":
        summary = (
            f"Run completed with {bookmarks_count} bookmarks."
            if bookmarks_count
            else "Run completed, but no bookmarks were detected."
        )
        severity = "success" if bookmarks_count else "warning"
    elif status in {"queued", "claimed"}:
        summary = f"Run is {status}; no processing output has been produced yet."
        severity = "info"
    elif status in {"running", "cancel_requested"}:
        summary = f"Run is {status} at stage {stage or 'unknown'}."
        severity = "warning" if status == "cancel_requested" else "info"
    elif status == "canceled":
        summary = str(job.error_message or "Run was canceled.")
        severity = "warning"
    else:
        summary = f"Run status is {status or 'unknown'}."
        severity = "info"

    log_rows = [job_log_to_read(item).model_dump() for item in logs]
    error_rows = [job_log_to_read(item).model_dump() for item in error_logs]
    warning_rows = [job_log_to_read(item).model_dump() for item in warning_logs]
    return {
        "job_id": job.id,
        "match_id": job.match_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "severity": severity,
        "summary": summary,
        "next_action": _next_action_for(summary, status, bookmarks_count),
        "error_message": job.error_message,
        "config_summary": _config_summary(config),
        "result_summary": {
            "bookmarks_count": bookmarks_count,
            "artifact_count": int(result.get("artifact_count", 0) or 0),
            "output_dir": result.get("output_dir"),
        },
        "latest_log": log_rows[0] if log_rows else None,
        "recent_logs": log_rows[:12],
        "error_logs": error_rows[:12],
        "warning_logs": warning_rows[:12],
    }


@router.post("/matches/{match_id}/jobs", response_model=JobRead, status_code=201)
def create_job(
    match_id: str,
    payload: JobCreate,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")

    job = ProcessingJob(tenant_id=match.tenant_id, match_id=match_id, config_json=payload.config)
    session.add(job)
    session.commit()
    session.refresh(job)
    append_job_log(
        session=session,
        job_id=job.id,
        tenant_id=job.tenant_id,
        level="info",
        stage="queued",
        message="Processing job created",
        detail_level="basic",
        data={
            "execution_mode": settings.job_execution_mode,
            "process_message": "A new processing run has been queued for this match.",
            "technical_message": "ProcessingJob row created and queued with the submitted config.",
            "log_profile": _log_profile(payload.config),
        },
    )
    append_job_log(
        session=session,
        job_id=job.id,
        tenant_id=job.tenant_id,
        level="info",
        stage="queued",
        message="Logging profile selected",
        detail_level="detailed",
        data={
            "log_profile": _log_profile(payload.config),
            "process_message": "This run will keep extra step-by-step notes for easier testing.",
            "technical_message": "Per-run log_profile controls which detailed worker checkpoints are persisted.",
        },
        force_persist=_log_profile(payload.config) in {"detailed", "diagnostic"},
    )
    append_job_log(
        session=session,
        job_id=job.id,
        tenant_id=job.tenant_id,
        level="debug",
        stage="queued",
        message="Job configuration accepted",
        detail_level="extreme",
        data={
            "config": payload.config,
            "process_message": "The exact run settings were captured for later comparison.",
            "technical_message": "Raw job config persisted for diagnostic replay.",
            "log_profile": _log_profile(payload.config),
        },
        force_persist=_log_profile(payload.config) == "diagnostic",
    )
    _record_match_processing_mechanics(session=session, match=match, job=job, reason="create_job")
    session.commit()

    if settings.job_execution_mode == "inline":
        job_runner.submit_processing_job(job.id)
    return job_to_read(job)


@router.get("/matches/{match_id}/jobs", response_model=Dict[str, object])
def list_match_jobs(
    match_id: str,
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    cursor: str | None = Query(default=None),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")

    offset = decode_cursor(cursor)
    stmt = select(ProcessingJob).where(ProcessingJob.match_id == match_id).where(
        ProcessingJob.tenant_id == tenant.tenant_id
    )
    if status:
        stmt = stmt.where(ProcessingJob.status == status)
    stmt = stmt.order_by(ProcessingJob.created_at.desc()).offset(offset).limit(limit + 1)
    rows = list(session.exec(stmt))
    has_more = len(rows) > limit
    items = [job_to_read(job) for job in rows[:limit]]
    next_cursor = encode_cursor(offset + limit) if has_more else None
    return {"items": [item.model_dump() for item in items], "next_cursor": next_cursor}


@router.get("/jobs/{job_id}", response_model=JobRead)
def get_job(
    job_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    return job_to_read(job)


@router.get("/jobs/{job_id}/bookmarks", response_model=Dict[str, object])
def get_job_bookmarks(
    job_id: str,
    limit: int = Query(default=2000, ge=1, le=10000),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)

    rows = list(
        session.exec(
            select(Event)
            .where(Event.job_id == job.id)
            .where(Event.tenant_id == tenant.tenant_id)
            .order_by(Event.occurred_at_ms.asc())
            .limit(limit)
        )
    )
    if rows:
        items = [
            {
                "event_id": item.id,
                "event_type": item.event_type,
                "status": item.status,
                "confidence": item.confidence,
                "occurred_at_ms": item.occurred_at_ms,
                "start_ms": item.start_ms,
                "end_ms": item.end_ms,
                "source": item.source_json or {},
                "explanations": item.explanations_json or [],
            }
            for item in rows
        ]
        return {"source": "events", "status": job.status, "items": items}

    result_bookmarks = list((job.result_json or {}).get("bookmarks", []) or [])
    if result_bookmarks:
        return {"source": "result", "status": job.status, "items": result_bookmarks[:limit]}

    live_bookmarks = _read_live_manifest_bookmarks(job)
    return {"source": "manifest", "status": job.status, "items": live_bookmarks[:limit]}


@router.get("/jobs/{job_id}/logs", response_model=Dict[str, object])
def list_job_logs(
    job_id: str,
    level: str | None = Query(default=None),
    stage: str | None = Query(default=None),
    detail_level: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=5000),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    _get_tenant_job_or_404(session, tenant.tenant_id, job_id)

    stmt = select(JobLogEntry).where(JobLogEntry.job_id == job_id).where(JobLogEntry.tenant_id == tenant.tenant_id)
    if level:
        stmt = stmt.where(JobLogEntry.level == level.lower())
    if stage:
        stmt = stmt.where(JobLogEntry.stage == stage)
    if detail_level:
        stmt = stmt.where(JobLogEntry.detail_level == detail_level.lower())
    stmt = stmt.order_by(JobLogEntry.created_at.desc()).limit(limit)
    rows = list(session.exec(stmt))
    items = [job_log_to_read(item).model_dump() for item in rows]
    return {"items": items}


@router.get("/jobs/{job_id}/notifications", response_model=Dict[str, object])
def list_job_notifications(
    job_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    stmt = (
        select(NotificationLog)
        .where(NotificationLog.job_id == job_id)
        .where(NotificationLog.tenant_id == tenant.tenant_id)
        .order_by(NotificationLog.created_at.desc())
        .limit(limit)
    )
    rows = list(session.exec(stmt))
    return {"items": [notification_to_read(item).model_dump() for item in rows]}


@router.get("/jobs/{job_id}/diagnostics", response_model=Dict[str, object])
def get_job_diagnostics(
    job_id: str,
    limit: int = Query(default=80, ge=1, le=500),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    logs = list(
        session.exec(
            select(JobLogEntry)
            .where(JobLogEntry.job_id == job.id)
            .where(JobLogEntry.tenant_id == tenant.tenant_id)
            .order_by(JobLogEntry.created_at.desc())
            .limit(limit)
        )
    )
    return _diagnostic_summary(job, logs)


@router.post("/jobs/{job_id}/cancel", response_model=JobRead)
def cancel_job(
    job_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    _mark_job_cancel_requested(session, job, reason="cancel_endpoint")
    session.commit()
    session.refresh(job)
    return job_to_read(job)


@router.delete("/jobs/{job_id}", response_model=Dict[str, object])
def delete_job(
    job_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    if job.status in {"running", "claimed"}:
        raise HTTPException(status_code=409, detail="Cannot delete active job. Cancel/kill it first.")

    match = session.get(Match, job.match_id)
    logs = list(
        session.exec(
            select(JobLogEntry)
            .where(JobLogEntry.job_id == job.id)
            .where(JobLogEntry.tenant_id == tenant.tenant_id)
        )
    )
    events = list(
        session.exec(
            select(Event)
            .where(Event.job_id == job.id)
            .where(Event.tenant_id == tenant.tenant_id)
        )
    )
    deleted_logs = len(logs)
    deleted_events = len(events)
    for item in logs:
        session.delete(item)
    for item in events:
        session.delete(item)
    session.delete(job)
    if match and match.tenant_id == tenant.tenant_id:
        _refresh_match_latest_processing_metadata(session, match)
    session.commit()
    return {
        "deleted": True,
        "job_id": job_id,
        "deleted_logs": deleted_logs,
        "deleted_events": deleted_events,
    }


@router.post("/jobs/{job_id}/kill-session", response_model=JobRead)
def kill_job_session(
    job_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    _mark_job_cancel_requested(session, job, reason="kill_session_endpoint")
    session.commit()
    session.refresh(job)
    return job_to_read(job)


@router.post("/jobs/{job_id}/retry", response_model=JobRead, status_code=201)
def retry_job(
    job_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    match = session.get(Match, job.match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found for job: {job_id}")

    retry = ProcessingJob(tenant_id=job.tenant_id, match_id=job.match_id, config_json=job.config_json or {})
    session.add(retry)
    session.commit()
    session.refresh(retry)
    append_job_log(
        session=session,
        job_id=retry.id,
        tenant_id=retry.tenant_id,
        level="info",
        stage="queued",
        message="Retry job created",
        detail_level="basic",
        data={"source_job_id": job.id},
    )
    _record_match_processing_mechanics(
        session=session,
        match=match,
        job=retry,
        source_job_id=job.id,
        reason="retry_job",
    )
    session.commit()

    if settings.job_execution_mode == "inline":
        job_runner.submit_processing_job(retry.id)
    return job_to_read(retry)


@router.post("/jobs/{job_id}/rerun", response_model=JobRead, status_code=201)
def rerun_job(
    job_id: str,
    payload: JobRerunRequest = Body(default_factory=JobRerunRequest),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    match = session.get(Match, job.match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found for job: {job_id}")

    new_config = dict(job.config_json or {})
    new_config.update(payload.config_overrides or {})
    rerun = ProcessingJob(tenant_id=job.tenant_id, match_id=job.match_id, config_json=new_config)
    session.add(rerun)
    session.commit()
    session.refresh(rerun)

    append_job_log(
        session=session,
        job_id=rerun.id,
        tenant_id=rerun.tenant_id,
        level="info",
        stage="queued",
        message="Rerun job created",
        detail_level="basic",
        data={
            "source_job_id": job.id,
            "reason": payload.reason,
            "config_overrides": payload.config_overrides,
            "process_message": "A rerun was queued using this match's existing source video and updated run settings.",
            "technical_message": "Created a new ProcessingJob by merging source config with config_overrides.",
            "log_profile": _log_profile(new_config),
        },
    )
    _record_match_processing_mechanics(
        session=session,
        match=match,
        job=rerun,
        source_job_id=job.id,
        reason=payload.reason or "rerun_job",
    )
    session.commit()

    if settings.job_execution_mode == "inline":
        job_runner.submit_processing_job(rerun.id)
    return job_to_read(rerun)


@router.post("/jobs/worker/run-once", response_model=Dict[str, object])
def worker_run_once(
    _: UserContext = Depends(require_roles("admin", "analyst", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    worked, job_id = job_runner.run_next_queued_job(tenant_id=tenant.tenant_id)
    return {"worked": worked, "job_id": job_id}
