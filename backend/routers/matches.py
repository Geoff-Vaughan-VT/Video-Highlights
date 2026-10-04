from __future__ import annotations

import json
import os
import subprocess
from hashlib import sha1
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..config import settings
from ..database import get_session
from ..models import Match, Tenant
from ..schemas import (
    MatchCreate,
    MatchLocalAssetRegister,
    MatchPatch,
    MatchRead,
    MatchStatsRead,
    UploadPolicyRead,
)
from ..serializers import match_to_read
from ..services.ffmpeg_tools import ffprobe_exe
from ..services.media_timeline import build_media_timeline
from ..services.source_catalog import RAW_UPLOAD, detect_source_from_url
from ..services.stat_catalog import compute_match_stat_catalog
from ..services.storage import get_storage_backend
from ..tenant import TenantContext, get_tenant_context
from ..utils import decode_cursor, encode_cursor, utcnow

router = APIRouter(prefix="/matches", tags=["matches"])

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def _gb_to_bytes(value_gb: float) -> int:
    return int(value_gb * 1024 * 1024 * 1024)


def _tenant_has_extended_uploads(session: Session, tenant_id: str | None) -> bool:
    if not tenant_id:
        return False
    tenant = session.get(Tenant, tenant_id)
    if not tenant:
        return False
    entitlements = dict((tenant.metadata_json or {}).get("entitlements", {}) or {})
    return bool(entitlements.get("extended_uploads", False))


def _resolve_upload_cap_bytes(session: Session, tenant_id: str | None) -> tuple[int, bool]:
    extended = _tenant_has_extended_uploads(session, tenant_id)
    cap_gb = settings.upload_extended_max_gb if extended else settings.upload_max_gb
    return _gb_to_bytes(cap_gb), extended


def _run_ffprobe(path: str) -> Dict[str, Any]:
    cmd = [
        ffprobe_exe(),
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_name,width,height,r_frame_rate",
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=20)
    except FileNotFoundError:
        return {"available": False, "ok": False, "error": "ffprobe was not found on PATH"}
    except Exception as exc:
        return {"available": True, "ok": False, "error": str(exc)}

    if result.returncode != 0:
        return {"available": True, "ok": False, "error": (result.stderr or result.stdout or "").strip()}

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {"available": True, "ok": False, "error": f"Could not parse ffprobe output: {exc}"}

    streams = [item for item in list(payload.get("streams", []) or []) if isinstance(item, dict)]
    video_stream = next((item for item in streams if item.get("width") and item.get("height")), streams[0] if streams else {})
    duration = None
    try:
        duration = float((payload.get("format", {}) or {}).get("duration"))
    except Exception:
        duration = None
    return {
        "available": True,
        "ok": True,
        "error": None,
        "duration_seconds": duration,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "codec_name": video_stream.get("codec_name"),
        "frame_rate": video_stream.get("r_frame_rate"),
    }


def _inspect_local_video_path(path: str) -> Dict[str, Any]:
    raw_path = str(path or "").strip().strip('"')
    if not raw_path:
        return {"ok": False, "code": "path_required", "path": "", "message": "Choose a local video file path."}

    clean_path = os.path.abspath(raw_path)
    extension = os.path.splitext(clean_path)[1].lower()
    payload: Dict[str, Any] = {
        "ok": False,
        "code": "unknown",
        "path": clean_path,
        "filename": os.path.basename(clean_path),
        "extension": extension,
        "extension_ok": extension in VIDEO_EXTENSIONS,
        "exists": os.path.exists(clean_path),
        "is_file": os.path.isfile(clean_path),
        "size_bytes": None,
        "ffprobe": {},
    }

    if not payload["exists"]:
        payload.update({"code": "not_found", "message": f"Local video file not found: {clean_path}"})
        return payload
    if not payload["is_file"]:
        payload.update({"code": "not_file", "message": f"Local video path is not a file: {clean_path}"})
        return payload
    if not payload["extension_ok"]:
        payload.update({"code": "bad_extension", "message": "Local video must end in .mp4, .mov, .mkv, .avi, or .m4v"})
        return payload

    try:
        size_bytes = os.path.getsize(clean_path)
    except OSError as exc:
        payload.update({"code": "unreadable", "message": f"Could not read local video file: {exc}"})
        return payload
    payload["size_bytes"] = int(size_bytes)
    if size_bytes <= 0:
        payload.update(
            {
                "code": "zero_bytes",
                "message": (
                    "Windows reports this file is 0 bytes. If it is still copying, syncing, or a cloud placeholder, "
                    "wait for the real local file to finish downloading before launching."
                ),
            }
        )
        return payload

    payload["ffprobe"] = _run_ffprobe(clean_path)
    payload.update({"ok": True, "code": "ready", "message": "Local video is readable by the API worker."})
    return payload


def _validate_local_video_path(path: str) -> Dict[str, Any]:
    inspected = _inspect_local_video_path(path)
    if not inspected.get("ok"):
        raise HTTPException(status_code=400, detail=str(inspected.get("message") or "Local video is not ready"))
    return {
        "path": inspected["path"],
        "filename": inspected["filename"],
        "size_bytes": int(inspected["size_bytes"]),
    }


def _resolve_source_video_path(match: Match) -> str:
    candidates = [str(match.source_video_path or "").strip()]
    metadata = dict(match.metadata_json or {})
    assets = list(metadata.get("assets", []) or [])
    for asset in assets:
        path = str(asset.get("path", "")).strip()
        if path:
            candidates.append(path)
    for path in candidates:
        if path and os.path.exists(path):
            return path
    for path in candidates:
        if path:
            return path
    return ""


@router.post("", response_model=MatchRead, status_code=201)
def create_match(
    payload: MatchCreate,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles("admin", "analyst", "coach")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> MatchRead:
    target_tenant_id = tenant.tenant_id
    if payload.tenant_id:
        if not user.is_global_admin and payload.tenant_id != tenant.tenant_id:
            raise HTTPException(status_code=403, detail="Cannot create a match in another tenant")
        target_tenant_id = payload.tenant_id

    # Classify the ingest source so stat coverage can be disclosed up front
    # and the stat catalog can mark link-limited statistics unavailable.
    metadata = dict(payload.metadata or {})
    if not metadata.get("source_type"):
        detected = detect_source_from_url(payload.source_video_path)
        metadata["source_type"] = detected.key if detected else RAW_UPLOAD
        if detected:
            metadata.setdefault("source_url", payload.source_video_path)

    match = Match(
        tenant_id=target_tenant_id,
        name=payload.name,
        home_team_name=payload.home_team_name,
        away_team_name=payload.away_team_name,
        match_date=payload.match_date,
        source_video_path=payload.source_video_path,
        metadata_json=metadata,
    )
    session.add(match)
    session.commit()
    session.refresh(match)
    return match_to_read(match)


@router.get("", response_model=Dict[str, object])
def list_matches(
    limit: int = Query(default=100, ge=1, le=500),
    cursor: str | None = Query(default=None),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    offset = decode_cursor(cursor)
    stmt = (
        select(Match)
        .where(Match.tenant_id == tenant.tenant_id)
        .order_by(Match.created_at.desc())
        .offset(offset)
        .limit(limit + 1)
    )
    rows = list(session.exec(stmt))
    has_more = len(rows) > limit
    items = [match_to_read(match) for match in rows[:limit]]
    next_cursor = encode_cursor(offset + limit) if has_more else None
    return {"items": [item.model_dump() for item in items], "next_cursor": next_cursor}


@router.get("/upload-policy", response_model=UploadPolicyRead)
def get_upload_policy(
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> UploadPolicyRead:
    cap_bytes, extended = _resolve_upload_cap_bytes(session, tenant.tenant_id)
    return UploadPolicyRead(
        max_upload_bytes=cap_bytes,
        max_upload_gb=settings.upload_extended_max_gb if extended else settings.upload_max_gb,
        extended_max_upload_bytes=_gb_to_bytes(settings.upload_extended_max_gb),
        extended_max_upload_gb=settings.upload_extended_max_gb,
        extended_upload_enabled=extended,
        min_duration_seconds=settings.upload_min_duration_seconds,
        allowed_extensions=sorted(VIDEO_EXTENSIONS),
        processing_sla_hours=[settings.processing_sla_hours_min, settings.processing_sla_hours_max],
    )


@router.post("/assets/inspect-local", response_model=Dict[str, Any])
def inspect_local_match_asset(
    payload: MatchLocalAssetRegister,
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    inspected = _inspect_local_video_path(payload.path)
    inspected["tenant_id"] = tenant.tenant_id
    return inspected


@router.get("/{match_id}", response_model=MatchRead)
def get_match(
    match_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> MatchRead:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    return match_to_read(match)


@router.get("/{match_id}/timeline", response_model=Dict[str, Any])
def get_match_timeline(
    match_id: str,
    thumbnail_count: int = Query(default=18, ge=4, le=48),
    waveform_bins: int = Query(default=96, ge=16, le=360),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    source_video = _resolve_source_video_path(match)
    if not source_video:
        raise HTTPException(status_code=400, detail="Match has no source video")
    if not os.path.exists(source_video):
        raise HTTPException(status_code=400, detail=f"Source video path not found: {source_video}")
    try:
        timeline = build_media_timeline(
            source_video,
            thumbnail_count=thumbnail_count,
            waveform_bins=waveform_bins,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build media timeline: {exc}") from exc
    timeline["match_id"] = match_id
    timeline["source_video_path"] = source_video
    return timeline


@router.get("/{match_id}/stats", response_model=MatchStatsRead)
def get_match_stats(
    match_id: str,
    job_id: str | None = Query(default=None),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> MatchStatsRead:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    return compute_match_stat_catalog(session, match, tenant.tenant_id, job_id=job_id)


@router.patch("/{match_id}", response_model=MatchRead)
def update_match(
    match_id: str,
    payload: MatchPatch,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> MatchRead:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")

    data = payload.model_dump(exclude_unset=True)
    if "tenant_id" in data:
        requested = data["tenant_id"]
        if requested and requested != match.tenant_id:
            if not user.is_global_admin:
                raise HTTPException(status_code=403, detail="Cannot move a match across tenants")
            match.tenant_id = requested
        data.pop("tenant_id", None)
    for key, value in data.items():
        if key == "metadata":
            match.metadata_json = value
        else:
            setattr(match, key, value)
    match.updated_at = utcnow()
    session.add(match)
    session.commit()
    session.refresh(match)
    return match_to_read(match)


@router.post("/{match_id}/assets/upload", response_model=Dict[str, Any], status_code=201)
def upload_match_asset(
    match_id: str,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")

    safe_name = file.filename or "upload.bin"
    extension = os.path.splitext(safe_name)[1].lower()
    if extension not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{extension or 'none'}'. Upload one of: {', '.join(sorted(VIDEO_EXTENSIONS))}",
        )

    cap_bytes, extended = _resolve_upload_cap_bytes(session, tenant.tenant_id)

    def _discard_stored(stored_result) -> None:
        if stored_result.backend == "local" and stored_result.path and os.path.exists(stored_result.path):
            try:
                os.remove(stored_result.path)
            except OSError:
                pass

    storage = get_storage_backend()
    stored = storage.save_file(file.file, key_prefix=match_id, filename=safe_name)
    file.file.close()
    if int(stored.size_bytes or 0) <= 0:
        _discard_stored(stored)
        raise HTTPException(status_code=400, detail="Uploaded video file is empty. Choose a non-empty MP4/MOV/MKV/AVI file.")

    if int(stored.size_bytes or 0) > cap_bytes:
        _discard_stored(stored)
        cap_gb = settings.upload_extended_max_gb if extended else settings.upload_max_gb
        upgrade_hint = "" if extended else " Larger matches are available as a paid add-on (extended uploads)."
        raise HTTPException(
            status_code=413,
            detail=f"Video is larger than the {cap_gb:g} GB upload limit for this account.{upgrade_hint}",
        )

    probe: Dict[str, Any] = {}
    if stored.backend == "local" and stored.path and os.path.exists(stored.path):
        probe = _run_ffprobe(stored.path)
        min_duration = settings.upload_min_duration_seconds
        duration = probe.get("duration_seconds") if probe.get("ok") else None
        if min_duration > 0 and duration is not None and float(duration) < min_duration:
            _discard_stored(stored)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video is {float(duration) / 60.0:.1f} minutes long; matches must be at least "
                    f"{min_duration / 60.0:.0f} minutes to analyze."
                ),
            )

    assets = list((match.metadata_json or {}).get("assets", []))
    asset_entry: Dict[str, Any] = {
        "asset_id": stored.object_id,
        "filename": safe_name,
        "path": stored.path,
        "size_bytes": stored.size_bytes,
        "storage_backend": stored.backend,
        "uploaded_at": utcnow().isoformat(),
    }
    if probe.get("ok"):
        asset_entry["duration_seconds"] = probe.get("duration_seconds")
        asset_entry["width"] = probe.get("width")
        asset_entry["height"] = probe.get("height")
    assets.append(asset_entry)
    metadata = dict(match.metadata_json or {})
    metadata["assets"] = assets
    match.metadata_json = metadata

    # Optional convenience: first valid upload, or a replacement for a bad empty source, becomes the source video path.
    current_source = str(match.source_video_path or "").strip()
    current_source_missing = not current_source or not os.path.exists(current_source)
    current_source_empty = False
    if current_source and os.path.exists(current_source):
        try:
            current_source_empty = os.path.getsize(current_source) <= 0
        except OSError:
            current_source_empty = True
    if current_source_missing or current_source_empty:
        match.source_video_path = stored.path

    match.updated_at = utcnow()
    session.add(match)
    session.commit()

    return {
        "asset_id": stored.object_id,
        "match_id": match_id,
        "filename": safe_name,
        "path": stored.path,
        "size_bytes": stored.size_bytes,
        "storage_backend": stored.backend,
        "duration_seconds": asset_entry.get("duration_seconds"),
        "width": asset_entry.get("width"),
        "height": asset_entry.get("height"),
    }


@router.post("/{match_id}/assets/register-local", response_model=Dict[str, Any], status_code=201)
def register_local_match_asset(
    match_id: str,
    payload: MatchLocalAssetRegister,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")

    local_video = _validate_local_video_path(payload.path)
    asset_id = f"asset_local_{sha1(str(local_video['path']).encode('utf-8')).hexdigest()[:24]}"
    asset_entry = {
        "asset_id": asset_id,
        "filename": local_video["filename"],
        "path": local_video["path"],
        "size_bytes": local_video["size_bytes"],
        "storage_backend": "local_path",
        "uploaded_at": utcnow().isoformat(),
        "registered_only": True,
    }

    metadata = dict(match.metadata_json or {})
    assets = list(metadata.get("assets", []) or [])
    assets = [item for item in assets if item.get("asset_id") != asset_id and item.get("path") != local_video["path"]]
    assets.append(asset_entry)
    metadata["assets"] = assets
    match.metadata_json = metadata

    if payload.set_as_source:
        match.source_video_path = str(local_video["path"])

    match.updated_at = utcnow()
    session.add(match)
    session.commit()
    session.refresh(match)

    return {
        "asset_id": asset_id,
        "match_id": match_id,
        "filename": local_video["filename"],
        "path": local_video["path"],
        "size_bytes": local_video["size_bytes"],
        "storage_backend": "local_path",
        "set_as_source": bool(payload.set_as_source),
    }


@router.get("/{match_id}/assets/{asset_id}/download-url", response_model=Dict[str, Any])
def get_asset_download_url(
    match_id: str,
    asset_id: str,
    expires_seconds: int = Query(default=3600, ge=60, le=86400),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")

    assets = list((match.metadata_json or {}).get("assets", []))
    asset = next((item for item in assets if item.get("asset_id") == asset_id), None)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")

    storage = get_storage_backend()
    stored_path = str(asset.get("path", ""))
    url = storage.get_download_url(stored_path=stored_path, expires_seconds=expires_seconds)
    return {
        "match_id": match_id,
        "asset_id": asset_id,
        "storage_backend": asset.get("storage_backend", storage.__class__.__name__),
        "path": stored_path,
        "download_url": url,
        "expires_seconds": expires_seconds,
    }
