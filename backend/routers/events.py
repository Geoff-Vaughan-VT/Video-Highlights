from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..config import settings
from ..database import get_session
from ..models import Event, Match
from ..schemas import (
    AudioEditRead,
    EventClipRead,
    EventClipRequest,
    EventPatch,
    EventRead,
    EventUpsert,
    HighlightExportRead,
    HighlightExportRequest,
)
from ..serializers import event_to_read
from ..services.audio_editor import render_audio_edit
from ..services.event_clip_renderer import concat_clips_ffmpeg, render_clip_ffmpeg
from ..services.follow_cam import ball_weight_for_mode, render_follow_cam_clip
from ..services.game_tracking import build_ball_track
from ..services.storage import get_storage_backend
from ..tenant import TenantContext, get_tenant_context
from ..utils import decode_cursor, encode_cursor, ensure_dir, generate_id, utcnow

router = APIRouter(tags=["events"])


def _safe_filename_slug(value: str, fallback: str = "audio-edit") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return slug[:80] or fallback


def _ensure_match(session: Session, match_id: str, tenant_id: str) -> Match:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    return match


def _get_event_or_404(session: Session, match_id: str, event_id: str, tenant_id: str) -> Event:
    event = session.exec(
        select(Event)
        .where(Event.match_id == match_id)
        .where(Event.tenant_id == tenant_id)
        .where(Event.id == event_id)
    ).first()
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    return event


def _clip_window_ms(event: Event, payload: EventClipRequest) -> tuple[int, int]:
    if payload.anchor == "occurred_at":
        anchor_s = float(event.occurred_at_ms) / 1000.0
        start_s = anchor_s - float(payload.pre_seconds)
        end_s = anchor_s + float(payload.post_seconds)
    else:
        start_s = (float(event.start_ms) / 1000.0) - float(payload.pre_seconds)
        end_s = (float(event.end_ms) / 1000.0) + float(payload.post_seconds)

    start_s = max(0.0, start_s)
    if end_s <= start_s:
        end_s = start_s + max(1.0, float(payload.pre_seconds) + float(payload.post_seconds))
    start_ms = max(0, int(round(start_s * 1000.0)))
    end_ms = max(start_ms + 1, int(round(end_s * 1000.0)))
    return start_ms, end_ms


def _find_cached_clip(match: Match, signature: str) -> tuple[Dict[str, object], Dict[str, object]] | None:
    metadata = dict(match.metadata_json or {})
    generated = list(metadata.get("generated_clips", []) or [])
    assets = list(metadata.get("assets", []) or [])

    clip_entry = next((item for item in generated if str(item.get("signature", "")) == signature), None)
    if not clip_entry:
        return None
    asset_id = str(clip_entry.get("asset_id", "")).strip()
    if not asset_id:
        return None
    asset_entry = next((item for item in assets if str(item.get("asset_id", "")) == asset_id), None)
    if not asset_entry:
        return None
    return clip_entry, asset_entry


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


def _read_json_file(path: str) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _resolve_tracking_manifest(event: Event) -> Dict[str, object]:
    evidence = dict(event.evidence_json or {})
    tracking_manifest_path = str(evidence.get("tracking_manifest_path", "")).strip()
    if tracking_manifest_path:
        return _read_json_file(tracking_manifest_path)

    analysis_manifest_path = str(evidence.get("analysis_manifest_path", "")).strip()
    manifest = _read_json_file(analysis_manifest_path)
    nested_tracking_manifest_path = str(manifest.get("tracking_manifest_path", "")).strip()
    if nested_tracking_manifest_path:
        return _read_json_file(nested_tracking_manifest_path)
    legacy_tracking = manifest.get("tracking")
    if isinstance(legacy_tracking, dict):
        return manifest
    return {}


def _manifest_track_to_samples(points: object) -> List[Tuple[float, float, float]]:
    if not isinstance(points, list):
        return []
    samples: List[Tuple[float, float, float]] = []
    for item in points:
        if not isinstance(item, dict):
            continue
        try:
            samples.append((float(item["t"]), float(item["x"]), float(item["y"])))
        except Exception:
            continue
    return samples


def _resolve_follow_cam_profile(event: Event) -> Tuple[str, float, List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    source = dict(event.source_json or {})
    manifest = _resolve_tracking_manifest(event)
    camera = manifest.get("camera", {}) if isinstance(manifest.get("camera", {}), dict) else {}
    tracking = manifest.get("tracking", {}) if isinstance(manifest.get("tracking", {}), dict) else {}

    mode = str(source.get("camera_mode") or camera.get("mode") or "wide").strip().lower()
    if mode not in {"follow_action", "follow_player", "follow_ball"}:
        return "wide", 1.0, [], []

    try:
        zoom_factor = float(source.get("zoom_factor") or camera.get("zoom_factor") or 1.6)
    except Exception:
        zoom_factor = 1.6

    player_track = _manifest_track_to_samples(tracking.get("target_track"))
    ball_track = _manifest_track_to_samples(tracking.get("ball_track"))
    if mode == "follow_ball":
        # Ball-centric clips lean on the ball track; the player track is the
        # fallback for frames where the ball was not detected. The manifest
        # stores RAW detections, so clean them the same way the pipeline does
        # (outlier rejection) before letting them steer the camera.
        ball_track = _cleaned_ball_samples(ball_track, manifest)
        if not ball_track and not player_track:
            return "wide", zoom_factor, [], []
        return mode, max(1.0, zoom_factor), player_track, ball_track
    if not player_track:
        return "wide", zoom_factor, [], []
    return mode, max(1.0, zoom_factor), player_track, ball_track


def _cleaned_ball_samples(
    raw_samples: List[Tuple[float, float, float]],
    manifest: Dict[str, object],
) -> List[Tuple[float, float, float]]:
    if not raw_samples:
        return []
    video = manifest.get("video", {}) if isinstance(manifest.get("video", {}), dict) else {}
    try:
        frame_w = int(video.get("frame_width") or 1920)
        frame_h = int(video.get("frame_height") or 1080)
    except Exception:
        frame_w, frame_h = 1920, 1080
    try:
        track = build_ball_track(raw_samples, (frame_w, frame_h))
    except Exception:
        return raw_samples
    return [
        (float(t), float(x), float(y))
        for t, x, y in zip(track.times, track.xs, track.ys)
    ]


def _render_window_clip(
    *,
    source_video: str,
    event: Event,
    output_path: str,
    start_seconds: float,
    end_seconds: float,
    include_audio: bool,
    prefer_gpu: bool,
) -> Tuple[str, float]:
    camera_mode, zoom_factor, player_track, ball_track = _resolve_follow_cam_profile(event)
    if camera_mode != "wide" and (player_track or ball_track):
        ball_weight = ball_weight_for_mode(camera_mode)
        render_follow_cam_clip(
            video_path=source_video,
            output_path=output_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            player_track=player_track,
            ball_track=ball_track,
            zoom_factor=zoom_factor,
            ball_weight=ball_weight,
            include_audio=include_audio,
        )
        return camera_mode, zoom_factor

    render_clip_ffmpeg(
        video_path=source_video,
        output_path=output_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        include_audio=include_audio,
        prefer_gpu=prefer_gpu,
    )
    return "wide", 1.0


@router.get("/matches/{match_id}/events", response_model=Dict[str, object])
def list_events(
    match_id: str,
    event_type: Optional[str] = Query(default=None),
    job_id: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    team_id: Optional[str] = Query(default=None),
    player_id: Optional[str] = Query(default=None),
    period: Optional[str] = Query(default=None),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    assigned: Optional[bool] = Query(default=None),
    from_ms: Optional[int] = Query(default=None, ge=0),
    to_ms: Optional[int] = Query(default=None, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    cursor: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    _ensure_match(session, match_id, tenant.tenant_id)
    offset = decode_cursor(cursor)

    stmt = select(Event).where(Event.match_id == match_id).where(Event.tenant_id == tenant.tenant_id)
    if event_type:
        stmt = stmt.where(Event.event_type == event_type)
    if job_id:
        stmt = stmt.where(Event.job_id == job_id)
    if status:
        stmt = stmt.where(Event.status == status)
    if team_id:
        stmt = stmt.where(Event.team_id == team_id)
    if player_id:
        stmt = stmt.where(Event.player_id == player_id)
    if period:
        stmt = stmt.where(Event.period == period)
    if min_confidence is not None:
        stmt = stmt.where(Event.confidence >= min_confidence)
    if assigned is not None:
        stmt = stmt.where(Event.player_id.is_not(None) if assigned else Event.player_id.is_(None))  # type: ignore[union-attr]
    if from_ms is not None:
        stmt = stmt.where(Event.occurred_at_ms >= from_ms)
    if to_ms is not None:
        stmt = stmt.where(Event.occurred_at_ms <= to_ms)

    stmt = stmt.order_by(Event.occurred_at_ms.desc()).offset(offset).limit(limit + 1)
    rows = list(session.exec(stmt))
    has_more = len(rows) > limit
    items = [event_to_read(evt) for evt in rows[:limit]]
    next_cursor = encode_cursor(offset + limit) if has_more else None
    return {"items": [item.model_dump() for item in items], "next_cursor": next_cursor}


@router.get("/matches/{match_id}/events/{event_id}", response_model=EventRead)
def get_event(
    match_id: str,
    event_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> EventRead:
    _ensure_match(session, match_id, tenant.tenant_id)
    event = _get_event_or_404(session, match_id, event_id, tenant.tenant_id)
    return event_to_read(event)


@router.post("/matches/{match_id}/events/{event_id}/clip-on-demand", response_model=EventClipRead)
def render_event_clip_on_demand(
    match_id: str,
    event_id: str,
    payload: EventClipRequest = Body(default_factory=EventClipRequest),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> EventClipRead:
    match = _ensure_match(session, match_id, tenant.tenant_id)
    event = _get_event_or_404(session, match_id, event_id, tenant.tenant_id)
    source_video = _resolve_source_video_path(match)
    if not source_video:
        raise HTTPException(status_code=400, detail="Match has no source_video_path")
    if not os.path.exists(source_video):
        raise HTTPException(status_code=400, detail=f"Source video path not found: {source_video}")

    start_ms, end_ms = _clip_window_ms(event, payload)
    effective_camera_mode, effective_zoom_factor, _, _ = _resolve_follow_cam_profile(event)
    signature = (
        f"{event.id}:{start_ms}:{end_ms}:{int(payload.include_audio)}:"
        f"{int(payload.prefer_gpu)}:{payload.anchor}:{effective_camera_mode}:{effective_zoom_factor:.2f}"
    )

    storage = get_storage_backend()
    cached = _find_cached_clip(match, signature)
    if cached and not payload.force_rebuild:
        clip_meta, asset_meta = cached
        path = str(asset_meta.get("path", ""))
        return EventClipRead(
            clip_id=str(clip_meta.get("clip_id", generate_id("eclip"))),
            match_id=match_id,
            event_id=event_id,
            asset_id=str(asset_meta.get("asset_id", "")),
            path=path,
            download_url=storage.get_download_url(path, expires_seconds=payload.expires_seconds),
            start_ms=int(clip_meta.get("start_ms", start_ms)),
            end_ms=int(clip_meta.get("end_ms", end_ms)),
            duration_ms=max(1, int(clip_meta.get("duration_ms", end_ms - start_ms))),
            include_audio=bool(clip_meta.get("include_audio", payload.include_audio)),
            anchor=str(clip_meta.get("anchor", payload.anchor)),
            reused_existing=True,
        )

    start_s = float(start_ms) / 1000.0
    end_s = float(end_ms) / 1000.0
    temp_dir = ensure_dir(os.path.join(settings.output_root, "event_clip_tmp", match_id))
    temp_filename = f"{event_id}_{start_ms}_{end_ms}_{'a' if payload.include_audio else 'na'}.mp4"
    temp_path = os.path.join(temp_dir, temp_filename)
    try:
        effective_camera_mode, effective_zoom_factor = _render_window_clip(
            source_video=source_video,
            event=event,
            output_path=temp_path,
            start_seconds=start_s,
            end_seconds=end_s,
            include_audio=payload.include_audio,
            prefer_gpu=payload.prefer_gpu,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render event clip: {exc}") from exc

    with open(temp_path, "rb") as stream:
        stored = storage.save_file(
            stream=stream,
            key_prefix=f"{match_id}/event_clips",
            filename=f"{event_id}_{start_ms}_{end_ms}.mp4",
        )
    try:
        Path(temp_path).unlink(missing_ok=True)
    except Exception:
        pass

    metadata = dict(match.metadata_json or {})
    assets = list(metadata.get("assets", []) or [])
    generated = list(metadata.get("generated_clips", []) or [])
    clip_id = generate_id("eclip")
    created_at = utcnow().isoformat()
    asset_entry = {
        "asset_id": stored.object_id,
        "filename": f"{event_id}_{start_ms}_{end_ms}.mp4",
        "path": stored.path,
        "size_bytes": stored.size_bytes,
        "storage_backend": stored.backend,
        "uploaded_at": created_at,
        "kind": "event_clip",
        "event_id": event_id,
    }
    clip_entry = {
        "clip_id": clip_id,
        "signature": signature,
        "event_id": event_id,
        "asset_id": stored.object_id,
        "path": stored.path,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": max(1, end_ms - start_ms),
        "include_audio": payload.include_audio,
        "anchor": payload.anchor,
        "camera_mode": effective_camera_mode,
        "zoom_factor": effective_zoom_factor,
        "created_at": created_at,
    }
    assets.append(asset_entry)
    generated.append(clip_entry)
    metadata["assets"] = assets[-5000:]
    metadata["generated_clips"] = generated[-5000:]
    match.metadata_json = metadata
    match.updated_at = utcnow()
    session.add(match)
    session.commit()

    return EventClipRead(
        clip_id=clip_id,
        match_id=match_id,
        event_id=event_id,
        asset_id=stored.object_id,
        path=stored.path,
        download_url=storage.get_download_url(stored.path, expires_seconds=payload.expires_seconds),
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=max(1, end_ms - start_ms),
        include_audio=payload.include_audio,
        anchor=payload.anchor,
        reused_existing=False,
    )


@router.post("/matches/{match_id}/exports/highlights", response_model=HighlightExportRead)
def export_selected_highlights(
    match_id: str,
    payload: HighlightExportRequest,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> HighlightExportRead:
    match = _ensure_match(session, match_id, tenant.tenant_id)
    source_video = _resolve_source_video_path(match)
    if not source_video:
        raise HTTPException(status_code=400, detail="Match has no source video")
    if not os.path.exists(source_video):
        raise HTTPException(status_code=400, detail=f"Source video path not found: {source_video}")

    ordered_event_ids: list[str] = []
    seen: set[str] = set()
    for raw in payload.event_ids:
        event_id = str(raw).strip()
        if event_id and event_id not in seen:
            ordered_event_ids.append(event_id)
            seen.add(event_id)
    if not ordered_event_ids:
        raise HTTPException(status_code=400, detail="At least one event_id is required")

    events = [_get_event_or_404(session, match_id, event_id, tenant.tenant_id) for event_id in ordered_event_ids]
    clip_payload = EventClipRequest(
        pre_seconds=payload.pre_seconds,
        post_seconds=payload.post_seconds,
        anchor=payload.anchor,
        include_audio=payload.include_audio,
        prefer_gpu=payload.prefer_gpu,
        force_rebuild=True,
        expires_seconds=payload.expires_seconds,
    )

    export_id = generate_id("export")
    temp_root = ensure_dir(os.path.join(settings.output_root, "highlight_export_tmp", match_id, export_id))
    clip_paths: list[str] = []
    duration_ms = 0
    try:
        for index, event in enumerate(events, start=1):
            start_ms, end_ms = _clip_window_ms(event, clip_payload)
            clip_path = os.path.join(temp_root, f"part_{index:04d}.mp4")
            _render_window_clip(
                source_video=source_video,
                event=event,
                output_path=clip_path,
                start_seconds=float(start_ms) / 1000.0,
                end_seconds=float(end_ms) / 1000.0,
                include_audio=payload.include_audio,
                prefer_gpu=payload.prefer_gpu,
            )
            clip_paths.append(clip_path)
            duration_ms += max(1, end_ms - start_ms)

        export_filename = f"{export_id}.mp4"
        export_temp_path = os.path.join(temp_root, export_filename)
        concat_clips_ffmpeg(clip_paths, export_temp_path, include_audio=payload.include_audio)

        storage = get_storage_backend()
        with open(export_temp_path, "rb") as stream:
            stored = storage.save_file(
                stream=stream,
                key_prefix=f"{match_id}/exports",
                filename=export_filename,
            )

        created_at = utcnow().isoformat()
        metadata = dict(match.metadata_json or {})
        assets = list(metadata.get("assets", []) or [])
        exports = list(metadata.get("highlight_exports", []) or [])
        asset_entry = {
            "asset_id": stored.object_id,
            "filename": export_filename,
            "path": stored.path,
            "size_bytes": stored.size_bytes,
            "storage_backend": stored.backend,
            "uploaded_at": created_at,
            "kind": "highlight_export",
            "export_id": export_id,
        }
        export_entry = {
            "export_id": export_id,
            "title": payload.title or "Selected Highlights",
            "event_ids": ordered_event_ids,
            "clip_count": len(ordered_event_ids),
            "duration_ms": duration_ms,
            "asset_id": stored.object_id,
            "path": stored.path,
            "created_at": created_at,
        }
        assets.append(asset_entry)
        exports.append(export_entry)
        metadata["assets"] = assets[-5000:]
        metadata["highlight_exports"] = exports[-5000:]
        match.metadata_json = metadata
        match.updated_at = utcnow()
        session.add(match)
        session.commit()

        return HighlightExportRead(
            export_id=export_id,
            match_id=match_id,
            event_ids=ordered_event_ids,
            clip_count=len(ordered_event_ids),
            asset_id=stored.object_id,
            path=stored.path,
            download_url=storage.get_download_url(stored.path, expires_seconds=payload.expires_seconds),
            duration_ms=duration_ms,
            created_at=created_at,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to export selected highlights: {exc}") from exc
    finally:
        try:
            shutil.rmtree(temp_root, ignore_errors=True)
        except Exception:
            pass


@router.post("/matches/{match_id}/audio/render", response_model=AudioEditRead)
def render_match_audio_edit(
    match_id: str,
    mode: str = Form(default="keep"),
    cleanup_profile: str = Form(default="none"),
    original_volume: float = Form(default=1.0, ge=0.0, le=2.0),
    music_volume: float = Form(default=0.35, ge=0.0, le=2.0),
    loop_external_audio: bool = Form(default=True),
    title: Optional[str] = Form(default=None),
    expires_seconds: int = Form(default=3600, ge=60, le=86400),
    audio_file: Optional[UploadFile] = File(default=None),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> AudioEditRead:
    match = _ensure_match(session, match_id, tenant.tenant_id)
    source_video = _resolve_source_video_path(match)
    if not source_video:
        raise HTTPException(status_code=400, detail="Match has no source video")
    if not os.path.exists(source_video):
        raise HTTPException(status_code=400, detail=f"Source video path not found: {source_video}")

    audio_edit_id = generate_id("audio")
    temp_root = ensure_dir(os.path.join(settings.output_root, "audio_edit_tmp", match_id, audio_edit_id))
    external_audio_path: Optional[str] = None
    output_temp_path = os.path.join(temp_root, f"{audio_edit_id}.mp4")
    try:
        if audio_file is not None and audio_file.filename:
            suffix = Path(audio_file.filename).suffix or ".mp3"
            external_audio_path = os.path.join(temp_root, f"source_audio{suffix}")
            with open(external_audio_path, "wb") as handle:
                shutil.copyfileobj(audio_file.file, handle)
            audio_file.file.close()
            if os.path.getsize(external_audio_path) <= 0:
                raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        render_audio_edit(
            source_video_path=source_video,
            output_path=output_temp_path,
            mode=mode,
            cleanup_profile=cleanup_profile,
            external_audio_path=external_audio_path,
            original_volume=original_volume,
            music_volume=music_volume,
            loop_external_audio=loop_external_audio,
        )

        storage = get_storage_backend()
        filename_base = _safe_filename_slug(title or f"{audio_edit_id}-{mode}-{cleanup_profile}")
        export_filename = f"{filename_base}.mp4"
        with open(output_temp_path, "rb") as stream:
            stored = storage.save_file(
                stream=stream,
                key_prefix=f"{match_id}/audio_edits",
                filename=export_filename,
            )

        created_at = utcnow().isoformat()
        metadata = dict(match.metadata_json or {})
        assets = list(metadata.get("assets", []) or [])
        audio_edits = list(metadata.get("audio_edits", []) or [])
        asset_entry = {
            "asset_id": stored.object_id,
            "filename": export_filename,
            "path": stored.path,
            "size_bytes": stored.size_bytes,
            "storage_backend": stored.backend,
            "uploaded_at": created_at,
            "kind": "audio_edit",
            "audio_edit_id": audio_edit_id,
        }
        edit_entry = {
            "audio_edit_id": audio_edit_id,
            "title": title or "Audio Edit",
            "asset_id": stored.object_id,
            "path": stored.path,
            "mode": str(mode).strip().lower(),
            "cleanup_profile": str(cleanup_profile).strip().lower(),
            "original_volume": float(original_volume),
            "music_volume": float(music_volume),
            "loop_external_audio": bool(loop_external_audio),
            "source_audio_filename": audio_file.filename if audio_file is not None else None,
            "created_at": created_at,
        }
        assets.append(asset_entry)
        audio_edits.append(edit_entry)
        metadata["assets"] = assets[-5000:]
        metadata["audio_edits"] = audio_edits[-5000:]
        match.metadata_json = metadata
        match.updated_at = utcnow()
        session.add(match)
        session.commit()

        return AudioEditRead(
            audio_edit_id=audio_edit_id,
            match_id=match_id,
            asset_id=stored.object_id,
            path=stored.path,
            download_url=storage.get_download_url(stored.path, expires_seconds=expires_seconds),
            mode=str(mode).strip().lower(),
            cleanup_profile=str(cleanup_profile).strip().lower(),
            size_bytes=stored.size_bytes,
            created_at=created_at,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render audio edit: {exc}") from exc
    finally:
        if audio_file is not None:
            try:
                audio_file.file.close()
            except Exception:
                pass
        try:
            shutil.rmtree(temp_root, ignore_errors=True)
        except Exception:
            pass


@router.put("/matches/{match_id}/events/{event_id}", response_model=EventRead)
def upsert_event(
    match_id: str,
    event_id: str,
    payload: EventUpsert,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> EventRead:
    match = _ensure_match(session, match_id, tenant.tenant_id)
    existing = session.get(Event, event_id)
    if existing and (existing.match_id != match_id or existing.tenant_id != tenant.tenant_id):
        raise HTTPException(
            status_code=409,
            detail=f"Event ID already exists on a different match: {event_id}",
        )
    event = existing

    if not event:
        event = Event(
            id=event_id,
            tenant_id=match.tenant_id,
            match_id=match_id,
            job_id=payload.job_id,
            event_type=payload.event_type,
            status=payload.status,
            confidence=payload.confidence,
            period=payload.period,
            occurred_at_ms=payload.occurred_at_ms,
            start_ms=payload.start_ms,
            end_ms=payload.end_ms,
            frame_index=payload.frame_index,
            team_id=payload.team_id,
            player_id=payload.player_id,
            jersey_number=payload.jersey_number,
            source_json=payload.source.model_dump(),
            location_json=payload.location.model_dump(),
            participants_json=[participant.model_dump() for participant in payload.participants],
            evidence_json=payload.evidence.model_dump(),
            explanations_json=[exp.model_dump() for exp in payload.explanations],
        )
    else:
        event.job_id = payload.job_id
        event.event_type = payload.event_type
        event.status = payload.status
        event.confidence = payload.confidence
        event.period = payload.period
        event.occurred_at_ms = payload.occurred_at_ms
        event.start_ms = payload.start_ms
        event.end_ms = payload.end_ms
        event.frame_index = payload.frame_index
        event.team_id = payload.team_id
        event.player_id = payload.player_id
        event.jersey_number = payload.jersey_number
        event.source_json = payload.source.model_dump()
        event.location_json = payload.location.model_dump()
        event.participants_json = [participant.model_dump() for participant in payload.participants]
        event.evidence_json = payload.evidence.model_dump()
        event.explanations_json = [exp.model_dump() for exp in payload.explanations]
        event.updated_at = utcnow()

    session.add(event)
    session.commit()
    session.refresh(event)
    return event_to_read(event)


@router.patch("/matches/{match_id}/events/{event_id}", response_model=EventRead)
def patch_event(
    match_id: str,
    event_id: str,
    payload: EventPatch,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> EventRead:
    _ensure_match(session, match_id, tenant.tenant_id)
    event = session.exec(
        select(Event)
        .where(Event.match_id == match_id)
        .where(Event.tenant_id == tenant.tenant_id)
        .where(Event.id == event_id)
    ).first()
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        if key in {"source", "location", "evidence"} and value is not None:
            setattr(event, f"{key}_json", value)
        elif key in {"participants", "explanations"} and value is not None:
            setattr(event, f"{key}_json", value)
        else:
            setattr(event, key, value)

    if event.start_ms > event.occurred_at_ms or event.occurred_at_ms > event.end_ms:
        raise HTTPException(
            status_code=400,
            detail="Must satisfy start_ms <= occurred_at_ms <= end_ms",
        )

    event.updated_at = utcnow()
    session.add(event)
    session.commit()
    session.refresh(event)
    return event_to_read(event)
