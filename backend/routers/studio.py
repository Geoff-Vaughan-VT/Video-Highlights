"""Studio endpoints: the match library, run review data and safe media streaming.

Powers the built-in web UI (served at ``/``). Everything here reads the run
artifacts described in ``docs/ARTIFACTS.md`` from the configured output root
and joins them with the DB (matches, jobs, events) so the UI never has to
issue N+1 requests.

Media endpoints are range-aware (HTTP 206) so browsers can seek long movies
without downloading them, and every file path is validated against the run
directory (no path traversal).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..config import settings
from ..database import get_session
from ..models import Event, JobLogEntry, Match, ProcessingJob
from ..services.tracking_types import TRACKS_FILENAME, TRACKS_META_FILENAME, TrackingResult
from ..tenant import TenantContext, get_tenant_context
from ..utils import utcnow

router = APIRouter(tags=["studio"])

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_ALLOWED_SUFFIXES = {".mp4", ".png", ".jpg", ".jpeg", ".json", ".csv", ".jsonl", ".log", ".md", ".txt"}
_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".csv": "text/csv; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}
_STATIC_MEDIA = {".mp4", ".m4v", ".mov", ".mkv", ".png", ".jpg", ".jpeg"}
_ASSET_SUFFIXES = {".mp4", ".m4v", ".mov", ".mkv", ".png", ".jpg", ".jpeg"}
_CHUNK = 256 * 1024
_ACTIVE_STATUSES = {"queued", "claimed", "running", "cancel_requested"}
# progress.json is only trusted once the job has started: a queued rerun can
# share its parent's output_dir and would otherwise show the parent's state.
_STARTED_STATUSES = {"claimed", "running", "cancel_requested", "completed", "failed", "canceled"}
_THUMB_INTERVAL_S = 10.0
LABELS_FILENAME = "player_labels.json"
CALIBRATION_FILENAME = "calibration.json"
POSTER_FILENAME = "studio_poster.jpg"

_READ_ROLES = ("admin", "analyst", "coach", "parent", "system", "tenant_admin")
_WRITE_ROLES = ("admin", "analyst", "coach", "tenant_admin")


# ---------------------------------------------------------------------------
# Paths and small file helpers
# ---------------------------------------------------------------------------


def _output_root() -> Path:
    return Path(settings.output_root).expanduser().resolve()


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _run_dir(run_id: str) -> Path:
    if not _SAFE_NAME.match(run_id or "") or run_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid run id")
    root = _output_root()
    path = (root / run_id).resolve()
    if path == root or not _is_within(path, root) or not path.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    return path


def _safe_child(base: Path, name: str) -> Path:
    if not _SAFE_NAME.match(name or "") or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid file name")
    path = (base / name).resolve()
    if not _is_within(path, base.resolve()):
        raise HTTPException(status_code=400, detail="Invalid file name")
    return path


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and any(
        (path / name).exists()
        for name in ("analysis_bookmarks.json", "analysis_events.json", TRACKS_FILENAME, "progress.json")
    )


# ---------------------------------------------------------------------------
# Range-aware media responses
# ---------------------------------------------------------------------------


class _Unsatisfiable(Exception):
    """Raised for a syntactically valid but unsatisfiable byte range (416)."""


def _parse_range(header: str, size: int) -> Optional[Tuple[int, int]]:
    """Parse a single ``bytes=`` range. Returns inclusive (start, end).

    Returns ``None`` for headers we ignore (non-bytes units, multi-range),
    raises ``_Unsatisfiable`` for ranges outside the file.
    """
    value = (header or "").strip()
    if not value.lower().startswith("bytes="):
        return None
    spec = value[6:].strip()
    if "," in spec:
        spec = spec.split(",", 1)[0].strip()  # serve the first range only
    if "-" not in spec:
        return None
    start_s, end_s = spec.split("-", 1)
    try:
        if start_s.strip() == "":
            suffix = int(end_s)
            if suffix <= 0:
                raise ValueError
            start, end = max(0, size - suffix), size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s.strip() else size - 1
    except ValueError:
        return None  # malformed: ignore the header and send the full body (RFC 9110)
    end = min(end, size - 1)
    if size == 0 or start >= size or start > end or start < 0:
        raise _Unsatisfiable()
    return start, end


def _iter_file(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def media_response(request: Request, path: Path, download_name: Optional[str] = None) -> Response:
    """Serve ``path`` with Range, ETag and Cache-Control support."""
    try:
        stat = path.stat()
    except OSError:
        raise HTTPException(status_code=404, detail="File not found")
    size = int(stat.st_size)
    suffix = path.suffix.lower()
    media_type = _MEDIA_TYPES.get(suffix, "application/octet-stream")
    etag = f'"{int(stat.st_mtime)}-{size}"'
    headers = {
        "Accept-Ranges": "bytes",
        "ETag": etag,
        "Last-Modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT"),
        "Cache-Control": "public, max-age=3600" if suffix in _STATIC_MEDIA else "no-cache",
    }
    if download_name:
        headers["Content-Disposition"] = f'inline; filename="{download_name}"'

    range_header = request.headers.get("range")
    if_range = request.headers.get("if-range")
    if range_header and if_range and if_range.strip() != etag:
        range_header = None  # resource changed: send the whole thing

    if not range_header and request.headers.get("if-none-match", "").strip() == etag:
        return Response(status_code=304, headers=headers)

    try:
        byte_range = _parse_range(range_header, size) if range_header else None
    except _Unsatisfiable:
        # Returned (not raised): the app's HTTPException handler drops headers.
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{size}"})
    head_only = request.method == "HEAD"
    if byte_range is None:
        headers["Content-Length"] = str(size)
        if head_only:
            return Response(status_code=200, headers=headers, media_type=media_type)
        return StreamingResponse(_iter_file(path, 0, size), status_code=200, headers=headers, media_type=media_type)

    start, end = byte_range
    length = end - start + 1
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(length)
    if head_only:
        return Response(status_code=206, headers=headers, media_type=media_type)
    return StreamingResponse(_iter_file(path, start, length), status_code=206, headers=headers, media_type=media_type)


# ---------------------------------------------------------------------------
# Tracking cache (tracks.npz is large; keep a few runs in memory)
# ---------------------------------------------------------------------------

_TRACK_CACHE: "OrderedDict[str, Tuple[float, TrackingResult]]" = OrderedDict()
_TRACK_CACHE_LOCK = threading.Lock()
_TRACK_CACHE_SIZE = 4


def _load_tracks(run: Path) -> TrackingResult:
    if not TrackingResult.exists(run):
        raise HTTPException(status_code=404, detail="No tracks.npz for this run")
    key = str(run)
    mtime = max((run / TRACKS_FILENAME).stat().st_mtime, (run / TRACKS_META_FILENAME).stat().st_mtime)
    with _TRACK_CACHE_LOCK:
        hit = _TRACK_CACHE.get(key)
        if hit and hit[0] == mtime:
            _TRACK_CACHE.move_to_end(key)
            return hit[1]
    try:
        result = TrackingResult.load(run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load tracks: {exc}") from exc
    with _TRACK_CACHE_LOCK:
        _TRACK_CACHE[key] = (mtime, result)
        _TRACK_CACHE.move_to_end(key)
        while len(_TRACK_CACHE) > _TRACK_CACHE_SIZE:
            _TRACK_CACHE.popitem(last=False)
    return result


def clear_track_cache() -> None:
    with _TRACK_CACHE_LOCK:
        _TRACK_CACHE.clear()


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


def _videos(path: Path) -> Dict[str, Any]:
    videos: Dict[str, Any] = {}
    if (path / "trimmed_working_video.mp4").exists():
        videos["original"] = "trimmed_working_video.mp4"
    proxies = sorted(path.glob("proxy_*.mp4"), key=lambda p: (p.name != "proxy_1080p.mp4", p.name))
    if proxies:
        videos["proxy"] = proxies[0].name
    if (path / "debug_camera_wide.mp4").exists():
        videos["debug"] = "debug_camera_wide.mp4"
    movies = sorted(path.glob("full_*.mp4"), key=lambda p: (p.name != "full_follow_ball_zoom.mp4", p.name))
    if movies:
        videos["movie"] = movies[0].name
        videos["zoom"] = movies[0].name  # legacy key
        videos["movies"] = [m.name for m in movies]
    if (path / "highlights_reel.mp4").exists():
        videos["reel"] = "highlights_reel.mp4"
    if (path / "highlights_montage.mp4").exists():
        videos["montage"] = "highlights_montage.mp4"
    clips = sorted(path.glob("highlight_*.mp4"))
    videos["clips"] = [f.name for f in clips if "spotlight" not in f.name]
    videos["spotlight"] = [f.name for f in clips if "spotlight" in f.name]
    return videos


def _thumbs(path: Path) -> List[Dict[str, Any]]:
    folder = path / "thumbs"
    if not folder.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for item in sorted(folder.glob("*.jpg")):
        match = re.match(r"^(\d+)", item.stem)
        index = int(match.group(1)) if match else len(out) + 1
        out.append({"name": item.name, "t": round(max(0, index - 1) * _THUMB_INTERVAL_S, 2)})
    return out


def _bookmark_events(bookmarks: List[Dict[str, Any]], trim: float) -> List[Dict[str, Any]]:
    """Fallback: present legacy bookmarks (source timebase) as v2 events (window timebase)."""
    events = []
    for bm in bookmarks:
        if not isinstance(bm, dict):
            continue
        occurred = float(bm.get("occurred_at_s", 0.0) or 0.0) - trim
        start = float(bm.get("start_s", occurred + trim) or 0.0) - trim
        end = float(bm.get("end_s", occurred + trim) or 0.0) - trim
        conf = float(bm.get("confidence", 0.0) or 0.0)
        events.append(
            {
                "id": str(bm.get("bookmark_id") or f"bm_{len(events):04d}"),
                "type": str(bm.get("event_type") or "highlight"),
                "t": round(occurred, 3),
                "t_start": round(start, 3),
                "t_end": round(end, 3),
                "team": None,
                "team_name": None,
                "player_track_id": None,
                "confidence": conf,
                "excitement": conf,
                "reason": str(bm.get("label") or ""),
                "sources": list(bm.get("sources") or []),
                "game_state": bm.get("game_state"),
                "bookmark_id": bm.get("bookmark_id"),
            }
        )
    return events


def _trim_offset(events_doc: Dict[str, Any], states: Dict[str, Any], tracks_meta: Dict[str, Any]) -> float:
    for value in (
        events_doc.get("trim_offset_seconds"),
        states.get("trim_offset_seconds"),
        tracks_meta.get("trim_offset_s"),
    ):
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _team_info(team_stats: Dict[str, Any], match: Optional[Match]) -> Tuple[List[str], List[Optional[str]], Optional[Dict[str, int]]]:
    """Normalize team names, colours and score from team stats v2 or v1."""
    names: List[str] = [
        (match.home_team_name if match and match.home_team_name else "Home"),
        (match.away_team_name if match and match.away_team_name else "Away"),
    ]
    colors: List[Optional[str]] = [None, None]
    score: Optional[Dict[str, int]] = None
    teams = team_stats.get("teams")
    if isinstance(teams, dict):  # v2
        score = {}
        for key in ("0", "1"):
            info = teams.get(key) or teams.get(int(key)) or {}
            if not isinstance(info, dict):
                continue
            idx = int(key)
            names[idx] = str(info.get("name") or names[idx])
            colors[idx] = info.get("color_hex")
            if info.get("goals") is not None:
                score[key] = int(info.get("goals") or 0)
        score = score if len(score) == 2 else None
    elif isinstance(teams, list):  # v1
        goals = team_stats.get("goals") or {}
        score = {}
        for idx, info in enumerate(teams[:2]):
            if not isinstance(info, dict):
                continue
            names[idx] = str(info.get("team") or names[idx])
            colors[idx] = info.get("color")
            if isinstance(goals, dict) and info.get("team") in goals:
                score[str(idx)] = int(goals.get(info.get("team")) or 0)
        score = score if len(score) == 2 else None
    return names, colors, score


def _score_from_events(events: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    goals = [e for e in events if e.get("type") == "goal" and e.get("team") in (0, 1)]
    if not goals:
        return None
    return {"0": sum(1 for g in goals if g.get("team") == 0), "1": sum(1 for g in goals if g.get("team") == 1)}


def _strip_player(player: Dict[str, Any]) -> Dict[str, Any]:
    slim = {k: v for k, v in player.items() if k != "speed_series"}
    slim["has_speed_series"] = bool(player.get("speed_series"))
    return slim


def _job_lookup(session: Optional[Session], tenant_id: Optional[str], run: Path) -> Tuple[Optional[ProcessingJob], Optional[Match]]:
    if session is None:
        return None, None
    job = session.get(ProcessingJob, run.name)
    if job is None or (tenant_id and job.tenant_id != tenant_id):
        job = None
        stmt = select(ProcessingJob).order_by(ProcessingJob.created_at.desc()).limit(2000)
        if tenant_id:
            stmt = stmt.where(ProcessingJob.tenant_id == tenant_id)
        for candidate in session.exec(stmt):
            if _job_run_path(candidate) == run:
                job = candidate
                break
    if job is None:
        return None, None
    match = session.get(Match, job.match_id)
    return job, match


def _job_output_dir(job: ProcessingJob) -> Path:
    config = dict(job.config_json or {})
    result = dict(job.result_json or {})
    raw = config.get("output_dir") or result.get("output_dir") or os.path.join(settings.output_root, job.id)
    try:
        return Path(str(raw)).expanduser().resolve()
    except Exception:
        return _output_root() / job.id


def _job_run_path(job: ProcessingJob) -> Optional[Path]:
    """The run dir for a job if it lives under the output root (servable)."""
    path = _job_output_dir(job)
    root = _output_root()
    if path != root and _is_within(path, root):
        return path
    return None


def _job_run_id(job: ProcessingJob) -> Optional[str]:
    path = _job_run_path(job)
    if path is None or not path.is_dir():
        return None
    rel = path.relative_to(_output_root())
    return rel.parts[0] if len(rel.parts) == 1 else None


def _map_db_events(
    events: List[Dict[str, Any]],
    db_events: List[Event],
    bookmarks: List[Dict[str, Any]],
    trim: float,
) -> None:
    """Attach ``db_event_id`` / ``db_status`` to analysis events (in place)."""
    if not db_events:
        return
    by_bookmark = {str(b.get("bookmark_id")): b for b in bookmarks if isinstance(b, dict)}
    by_id = {str(e.get("id")): e for e in events}
    claimed: set[str] = set()
    remaining: List[Event] = []
    for row in db_events:
        source = dict(row.source_json or {})
        evidence = dict(row.evidence_json or {})
        linked = source.get("analysis_event_id") or evidence.get("analysis_event_id")
        if not linked and evidence.get("bookmark_id"):
            bm = by_bookmark.get(str(evidence.get("bookmark_id"))) or {}
            linked = bm.get("event_id") or bm.get("analysis_event_id")
            if not linked and str(evidence.get("bookmark_id")) in by_id:
                linked = evidence.get("bookmark_id")
        target = by_id.get(str(linked)) if linked else None
        if target is not None and str(target.get("id")) not in claimed:
            target["db_event_id"] = row.id
            target["db_status"] = row.status
            claimed.add(str(target.get("id")))
        else:
            remaining.append(row)
    for row in remaining:
        t_src = float(row.occurred_at_ms) / 1000.0
        best = None
        best_dt = 1.25
        for ev in events:
            if str(ev.get("id")) in claimed:
                continue
            same_type = ev.get("type") == row.event_type or row.event_type == "shot"
            delta = abs(float(ev.get("t", 0.0) or 0.0) + trim - t_src)
            if same_type and delta <= best_dt:
                best, best_dt = ev, delta
        if best is not None:
            best["db_event_id"] = row.id
            best["db_status"] = row.status
            claimed.add(str(best.get("id")))


def _run_summary(
    path: Path,
    *,
    session: Optional[Session] = None,
    tenant_id: Optional[str] = None,
    light: bool = False,
    job: Optional[ProcessingJob] = None,
    match: Optional[Match] = None,
) -> Dict[str, Any]:
    manifest = _read_json(path / "analysis_bookmarks.json")
    states = _read_json(path / "analysis_game_states.json")
    events_doc = _read_json(path / "analysis_events.json")
    team_stats = _read_json(path / "analysis_team_stats.json")
    tracks_meta = _read_json(path / TRACKS_META_FILENAME)
    if job is None and session is not None:
        job, match = _job_lookup(session, tenant_id, path)
    trim = _trim_offset(events_doc, states, tracks_meta)
    bookmarks = [b for b in (manifest.get("bookmarks") or []) if isinstance(b, dict)]
    events = [dict(e) for e in (events_doc.get("events") or []) if isinstance(e, dict)]
    events_source = "analysis_events" if events else ("bookmarks" if bookmarks else "none")
    if not events:
        events = _bookmark_events(bookmarks, trim)
    names, colors, score = _team_info(team_stats, match)
    if score is None:
        score = _score_from_events(events)
    counts: Dict[str, int] = {}
    for ev in events:
        counts[str(ev.get("type"))] = counts.get(str(ev.get("type")), 0) + 1
    thumbs = _thumbs(path)
    videos = _videos(path)
    progress = _read_json(path / "progress.json")
    stats = manifest.get("stats", {}) if isinstance(manifest.get("stats"), dict) else {}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    summary: Dict[str, Any] = {
        "run_id": path.name,
        "match_id": job.match_id if job else None,
        "job_id": job.id if job else None,
        "job_status": job.status if job else None,
        "match_name": (match.name if match else None),
        "match_date": (match.match_date if match else None),
        "generated_at": events_doc.get("generated_at") or manifest.get("generated_at"),
        "modified_at": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat() if mtime else None,
        "video_path": manifest.get("video_path") or tracks_meta.get("source_video_path"),
        "trim_offset_seconds": trim,
        "duration_s": tracks_meta.get("duration_s") or stats.get("duration_s"),
        "team_names": names,
        "team_colors": colors,
        "score": score,
        "event_counts": counts,
        "events_source": events_source,
        "videos": videos,
        "thumb": thumbs[0]["name"] if thumbs else None,
        "tracks_available": TrackingResult.exists(path),
        "progress": progress or None,
        "stats": stats,
    }
    if light:
        return summary

    player_doc = _read_json(path / "analysis_player_stats.json")
    if player_doc:
        player_doc = dict(player_doc)
        player_doc["players"] = [_strip_player(p) for p in (player_doc.get("players") or []) if isinstance(p, dict)]
    if session is not None and job is not None:
        db_rows = list(session.exec(select(Event).where(Event.job_id == job.id).where(Event.tenant_id == job.tenant_id)))
        _map_db_events(events, db_rows, bookmarks, trim)
    crops_dir = path / "card_crops"
    summary.update(
        {
            "events": events,
            "reel_plan": events_doc.get("reel_plan") or {},
            "events_summary": events_doc.get("summary") or {},
            "player_stats": player_doc or {},
            "team_stats": team_stats,
            "camera_quality": _read_json(path / "camera_quality.json"),
            "thumbs": thumbs,
            "tracks_meta": {k: v for k, v in tracks_meta.items() if k != "players"},
            "player_labels": _read_json(path / LABELS_FILENAME).get("labels", {}),
            "calibration": _read_json(path / CALIBRATION_FILENAME) or None,
            "state_summary_s": states.get("state_summary_s", {}),
            "goal_events": states.get("goal_events", []),
            "card_events": states.get("card_events", []),
            "set_piece_events": states.get("set_piece_events", []),
            "segments": states.get("segments", [])[:2000] if isinstance(states.get("segments"), list) else [],
            "match_report": (path / "match_report.md").read_text(encoding="utf-8")[:60000]
            if (path / "match_report.md").exists()
            else None,
            "card_crops": sorted(f.name for f in crops_dir.glob("*.png")) if crops_dir.is_dir() else [],
            "bookmarks": bookmarks,
            "match": {
                "match_id": match.id,
                "name": match.name,
                "home_team_name": match.home_team_name,
                "away_team_name": match.away_team_name,
                "match_date": match.match_date,
            }
            if match
            else None,
            "job": {
                "job_id": job.id,
                "status": job.status,
                "stage": job.stage,
                "progress": job.progress,
                "config": dict(job.config_json or {}),
                "created_at": _iso(job.created_at),
                "completed_at": _iso(job.completed_at),
            }
            if job
            else None,
        }
    )
    return summary


def _optional_context(
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*_READ_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Tuple[Session, str]:
    return session, tenant.tenant_id


# ---------------------------------------------------------------------------
# Library / runs
# ---------------------------------------------------------------------------


def _scan_runs(limit: int = 300) -> List[Path]:
    root = _output_root()
    if not root.is_dir():
        return []
    runs = [child for child in root.iterdir() if _is_run_dir(child)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[:limit]


@router.get("/studio/runs")
def list_runs() -> Dict[str, object]:
    runs = [_run_summary(child, light=True) for child in _scan_runs(200)]
    return {"output_root": str(_output_root()), "runs": runs}


@router.get("/studio/library")
def library(
    limit: int = Query(default=200, ge=1, le=500),
    ctx: Tuple[Session, str] = Depends(_optional_context),
) -> Dict[str, object]:
    """Match-centric library: matches + their jobs + the latest run, in 2 queries."""
    session, tenant_id = ctx
    matches = list(
        session.exec(select(Match).where(Match.tenant_id == tenant_id).order_by(Match.created_at.desc()).limit(limit))
    )
    match_ids = [m.id for m in matches]
    jobs: List[ProcessingJob] = []
    if match_ids:
        jobs = list(
            session.exec(
                select(ProcessingJob)
                .where(ProcessingJob.tenant_id == tenant_id)
                .where(ProcessingJob.match_id.in_(match_ids))  # type: ignore[attr-defined]
                .order_by(ProcessingJob.created_at.desc())
            )
        )
    jobs_by_match: Dict[str, List[ProcessingJob]] = {}
    for job in jobs:
        jobs_by_match.setdefault(job.match_id, []).append(job)

    linked_runs: set[str] = set()
    items: List[Dict[str, Any]] = []
    for match in matches:
        match_jobs = jobs_by_match.get(match.id, [])
        latest = match_jobs[0] if match_jobs else None
        run_summary = None
        for job in match_jobs:
            run_path = _job_run_path(job)
            if run_path is not None and _is_run_dir(run_path):
                linked_runs.add(run_path.name)
                if run_summary is None and job.status not in _ACTIVE_STATUSES:
                    run_summary = _run_summary(run_path, light=True, job=job, match=match)
        items.append(
            {
                "match_id": match.id,
                "name": match.name,
                "home_team_name": match.home_team_name,
                "away_team_name": match.away_team_name,
                "match_date": match.match_date,
                "created_at": _iso(match.created_at),
                "source_video_path": match.source_video_path,
                "job_count": len(match_jobs),
                "latest_job": {
                    "job_id": latest.id,
                    "status": latest.status,
                    "stage": latest.stage,
                    "progress": latest.progress,
                    "created_at": _iso(latest.created_at),
                    "error_message": latest.error_message,
                }
                if latest
                else None,
                "run": run_summary,
            }
        )
    unlinked = [
        _run_summary(child, light=True)
        for child in _scan_runs(200)
        if child.name not in linked_runs and session.get(ProcessingJob, child.name) is None
    ]
    return {"output_root": str(_output_root()), "matches": items, "unlinked_runs": unlinked}


@router.get("/studio/runs/{run_id}")
def get_run(run_id: str, ctx: Tuple[Session, str] = Depends(_optional_context)) -> Dict[str, object]:
    session, tenant_id = ctx
    return _run_summary(_run_dir(run_id), session=session, tenant_id=tenant_id)


@router.get("/studio/runs/{run_id}/player-stats/{track_id}")
def get_player_stats(run_id: str, track_id: int) -> Dict[str, object]:
    doc = _read_json(_run_dir(run_id) / "analysis_player_stats.json")
    for player in doc.get("players") or []:
        if isinstance(player, dict) and int(player.get("track_id", -10**9)) == int(track_id):
            return {"run_id": run_id, "player": player, "pitch_calibration": doc.get("pitch_calibration") or {}}
    raise HTTPException(status_code=404, detail=f"No stats for track {track_id}")


@router.get("/studio/runs/{run_id}/file/{name}")
@router.head("/studio/runs/{run_id}/file/{name}", include_in_schema=False)
def get_run_file(run_id: str, name: str, request: Request) -> Response:
    run = _run_dir(run_id)
    if Path(name).suffix.lower() not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Invalid file name")
    path = _safe_child(run, name)
    if not path.is_file():
        path = _safe_child(run / "card_crops", name)  # card crops live one level down
        if not path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
    return media_response(request, path, download_name=path.name)


@router.get("/studio/runs/{run_id}/thumb/{name}")
@router.head("/studio/runs/{run_id}/thumb/{name}", include_in_schema=False)
def get_run_thumb(run_id: str, name: str, request: Request) -> Response:
    run = _run_dir(run_id)
    if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Invalid thumbnail name")
    path = _safe_child(run / "thumbs", name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return media_response(request, path)


@router.get("/studio/runs/{run_id}/poster")
@router.head("/studio/runs/{run_id}/poster", include_in_schema=False)
def get_run_poster(run_id: str, request: Request) -> Response:
    """First thumbnail, else a cached poster frame grabbed from the proxy/movie."""
    run = _run_dir(run_id)
    thumbs = _thumbs(run)
    if thumbs:
        return media_response(request, run / "thumbs" / thumbs[min(1, len(thumbs) - 1)]["name"])
    poster = run / POSTER_FILENAME
    if not poster.is_file():
        videos = _videos(run)
        source_name = videos.get("proxy") or videos.get("movie") or videos.get("original") or videos.get("reel")
        if not source_name:
            raise HTTPException(status_code=404, detail="No video to make a poster from")
        source = run / str(source_name)
        tmp = run / f".{POSTER_FILENAME}.{os.getpid()}.jpg"
        for seek in ("5", "0"):
            try:
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", seek, "-i", str(source),
                     "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", str(tmp)],
                    check=False, timeout=30, capture_output=True,
                )
            except Exception:
                break
            if tmp.is_file() and tmp.stat().st_size > 0:
                os.replace(tmp, poster)
                break
        if not poster.is_file():
            raise HTTPException(status_code=404, detail="Poster frame unavailable")
    return media_response(request, poster)


# ---------------------------------------------------------------------------
# Player picker
# ---------------------------------------------------------------------------


@router.get("/studio/runs/{run_id}/tracks")
def get_tracks_at(
    run_id: str,
    t: float = Query(..., ge=0.0, description="Seconds in the processing-window timebase"),
    tolerance: float = Query(default=0.25, gt=0.0, le=5.0),
) -> Dict[str, object]:
    run = _run_dir(run_id)
    tracking = _load_tracks(run)
    labels = _read_json(run / LABELS_FILENAME).get("labels", {})
    players = tracking.tracks_at(float(t), tolerance_s=float(tolerance))
    for player in players:
        label = labels.get(str(player["track_id"])) or {}
        player["name"] = label.get("name")
        player["number"] = label.get("number") or (
            str(player["jersey_number"]) if player.get("jersey_number") is not None else None
        )
        track = tracking.players.get(int(player["track_id"]))
        if track is not None:
            player["track_start_s"] = round(track.start_s, 2)
            player["track_end_s"] = round(track.end_s, 2)
            player["jersey_color_hex"] = track.jersey_color_hex
    ball = None
    if len(tracking.ball):
        import numpy as np

        idx = int(np.argmin(np.abs(tracking.ball.t - float(t))))
        if abs(float(tracking.ball.t[idx]) - float(t)) <= float(tolerance):
            ball = {
                "x": round(float(tracking.ball.x[idx]), 1),
                "y": round(float(tracking.ball.y[idx]), 1),
                "w": round(float(tracking.ball.w[idx]), 1),
                "h": round(float(tracking.ball.h[idx]), 1),
                "conf": round(float(tracking.ball.conf[idx]), 3),
            }
    return {
        "run_id": run_id,
        "t": float(t),
        "t_source": round(float(t) + float(tracking.trim_offset_s), 3),
        "tolerance_s": float(tolerance),
        "frame_width": tracking.frame_width,
        "frame_height": tracking.frame_height,
        "fps": tracking.fps,
        "duration_s": tracking.duration_s,
        "trim_offset_s": tracking.trim_offset_s,
        "focus_track_id": tracking.focus_track_id,
        "players": players,
        "ball": ball,
    }


class PlayerLabel(BaseModel):
    name: Optional[str] = Field(default=None, max_length=64)
    number: Optional[str] = Field(default=None, max_length=4)
    team: Optional[int] = Field(default=None, ge=-1, le=2)
    notes: Optional[str] = Field(default=None, max_length=280)


class PlayerLabelsPut(BaseModel):
    labels: Dict[str, PlayerLabel] = Field(default_factory=dict)


@router.get("/studio/runs/{run_id}/player-labels")
def get_player_labels(run_id: str) -> Dict[str, object]:
    doc = _read_json(_run_dir(run_id) / LABELS_FILENAME)
    return {"run_id": run_id, "updated_at": doc.get("updated_at"), "labels": doc.get("labels", {})}


@router.put("/studio/runs/{run_id}/player-labels")
def put_player_labels(
    run_id: str,
    payload: PlayerLabelsPut,
    _: UserContext = Depends(require_roles(*_WRITE_ROLES)),
) -> Dict[str, object]:
    """Replace the run's player labels (track_id -> {name, number, team, notes})."""
    run = _run_dir(run_id)
    labels: Dict[str, Dict[str, Any]] = {}
    for key, label in payload.labels.items():
        try:
            track_id = int(str(key))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Label keys must be track ids, got {key!r}")
        data = {k: (v.strip() if isinstance(v, str) else v) for k, v in label.model_dump().items()}
        data = {k: v for k, v in data.items() if v not in (None, "")}
        if data:
            labels[str(track_id)] = data
    doc = {"updated_at": utcnow().isoformat(), "labels": labels}
    _write_json_atomic(run / LABELS_FILENAME, doc)
    return {"run_id": run_id, **doc}


# ---------------------------------------------------------------------------
# Manual pitch calibration (calibration.json)
# ---------------------------------------------------------------------------
#
# The Studio "Calibrate pitch" mode stores the 4 playing-area corners (TL, TR,
# BR, BL) and optional goal mouths per run, always as normalized [0, 1]
# coordinates of the source frame. They are sent unchanged as the job config
# keys ``pitch_corners`` / ``goal_box_left`` / ``goal_box_right`` on
# "Re-analyze with calibration" (the pipeline scales normalized values by the
# source frame size; see VideoHighlights._normalize_corners and
# game_tracking._goal_box_from_override).


class CalibrationPut(BaseModel):
    # Loosely typed so shape errors come back as 400 with a clear message.
    pitch_corners: Any = None
    goal_box_left: Any = None
    goal_box_right: Any = None
    frame_width: Optional[int] = Field(default=None, ge=1, le=16384)
    frame_height: Optional[int] = Field(default=None, ge=1, le=16384)
    normalized: bool = True


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _convex_quad(points: List[List[float]]) -> bool:
    signs = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross)
    return all(v > 1e-9 for v in signs) or all(v < -1e-9 for v in signs)


def _calibration_frame(run: Path, payload: CalibrationPut) -> Tuple[Optional[int], Optional[int]]:
    meta = _read_json(run / TRACKS_META_FILENAME)
    try:
        width = int(meta.get("frame_width") or 0) or None
        height = int(meta.get("frame_height") or 0) or None
    except (TypeError, ValueError):
        width = height = None
    return width or payload.frame_width, height or payload.frame_height


def _normalize_point(x: Any, y: Any, normalized: bool, size: Tuple[Optional[int], Optional[int]], what: str) -> List[float]:
    fx, fy = _finite(x), _finite(y)
    if fx is None or fy is None:
        raise HTTPException(status_code=400, detail=f"{what}: coordinates must be finite numbers")
    if not normalized:
        width, height = size
        if not width or not height:
            raise HTTPException(status_code=400, detail=f"{what}: pixel coordinates need frame_width/frame_height")
        fx, fy = fx / width, fy / height
    if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
        raise HTTPException(status_code=400, detail=f"{what}: coordinates must lie inside the frame ([0, 1] normalized)")
    return [round(fx, 6), round(fy, 6)]


def _normalize_goal_box(raw: Any, side: str, normalized: bool,
                        size: Tuple[Optional[int], Optional[int]]) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if not isinstance(raw, dict) or any(k not in raw for k in ("x1", "y1", "x2", "y2")):
        raise HTTPException(status_code=400, detail=f"goal_box_{side} must be {{x1, y1, x2, y2}}")
    box_normalized = bool(raw.get("normalized", normalized))
    x1, y1 = _normalize_point(raw["x1"], raw["y1"], box_normalized, size, f"goal_box_{side}")
    x2, y2 = _normalize_point(raw["x2"], raw["y2"], box_normalized, size, f"goal_box_{side}")
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
        raise HTTPException(status_code=400, detail=f"goal_box_{side} is empty")
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "normalized": True}


@router.get("/studio/runs/{run_id}/calibration")
def get_calibration(run_id: str) -> Dict[str, object]:
    """The run's saved manual calibration (``null`` when none was saved)."""
    doc = _read_json(_run_dir(run_id) / CALIBRATION_FILENAME)
    return {"run_id": run_id, "calibration": doc or None}


@router.put("/studio/runs/{run_id}/calibration")
def put_calibration(
    run_id: str,
    payload: CalibrationPut,
    _: UserContext = Depends(require_roles(*_WRITE_ROLES)),
) -> Dict[str, object]:
    """Save the manual pitch calibration: 4 corners TL, TR, BR, BL (+ goal mouths).

    Stored normalized to the source frame; pixel input (``normalized: false``)
    is converted with the run's frame size.
    """
    run = _run_dir(run_id)
    size = _calibration_frame(run, payload)
    raw_corners = payload.pitch_corners
    if (not isinstance(raw_corners, list) or len(raw_corners) != 4
            or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in raw_corners)):
        raise HTTPException(status_code=400, detail="pitch_corners must be exactly 4 [x, y] points (TL, TR, BR, BL)")
    corners = [_normalize_point(p[0], p[1], payload.normalized, size, "pitch_corners") for p in raw_corners]
    width, height = size
    aspect = [float(width or 1), float(height or 1)]
    if not _convex_quad([[x * aspect[0], y * aspect[1]] for x, y in corners]):
        raise HTTPException(status_code=400, detail="pitch_corners must form a convex quadrilateral in order TL, TR, BR, BL")
    doc = {
        "pitch_corners": corners,
        "goal_box_left": _normalize_goal_box(payload.goal_box_left, "left", payload.normalized, size),
        "goal_box_right": _normalize_goal_box(payload.goal_box_right, "right", payload.normalized, size),
        "frame_width": width,
        "frame_height": height,
        "normalized": True,
        "updated_at": utcnow().isoformat(),
    }
    _write_json_atomic(run / CALIBRATION_FILENAME, doc)
    return {"run_id": run_id, "calibration": doc}


@router.delete("/studio/runs/{run_id}/calibration")
def delete_calibration(
    run_id: str,
    _: UserContext = Depends(require_roles(*_WRITE_ROLES)),
) -> Dict[str, object]:
    path = _run_dir(run_id) / CALIBRATION_FILENAME
    existed = path.exists()
    if existed:
        path.unlink()
    return {"run_id": run_id, "calibration": None, "deleted": existed}


# ---------------------------------------------------------------------------
# DB event bridge: analysis_events.json event -> Event row (feedback, clips, export)
# ---------------------------------------------------------------------------


@router.post("/studio/runs/{run_id}/events/{event_id}/db-event")
def ensure_db_event(
    run_id: str,
    event_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*_WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    """Return (creating if needed) the DB Event row for an analysis event.

    The clip-on-demand, export and feedback endpoints work on DB events; the
    engine only syncs reel-selected bookmarks, so any other event is
    materialized here on demand (source timebase, like the job sync).
    """
    run = _run_dir(run_id)
    summary = _run_summary(run, session=session, tenant_id=tenant.tenant_id)
    if not summary.get("job_id"):
        raise HTTPException(status_code=409, detail="This run is not linked to a processing job")
    event = next((e for e in summary.get("events", []) if str(e.get("id")) == event_id), None)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event not found in run: {event_id}")
    if event.get("db_event_id"):
        return {"db_event_id": event["db_event_id"], "created": False, "match_id": summary["match_id"]}
    job = session.get(ProcessingJob, summary["job_id"])
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    trim = float(summary.get("trim_offset_seconds") or 0.0)
    t = float(event.get("t", 0.0) or 0.0)
    t_start = float(event.get("t_start", t) if event.get("t_start") is not None else t)
    t_end = float(event.get("t_end", t) if event.get("t_end") is not None else t)
    occurred_ms = int(round((t + trim) * 1000))
    start_ms = max(0, min(occurred_ms, int(round((t_start + trim) * 1000))))
    end_ms = max(occurred_ms, int(round((t_end + trim) * 1000)))
    team = event.get("team")
    row = Event(
        tenant_id=job.tenant_id,
        match_id=job.match_id,
        job_id=job.id,
        event_type=str(event.get("type") or "highlight"),
        status="auto_detected",
        confidence=max(0.0, min(1.0, float(event.get("confidence", 0.0) or 0.0))),
        occurred_at_ms=occurred_ms,
        start_ms=start_ms,
        end_ms=end_ms,
        # team_id uses the stat catalog's home/away buckets; player_id is
        # reserved for roster assignment, so the track id goes in participants.
        team_id={0: "home", 1: "away"}.get(team) if isinstance(team, int) else None,
        participants_json=[{"player_id": f"track:{event['player_track_id']}", "role": "primary"}]
        if event.get("player_track_id") is not None
        else [],
        source_json={
            "detector": "event-engine",
            "analysis_event_id": event_id,
            "analysis_type": event.get("type"),
            "player_track_id": event.get("player_track_id"),
            "sources": list(event.get("sources") or []),
            "camera_mode": str((job.config_json or {}).get("camera_mode") or "wide"),
        },
        evidence_json={
            "analysis_event_id": event_id,
            "analysis_manifest_path": str(run / "analysis_events.json"),
            "tracking_manifest_path": str(run / "analysis_tracking.json"),
        },
        explanations_json=[
            {"signal": "excitement", "value": float(event.get("excitement", 0.0) or 0.0)},
        ],
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return {"db_event_id": row.id, "created": True, "match_id": job.match_id}


@router.get("/studio/matches/{match_id}/assets/{asset_id}/file")
@router.head("/studio/matches/{match_id}/assets/{asset_id}/file", include_in_schema=False)
def get_match_asset_file(
    match_id: str,
    asset_id: str,
    request: Request,
    session: Session = Depends(get_session),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Response:
    """Stream a locally stored match asset (exported reels, on-demand clips)."""
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    assets = list((match.metadata_json or {}).get("assets", []) or [])
    asset = next((a for a in assets if str(a.get("asset_id")) == asset_id), None)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")
    path = Path(str(asset.get("path") or ""))
    if path.suffix.lower() not in _ASSET_SUFFIXES or not path.is_file():
        raise HTTPException(status_code=404, detail="Asset file is not available locally")
    return media_response(request, path, download_name=str(asset.get("filename") or path.name))


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def _config_brief(config: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "profile", "camera_mode", "camera_style", "output_height", "focus_track_id",
        "reuse_tracking_from_job", "trim_start", "trim_end", "debug_video", "broadcast_reel",
        "player_spotlight_reel",
    )
    return {k: config.get(k) for k in keys if config.get(k) is not None}


def _derive_progress(job: ProcessingJob, latest_log: Optional[JobLogEntry]) -> Dict[str, Any]:
    data = dict(latest_log.data_json or {}) if latest_log else {}
    progress = float(job.progress or 0.0)
    if isinstance(data.get("progress"), (int, float)) and job.status in _ACTIVE_STATUSES:
        progress = max(progress, float(data["progress"]))
    started = _as_utc(job.started_at)
    finished = _as_utc(job.completed_at)
    now = datetime.now(timezone.utc)
    elapsed = ((finished or now) - started).total_seconds() if started else None
    eta = None
    if elapsed is not None and job.status in _ACTIVE_STATUSES and 0.02 < progress < 1.0:
        eta = elapsed * (1.0 - progress) / progress
    return {
        "source": "job_log" if latest_log else "job",
        "stage": (latest_log.stage if latest_log and latest_log.stage else job.stage),
        "stage_index": None,
        "stage_count": None,
        "progress": round(progress, 4),
        "stage_progress": data.get("stage_progress"),
        "eta_s": round(eta, 1) if eta is not None else None,
        "elapsed_s": round(elapsed, 1) if elapsed is not None else None,
        "fps_processing": data.get("fps_processing") or data.get("fps"),
        "message": latest_log.message if latest_log else None,
        "device": data.get("device"),
        "cancelled": bool(job.cancel_requested),
    }


def _stage_timings_from_logs(logs: List[JobLogEntry]) -> Dict[str, float]:
    """Seconds spent per stage, from consecutive log timestamps (oldest first)."""
    timings: Dict[str, float] = {}
    rows = sorted(logs, key=lambda r: r.created_at)
    for current, nxt in zip(rows, rows[1:]):
        stage = current.stage or "unknown"
        delta = (_as_utc(nxt.created_at) - _as_utc(current.created_at)).total_seconds()  # type: ignore[operator]
        if delta >= 0:
            timings[stage] = round(timings.get(stage, 0.0) + delta, 2)
    return timings


def _progress_payload(session: Session, job: ProcessingJob) -> Dict[str, Any]:
    out_dir = _job_output_dir(job)
    file_progress = (
        _read_json(out_dir / "progress.json") if out_dir.is_dir() and job.status in _STARTED_STATUSES else {}
    )
    latest_log = session.exec(
        select(JobLogEntry).where(JobLogEntry.job_id == job.id).order_by(JobLogEntry.created_at.desc()).limit(1)
    ).first()
    if file_progress:
        payload = {
            "source": "progress_file",
            "stage": file_progress.get("stage"),
            "stage_index": file_progress.get("stage_index"),
            "stage_count": file_progress.get("stage_count"),
            "progress": file_progress.get("progress"),
            "stage_progress": file_progress.get("stage_progress"),
            "eta_s": file_progress.get("eta_s"),
            "elapsed_s": file_progress.get("elapsed_s"),
            "fps_processing": file_progress.get("fps_processing"),
            "message": file_progress.get("message"),
            "device": file_progress.get("device"),
            "cancelled": bool(file_progress.get("cancelled") or job.cancel_requested),
            "stages": file_progress.get("stages"),
        }
        if job.status not in _ACTIVE_STATUSES:
            payload["eta_s"] = None
            if job.status == "completed":
                payload["progress"] = 1.0
    else:
        payload = _derive_progress(job, latest_log)
    timings = file_progress.get("stage_timings") or file_progress.get("timings") if file_progress else None
    if not timings:
        timings = _read_json(out_dir / TRACKS_META_FILENAME).get("timings") if out_dir.is_dir() else None
    if not timings and job.status not in _ACTIVE_STATUSES:
        logs = list(session.exec(select(JobLogEntry).where(JobLogEntry.job_id == job.id).limit(5000)))
        timings = _stage_timings_from_logs(logs)
    payload.update(
        {
            "job_id": job.id,
            "match_id": job.match_id,
            "status": job.status,
            "job_stage": job.stage,
            "job_progress": job.progress,
            "error_message": job.error_message,
            "run_id": _job_run_id(job),
            "timings": timings or {},
            "started_at": _iso(job.started_at),
            "completed_at": _iso(job.completed_at),
            "updated_at": _iso(job.updated_at),
        }
    )
    return payload


@router.get("/studio/jobs/{job_id}/progress")
def get_job_progress(
    job_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*_READ_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    job = session.get(ProcessingJob, job_id)
    if not job or job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return _progress_payload(session, job)


@router.get("/studio/jobs")
def list_studio_jobs(
    limit: int = Query(default=50, ge=1, le=500),
    status: Optional[str] = Query(default=None),
    match_id: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*_READ_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    """Jobs joined with their match in one query; live progress for active jobs."""
    stmt = (
        select(ProcessingJob, Match)
        .join(Match, Match.id == ProcessingJob.match_id)
        .where(ProcessingJob.tenant_id == tenant.tenant_id)
    )
    if status:
        statuses = [s.strip() for s in status.split(",") if s.strip()]
        stmt = stmt.where(ProcessingJob.status.in_(statuses))  # type: ignore[attr-defined]
    if match_id:
        stmt = stmt.where(ProcessingJob.match_id == match_id)
    stmt = stmt.order_by(ProcessingJob.created_at.desc()).limit(limit)
    items: List[Dict[str, Any]] = []
    active = 0
    for job, match in session.exec(stmt):
        config = dict(job.config_json or {})
        live = None
        if job.status in _ACTIVE_STATUSES:
            active += 1
            out_dir = _job_output_dir(job)
            live = _read_json(out_dir / "progress.json") if out_dir.is_dir() and job.status != "queued" else None
        items.append(
            {
                "job_id": job.id,
                "match_id": job.match_id,
                "match_name": match.name,
                "home_team_name": match.home_team_name,
                "away_team_name": match.away_team_name,
                "status": job.status,
                "stage": job.stage,
                "progress": job.progress,
                "cancel_requested": job.cancel_requested,
                "error_message": job.error_message,
                "created_at": _iso(job.created_at),
                "started_at": _iso(job.started_at),
                "completed_at": _iso(job.completed_at),
                "updated_at": _iso(job.updated_at),
                "config": _config_brief(config),
                "run_id": _job_run_id(job),
                "live": live or None,
            }
        )
    return {"items": items, "active_count": active}


# ---------------------------------------------------------------------------
# System / hardware
# ---------------------------------------------------------------------------

_ENCODERS = (
    "libx264", "libx265", "h264_nvenc", "hevc_nvenc", "av1_nvenc", "h264_videotoolbox",
    "hevc_videotoolbox", "h264_qsv", "h264_vaapi", "h264_amf",
)


@lru_cache(maxsize=1)
def _ffmpeg_capabilities() -> Dict[str, Any]:
    exe = shutil.which("ffmpeg")
    info: Dict[str, Any] = {"path": exe, "version": None, "encoders": {}, "hwaccels": []}
    if not exe:
        return info
    try:
        out = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, text=True, timeout=5).stdout
        info["version"] = (out.splitlines() or [""])[0].replace("ffmpeg version ", "").split(" Copyright")[0]
        enc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=5).stdout
        names = {line.split()[1] for line in enc.splitlines() if len(line.split()) > 1 and line.startswith(" ")}
        info["encoders"] = {name: name in names for name in _ENCODERS}
        hw = subprocess.run([exe, "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=5).stdout
        info["hwaccels"] = [line.strip() for line in hw.splitlines()[1:] if line.strip()]
    except Exception as exc:  # pragma: no cover - depends on host
        info["error"] = str(exc)
    return info


@router.get("/studio/system")
def system_info() -> Dict[str, object]:
    root = _output_root()
    disk = None
    try:
        usage = shutil.disk_usage(root if root.exists() else root.parent)
        disk = {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
    except Exception:
        disk = None
    return {
        "api_version": settings.api_version,
        "python": sys.version.split()[0],
        "output_root": str(root),
        "storage_backend": settings.storage_backend,
        "local_storage_root": str(Path(settings.local_storage_root).expanduser().resolve()),
        "job_execution_mode": settings.job_execution_mode,
        "auth_required": settings.auth_required,
        "disk": disk,
        "ffmpeg": _ffmpeg_capabilities(),
        "server_time": datetime.now(timezone.utc).isoformat(),
        "uptime_hint_s": round(time.monotonic(), 1),
    }
