"""Baseline per-team match statistics catalog (FR-STATS-01/02/04).

Computes the guaranteed 15-stat catalog from persisted Event rows and the
``analysis_team_stats.json`` artifact of the most recent completed run.
Stats the pipeline cannot yet produce are reported as unavailable with a
reason instead of a misleading zero, and every event-derived stat carries
the contributing event ids so the UI can drill down to video evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlmodel import Session, select

from ..config import settings
from ..models import Event, Match, ProcessingJob
from ..schemas import MatchStatsRead, StatValue
from ..utils import utcnow
from .source_catalog import get_source_type

BASELINE_STATS: List[Tuple[str, str, str]] = [
    ("goals", "Goals", "count"),
    ("assists", "Assists", "count"),
    ("possession", "Possession", "percent"),
    ("total_shots", "Total Shots", "count"),
    ("shots_on_target", "Shots on Target", "count"),
    ("saves", "Saves", "count"),
    ("offsides", "Offsides", "count"),
    ("total_passes", "Total Passes", "count"),
    ("pass_accuracy", "Pass Accuracy", "percent"),
    ("key_passes", "Key Passes", "count"),
    ("duels", "Duels", "count"),
    ("fouls", "Fouls", "count"),
    ("corners", "Corners", "count"),
    ("free_kicks", "Free Kicks", "count"),
    ("penalties", "Penalties", "count"),
]

# Simple 1:1 event-count stats. Shots and shots-on-target are derived below.
EVENT_COUNT_STATS: Dict[str, str] = {
    "goals": "goal",
    "saves": "save",
    "fouls": "foul",
    "corners": "corner_kick",
    "free_kicks": "free_kick",
    "penalties": "penalty_kick",
}

NOT_YET_DETECTED = {"assists", "offsides", "total_passes", "pass_accuracy", "key_passes", "duels"}

# Two shot-ish events closer than this are treated as one attempt
# (a goal bookmark and a shot bookmark for the same strike).
SHOT_DEDUP_WINDOW_MS = 4000


def latest_completed_job(session: Session, match_id: str, tenant_id: Optional[str]) -> Optional[ProcessingJob]:
    stmt = (
        select(ProcessingJob)
        .where(ProcessingJob.match_id == match_id)
        .where(ProcessingJob.tenant_id == tenant_id)
        .where(ProcessingJob.status == "completed")
        .order_by(ProcessingJob.completed_at.desc())
    )
    return session.exec(stmt).first()


def _team_bucket(team_id: Optional[str], home_name: Optional[str], away_name: Optional[str]) -> str:
    value = str(team_id or "").strip().lower()
    if not value:
        return "unattributed"
    if value in {"home", "team_home"} or (home_name and value == home_name.strip().lower()):
        return "home"
    if value in {"away", "team_away"} or (away_name and value == away_name.strip().lower()):
        return "away"
    return "unattributed"


def _count_stat(events: Sequence[Event], event_type: str, home: Optional[str], away: Optional[str]) -> Dict[str, Any]:
    buckets = {"home": 0.0, "away": 0.0, "unattributed": 0.0}
    event_ids: List[str] = []
    for event in events:
        if event.event_type != event_type:
            continue
        buckets[_team_bucket(event.team_id, home, away)] += 1
        event_ids.append(event.id)
    return {**buckets, "total": sum(buckets.values()), "event_ids": event_ids}


def _dedup_attempts(events: Sequence[Event], types: set, home: Optional[str], away: Optional[str]) -> Dict[str, Any]:
    """Count distinct attempts, merging same-team events inside the dedup window."""
    rows = sorted((e for e in events if e.event_type in types), key=lambda e: e.occurred_at_ms)
    buckets = {"home": 0.0, "away": 0.0, "unattributed": 0.0}
    event_ids: List[str] = []
    last_ms: Optional[int] = None
    last_bucket: Optional[str] = None
    for event in rows:
        bucket = _team_bucket(event.team_id, home, away)
        event_ids.append(event.id)
        if last_ms is not None and bucket == last_bucket and event.occurred_at_ms - last_ms <= SHOT_DEDUP_WINDOW_MS:
            last_ms = event.occurred_at_ms
            continue
        buckets[bucket] += 1
        last_ms = event.occurred_at_ms
        last_bucket = bucket
    return {**buckets, "total": sum(buckets.values()), "event_ids": event_ids}


def _load_team_stats_artifact(job: Optional[ProcessingJob]) -> Dict[str, Any]:
    if not job:
        return {}
    result = job.result_json or {}
    candidates = []
    output_dir = str(result.get("output_dir") or "").strip()
    if output_dir:
        candidates.append(Path(output_dir) / "analysis_team_stats.json")
    manifest_path = str(result.get("analysis_manifest_path") or "").strip()
    if manifest_path:
        candidates.append(Path(manifest_path).parent / "analysis_team_stats.json")
    candidates.append(Path(settings.output_root) / job.id / "analysis_team_stats.json")
    for path in candidates:
        try:
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
        except Exception:
            continue
    return {}


def _possession_stat(artifact: Dict[str, Any], home: Optional[str], away: Optional[str]) -> Dict[str, Any]:
    possession = artifact.get("possession_pct") or {}
    if not isinstance(possession, dict) or not possession:
        return {}
    mapped: Dict[str, float] = {}
    raw: Dict[str, float] = {}
    for name, pct in possession.items():
        try:
            value = float(pct)
        except (TypeError, ValueError):
            continue
        raw[str(name)] = value
        bucket = _team_bucket(str(name), home, away)
        if bucket in {"home", "away"} and bucket not in mapped:
            mapped[bucket] = value
    return {"mapped": mapped, "raw": raw}


def compute_match_stat_catalog(
    session: Session,
    match: Match,
    tenant_id: Optional[str],
    job_id: Optional[str] = None,
) -> MatchStatsRead:
    job: Optional[ProcessingJob] = None
    if job_id:
        job = session.get(ProcessingJob, job_id)
        if job and (job.match_id != match.id or job.tenant_id != tenant_id):
            job = None
    if job is None:
        job = latest_completed_job(session, match.id, tenant_id)

    stmt = select(Event).where(Event.match_id == match.id).where(Event.tenant_id == tenant_id)
    if job is not None:
        stmt = stmt.where(Event.job_id == job.id)
    events = list(session.exec(stmt))

    home = match.home_team_name
    away = match.away_team_name
    artifact = _load_team_stats_artifact(job)
    has_analysis = job is not None or bool(events)
    source = get_source_type(str((match.metadata_json or {}).get("source_type") or "") or None)

    stats: List[StatValue] = []
    for key, label, unit in BASELINE_STATS:
        value = StatValue(key=key, label=label, unit=unit)
        if not has_analysis:
            value.reason = "no_completed_analysis"
            stats.append(value)
            continue
        # A link source that cannot supply this statistic is reported as
        # unavailable no matter what the pipeline produced.
        if not source.supports(key):
            value.reason = "not_available_for_source"
            value.raw = {"source": source.key, "source_label": source.label}
            stats.append(value)
            continue

        if key in EVENT_COUNT_STATS:
            counted = _count_stat(events, EVENT_COUNT_STATS[key], home, away)
            value.available = True
            value.method = f"count of detected '{EVENT_COUNT_STATS[key]}' events"
            value.home, value.away = counted["home"], counted["away"]
            value.unattributed, value.total = counted["unattributed"], counted["total"]
            value.event_ids = counted["event_ids"]
        elif key == "total_shots":
            counted = _dedup_attempts(events, {"shot", "goal"}, home, away)
            value.available = True
            value.method = "distinct shot/goal attempts (same-team events within 4s merged)"
            value.home, value.away = counted["home"], counted["away"]
            value.unattributed, value.total = counted["unattributed"], counted["total"]
            value.event_ids = counted["event_ids"]
        elif key == "shots_on_target":
            counted = _dedup_attempts(events, {"goal", "save"}, home, away)
            value.available = True
            value.method = "goals plus saves (every goal and saved shot was on target); save events attribute to the saving team when tagged"
            value.home, value.away = counted["home"], counted["away"]
            value.unattributed, value.total = counted["unattributed"], counted["total"]
            value.event_ids = counted["event_ids"]
        elif key == "possession":
            possession = _possession_stat(artifact, home, away)
            if possession:
                value.available = True
                value.method = "nearest-player-to-ball sampling from analysis_team_stats.json"
                value.raw = possession["raw"]
                value.home = possession["mapped"].get("home")
                value.away = possession["mapped"].get("away")
            else:
                value.reason = "team_stats_artifact_missing"
        elif key in NOT_YET_DETECTED:
            value.reason = "not_detected_by_pipeline"
        stats.append(value)

    return MatchStatsRead(
        match_id=match.id,
        job_id=job.id if job else None,
        teams={"home": home, "away": away},
        generated_at=utcnow().isoformat(),
        analysis={
            "has_completed_job": job is not None,
            "event_count": len(events),
            "team_stats_artifact": bool(artifact),
            "source_type": source.key,
            "source_label": source.label,
            "source_supported_stat_count": len(source.supported_stats),
        },
        stats=stats,
    )
