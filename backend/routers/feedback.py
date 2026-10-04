from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..database import get_session
from ..models import Event, EventFeedback, Match
from ..schemas import FeedbackCreate, FeedbackRead, FeedbackReviewRequest
from ..serializers import feedback_to_read
from ..tenant import TenantContext, get_tenant_context
from ..utils import decode_cursor, encode_cursor, generate_id, utcnow

router = APIRouter(tags=["feedback"])

VALID_REVIEW_DECISIONS = {"approved", "rejected", "needs_more_info", "merged"}


def _ensure_match(session: Session, match_id: str, tenant_id: str) -> Match:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    return match


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ISO timestamp: {value}") from exc
    # Stored timestamps are timezone-aware (UTC); a date-only or naive input
    # is interpreted as UTC so the comparison is valid on every backend.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _create_feedback_record(
    tenant_id: str,
    match_id: str,
    event_id: Optional[str],
    payload: FeedbackCreate,
    user: UserContext,
) -> EventFeedback:
    submit_user_id = payload.submitted_by.user_id or user.user_id
    submit_role = payload.submitted_by.role or user.role
    return EventFeedback(
        tenant_id=tenant_id,
        match_id=match_id,
        event_id=event_id,
        feedback_type=payload.feedback_type,
        status=payload.status,
        severity=payload.severity,
        comment=payload.comment,
        submitted_by_user_id=submit_user_id,
        submitted_by_role=submit_role,
        correction_json=payload.correction.model_dump(exclude_none=True),
        evidence_json=[item.model_dump(exclude_none=True) for item in payload.evidence],
        review_json={},
    )


@router.post("/matches/{match_id}/events/{event_id}/feedback", response_model=FeedbackRead, status_code=201)
def submit_event_feedback(
    match_id: str,
    event_id: str,
    payload: FeedbackCreate,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> FeedbackRead:
    _ensure_match(session, match_id, tenant.tenant_id)
    event = session.exec(
        select(Event)
        .where(Event.match_id == match_id)
        .where(Event.tenant_id == tenant.tenant_id)
        .where(Event.id == event_id)
    ).first()
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    feedback = _create_feedback_record(tenant.tenant_id, match_id, event_id, payload, user)
    session.add(feedback)
    session.commit()
    session.refresh(feedback)
    return feedback_to_read(feedback)


@router.post("/matches/{match_id}/feedback", response_model=FeedbackRead, status_code=201)
def submit_missed_event_feedback(
    match_id: str,
    payload: FeedbackCreate,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> FeedbackRead:
    _ensure_match(session, match_id, tenant.tenant_id)
    if payload.feedback_type != "missed_event":
        raise HTTPException(
            status_code=400,
            detail="Endpoint /matches/{match_id}/feedback is reserved for feedback_type=missed_event",
        )

    feedback = _create_feedback_record(tenant.tenant_id, match_id, None, payload, user)
    session.add(feedback)
    session.commit()
    session.refresh(feedback)
    return feedback_to_read(feedback)


@router.get("/matches/{match_id}/feedback", response_model=Dict[str, object])
def list_feedback(
    match_id: str,
    feedback_type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    submitted_by_user_id: Optional[str] = Query(default=None),
    from_created_at: Optional[str] = Query(default=None),
    to_created_at: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    cursor: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    _ensure_match(session, match_id, tenant.tenant_id)
    offset = decode_cursor(cursor)
    from_ts = _parse_iso(from_created_at)
    to_ts = _parse_iso(to_created_at)

    stmt = (
        select(EventFeedback)
        .where(EventFeedback.match_id == match_id)
        .where(EventFeedback.tenant_id == tenant.tenant_id)
    )
    if feedback_type:
        stmt = stmt.where(EventFeedback.feedback_type == feedback_type)
    if status:
        stmt = stmt.where(EventFeedback.status == status)
    if severity:
        stmt = stmt.where(EventFeedback.severity == severity)
    if submitted_by_user_id:
        stmt = stmt.where(EventFeedback.submitted_by_user_id == submitted_by_user_id)
    if from_ts:
        stmt = stmt.where(EventFeedback.created_at >= from_ts)
    if to_ts:
        stmt = stmt.where(EventFeedback.created_at <= to_ts)

    stmt = stmt.order_by(EventFeedback.created_at.desc()).offset(offset).limit(limit + 1)
    rows = list(session.exec(stmt))
    has_more = len(rows) > limit
    items = [feedback_to_read(item) for item in rows[:limit]]
    next_cursor = encode_cursor(offset + limit) if has_more else None
    return {"items": [item.model_dump() for item in items], "next_cursor": next_cursor}


@router.get("/matches/{match_id}/feedback/{feedback_id}", response_model=FeedbackRead)
def get_feedback(
    match_id: str,
    feedback_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> FeedbackRead:
    _ensure_match(session, match_id, tenant.tenant_id)
    feedback = session.exec(
        select(EventFeedback)
        .where(EventFeedback.match_id == match_id)
        .where(EventFeedback.tenant_id == tenant.tenant_id)
        .where(EventFeedback.id == feedback_id)
    ).first()
    if not feedback:
        raise HTTPException(status_code=404, detail=f"Feedback not found: {feedback_id}")
    return feedback_to_read(feedback)


@router.post("/matches/{match_id}/feedback/{feedback_id}/review", response_model=FeedbackRead)
def review_feedback(
    match_id: str,
    feedback_id: str,
    payload: FeedbackReviewRequest,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles("admin", "analyst", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> FeedbackRead:
    _ensure_match(session, match_id, tenant.tenant_id)
    feedback = session.exec(
        select(EventFeedback)
        .where(EventFeedback.match_id == match_id)
        .where(EventFeedback.tenant_id == tenant.tenant_id)
        .where(EventFeedback.id == feedback_id)
    ).first()
    if not feedback:
        raise HTTPException(status_code=404, detail=f"Feedback not found: {feedback_id}")
    if payload.review_decision not in VALID_REVIEW_DECISIONS:
        raise HTTPException(status_code=400, detail="Invalid review_decision")

    feedback.status = payload.review_decision
    feedback.review_json = {
        "review_decision": payload.review_decision,
        "review_note": payload.review_note,
        "reviewed_by_user_id": user.user_id,
        "reviewed_at": utcnow().isoformat(),
    }
    feedback.updated_at = utcnow()

    if payload.review_decision == "approved":
        correction = feedback.correction_json or {}
        if feedback.event_id:
            event = session.exec(
                select(Event)
                .where(Event.match_id == match_id)
                .where(Event.tenant_id == tenant.tenant_id)
                .where(Event.id == feedback.event_id)
            ).first()
            if event:
                if correction.get("expected_event_type"):
                    event.event_type = correction["expected_event_type"]
                if correction.get("corrected_occurred_at_ms") is not None:
                    event.occurred_at_ms = int(correction["corrected_occurred_at_ms"])
                if correction.get("corrected_start_ms") is not None:
                    event.start_ms = int(correction["corrected_start_ms"])
                if correction.get("corrected_end_ms") is not None:
                    event.end_ms = int(correction["corrected_end_ms"])
                if correction.get("corrected_team_id"):
                    event.team_id = correction["corrected_team_id"]
                if correction.get("corrected_player_id"):
                    event.player_id = correction["corrected_player_id"]
                if correction.get("corrected_jersey_number"):
                    event.jersey_number = correction["corrected_jersey_number"]
                event.status = "corrected"
                event.updated_at = utcnow()
                session.add(event)
        elif feedback.feedback_type == "missed_event":
            event = Event(
                id=generate_id("evt"),
                tenant_id=tenant.tenant_id,
                match_id=match_id,
                event_type=correction.get("expected_event_type", "kickoff"),
                status="confirmed",
                confidence=1.0,
                period=None,
                occurred_at_ms=int(correction.get("corrected_occurred_at_ms") or 0),
                start_ms=int(correction.get("corrected_start_ms") or correction.get("corrected_occurred_at_ms") or 0),
                end_ms=int(correction.get("corrected_end_ms") or correction.get("corrected_occurred_at_ms") or 0),
                frame_index=0,
                team_id=correction.get("corrected_team_id"),
                player_id=correction.get("corrected_player_id"),
                jersey_number=correction.get("corrected_jersey_number"),
                source_json={"detector": "human_review"},
            )
            session.add(event)

    session.add(feedback)
    session.commit()
    session.refresh(feedback)
    return feedback_to_read(feedback)
