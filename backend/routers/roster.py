"""Match roster management and highlight assignment (FR-ROSTER-01/04/06).

Coaches upload a roster (player name, jersey number, position, email) after
team-level stats complete, via single entries or a CSV template. Highlights
that automatic attribution misses stay listed as unassigned and can be
manually assigned to a roster entry here.
"""

from __future__ import annotations

import csv
import io
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..database import get_session
from ..models import Event, Match, RosterEntry, RosterTemplate
from ..schemas import (
    EventAssignRequest,
    EventRead,
    PlayerCardRead,
    PlayerCardSendResult,
    RosterEntryCreate,
    RosterEntryPatch,
    RosterEntryRead,
    RosterImportError,
    RosterImportRequest,
    RosterImportResult,
    RosterTemplateApply,
    RosterTemplateCreate,
    RosterTemplateFromMatch,
    RosterTemplateRead,
    RoutingResult,
)
from ..serializers import event_to_read, roster_template_to_read, roster_to_read
from ..services.notifications import send_notification
from ..services.player_routing import build_player_card, route_match_events
from ..services.sharing import create_share_link, share_url_path
from ..tenant import TenantContext, get_tenant_context
from ..utils import utcnow

router = APIRouter(tags=["roster"])

READ_ROLES = ("admin", "analyst", "coach", "parent", "system", "tenant_admin")
WRITE_ROLES = ("admin", "analyst", "coach", "tenant_admin")

TEMPLATE_CSV = (
    "player_name,jersey_number,position,email\n"
    "Alex Morgan,13,Forward,alex@example.com\n"
    "Sam Kerr,20,Forward,sam@example.com\n"
)

# Flexible header aliases so common spreadsheet exports import cleanly.
HEADER_ALIASES = {
    "player_name": {"player_name", "player", "name", "full_name"},
    "jersey_number": {"jersey_number", "jersey", "number", "shirt", "shirt_number", "kit_number"},
    "position": {"position", "pos", "field_position"},
    "email": {"email", "e-mail", "email_address"},
}


def _ensure_match(session: Session, match_id: str, tenant_id: Optional[str]) -> Match:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    return match


def _map_headers(fieldnames: list[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw in fieldnames or []:
        normalized = str(raw or "").strip().lower().replace(" ", "_")
        for canonical, aliases in HEADER_ALIASES.items():
            if normalized in aliases and canonical not in mapping:
                mapping[canonical] = raw
    return mapping


@router.get("/matches/roster-template.csv", response_class=PlainTextResponse)
def get_roster_template(
    _: UserContext = Depends(require_roles(*READ_ROLES)),
) -> PlainTextResponse:
    return PlainTextResponse(
        TEMPLATE_CSV,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="roster_template.csv"'},
    )


@router.get("/matches/{match_id}/roster", response_model=Dict[str, object])
def list_roster(
    match_id: str,
    team_side: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*READ_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    _ensure_match(session, match_id, tenant.tenant_id)
    stmt = (
        select(RosterEntry)
        .where(RosterEntry.match_id == match_id)
        .where(RosterEntry.tenant_id == tenant.tenant_id)
    )
    if team_side:
        stmt = stmt.where(RosterEntry.team_side == team_side)
    rows = list(session.exec(stmt))
    rows.sort(key=lambda item: (item.team_side, len(item.jersey_number), item.jersey_number))
    return {"items": [roster_to_read(row).model_dump() for row in rows]}


@router.post("/matches/{match_id}/roster", response_model=RosterEntryRead, status_code=201)
def create_roster_entry(
    match_id: str,
    payload: RosterEntryCreate,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> RosterEntryRead:
    _ensure_match(session, match_id, tenant.tenant_id)
    jersey = payload.jersey_number.strip()
    name = payload.player_name.strip()
    if not name or not jersey:
        raise HTTPException(status_code=400, detail="player_name and jersey_number are required")
    existing = session.exec(
        select(RosterEntry)
        .where(RosterEntry.match_id == match_id)
        .where(RosterEntry.tenant_id == tenant.tenant_id)
        .where(RosterEntry.team_side == payload.team_side)
        .where(RosterEntry.jersey_number == jersey)
    ).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Jersey #{jersey} already exists on the {payload.team_side} roster",
        )
    entry = RosterEntry(
        tenant_id=tenant.tenant_id,
        match_id=match_id,
        player_name=name,
        jersey_number=jersey,
        position=(payload.position or "").strip() or None,
        email=(payload.email or "").strip() or None,
        team_side=payload.team_side,
        metadata_json=payload.metadata,
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return roster_to_read(entry)


@router.post("/matches/{match_id}/roster/import", response_model=RosterImportResult)
def import_roster(
    match_id: str,
    payload: RosterImportRequest,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> RosterImportResult:
    _ensure_match(session, match_id, tenant.tenant_id)

    reader = csv.DictReader(io.StringIO(payload.csv_text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV is empty")
    headers = _map_headers(list(reader.fieldnames))
    if "player_name" not in headers or "jersey_number" not in headers:
        raise HTTPException(
            status_code=400,
            detail="CSV must include player_name and jersey_number columns (see roster-template.csv)",
        )

    existing_stmt = (
        select(RosterEntry)
        .where(RosterEntry.match_id == match_id)
        .where(RosterEntry.tenant_id == tenant.tenant_id)
        .where(RosterEntry.team_side == payload.team_side)
    )
    existing = {entry.jersey_number: entry for entry in session.exec(existing_stmt)}
    if payload.replace_existing:
        for entry in existing.values():
            session.delete(entry)
        session.flush()
        existing = {}

    result = RosterImportResult()
    seen_jerseys: set[str] = set()
    for line_number, row in enumerate(reader, start=2):
        name = str(row.get(headers["player_name"], "") or "").strip()
        jersey = str(row.get(headers["jersey_number"], "") or "").strip()
        position = str(row.get(headers.get("position", ""), "") or "").strip() or None
        email = str(row.get(headers.get("email", ""), "") or "").strip() or None
        if not name and not jersey:
            result.skipped += 1
            continue
        if not name or not jersey:
            result.errors.append(RosterImportError(line=line_number, issue="player_name and jersey_number are required"))
            continue
        if jersey in seen_jerseys:
            result.errors.append(RosterImportError(line=line_number, issue=f"Duplicate jersey #{jersey} in file"))
            continue
        seen_jerseys.add(jersey)

        entry = existing.get(jersey)
        if entry:
            entry.player_name = name
            entry.position = position
            entry.email = email
            entry.updated_at = utcnow()
            session.add(entry)
            result.updated += 1
        else:
            entry = RosterEntry(
                tenant_id=tenant.tenant_id,
                match_id=match_id,
                player_name=name,
                jersey_number=jersey,
                position=position,
                email=email,
                team_side=payload.team_side,
            )
            session.add(entry)
            existing[jersey] = entry
            result.created += 1

    session.commit()
    rows = list(session.exec(existing_stmt))
    rows.sort(key=lambda item: (len(item.jersey_number), item.jersey_number))
    result.entries = [roster_to_read(row) for row in rows]
    return result


@router.patch("/matches/{match_id}/roster/{entry_id}", response_model=RosterEntryRead)
def update_roster_entry(
    match_id: str,
    entry_id: str,
    payload: RosterEntryPatch,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> RosterEntryRead:
    _ensure_match(session, match_id, tenant.tenant_id)
    entry = session.get(RosterEntry, entry_id)
    if not entry or entry.match_id != match_id or entry.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Roster entry not found: {entry_id}")

    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        if key == "metadata":
            entry.metadata_json = value or {}
        else:
            setattr(entry, key, value)
    entry.updated_at = utcnow()
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return roster_to_read(entry)


@router.delete("/matches/{match_id}/roster/{entry_id}", response_model=Dict[str, object])
def delete_roster_entry(
    match_id: str,
    entry_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    _ensure_match(session, match_id, tenant.tenant_id)
    entry = session.get(RosterEntry, entry_id)
    if not entry or entry.match_id != match_id or entry.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Roster entry not found: {entry_id}")

    # Keep events but drop the dangling assignment so they show as unassigned again.
    assigned_events = list(
        session.exec(
            select(Event)
            .where(Event.match_id == match_id)
            .where(Event.tenant_id == tenant.tenant_id)
            .where(Event.player_id == entry_id)
        )
    )
    for event in assigned_events:
        event.player_id = None
        event.jersey_number = None
        event.updated_at = utcnow()
        session.add(event)

    session.delete(entry)
    session.commit()
    return {"deleted": True, "roster_entry_id": entry_id, "unassigned_events": len(assigned_events)}


@router.post("/matches/{match_id}/roster/route", response_model=RoutingResult)
def route_highlights_to_players(
    match_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> RoutingResult:
    """Attach highlights carrying a jersey number to their roster entry."""
    _ensure_match(session, match_id, tenant.tenant_id)
    result = route_match_events(session, match_id, tenant.tenant_id)
    session.commit()
    return result


@router.get("/matches/{match_id}/roster/{entry_id}/card", response_model=PlayerCardRead)
def get_player_card(
    match_id: str,
    entry_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*READ_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> PlayerCardRead:
    match = _ensure_match(session, match_id, tenant.tenant_id)
    entry = session.get(RosterEntry, entry_id)
    if not entry or entry.match_id != match_id or entry.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Roster entry not found: {entry_id}")
    return build_player_card(session, match, entry)


@router.post("/matches/{match_id}/roster/cards/send", response_model=PlayerCardSendResult)
def send_player_cards(
    match_id: str,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> PlayerCardSendResult:
    """Email each rostered player a shareable link to their player card."""
    match = _ensure_match(session, match_id, tenant.tenant_id)
    entries = list(
        session.exec(
            select(RosterEntry)
            .where(RosterEntry.match_id == match_id)
            .where(RosterEntry.tenant_id == tenant.tenant_id)
        )
    )
    result = PlayerCardSendResult(match_id=match_id)
    for entry in entries:
        card = build_player_card(session, match, entry)
        if not entry.email:
            result.skipped += 1
            result.details.append({
                "roster_entry_id": entry.id,
                "player_name": entry.player_name,
                "status": "skipped",
                "reason": "no email address on the roster entry",
            })
            continue

        link = create_share_link(
            session=session,
            match=match,
            scope="player_card",
            roster_entry_id=entry.id,
            label=f"Player card: {entry.player_name}",
            created_by_user_id=user.user_id,
        )
        session.flush()  # assign the token before it goes into the email body
        url_path = share_url_path(link.token)
        match_label = match.name or match_id
        body = (
            f"Hi {entry.player_name},\n\n"
            f"Your player card for '{match_label}' is ready with "
            f"{card.highlight_count} highlight(s) our system attributed to #{entry.jersey_number}.\n\n"
            f"View and share it here: {url_path}\n\n"
            f"The link is public — send it to family, recruiters, or agents."
        )
        entry_log = send_notification(
            session=session,
            recipient=entry.email,
            subject=f"Your player card is ready: {match_label}",
            body=body,
            tenant_id=tenant.tenant_id,
            match_id=match_id,
        )
        sent = entry_log.status == "sent"
        result.sent += 1 if sent else 0
        result.skipped += 0 if sent else 1
        result.details.append({
            "roster_entry_id": entry.id,
            "player_name": entry.player_name,
            "status": entry_log.status,
            "reason": entry_log.error_message,
            "share_url_path": url_path,
            "highlight_count": card.highlight_count,
        })

    session.commit()
    return result


# ---- reusable roster templates (FR-ROSTER-05) ----


@router.get("/roster-templates", response_model=Dict[str, object])
def list_roster_templates(
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*READ_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    rows = list(
        session.exec(
            select(RosterTemplate)
            .where(RosterTemplate.tenant_id == tenant.tenant_id)
            .order_by(RosterTemplate.name)
        )
    )
    return {"items": [roster_template_to_read(row).model_dump() for row in rows]}


@router.post("/roster-templates", response_model=RosterTemplateRead, status_code=201)
def create_roster_template(
    payload: RosterTemplateCreate,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> RosterTemplateRead:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Template name is required")
    template = _upsert_template(
        session=session,
        tenant_id=tenant.tenant_id,
        name=name,
        description=payload.description,
        entries=[entry.model_dump() for entry in payload.entries],
        user_id=user.user_id,
    )
    session.commit()
    session.refresh(template)
    return roster_template_to_read(template)


@router.post("/matches/{match_id}/roster/save-template", response_model=RosterTemplateRead, status_code=201)
def save_roster_as_template(
    match_id: str,
    payload: RosterTemplateFromMatch,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> RosterTemplateRead:
    _ensure_match(session, match_id, tenant.tenant_id)
    stmt = (
        select(RosterEntry)
        .where(RosterEntry.match_id == match_id)
        .where(RosterEntry.tenant_id == tenant.tenant_id)
    )
    if payload.team_side:
        stmt = stmt.where(RosterEntry.team_side == payload.team_side)
    entries = list(session.exec(stmt))
    if not entries:
        raise HTTPException(status_code=400, detail="This match has no roster entries to save")

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Template name is required")
    template = _upsert_template(
        session=session,
        tenant_id=tenant.tenant_id,
        name=name,
        description=payload.description,
        entries=[
            {
                "player_name": entry.player_name,
                "jersey_number": entry.jersey_number,
                "position": entry.position,
                "email": entry.email,
                "team_side": entry.team_side,
            }
            for entry in entries
        ],
        user_id=user.user_id,
    )
    session.commit()
    session.refresh(template)
    return roster_template_to_read(template)


@router.post("/matches/{match_id}/roster/apply-template/{template_id}", response_model=RosterImportResult)
def apply_roster_template(
    match_id: str,
    template_id: str,
    payload: RosterTemplateApply,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> RosterImportResult:
    _ensure_match(session, match_id, tenant.tenant_id)
    template = session.get(RosterTemplate, template_id)
    if not template or template.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Roster template not found: {template_id}")

    result = RosterImportResult()
    for index, raw in enumerate(list(template.entries_json or []), start=1):
        name = str(raw.get("player_name", "") or "").strip()
        jersey = str(raw.get("jersey_number", "") or "").strip()
        side = payload.team_side or str(raw.get("team_side", "home") or "home")
        if not name or not jersey:
            result.errors.append(RosterImportError(line=index, issue="player_name and jersey_number are required"))
            continue

        existing = session.exec(
            select(RosterEntry)
            .where(RosterEntry.match_id == match_id)
            .where(RosterEntry.tenant_id == tenant.tenant_id)
            .where(RosterEntry.team_side == side)
            .where(RosterEntry.jersey_number == jersey)
        ).first()
        if existing and not payload.replace_existing:
            result.skipped += 1
            continue
        if existing:
            existing.player_name = name
            existing.position = raw.get("position") or None
            existing.email = raw.get("email") or None
            existing.updated_at = utcnow()
            session.add(existing)
            result.updated += 1
            continue

        session.add(RosterEntry(
            tenant_id=tenant.tenant_id,
            match_id=match_id,
            player_name=name,
            jersey_number=jersey,
            position=raw.get("position") or None,
            email=raw.get("email") or None,
            team_side=side,
        ))
        result.created += 1

    session.commit()
    rows = list(
        session.exec(
            select(RosterEntry)
            .where(RosterEntry.match_id == match_id)
            .where(RosterEntry.tenant_id == tenant.tenant_id)
        )
    )
    rows.sort(key=lambda item: (item.team_side, len(item.jersey_number), item.jersey_number))
    result.entries = [roster_to_read(row) for row in rows]
    return result


@router.delete("/roster-templates/{template_id}", response_model=Dict[str, object])
def delete_roster_template(
    template_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    template = session.get(RosterTemplate, template_id)
    if not template or template.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Roster template not found: {template_id}")
    session.delete(template)
    session.commit()
    return {"deleted": True, "template_id": template_id}


def _upsert_template(
    session: Session,
    tenant_id: Optional[str],
    name: str,
    description: Optional[str],
    entries: list,
    user_id: Optional[str],
) -> RosterTemplate:
    """Saving under an existing name replaces it — templates are living rosters."""
    template = session.exec(
        select(RosterTemplate)
        .where(RosterTemplate.tenant_id == tenant_id)
        .where(RosterTemplate.name == name)
    ).first()
    if template:
        template.description = description
        template.entries_json = entries
        template.updated_at = utcnow()
    else:
        template = RosterTemplate(
            tenant_id=tenant_id,
            name=name,
            description=description,
            entries_json=entries,
            created_by_user_id=user_id,
        )
    session.add(template)
    return template


@router.post("/matches/{match_id}/events/{event_id}/assign", response_model=EventRead)
def assign_event_to_player(
    match_id: str,
    event_id: str,
    payload: EventAssignRequest,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> EventRead:
    _ensure_match(session, match_id, tenant.tenant_id)
    event = session.get(Event, event_id)
    if not event or event.match_id != match_id or event.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    if payload.roster_entry_id is None:
        event.player_id = None
        event.jersey_number = None
        event.team_id = None
    else:
        entry = session.get(RosterEntry, payload.roster_entry_id)
        if not entry or entry.match_id != match_id or entry.tenant_id != tenant.tenant_id:
            raise HTTPException(status_code=404, detail=f"Roster entry not found: {payload.roster_entry_id}")
        event.player_id = entry.id
        event.jersey_number = entry.jersey_number
        event.team_id = entry.team_side

    event.updated_at = utcnow()
    session.add(event)
    session.commit()
    session.refresh(event)
    return event_to_read(event)
