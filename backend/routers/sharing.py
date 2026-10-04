"""Share links and public viewing (FR-SHARE-01/03), plus the ingest source
capability matrix (FR-SOURCE-02/03).

Everything under ``/v1/public/*`` is deliberately unauthenticated and
tenant-header free: a share link is meant to work for anyone the customer
sends it to. Those handlers take no user/tenant dependency and return only
the fields assembled in ``services.sharing``.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..database import get_session
from ..models import Event, Match, RosterEntry, ShareLink
from ..schemas import ShareLinkCreate, ShareLinkRead
from ..services.sharing import (
    build_public_payload,
    create_share_link,
    get_usable_link,
    list_match_shares,
    record_view,
    share_to_read,
)
from ..services.source_catalog import detect_source_from_url, list_source_types, source_to_dict
from ..tenant import TenantContext, get_tenant_context

router = APIRouter(tags=["sharing"])

READ_ROLES = ("admin", "analyst", "coach", "parent", "system", "tenant_admin")
WRITE_ROLES = ("admin", "analyst", "coach", "tenant_admin")


def _ensure_match(session: Session, match_id: str, tenant_id: str) -> Match:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    return match


@router.get("/sources", response_model=Dict[str, object])
def list_sources(
    url: str | None = Query(default=None, description="Optional URL to classify"),
) -> Dict[str, object]:
    """Ingest sources and the statistics each can support."""
    payload: Dict[str, object] = {"sources": list_source_types()}
    if url:
        detected = detect_source_from_url(url)
        payload["detected"] = source_to_dict(detected) if detected else None
    return payload


@router.post("/matches/{match_id}/shares", response_model=ShareLinkRead, status_code=201)
def create_match_share(
    match_id: str,
    payload: ShareLinkCreate,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> ShareLinkRead:
    match = _ensure_match(session, match_id, tenant.tenant_id)

    if payload.scope == "highlight":
        if not payload.event_id:
            raise HTTPException(status_code=400, detail="event_id is required for a highlight share")
        event = session.get(Event, payload.event_id)
        if not event or event.match_id != match_id or event.tenant_id != tenant.tenant_id:
            raise HTTPException(status_code=404, detail=f"Event not found: {payload.event_id}")
    elif payload.scope == "player_card":
        if not payload.roster_entry_id:
            raise HTTPException(status_code=400, detail="roster_entry_id is required for a player card share")
        entry = session.get(RosterEntry, payload.roster_entry_id)
        if not entry or entry.match_id != match_id or entry.tenant_id != tenant.tenant_id:
            raise HTTPException(status_code=404, detail=f"Roster entry not found: {payload.roster_entry_id}")

    link = create_share_link(
        session=session,
        match=match,
        scope=payload.scope,
        event_id=payload.event_id,
        roster_entry_id=payload.roster_entry_id,
        label=payload.label,
        expires_in_days=payload.expires_in_days,
        created_by_user_id=user.user_id,
    )
    session.commit()
    session.refresh(link)
    return share_to_read(link)


@router.get("/matches/{match_id}/shares", response_model=Dict[str, object])
def list_shares(
    match_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*READ_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    _ensure_match(session, match_id, tenant.tenant_id)
    rows = list_match_shares(session, match_id, tenant.tenant_id)
    return {"items": [share_to_read(row).model_dump() for row in rows]}


@router.delete("/shares/{share_id}", response_model=Dict[str, object])
def revoke_share(
    share_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles(*WRITE_ROLES)),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    link = session.get(ShareLink, share_id)
    if not link or link.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Share link not found: {share_id}")
    link.revoked = True
    session.add(link)
    session.commit()
    return {"revoked": True, "share_id": share_id}


# --------------------------------------------------------------------------
# Public (no auth, no tenant header) — anyone holding the token can read.
# --------------------------------------------------------------------------


@router.get("/public/shares/{token}", response_model=Dict[str, Any])
def view_shared(token: str, session: Session = Depends(get_session)) -> Dict[str, Any]:
    link = get_usable_link(session, token)
    if not link:
        raise HTTPException(status_code=404, detail="This share link is invalid, expired, or revoked")
    try:
        payload = build_public_payload(session, link)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    record_view(session, link)
    session.commit()
    return payload
