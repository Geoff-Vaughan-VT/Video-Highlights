"""Share-link creation and public payload building (FR-SHARE-01/03).

A share link is an unguessable token that grants read-only access to one
match, one highlight, or one player card — no account required. Public
payloads are built explicitly field by field: never spread a model, so
filesystem paths, tenant internals, and reviewer metadata cannot leak.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..models import Event, Match, RosterEntry, ShareLink
from ..schemas import ShareLinkRead
from ..utils import utcnow
from .player_routing import build_player_card
from .source_catalog import get_source_type
from .stat_catalog import compute_match_stat_catalog

TOKEN_BYTES = 24


def share_url_path(token: str) -> str:
    """Client-facing path for a share token (hash route in the web app)."""
    return f"/#share/{token}"


def create_share_link(
    session: Session,
    match: Match,
    scope: str = "match",
    event_id: Optional[str] = None,
    roster_entry_id: Optional[str] = None,
    label: Optional[str] = None,
    expires_in_days: Optional[int] = None,
    created_by_user_id: Optional[str] = None,
    reuse_existing: bool = True,
) -> ShareLink:
    if reuse_existing:
        existing = session.exec(
            select(ShareLink)
            .where(ShareLink.match_id == match.id)
            .where(ShareLink.tenant_id == match.tenant_id)
            .where(ShareLink.scope == scope)
            .where(ShareLink.event_id == event_id)
            .where(ShareLink.roster_entry_id == roster_entry_id)
            .where(ShareLink.revoked == False)  # noqa: E712 - SQL boolean comparison
        ).first()
        if existing and not _is_expired(existing):
            return existing

    expires_at = None
    if expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

    link = ShareLink(
        token=secrets.token_urlsafe(TOKEN_BYTES),
        tenant_id=match.tenant_id,
        match_id=match.id,
        scope=scope,
        event_id=event_id,
        roster_entry_id=roster_entry_id,
        label=label,
        expires_at=expires_at,
        created_by_user_id=created_by_user_id,
    )
    session.add(link)
    return link


def _is_expired(link: ShareLink) -> bool:
    if not link.expires_at:
        return False
    expires_at = link.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at < datetime.now(timezone.utc)


def is_usable(link: Optional[ShareLink]) -> bool:
    return bool(link) and not link.revoked and not _is_expired(link)


def share_to_read(link: ShareLink) -> ShareLinkRead:
    return ShareLinkRead(
        share_id=link.id,
        token=link.token,
        url_path=share_url_path(link.token),
        tenant_id=link.tenant_id,
        match_id=link.match_id,
        scope=link.scope,
        event_id=link.event_id,
        roster_entry_id=link.roster_entry_id,
        label=link.label,
        revoked=link.revoked,
        expires_at=link.expires_at,
        view_count=link.view_count,
        created_at=link.created_at,
    )


def _public_event(event: Event, player_by_id: Dict[str, RosterEntry]) -> Dict[str, Any]:
    player = player_by_id.get(event.player_id or "")
    return {
        "event_id": event.id,
        "event_type": event.event_type,
        "occurred_at_ms": event.occurred_at_ms,
        "start_ms": event.start_ms,
        "end_ms": event.end_ms,
        "confidence": round(float(event.confidence or 0.0), 3),
        "team": event.team_id,
        "player_name": player.player_name if player else None,
        "jersey_number": player.jersey_number if player else event.jersey_number,
    }


def build_public_payload(session: Session, link: ShareLink) -> Dict[str, Any]:
    """Assemble the read-only payload for a share token."""
    match = session.get(Match, link.match_id)
    if not match:
        raise LookupError("Match not found for this share link")

    roster = list(
        session.exec(
            select(RosterEntry)
            .where(RosterEntry.match_id == match.id)
            .where(RosterEntry.tenant_id == match.tenant_id)
        )
    )
    player_by_id = {entry.id: entry for entry in roster}
    source = get_source_type(str((match.metadata_json or {}).get("source_type") or "") or None)

    payload: Dict[str, Any] = {
        "scope": link.scope,
        "label": link.label,
        "match": {
            "name": match.name,
            "home_team_name": match.home_team_name,
            "away_team_name": match.away_team_name,
            "match_date": match.match_date,
            "source_label": source.label,
        },
        "shared_at": utcnow().isoformat(),
    }

    if link.scope == "highlight":
        event = session.get(Event, link.event_id) if link.event_id else None
        if not event or event.match_id != match.id:
            raise LookupError("Highlight not found for this share link")
        payload["highlight"] = _public_event(event, player_by_id)
        return payload

    if link.scope == "player_card":
        entry = session.get(RosterEntry, link.roster_entry_id) if link.roster_entry_id else None
        if not entry or entry.match_id != match.id:
            raise LookupError("Player card not found for this share link")
        card = build_player_card(session, match, entry, share_url_path(link.token))
        payload["player_card"] = card.model_dump()
        return payload

    stats = compute_match_stat_catalog(session, match, match.tenant_id)
    events = list(
        session.exec(
            select(Event).where(Event.match_id == match.id).where(Event.tenant_id == match.tenant_id)
        )
    )
    events.sort(key=lambda item: item.occurred_at_ms)
    payload["stats"] = [stat.model_dump() for stat in stats.stats]
    payload["analysis"] = stats.analysis
    payload["highlights"] = [_public_event(event, player_by_id) for event in events]
    payload["roster"] = [
        {
            "player_name": entry.player_name,
            "jersey_number": entry.jersey_number,
            "position": entry.position,
            "team_side": entry.team_side,
        }
        for entry in roster
    ]
    return payload


def record_view(session: Session, link: ShareLink) -> None:
    link.view_count = int(link.view_count or 0) + 1
    link.last_viewed_at = utcnow()
    session.add(link)


def get_usable_link(session: Session, token: str) -> Optional[ShareLink]:
    link = session.exec(select(ShareLink).where(ShareLink.token == token)).first()
    return link if is_usable(link) else None


def list_match_shares(session: Session, match_id: str, tenant_id: Optional[str]) -> List[ShareLink]:
    rows = list(
        session.exec(
            select(ShareLink)
            .where(ShareLink.match_id == match_id)
            .where(ShareLink.tenant_id == tenant_id)
            .order_by(ShareLink.created_at.desc())
        )
    )
    return rows
