"""Route detected highlights to rostered players (FR-ROSTER-02) and build
player cards (FR-ROSTER-03).

Routing matches an event's recognized ``jersey_number`` to a roster entry on
the same team side. Jersey-number *recognition* is a separate pipeline
concern: until the CV layer populates ``Event.jersey_number``, routing is
driven by numbers set through manual assignment or reviewer corrections, and
this service reports exactly how many events it could and could not place.
Unrouted highlights stay on the match, shareable and manually assignable
(FR-ROSTER-04).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from sqlmodel import Session, select

from ..models import Event, Match, RosterEntry
from ..schemas import PlayerCardRead, PlayerCardStat, RoutingResult
from ..utils import utcnow

# Event types worth counting on a player card, in display order.
CARD_STATS: List[Tuple[str, str]] = [
    ("goal", "Goals"),
    ("shot", "Shots"),
    ("save", "Saves"),
    ("corner_kick", "Corners"),
    ("free_kick", "Free Kicks"),
    ("penalty_kick", "Penalties"),
    ("foul", "Fouls"),
]


def _normalize_jersey(value: Optional[str]) -> str:
    """'07' and '7' are the same shirt."""
    text = str(value or "").strip().lstrip("0")
    return text or str(value or "").strip()


def _roster_index(entries: List[RosterEntry]) -> Dict[Tuple[str, str], RosterEntry]:
    index: Dict[Tuple[str, str], RosterEntry] = {}
    for entry in entries:
        index[(entry.team_side, _normalize_jersey(entry.jersey_number))] = entry
    return index


def route_match_events(session: Session, match_id: str, tenant_id: Optional[str]) -> RoutingResult:
    """Attach events carrying a jersey number to the matching roster entry."""
    entries = list(
        session.exec(
            select(RosterEntry)
            .where(RosterEntry.match_id == match_id)
            .where(RosterEntry.tenant_id == tenant_id)
        )
    )
    result = RoutingResult(match_id=match_id, roster_size=len(entries))
    events = list(
        session.exec(
            select(Event).where(Event.match_id == match_id).where(Event.tenant_id == tenant_id)
        )
    )
    if not entries:
        result.unassigned_remaining = sum(1 for event in events if not event.player_id)
        return result

    index = _roster_index(entries)
    by_jersey_any_side: Dict[str, List[RosterEntry]] = {}
    for entry in entries:
        by_jersey_any_side.setdefault(_normalize_jersey(entry.jersey_number), []).append(entry)

    unmatched: set[str] = set()
    for event in events:
        jersey = _normalize_jersey(event.jersey_number)
        if not jersey:
            continue
        if event.player_id:
            result.already_routed += 1
            continue

        entry = index.get((str(event.team_id or ""), jersey))
        if entry is None:
            # No team side on the event: accept only an unambiguous number.
            candidates = by_jersey_any_side.get(jersey, [])
            entry = candidates[0] if len(candidates) == 1 else None
        if entry is None:
            unmatched.add(jersey)
            continue

        event.player_id = entry.id
        event.team_id = entry.team_side
        event.updated_at = utcnow()
        session.add(event)
        result.routed += 1

    result.unmatched_jersey_numbers = sorted(unmatched)
    # The in-memory rows already carry this pass's assignments.
    result.unassigned_remaining = sum(1 for event in events if not event.player_id)
    return result


def build_player_card(
    session: Session,
    match: Match,
    entry: RosterEntry,
    share_url_path: Optional[str] = None,
) -> PlayerCardRead:
    events = list(
        session.exec(
            select(Event)
            .where(Event.match_id == match.id)
            .where(Event.tenant_id == entry.tenant_id)
            .where(Event.player_id == entry.id)
        )
    )
    events.sort(key=lambda item: item.occurred_at_ms)

    counts: Dict[str, int] = {}
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1

    stats = [
        PlayerCardStat(key=key, label=label, count=counts.get(key, 0))
        for key, label in CARD_STATS
        if counts.get(key, 0) > 0
    ]

    team_name = match.home_team_name if entry.team_side == "home" else match.away_team_name
    return PlayerCardRead(
        match_id=match.id,
        roster_entry_id=entry.id,
        player_name=entry.player_name,
        jersey_number=entry.jersey_number,
        position=entry.position,
        team_side=entry.team_side,
        team_name=team_name,
        match_name=match.name,
        match_date=match.match_date,
        highlight_count=len(events),
        stats=stats,
        highlights=[
            {
                "event_id": event.id,
                "event_type": event.event_type,
                "occurred_at_ms": event.occurred_at_ms,
                "start_ms": event.start_ms,
                "end_ms": event.end_ms,
                "confidence": round(float(event.confidence or 0.0), 3),
            }
            for event in events
        ],
        share_url_path=share_url_path,
    )
