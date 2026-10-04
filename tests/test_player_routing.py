from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def _create_match(client: TestClient) -> str:
    response = client.post(
        "/v1/matches",
        json={
            "name": "Routing Match",
            "home_team_name": "Lions",
            "away_team_name": "Rovers",
            "source_video_path": "C:/tmp/nonexistent.mp4",
            "metadata": {},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def _put_event(
    client: TestClient,
    match_id: str,
    event_type: str = "goal",
    occurred_ms: int = 10_000,
    jersey: str | None = None,
    team_id: str | None = None,
) -> str:
    event_id = f"evt_route_{uuid4().hex[:10]}"
    body = {
        "event_type": event_type,
        "occurred_at_ms": occurred_ms,
        "start_ms": occurred_ms - 500,
        "end_ms": occurred_ms + 500,
    }
    if jersey:
        body["jersey_number"] = jersey
    if team_id:
        body["team_id"] = team_id
    response = client.put(f"/v1/matches/{match_id}/events/{event_id}", json=body)
    assert response.status_code == 200, response.text
    return event_id


def _add_player(client: TestClient, match_id: str, name: str, jersey: str, **extra) -> str:
    body = {"player_name": name, "jersey_number": jersey}
    body.update(extra)
    response = client.post(f"/v1/matches/{match_id}/roster", json=body)
    assert response.status_code == 201, response.text
    return response.json()["roster_entry_id"]


def test_routing_matches_jersey_numbers(client: TestClient) -> None:
    match_id = _create_match(client)
    entry_id = _add_player(client, match_id, "Alex Morgan", "13", team_side="home")
    _add_player(client, match_id, "Away Keeper", "1", team_side="away")

    goal = _put_event(client, match_id, "goal", 10_000, jersey="13", team_id="home")
    # Leading zeros are the same shirt.
    shot = _put_event(client, match_id, "shot", 20_000, jersey="013", team_id="home")
    # Unknown number stays unrouted.
    _put_event(client, match_id, "shot", 30_000, jersey="77", team_id="home")
    # No jersey at all stays unrouted.
    _put_event(client, match_id, "corner_kick", 40_000)

    result = client.post(f"/v1/matches/{match_id}/roster/route")
    assert result.status_code == 200, result.text
    payload = result.json()
    assert payload["routed"] == 2
    assert payload["roster_size"] == 2
    assert payload["unmatched_jersey_numbers"] == ["77"]
    assert payload["unassigned_remaining"] == 2

    events = {item["event_id"]: item for item in client.get(f"/v1/matches/{match_id}/events").json()["items"]}
    assert events[goal]["player_id"] == entry_id
    assert events[shot]["player_id"] == entry_id

    # Re-running is idempotent: already-routed events are not re-counted.
    again = client.post(f"/v1/matches/{match_id}/roster/route").json()
    assert again["routed"] == 0
    assert again["already_routed"] == 2


def test_routing_requires_unambiguous_number_without_team(client: TestClient) -> None:
    match_id = _create_match(client)
    _add_player(client, match_id, "Home Ten", "10", team_side="home")
    _add_player(client, match_id, "Away Ten", "10", team_side="away")
    _put_event(client, match_id, "goal", 10_000, jersey="10")  # no team side

    result = client.post(f"/v1/matches/{match_id}/roster/route").json()
    assert result["routed"] == 0
    assert result["unmatched_jersey_numbers"] == ["10"]


def test_routing_without_roster_is_a_noop(client: TestClient) -> None:
    match_id = _create_match(client)
    _put_event(client, match_id, "goal", 10_000, jersey="9")
    result = client.post(f"/v1/matches/{match_id}/roster/route").json()
    assert result["routed"] == 0
    assert result["roster_size"] == 0
    assert result["unassigned_remaining"] == 1


def test_player_card_contents(client: TestClient) -> None:
    match_id = _create_match(client)
    entry_id = _add_player(client, match_id, "Alex Morgan", "13", position="Forward", team_side="home")
    _put_event(client, match_id, "goal", 10_000, jersey="13", team_id="home")
    _put_event(client, match_id, "goal", 20_000, jersey="13", team_id="home")
    _put_event(client, match_id, "shot", 30_000, jersey="13", team_id="home")
    client.post(f"/v1/matches/{match_id}/roster/route")

    card = client.get(f"/v1/matches/{match_id}/roster/{entry_id}/card")
    assert card.status_code == 200, card.text
    payload = card.json()
    assert payload["player_name"] == "Alex Morgan"
    assert payload["jersey_number"] == "13"
    assert payload["team_name"] == "Lions"
    assert payload["highlight_count"] == 3
    counts = {stat["key"]: stat["count"] for stat in payload["stats"]}
    assert counts == {"goal": 2, "shot": 1}
    assert [item["occurred_at_ms"] for item in payload["highlights"]] == [10_000, 20_000, 30_000]


def test_send_player_cards_emails_and_shares(client: TestClient) -> None:
    match_id = _create_match(client)
    entry_id = _add_player(client, match_id, "Alex Morgan", "13", email="alex@example.com", team_side="home")
    _add_player(client, match_id, "No Email", "7", team_side="home")
    _put_event(client, match_id, "goal", 10_000, jersey="13", team_id="home")
    client.post(f"/v1/matches/{match_id}/roster/route")

    response = client.post(f"/v1/matches/{match_id}/roster/cards/send")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["sent"] == 1
    assert payload["skipped"] == 1

    sent = next(item for item in payload["details"] if item["status"] == "sent")
    assert sent["roster_entry_id"] == entry_id
    assert sent["highlight_count"] == 1
    token = sent["share_url_path"].rsplit("/", 1)[-1]

    # The emailed link is a working public player card.
    public = client.get(f"/v1/public/shares/{token}")
    assert public.status_code == 200, public.text
    card = public.json()["player_card"]
    assert card["player_name"] == "Alex Morgan"
    assert card["highlight_count"] == 1

    skipped = next(item for item in payload["details"] if item["status"] == "skipped")
    assert "no email" in skipped["reason"].lower()
