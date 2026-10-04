from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def _create_match(client: TestClient, **overrides) -> str:
    payload = {
        "name": "Roster Match",
        "home_team_name": "Lions",
        "away_team_name": "Rovers",
        "source_video_path": "C:/tmp/nonexistent.mp4",
        "metadata": {},
    }
    payload.update(overrides)
    response = client.post("/v1/matches", json=payload)
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def _put_event(client: TestClient, match_id: str, event_type: str = "goal", occurred_ms: int = 10_000) -> str:
    event_id = f"evt_roster_{uuid4().hex[:10]}"
    response = client.put(
        f"/v1/matches/{match_id}/events/{event_id}",
        json={
            "event_type": event_type,
            "occurred_at_ms": occurred_ms,
            "start_ms": occurred_ms - 500,
            "end_ms": occurred_ms + 500,
        },
    )
    assert response.status_code == 200, response.text
    return event_id


def test_roster_template_download(client: TestClient) -> None:
    response = client.get("/v1/matches/roster-template.csv")
    assert response.status_code == 200, response.text
    assert "player_name,jersey_number,position,email" in response.text
    assert "attachment" in response.headers.get("content-disposition", "")


def test_roster_import_with_aliases_and_errors(client: TestClient) -> None:
    match_id = _create_match(client)
    csv_text = (
        "Name,Jersey,Position,Email\n"
        "Alex Morgan,13,Forward,alex@example.com\n"
        "Sam Kerr,20,Forward,sam@example.com\n"
        "No Jersey,,Midfield,missing@example.com\n"
        "Dup Jersey,13,Defense,dup@example.com\n"
        ",,,\n"
    )
    response = client.post(
        f"/v1/matches/{match_id}/roster/import",
        json={"csv_text": csv_text, "team_side": "home"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["created"] == 2
    assert result["updated"] == 0
    assert result["skipped"] == 1
    issues = [error["issue"] for error in result["errors"]]
    assert len(issues) == 2
    assert any("required" in issue for issue in issues)
    assert any("Duplicate" in issue for issue in issues)
    assert len(result["entries"]) == 2

    # Re-import updates in place instead of duplicating.
    again = client.post(
        f"/v1/matches/{match_id}/roster/import",
        json={"csv_text": "player_name,jersey_number\nAlexandra Morgan,13\n", "team_side": "home"},
    )
    assert again.status_code == 200
    assert again.json()["updated"] == 1
    assert again.json()["created"] == 0

    listing = client.get(f"/v1/matches/{match_id}/roster")
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert len(items) == 2
    names = {item["jersey_number"]: item["player_name"] for item in items}
    assert names["13"] == "Alexandra Morgan"


def test_roster_entry_conflict_and_patch(client: TestClient) -> None:
    match_id = _create_match(client)
    created = client.post(
        f"/v1/matches/{match_id}/roster",
        json={"player_name": "Keeper One", "jersey_number": "1", "position": "GK"},
    )
    assert created.status_code == 201, created.text
    entry_id = created.json()["roster_entry_id"]

    conflict = client.post(
        f"/v1/matches/{match_id}/roster",
        json={"player_name": "Other Keeper", "jersey_number": "1"},
    )
    assert conflict.status_code == 409

    # Same jersey on the other team is fine.
    away_ok = client.post(
        f"/v1/matches/{match_id}/roster",
        json={"player_name": "Away Keeper", "jersey_number": "1", "team_side": "away"},
    )
    assert away_ok.status_code == 201, away_ok.text

    patched = client.patch(
        f"/v1/matches/{match_id}/roster/{entry_id}",
        json={"email": "keeper@example.com"},
    )
    assert patched.status_code == 200
    assert patched.json()["email"] == "keeper@example.com"


def test_event_assignment_and_unassigned_filter(client: TestClient) -> None:
    match_id = _create_match(client)
    event_id = _put_event(client, match_id)
    other_event_id = _put_event(client, match_id, occurred_ms=50_000)

    created = client.post(
        f"/v1/matches/{match_id}/roster",
        json={"player_name": "Striker", "jersey_number": "9", "team_side": "home"},
    )
    assert created.status_code == 201
    entry_id = created.json()["roster_entry_id"]

    assigned = client.post(
        f"/v1/matches/{match_id}/events/{event_id}/assign",
        json={"roster_entry_id": entry_id},
    )
    assert assigned.status_code == 200, assigned.text
    body = assigned.json()
    assert body["player_id"] == entry_id
    assert body["jersey_number"] == "9"
    assert body["team_id"] == "home"

    unassigned = client.get(f"/v1/matches/{match_id}/events?assigned=false")
    ids = [item["event_id"] for item in unassigned.json()["items"]]
    assert other_event_id in ids
    assert event_id not in ids

    assigned_list = client.get(f"/v1/matches/{match_id}/events?assigned=true")
    ids = [item["event_id"] for item in assigned_list.json()["items"]]
    assert ids == [event_id]

    # Clearing the assignment puts the event back in the unassigned pool.
    cleared = client.post(
        f"/v1/matches/{match_id}/events/{event_id}/assign",
        json={"roster_entry_id": None},
    )
    assert cleared.status_code == 200
    assert cleared.json()["player_id"] is None

    # Deleting a roster entry unassigns its events.
    reassigned = client.post(
        f"/v1/matches/{match_id}/events/{event_id}/assign",
        json={"roster_entry_id": entry_id},
    )
    assert reassigned.status_code == 200
    deleted = client.delete(f"/v1/matches/{match_id}/roster/{entry_id}")
    assert deleted.status_code == 200
    assert deleted.json()["unassigned_events"] == 1
    event = client.get(f"/v1/matches/{match_id}/events/{event_id}")
    assert event.json()["player_id"] is None


def test_parent_role_cannot_import_roster(auth_client: TestClient) -> None:
    coach_headers = {"Authorization": "Bearer coach-token"}
    create_match = auth_client.post(
        "/v1/matches",
        headers=coach_headers,
        json={"name": "RBAC Roster", "source_video_path": "", "metadata": {}},
    )
    assert create_match.status_code == 201
    match_id = create_match.json()["match_id"]

    response = auth_client.post(
        f"/v1/matches/{match_id}/roster/import",
        headers={"Authorization": "Bearer parent-token"},
        json={"csv_text": "player_name,jersey_number\nKid,10\n"},
    )
    assert response.status_code == 403
