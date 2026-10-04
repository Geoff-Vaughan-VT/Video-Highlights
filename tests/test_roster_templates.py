from __future__ import annotations

from fastapi.testclient import TestClient


def _create_match(client: TestClient, name: str = "Template Match") -> str:
    response = client.post(
        "/v1/matches",
        json={"name": name, "source_video_path": "C:/tmp/nonexistent.mp4", "metadata": {}},
    )
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def _add_player(client: TestClient, match_id: str, name: str, jersey: str, **extra) -> str:
    body = {"player_name": name, "jersey_number": jersey}
    body.update(extra)
    response = client.post(f"/v1/matches/{match_id}/roster", json=body)
    assert response.status_code == 201, response.text
    return response.json()["roster_entry_id"]


def test_save_roster_as_template_and_apply_to_another_match(client: TestClient) -> None:
    source_match = _create_match(client, "Week 1")
    _add_player(client, source_match, "Alex Morgan", "13", position="Forward", email="alex@example.com")
    _add_player(client, source_match, "Sam Kerr", "20", position="Forward")

    saved = client.post(
        f"/v1/matches/{source_match}/roster/save-template",
        json={"name": "U12 Lions", "description": "Season roster"},
    )
    assert saved.status_code == 201, saved.text
    template = saved.json()
    assert template["entry_count"] == 2
    assert template["name"] == "U12 Lions"

    listing = client.get("/v1/roster-templates").json()["items"]
    assert [item["name"] for item in listing] == ["U12 Lions"]

    # Apply to a brand-new match — no re-entry needed.
    next_match = _create_match(client, "Week 2")
    applied = client.post(
        f"/v1/matches/{next_match}/roster/apply-template/{template['template_id']}",
        json={},
    )
    assert applied.status_code == 200, applied.text
    result = applied.json()
    assert result["created"] == 2
    assert result["updated"] == 0
    names = {entry["jersey_number"]: entry["player_name"] for entry in result["entries"]}
    assert names == {"13": "Alex Morgan", "20": "Sam Kerr"}
    emails = {entry["jersey_number"]: entry["email"] for entry in result["entries"]}
    assert emails["13"] == "alex@example.com"


def test_apply_template_skips_or_replaces_existing(client: TestClient) -> None:
    source_match = _create_match(client, "Source")
    _add_player(client, source_match, "Alex Morgan", "13")
    template = client.post(
        f"/v1/matches/{source_match}/roster/save-template",
        json={"name": "Squad"},
    ).json()

    target = _create_match(client, "Target")
    _add_player(client, target, "Someone Else", "13")

    skipped = client.post(
        f"/v1/matches/{target}/roster/apply-template/{template['template_id']}",
        json={},
    ).json()
    assert skipped["created"] == 0
    assert skipped["skipped"] == 1
    assert skipped["entries"][0]["player_name"] == "Someone Else"

    replaced = client.post(
        f"/v1/matches/{target}/roster/apply-template/{template['template_id']}",
        json={"replace_existing": True},
    ).json()
    assert replaced["updated"] == 1
    assert replaced["entries"][0]["player_name"] == "Alex Morgan"


def test_template_apply_can_override_team_side(client: TestClient) -> None:
    source_match = _create_match(client, "Source")
    _add_player(client, source_match, "Alex Morgan", "13", team_side="home")
    template = client.post(
        f"/v1/matches/{source_match}/roster/save-template", json={"name": "Squad"}
    ).json()

    target = _create_match(client, "Target")
    applied = client.post(
        f"/v1/matches/{target}/roster/apply-template/{template['template_id']}",
        json={"team_side": "away"},
    ).json()
    assert applied["entries"][0]["team_side"] == "away"


def test_saving_same_name_replaces_template(client: TestClient) -> None:
    match_id = _create_match(client)
    _add_player(client, match_id, "One Player", "1")
    first = client.post(f"/v1/matches/{match_id}/roster/save-template", json={"name": "Squad"}).json()
    assert first["entry_count"] == 1

    _add_player(client, match_id, "Two Player", "2")
    second = client.post(f"/v1/matches/{match_id}/roster/save-template", json={"name": "Squad"}).json()
    assert second["template_id"] == first["template_id"]
    assert second["entry_count"] == 2
    assert len(client.get("/v1/roster-templates").json()["items"]) == 1


def test_create_and_delete_template_directly(client: TestClient) -> None:
    created = client.post(
        "/v1/roster-templates",
        json={
            "name": "Manual Squad",
            "entries": [{"player_name": "Keeper", "jersey_number": "1", "position": "GK"}],
        },
    )
    assert created.status_code == 201, created.text
    template_id = created.json()["template_id"]

    deleted = client.delete(f"/v1/roster-templates/{template_id}")
    assert deleted.status_code == 200
    assert client.get("/v1/roster-templates").json()["items"] == []


def test_saving_template_from_empty_roster_is_rejected(client: TestClient) -> None:
    match_id = _create_match(client)
    response = client.post(f"/v1/matches/{match_id}/roster/save-template", json={"name": "Empty"})
    assert response.status_code == 400
    assert "no roster" in response.json()["error"]["message"].lower()
