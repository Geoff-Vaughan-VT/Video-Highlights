from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def _create_match(client: TestClient, **overrides) -> str:
    payload = {
        "name": "Share Match",
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
    event_id = f"evt_share_{uuid4().hex[:10]}"
    response = client.put(
        f"/v1/matches/{match_id}/events/{event_id}",
        json={
            "event_type": event_type,
            "occurred_at_ms": occurred_ms,
            "start_ms": occurred_ms - 500,
            "end_ms": occurred_ms + 500,
            "team_id": "home",
        },
    )
    assert response.status_code == 200, response.text
    return event_id


def test_match_share_link_is_publicly_viewable(client: TestClient) -> None:
    match_id = _create_match(client)
    event_id = _put_event(client, match_id)

    created = client.post(f"/v1/matches/{match_id}/shares", json={"scope": "match"})
    assert created.status_code == 201, created.text
    share = created.json()
    assert share["token"]
    assert share["url_path"].endswith(share["token"])

    # Public read needs no auth headers and no tenant context.
    public = client.get(f"/v1/public/shares/{share['token']}")
    assert public.status_code == 200, public.text
    payload = public.json()
    assert payload["scope"] == "match"
    assert payload["match"]["name"] == "Share Match"
    assert payload["match"]["home_team_name"] == "Lions"
    assert len(payload["stats"]) == 15
    assert [item["event_id"] for item in payload["highlights"]] == [event_id]

    # No internal identifiers or filesystem paths leak into the public view.
    body = public.text
    assert match_id not in body
    assert "source_video_path" not in body
    assert "tenant" not in body.lower()


def test_share_link_is_reused_and_view_counted(client: TestClient) -> None:
    match_id = _create_match(client)
    first = client.post(f"/v1/matches/{match_id}/shares", json={"scope": "match"}).json()
    second = client.post(f"/v1/matches/{match_id}/shares", json={"scope": "match"}).json()
    assert first["token"] == second["token"]

    client.get(f"/v1/public/shares/{first['token']}")
    client.get(f"/v1/public/shares/{first['token']}")
    listing = client.get(f"/v1/matches/{match_id}/shares").json()["items"]
    assert len(listing) == 1
    assert listing[0]["view_count"] == 2


def test_highlight_share_scope(client: TestClient) -> None:
    match_id = _create_match(client)
    event_id = _put_event(client, match_id, event_type="shot", occurred_ms=42_000)

    created = client.post(
        f"/v1/matches/{match_id}/shares",
        json={"scope": "highlight", "event_id": event_id, "label": "Great strike"},
    )
    assert created.status_code == 201, created.text
    payload = client.get(f"/v1/public/shares/{created.json()['token']}").json()
    assert payload["scope"] == "highlight"
    assert payload["label"] == "Great strike"
    assert payload["highlight"]["event_id"] == event_id
    assert payload["highlight"]["occurred_at_ms"] == 42_000
    assert "highlights" not in payload


def test_highlight_share_requires_valid_event(client: TestClient) -> None:
    match_id = _create_match(client)
    missing = client.post(f"/v1/matches/{match_id}/shares", json={"scope": "highlight"})
    assert missing.status_code == 400

    wrong = client.post(
        f"/v1/matches/{match_id}/shares",
        json={"scope": "highlight", "event_id": "evt_does_not_exist"},
    )
    assert wrong.status_code == 404


def test_revoked_and_unknown_tokens_are_rejected(client: TestClient) -> None:
    match_id = _create_match(client)
    share = client.post(f"/v1/matches/{match_id}/shares", json={"scope": "match"}).json()

    assert client.get("/v1/public/shares/not-a-real-token").status_code == 404

    revoked = client.delete(f"/v1/shares/{share['share_id']}")
    assert revoked.status_code == 200
    assert client.get(f"/v1/public/shares/{share['token']}").status_code == 404


def test_share_links_are_tenant_isolated(client: TestClient) -> None:
    # The default client works in the "default" tenant; act as a global admin
    # scoped to the other seeded tenant and confirm the link is invisible there.
    match_id = _create_match(client)
    share = client.post(f"/v1/matches/{match_id}/shares", json={"scope": "match"}).json()

    other_tenant = {"x-user-role": "system", "x-user-id": "sys_user", "X-Tenant-Id": "sandbox"}
    assert client.delete(f"/v1/shares/{share['share_id']}", headers=other_tenant).status_code == 404
    assert client.get(f"/v1/matches/{match_id}/shares", headers=other_tenant).status_code == 404

    # The link still works for its own tenant and for the public.
    assert client.get(f"/v1/public/shares/{share['token']}").status_code == 200


def test_parent_role_cannot_create_share(auth_client: TestClient) -> None:
    coach_headers = {"Authorization": "Bearer coach-token"}
    match_id = auth_client.post(
        "/v1/matches",
        headers=coach_headers,
        json={"name": "RBAC Share", "source_video_path": "", "metadata": {}},
    ).json()["match_id"]

    response = auth_client.post(
        f"/v1/matches/{match_id}/shares",
        headers={"Authorization": "Bearer parent-token"},
        json={"scope": "match"},
    )
    assert response.status_code == 403
