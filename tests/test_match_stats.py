from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def _create_match(client: TestClient) -> str:
    response = client.post(
        "/v1/matches",
        json={
            "name": "Stats Match",
            "home_team_name": "Lions",
            "away_team_name": "Rovers",
            "source_video_path": "C:/tmp/nonexistent.mp4",
            "metadata": {},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def _put_event(client: TestClient, match_id: str, event_type: str, occurred_ms: int, team_id: str | None) -> str:
    event_id = f"evt_stats_{uuid4().hex[:10]}"
    response = client.put(
        f"/v1/matches/{match_id}/events/{event_id}",
        json={
            "event_type": event_type,
            "status": "auto_detected",
            "confidence": 0.9,
            "occurred_at_ms": occurred_ms,
            "start_ms": max(0, occurred_ms - 500),
            "end_ms": occurred_ms + 500,
            "team_id": team_id,
        },
    )
    assert response.status_code == 200, response.text
    return event_id


def test_stats_unavailable_without_analysis(client: TestClient) -> None:
    match_id = _create_match(client)
    response = client.get(f"/v1/matches/{match_id}/stats")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["match_id"] == match_id
    assert payload["analysis"]["has_completed_job"] is False
    assert len(payload["stats"]) == 15
    for stat in payload["stats"]:
        assert stat["available"] is False
        assert stat["reason"] == "no_completed_analysis"


def test_stats_catalog_counts_and_attribution(client: TestClient) -> None:
    match_id = _create_match(client)

    goal_home = _put_event(client, match_id, "goal", 10_000, "home")
    # Same-team shot 2s after the goal: one attempt after dedup.
    _put_event(client, match_id, "shot", 12_000, "home")
    # Away goal attributed via team name mapping.
    _put_event(client, match_id, "goal", 100_000, "Rovers")
    _put_event(client, match_id, "shot", 30_000, None)
    _put_event(client, match_id, "save", 31_000, "home")
    _put_event(client, match_id, "corner_kick", 40_000, "home")
    _put_event(client, match_id, "foul", 50_000, None)
    _put_event(client, match_id, "penalty_kick", 60_000, "Rovers")
    _put_event(client, match_id, "free_kick", 70_000, "home")

    response = client.get(f"/v1/matches/{match_id}/stats")
    assert response.status_code == 200, response.text
    payload = response.json()
    stats = {stat["key"]: stat for stat in payload["stats"]}

    assert payload["teams"] == {"home": "Lions", "away": "Rovers"}
    assert len(payload["stats"]) == 15

    # Goals: 1 home + 1 away, with drilldown event ids.
    assert stats["goals"]["available"] is True
    assert stats["goals"]["home"] == 1
    assert stats["goals"]["away"] == 1
    assert stats["goals"]["total"] == 2
    assert goal_home in stats["goals"]["event_ids"]

    # Shots: home goal+shot dedup to 1 attempt, away goal 1, unattributed shot 1.
    assert stats["total_shots"]["home"] == 1
    assert stats["total_shots"]["away"] == 1
    assert stats["total_shots"]["unattributed"] == 1
    assert stats["total_shots"]["total"] == 3

    # On target: goals + saves.
    assert stats["shots_on_target"]["available"] is True
    assert stats["shots_on_target"]["total"] >= 2

    assert stats["saves"]["total"] == 1
    assert stats["corners"]["home"] == 1
    assert stats["fouls"]["unattributed"] == 1
    assert stats["penalties"]["away"] == 1
    assert stats["free_kicks"]["home"] == 1

    # Not yet detectable stats are flagged, not zeroed.
    for key in ("assists", "offsides", "total_passes", "pass_accuracy", "key_passes", "duels"):
        assert stats[key]["available"] is False
        assert stats[key]["reason"] == "not_detected_by_pipeline"

    # No completed job -> no team-stats artifact -> possession unavailable.
    assert stats["possession"]["available"] is False
    assert stats["possession"]["reason"] == "team_stats_artifact_missing"
