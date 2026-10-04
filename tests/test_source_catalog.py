from __future__ import annotations

from fastapi.testclient import TestClient

from backend.services.source_catalog import ALL_STAT_KEYS, detect_source_from_url, get_source_type


def test_detects_known_providers() -> None:
    assert detect_source_from_url("https://www.youtube.com/watch?v=abc123").key == "youtube"
    assert detect_source_from_url("https://youtu.be/abc123").key == "youtube"
    assert detect_source_from_url("https://vimeo.com/12345").key == "vimeo"
    assert detect_source_from_url("https://app.veo.co/matches/xyz").key == "veo"
    assert detect_source_from_url("https://www.hudl.com/video/abc").key == "hudl"
    assert detect_source_from_url("https://someone-elses-site.example/video").key == "other_link"


def test_non_urls_are_not_links() -> None:
    assert detect_source_from_url("C:/videos/match.mp4") is None
    assert detect_source_from_url("/data/match.mp4") is None
    assert detect_source_from_url("") is None


def test_raw_upload_supports_every_stat() -> None:
    raw = get_source_type("raw_upload")
    assert raw.supported_stats == ALL_STAT_KEYS
    assert raw.premium_stats_supported is True

    youtube = get_source_type("youtube")
    assert youtube.premium_stats_supported is False
    assert "goals" in youtube.supported_stats
    assert "total_passes" in youtube.unsupported_stats
    assert youtube.supports("goals") is True
    assert youtube.supports("total_passes") is False


def test_unknown_source_key_falls_back_to_raw() -> None:
    assert get_source_type("something-new").key == "raw_upload"
    assert get_source_type(None).key == "raw_upload"


def test_sources_endpoint_lists_matrix_and_classifies(client: TestClient) -> None:
    response = client.get("/v1/sources")
    assert response.status_code == 200, response.text
    sources = response.json()["sources"]
    keys = [item["key"] for item in sources]
    for expected in ("raw_upload", "youtube", "vimeo", "veo", "hudl", "pixellot", "xbotgo", "nbc_sports_engine"):
        assert expected in keys
    raw = next(item for item in sources if item["key"] == "raw_upload")
    assert raw["supported_stat_count"] == raw["total_stat_count"] == 15

    classified = client.get("/v1/sources?url=https://youtu.be/abc").json()
    assert classified["detected"]["key"] == "youtube"
    assert classified["detected"]["supported_stat_count"] < 15


def test_match_creation_classifies_pasted_link(client: TestClient) -> None:
    link_match = client.post(
        "/v1/matches",
        json={"name": "YouTube Match", "source_video_path": "https://www.youtube.com/watch?v=abc", "metadata": {}},
    ).json()
    assert link_match["metadata"]["source_type"] == "youtube"
    assert link_match["metadata"]["source_url"] == "https://www.youtube.com/watch?v=abc"

    file_match = client.post(
        "/v1/matches",
        json={"name": "File Match", "source_video_path": "C:/videos/match.mp4", "metadata": {}},
    ).json()
    assert file_match["metadata"]["source_type"] == "raw_upload"


def test_stat_catalog_marks_source_limited_stats_unavailable(client: TestClient) -> None:
    match_id = client.post(
        "/v1/matches",
        json={
            "name": "YouTube Match",
            "home_team_name": "Lions",
            "away_team_name": "Rovers",
            "source_video_path": "https://www.youtube.com/watch?v=abc",
            "metadata": {},
        },
    ).json()["match_id"]

    client.put(
        f"/v1/matches/{match_id}/events/evt_src_1",
        json={"event_type": "goal", "occurred_at_ms": 1000, "start_ms": 500, "end_ms": 1500, "team_id": "home"},
    )

    payload = client.get(f"/v1/matches/{match_id}/stats").json()
    stats = {item["key"]: item for item in payload["stats"]}
    assert payload["analysis"]["source_type"] == "youtube"

    # Event-derived stats still work from a link.
    assert stats["goals"]["available"] is True
    assert stats["goals"]["home"] == 1

    # Aggregate stats the provider cannot support are flagged, not zeroed.
    for key in ("total_passes", "pass_accuracy", "key_passes", "duels", "possession"):
        assert stats[key]["available"] is False
        assert stats[key]["reason"] == "not_available_for_source"
        assert stats[key]["raw"]["source_label"] == "YouTube"
