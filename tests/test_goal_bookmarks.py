from __future__ import annotations

from VideoHighlights import build_analysis_bookmarks


def test_goal_event_upgrades_overlapping_bookmark() -> None:
    bookmarks = build_analysis_bookmarks(
        original_intervals=[(10.0, 20.0), (40.0, 48.0)],
        speed_intervals=[(11.0, 13.0)],
        audio_intervals=[(14.0, 15.0)],
        requested_targets=[],
        goal_events=[
            {"t": 15.5, "side": "left", "confidence": 0.9, "reason": "ball observed inside left goal"}
        ],
        game_states=[
            {"start_s": 0.0, "end_s": 15.0, "state": "in_play"},
            {"start_s": 15.0, "end_s": 25.0, "state": "goal_left"},
            {"start_s": 25.0, "end_s": 60.0, "state": "in_play"},
        ],
    )

    assert len(bookmarks) == 2
    goal_bm = bookmarks[0]
    assert goal_bm["event_type"] == "goal"
    assert goal_bm["label"] == "goal_detected"
    assert goal_bm["confidence"] >= 0.9
    assert "ball_tracking" in goal_bm["sources"]
    assert goal_bm["occurred_at_s"] == 15.5
    assert goal_bm["signals"]["goal_side"] == "left"
    assert goal_bm["game_state"] == "goal_left"

    other = bookmarks[1]
    assert other["event_type"] != "goal" or other["label"] != "goal_detected"
    assert other["game_state"] == "in_play"


def test_bookmarks_without_goal_events_unchanged() -> None:
    bookmarks = build_analysis_bookmarks(
        original_intervals=[(5.0, 12.0)],
        speed_intervals=[(6.0, 7.0)],
        audio_intervals=[],
        requested_targets=["shot"],
    )
    assert len(bookmarks) == 1
    assert bookmarks[0]["event_type"] == "shot"
    assert bookmarks[0]["game_state"] is None


def test_event_engine_bookmarks_match_legacy_shape() -> None:
    """events_to_bookmarks (event engine) must produce rows with every key the
    legacy build_analysis_bookmarks rows carry, so job_runner/Studio keep
    working when the integration swaps the bookmark source - and every
    selected event gets its own bookmark, even when two share a clip."""
    from backend.services.event_engine import Event, EventSet, events_to_bookmarks, plan_reel

    legacy = build_analysis_bookmarks(
        original_intervals=[(10.0, 20.0)], speed_intervals=[(11.0, 13.0)], audio_intervals=[],
        requested_targets=[],
        goal_events=[{"t": 15.5, "side": "left", "confidence": 0.9, "reason": "r"}],
    )[0]

    events = EventSet(events=[
        Event(type="shot", t=14.0, side="left", team=1, team_name="AWAY", player_track_id=7, confidence=0.75,
              excitement=0.8, t_start=10.0, t_end=18.0, id="ev_0001", sources=["ball_tracking"]),
        Event(type="goal", t=15.5, side="left", team=1, team_name="AWAY", player_track_id=7, confidence=0.9,
              excitement=1.0, t_start=10.0, t_end=22.0, id="ev_0002", sources=["ball_tracking"],
              evidence={"game_state": "in_play"}),
        Event(type="corner_kick", t=60.0, side="right", team=0, confidence=0.8, excitement=0.4,
              t_start=57.0, t_end=67.0, id="ev_0003"),
    ], duration_s=90.0)
    plan = plan_reel(events, target_duration_s=60.0)
    rows = events_to_bookmarks(events, plan, trim_offset_s=30.0)

    assert set(legacy) <= set(rows[0])
    assert [r["event_type"] for r in rows] == ["shot", "goal", "corner_kick"]
    goal = rows[1]
    assert goal["label"] == "goal_detected"
    assert goal["occurred_at_s"] == 45.5 and goal["start_s"] <= 45.5 <= goal["end_s"]
    assert goal["signals"]["goal_side"] == "left"
    assert goal["game_state"] == "in_play"
    assert goal["team"] == 1 and goal["player_track_id"] == 7 and goal["excitement"] == 1.0
    assert rows[0]["clip_index"] == rows[1]["clip_index"]  # one clip, two bookmarks
    assert rows[2]["signals"]["set_piece_side"] == "right"
