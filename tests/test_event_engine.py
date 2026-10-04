"""Event engine + goal detection v2 on the synthetic match and adversarial tracks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.services.event_engine import (
    EVENTS_FILENAME,
    REEL_PRESETS,
    EventEngineConfig,
    detect_events,
    events_to_bookmarks,
    load_events,
    plan_reel,
    write_events,
)
from backend.services.game_tracking import (
    analyze_game_states,
    build_ball_track,
    detect_goal_candidates,
    detect_goal_events,
    detect_set_pieces,
    estimate_field_geometry,
    overlay_set_piece_states,
)
from backend.services.pitch_calibration import calibrate_from_corners
from backend.services.synthetic_match import SyntheticMatchSpec, generate_synthetic_match
from backend.services.tracking_types import (
    TEAM_A,
    TEAM_B,
    TrackingResult,
    ball_detections_from_rows,
    player_track_from_rows,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic") / "match.mp4"
    gt = generate_synthetic_match(out, SyntheticMatchSpec(duration_s=28.0))
    tracking = gt.tracking
    ball_track = build_ball_track(tracking.ball.to_samples(), tracking.frame_size)
    boxes = {k: {"x1": v[0], "y1": v[1], "x2": v[2], "y2": v[3]} for k, v in gt.goal_boxes_px.items()}
    # Calibrated pitch (known field rectangle + goal mouths), as a manual
    # calibration / pitch_calibration corners would provide.
    geometry = estimate_field_geometry(
        tracking.all_player_positions(), tracking.frame_size,
        goal_box_left=boxes["left"], goal_box_right=boxes["right"], field_bounds=gt.pitch_bounds_px,
    )
    goals = detect_goal_events(ball_track, geometry, 0.0, tracking.duration_s, player_tracks=tracking)
    segments = analyze_game_states(ball_track, geometry, 0.0, tracking.duration_s, goal_events=goals)
    set_pieces = detect_set_pieces(ball_track, geometry, 0.0, tracking.duration_s)
    segments = overlay_set_piece_states(segments, set_pieces)
    # Metric event thresholds (shot speed in m/s): the simulator moves the ball
    # at realistic speeds (25 m/s shots), below the uncalibrated fallback
    # threshold of 0.25 frame widths/s (~30 m/s on this framing).
    x0, y0, x1, y1 = gt.pitch_bounds_px
    calibration = calibrate_from_corners([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    events = detect_events(
        tracking, ball_track, geometry, segments, goals, set_pieces, [],
        team_names={0: "RED", 1: "BLUE"}, calibration=calibration,
    )
    return {"gt": gt, "tracking": tracking, "ball_track": ball_track, "geometry": geometry,
            "goals": goals, "segments": segments, "events": events}


def _ball_track(rows, frame=(1280, 720)):
    return build_ball_track(rows, frame)


def _line(t0, t1, p0, p1, hz=25.0):
    steps = max(1, int(round((t1 - t0) * hz)))
    return [(t0 + i / hz, p0[0] + (p1[0] - p0[0]) * i / steps, p0[1] + (p1[1] - p0[1]) * i / steps)
            for i in range(steps + 1)]


def _hold(t0, t1, p, hz=25.0):
    return [(t0 + i / hz, p[0], p[1]) for i in range(int(round((t1 - t0) * hz)) + 1)]


# ---------------------------------------------------------------------------
# Goal detection v2 on the synthetic match
# ---------------------------------------------------------------------------


def test_v2_finds_exactly_the_scripted_goals(synthetic) -> None:
    truth = synthetic["gt"].goal_times_s
    goals = synthetic["goals"]
    assert len(truth) == 2
    assert len(goals) == len(truth), [g.to_dict() for g in goals]
    for (t_true, side), goal in zip(sorted(truth), sorted(goals, key=lambda g: g.t)):
        assert abs(goal.t - t_true) <= 1.0, (goal.t, t_true)
        assert goal.side == side
        assert goal.confidence >= 0.7
        assert goal.verdict == "goal"
        assert goal.evidence["kickoff_validation"] == "players"
        assert goal.evidence["corroborated"] is True


def test_synthetic_goals_need_no_player_tracks_but_lose_confidence(synthetic) -> None:
    tracking, geometry = synthetic["tracking"], synthetic["geometry"]
    goals = detect_goal_events(synthetic["ball_track"], geometry, 0.0, tracking.duration_s)
    assert len(goals) == 2
    with_tracks = sorted(g.confidence for g in synthetic["goals"])
    without = sorted(g.confidence for g in goals)
    assert all(a > b for a, b in zip(with_tracks, without))


# ---------------------------------------------------------------------------
# Adversarial ball tracks: none of these is a goal
# ---------------------------------------------------------------------------


def _audio_peak_at(t_peak, duration=40.0):
    times = np.arange(0.0, duration, 0.1)
    rms = np.full_like(times, 0.05) + 0.002 * np.sin(times)
    rms[np.abs(times - t_peak) < 0.6] = 0.6
    return times, rms


def test_wide_shot_over_the_end_line_is_not_a_goal(synthetic) -> None:
    geometry, tracking = synthetic["geometry"], synthetic["tracking"]
    goal = geometry.right_goal
    gh = goal.y2 - goal.y1
    wide_y = goal.y2 + 0.5 * gh
    rows = _line(0.0, 1.0, (850.0, wide_y - 40.0), (geometry.x_max + 60.0, wide_y))
    rows += _hold(1.0, 3.0, (geometry.x_max + 60.0, wide_y))  # rests behind the line, wide
    cx = (geometry.x_min + geometry.x_max) / 2.0
    cy = (geometry.y_min + geometry.y_max) / 2.0
    rows += _hold(20.0, 21.0, (cx, cy))  # later restart at the centre
    track = _ball_track(rows)
    goals = detect_goal_events(track, geometry, 0.0, 30.0, player_tracks=tracking,
                               audio_envelope=_audio_peak_at(1.0))
    assert goals == []


def test_ball_out_for_a_corner_near_the_post_is_not_a_goal(synthetic) -> None:
    geometry, tracking = synthetic["geometry"], synthetic["tracking"]
    goal = geometry.right_goal
    gh = goal.y2 - goal.y1
    near_post_y = goal.y1 - 0.15 * gh  # just outside the near post
    behind = (geometry.x_max + 12.0, near_post_y)
    rows = _line(0.0, 2.0, (950.0, near_post_y + 30.0), behind)  # slow roll over the line
    rows += _hold(2.0, 5.0, behind)
    corner = (geometry.x_max - 6.0, geometry.y_min + 6.0)
    rows += _hold(14.0, 16.0, corner)  # corner kick taken
    rows += _line(16.04, 17.0, corner, (geometry.x_max - 150.0, goal.center[1]))
    track = _ball_track(rows)
    candidates = detect_goal_candidates(track, geometry, 0.0, 30.0, player_tracks=tracking,
                                        audio_envelope=_audio_peak_at(2.0))
    assert all(c.verdict != "goal" for c in candidates)
    assert detect_goal_events(track, geometry, 0.0, 30.0, player_tracks=tracking,
                              audio_envelope=_audio_peak_at(2.0)) == []


def test_ball_over_the_line_in_the_post_margin_with_crowd_noise_is_not_a_goal(synthetic) -> None:
    """Weak-only evidence (crossing + resting in the post margin band) plus a
    crowd roar must not make a goal; the ball is then taken for a corner."""
    geometry, tracking = synthetic["geometry"], synthetic["tracking"]
    goal = geometry.right_goal
    y = goal.y2 + 3.0  # just outside the post, inside the uncertainty band
    rows = _line(0.0, 1.0, (900.0, y), ((goal.x1 + goal.x2) / 2.0, y))
    rows += _hold(1.04, 4.0, ((goal.x1 + goal.x2) / 2.0, y))
    corner = (geometry.x_max - 6.0, geometry.y_max - 6.0)
    rows += _hold(14.0, 16.0, corner)
    rows += _line(16.04, 17.0, corner, (geometry.x_max - 150.0, goal.center[1]))
    track = _ball_track(rows)
    candidates = detect_goal_candidates(track, geometry, 0.0, 30.0, player_tracks=tracking,
                                        audio_envelope=_audio_peak_at(1.0))
    assert candidates and all(c.verdict != "goal" for c in candidates)
    assert all(not c.evidence["strong_signals"] for c in candidates)
    assert all(c.confidence < 0.6 for c in candidates)


def test_ball_hidden_then_back_at_midfield_is_not_a_goal(synthetic) -> None:
    geometry, tracking = synthetic["geometry"], synthetic["tracking"]
    goal = geometry.right_goal
    gy = goal.center[1]
    rows = _line(0.0, 0.8, (800.0, gy), (geometry.x_max - 25.0, gy))  # fast, at the mouth
    cx = (geometry.x_min + geometry.x_max) / 2.0
    rows += _line(2.8, 6.0, (cx + 60.0, gy), (cx - 200.0, gy - 80.0))  # back in play 2s later
    track = _ball_track(rows)
    candidates = detect_goal_candidates(track, geometry, 0.0, 10.0, player_tracks=tracking,
                                        audio_envelope=_audio_peak_at(0.8))
    assert candidates, "the vanish should still produce a (rejected) candidate"
    assert all(c.verdict != "goal" for c in candidates)
    assert all(c.confidence < 0.6 for c in candidates)
    assert any("reappeared_in_play_s" in c.evidence for c in candidates)
    assert detect_goal_events(track, geometry, 0.0, 10.0, player_tracks=tracking) == []


def test_single_weak_signal_is_a_low_confidence_candidate(synthetic) -> None:
    geometry = synthetic["geometry"]
    goal = geometry.left_goal
    # One sighting inside the goal box margin band (just outside the post).
    margin_y = goal.y2 + 3.0
    rows = _line(0.0, 1.0, (500.0, margin_y - 60.0), (geometry.x_min + 30.0, margin_y))
    rows += [(1.04, (goal.x1 + goal.x2) / 2.0, margin_y)]
    track = _ball_track(rows)
    candidates = detect_goal_candidates(track, geometry, 0.0, 30.0)
    assert candidates
    assert all(c.verdict in ("shot", "chance") for c in candidates)
    assert max(c.confidence for c in candidates) < 0.6
    assert detect_goal_events(track, geometry, 0.0, 30.0) == []


# ---------------------------------------------------------------------------
# Event engine on the synthetic match
# ---------------------------------------------------------------------------


def test_detect_events_finds_goal_shots_with_teams(synthetic) -> None:
    events = synthetic["events"]
    goals = events.by_type("goal")
    assert len(goals) == 2
    # Team A (0) defends left, so it scores in the right goal.
    for goal in goals:
        assert goal.team == (TEAM_A if goal.side == "right" else TEAM_B)
        assert goal.team_name == ("RED" if goal.team == 0 else "BLUE")
        assert goal.player_track_id is not None
        assert synthetic["tracking"].team_of(goal.player_track_id) == goal.team
    scripted_shot_starts = {"right": 8.0, "left": 15.0}
    for side, t_shot in scripted_shot_starts.items():
        shots = [e for e in events.by_type("shot") if e.side == side and e.evidence.get("outcome") == "goal"]
        assert len(shots) == 1, [e.to_dict() for e in events.by_type("shot")]
        shot = shots[0]
        assert abs(shot.t - t_shot) <= 0.5
        assert shot.team == (TEAM_A if side == "right" else TEAM_B)
        assert shot.evidence["on_target"] is True
        goal = next(g for g in goals if g.side == side)
        assert goal.player_track_id == shot.player_track_id


def test_excitement_ordering_and_event_contract(synthetic) -> None:
    events = synthetic["events"]
    goals = events.by_type("goal")
    others = [e for e in events if e.type != "goal"]
    assert all(g.excitement == pytest.approx(1.0) for g in goals)
    assert all(0.0 <= e.excitement <= 1.0 for e in events)
    assert min(g.excitement for g in goals) >= max(e.excitement for e in others)
    shots = events.by_type("shot")
    low = [e for e in events if e.type in ("turnover", "sprint")]
    if shots and low:
        assert np.mean([e.excitement for e in shots]) > np.mean([e.excitement for e in low])
    ids = [e.id for e in events]
    assert len(set(ids)) == len(ids)
    for ev in events:
        d = ev.to_dict()
        for key in ("id", "type", "t", "t_start", "t_end", "team", "team_name", "side", "player_track_id",
                    "secondary_track_id", "confidence", "excitement", "reason", "evidence", "sources"):
            assert key in d
        assert d["t_start"] <= d["t"] <= d["t_end"]
        assert d["t"] - d["t_start"] <= 15.0 + 1e-6
        assert d["t_end"] - d["t"] <= 10.0 + 1e-6
    # same-type events are merged within 3 s
    for etype in {e.type for e in events}:
        if etype in ("sprint", "dribble"):
            continue
        same = sorted((e for e in events if e.type == etype), key=lambda e: e.t)
        for a, b in zip(same, same[1:]):
            assert b.t - a.t > 3.0 or a.side != b.side


def test_plan_reel_respects_target_and_includes_goals(synthetic) -> None:
    events = synthetic["events"]
    plan = plan_reel(events, target_duration_s=12.0, pre_s=2.0, post_s=3.0)
    goal_ids = {e.id for e in events.by_type("goal")}
    assert goal_ids <= set(plan["selected_event_ids"])
    clips = plan["clips"]
    starts = [c["t_start"] for c in clips]
    assert starts == sorted(starts)
    for a, b in zip(clips, clips[1:]):
        assert a["t_end"] <= b["t_start"]
    # The goal windows alone exceed 12 s, so nothing else may be added.
    selected = [events.get(i) for i in plan["selected_event_ids"]]
    assert all(e.type in ("goal", "red_card") or any(e.id in c["event_ids"] for c in clips) for e in selected)

    roomy = plan_reel(events, target_duration_s=60.0)
    assert roomy["total_duration_s"] <= 60.0 + 1e-6
    assert len(roomy["selected_event_ids"]) >= len(plan["selected_event_ids"])
    assert goal_ids <= set(roomy["selected_event_ids"])

    tight = plan_reel(events, target_duration_s=40.0, must_include=())
    assert tight["total_duration_s"] <= 40.0 + 1e-6

    assert plan_reel(events, preset="1min")["target_duration_s"] == 60.0
    assert set(REEL_PRESETS) == {"1min", "3min", "5min", "10min"}


def test_events_to_bookmarks_legacy_shape(synthetic) -> None:
    events = synthetic["events"]
    plan = plan_reel(events, target_duration_s=60.0)
    rows = events_to_bookmarks(events, plan, trim_offset_s=100.0)
    assert len(rows) == len(plan["selected_event_ids"])
    legacy = {"bookmark_id", "index", "event_type", "label", "confidence", "start_s", "occurred_at_s",
              "end_s", "duration_s", "sources", "game_state", "signals"}
    for i, row in enumerate(rows, start=1):
        assert legacy <= set(row)
        assert {"team", "player_track_id", "excitement"} <= set(row)
        assert row["index"] == i and row["bookmark_id"] == f"bm_{i:04d}"
        assert row["label"] == f"{row['event_type']}_detected"
        assert row["start_s"] <= row["occurred_at_s"] <= row["end_s"]
        assert row["occurred_at_s"] >= 100.0  # source timebase
        assert isinstance(row["signals"], dict) and isinstance(row["sources"], list)
    goal_rows = [r for r in rows if r["event_type"] == "goal"]
    assert len(goal_rows) == 2
    assert all(r["signals"]["goal_side"] in ("left", "right") for r in goal_rows)
    # Several events can share one clip and each keeps its own bookmark.
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip_index"], []).append(r)
    assert any(len(v) > 1 for v in by_clip.values())


def test_write_and_load_events_roundtrip(synthetic, tmp_path) -> None:
    events = synthetic["events"]
    plan = plan_reel(events, target_duration_s=30.0)
    path = write_events(tmp_path, events, plan)
    assert path.endswith(EVENTS_FILENAME)
    doc = json.loads((tmp_path / EVENTS_FILENAME).read_text())
    assert {"generated_at", "trim_offset_seconds", "events", "reel_plan", "summary"} <= set(doc)
    assert {"target_duration_s", "selected_event_ids", "total_duration_s"} <= set(doc["reel_plan"])
    assert doc["summary"]["counts_by_type"]["goal"] == 2
    assert doc["summary"]["per_team"]["0"]["goals"] == 1
    assert doc["summary"]["per_team"]["1"]["goals"] == 1
    loaded = load_events(tmp_path)
    assert loaded is not None and len(loaded) == len(events)
    assert loaded.reel_plan["selected_event_ids"] == plan["selected_event_ids"]
    assert [e.type for e in loaded] == [e.type for e in events]
    assert load_events(tmp_path / "missing") is None


# ---------------------------------------------------------------------------
# Hand-built scenarios: saves, sprints, dribbles, calibration
# ---------------------------------------------------------------------------

FRAME = (1280, 720)


def _static_player(track_id, team, x, y, t1=40.0, hz=10.0, label=None):
    rows = [(i / hz, x - 9, y - 44, x + 9, y, 1.0) for i in range(int(t1 * hz) + 1)]
    return player_track_from_rows(track_id, rows, team=team, team_confidence=1.0, label=label)


def _moving_player(track_id, team, path, hz=10.0):
    rows = [(t, x - 9, y - 44, x + 9, y, 1.0) for t, x, y in path]
    return player_track_from_rows(track_id, rows, team=team, team_confidence=1.0)


def _geometry():
    return estimate_field_geometry(None, FRAME, field_bounds=(80.0, 90.0, 1200.0, 630.0),
                                   goal_box_left={"x1": 35.0, "y1": 290.0, "x2": 80.0, "y2": 430.0},
                                   goal_box_right={"x1": 1200.0, "y1": 290.0, "x2": 1245.0, "y2": 430.0})


def _tracking(players, ball_rows, duration=40.0, focus=None):
    return TrackingResult(fps=25.0, frame_size=FRAME, duration_s=duration,
                          players={p.track_id: p for p in players},
                          ball=ball_detections_from_rows(ball_rows), focus_track_id=focus)


def test_save_detected_for_shot_stopped_by_keeper() -> None:
    geometry = _geometry()
    keeper = _static_player(20, TEAM_B, 1180.0, 365.0)
    players = [keeper] + [_static_player(10 + i, TEAM_A, 300.0 + 80 * i, 200.0 + 60 * i) for i in range(4)]
    players += [_static_player(30 + i, TEAM_B, 800.0 + 60 * i, 220.0 + 70 * i) for i in range(4)]
    shooter = _static_player(9, TEAM_A, 880.0, 360.0)
    players.append(shooter)
    ball = _hold(0.0, 2.0, (890.0, 358.0))
    ball += _line(2.04, 2.32, (890.0, 358.0), (1160.0, 362.0))  # ~960 px/s at the goal
    ball += _line(2.36, 3.5, (1160.0, 362.0), (1100.0, 380.0))  # parried back out
    tracking = _tracking(players, ball)
    bt = build_ball_track(tracking.ball.to_samples(), FRAME)
    events = detect_events(tracking, bt, geometry, [], [], [], [])
    saves = events.by_type("save")
    assert len(saves) == 1, [e.to_dict() for e in events]
    assert saves[0].player_track_id == 20 and saves[0].team == TEAM_B and saves[0].side == "right"
    shots = [e for e in events.by_type("shot") if e.side == "right"]
    assert shots and shots[0].team == TEAM_A and shots[0].evidence["outcome"] == "saved"
    assert shots[0].player_track_id == 9
    assert saves[0].excitement > 0.7


def test_sprint_and_focus_threshold() -> None:
    geometry = _geometry()  # 1120 px wide -> 10.67 px/m
    px_per_m = geometry.width / 105.0
    fast = _moving_player(1, TEAM_A, [(i / 10.0, 200.0 + 8.0 * px_per_m * i / 10.0, 300.0) for i in range(31)])
    medium = _moving_player(2, TEAM_A, [(i / 10.0, 200.0 + 6.5 * px_per_m * i / 10.0, 500.0) for i in range(31)])
    others = [_static_player(30 + i, TEAM_B, 900.0, 200.0 + 80 * i, t1=3.0) for i in range(3)]
    ball = _hold(0.0, 3.0, (640.0, 100.0))
    tracking = _tracking([fast, medium] + others, ball, duration=3.0)
    bt = build_ball_track(tracking.ball.to_samples(), FRAME)

    events = detect_events(tracking, bt, geometry, [], [], [], [], focus_track_id=None)
    sprinters = {e.player_track_id for e in events.by_type("sprint")}
    assert sprinters == {1}
    sprint = events.by_type("sprint")[0]
    assert 7.0 < sprint.evidence["top_speed_mps"] < 9.0
    assert sprint.team == TEAM_A

    focused = detect_events(tracking, bt, geometry, [], [], [], [], focus_track_id=2)
    assert {e.player_track_id for e in focused.by_type("sprint")} == {1, 2}
    focus_sprint = next(e for e in focused.by_type("sprint") if e.player_track_id == 2)
    assert focus_sprint.evidence.get("focus_involved") is True


def test_dribble_and_turnover_from_possession() -> None:
    geometry = _geometry()
    px_per_m = geometry.width / 105.0
    path = [(i / 10.0, 300.0 + 6.0 * px_per_m * i / 10.0, 360.0) for i in range(41)]  # 24 m in 4 s
    dribbler = _moving_player(1, TEAM_A, path)
    chaser = _moving_player(5, TEAM_B, [(t, x + 2.0 * px_per_m, y - 10.0) for t, x, y in path])
    # Ball at the dribbler's feet, then a defender (team B) takes it.
    ball = [(t, x + 3.0, y) for t, x, y in path]
    thief = _static_player(6, TEAM_B, path[-1][1] + 60.0, 360.0, t1=8.0)
    ball += _line(4.04, 4.4, (path[-1][1] + 3.0, 360.0), (path[-1][1] + 60.0, 360.0), hz=10.0)
    ball += _hold(4.5, 6.0, (path[-1][1] + 60.0, 360.0), hz=10.0)
    tracking = _tracking([dribbler, chaser, thief], ball, duration=6.0)
    bt = build_ball_track(tracking.ball.to_samples(), FRAME)
    events = detect_events(tracking, bt, geometry, [], [], [], [], config=EventEngineConfig())
    dribbles = events.by_type("dribble")
    assert dribbles and dribbles[0].player_track_id == 1 and dribbles[0].team == TEAM_A
    assert dribbles[0].evidence["distance_m"] >= 15.0
    turnovers = events.by_type("turnover")
    assert any(e.team == TEAM_B and e.secondary_track_id == 1 for e in turnovers)


class _FakeCalibration:
    """Linear px -> metres map, duck-typing pitch_calibration.PitchCalibration."""

    source = "manual"
    confidence = 0.95
    pitch_length_m = 105.0
    pitch_width_m = 68.0

    def __init__(self, scale):
        self.scale = scale
        self.image_corners_px = np.array([[80.0, 90.0], [1200.0, 90.0], [1200.0, 630.0], [80.0, 630.0]])

    def to_pitch(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return (pts - np.array([640.0, 360.0])) / self.scale


def test_calibration_switches_to_metres() -> None:
    geometry = _geometry()
    cal = _FakeCalibration(scale=5.0)  # 5 px per metre: the same pixels are twice as fast
    px_per_m_uncal = geometry.width / 105.0
    path = [(i / 10.0, 200.0 + 4.0 * px_per_m_uncal * i / 10.0, 300.0) for i in range(31)]
    runner = _moving_player(1, TEAM_A, path)
    others = [_static_player(30, TEAM_B, 900.0, 300.0, t1=3.0)]
    tracking = _tracking([runner] + others, _hold(0.0, 3.0, (640.0, 100.0)), duration=3.0)
    bt = build_ball_track(tracking.ball.to_samples(), FRAME)
    plain = detect_events(tracking, bt, geometry, [], [], [], [])
    assert not plain.calibrated and plain.by_type("sprint") == []
    metric = detect_events(tracking, bt, geometry, [], [], [], [], calibration=cal)
    assert metric.calibrated
    assert [e.player_track_id for e in metric.by_type("sprint")] == [1]
    # The calibration also pins the field geometry.
    calibrated_geometry = estimate_field_geometry(None, FRAME, calibration=cal)
    assert calibrated_geometry.x_min == pytest.approx(80.0) and calibrated_geometry.x_max == pytest.approx(1200.0)
    assert "calibration" in calibrated_geometry.source
