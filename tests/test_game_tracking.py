from __future__ import annotations

import numpy as np
import pytest

from backend.services.game_tracking import (
    BallTrackConfig,
    GameStateConfig,
    STATE_BALL_LOST,
    STATE_IN_PLAY,
    STATE_RESTART_RIGHT,
    analyze_game_states,
    build_ball_track,
    detect_goal_events,
    estimate_field_geometry,
    state_at,
    summarize_states,
)

FRAME = (1920, 1080)


def _linear_detections(t0: float, t1: float, x0: float, x1: float, y: float, hz: float = 10.0):
    steps = int(round((t1 - t0) * hz))
    return [
        (t0 + i / hz, x0 + (x1 - x0) * (i / max(1, steps)), y)
        for i in range(steps + 1)
    ]


def _player_cloud(x_min=100.0, x_max=1820.0, y_min=200.0, y_max=880.0, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    ts = rng.uniform(0.0, 60.0, n)
    xs = rng.uniform(x_min, x_max, n)
    ys = rng.uniform(y_min, y_max, n)
    return np.stack([ts, xs, ys], axis=1)


# ---------------------------------------------------------------------------
# Ball track
# ---------------------------------------------------------------------------


def test_build_ball_track_rejects_teleporting_outlier() -> None:
    detections = _linear_detections(0.0, 2.0, 300.0, 600.0, 500.0)
    # Scoreboard-style false positive far away for a single frame.
    detections.append((1.05, 1900.0, 40.0))

    track = build_ball_track(detections, FRAME)

    assert track.stats["rejected_outliers"] >= 1
    pos = track.position_at(1.05)
    assert pos is not None
    assert abs(pos[0] - 457.5) < 60.0  # stays on the real path
    assert pos[1] > 400.0


def test_build_ball_track_interpolates_short_gaps_only() -> None:
    detections = _linear_detections(0.0, 1.0, 100.0, 200.0, 300.0)
    detections += _linear_detections(1.8, 2.8, 280.0, 380.0, 300.0)
    cfg = BallTrackConfig(max_interpolation_gap_s=1.25)
    track = build_ball_track(detections, FRAME, cfg)

    mid = track.position_at(1.4)
    assert mid is not None
    assert mid[2] == "interpolated"
    assert 200.0 < mid[0] < 280.0

    # A much longer gap must NOT be bridged.
    detections2 = _linear_detections(0.0, 1.0, 100.0, 200.0, 300.0)
    detections2 += _linear_detections(5.0, 6.0, 300.0, 400.0, 300.0)
    track2 = build_ball_track(detections2, FRAME, cfg)
    assert track2.position_at(3.0) is None
    gaps = track2.visibility_gaps(0.0, 6.0)
    assert any(start <= 1.1 and end >= 4.9 for start, end in gaps)


def test_build_ball_track_rejects_outlier_after_detector_dropout() -> None:
    """The association gate must stay bounded during detection gaps: after a
    ~0.6s dropout a single far-away false positive (scoreboard) must not
    capture the track away from the real ball."""
    detections = _linear_detections(0.0, 2.0, 380.0, 400.0, 500.0)
    detections.append((2.6, 1900.0, 40.0))  # scoreboard flash mid-dropout
    detections += _linear_detections(2.7, 3.5, 402.0, 410.0, 500.0)

    track = build_ball_track(detections, FRAME)

    pos = track.position_at(2.6)
    assert pos is not None
    assert pos[0] < 600.0, f"track teleported to the outlier: {pos}"
    late = track.position_at(3.2)
    assert late is not None
    assert abs(late[0] - 406.0) < 40.0


def test_build_ball_track_reacquires_after_silence() -> None:
    detections = _linear_detections(0.0, 2.0, 300.0, 400.0, 500.0)
    # 2s silence, ball re-appears on the other wing.
    detections += _linear_detections(4.0, 5.0, 1500.0, 1450.0, 350.0)
    track = build_ball_track(detections, FRAME)

    pos = track.position_at(4.5)
    assert pos is not None
    assert pos[0] > 1300.0


def test_ball_track_coverage_fraction() -> None:
    detections = _linear_detections(0.0, 5.0, 100.0, 600.0, 300.0)
    track = build_ball_track(detections, FRAME)
    assert track.coverage_fraction(0.0, 5.0) > 0.9
    assert track.coverage_fraction(10.0, 20.0) < 0.1


# ---------------------------------------------------------------------------
# Field geometry
# ---------------------------------------------------------------------------


def test_estimate_field_geometry_from_player_cloud() -> None:
    geometry = estimate_field_geometry(_player_cloud(), FRAME)

    assert 80.0 <= geometry.x_min <= 180.0
    assert 1750.0 <= geometry.x_max <= 1850.0
    assert geometry.left_goal.x2 <= geometry.x_min + 1.0
    assert geometry.right_goal.x1 >= geometry.x_max - 1.0
    # Goal mouths vertically inside the field band.
    assert geometry.y_min < geometry.left_goal.center[1] < geometry.y_max


def test_estimate_field_geometry_manual_override_normalized() -> None:
    manual_left = {"x1": 0.0, "y1": 0.4, "x2": 0.05, "y2": 0.6}
    geometry = estimate_field_geometry(_player_cloud(), FRAME, goal_box_left=manual_left)
    assert geometry.left_goal.x1 == 0.0
    assert abs(geometry.left_goal.x2 - 0.05 * FRAME[0]) < 1.0
    assert abs(geometry.left_goal.y1 - 0.4 * FRAME[1]) < 1.0
    assert "manual" in geometry.source


def test_estimated_goal_mouth_is_a_real_goal_not_a_third_of_the_field() -> None:
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    for goal in (geometry.left_goal, geometry.right_goal):
        # 7.32 m of a 68 m wide pitch ~ 11 % of the field height (was 30 %).
        assert goal.y2 - goal.y1 == pytest.approx(0.11 * geometry.height, rel=1e-6)
        assert goal.x2 - goal.x1 == pytest.approx(0.05 * geometry.width, rel=1e-6)
    wider = estimate_field_geometry(_player_cloud(), FRAME, goal_mouth_frac=0.2)
    assert wider.left_goal.y2 - wider.left_goal.y1 == pytest.approx(0.2 * wider.height, rel=1e-6)


def test_calibrated_goal_boxes_project_the_real_goal_mouth() -> None:
    from backend.services.pitch_calibration import calibrate_from_corners

    x0, y0, x1, y1 = 100.0, 200.0, 1820.0, 880.0  # top-down: 16.38 px/m along x, 10.0 px/m along y
    cal = calibrate_from_corners([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    geometry = estimate_field_geometry(_player_cloud(), FRAME, calibration=cal)
    px_x, px_y = (x1 - x0) / 105.0, (y1 - y0) / 68.0
    cy = (y0 + y1) / 2.0
    left, right = geometry.left_goal, geometry.right_goal
    for goal in (left, right):
        # Posts at +/-3.66 m plus 1 m beyond each post; 2.5 m deep behind the line.
        assert goal.y1 == pytest.approx(cy - 4.66 * px_y, abs=0.05)
        assert goal.y2 == pytest.approx(cy + 4.66 * px_y, abs=0.05)
        assert goal.x2 - goal.x1 == pytest.approx(2.5 * px_x, abs=0.05)
    assert left.x2 == pytest.approx(x0, abs=0.05) and right.x1 == pytest.approx(x1, abs=0.05)
    tight = estimate_field_geometry(None, FRAME, calibration=cal, goal_margin_m=0.0, goal_depth_m=2.0)
    assert tight.right_goal.y2 - tight.right_goal.y1 == pytest.approx(7.32 * px_y, abs=0.1)
    assert tight.right_goal.x2 - tight.right_goal.x1 == pytest.approx(2.0 * px_x, abs=0.05)
    # Manual goal boxes still override the calibrated mouth.
    manual = estimate_field_geometry(None, FRAME, calibration=cal,
                                     goal_box_right={"x1": 1820, "y1": 500, "x2": 1860, "y2": 580})
    assert (manual.right_goal.y1, manual.right_goal.y2) == (500.0, 580.0)
    assert manual.left_goal.y1 == pytest.approx(left.y1)


def test_estimate_field_geometry_without_players_falls_back() -> None:
    geometry = estimate_field_geometry(None, FRAME)
    assert geometry.source == "frame_default"
    assert geometry.x_min < geometry.x_max


# ---------------------------------------------------------------------------
# Game states: don't leave the goal during restart waits
# ---------------------------------------------------------------------------


def test_states_hold_at_goal_when_ball_goes_out_over_goal_line() -> None:
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    # Ball rolls out over the right goal line at ~2s, is gone until 10s,
    # then play restarts in midfield.
    detections = _linear_detections(0.0, 2.0, 1400.0, geometry.x_max + 60.0, geometry.right_goal.center[1])
    detections += _linear_detections(10.0, 12.0, 960.0, 900.0, 540.0)
    track = build_ball_track(detections, FRAME)

    segments = analyze_game_states(track, geometry, 0.0, 12.0)
    restart = [s for s in segments if s.state == STATE_RESTART_RIGHT]
    assert restart, f"expected a restart_right hold, got {[s.to_dict() for s in segments]}"
    hold = max(restart, key=lambda s: s.end_s - s.start_s)
    # The hold must span the entire out-of-play wait (~2s..10s): the camera
    # never leaves the goal while everyone waits for the goal kick/corner.
    assert hold.start_s <= 3.5
    assert hold.end_s >= 9.5
    assert "goal" in hold.reason
    # And play resumes afterwards.
    tail = state_at(segments, 11.5)
    assert tail is not None and tail.state == STATE_IN_PLAY


def test_states_hold_at_goal_when_ball_vanishes_near_goal() -> None:
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    near_goal_x = geometry.x_min + 0.05 * geometry.width
    detections = _linear_detections(0.0, 3.0, 700.0, near_goal_x, geometry.left_goal.center[1])
    detections += _linear_detections(9.0, 11.0, 960.0, 1000.0, 540.0)
    track = build_ball_track(detections, FRAME)

    segments = analyze_game_states(track, geometry, 0.0, 11.0)
    assert any(s.state == "restart_left" for s in segments)
    mid = state_at(segments, 6.0)
    assert mid is not None and mid.state == "restart_left"


def test_states_ball_lost_midfield_does_not_pin_to_goal() -> None:
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    detections = _linear_detections(0.0, 2.0, 900.0, 1000.0, 540.0)
    detections += _linear_detections(8.0, 9.0, 1000.0, 1050.0, 540.0)
    track = build_ball_track(detections, FRAME)

    segments = analyze_game_states(track, geometry, 0.0, 9.0)
    mid = state_at(segments, 5.0)
    assert mid is not None and mid.state == STATE_BALL_LOST
    summary = summarize_states(segments)
    assert summary.get(STATE_BALL_LOST, 0.0) > 2.0


# ---------------------------------------------------------------------------
# Goal detection
# ---------------------------------------------------------------------------


def test_goal_flagged_when_ball_observed_inside_goal() -> None:
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    goal = geometry.left_goal
    gx, gy = goal.center
    detections = _linear_detections(0.0, 1.5, 700.0, geometry.x_min + 40.0, gy)
    # Ball seen inside the goal box.
    detections += [(1.6, gx, gy), (1.7, gx, gy), (1.8, gx, gy)]
    track = build_ball_track(detections, FRAME)

    events = detect_goal_events(track, geometry, 0.0, 20.0)
    assert len(events) == 1
    assert events[0].side == "left"
    assert events[0].confidence >= 0.7
    assert events[0].evidence.get("observed_in_goal_box") is True


def _vanish_then_kickoff(geometry):
    gy = geometry.right_goal.center[1]
    # Fast shot toward the right goal, ball disappears just before the line.
    detections = _linear_detections(0.0, 1.0, 1100.0, geometry.x_max - 20.0, gy, hz=20.0)
    # Kickoff from the center circle a while later.
    center_x = (geometry.x_min + geometry.x_max) / 2.0
    center_y = (geometry.y_min + geometry.y_max) / 2.0
    detections += _linear_detections(20.0, 21.0, center_x, center_x + 50.0, center_y)
    return build_ball_track(detections, FRAME), center_x


def _teams_in_halves(center_x, t0=15.0, t1=25.0):
    from backend.services.tracking_types import player_track_from_rows

    players = {}
    for k in range(5):
        for team, x in ((0, center_x - 200.0 - 120.0 * k), (1, center_x + 200.0 + 120.0 * k)):
            tid = len(players) + 1
            y = 300.0 + 100.0 * k
            rows = [(t0 + i / 5.0, x - 10, y - 44, x + 10, y) for i in range(int((t1 - t0) * 5))]
            players[tid] = player_track_from_rows(tid, rows, team=team)
    return players


def test_vanish_into_mouth_plus_ball_only_kickoff_is_not_enough() -> None:
    """v2: the ball re-appearing near the centre is no longer a near-free
    +0.15. One strong signal (ball vanished into the mouth and stayed gone)
    plus a ball-only kickoff (+0.05) stays an uncorroborated shot (< 0.6)."""
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    track, _cx = _vanish_then_kickoff(geometry)
    assert detect_goal_events(track, geometry, 0.0, 25.0) == []
    from backend.services.game_tracking import detect_goal_candidates

    cands = detect_goal_candidates(track, geometry, 0.0, 25.0)
    assert len(cands) == 1 and cands[0].verdict == "shot"
    assert cands[0].confidence < 0.6
    assert cands[0].evidence["kickoff_validation"] == "ball_only"


def test_goal_flagged_when_ball_vanishes_and_both_teams_line_up_for_kickoff() -> None:
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    track, center_x = _vanish_then_kickoff(geometry)
    events = detect_goal_events(track, geometry, 0.0, 25.0, player_tracks=_teams_in_halves(center_x))
    assert len(events) == 1
    assert events[0].side == "right"
    assert "kickoff_reappearance_s" in events[0].evidence
    assert events[0].evidence["kickoff_validation"] == "players"
    assert events[0].confidence >= 0.7


def test_goal_flagged_when_ball_vanishes_with_crowd_peak() -> None:
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    track, _cx = _vanish_then_kickoff(geometry)
    times = np.arange(0.0, 30.0, 0.1)
    rms = np.full_like(times, 0.05)
    rms[(times > 1.0) & (times < 2.5)] = 0.6
    events = detect_goal_events(track, geometry, 0.0, 25.0, audio_envelope=(times, rms))
    assert len(events) == 1 and events[0].confidence >= 0.7
    assert events[0].evidence["audio_peak_s"] == pytest.approx(1.1, abs=0.2)


def test_teams_not_in_halves_reject_kickoff_bonus() -> None:
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    track, center_x = _vanish_then_kickoff(geometry)
    mixed = _teams_in_halves(center_x)
    for track_obj in mixed.values():  # everyone on one side: not a kickoff line-up
        if track_obj.team == 1:
            track_obj.x1 -= 900.0
            track_obj.x2 -= 900.0
    assert detect_goal_events(track, geometry, 0.0, 25.0, player_tracks=mixed) == []


def test_no_goal_when_ball_vanishes_near_goal_as_recording_ends() -> None:
    """A keeper catch / recording cut right after motion toward the goal must
    not be flagged: the trailing visibility gap earns no disappearance bonus
    and the event falls below the confidence floor."""
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    gy = geometry.right_goal.center[1]
    detections = _linear_detections(0.0, 1.0, 1100.0, geometry.x_max - 60.0, gy, hz=20.0)
    track = build_ball_track(detections, FRAME)

    # Window extends well past the last sighting with no kickoff ever seen.
    events = detect_goal_events(track, geometry, 0.0, 30.0)
    assert events == []


def test_no_goal_for_shot_wide_of_the_posts() -> None:
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    wide_y = geometry.right_goal.y2 + 0.35 * geometry.height
    detections = _linear_detections(0.0, 1.0, 1100.0, geometry.x_max - 20.0, wide_y, hz=20.0)
    track = build_ball_track(detections, FRAME)

    events = detect_goal_events(track, geometry, 0.0, 10.0)
    assert events == []


def test_ball_resting_a_few_metres_wide_of_the_estimated_mouth_is_not_a_goal() -> None:
    """Regression: with the old 30 %-of-the-field goal box a ball crossing the
    line ~5 m wide of the post (8 % of the field height off-centre) and
    resting there scored as a strong crossing + ball in the net."""
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    goal = geometry.right_goal
    y = goal.center[1] + 0.08 * geometry.height
    detections = _linear_detections(0.0, 1.0, 1300.0, geometry.x_max + 15.0, y, hz=20.0)
    detections += [(1.0 + i / 20.0, geometry.x_max + 15.0, y) for i in range(1, 40)]
    track = build_ball_track(detections, FRAME)
    times = np.arange(0.0, 30.0, 0.1)
    rms = np.full_like(times, 0.05)
    rms[(times > 1.0) & (times < 2.5)] = 0.6
    from backend.services.game_tracking import detect_goal_candidates

    cands = detect_goal_candidates(track, geometry, 0.0, 10.0, audio_envelope=(times, rms))
    assert cands and all(not c.evidence["strong_signals"] for c in cands)
    assert detect_goal_events(track, geometry, 0.0, 10.0, audio_envelope=(times, rms)) == []


def test_goal_hold_appears_in_state_timeline() -> None:
    geometry = estimate_field_geometry(_player_cloud(), FRAME)
    goal = geometry.left_goal
    gx, gy = goal.center
    detections = _linear_detections(0.0, 1.5, 700.0, geometry.x_min + 40.0, gy)
    detections += [(1.6, gx, gy), (1.7, gx, gy)]
    track = build_ball_track(detections, FRAME)
    events = detect_goal_events(track, geometry, 0.0, 20.0)

    segments = analyze_game_states(track, geometry, 0.0, 20.0, goal_events=events)
    goal_states = [s for s in segments if s.state == "goal_left"]
    assert goal_states
    cfg = GameStateConfig()
    longest = max(goal_states, key=lambda s: s.end_s - s.start_s)
    assert (longest.end_s - longest.start_s) >= cfg.goal_hold_s * 0.8
