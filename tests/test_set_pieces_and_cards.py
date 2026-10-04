from __future__ import annotations

import numpy as np
import pytest

from backend.services.camera_planner import CameraPlannerConfig, plan_camera
from backend.services.card_detection import (
    CardDetectionConfig,
    detect_card_events,
    stopped_play_windows,
)
from backend.services.game_tracking import (
    GameStateSegment,
    STATE_CORNER_SETUP,
    STATE_FREE_KICK_SETUP,
    STATE_IN_PLAY,
    build_ball_track,
    detect_set_pieces,
    estimate_field_geometry,
    overlay_set_piece_states,
)

FRAME = (1920, 1080)


def _geometry():
    rng = np.random.default_rng(3)
    n = 3000
    cloud = np.stack(
        [rng.uniform(0, 60, n), rng.uniform(100, 1820, n), rng.uniform(200, 880, n)],
        axis=1,
    )
    return estimate_field_geometry(cloud, FRAME)


def _stationary_then_kick(x, y, t0=2.0, hold_s=2.0, kick_to=None, hz=10.0, kick_s=1.2):
    """Ball moves in, sits at (x, y), then is kicked toward kick_to."""
    if kick_to is None:
        # Default: a solid kick toward the far side of the field.
        kick_to = (x - 700.0, 500.0) if x > 960.0 else (x + 700.0, 500.0)
    detections = []
    steps = int(t0 * hz)
    for i in range(steps):
        f = i / steps
        detections.append((i / hz, x - 200.0 * (1 - f), y + 100.0 * (1 - f)))
    hold_steps = int(hold_s * hz)
    for i in range(hold_steps):
        detections.append((t0 + i / hz, x + (i % 3) * 2.0, y + (i % 2) * 2.0))
    t_kick = t0 + hold_s
    for i in range(int(kick_s * hz)):
        f = i / (kick_s * hz)
        detections.append((t_kick + i / hz, x + (kick_to[0] - x) * f, y + (kick_to[1] - y) * f))
    return detections, t_kick


def test_corner_kick_detected_at_field_corner() -> None:
    geometry = _geometry()
    detections, t_kick = _stationary_then_kick(geometry.x_max - 15.0, geometry.y_max - 15.0)
    track = build_ball_track(detections, FRAME)

    events = detect_set_pieces(track, geometry, 0.0, 10.0)
    corners = [e for e in events if e.kind == "corner_kick"]
    assert corners, f"no corner found: {[e.to_dict() for e in events]}"
    assert corners[0].side == "right"
    assert abs(corners[0].t_kick - t_kick) < 1.0


def test_free_kick_near_goal_detected_with_threatened_side() -> None:
    geometry = _geometry()
    x = geometry.x_min + 0.25 * geometry.width  # shooting range of the left goal
    y = (geometry.y_min + geometry.y_max) / 2.0 + 120.0
    detections, _ = _stationary_then_kick(x, y)
    track = build_ball_track(detections, FRAME)

    events = detect_set_pieces(track, geometry, 0.0, 10.0)
    frees = [e for e in events if e.kind in {"free_kick", "penalty_kick"}]
    assert frees
    assert frees[0].side == "left"


def test_kickoff_classified_at_center() -> None:
    geometry = _geometry()
    cx = (geometry.x_min + geometry.x_max) / 2.0
    cy = (geometry.y_min + geometry.y_max) / 2.0
    detections, _ = _stationary_then_kick(cx, cy)
    track = build_ball_track(detections, FRAME)

    events = detect_set_pieces(track, geometry, 0.0, 10.0)
    assert any(e.kind == "kickoff" for e in events)


def test_moving_ball_produces_no_set_pieces() -> None:
    geometry = _geometry()
    hz = 10.0
    detections = [(i / hz, 300.0 + i * 12.0, 500.0) for i in range(80)]
    track = build_ball_track(detections, FRAME)
    assert detect_set_pieces(track, geometry, 0.0, 8.0) == []


def test_overlay_set_piece_states_carves_setup_window() -> None:
    geometry = _geometry()
    detections, t_kick = _stationary_then_kick(geometry.x_max - 15.0, geometry.y_max - 15.0)
    track = build_ball_track(detections, FRAME)
    events = detect_set_pieces(track, geometry, 0.0, 10.0)
    base = [GameStateSegment(0.0, 10.0, STATE_IN_PLAY, reason="ball visible in field")]

    segments = overlay_set_piece_states(base, events)
    setups = [s for s in segments if s.state == STATE_CORNER_SETUP]
    assert setups
    assert setups[0].side == "right"
    # Timeline stays contiguous.
    for a, b in zip(segments[:-1], segments[1:]):
        assert abs(a.end_s - b.start_s) < 1e-6


def test_camera_keeps_goal_in_view_during_free_kick_setup() -> None:
    geometry = _geometry()
    x = geometry.x_min + 0.25 * geometry.width
    y = geometry.left_goal.center[1] + 100.0
    detections, _ = _stationary_then_kick(x, y, t0=1.0, hold_s=4.0)
    track = build_ball_track(detections, FRAME)
    segments = [
        GameStateSegment(0.0, 1.0, STATE_IN_PLAY, reason="in play"),
        GameStateSegment(1.0, 5.0, STATE_FREE_KICK_SETUP, side="left",
                         reason="free kick setup - keeping the left goal in view"),
        GameStateSegment(5.0, 8.0, STATE_IN_PLAY, reason="in play"),
    ]

    plan = plan_camera(
        ball_track=track, player_positions=None, geometry=geometry,
        segments=segments, start_seconds=0.0, end_seconds=8.0, fps=10.0,
        frame_size=FRAME, base_zoom=1.8,
    )

    gx, gy = geometry.left_goal.center
    settled = [d for d in plan.decisions if 3.0 <= d.t <= 4.8]
    assert settled
    for d in settled:
        assert d.focus == "set_piece"
        half_w = FRAME[0] / d.zoom / 2.0
        half_h = FRAME[1] / d.zoom / 2.0
        assert abs(gx - d.center_x) <= half_w + 1.0, "goal left the frame during free kick"
        assert abs(gy - d.center_y) <= half_h + 1.0
        # Ball must be in frame too.
        assert abs(x - d.center_x) <= half_w + 1.0, "ball left the frame during free kick"


def test_threat_zoom_tightens_as_ball_approaches_goal() -> None:
    geometry = _geometry()
    gy = geometry.right_goal.center[1]
    hz = 15.0
    start_x = geometry.x_max - 0.31 * geometry.width
    detections = [
        (i / hz, start_x + (geometry.x_max - 40.0 - start_x) * (i / (6.0 * hz)), gy)
        for i in range(int(6.0 * hz))
    ]
    track = build_ball_track(detections, FRAME)
    segments = [GameStateSegment(0.0, 6.0, STATE_IN_PLAY, reason="in play")]

    plan = plan_camera(
        ball_track=track, player_positions=None, geometry=geometry,
        segments=segments, start_seconds=0.0, end_seconds=6.0, fps=10.0,
        frame_size=FRAME, base_zoom=1.8,
    )

    threat = [d for d in plan.decisions if d.focus == "ball_goal_threat"]
    assert len(threat) >= 20
    # Zoom tightens (increases) as the ball nears the goal.
    early = np.mean([d.zoom for d in threat[: len(threat) // 3]])
    late = np.mean([d.zoom for d in threat[-len(threat) // 3 :]])
    assert late > early + 0.1
    # Goal stays in frame throughout the attack.
    gx = geometry.right_goal.center[0]
    for d in threat:
        assert abs(gx - d.center_x) <= FRAME[0] / d.zoom / 2.0 + 1.0


def test_smoothed_plan_obeys_acceleration_limit() -> None:
    geometry = _geometry()
    hz = 15.0
    # Ball teleports mid-sequence (re-acquisition on the far wing).
    detections = [(i / hz, 300.0, 500.0) for i in range(int(2 * hz))]
    detections += [(2.0 + 1.5 + i / hz, 1500.0, 400.0) for i in range(int(2 * hz))]
    track = build_ball_track(detections, FRAME)
    segments = [GameStateSegment(0.0, 6.0, STATE_IN_PLAY, reason="in play")]
    cfg = CameraPlannerConfig()
    fps = 30.0

    plan = plan_camera(
        ball_track=track, player_positions=None, geometry=geometry,
        segments=segments, start_seconds=0.0, end_seconds=6.0, fps=fps,
        frame_size=FRAME, base_zoom=1.8, config=cfg,
    )

    dt = 1.0 / fps
    prev_v = None
    for a, b in zip(plan.decisions[:-1], plan.decisions[1:]):
        v = ((b.center_x - a.center_x) / dt, (b.center_y - a.center_y) / dt)
        crop_w = FRAME[0] / a.zoom
        assert np.hypot(*v) <= cfg.max_pan_speed_crop_frac * crop_w * 1.05
        if prev_v is not None:
            accel = np.hypot(v[0] - prev_v[0], v[1] - prev_v[1]) / dt
            assert accel <= cfg.max_pan_accel_crop_frac * crop_w * 1.6, f"jerk at t={a.t:.2f}"
        prev_v = v


# ---------------------------------------------------------------------------
# Card detection
# ---------------------------------------------------------------------------

cv2 = pytest.importorskip("cv2")


def _write_card_video(path, card_bgr, frames=40, card_frames=range(10, 30)):
    w, h = 640, 360
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    assert writer.isOpened()
    try:
        for i in range(frames):
            frame = np.full((h, w, 3), (40, 90, 40), dtype=np.uint8)
            # A couple of players.
            cv2.circle(frame, (200, 200), 12, (255, 255, 255), -1)
            cv2.circle(frame, (260, 210), 12, (128, 128, 230), -1)
            if i in card_frames:
                cv2.rectangle(frame, (240, 130), (252, 150), card_bgr, -1)
            writer.write(frame)
    finally:
        writer.release()


def test_yellow_card_flagged(tmp_path) -> None:
    video = tmp_path / "cards.mp4"
    _write_card_video(video, (0, 220, 240))  # BGR yellow

    events = detect_card_events(str(video), [(0.0, 4.0)], debug_dir=str(tmp_path / "crops"))

    assert len(events) == 1
    assert events[0].kind == "yellow_card"
    assert events[0].confidence >= 0.55
    assert events[0].crop_path and (tmp_path / "crops").exists()


def test_red_card_flagged_and_no_false_positive_without_card(tmp_path) -> None:
    video = tmp_path / "red.mp4"
    _write_card_video(video, (0, 0, 230))  # BGR red
    events = detect_card_events(str(video), [(0.0, 4.0)])
    assert len(events) == 1 and events[0].kind == "red_card"

    clean = tmp_path / "clean.mp4"
    _write_card_video(clean, (0, 220, 240), card_frames=range(0))  # never shown
    assert detect_card_events(str(clean), [(0.0, 4.0)]) == []


def test_large_jersey_region_not_flagged(tmp_path) -> None:
    video = tmp_path / "jersey.mp4"
    w, h = 640, 360
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    try:
        for _ in range(40):
            frame = np.full((h, w, 3), (40, 90, 40), dtype=np.uint8)
            # Big yellow jersey blob - far larger than a card.
            cv2.rectangle(frame, (200, 120), (280, 260), (0, 220, 240), -1)
            writer.write(frame)
    finally:
        writer.release()
    assert detect_card_events(str(video), [(0.0, 4.0)]) == []


def test_round_yellow_ball_not_flagged_as_card(tmp_path) -> None:
    """A stationary or slow-moving yellow BALL (circular) must not flag."""
    video = tmp_path / "ball.mp4"
    w, h = 640, 360
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    try:
        for i in range(40):
            frame = np.full((h, w, 3), (40, 90, 40), dtype=np.uint8)
            cv2.circle(frame, (300 + i, 250), 7, (0, 220, 240), -1)
            writer.write(frame)
    finally:
        writer.release()
    assert detect_card_events(str(video), [(0.0, 4.0)]) == []


def test_walking_player_bib_not_flagged_as_card(tmp_path) -> None:
    """A small card-shaped patch that MOVES across the frame (bib on a
    walking player) must be rejected by the stationarity gate."""
    video = tmp_path / "bib.mp4"
    w, h = 640, 360
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    try:
        for i in range(40):
            frame = np.full((h, w, 3), (40, 90, 40), dtype=np.uint8)
            x = 100 + i * 9  # ~90 px/s walk
            cv2.rectangle(frame, (x, 150), (x + 12, 170), (0, 220, 240), -1)
            writer.write(frame)
    finally:
        writer.release()
    assert detect_card_events(str(video), [(0.0, 4.0)]) == []


def test_stopped_play_windows_merges_and_pads() -> None:
    segments = [
        GameStateSegment(0.0, 10.0, STATE_IN_PLAY, reason=""),
        GameStateSegment(10.0, 14.0, "ball_lost", reason=""),
        GameStateSegment(14.0, 15.0, STATE_IN_PLAY, reason=""),
        GameStateSegment(15.0, 20.0, "restart_left", side="left", reason=""),
        GameStateSegment(20.0, 60.0, STATE_IN_PLAY, reason=""),
        GameStateSegment(60.0, 63.0, STATE_FREE_KICK_SETUP, side="right", reason=""),
    ]
    windows = stopped_play_windows(segments, pad_s=2.0)
    assert windows == [(8.0, 22.0), (58.0, 65.0)]


def test_set_pieces_and_cards_pass_through_event_engine() -> None:
    from backend.services.card_detection import CardEvent
    from backend.services.event_engine import detect_events
    from backend.services.tracking_types import TrackingResult, ball_detections_from_rows, player_track_from_rows

    geometry = _geometry()
    detections, t_kick = _stationary_then_kick(geometry.x_max - 15.0, geometry.y_max - 15.0)
    track = build_ball_track(detections, FRAME)
    set_pieces = detect_set_pieces(track, geometry, 0.0, 10.0)
    assert any(sp.kind == "corner_kick" for sp in set_pieces)

    def _player(tid, team, x, y):
        rows = [(i / 10.0, x - 10, y - 50, x + 10, y) for i in range(101)]
        return player_track_from_rows(tid, rows, team=team)

    taker = _player(1, 0, geometry.x_max - 25.0, geometry.y_max - 10.0)
    players = {1: taker, 2: _player(2, 0, 700.0, 500.0), 3: _player(3, 1, 1500.0, 500.0)}
    tracking = TrackingResult(fps=10.0, frame_size=FRAME, duration_s=10.0, players=players,
                              ball=ball_detections_from_rows(detections))
    card = CardEvent(t=8.0, kind="red_card", confidence=0.7, x=900.0, y=400.0, reason="red card shown")
    events = detect_events(tracking, track, geometry, [], [], set_pieces, [card])

    corners = events.by_type("corner_kick")
    assert len(corners) == 1
    assert corners[0].player_track_id == 1 and corners[0].team == 0
    assert corners[0].side == "right"
    assert abs(corners[0].t - t_kick) < 1.0
    assert 0.3 <= corners[0].excitement <= 0.6
    reds = events.by_type("red_card")
    assert len(reds) == 1 and reds[0].confidence == pytest.approx(0.7)
    assert "vision" in reds[0].sources
