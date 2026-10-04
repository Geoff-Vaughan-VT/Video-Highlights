from __future__ import annotations

import json
import math

import numpy as np
import pytest

from backend.services.camera_planner import (
    CameraPlannerConfig,
    plan_camera,
    slice_plan,
)
from backend.services.game_tracking import (
    GameStateSegment,
    STATE_BALL_LOST,
    STATE_IN_PLAY,
    build_ball_track,
    estimate_field_geometry,
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


def _moving_ball(t0=0.0, t1=6.0, x0=300.0, x1=1500.0, y=540.0, hz=15.0):
    steps = int((t1 - t0) * hz)
    return build_ball_track(
        [(t0 + i / hz, x0 + (x1 - x0) * i / steps, y) for i in range(steps + 1)],
        FRAME,
    )


def test_plan_follows_moving_ball() -> None:
    geometry = _geometry()
    ball = _moving_ball()
    segments = [GameStateSegment(0.0, 6.0, STATE_IN_PLAY, reason="ball visible in field")]

    plan = plan_camera(
        ball_track=ball,
        player_positions=None,
        geometry=geometry,
        segments=segments,
        start_seconds=0.0,
        end_seconds=6.0,
        fps=10.0,
        frame_size=FRAME,
        base_zoom=1.6,
    )

    assert len(plan) == 60
    xs = [d.center_x for d in plan.decisions]
    assert xs[-1] > xs[0] + 400.0  # camera panned with the ball
    assert all(d.reason for d in plan.decisions)
    # The run ends near the right goal, so late frames legitimately switch
    # to goal-threat framing - still ball-driven camera work.
    following = [d for d in plan.decisions if d.focus in {"ball", "ball_lead", "ball_goal_threat"}]
    assert len(following) >= 50
    assert plan.summary()["mean_confidence"] > 0.8


def test_plan_holds_at_goal_during_restart_and_does_not_leave() -> None:
    geometry = _geometry()
    # Ball visible only for the first second, then gone for the whole wait.
    ball = _moving_ball(0.0, 1.0, 1500.0, geometry.x_max - 10.0, geometry.right_goal.center[1])
    segments = [
        GameStateSegment(0.0, 1.2, STATE_IN_PLAY, reason="ball visible in field"),
        GameStateSegment(1.2, 15.0, "restart_right", side="right",
                         reason="ball out over the right goal line - waiting for goal kick/corner"),
    ]
    cfg = CameraPlannerConfig()

    plan = plan_camera(
        ball_track=ball,
        player_positions=None,
        geometry=geometry,
        segments=segments,
        start_seconds=0.0,
        end_seconds=15.0,
        fps=10.0,
        frame_size=FRAME,
        base_zoom=1.6,
        config=cfg,
    )

    goal_x, goal_y = geometry.right_goal.center
    aim_x = goal_x - cfg.goal_infield_offset_frac * geometry.width
    # Give the spring 2.5s to settle, then the camera must sit at the goal
    # for the entire remaining wait - it never drifts back to midfield.
    settled = [d for d in plan.decisions if 4.0 <= d.t <= 14.5]
    assert settled
    for decision in settled:
        # Camera center is clamped by the crop, so compare against the
        # clamped aim point rather than the raw goal center.
        crop_half_w = FRAME[0] / decision.zoom / 2.0
        expected_x = min(aim_x, FRAME[0] - crop_half_w)
        assert abs(decision.center_x - expected_x) < 60.0, (
            f"camera left the goal at t={decision.t:.1f}s: {decision.center_x:.0f} vs {expected_x:.0f}"
        )
        assert decision.focus == "goal_right"
        assert "goal" in decision.reason
    # v2: zoom is rate limited (<= 0.12x/s), so the change from the opening
    # goal-threat framing (~1.9x) down to the restart cap is a slow, monotone
    # zoom-out instead of the old near-instant jump; it settles by ~6 s.
    zooms = [d.zoom for d in settled]
    assert all(b <= a + 1e-6 for a, b in zip(zooms, zooms[1:]))
    for decision in settled:
        if decision.t >= 6.5:
            assert decision.zoom <= cfg.restart_zoom_cap + 0.05
    rates = np.abs(np.diff([d.zoom for d in plan.decisions])) * 10.0
    assert rates.max() <= cfg.max_zoom_rate_per_s + 0.02


def test_plan_zooms_out_and_follows_players_when_ball_lost() -> None:
    geometry = _geometry()
    ball = _moving_ball(0.0, 1.0, 900.0, 950.0, 500.0)
    # Player cluster sits far from the last ball spot.
    rng = np.random.default_rng(11)
    n = 800
    players = np.stack(
        [rng.uniform(0, 20, n), rng.normal(1400.0, 40.0, n), rng.normal(700.0, 30.0, n)],
        axis=1,
    )
    segments = [
        GameStateSegment(0.0, 1.2, STATE_IN_PLAY, reason="ball visible in field"),
        GameStateSegment(1.2, 12.0, STATE_BALL_LOST, reason="ball not visible"),
    ]
    base_zoom = 1.6

    plan = plan_camera(
        ball_track=ball,
        player_positions=players,
        geometry=geometry,
        segments=segments,
        start_seconds=0.0,
        end_seconds=12.0,
        fps=10.0,
        frame_size=FRAME,
        base_zoom=base_zoom,
    )

    late = [d for d in plan.decisions if d.t >= 8.0]
    assert late
    for decision in late:
        assert decision.zoom < base_zoom - 0.1  # zoomed out while searching
    assert any(d.focus == "action_centroid" for d in late)
    # Drifted toward the player cluster.
    assert late[-1].center_x > 1150.0


def test_plan_centers_always_within_legal_crop() -> None:
    geometry = _geometry()
    ball = _moving_ball(0.0, 4.0, -200.0, 2200.0, 100.0)  # deliberately out of frame
    segments = [GameStateSegment(0.0, 4.0, STATE_IN_PLAY, reason="test")]

    plan = plan_camera(
        ball_track=ball,
        player_positions=None,
        geometry=geometry,
        segments=segments,
        start_seconds=0.0,
        end_seconds=4.0,
        fps=12.0,
        frame_size=FRAME,
        base_zoom=2.0,
    )

    for decision in plan.decisions:
        half_w = FRAME[0] / decision.zoom / 2.0
        half_h = FRAME[1] / decision.zoom / 2.0
        assert half_w - 0.51 <= decision.center_x <= FRAME[0] - half_w + 0.51
        assert half_h - 0.51 <= decision.center_y <= FRAME[1] - half_h + 0.51
        assert decision.zoom >= 1.0


def test_slice_plan_reindexes_decisions() -> None:
    geometry = _geometry()
    ball = _moving_ball()
    segments = [GameStateSegment(0.0, 6.0, STATE_IN_PLAY, reason="test")]
    plan = plan_camera(
        ball_track=ball, player_positions=None, geometry=geometry, segments=segments,
        start_seconds=0.0, end_seconds=6.0, fps=10.0, frame_size=FRAME, base_zoom=1.6,
    )

    sub = slice_plan(plan, 2.0, 4.0)
    assert len(sub) == 20
    assert sub.decisions[0].index == 0
    assert math.isclose(sub.start_seconds, 2.0, abs_tol=1e-6)
    assert math.isclose(sub.decisions[0].t, 2.0, abs_tol=1e-6)
    # Original untouched.
    assert plan.decisions[20].index == 20


def test_plan_writes_jsonl_decisions(tmp_path) -> None:
    geometry = _geometry()
    ball = _moving_ball(0.0, 1.0)
    segments = [GameStateSegment(0.0, 1.0, STATE_IN_PLAY, reason="test")]
    plan = plan_camera(
        ball_track=ball, player_positions=None, geometry=geometry, segments=segments,
        start_seconds=0.0, end_seconds=1.0, fps=5.0, frame_size=FRAME, base_zoom=1.6,
    )

    path = tmp_path / "decisions.jsonl"
    plan.write_jsonl(str(path))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len(plan)
    assert {"t", "center_x", "center_y", "zoom", "state", "focus", "reason", "confidence"} <= set(rows[0])


# ---------------------------------------------------------------------------
# Planner v2
# ---------------------------------------------------------------------------

import re  # noqa: E402

from backend.services.camera_planner import (  # noqa: E402
    CameraPlan,
    _zoom_hysteresis,
    crop_rects_from_centers,
    resolve_output_size,
)
from backend.services.camera_quality import compute_plan_quality  # noqa: E402

UHD = (3840, 2160)


def _uhd_geometry():
    rng = np.random.default_rng(4)
    n = 3000
    cloud = np.stack(
        [rng.uniform(0, 60, n), rng.uniform(200, 3640, n), rng.uniform(400, 1760, n)], axis=1
    )
    return estimate_field_geometry(cloud, UHD)


def test_max_zoom_is_bounded_so_no_output_pixel_is_upscaled() -> None:
    geometry = _uhd_geometry()
    hz = 15.0
    track = build_ball_track([(i / hz, 1900.0 + 2 * i, 1080.0) for i in range(int(8 * hz))], UHD)
    segments = [GameStateSegment(0.0, 8.0, STATE_IN_PLAY, reason="in play")]

    plan = plan_camera(
        ball_track=track, player_positions=None, geometry=geometry, segments=segments,
        start_seconds=0.0, end_seconds=8.0, fps=25.0, frame_size=UHD, base_zoom=3.0,
        output_size=(1920, 1080),
    )

    assert plan.max_zoom == 2.0
    assert plan.output_size == (1920, 1080)
    assert max(d.zoom for d in plan.decisions) <= 2.0 + 1e-9
    rects = plan.crop_rects
    assert rects is not None and rects.shape == (len(plan), 4)
    assert (rects % 2 == 0).all(), "crop rects must be even integers"
    assert (rects[:, 2] >= 1920).all() and (rects[:, 3] >= 1080).all()
    # Crop aspect == output aspect.
    assert np.allclose(rects[:, 3] / rects[:, 2], 1080 / 1920, atol=2.0 / 1080)
    assert (rects[:, 0] + rects[:, 2] <= UHD[0]).all() and (rects[:, 1] + rects[:, 3] <= UHD[1]).all()


def test_resolve_output_size_defaults_never_exceed_source() -> None:
    assert resolve_output_size((3840, 2160)) == (1920, 1080)
    assert resolve_output_size((1280, 720)) == (1280, 720)
    assert resolve_output_size((3840, 2160), (1280, 720)) == (1280, 720)


def test_zoom_hysteresis_ignores_flicker_between_two_zooms() -> None:
    fps = 25.0
    # Required zoom flips between 1.4 and 1.8 every 0.5 s for 30 s.
    t = np.arange(int(30 * fps)) / fps
    required = np.where((t // 0.5) % 2 == 0, 1.4, 1.8)
    setpoint, changes = _zoom_hysteresis(required, fps, 0.12, 1.0, 4.0)
    assert changes <= 1
    assert len(np.unique(np.round(setpoint, 6))) <= 2

    # A sustained change (> 12 % for > 1 s) is followed exactly once, from
    # its onset, and short blips during the dwell are ignored.
    required = np.full(len(t), 1.8)
    required[250:] = 1.3
    required[300:310] = 1.8
    setpoint, changes = _zoom_hysteresis(required, fps, 0.12, 1.0, 4.0)
    assert changes == 1
    assert setpoint[249] == pytest.approx(1.8) and setpoint[250] == pytest.approx(1.3)
    assert setpoint[305] == pytest.approx(1.3)


def test_plan_zoom_does_not_pump_with_ball_speed() -> None:
    geometry = _geometry()
    hz = 15.0
    # Ball alternates fast/slow every 0.6 s: v1 pumped the zoom with speed.
    rows, x, direction = [], 400.0, 1.0
    for i in range(int(12 * hz)):
        t = i / hz
        speed = 1400.0 if int(t / 0.6) % 2 == 0 else 80.0
        x += direction * speed / hz
        if x > 1500 or x < 400:
            direction *= -1.0
        rows.append((t, x, 540.0))
    track = build_ball_track(rows, FRAME)
    segments = [GameStateSegment(0.0, 12.0, STATE_IN_PLAY, reason="in play")]
    plan = plan_camera(
        ball_track=track, player_positions=None, geometry=geometry, segments=segments,
        start_seconds=0.0, end_seconds=12.0, fps=25.0, frame_size=FRAME, base_zoom=1.8,
    )
    quality = compute_plan_quality(plan, ball_track=track, segments=segments)
    assert quality["zoom_rate_p95_per_s"] < 0.15
    assert quality["zoom_reversals_per_min"] <= 10.0
    assert quality["hard_snaps"] == 0


def test_state_transition_is_eased_not_jumped() -> None:
    geometry = _geometry()
    # Ball on the left wing, then a restart at the right goal.
    track = build_ball_track([(i / 15.0, 500.0, 540.0) for i in range(int(3 * 15))], FRAME)
    segments = [
        GameStateSegment(0.0, 3.0, STATE_IN_PLAY, reason="in play"),
        GameStateSegment(3.0, 12.0, "restart_right", side="right", reason="ball out over the right goal line"),
    ]
    fps = 25.0
    plan = plan_camera(
        ball_track=track, player_positions=None, geometry=geometry, segments=segments,
        start_seconds=0.0, end_seconds=12.0, fps=fps, frame_size=FRAME, base_zoom=1.6,
    )
    xs = np.array([d.center_x for d in plan.decisions])
    steps = np.abs(np.diff(xs))
    crop_w = FRAME[0] / np.array([d.zoom for d in plan.decisions])
    # No frame-to-frame step above the pan speed limit (1.0 crop widths/s).
    assert (steps * fps <= 1.0 * crop_w[:-1] * 1.02).all()
    # The move takes time (eased + smoothed), and ends at the goal side.
    i_start = int(np.argmax(np.abs(xs - xs[0]) > 5.0))
    i_end = int(np.argmax(np.abs(xs - xs[-1]) < 5.0))
    assert (i_end - i_start) / fps >= 1.0
    assert xs[-1] > 1100.0
    assert not plan.cuts


def test_hard_cut_only_after_long_ball_loss_and_far_jump() -> None:
    geometry = _geometry()
    hz = 15.0
    rows = [(i / hz, 300.0, 540.0) for i in range(int(2 * hz))]
    rows += [(5.0 + i / hz, 1650.0, 540.0) for i in range(int(3 * hz))]  # lost 3 s, far side
    track = build_ball_track(rows, FRAME)
    segments = [GameStateSegment(0.0, 8.0, STATE_IN_PLAY, reason="in play")]
    # While the ball is lost the camera follows the player cluster on the
    # left wing; the ball then reappears on the far right.
    rng = np.random.default_rng(2)
    t_s = np.repeat(np.arange(0.0, 8.0, 0.1), 6)
    players = np.stack([t_s, rng.normal(320.0, 30.0, len(t_s)), rng.normal(540.0, 40.0, len(t_s))], axis=1)
    plan = plan_camera(
        ball_track=track, player_positions=players, geometry=geometry, segments=segments,
        start_seconds=0.0, end_seconds=8.0, fps=25.0, frame_size=FRAME, base_zoom=1.6,
    )
    assert len(plan.cuts) == 1
    cut = plan.decisions[plan.cuts[0]]
    assert cut.focus == "cut" and "hard cut" in cut.reason
    assert 4.8 <= cut.t <= 5.2  # ball counts as detected within 0.14 s of a sample
    quality = compute_plan_quality(plan, ball_track=track, segments=segments)
    assert quality["cut_count"] == 1
    assert quality["hard_snaps"] == 0  # the deliberate cut is not a snap

    # Same jump after only a 1.2 s loss: eased, no cut.
    rows = [(i / hz, 300.0, 540.0) for i in range(int(2 * hz))]
    rows += [(3.2 + i / hz, 1650.0, 540.0) for i in range(int(3 * hz))]
    track = build_ball_track(rows, FRAME)
    plan = plan_camera(
        ball_track=track, player_positions=players, geometry=geometry, segments=segments,
        start_seconds=0.0, end_seconds=8.0, fps=25.0, frame_size=FRAME, base_zoom=1.6,
    )
    assert not plan.cuts
    assert compute_plan_quality(plan, ball_track=track, segments=segments)["hard_snaps"] == 0


def test_crop_size_does_not_flicker_around_rounding_boundary() -> None:
    n = 200
    # Float crop width hovers around 1201 (between even integers 1200/1202).
    widths = 1201.0 + 0.9 * np.sin(np.arange(n) * 1.7)
    zooms = 1920.0 / widths
    rects = crop_rects_from_centers(np.full(n, 960.0), np.full(n, 540.0), zooms, FRAME, FRAME)
    assert len(np.unique(rects[:, 2])) == 1
    assert len(np.unique(rects[:, 3])) == 1
    # A real zoom change still moves the size.
    zooms = np.linspace(1.0, 2.0, n)
    rects = crop_rects_from_centers(np.full(n, 960.0), np.full(n, 540.0), zooms, FRAME, FRAME)
    assert rects[0, 2] == 1920 and rects[-1, 2] == 960
    assert (np.diff(rects[:, 2]) <= 0).all()


def test_write_sendcmd_syntax_and_only_changes(tmp_path) -> None:
    geometry = _geometry()
    plan = plan_camera(
        ball_track=_moving_ball(), player_positions=None, geometry=geometry,
        segments=[GameStateSegment(0.0, 6.0, STATE_IN_PLAY, reason="test")],
        start_seconds=0.0, end_seconds=6.0, fps=25.0, frame_size=FRAME, base_zoom=1.6,
    )
    path = tmp_path / "camera_crops.txt"
    plan.write_sendcmd(str(path))
    lines = path.read_text().splitlines()
    pattern = re.compile(
        r"^(\d+\.\d{4}) crop w (\d+), crop h (\d+), crop x (\d+), crop y (\d+);$"
    )
    assert lines and len(lines) <= len(plan)
    times, rects = [], []
    for line in lines:
        m = pattern.match(line)
        assert m, line
        times.append(float(m.group(1)))
        rects.append(tuple(int(v) for v in m.groups()[1:]))
    assert times[0] == 0.0
    assert all(b > a for a, b in zip(times, times[1:]))
    assert all(v % 2 == 0 for r in rects for v in r)
    assert all(a != b for a, b in zip(rects, rects[1:])), "unchanged rects must not be repeated"
    # Each command lands half a frame before its frame.
    first_change = next(i for i in range(1, len(plan)) if tuple(plan.crop_rects[i]) != tuple(plan.crop_rects[i - 1]))
    assert times[1] == pytest.approx((first_change - 0.5) / plan.fps, abs=1e-4)


def test_focus_track_follows_selected_player() -> None:
    from backend.services.tracking_types import BallDetections, TrackingResult, player_track_from_rows

    geometry = _geometry()
    fps = 25.0
    rows = [(i / fps, 400 + 60 * i / fps - 10, 500, 400 + 60 * i / fps + 10, 560, 1.0) for i in range(int(10 * fps))]
    other = [(i / fps, 1500.0, 300.0, 1520.0, 360.0, 1.0) for i in range(int(10 * fps))]
    tracking = TrackingResult(
        fps=fps, frame_size=FRAME, duration_s=10.0,
        players={7: player_track_from_rows(7, rows, team=0), 8: player_track_from_rows(8, other, team=1)},
        ball=BallDetections.empty(),
    )
    track = build_ball_track([(i / 15.0, 1500.0, 330.0) for i in range(150)], FRAME)  # ball far away
    segments = [GameStateSegment(0.0, 10.0, STATE_IN_PLAY, reason="in play")]
    plan = plan_camera(
        ball_track=track, player_positions=tracking.all_player_positions(), geometry=geometry,
        segments=segments, start_seconds=0.0, end_seconds=10.0, fps=fps, frame_size=FRAME,
        base_zoom=1.6, player_tracks=tracking, focus_track_id=7,
    )
    late = [d for d in plan.decisions if d.t >= 2.0]
    assert all(d.focus == "player" for d in late)
    for d in late:
        px = 400 + 60 * d.t
        half_w = FRAME[0] / d.zoom / 2.0
        assert abs(px - d.center_x) < half_w * 0.9, "focus player left the frame"


def test_slice_plan_keeps_crop_rects_aligned() -> None:
    geometry = _geometry()
    plan = plan_camera(
        ball_track=_moving_ball(), player_positions=None, geometry=geometry,
        segments=[GameStateSegment(0.0, 6.0, STATE_IN_PLAY, reason="test")],
        start_seconds=0.0, end_seconds=6.0, fps=10.0, frame_size=FRAME, base_zoom=1.6,
    )
    sub = slice_plan(plan, 2.0, 4.0)
    assert isinstance(sub, CameraPlan)
    assert sub.crop_rects is not None and len(sub.crop_rects) == len(sub) == 20
    assert (sub.crop_rects == plan.crop_rects[20:40]).all()
    assert sub.output_size == plan.output_size
