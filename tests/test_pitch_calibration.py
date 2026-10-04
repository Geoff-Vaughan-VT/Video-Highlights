from __future__ import annotations

import json

import numpy as np
import pytest

from backend.services.game_tracking import build_ball_track, detect_goal_candidates, detect_goal_events, estimate_field_geometry
from backend.services.pitch_calibration import (
    PitchCalibration,
    calibrate_auto,
    calibrate_from_corners,
    calibrate_from_geometry,
    estimate_pitch_bounds,
    foot_positions,
)
from backend.services.tracking_types import TEAM_REFEREE, TrackingResult, player_track_from_rows, BallDetections

pytest.importorskip("cv2")

# A side-on view: the far touchline (top) is shorter than the near one.
CORNERS = [[420.0, 210.0], [1500.0, 210.0], [1860.0, 900.0], [60.0, 900.0]]


def test_manual_corners_map_to_pitch_corners_and_roundtrip() -> None:
    cal = calibrate_from_corners(CORNERS)
    assert cal.source == "manual"
    assert cal.confidence == pytest.approx(0.95)
    mapped = cal.to_pitch(np.asarray(CORNERS))
    expected = [[-52.5, -34.0], [52.5, -34.0], [52.5, 34.0], [-52.5, 34.0]]
    assert np.allclose(mapped, expected, atol=1e-3)

    rng = np.random.default_rng(0)
    pts_m = np.column_stack([rng.uniform(-52.5, 52.5, 200), rng.uniform(-34, 34, 200)])
    back = cal.to_pitch(cal.to_image(pts_m))
    assert np.allclose(back, pts_m, atol=1e-6)
    px = np.column_stack([rng.uniform(100, 1800, 200), rng.uniform(250, 880, 200)])
    assert np.allclose(cal.to_image(cal.to_pitch(px)), px, atol=1e-6)
    # Single points work too.
    assert cal.to_pitch(np.asarray(CORNERS[0])).shape == (2,)


def test_perspective_far_side_metres_are_larger_per_pixel() -> None:
    cal = calibrate_from_corners(CORNERS)
    far = cal.to_pitch([[900.0, 230.0], [1000.0, 230.0]])
    near = cal.to_pitch([[900.0, 880.0], [1000.0, 880.0]])
    assert abs(far[1, 0] - far[0, 0]) > abs(near[1, 0] - near[0, 0])


def test_normalized_corners_and_invalid_corners() -> None:
    norm = [[x / 1920.0, y / 1080.0] for x, y in CORNERS]
    cal = calibrate_from_corners(norm, frame_size=(1920, 1080))
    assert np.allclose(cal.image_corners_px, CORNERS)
    with pytest.raises(ValueError):
        calibrate_from_corners(CORNERS[:3])
    with pytest.raises(ValueError):  # self-intersecting (TL, BR, TR, BL)
        calibrate_from_corners([CORNERS[0], CORNERS[2], CORNERS[1], CORNERS[3]])


def test_auto_calibration_maps_field_bounds_to_pitch_extent() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    cloud = np.stack([rng.uniform(0, 60, n), rng.uniform(100, 1820, n), rng.uniform(200, 880, n)], axis=1)
    geo = estimate_field_geometry(cloud, (1920, 1080))
    cal = calibrate_from_geometry(geo, (1920, 1080))
    assert cal.source == "auto"
    assert 0.3 <= cal.confidence <= 0.7
    # Near touchline spans the full estimated width.
    near = cal.to_pitch([[geo.x_min, geo.y_max], [geo.x_max, geo.y_max]])
    assert np.allclose(near, [[-52.5, 34.0], [52.5, 34.0]], atol=1e-3)
    # Image corners (trapezoid; far line 85% of the near line) -> pitch corners.
    assert np.allclose(cal.to_pitch(cal.image_corners_px),
                       [[-52.5, -34.0], [52.5, -34.0], [52.5, 34.0], [-52.5, 34.0]], atol=1e-3)
    far_len = cal.image_corners_px[1, 0] - cal.image_corners_px[0, 0]
    assert far_len == pytest.approx(0.85 * (geo.x_max - geo.x_min), rel=1e-6)
    # Mid-field on the near touchline is the halfway line.
    mid = cal.to_pitch([(geo.x_min + geo.x_max) / 2.0, geo.y_max])
    assert mid[0] == pytest.approx(0.0, abs=1e-3)

    # A plain rectangle (top-down camera): every bound maps to +-52.5 / +-34.
    flat = calibrate_from_geometry(geo, (1920, 1080), far_line_ratio=1.0)
    box = flat.to_pitch([[geo.x_min, geo.y_min], [geo.x_max, geo.y_min], [geo.x_max, geo.y_max],
                         [geo.x_min, geo.y_max]])
    assert np.allclose(box, [[-52.5, -34.0], [52.5, -34.0], [52.5, 34.0], [-52.5, 34.0]], atol=1e-3)


def test_auto_calibration_without_geometry_uses_frame_default() -> None:
    cal = calibrate_from_geometry(None, (1280, 720))
    assert cal.confidence < 0.5
    assert cal.notes["geometry_source"] == "frame_default"


def test_thirds_in_bounds_and_dict_roundtrip() -> None:
    cal = calibrate_from_corners(CORNERS)
    x = np.asarray([-40.0, 0.0, 40.0])
    assert cal.thirds(x, "right").tolist() == [0, 1, 2]
    assert cal.thirds(x, -1).tolist() == [2, 1, 0]
    assert cal.in_bounds([[0.0, 0.0], [60.0, 0.0], [0.0, -40.0]]).tolist() == [True, False, False]
    assert cal.in_bounds([53.0, 0.0], margin_m=1.0) is True

    doc = json.loads(json.dumps(cal.to_dict()))
    assert set(["source", "confidence", "pitch_length_m", "pitch_width_m", "image_corners_px",
                "homography"]).issubset(doc)
    back = PitchCalibration.from_dict(doc)
    assert np.allclose(back.homography, cal.homography)
    doc.pop("homography")
    rebuilt = PitchCalibration.from_dict(doc)
    assert np.allclose(rebuilt.to_pitch(CORNERS[2]), [52.5, 34.0], atol=1e-3)


def test_points_beyond_the_horizon_are_nan() -> None:
    cal = calibrate_from_corners(CORNERS)
    out = cal.to_pitch([[960.0, -5000.0]])
    assert np.all(np.isnan(out))


def _tracking_with_referee() -> TrackingResult:
    rows_a = [(i * 0.1, 100 + i, 300, 120 + i, 360) for i in range(100)]
    rows_r = [(i * 0.1, 900, 300, 920, 360) for i in range(100)]
    players = {
        1: player_track_from_rows(1, rows_a, team=0),
        2: player_track_from_rows(2, rows_r, team=TEAM_REFEREE),
    }
    return TrackingResult(fps=10.0, frame_size=(1280, 720), duration_s=10.0, players=players,
                          ball=BallDetections.empty())


def test_foot_positions_use_box_bottom_and_skip_referee() -> None:
    feet = foot_positions(_tracking_with_referee())
    assert feet.shape[1] == 3
    assert np.allclose(feet[:, 2], 360.0)
    assert np.all(feet[:, 1] < 900)


def test_calibrate_auto_widens_bounds_with_the_ball() -> None:
    tracking = _tracking_with_referee()
    ball = build_ball_track([(i * 0.1, 80.0 + i * 11.0, 200.0 + (i % 7) * 50.0) for i in range(100)], (1280, 720))
    cal = calibrate_auto(tracking, ball, None, far_line_ratio=1.0)
    assert "ball" in cal.notes["geometry_source"]
    # The ball reached x~80 and x~1170: those are near the goal lines now.
    left = cal.to_pitch([90.0, 360.0])
    right = cal.to_pitch([1160.0, 360.0])
    assert left[0] < -50.0 and right[0] > 50.0


def test_goal_line_model_projects_goal_mouths_onto_the_observed_goal_lines() -> None:
    """With a centre-spot height the x bounds are the goal lines at the goal
    mouths, whatever the assumed camera tilt (far-line ratio)."""
    from types import SimpleNamespace

    geo = SimpleNamespace(x_min=100.0, x_max=1800.0, y_min=250.0, y_max=850.0, source="estimated",
                          centre_y=520.0)
    for ratio in (1.0, 0.85, 0.6):
        cal = calibrate_from_geometry(geo, (1920, 1080), far_line_ratio=ratio)
        assert cal.notes["model"] == "goal_lines_at_centre_y"
        goals = cal.to_image(np.array([[-52.5, 0.0], [52.5, 0.0], [0.0, 0.0]]))
        assert np.allclose(goals, [[100.0, 520.0], [1800.0, 520.0], [950.0, 520.0]], atol=1e-3)
        # The observed y bounds stay on the pitch.
        top = cal.to_pitch([950.0, 250.0])
        bottom = cal.to_pitch([950.0, 850.0])
        assert top[1] >= -34.0 - 1e-6 and bottom[1] <= 34.0 + 1e-6
    # Without centre_y: the legacy near-touchline model.
    legacy = calibrate_from_geometry(SimpleNamespace(x_min=100.0, x_max=1800.0, y_min=250.0, y_max=850.0,
                                                     source="estimated"), (1920, 1080))
    assert "model" not in legacy.notes
    assert np.allclose(legacy.to_pitch([[100.0, 850.0], [1800.0, 850.0]]), [[-52.5, 34.0], [52.5, 34.0]], atol=1e-3)


def test_user_goal_boxes_are_the_length_bounds() -> None:
    tracking = _tracking_with_referee()
    geo = estimate_field_geometry(None, (1280, 720), goal_box_left={"x1": 40, "y1": 330, "x2": 70, "y2": 390},
                                  goal_box_right={"x1": 1210, "y1": 330, "x2": 1240, "y2": 390})
    bounds = estimate_pitch_bounds(tracking, None, None, field_geometry=geo)
    assert (bounds.x_min, bounds.x_max) == (70.0, 1210.0)
    assert bounds.centre_y == pytest.approx(360.0)
    assert "goal_lines" in bounds.source
    cal = calibrate_auto(tracking, None, None, field_geometry=geo)
    assert np.allclose(cal.to_image(np.array([[-52.5, 0.0], [52.5, 0.0]])), [[70.0, 360.0], [1210.0, 360.0]],
                       atol=1e-3)


@pytest.fixture(scope="module")
def synthetic_match(tmp_path_factory):
    from backend.services.synthetic_match import SyntheticMatchSpec, generate_synthetic_match

    gt = generate_synthetic_match(tmp_path_factory.mktemp("autocal") / "m.mp4", SyntheticMatchSpec(duration_s=20.0))
    return gt, build_ball_track(gt.tracking.ball.to_samples(), gt.tracking.frame_size)


def test_auto_calibration_finds_both_synthetic_goals_without_corners(synthetic_match) -> None:
    """No manual corners, no goal boxes, no player tracks for goal detection:
    tracking -> calibrate_auto -> calibrated goal mouths -> both scripted goals."""
    gt, ball = synthetic_match
    tr = gt.tracking
    player_geo = estimate_field_geometry(tr.all_player_positions(), tr.frame_size)
    cal = calibrate_auto(tr, ball, None, field_geometry=player_geo)  # as the pipeline calls it
    assert cal.source == "auto" and cal.confidence == pytest.approx(0.5)
    assert "centre_spot" in cal.notes["geometry_source"]
    assert cal.notes["ball_samples_at_rest_beyond_players"] > 0  # ball-in-net samples were excluded

    x0, y0, x1, y1 = gt.pitch_bounds_px
    px_per_m = (x1 - x0) / 105.0
    centre = cal.to_image(np.array([0.0, 0.0]))
    assert centre[1] == pytest.approx((y0 + y1) / 2.0, abs=2.0)  # centre spot from the kickoffs
    goal_lines = cal.to_image(np.array([[-52.5, 0.0], [52.5, 0.0]]))
    # Goal lines from the last in-play ball samples: at most ~2.5 m short.
    assert 0.0 <= goal_lines[0, 0] - x0 <= 2.5 * px_per_m
    assert 0.0 <= x1 - goal_lines[1, 0] <= 2.5 * px_per_m

    geometry = estimate_field_geometry(tr.all_player_positions(), tr.frame_size, calibration=cal)
    assert "calibration_auto" in geometry.source
    for side, (bx1, by1, bx2, by2) in gt.goal_boxes_px.items():
        goal = geometry.goal_for_side(side)
        # The calibrated mouth covers the real one up to the vertical scale error.
        assert goal.y1 <= (by1 + by2) / 2.0 <= goal.y2
        assert goal.y2 - goal.y1 >= 0.7 * (by2 - by1)

    goals = detect_goal_events(ball, geometry, 0.0, tr.duration_s)
    truth = sorted(gt.goal_times_s)
    assert len(truth) == 2
    assert [g.side for g in goals] == [side for _t, side in truth]
    for goal, (t_true, _side) in zip(goals, truth):
        assert abs(goal.t - t_true) <= 0.5
        assert goal.confidence >= 0.7
    # No false goals, and nothing else even looks like one.
    candidates = detect_goal_candidates(ball, geometry, 0.0, tr.duration_s)
    assert len(candidates) == 2 and all(c.verdict == "goal" for c in candidates)
