from __future__ import annotations

import json

import numpy as np
import pytest

from backend.services.camera_planner import CameraDecision, CameraPlan, plan_camera
from backend.services.camera_quality import THRESHOLDS, compute_plan_quality, write_plan_quality
from backend.services.game_tracking import (
    analyze_game_states,
    build_ball_track,
    detect_goal_events,
    estimate_field_geometry,
)

pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    from backend.services.synthetic_match import generate_synthetic_match

    out = tmp_path_factory.mktemp("synthetic") / "match.mp4"
    gt = generate_synthetic_match(out)
    tracking = gt.tracking
    frame_size = tracking.frame_size
    ball = build_ball_track(tracking.ball.to_samples(), frame_size)
    positions = tracking.all_player_positions()
    geometry = estimate_field_geometry(positions, frame_size)
    goals = detect_goal_events(ball, geometry, 0.0, tracking.duration_s)
    segments = analyze_game_states(ball, geometry, 0.0, tracking.duration_s, goal_events=goals)
    return {"gt": gt, "ball": ball, "positions": positions, "geometry": geometry, "segments": segments}


def _plan(synthetic, **kwargs):
    tracking = synthetic["gt"].tracking
    return plan_camera(
        ball_track=synthetic["ball"],
        player_positions=synthetic["positions"],
        geometry=synthetic["geometry"],
        segments=synthetic["segments"],
        start_seconds=0.0,
        end_seconds=tracking.duration_s,
        fps=tracking.fps,
        frame_size=tracking.frame_size,
        base_zoom=1.8,
        # 1280x720 -> 640x360 mirrors 4K -> 1080p: zoom bound 2.0.
        output_size=(640, 360),
        **kwargs,
    )


@pytest.mark.parametrize(
    "variant",
    [
        {},
        {"use_tracks": True},
        {"use_tracks": True, "style": "tight"},
        {"use_tracks": True, "focus_track_id": 3},
    ],
    ids=["ball", "team_shape", "tight", "focus_player"],
)
def test_synthetic_match_plan_meets_smoothness_thresholds(synthetic, variant) -> None:
    kwargs = dict(variant)
    if kwargs.pop("use_tracks", False):
        kwargs["player_tracks"] = synthetic["gt"].tracking
    plan = _plan(synthetic, **kwargs)
    quality = compute_plan_quality(plan, ball_track=synthetic["ball"], segments=synthetic["segments"])

    assert quality["pan_speed_p95_cropw_per_s"] < THRESHOLDS["pan_speed_p95_cropw_per_s"]
    assert quality["pan_speed_p95_cropw_per_s"] <= 0.8
    assert quality["pan_accel_p95"] <= 1.2
    assert quality["zoom_rate_p95_per_s"] < THRESHOLDS["zoom_rate_p95_per_s"]
    assert quality["hard_snaps"] == 0
    assert quality["ball_in_frame_fraction"] > 0.97
    assert quality["ball_frames_counted"] > 100
    assert quality["passes_thresholds"] is True
    # Zoom bounded so no output pixel is upscaled; never wider than the frame.
    assert plan.max_zoom == pytest.approx(2.0)
    assert 1.0 <= quality["zoom_min"] <= quality["zoom_max"] <= 2.0 + 1e-9
    assert quality["cut_count"] == 0


def test_quality_counts_snaps_but_not_recorded_cuts() -> None:
    plan = CameraPlan(start_seconds=0.0, fps=25.0, frame_size=(1920, 1080), base_zoom=1.5)
    for i in range(100):
        x = 600.0 if i < 50 else 1300.0  # 700 px jump (> 25 % of a 1280 crop)
        plan.decisions.append(CameraDecision(index=i, t=i / 25.0, center_x=x, center_y=540.0, zoom=1.5,
                                             state="in_play", focus="ball", reason="t", confidence=1.0,
                                             ball_x=x, ball_y=540.0))
    quality = compute_plan_quality(plan)
    assert quality["hard_snaps"] == 1
    assert quality["passes_thresholds"] is False

    plan.cuts = [50]
    plan.decisions[50].focus = "cut"
    quality = compute_plan_quality(plan)
    assert quality["hard_snaps"] == 0
    assert quality["cut_count"] == 1
    assert quality["pan_speed_p95_cropw_per_s"] == 0.0
    assert quality["ball_in_frame_fraction"] == 1.0


def test_quality_reports_ball_out_of_frame_and_zoom_reversals() -> None:
    plan = CameraPlan(start_seconds=0.0, fps=10.0, frame_size=(1920, 1080), base_zoom=2.0)
    for i in range(120):
        zoom = 1.5 + 0.3 * np.sin(i / 3.0)  # pumping zoom
        plan.decisions.append(CameraDecision(index=i, t=i / 10.0, center_x=960.0, center_y=540.0, zoom=zoom,
                                             state="in_play", focus="ball", reason="t", confidence=1.0,
                                             ball_x=100.0 if i % 2 else 960.0, ball_y=540.0))
    quality = compute_plan_quality(plan)
    assert quality["ball_in_frame_fraction"] == pytest.approx(0.5, abs=0.01)
    assert quality["zoom_reversals_per_min"] > 20
    assert quality["zoom_rate_p95_per_s"] > 0.15


def test_write_plan_quality(tmp_path, synthetic) -> None:
    plan = _plan(synthetic)
    path = write_plan_quality(plan, str(tmp_path / "camera_quality.json"), ball_track=synthetic["ball"],
                              segments=synthetic["segments"])
    data = json.loads((tmp_path / "camera_quality.json").read_text())
    assert path.endswith("camera_quality.json")
    for key in ("pan_speed_p95_cropw_per_s", "pan_accel_p95", "zoom_rate_p95_per_s",
                "zoom_reversals_per_min", "hard_snaps", "ball_in_frame_fraction", "cut_count"):
        assert key in data
