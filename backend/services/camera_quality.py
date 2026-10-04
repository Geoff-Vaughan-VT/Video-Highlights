"""Smoothness / watchability metrics for a :class:`CameraPlan`.

``camera_quality.json`` (docs/ARTIFACTS.md) is what the tests and the Studio
use to decide whether a plan is comfortable to watch. Thresholds (PLAN.md
acceptance criteria): pan speed p95 < 0.9 crop-widths/s, zoom rate p95 <
0.15x/s, zero hard snaps, ball in frame > 97 % of in-play frames.

Speeds/accelerations are measured on the planner's float camera path (the
intent; the even-integer crop quantisation would otherwise dominate the
second difference), hard snaps on the integer crop rectangles that are
actually rendered.
"""

from __future__ import annotations

import json
from typing import Dict, Optional, Sequence

import numpy as np

from .camera_planner import CameraPlan, base_crop_size
from .game_tracking import STATE_IN_PLAY, BallTrack, GameStateSegment

# Frame-to-frame center jump (fraction of crop width) that counts as a snap.
HARD_SNAP_CROP_FRAC = 0.25
# Zoom velocity below this (x/s) is "not moving" for reversal counting.
ZOOM_REVERSAL_MIN_RATE = 0.01

THRESHOLDS = {
    "pan_speed_p95_cropw_per_s": 0.9,
    "zoom_rate_p95_per_s": 0.15,
    "hard_snaps": 0,
    "ball_in_frame_fraction": 0.97,
}


def _p95(values: np.ndarray) -> float:
    return float(np.percentile(values, 95)) if len(values) else 0.0


def _in_play_mask(plan: CameraPlan, segments: Optional[Sequence[GameStateSegment]]) -> np.ndarray:
    times = np.array([d.t for d in plan.decisions], dtype=np.float64)
    if not segments:
        return np.array([d.state == STATE_IN_PLAY for d in plan.decisions], dtype=bool)
    mask = np.zeros(len(times), dtype=bool)
    for seg in segments:
        if seg.state == STATE_IN_PLAY:
            mask |= (times >= seg.start_s) & (times < seg.end_s)
    return mask


def compute_plan_quality(
    plan: CameraPlan,
    ball_track: Optional[BallTrack] = None,
    segments: Optional[Sequence[GameStateSegment]] = None,
) -> Dict[str, object]:
    """Smoothness metrics for ``plan`` (see module docstring)."""
    n = len(plan.decisions)
    out_size = plan.resolved_output_size()
    base_w, _ = base_crop_size(plan.frame_size, out_size)
    result: Dict[str, object] = {
        "frames": n,
        "fps": round(float(plan.fps), 3),
        "duration_s": round(n / plan.fps, 3) if plan.fps > 0 else 0.0,
        "output_size": list(out_size),
        "max_zoom_bound": round(float(plan.max_zoom), 3) if plan.max_zoom is not None else None,
    }
    if n < 3 or plan.fps <= 0:
        result.update({
            "pan_speed_p95_cropw_per_s": 0.0, "pan_speed_max_cropw_per_s": 0.0,
            "pan_accel_p95": 0.0, "zoom_rate_p95_per_s": 0.0, "zoom_rate_max_per_s": 0.0,
            "zoom_reversals_per_min": 0.0, "hard_snaps": 0, "ball_in_frame_fraction": 1.0,
            "cut_count": len(plan.cuts), "zoom_min": None, "zoom_max": None, "zoom_mean": None,
        })
        return result

    dt = 1.0 / plan.fps
    cx = np.array([d.center_x for d in plan.decisions], dtype=np.float64)
    cy = np.array([d.center_y for d in plan.decisions], dtype=np.float64)
    zoom = np.array([d.zoom for d in plan.decisions], dtype=np.float64)
    crop_w = base_w / np.maximum(zoom, 1e-6)

    cut_frames = set(int(c) for c in plan.cuts)
    cut_frames.update(i for i, d in enumerate(plan.decisions) if d.focus == "cut")
    # Velocity sample i covers frames i -> i+1; drop samples touching a cut.
    vel_ok = np.array([(i + 1) not in cut_frames for i in range(n - 1)], dtype=bool)
    speed = np.hypot(np.diff(cx), np.diff(cy)) / dt / crop_w[:-1]
    speed_ok = speed[vel_ok]
    acc_ok = vel_ok[:-1] & vel_ok[1:]
    accel_vec = np.hypot(np.diff(cx, 2), np.diff(cy, 2)) / (dt * dt) / crop_w[:-2]
    accel_ok = accel_vec[acc_ok]

    zoom_rate = np.diff(zoom) / dt
    zr_ok = np.abs(zoom_rate[vel_ok])
    # Reversals: direction changes of a zoom that is actually moving.
    signs = np.sign(np.where(np.abs(zoom_rate) >= ZOOM_REVERSAL_MIN_RATE, zoom_rate, 0.0))
    moving = signs[signs != 0]
    reversals = int(np.count_nonzero(np.diff(moving) != 0)) if len(moving) > 1 else 0
    minutes = max(1e-9, n * dt / 60.0)

    rects = plan.get_crop_rects()
    rcx = rects[:, 0] + rects[:, 2] / 2.0
    rcy = rects[:, 1] + rects[:, 3] / 2.0
    jumps = np.hypot(np.diff(rcx), np.diff(rcy)) / np.maximum(rects[:-1, 2], 1)
    snaps = int(np.count_nonzero((jumps > HARD_SNAP_CROP_FRAC) & vel_ok))

    in_play = _in_play_mask(plan, segments)
    inside = 0
    counted = 0
    for i, d in enumerate(plan.decisions):
        if not in_play[i]:
            continue
        if ball_track is not None:
            pos = ball_track.position_at(d.t)
            bx, by = (pos[0], pos[1]) if pos is not None else (None, None)
        else:
            bx, by = d.ball_x, d.ball_y
        if bx is None or by is None:
            continue
        counted += 1
        x, y, w, h = (int(v) for v in rects[i])
        if x <= bx <= x + w and y <= by <= y + h:
            inside += 1

    result.update({
        "pan_speed_p95_cropw_per_s": round(_p95(speed_ok), 4),
        "pan_speed_max_cropw_per_s": round(float(speed_ok.max()) if len(speed_ok) else 0.0, 4),
        "pan_accel_p95": round(_p95(accel_ok), 4),
        "zoom_rate_p95_per_s": round(_p95(zr_ok), 4),
        "zoom_rate_max_per_s": round(float(zr_ok.max()) if len(zr_ok) else 0.0, 4),
        "zoom_reversals_per_min": round(reversals / minutes, 3),
        "hard_snaps": snaps,
        "ball_in_frame_fraction": round(inside / counted, 4) if counted else 1.0,
        "ball_frames_counted": counted,
        "cut_count": len(cut_frames),
        "zoom_min": round(float(zoom.min()), 3),
        "zoom_max": round(float(zoom.max()), 3),
        "zoom_mean": round(float(zoom.mean()), 3),
    })
    result["passes_thresholds"] = bool(
        result["pan_speed_p95_cropw_per_s"] < THRESHOLDS["pan_speed_p95_cropw_per_s"]
        and result["zoom_rate_p95_per_s"] < THRESHOLDS["zoom_rate_p95_per_s"]
        and result["hard_snaps"] == THRESHOLDS["hard_snaps"]
        and result["ball_in_frame_fraction"] > THRESHOLDS["ball_in_frame_fraction"]
    )
    return result


def write_plan_quality(
    plan: CameraPlan,
    path: str,
    ball_track: Optional[BallTrack] = None,
    segments: Optional[Sequence[GameStateSegment]] = None,
) -> str:
    """Write ``camera_quality.json`` and return its path."""
    metrics = compute_plan_quality(plan, ball_track=ball_track, segments=segments)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return path
