"""Ball tracking, field geometry, and game-state analysis.

This module turns raw per-frame ball detections into a cleaned ball track,
estimates where the field and the two goals are inside the frame, and then
classifies the match timeline into game states:

* ``in_play``            - the ball is visible and inside the field of play.
* ``ball_lost``          - the ball is not visible and we have no strong reason
                           to believe it left the field (occlusion, missed
                           detections).
* ``restart_left/right`` - the ball went out over a goal line (or vanished next
                           to one). The game is waiting for a goal kick or a
                           corner, so the camera must stay at that goal.
* ``restart_touchline``  - the ball went out over a touchline (throw-in wait).
* ``goal_left/right``    - the ball entered the goal. These are also emitted
                           as explicit goal events so they can be flagged as
                           bookmarks.

Goal detection (v2, :func:`detect_goal_candidates`) demands corroboration:
a strong ball signal (crossing between the posts, ball arrested in the net,
ball vanishing into the mouth and staying gone) plus a second independent
signal (another ball signal, a kickoff with both teams in their halves, a
crowd-noise peak, the scoring team converging). A ball seen back in play far
from the goal within 3 s vetoes the goal. Uncorroborated candidates are
returned as ``shot``/``chance`` with confidence < 0.6.

Everything is heuristic but fully explainable: each segment and each goal
event carries a human-readable ``reason`` plus the raw evidence values, so the
output can be reviewed, debugged, and used as training data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("videohighlights.game_tracking")

BALL_SOURCE_DETECTED = "detected"
BALL_SOURCE_INTERPOLATED = "interpolated"

STATE_IN_PLAY = "in_play"
STATE_BALL_LOST = "ball_lost"
STATE_RESTART_LEFT = "restart_left"
STATE_RESTART_RIGHT = "restart_right"
STATE_RESTART_TOUCHLINE = "restart_touchline"
STATE_GOAL_LEFT = "goal_left"
STATE_GOAL_RIGHT = "goal_right"
STATE_CORNER_SETUP = "corner_kick_setup"
STATE_FREE_KICK_SETUP = "free_kick_setup"

RESTART_STATES = {STATE_RESTART_LEFT, STATE_RESTART_RIGHT, STATE_RESTART_TOUCHLINE}
GOAL_STATES = {STATE_GOAL_LEFT, STATE_GOAL_RIGHT}
SET_PIECE_STATES = {STATE_CORNER_SETUP, STATE_FREE_KICK_SETUP}


# ---------------------------------------------------------------------------
# Ball track building
# ---------------------------------------------------------------------------


@dataclass
class BallTrackConfig:
    """Tuning knobs for turning raw detections into a clean ball track."""

    # Hard physical gate: reject detections implying speeds above this many
    # frame-widths per second relative to the current filtered state.
    max_speed_frame_widths_per_s: float = 2.2
    # Base association gate in pixels (grows with elapsed time).
    base_gate_px: float = 80.0
    # Cap on the association gate as a fraction of the frame width. Without
    # a cap, the gate outgrows the frame after ~0.4s of detector dropout and
    # a single false positive (scoreboard, spare ball) teleports the track;
    # gaps longer than this are handled by the re-acquisition path instead.
    max_gate_frac: float = 0.35
    # A fresh/re-acquired track is confirmed once this many mutually
    # consistent detections land inside a short window.
    confirm_detections: int = 2
    confirm_window_s: float = 0.7
    # After this long without an accepted detection, a consistent cluster of
    # detections anywhere in the frame restarts the track (ball re-appeared).
    reacquire_after_s: float = 0.9
    # Gaps up to this long are bridged by interpolation; longer gaps mean the
    # ball is genuinely "not visible".
    max_interpolation_gap_s: float = 1.25
    # Alpha-beta filter gains.
    smoothing_alpha: float = 0.5
    smoothing_beta: float = 0.3
    # A query time counts as "detected" (vs interpolated) when a real sample
    # is within this distance in time.
    detected_time_tolerance_s: float = 0.14


@dataclass
class BallTrack:
    """Cleaned ball trajectory with interpolation-aware queries."""

    times: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    frame_size: Tuple[int, int]
    config: BallTrackConfig
    stats: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.times.shape[0])

    def position_at(self, t: float) -> Optional[Tuple[float, float, str]]:
        """Return (x, y, source) at time ``t`` or None when not visible.

        ``source`` is ``detected`` when a real detection is nearby in time and
        ``interpolated`` when the value is bridged across a short gap.
        """
        if len(self) == 0:
            return None
        times = self.times
        idx = int(np.searchsorted(times, t))
        tol = self.config.detected_time_tolerance_s
        if idx <= 0:
            gap = float(times[0] - t)
            if gap > tol:
                return None
            return float(self.xs[0]), float(self.ys[0]), BALL_SOURCE_DETECTED
        if idx >= len(times):
            gap = float(t - times[-1])
            if gap > tol:
                return None
            return float(self.xs[-1]), float(self.ys[-1]), BALL_SOURCE_DETECTED

        left_t = float(times[idx - 1])
        right_t = float(times[idx])
        nearest_gap = min(t - left_t, right_t - t)
        span = right_t - left_t
        if span > self.config.max_interpolation_gap_s and nearest_gap > tol:
            return None
        if span <= 1e-9:
            return float(self.xs[idx - 1]), float(self.ys[idx - 1]), BALL_SOURCE_DETECTED
        alpha = (t - left_t) / span
        x = float(self.xs[idx - 1] + (self.xs[idx] - self.xs[idx - 1]) * alpha)
        y = float(self.ys[idx - 1] + (self.ys[idx] - self.ys[idx - 1]) * alpha)
        source = BALL_SOURCE_DETECTED if nearest_gap <= tol else BALL_SOURCE_INTERPOLATED
        return x, y, source

    def velocity_at(self, t: float, window_s: float = 0.4) -> Tuple[float, float]:
        """Finite-difference velocity (px/s) from samples within ``t`` +/- ``window_s``.

        Uses the actual samples inside the window (not interpolated queries),
        so it stays accurate right up to the edge of a visible segment - e.g.
        the instant a shot disappears into the goal.
        """
        if len(self) < 2:
            return 0.0, 0.0
        # searchsorted keeps this O(log n); it is called once per planned
        # frame, so a linear scan here would make the planner O(n^2).
        first = int(np.searchsorted(self.times, t - window_s, side="left"))
        last = int(np.searchsorted(self.times, t + window_s, side="right")) - 1
        if last - first < 1:
            return 0.0, 0.0
        dt = float(self.times[last] - self.times[first])
        if dt <= 1e-6:
            return 0.0, 0.0
        return (
            float(self.xs[last] - self.xs[first]) / dt,
            float(self.ys[last] - self.ys[first]) / dt,
        )

    def visibility_gaps(self, start_s: float, end_s: float) -> List[Tuple[float, float]]:
        """Time ranges within [start_s, end_s] where the ball is not visible."""
        gaps: List[Tuple[float, float]] = []
        max_gap = self.config.max_interpolation_gap_s
        if len(self) == 0:
            return [(start_s, end_s)] if end_s > start_s else []
        if float(self.times[0]) - start_s > max_gap:
            gaps.append((start_s, float(self.times[0])))
        diffs = np.diff(self.times)
        for i in np.where(diffs > max_gap)[0]:
            gap_start = float(self.times[i])
            gap_end = float(self.times[i + 1])
            if gap_end > start_s and gap_start < end_s:
                gaps.append((max(gap_start, start_s), min(gap_end, end_s)))
        if end_s - float(self.times[-1]) > max_gap:
            gaps.append((float(self.times[-1]), end_s))
        return gaps

    def coverage_fraction(self, start_s: float, end_s: float, step_s: float = 0.25) -> float:
        if end_s <= start_s:
            return 0.0
        steps = max(1, int(math.ceil((end_s - start_s) / step_s)))
        visible = 0
        for i in range(steps):
            if self.position_at(start_s + i * step_s) is not None:
                visible += 1
        return visible / steps

    def to_rows(self) -> List[Dict[str, float]]:
        return [
            {"t": round(float(t), 3), "x": round(float(x), 2), "y": round(float(y), 2)}
            for t, x, y in zip(self.times, self.xs, self.ys)
        ]


def _extract_detection(item: object) -> Optional[Tuple[float, float, float]]:
    """Accept TrackPoint-like objects or (t, x, y[, ...]) tuples."""
    t = getattr(item, "t", None)
    xy = getattr(item, "xy", None)
    if t is not None and xy is not None:
        try:
            return float(t), float(xy[0]), float(xy[1])
        except Exception:
            return None
    try:
        seq = tuple(item)  # type: ignore[arg-type]
        return float(seq[0]), float(seq[1]), float(seq[2])
    except Exception:
        return None


def build_ball_track(
    raw_detections: Iterable[object],
    frame_size: Tuple[int, int],
    config: Optional[BallTrackConfig] = None,
) -> BallTrack:
    """Filter raw ball detections into a clean, physically plausible track.

    The filter keeps a constant-velocity state, associates the nearest
    detection per frame inside a speed-based gate, rejects teleporting
    outliers (scoreboards, bald heads, spare balls), and re-acquires the ball
    after long gaps once a consistent cluster of detections appears.
    """
    cfg = config or BallTrackConfig()
    frame_w = max(1, int(frame_size[0]))
    speed_limit = cfg.max_speed_frame_widths_per_s * frame_w

    detections: List[Tuple[float, float, float]] = []
    for item in raw_detections or []:
        parsed = _extract_detection(item)
        if parsed is not None:
            detections.append(parsed)
    detections.sort(key=lambda row: row[0])

    accepted: List[Tuple[float, float, float]] = []
    rejected = 0

    # Filter state.
    pos: Optional[Tuple[float, float]] = None
    vel = (0.0, 0.0)
    last_t: Optional[float] = None
    pending: List[Tuple[float, float, float]] = []

    def _pending_consistent() -> Optional[List[Tuple[float, float, float]]]:
        """Return a confirmable cluster from pending detections, if any."""
        if len(pending) < cfg.confirm_detections:
            return None
        window = [p for p in pending if pending[-1][0] - p[0] <= cfg.confirm_window_s]
        if len(window) < cfg.confirm_detections:
            return None
        for a, b in zip(window[:-1], window[1:]):
            dt = max(1e-3, b[0] - a[0])
            dist = math.hypot(b[1] - a[1], b[2] - a[2])
            if dist / dt > speed_limit:
                return None
        return window

    # Group detections that share (nearly) the same timestamp.
    groups: List[List[Tuple[float, float, float]]] = []
    for det in detections:
        if groups and abs(det[0] - groups[-1][0][0]) <= 1e-4:
            groups[-1].append(det)
        else:
            groups.append([det])

    for group in groups:
        t = group[0][0]
        if pos is None:
            pending.extend(group)
            pending = [p for p in pending if t - p[0] <= max(cfg.confirm_window_s, 1.0)]
            window = _pending_consistent()
            if window is not None:
                for row in window:
                    accepted.append(row)
                tail, prev = window[-1], window[-2] if len(window) >= 2 else window[-1]
                dt = max(1e-3, tail[0] - prev[0])
                pos = (tail[1], tail[2])
                vel = ((tail[1] - prev[1]) / dt, (tail[2] - prev[2]) / dt) if len(window) >= 2 else (0.0, 0.0)
                last_t = tail[0]
                pending = []
            continue

        assert last_t is not None
        dt = max(1e-3, t - last_t)
        # Prediction with capped extrapolation so long gaps don't fling the
        # predicted point off-screen.
        pred_dt = min(dt, 0.5)
        pred = (pos[0] + vel[0] * pred_dt, pos[1] + vel[1] * pred_dt)
        gate = min(cfg.base_gate_px + speed_limit * dt, cfg.max_gate_frac * frame_w)

        best = min(group, key=lambda row: math.hypot(row[1] - pred[0], row[2] - pred[1]))
        dist = math.hypot(best[1] - pred[0], best[2] - pred[1])

        if dist <= gate:
            residual = (best[1] - pred[0], best[2] - pred[1])
            new_x = pred[0] + cfg.smoothing_alpha * residual[0]
            new_y = pred[1] + cfg.smoothing_alpha * residual[1]
            vel = (
                vel[0] + cfg.smoothing_beta * residual[0] / dt,
                vel[1] + cfg.smoothing_beta * residual[1] / dt,
            )
            speed = math.hypot(*vel)
            if speed > speed_limit:
                scale = speed_limit / speed
                vel = (vel[0] * scale, vel[1] * scale)
            pos = (new_x, new_y)
            last_t = t
            accepted.append((t, new_x, new_y))
            pending = []
        else:
            rejected += len(group)
            pending.extend(group)
            pending = [p for p in pending if t - p[0] <= max(cfg.confirm_window_s, 1.0)]
            if t - last_t > cfg.reacquire_after_s:
                window = _pending_consistent()
                if window is not None:
                    LOGGER.debug(
                        "ball re-acquired at t=%.2fs (%.0f, %.0f) after %.2fs silence",
                        window[-1][0], window[-1][1], window[-1][2], t - last_t,
                    )
                    for row in window:
                        accepted.append(row)
                    tail, prev = window[-1], window[-2] if len(window) >= 2 else window[-1]
                    dtw = max(1e-3, tail[0] - prev[0])
                    pos = (tail[1], tail[2])
                    vel = ((tail[1] - prev[1]) / dtw, (tail[2] - prev[2]) / dtw) if len(window) >= 2 else (0.0, 0.0)
                    last_t = tail[0]
                    pending = []

    accepted.sort(key=lambda row: row[0])
    # Deduplicate identical timestamps (keep the filtered value emitted last).
    deduped: List[Tuple[float, float, float]] = []
    for row in accepted:
        if deduped and abs(row[0] - deduped[-1][0]) <= 1e-6:
            deduped[-1] = row
        else:
            deduped.append(row)

    arr = np.asarray(deduped, dtype=np.float64) if deduped else np.empty((0, 3), dtype=np.float64)
    track = BallTrack(
        times=arr[:, 0] if arr.size else np.empty(0),
        xs=arr[:, 1] if arr.size else np.empty(0),
        ys=arr[:, 2] if arr.size else np.empty(0),
        frame_size=(int(frame_size[0]), int(frame_size[1])),
        config=cfg,
        stats={
            "raw_detections": len(detections),
            "accepted": len(deduped),
            "rejected_outliers": rejected,
        },
    )
    LOGGER.info(
        "ball track built: %d raw detections -> %d accepted, %d rejected outliers",
        len(detections), len(deduped), rejected,
    )
    return track


# ---------------------------------------------------------------------------
# Field / goal geometry
# ---------------------------------------------------------------------------


@dataclass
class GoalBox:
    side: str  # "left" | "right"
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            self.x1 - margin <= x <= self.x2 + margin
            and self.y1 - margin <= y <= self.y2 + margin
        )

    def to_dict(self) -> Dict[str, float]:
        return {"side": self.side, "x1": round(self.x1, 1), "y1": round(self.y1, 1),
                "x2": round(self.x2, 1), "y2": round(self.y2, 1)}


@dataclass
class FieldGeometry:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    left_goal: GoalBox
    right_goal: GoalBox
    frame_size: Tuple[int, int]
    source: str = "estimated"

    @property
    def width(self) -> float:
        return max(1.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(1.0, self.y_max - self.y_min)

    def goal_for_side(self, side: str) -> GoalBox:
        return self.left_goal if side == "left" else self.right_goal

    def side_if_near_goal(self, x: float, y: float, near_frac: float) -> Optional[str]:
        """Which goal (if any) the point is close to, in field-width units."""
        near_px = near_frac * self.width
        for goal in (self.left_goal, self.right_goal):
            gx, gy = goal.center
            goal_half_h = max(abs(goal.y2 - goal.y1) / 2.0, self.height * 0.25)
            if abs(x - gx) <= near_px and abs(y - gy) <= goal_half_h + near_px:
                return goal.side
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "x_min": round(self.x_min, 1),
            "x_max": round(self.x_max, 1),
            "y_min": round(self.y_min, 1),
            "y_max": round(self.y_max, 1),
            "left_goal": self.left_goal.to_dict(),
            "right_goal": self.right_goal.to_dict(),
            "frame_width": int(self.frame_size[0]),
            "frame_height": int(self.frame_size[1]),
            "source": self.source,
        }


def _goal_box_from_override(raw: Optional[Dict[str, object]], side: str,
                            frame_size: Tuple[int, int]) -> Optional[GoalBox]:
    if not isinstance(raw, dict):
        return None
    try:
        x1, y1, x2, y2 = (float(raw["x1"]), float(raw["y1"]), float(raw["x2"]), float(raw["y2"]))
    except Exception:
        return None
    w, h = frame_size
    # Values <= 1.0 are treated as normalized coordinates.
    if max(x1, y1, x2, y2) <= 1.0:
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return GoalBox(side=side, x1=min(x1, x2), y1=min(y1, y2), x2=max(x1, x2), y2=max(y1, y2))


# Real goal: 7.32 m between the posts on a 68 m wide pitch (10.8 %).
GOAL_MOUTH_HALF_M = 3.66
# Uncalibrated goal mouth height as a fraction of the estimated field height
# (7.32 m / 68 m, rounded up a little for the estimate's uncertainty).
DEFAULT_GOAL_MOUTH_FRAC = 0.11
# Uncalibrated goal box depth behind the line (fraction of field width): the
# estimated goal line itself is uncertain by a few metres, so the box is
# deeper than a real net.
DEFAULT_GOAL_DEPTH_FRAC = 0.05
# Calibrated goal boxes: tolerance beyond each post and depth behind the line.
DEFAULT_GOAL_MARGIN_M = 1.0
DEFAULT_GOAL_DEPTH_M = 2.5


def estimate_field_geometry(
    player_positions: Optional[np.ndarray],
    frame_size: Tuple[int, int],
    goal_box_left: Optional[Dict[str, object]] = None,
    goal_box_right: Optional[Dict[str, object]] = None,
    *,
    field_bounds: Optional[Sequence[float]] = None,
    calibration: object = None,
    goal_mouth_frac: float = DEFAULT_GOAL_MOUTH_FRAC,
    goal_depth_frac: float = DEFAULT_GOAL_DEPTH_FRAC,
    goal_margin_m: float = DEFAULT_GOAL_MARGIN_M,
    goal_depth_m: float = DEFAULT_GOAL_DEPTH_M,
) -> FieldGeometry:
    """Estimate the playable area and goal mouths from player positions.

    Player positions accumulated over a match trace out the field: robust
    percentiles of x/y give the field bounds, and the vertical position of
    players close to each end line (mostly the goalkeepers) centers the goal
    mouth. Manual goal boxes (pixel or normalized) override everything.

    ``field_bounds`` ``(x_min, y_min, x_max, y_max)`` (pixels) or a
    ``pitch_calibration.PitchCalibration`` (its ``image_corners_px``) pin the
    field rectangle exactly instead of estimating it from players, who rarely
    reach the lines.

    Goal boxes (the area the ball must reach to count as "in the goal"):

    * with a ``calibration`` (duck-typed: ``to_image(points_m)`` and
      ``pitch_length_m``): the real goal mouth (posts at +/-3.66 m on the
      goal line ``x = +/-length/2``) widened by ``goal_margin_m`` beyond each
      post, and ``goal_depth_m`` deep behind the line, projected to pixels.
      The goal line x is the projected line between the posts.
    * without: ``goal_mouth_frac`` (default 0.11 = 7.32 m / 68 m) of the field
      height, centred on the estimated goal-line y, ``goal_depth_frac`` of the
      field width deep.
    """
    w, h = int(frame_size[0]), int(frame_size[1])
    positions = None
    if player_positions is not None and len(player_positions) >= 50:
        positions = np.asarray(player_positions, dtype=np.float64)

    pinned: Optional[Tuple[float, float, float, float]] = None
    pinned_source = ""
    if field_bounds is not None:
        fb = [float(v) for v in list(field_bounds)[:4]]
        if len(fb) == 4 and fb[2] > fb[0] and fb[3] > fb[1]:
            pinned = (fb[0], fb[1], fb[2], fb[3])
            pinned_source = "field_bounds"
    corners = getattr(calibration, "image_corners_px", None) if calibration is not None else None
    if pinned is None and corners is not None:
        try:
            c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            pinned = (float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max()))
            pinned_source = f"calibration_{getattr(calibration, 'source', 'auto')}"
        except Exception:
            pinned = None

    if pinned is not None:
        x_min, y_min, x_max, y_max = pinned
        source = pinned_source
    elif positions is not None:
        xs = positions[:, 1]
        ys = positions[:, 2]
        x_min, x_max = float(np.percentile(xs, 0.5)), float(np.percentile(xs, 99.5))
        y_min, y_max = float(np.percentile(ys, 1.5)), float(np.percentile(ys, 98.5))
        source = "estimated"
    else:
        x_min, x_max = w * 0.02, w * 0.98
        y_min, y_max = h * 0.10, h * 0.90
        source = "frame_default"

    field_w = max(1.0, x_max - x_min)
    field_h = max(1.0, y_max - y_min)

    def _goal_y_center(near_x: float) -> float:
        if positions is None:
            return (y_min + y_max) / 2.0
        band = positions[np.abs(positions[:, 1] - near_x) <= field_w * 0.08]
        if len(band) < 20:
            return (y_min + y_max) / 2.0
        return float(np.median(band[:, 2]))

    goal_h = max(16.0, field_h * float(goal_mouth_frac))
    goal_depth = max(16.0, field_w * float(goal_depth_frac))

    def _calibrated_goal(side: str) -> Optional[GoalBox]:
        if calibration is None or not hasattr(calibration, "to_image"):
            return None
        try:
            half_l = float(getattr(calibration, "pitch_length_m", 105.0)) / 2.0
            line_m = -half_l if side == "left" else half_l
            back_m = line_m + (-1.0 if side == "left" else 1.0) * max(0.0, float(goal_depth_m))
            half_mouth = GOAL_MOUTH_HALF_M + max(0.0, float(goal_margin_m))
            # Posts (+margin) on the goal line, then the back of the box.
            pts_m = np.array([[line_m, -half_mouth], [line_m, half_mouth],
                              [back_m, -half_mouth], [back_m, half_mouth]])
            pts = np.asarray(calibration.to_image(pts_m), dtype=np.float64).reshape(-1, 2)
            if pts.shape[0] != 4 or not np.all(np.isfinite(pts)):
                return None
            line_x = float(np.mean(pts[:2, 0]))
            back_x = float(np.mean(pts[2:, 0]))
            y1, y2 = float(pts[:, 1].min()), float(pts[:, 1].max())
            if y2 - y1 < 4.0:
                return None
            depth = max(4.0, abs(back_x - line_x))
            if side == "left":
                return GoalBox(side="left", x1=max(0.0, line_x - depth), y1=y1, x2=line_x, y2=y2)
            return GoalBox(side="right", x1=line_x, y1=y1, x2=min(float(w), line_x + depth), y2=y2)
        except Exception:
            return None

    def _build_goal(side: str) -> GoalBox:
        calibrated = _calibrated_goal(side)
        if calibrated is not None:
            return calibrated
        if side == "left":
            y_c = _goal_y_center(x_min)
            return GoalBox(side="left", x1=max(0.0, x_min - goal_depth), y1=y_c - goal_h / 2.0,
                           x2=x_min, y2=y_c + goal_h / 2.0)
        y_c = _goal_y_center(x_max)
        return GoalBox(side="right", x1=x_max, y1=y_c - goal_h / 2.0,
                       x2=min(float(w), x_max + goal_depth), y2=y_c + goal_h / 2.0)

    left_manual = _goal_box_from_override(goal_box_left, "left", (w, h))
    right_manual = _goal_box_from_override(goal_box_right, "right", (w, h))
    left = left_manual or _build_goal("left")
    right = right_manual or _build_goal("right")
    if goal_box_left or goal_box_right:
        source = "manual" if (goal_box_left and goal_box_right) else f"{source}+manual"
    if left_manual is not None and right_manual is not None and pinned is None:
        # Both goal mouths known: the goal lines ARE the field's x bounds.
        x_min, x_max = left_manual.x2, right_manual.x1

    geometry = FieldGeometry(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        left_goal=left, right_goal=right, frame_size=(w, h), source=source,
    )
    LOGGER.info(
        "field geometry (%s): x=[%.0f, %.0f] y=[%.0f, %.0f], left goal %s, right goal %s",
        source, x_min, x_max, y_min, y_max, left.to_dict(), right.to_dict(),
    )
    return geometry


# ---------------------------------------------------------------------------
# Goal events
# ---------------------------------------------------------------------------


@dataclass
class GoalEvent:
    t: float
    side: str
    confidence: float
    reason: str
    evidence: Dict[str, object] = field(default_factory=dict)
    # "goal" for kept goals; rejected candidates are "shot" (the ball reached
    # the goal mouth but the goal was not corroborated) or "chance".
    verdict: str = "goal"

    def to_dict(self) -> Dict[str, object]:
        return {
            "t": round(self.t, 3),
            "side": self.side,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "evidence": self.evidence,
            "verdict": self.verdict,
        }


@dataclass
class GameStateConfig:
    step_s: float = 0.2
    # Distance from a goal line (fraction of field width) that counts as
    # "near the goal" when the ball disappears.
    near_goal_frac: float = 0.14
    # How far beyond the field bounds a visible ball counts as out of play.
    out_margin_frac: float = 0.01
    # The ball must be invisible at least this long before we call it lost.
    lost_grace_s: float = 0.7
    # The ball must be back inside the field this long before a restart hold
    # is released (prevents flicker off single false detections).
    return_confirm_s: float = 0.6
    # Safety valve: never hold a restart longer than this.
    max_restart_hold_s: float = 50.0
    # Goal detection.
    goal_disappear_confirm_s: float = 2.0
    goal_lookback_s: float = 0.7
    goal_hold_s: float = 8.0
    kickoff_center_frac: float = 0.16
    kickoff_search_s: float = 60.0
    min_shot_speed_frame_widths_per_s: float = 0.25
    # Minimum x-velocity (px/s) that counts as the ball entering a goal.
    goal_entry_speed_px_s: float = 20.0
    # Goal signals on the same side chain into ONE candidate when each is
    # within this many seconds of the previous one (crossing -> ball in the
    # net -> ball vanishes behind the net).
    goal_merge_window_s: float = 4.0
    # Two kept goals on the same side closer than this are the same goal
    # (late secondary sightings); the more confident one is kept.
    goal_min_separation_s: float = 25.0
    # How far ahead a vanishing ball's path is extrapolated to the goal line.
    goal_extrapolation_s: float = 0.8
    # Consecutive in-goal sightings further apart than this start a new run.
    goal_run_gap_s: float = 1.5
    # Weak band around the posts (max of these two, the fraction is of the
    # goal box height). Inside the goal box is a "strong" location; inside
    # this band only "weak". Calibrated boxes already include ~1 m beyond
    # each post, so the band stays small.
    goal_mouth_margin_px: float = 6.0
    goal_mouth_margin_frac: float = 0.06
    # When the goal boxes are only ESTIMATED from player positions (a real
    # 7.32 m mouth, ~11 % of the field height, centred on the goalkeepers'
    # median y), the central ``estimated_goal_strict_frac`` of the box counts
    # as "between the posts" (strong) and the weak band extends
    # ``estimated_goal_margin_frac`` box heights beyond each post to absorb
    # the uncertainty of the estimated centre (strong 11 %, weak 22 % of the
    # field height; the old 30 %-tall estimate gave 18 % / 34 %).
    estimated_goal_strict_frac: float = 1.0
    estimated_goal_margin_frac: float = 0.5
    # Ball arrested by the net: at least this long / many samples inside the
    # goal box with a low median speed.
    goal_dwell_min_s: float = 0.3
    goal_dwell_min_samples: int = 3
    goal_dwell_max_speed_frame_widths_per_s: float = 0.2
    # A crossing followed by the ball back on the field side (deeper than
    # depth_frac of the field width) within this window is a rebound/post.
    goal_rebound_window_s: float = 1.5
    goal_rebound_depth_frac: float = 0.02
    # Negative signal: ball seen in play far from the goal soon after.
    goal_reappear_window_s: float = 3.0
    goal_reappear_far_frac: float = 0.30
    # Audio peak (crowd) must be within this many seconds of the goal.
    goal_audio_window_s: float = 4.0
    # Kickoff validation: the ball is placed on the centre spot (first
    # sighting after a gap, or resting there) and - with player tracks - both
    # teams stand in their own halves.
    kickoff_spot_frac: float = 0.06
    kickoff_min_rest_s: float = 0.5
    kickoff_gap_s: float = 1.0
    kickoff_half_majority: float = 0.75
    kickoff_min_players_per_team: int = 3
    kickoff_halfway_tolerance_frac: float = 0.03
    # The kickoff search stops once the ball is clearly back in open play.
    open_play_speed_frame_widths_per_s: float = 0.15
    open_play_sustain_s: float = 1.5
    # Confidence calibration (see detect_goal_candidates).
    goal_strong_base: float = 0.55
    goal_weak_base: float = 0.40
    goal_extra_strong: float = 0.25
    goal_extra_weak: float = 0.17
    goal_kickoff_players_bonus: float = 0.20
    goal_kickoff_ball_only_bonus: float = 0.05
    goal_audio_bonus: float = 0.17
    goal_celebration_bonus: float = 0.08
    goal_rejected_cap: float = 0.59
    goal_negative_cap: float = 0.35
    # Goal events below this confidence are not goals (they are emitted as
    # shot/chance candidates by detect_goal_candidates).
    min_goal_confidence: float = 0.7
    # --- Set pieces (corners, free kicks, goal kicks, kickoffs) ---
    # The ball must sit still (within the radius) at least this long.
    set_piece_min_stationary_s: float = 1.2
    set_piece_stationary_radius_frac: float = 0.012  # of frame width
    # ...and then accelerate away at least this fast to count as the kick.
    set_piece_kick_speed_frame_widths_per_s: float = 0.15
    # Location classification (fractions of field size).
    corner_radius_frac: float = 0.06
    goal_kick_zone_depth_frac: float = 0.12
    goal_kick_zone_half_height_frac: float = 0.25
    penalty_depth_range_frac: Tuple[float, float] = (0.06, 0.18)
    penalty_half_height_frac: float = 0.12
    # A free kick within this distance of a goal line threatens that goal,
    # so the camera must keep the goal in view during the run-up.
    free_kick_threat_frac: float = 0.38


@dataclass
class GameStateSegment:
    start_s: float
    end_s: float
    state: str
    side: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "state": self.state,
            "side": self.side,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Player / audio helpers shared by goal detection and the event engine
# ---------------------------------------------------------------------------


def iter_player_tracks(player_tracks: object) -> List[object]:
    """Normalize a TrackingResult / dict / list of PlayerTrack into a list."""
    if player_tracks is None:
        return []
    players = getattr(player_tracks, "players", player_tracks)
    if isinstance(players, dict):
        players = list(players.values())
    try:
        return [p for p in players if len(p) > 0]  # type: ignore[arg-type]
    except TypeError:
        return []


def team_positions_at(
    player_tracks: object,
    t: float,
    window_s: float = 1.0,
    teams: Sequence[int] = (0, 1),
) -> Dict[int, np.ndarray]:
    """Median foot position of every team player seen within ``t +/- window_s``.

    Returns ``{team: array[k, 3]}`` with rows ``(x, y, track_id)``.
    """
    rows: Dict[int, List[Tuple[float, float, float]]] = {int(team): [] for team in teams}
    for track in iter_player_tracks(player_tracks):
        team = int(getattr(track, "team", -1))
        if team not in rows:
            continue
        lo = int(np.searchsorted(track.t, t - window_s, side="left"))
        hi = int(np.searchsorted(track.t, t + window_s, side="right"))
        if hi <= lo:
            continue
        fx = float(np.median((track.x1[lo:hi] + track.x2[lo:hi]) * 0.5))
        fy = float(np.median(track.y2[lo:hi]))
        rows[team].append((fx, fy, float(track.track_id)))
    return {team: np.asarray(r, dtype=np.float64).reshape(-1, 3) for team, r in rows.items()}


def defending_sides(
    player_tracks: object,
    t: float,
    window_s: float = 90.0,
) -> Optional[Dict[str, int]]:
    """Which team defends which goal around time ``t`` (handles half-time swaps).

    The team whose players sit further left (median foot x over the window)
    defends the left goal. Returns ``{"left": team, "right": team}`` or None
    when the two teams cannot be told apart.
    """
    medians: Dict[int, List[float]] = {0: [], 1: []}
    for track in iter_player_tracks(player_tracks):
        team = int(getattr(track, "team", -1))
        if team not in medians:
            continue
        lo = int(np.searchsorted(track.t, t - window_s, side="left"))
        hi = int(np.searchsorted(track.t, t + window_s, side="right"))
        if hi <= lo:
            continue
        medians[team].append(float(np.median((track.x1[lo:hi] + track.x2[lo:hi]) * 0.5)))
    if not medians[0] or not medians[1]:
        return None
    m0, m1 = float(np.median(medians[0])), float(np.median(medians[1]))
    if abs(m0 - m1) < 1e-6:
        return None
    return {"left": 0, "right": 1} if m0 < m1 else {"left": 1, "right": 0}


def find_audio_peaks(
    envelope: Optional[Tuple[np.ndarray, np.ndarray]],
    k: float = 3.0,
    min_separation_s: float = 1.5,
) -> List[Tuple[float, float]]:
    """Crowd-noise peaks ``[(t, strength 0..1)]`` from an RMS envelope.

    Threshold: median + k * MAD (scaled). Strength is the peak height between
    the median and the 99.5th percentile of the envelope.
    """
    if envelope is None:
        return []
    times = np.asarray(envelope[0], dtype=np.float64)
    rms = np.asarray(envelope[1], dtype=np.float64)
    if len(rms) < 8 or len(times) != len(rms):
        return []
    med = float(np.median(rms))
    mad = float(np.median(np.abs(rms - med))) * 1.4826 + 1e-12
    thr = med + k * mad
    hi_ref = max(float(np.percentile(rms, 99.5)), thr + 1e-12)
    left = np.concatenate([[-np.inf], rms[:-1]])
    right = np.concatenate([rms[1:], [-np.inf]])
    cand = np.flatnonzero((rms >= thr) & (rms >= left) & (rms >= right))
    picked: List[int] = []
    for i in sorted(cand, key=lambda j: -rms[j]):
        if all(abs(times[i] - times[p]) >= min_separation_s for p in picked):
            picked.append(int(i))
    out = [
        (float(times[i]), float(np.clip((rms[i] - med) / (hi_ref - med + 1e-12), 0.0, 1.0)))
        for i in picked
    ]
    return sorted(out)


def audio_peak_near(
    peaks: Sequence[Tuple[float, float]], t: float, window_s: float
) -> Optional[Tuple[float, float]]:
    """Strongest peak within ``t +/- window_s``."""
    near = [p for p in peaks if abs(p[0] - t) <= window_s]
    if not near:
        return None
    return max(near, key=lambda p: p[1])


def _goal_line_x(goal: GoalBox) -> float:
    return goal.x2 if goal.side == "left" else goal.x1


def _geometry_is_estimated(geometry: FieldGeometry) -> bool:
    source = str(geometry.source or "")
    return not any(tag in source for tag in ("manual", "calibrat", "field_bounds"))


def detect_goal_candidates(
    ball_track: BallTrack,
    geometry: FieldGeometry,
    start_s: float,
    end_s: float,
    config: Optional[GameStateConfig] = None,
    *,
    player_tracks: object = None,
    audio_envelope: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> List[GoalEvent]:
    """Score every goal-like ball event; ``verdict`` says goal / shot / chance.

    Ball signals (each ``strong`` or ``weak``):

    * ``line_crossing``: the visible ball crosses the goal line moving into
      the goal. Strong between the posts with no rebound onto the field
      within ``goal_rebound_window_s``; weak inside the post margin band or
      after a rebound.
    * ``in_goal_box``: the ball is observed inside the goal box having entered
      from the field. Strong only when it is *arrested by the net* (dwell of
      >= ``goal_dwell_min_s`` / ``goal_dwell_min_samples`` at low speed); a
      single sighting or a ball flying through (over the bar) is weak.
    * ``vanished_into_mouth``: the ball disappears while moving fast toward
      the mouth and stays gone >= ``goal_disappear_confirm_s`` (strong); a
      short or trailing (end-of-recording) gap is weak.

    Corroboration (all required for ``verdict == "goal"``):

    * at least one STRONG ball signal plus one more independent signal
      (another ball signal, a kickoff validated by player positions - or by
      the ball alone when no player tracks exist -, an audio peak within
      ``goal_audio_window_s``, players converging); weak-only ball evidence
      additionally needs a players-validated kickoff and >= 3 signals;
    * no negative signal: the ball is not seen in play far from the goal
      within ``goal_reappear_window_s``;
    * calibrated confidence >= ``min_goal_confidence``.

    Confidence: ``goal_strong_base`` (0.55) or ``goal_weak_base`` (0.40) for
    the best ball signal, +0.25/+0.17 per additional strong/weak ball signal,
    +0.20 kickoff with both teams in their halves (+0.05 ball-only kickoff),
    +0.17 audio peak, +0.08 celebration. A single signal therefore stays
    below 0.6; rejected candidates are capped at ``goal_rejected_cap``.
    """
    cfg = config or GameStateConfig()
    n = len(ball_track)
    if n == 0:
        return []

    times, xs, ys = ball_track.times, ball_track.xs, ball_track.ys
    frame_w = float(ball_track.frame_size[0])
    min_speed = cfg.min_shot_speed_frame_widths_per_s * frame_w
    center_x = (geometry.x_min + geometry.x_max) / 2.0
    center_y = (geometry.y_min + geometry.y_max) / 2.0
    estimated = _geometry_is_estimated(geometry)
    tracks = iter_player_tracks(player_tracks)
    have_tracks = any(int(getattr(tr, "team", -1)) in (0, 1) for tr in tracks)
    peaks = find_audio_peaks(audio_envelope)

    # Speed of the segment ending at each sample (NaN across gaps).
    seg_speed = np.full(n, np.nan)
    if n >= 2:
        dts_raw = np.diff(times)
        step = np.hypot(np.diff(xs), np.diff(ys)) / np.maximum(dts_raw, 1e-3)
        seg_speed[1:] = np.where(dts_raw <= 0.5, step, np.nan)

    def _bands(goal: GoalBox) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        h = abs(goal.y2 - goal.y1)
        if estimated:
            shrink = (1.0 - min(1.0, cfg.estimated_goal_strict_frac)) / 2.0 * h
            strict = (goal.y1 + shrink, goal.y2 - shrink)
            margin_frac = max(cfg.goal_mouth_margin_frac, cfg.estimated_goal_margin_frac)
        else:
            strict = (goal.y1, goal.y2)
            margin_frac = cfg.goal_mouth_margin_frac
        margin = max(cfg.goal_mouth_margin_px, h * margin_frac)
        return strict, (goal.y1 - margin, goal.y2 + margin)

    def _band(y: float, goal: GoalBox) -> Optional[str]:
        strict, loose = _bands(goal)
        if strict[0] <= y <= strict[1]:
            return "strong"
        if loose[0] <= y <= loose[1]:
            return "weak"
        return None

    def _entered_from_field(goal: GoalBox, first_idx: int) -> bool:
        """True when the ball moved INTO the goal from the field side (rejects
        goal kicks: the ball appears in/near the box and moves infield)."""
        t_first = float(times[first_idx])
        vx, _vy = ball_track.velocity_at(t_first, window_s=0.5)
        gate = cfg.goal_entry_speed_px_s
        if (vx < -gate) if goal.side == "left" else (vx > gate):
            return True
        if first_idx > 0 and t_first - float(times[first_idx - 1]) <= 1.0:
            prev_x = float(xs[first_idx - 1])
            line_x = _goal_line_x(goal)
            return prev_x > line_x if goal.side == "left" else prev_x < line_x
        return False

    signals: Dict[str, List[Dict[str, object]]] = {"left": [], "right": []}
    in_window = (times >= start_s) & (times <= end_s)

    for goal in (geometry.left_goal, geometry.right_goal):
        line_x = _goal_line_x(goal)
        into = -1.0 if goal.side == "left" else 1.0

        # --- line crossings -------------------------------------------------
        if n >= 2:
            if goal.side == "left":
                crossed = (xs[:-1] >= line_x) & (xs[1:] < line_x)
            else:
                crossed = (xs[:-1] <= line_x) & (xs[1:] > line_x)
            crossed &= np.diff(times) <= 0.5
            crossed &= (times[1:] >= start_s) & (times[:-1] <= end_s)
            for i in np.flatnonzero(crossed):
                t0, t1 = float(times[i]), float(times[i + 1])
                x0, x1 = float(xs[i]), float(xs[i + 1])
                alpha = (line_x - x0) / (x1 - x0) if abs(x1 - x0) > 1e-6 else 0.0
                y_cross = float(ys[i]) + (float(ys[i + 1]) - float(ys[i])) * alpha
                band = _band(y_cross, goal)
                if band is None:
                    continue
                vx = (x1 - x0) / max(1e-3, t1 - t0)
                if vx * into < cfg.goal_entry_speed_px_s:
                    continue
                t_cross = t0 + (t1 - t0) * alpha
                hi = int(np.searchsorted(times, t_cross + cfg.goal_rebound_window_s, side="right"))
                depth = cfg.goal_rebound_depth_frac * geometry.width
                after = xs[i + 1:hi]
                back = (after > line_x + depth) if goal.side == "left" else (after < line_x - depth)
                rebound_t = float(times[i + 1 + int(np.argmax(back))]) if back.any() else None
                strength = band if rebound_t is None else "weak"
                evidence: Dict[str, object] = {
                    "crossing_y": round(y_cross, 1),
                    "goal_line_x": round(line_x, 1),
                    "crossing_speed_px_s": round(abs(vx), 1),
                    "crossing_between_posts": band == "strong",
                }
                if rebound_t is not None:
                    evidence["rebound_to_field_s"] = round(rebound_t, 3)
                signals[goal.side].append({
                    "kind": "line_crossing", "strength": strength, "t": t_cross, "t_end": t_cross,
                    "evidence": evidence,
                })

        # --- ball observed inside the goal box ---------------------------------
        strict, loose = _bands(goal)
        mask = (
            (xs >= goal.x1) & (xs <= goal.x2) & (ys >= loose[0]) & (ys <= loose[1]) & in_window
        )
        idxs = np.flatnonzero(mask)
        if len(idxs):
            runs: List[Tuple[int, int]] = []
            run_start = prev = int(idxs[0])
            line_x = _goal_line_x(goal)
            for i in idxs[1:]:
                between = xs[prev + 1:i]
                back_in_field = bool(np.any(between > line_x if goal.side == "left" else between < line_x))
                if float(times[i] - times[prev]) > cfg.goal_run_gap_s or back_in_field:
                    runs.append((run_start, prev))
                    run_start = int(i)
                prev = int(i)
            runs.append((run_start, prev))
            for a, b in runs:
                t_first = float(times[a])
                if not _entered_from_field(goal, a):
                    LOGGER.debug("goal-box sighting t=%.2fs (%s) ignored: did not enter from the field",
                                 t_first, goal.side)
                    continue
                run_y = ys[a:b + 1]
                strict_frac = float(np.mean((run_y >= strict[0]) & (run_y <= strict[1])))
                dwell_s = float(times[b] - times[a])
                n_samples = int(b - a + 1)
                inner = seg_speed[a + 1:b + 1]
                inner = inner[np.isfinite(inner)]
                med_speed = float(np.median(inner)) if len(inner) else float("inf")
                arrested = (
                    dwell_s >= cfg.goal_dwell_min_s
                    and n_samples >= cfg.goal_dwell_min_samples
                    and med_speed <= cfg.goal_dwell_max_speed_frame_widths_per_s * frame_w
                )
                strength = "strong" if strict_frac >= 0.5 and arrested else "weak"
                signals[goal.side].append({
                    "kind": "in_goal_box", "strength": strength, "t": t_first,
                    "t_end": float(times[b]),
                    "evidence": {
                        "first_seen_in_goal_s": round(t_first, 3),
                        "samples_in_goal": n_samples,
                        "dwell_s": round(dwell_s, 3),
                        "dwell_median_speed_px_s": round(med_speed, 1) if math.isfinite(med_speed) else None,
                        "arrested_by_net": bool(arrested),
                    },
                })

    # --- ball vanishes heading into a goal mouth -----------------------------
    track_end_t = float(times[-1])
    for gap_start, gap_end in ball_track.visibility_gaps(start_s, end_s):
        if gap_start <= float(times[0]):
            continue
        trailing = gap_start >= track_end_t - 1e-6
        last = ball_track.position_at(gap_start)
        if last is None:
            continue
        vx, vy = ball_track.velocity_at(gap_start - 0.05, window_s=cfg.goal_lookback_s)
        if math.hypot(vx, vy) < min_speed:
            continue
        for goal in (geometry.left_goal, geometry.right_goal):
            line_x = _goal_line_x(goal)
            if not (vx < 0 if goal.side == "left" else vx > 0):
                continue
            time_to_line = (line_x - last[0]) / vx if abs(vx) > 1e-6 else float("inf")
            if not (0.0 <= time_to_line <= cfg.goal_extrapolation_s):
                continue
            y_at_line = last[1] + vy * time_to_line
            band = _band(y_at_line, goal)
            if band is None:
                continue
            gap_len = gap_end - gap_start
            long_gap = gap_len >= cfg.goal_disappear_confirm_s and not trailing
            strength = "strong" if band == "strong" and long_gap else "weak"
            t_sig = gap_start + max(0.0, min(time_to_line, gap_len))
            signals[goal.side].append({
                "kind": "vanished_into_mouth", "strength": strength, "t": t_sig, "t_end": t_sig,
                "evidence": {
                    "last_seen_xy": [round(last[0], 1), round(last[1], 1)],
                    "velocity_px_s": [round(vx, 1), round(vy, 1)],
                    "projected_goal_line_y": round(y_at_line, 1),
                    "disappeared_for_s": round(gap_len, 3),
                    "trailing_gap": bool(trailing),
                },
            })

    def _find_kickoff(after_t: float, side: str) -> Optional[float]:
        """Ball placed on the centre spot after ``after_t`` (before open play resumes)."""
        lo = int(np.searchsorted(times, after_t, side="left"))
        hi = int(np.searchsorted(times, after_t + cfg.kickoff_search_s, side="right"))
        spot_r = cfg.kickoff_spot_frac * geometry.width
        rest_r = cfg.set_piece_stationary_radius_frac * frame_w
        open_speed = cfg.open_play_speed_frame_widths_per_s * frame_w
        line_x = _goal_line_x(geometry.goal_for_side(side))
        open_since: Optional[float] = None
        for i in range(lo, hi):
            t = float(times[i])
            d_centre = math.hypot(float(xs[i]) - center_x, float(ys[i]) - center_y)
            if d_centre <= spot_r:
                gap_before = t - float(times[i - 1]) if i > 0 else float("inf")
                if gap_before >= cfg.kickoff_gap_s:
                    return t
                j = i
                while (j + 1 < n and float(times[j + 1] - times[j]) <= 0.5
                       and math.hypot(float(xs[j + 1] - xs[i]), float(ys[j + 1] - ys[i])) <= rest_r):
                    j += 1
                if float(times[j]) - t >= cfg.kickoff_min_rest_s:
                    return t
            sp = seg_speed[i]
            far = abs(float(xs[i]) - line_x) > cfg.goal_reappear_far_frac * geometry.width
            if np.isfinite(sp) and sp >= open_speed and far and d_centre > spot_r:
                if open_since is None:
                    open_since = t
                elif t - open_since >= cfg.open_play_sustain_s:
                    return None
            else:
                open_since = None
        return None

    def _kickoff_players(t_k: float) -> Tuple[Optional[bool], Dict[str, object]]:
        """None = not enough players to judge; True/False = both teams in halves."""
        if not have_tracks:
            return None, {}
        pos = team_positions_at(tracks, t_k, window_s=1.0)
        tol = cfg.kickoff_halfway_tolerance_frac * geometry.width
        sides: Dict[int, Optional[str]] = {}
        detail: Dict[str, object] = {}
        for team, rows in pos.items():
            if len(rows) < cfg.kickoff_min_players_per_team:
                return None, {"kickoff_players_visible": {str(k): len(v) for k, v in pos.items()}}
            px = rows[:, 0]
            amb = np.abs(px - center_x) <= tol
            left_frac = float(np.mean((px < center_x) | amb))
            right_frac = float(np.mean((px > center_x) | amb))
            detail[f"team{team}_left_frac"] = round(left_frac, 2)
            detail[f"team{team}_right_frac"] = round(right_frac, 2)
            if left_frac >= cfg.kickoff_half_majority and left_frac >= right_frac:
                sides[team] = "left"
            elif right_frac >= cfg.kickoff_half_majority:
                sides[team] = "right"
            else:
                sides[team] = None
        ok = sides.get(0) is not None and sides.get(1) is not None and sides[0] != sides[1]
        detail["team_halves"] = {str(k): v for k, v in sides.items()}
        return bool(ok), detail

    def _celebration(t_event: float, side: str) -> Optional[float]:
        """Spread ratio (after/before) of the scoring team; < 0.75 = converging."""
        if not have_tracks:
            return None
        sides = defending_sides(tracks, t_event)
        if sides is None:
            return None
        attackers = sides["left" if side == "right" else "right"]
        before = team_positions_at(tracks, t_event - 1.0, window_s=1.0).get(attackers)
        after = team_positions_at(tracks, t_event + 5.0, window_s=3.0).get(attackers)
        if before is None or after is None or len(before) < 3 or len(after) < 3:
            return None

        def _spread(rows: np.ndarray) -> float:
            c = rows[:, :2].mean(axis=0)
            return float(np.mean(np.hypot(rows[:, 0] - c[0], rows[:, 1] - c[1])))

        sb = _spread(before)
        if sb <= 1e-6:
            return None
        return _spread(after) / sb

    candidates: List[GoalEvent] = []
    for side, sigs in signals.items():
        if not sigs:
            continue
        goal = geometry.goal_for_side(side)
        line_x = _goal_line_x(goal)
        sigs.sort(key=lambda s: float(s["t"]))  # type: ignore[arg-type]
        clusters: List[List[Dict[str, object]]] = []
        last_end = -1e18
        for sig in sigs:
            if clusters and float(sig["t"]) - last_end <= cfg.goal_merge_window_s:  # type: ignore[arg-type]
                clusters[-1].append(sig)
                last_end = max(last_end, float(sig["t_end"]))  # type: ignore[arg-type]
            else:
                clusters.append([sig])
                last_end = float(sig["t_end"])  # type: ignore[arg-type]

        for cluster in clusters:
            best: Dict[str, Dict[str, object]] = {}
            for sig in cluster:
                kind = str(sig["kind"])
                if kind not in best or (sig["strength"] == "strong" and best[kind]["strength"] != "strong"):
                    best[kind] = sig
            strong_kinds = sorted(k for k, s in best.items() if s["strength"] == "strong")
            strong_ts = [float(s["t"]) for s in cluster if s["strength"] == "strong"]  # type: ignore[arg-type]
            t_event = min(strong_ts) if strong_ts else min(float(s["t"]) for s in cluster)  # type: ignore[arg-type]
            # When the ball is seen arrested in the net, the goal happened at
            # the crossing that led into it (not an earlier crossing in the
            # same chain, e.g. a ball skimming the line moments before).
            dwell = [s for s in cluster if s["kind"] == "in_goal_box" and s["strength"] == "strong"]
            if dwell:
                t_box = min(float(s["t"]) for s in dwell)  # type: ignore[arg-type]
                leading = [float(s["t"]) for s in cluster  # type: ignore[arg-type]
                           if s["kind"] == "line_crossing" and 0.0 <= t_box - float(s["t"]) <= 1.5]  # type: ignore[arg-type]
                t_event = max(leading) if leading else t_box
            t_last = max(float(s["t_end"]) for s in cluster)  # type: ignore[arg-type]
            ordered = sorted(best.values(), key=lambda s: 0 if s["strength"] == "strong" else 1)
            confidence = cfg.goal_strong_base if strong_kinds else cfg.goal_weak_base
            for sig in ordered[1:]:
                confidence += cfg.goal_extra_strong if sig["strength"] == "strong" else cfg.goal_extra_weak

            evidence: Dict[str, object] = {
                "signals": [
                    {"kind": s["kind"], "strength": s["strength"], "t": round(float(s["t"]), 3)}  # type: ignore[arg-type]
                    for s in cluster
                ],
                "strong_signals": strong_kinds,
                "observed_in_goal_box": "in_goal_box" in best,
                "line_crossed_while_visible": "line_crossing" in best,
                "vanished_into_mouth": "vanished_into_mouth" in best,
                "geometry_source": geometry.source,
            }
            for sig in ordered[::-1]:
                evidence.update(sig["evidence"])  # type: ignore[arg-type]
            if "line_crossing" in best and best["line_crossing"]["strength"] == "strong":
                evidence["stayed_behind_line"] = True
            reasons = [
                f"{s['kind'].replace('_', ' ')} ({s['strength']})" for s in ordered  # type: ignore[union-attr]
            ]

            # Negative: ball back in play far from this goal right after.
            lo = int(np.searchsorted(times, t_event, side="right"))
            hi = int(np.searchsorted(times, t_event + cfg.goal_reappear_window_s, side="right"))
            far = np.flatnonzero(np.abs(xs[lo:hi] - line_x) > cfg.goal_reappear_far_frac * geometry.width)
            negative_t = float(times[lo + int(far[0])]) if len(far) else None

            independent = len(best)
            kickoff_mode: Optional[str] = None
            kickoff_t = _find_kickoff(t_last + 0.5, side)
            if kickoff_t is not None:
                evidence["kickoff_reappearance_s"] = round(kickoff_t, 3)
                ok, detail = _kickoff_players(kickoff_t)
                evidence.update(detail)
                if ok is True:
                    kickoff_mode = "players"
                    confidence += cfg.goal_kickoff_players_bonus
                    independent += 1
                    reasons.append("kickoff with both teams in their halves")
                elif ok is None:
                    kickoff_mode = "ball_only"
                    confidence += cfg.goal_kickoff_ball_only_bonus
                    reasons.append("ball placed on the centre spot")
                else:
                    kickoff_mode = "players_rejected"
                    reasons.append("centre-spot restart but teams not in their halves")
            evidence["kickoff_validation"] = kickoff_mode

            peak = audio_peak_near(peaks, t_event, cfg.goal_audio_window_s)
            if peak is not None:
                confidence += cfg.goal_audio_bonus
                independent += 1
                evidence["audio_peak_s"] = round(peak[0], 3)
                evidence["audio_peak_strength"] = round(peak[1], 3)
                reasons.append("crowd-noise peak")

            spread_ratio = _celebration(t_event, side)
            if spread_ratio is not None:
                evidence["scoring_team_spread_ratio"] = round(spread_ratio, 3)
                if spread_ratio < 0.75:
                    confidence += cfg.goal_celebration_bonus
                    independent += 1
                    reasons.append("scoring team converged (celebration)")

            kickoff_ok = kickoff_mode == "players" or (kickoff_mode == "ball_only" and not have_tracks)
            if strong_kinds:
                corroborated = independent >= 2 or kickoff_ok or peak is not None
            else:
                # Weak-only ball evidence (ball near a post, short vanish) is
                # never enough with audio alone: it needs the players lining
                # up for a kickoff plus at least one more signal.
                corroborated = kickoff_mode == "players" and independent >= 3
            if negative_t is not None:
                evidence["reappeared_in_play_s"] = round(negative_t, 3)
                reasons.append(f"ball back in play far from goal at {negative_t:.1f}s - not a goal")
                confidence = min(confidence, cfg.goal_negative_cap)
                corroborated = False
            confidence = float(min(0.98, confidence))
            evidence["independent_signals"] = int(independent)
            evidence["corroborated"] = bool(corroborated)

            if corroborated and confidence >= cfg.min_goal_confidence - 1e-9:
                verdict = "goal"
                reason = f"goal at {side} goal: " + "; ".join(reasons)
            else:
                reached_mouth = bool(strong_kinds) or "line_crossing" in best or "in_goal_box" in best
                verdict = "shot" if reached_mouth else "chance"
                confidence = min(confidence, cfg.goal_rejected_cap)
                reason = f"uncorroborated goal candidate at {side} goal ({verdict}): " + "; ".join(reasons)
            candidates.append(GoalEvent(t=float(t_event), side=side, confidence=confidence,
                                        reason=reason, evidence=evidence, verdict=verdict))

    candidates.sort(key=lambda e: e.t)
    return candidates


def detect_goal_events(
    ball_track: BallTrack,
    geometry: FieldGeometry,
    start_s: float,
    end_s: float,
    config: Optional[GameStateConfig] = None,
    *,
    player_tracks: object = None,
    audio_envelope: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    candidates_out: Optional[List[GoalEvent]] = None,
) -> List[GoalEvent]:
    """Corroborated goals only (see :func:`detect_goal_candidates`).

    ``player_tracks`` (a ``TrackingResult`` or its ``players``) enables the
    kickoff-in-halves check and the celebration signal; ``audio_envelope``
    (``(times, rms)``) enables the crowd-peak signal. Pass a list as
    ``candidates_out`` to receive the rejected candidates (verdict
    ``shot``/``chance``) for the event engine.
    """
    cfg = config or GameStateConfig()
    candidates = detect_goal_candidates(
        ball_track, geometry, start_s, end_s, cfg,
        player_tracks=player_tracks, audio_envelope=audio_envelope,
    )
    kept: List[GoalEvent] = []
    for event in candidates:
        if event.verdict != "goal":
            LOGGER.info("goal candidate rejected (%s, confidence %.2f): t=%.2fs side=%s (%s)",
                        event.verdict, event.confidence, event.t, event.side, event.reason)
            if candidates_out is not None:
                candidates_out.append(event)
            continue
        dup = next((k for k in kept if k.side == event.side
                    and abs(k.t - event.t) < cfg.goal_min_separation_s), None)
        if dup is not None:
            if event.confidence > dup.confidence:
                kept[kept.index(dup)] = event
            continue
        kept.append(event)
    for event in kept:
        LOGGER.info("goal flagged: t=%.2fs side=%s confidence=%.2f (%s)",
                    event.t, event.side, event.confidence, event.reason)
    return kept


# ---------------------------------------------------------------------------
# Game state timeline
# ---------------------------------------------------------------------------


def analyze_game_states(
    ball_track: BallTrack,
    geometry: FieldGeometry,
    start_s: float,
    end_s: float,
    config: Optional[GameStateConfig] = None,
    goal_events: Optional[Sequence[GoalEvent]] = None,
) -> List[GameStateSegment]:
    """Classify the timeline into game states with restart/goal holds.

    The key behavior: when the ball leaves play over a goal line (or vanishes
    right next to a goal), the state pins to ``restart_<side>`` until the ball
    is confirmed back in play - this is what keeps the camera at the goal
    while everyone waits for the goal kick or corner.
    """
    cfg = config or GameStateConfig()
    if end_s <= start_s:
        return []

    goals = sorted(goal_events or [], key=lambda e: e.t)
    out_margin = cfg.out_margin_frac * geometry.width

    steps = max(1, int(math.ceil((end_s - start_s) / cfg.step_s)))
    raw_states: List[Tuple[float, str, Optional[str], str]] = []  # (t, state, side, reason)

    state = STATE_IN_PLAY
    side: Optional[str] = None
    reason = "start of window"
    invisible_since: Optional[float] = None
    in_play_streak_start: Optional[float] = None
    restart_started_at: Optional[float] = None
    last_seen_xy: Optional[Tuple[float, float]] = None
    goal_idx = 0
    goal_hold_until: Optional[float] = None
    goal_side: Optional[str] = None

    for i in range(steps + 1):
        t = min(end_s, start_s + i * cfg.step_s)

        # Goal holds take priority over everything else.
        while goal_idx < len(goals) and goals[goal_idx].t <= t:
            goal_hold_until = goals[goal_idx].t + cfg.goal_hold_s
            goal_side = goals[goal_idx].side
            goal_idx += 1
        if goal_hold_until is not None and t <= goal_hold_until:
            state = STATE_GOAL_LEFT if goal_side == "left" else STATE_GOAL_RIGHT
            side = goal_side
            reason = f"goal scored at {goal_side} goal - holding on goal celebration/kickoff"
            raw_states.append((t, state, side, reason))
            invisible_since = None
            in_play_streak_start = None
            restart_started_at = None
            continue
        if goal_hold_until is not None and t > goal_hold_until:
            goal_hold_until = None
            goal_side = None
            state = STATE_IN_PLAY
            reason = "goal hold released"
            side = None

        pos = ball_track.position_at(t)
        if pos is not None:
            x, y, _source = pos
            last_seen_xy = (x, y)
            invisible_since = None
            beyond_left = x < geometry.x_min - out_margin
            beyond_right = x > geometry.x_max + out_margin
            beyond_touch = y < geometry.y_min - out_margin or y > geometry.y_max + out_margin
            in_field = not (beyond_left or beyond_right or beyond_touch)

            if in_field:
                if in_play_streak_start is None:
                    in_play_streak_start = t
                if state in RESTART_STATES:
                    held_long_enough = (t - in_play_streak_start) >= cfg.return_confirm_s
                    hit_cap = restart_started_at is not None and (t - restart_started_at) >= cfg.max_restart_hold_s
                    if held_long_enough or hit_cap:
                        state = STATE_IN_PLAY
                        side = None
                        reason = "ball back in play"
                        restart_started_at = None
                else:
                    state = STATE_IN_PLAY
                    side = None
                    reason = "ball visible in field"
            else:
                in_play_streak_start = None
                if beyond_left or beyond_right:
                    new_side = "left" if beyond_left else "right"
                    if state not in RESTART_STATES or side != new_side:
                        restart_started_at = t
                    state = STATE_RESTART_LEFT if new_side == "left" else STATE_RESTART_RIGHT
                    side = new_side
                    reason = (
                        f"ball out over the {new_side} goal line - waiting for goal kick/corner, "
                        "holding camera at the goal"
                    )
                else:
                    if state != STATE_RESTART_TOUCHLINE:
                        restart_started_at = t
                    state = STATE_RESTART_TOUCHLINE
                    side = None
                    reason = "ball out over the touchline - waiting for throw-in"
        else:
            in_play_streak_start = None
            if invisible_since is None:
                invisible_since = t
            invisible_for = t - invisible_since
            if state in RESTART_STATES or state in GOAL_STATES:
                # Keep holding; the ball being invisible is expected while
                # someone fetches it.
                if restart_started_at is not None and (t - restart_started_at) >= cfg.max_restart_hold_s:
                    state = STATE_BALL_LOST
                    side = None
                    reason = "restart hold exceeded safety cap - reverting to ball_lost"
                    restart_started_at = None
            elif invisible_for >= cfg.lost_grace_s:
                near_side = None
                if last_seen_xy is not None:
                    near_side = geometry.side_if_near_goal(
                        last_seen_xy[0], last_seen_xy[1], cfg.near_goal_frac
                    )
                if near_side is not None:
                    if state not in RESTART_STATES or side != near_side:
                        restart_started_at = t
                    state = STATE_RESTART_LEFT if near_side == "left" else STATE_RESTART_RIGHT
                    side = near_side
                    reason = (
                        f"ball vanished near the {near_side} goal - assuming goal kick/corner wait, "
                        "holding camera at the goal"
                    )
                else:
                    state = STATE_BALL_LOST
                    side = None
                    reason = f"ball not visible for {invisible_for:.1f}s - following player cluster"

        raw_states.append((t, state, side, reason))

    # Collapse consecutive identical states into segments.
    segments: List[GameStateSegment] = []
    for t, st, sd, rsn in raw_states:
        if segments and segments[-1].state == st and segments[-1].side == sd:
            segments[-1].end_s = t
        else:
            if segments:
                segments[-1].end_s = t
            segments.append(GameStateSegment(start_s=t, end_s=t, state=st, side=sd, reason=rsn))
    if segments:
        segments[-1].end_s = end_s

    # Drop zero-length artifacts.
    segments = [seg for seg in segments if seg.end_s - seg.start_s > 1e-6]
    for seg in segments:
        LOGGER.debug(
            "game state %.1fs-%.1fs: %s%s (%s)",
            seg.start_s, seg.end_s, seg.state, f"[{seg.side}]" if seg.side else "", seg.reason,
        )
    return segments


@dataclass
class SetPieceEvent:
    """A dead-ball restart: the ball sat still, then was kicked."""

    kind: str  # corner_kick | free_kick | penalty_kick | goal_kick | kickoff
    t_start: float  # when the ball became stationary
    t_kick: float  # when it accelerated away
    x: float
    y: float
    side: Optional[str]  # threatened goal ("left"/"right"), if any
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "t_start": round(self.t_start, 3),
            "t_kick": round(self.t_kick, 3),
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "side": self.side,
            "reason": self.reason,
        }


def detect_set_pieces(
    ball_track: BallTrack,
    geometry: FieldGeometry,
    start_s: float,
    end_s: float,
    config: Optional[GameStateConfig] = None,
) -> List[SetPieceEvent]:
    """Find dead-ball restarts from the stationary-ball + kick signature.

    A set piece is a window where the visible ball stays inside a small
    radius for a minimum time and then accelerates away. The location of the
    stationary spot classifies it: field corner -> corner kick, in front of a
    goal on the penalty spot -> penalty, inside the goal-kick zone -> goal
    kick, center circle -> kickoff, anywhere else -> free kick (with the
    threatened goal recorded when it is within shooting range).
    """
    cfg = config or GameStateConfig()
    events: List[SetPieceEvent] = []
    n = len(ball_track)
    if n < 3:
        return events

    frame_w = ball_track.frame_size[0]
    radius = cfg.set_piece_stationary_radius_frac * frame_w
    kick_speed = cfg.set_piece_kick_speed_frame_widths_per_s * frame_w
    times, xs, ys = ball_track.times, ball_track.xs, ball_track.ys
    center_x = (geometry.x_min + geometry.x_max) / 2.0
    center_y = (geometry.y_min + geometry.y_max) / 2.0

    def _classify(x: float, y: float) -> Tuple[str, Optional[str]]:
        corner_r = cfg.corner_radius_frac * geometry.width
        for cx in (geometry.x_min, geometry.x_max):
            for cy in (geometry.y_min, geometry.y_max):
                if math.hypot(x - cx, y - cy) <= corner_r:
                    return "corner_kick", "left" if cx == geometry.x_min else "right"
        for goal in (geometry.left_goal, geometry.right_goal):
            line_x = goal.x2 if goal.side == "left" else goal.x1
            depth = abs(x - line_x)
            toward_field = (x > line_x) if goal.side == "left" else (x < line_x)
            if not toward_field:
                continue
            gy = goal.center[1]
            lo, hi = cfg.penalty_depth_range_frac
            if lo * geometry.width <= depth <= hi * geometry.width and abs(y - gy) <= cfg.penalty_half_height_frac * geometry.height:
                return "penalty_kick", goal.side
            if depth <= cfg.goal_kick_zone_depth_frac * geometry.width and abs(y - gy) <= cfg.goal_kick_zone_half_height_frac * geometry.height:
                return "goal_kick", goal.side
        if math.hypot(x - center_x, y - center_y) <= cfg.kickoff_center_frac * geometry.width:
            return "kickoff", None
        threat = cfg.free_kick_threat_frac * geometry.width
        side = None
        if x - geometry.x_min <= threat:
            side = "left"
        elif geometry.x_max - x <= threat:
            side = "right"
        return "free_kick", side

    i = 0
    while i < n - 1:
        if times[i] < start_s:
            i += 1
            continue
        if times[i] > end_s:
            break
        # Grow a stationary window anchored at sample i.
        j = i + 1
        anchor_x, anchor_y = float(xs[i]), float(ys[i])
        while j < n and times[j] <= end_s:
            if (times[j] - times[j - 1]) > ball_track.config.max_interpolation_gap_s:
                break
            if math.hypot(float(xs[j]) - anchor_x, float(ys[j]) - anchor_y) > radius:
                break
            j += 1
        window_len = float(times[j - 1] - times[i])
        if window_len >= cfg.set_piece_min_stationary_s and j < n:
            t_kick = float(times[j - 1])
            # Probe twice after the window ends: a kicked ball is still
            # accelerating, so the later probe catches slower restarts.
            speeds = [
                math.hypot(*ball_track.velocity_at(t_kick + 0.3, window_s=0.5)),
                math.hypot(*ball_track.velocity_at(t_kick + 0.7, window_s=0.5)),
            ]
            if max(speeds) >= kick_speed:
                kind, side = _classify(anchor_x, anchor_y)
                reason = f"{kind.replace('_', ' ')} detected: ball held still {window_len:.1f}s then kicked"
                if side is not None and kind in {"corner_kick", "free_kick", "penalty_kick"}:
                    reason += f"; threatens the {side} goal"
                events.append(
                    SetPieceEvent(
                        kind=kind, t_start=float(times[i]), t_kick=t_kick,
                        x=anchor_x, y=anchor_y, side=side, reason=reason,
                    )
                )
                LOGGER.info("set piece: %s at t=%.1fs-%.1fs (%.0f, %.0f) side=%s",
                            kind, float(times[i]), t_kick, anchor_x, anchor_y, side)
                i = j
                continue
        i += 1
    return events


def overlay_set_piece_states(
    segments: List[GameStateSegment],
    set_pieces: Sequence[SetPieceEvent],
) -> List[GameStateSegment]:
    """Carve corner/free-kick setup states into the base state timeline.

    Goal celebrations keep priority; set-piece setup windows replace whatever
    other state covered [t_start, t_kick] so the camera planner can frame the
    restart (ball AND threatened goal in view).
    """
    overlays: List[GameStateSegment] = []
    for sp in set_pieces:
        if sp.kind == "corner_kick" and sp.side:
            state = STATE_CORNER_SETUP
        elif sp.kind in {"free_kick", "penalty_kick"} and sp.side:
            state = STATE_FREE_KICK_SETUP
        else:
            continue  # goal kicks/kickoffs already behave correctly
        overlays.append(GameStateSegment(
            start_s=sp.t_start, end_s=sp.t_kick, state=state, side=sp.side,
            reason=sp.reason,
        ))

    result = list(segments)
    for overlay in sorted(overlays, key=lambda s: s.start_s):
        updated: List[GameStateSegment] = []
        for seg in result:
            if seg.state in GOAL_STATES or seg.end_s <= overlay.start_s or seg.start_s >= overlay.end_s:
                updated.append(seg)
                continue
            if seg.start_s < overlay.start_s:
                updated.append(GameStateSegment(seg.start_s, overlay.start_s, seg.state, seg.side, seg.reason))
            updated.append(GameStateSegment(
                max(seg.start_s, overlay.start_s), min(seg.end_s, overlay.end_s),
                overlay.state, overlay.side, overlay.reason,
            ))
            if seg.end_s > overlay.end_s:
                updated.append(GameStateSegment(overlay.end_s, seg.end_s, seg.state, seg.side, seg.reason))
        result = updated

    # Merge adjacent identical states created by the splitting.
    merged: List[GameStateSegment] = []
    for seg in sorted(result, key=lambda s: s.start_s):
        if seg.end_s - seg.start_s <= 1e-6:
            continue
        if merged and merged[-1].state == seg.state and merged[-1].side == seg.side \
                and abs(merged[-1].end_s - seg.start_s) < 1e-6:
            merged[-1].end_s = seg.end_s
        else:
            merged.append(seg)
    return merged


def state_at(segments: Sequence[GameStateSegment], t: float) -> Optional[GameStateSegment]:
    """Return the segment covering time ``t`` (or the last one before it)."""
    current: Optional[GameStateSegment] = None
    for seg in segments:
        if seg.start_s <= t:
            current = seg
        else:
            break
        if t < seg.end_s:
            return seg
    return current


def summarize_states(segments: Sequence[GameStateSegment]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for seg in segments:
        totals[seg.state] = totals.get(seg.state, 0.0) + (seg.end_s - seg.start_s)
    return {state: round(duration, 2) for state, duration in sorted(totals.items())}
