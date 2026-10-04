"""Game-centric virtual camera planning (planner v2).

Given a cleaned ball track, the players, the field geometry, and the
game-state timeline, this module plans one camera decision per output frame:
where the virtual camera points ("the center of the game"), how far it zooms,
and - crucially - *why*. Every decision carries a human-readable reason plus
the underlying evidence so the plan can be rendered as a debug overlay,
dumped as training data, and audited frame by frame.

Camera behaviour by game state:

* ``in_play``      - follow a ball-led "play center": the ball's lead point
                     blended with the players engaged in the play, framed
                     wide enough to show them (team-shape aware when
                     per-player tracks are available).
* ``ball_lost``    - hold briefly, then ease toward the player cluster and
                     zoom out until the ball is found again.
* ``restart_*``    - lock onto the relevant goal (or the exit point) and do
                     not drift away until the ball is back in play.
* ``goal_*``       - hold on that goal.
* focus player     - ``focus_track_id`` follows one player, blended with the
                     ball, framed so player and ball both fit.

Motion architecture (why the v1 camera made people sick, and the fix):

1. Raw per-frame targets and desired zooms are computed first. Operator
   deadband is applied to the *setpoint* (it creates steps, which the later
   low-pass stages turn into smooth moves). State transitions are eased over
   ``transition_s`` with an ease-in-out curve; a hard *cut* is used only when
   the target jumps > ``cut_min_jump_frac`` of the frame width after the
   ball was lost for > ``cut_min_ball_lost_s``.
2. Zoom is a slow variable: a hysteresis setpoint (changes only when the
   required zoom differs by > 12 % for > 1 s, at most once per 4 s dwell),
   slew-rate limited (0.12x/s) and zero-phase smoothed.
3. Pan: confidence-weighted zero-phase Gaussian low-pass, then a
   critically-damped velocity/acceleration limiter run forward and backward
   (averaged, so it has no net lag and both limits still hold).
4. Hard framing constraints (ball / goal / anchors in frame with margin) are
   enforced by solving for the minimum zoom-OUT given the smoothed pan - a
   rate-limited lower envelope that is guaranteed to stay below the
   constraint - never by snapping the pan. smooth -> constrain -> smooth is
   iterated; residual violations get an explicit, rate-limited pan
   correction (not a clamp).
5. Crop rectangles are emitted as even integers with size hysteresis (the
   crop size only changes when it moves >= 2 px), center and size rounded
   separately, and are written as an ffmpeg ``sendcmd`` script for the
   ffmpeg-native renderer.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .follow_cam import _clamp_center  # noqa: F401  (re-exported for callers)
from .game_tracking import (
    BALL_SOURCE_DETECTED,
    GOAL_STATES,
    RESTART_STATES,
    SET_PIECE_STATES,
    STATE_CORNER_SETUP,
    STATE_IN_PLAY,
    STATE_RESTART_TOUCHLINE,
    BallTrack,
    FieldGeometry,
    GameStateSegment,
)

LOGGER = logging.getLogger("videohighlights.camera_planner")

DEFAULT_OUTPUT_SIZE: Tuple[int, int] = (1920, 1080)

STYLE_ZOOM_SCALE: Dict[str, float] = {"broadcast": 1.0, "tight": 1.18, "wide": 0.8}


@dataclass
class CameraPlannerConfig:
    # How far ahead of the ball (in seconds of ball velocity) to aim.
    lead_time_s: float = 0.35
    # Blend between the ball lead point and the engaged-player centroid when
    # only an anonymous player point cloud is available. Kept LOW: the ball
    # outranks the player cluster.
    action_blend: float = 0.12
    # Same blend when per-player tracks are available (team-shape aware).
    team_shape_blend: float = 0.25
    # Players within this fraction of the frame width around the ball count
    # as part of "the action".
    action_radius_frac: float = 0.22
    # Number of nearest engaged players fitted into the shot (tracks only).
    engaged_players: int = 4
    # After losing the ball, hold the camera this long before drifting.
    hold_last_s: float = 1.2
    # Pan low-pass: Gaussian sigma (seconds) of the zero-phase smoother.
    smooth_time_constant_s: float = 0.55
    # Zoom low-pass: smoothing radius (seconds) for the zoom path.
    zoom_smooth_time_constant_s: float = 1.2
    # Hard motion limits (crop-widths per second / per second^2).
    max_pan_speed_crop_frac: float = 1.0
    max_pan_accel_crop_frac: float = 1.2
    # --- Goal-threat framing ---
    threat_zoom_dist_frac: float = 0.32
    threat_goal_blend_max: float = 0.35
    threat_tighten_scale: float = 1.3
    # Margin (fraction of the full-frame crop) kept around point pairs when
    # zooming to fit them.
    both_in_frame_margin_frac: float = 0.12
    # Operator deadband on the aim SETPOINT (fraction of frame width) and the
    # raw zoom request; the low-pass stages smooth the resulting steps.
    deadband_frac: float = 0.02
    zoom_deadband: float = 0.05
    # Zoom levels relative to the configured base zoom.
    lost_zoom_scale: float = 0.8
    restart_zoom_cap: float = 1.35
    goal_zoom_scale: float = 1.0
    min_zoom: float = 1.0
    # Absolute zoom ceiling. The effective ceiling is further bounded so no
    # output pixel is upscaled (source/output size ratio).
    max_zoom: float = 3.0
    # Fast-ball zoom-out (feeds the zoom request; hysteresis stops pumping).
    fast_ball_speed_frame_widths_per_s: float = 1.2
    fast_ball_zoom_out_frac: float = 0.25
    # Aim slightly infield from the goal center during restarts.
    goal_infield_offset_frac: float = 0.05
    # Player-position time bin used for centroid lookups (point cloud input).
    player_bin_s: float = 0.5
    # --- v2: zoom as a slow variable ---
    zoom_hysteresis_frac: float = 0.12
    zoom_hysteresis_s: float = 1.0
    zoom_min_dwell_s: float = 4.0
    max_zoom_rate_per_s: float = 0.12
    # Zoom-out rate allowed when a framing constraint forces it (the plan is
    # offline, so the zoom-out starts early instead of reacting late).
    constraint_zoom_rate_per_s: float = 0.14
    # Anchors must sit at least this fraction of the crop size inside the
    # crop edge (unless the crop already touches that frame edge).
    keep_margin_frac: float = 0.06
    # --- v2: transitions ---
    transition_s: float = 1.5
    transition_jump_frac: float = 0.05
    cut_min_jump_frac: float = 0.6
    cut_min_ball_lost_s: float = 2.0
    # smooth -> constrain -> smooth iterations.
    smoothing_passes: int = 3
    # Focus-player mode: ball weight in the aim point and max ball distance
    # (fraction of frame width) for which both are framed together.
    focus_ball_weight: float = 0.35
    focus_ball_max_dist_frac: float = 0.45
    focus_player_max_gap_s: float = 1.0
    # Integer crop sizes only change when the float size moves this much.
    size_hold_px: float = 2.0


@dataclass(slots=True)
class CameraDecision:
    """One per output frame: where the camera points and why.

    slots=True matters here: an hour of 60fps video produces ~216k of these,
    and slotted instances roughly halve the per-decision memory.
    """

    index: int
    t: float
    center_x: float
    center_y: float
    zoom: float
    state: str
    focus: str  # ball | ball_lead | ball_goal_threat | action_centroid | goal_left | goal_right | exit_point | hold | set_piece | player | cut | frame_center
    reason: str
    confidence: float
    ball_x: Optional[float] = None
    ball_y: Optional[float] = None
    ball_source: Optional[str] = None
    target_x: Optional[float] = None
    target_y: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "t": round(self.t, 3),
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "zoom": round(self.zoom, 3),
            "state": self.state,
            "focus": self.focus,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "ball_x": round(self.ball_x, 2) if self.ball_x is not None else None,
            "ball_y": round(self.ball_y, 2) if self.ball_y is not None else None,
            "ball_source": self.ball_source,
            "target_x": round(self.target_x, 2) if self.target_x is not None else None,
            "target_y": round(self.target_y, 2) if self.target_y is not None else None,
        }


# ---------------------------------------------------------------------------
# Crop geometry helpers
# ---------------------------------------------------------------------------


def _even(value: float) -> int:
    return int(2 * round(float(value) / 2.0))


def resolve_output_size(
    frame_size: Tuple[int, int], output_size: Optional[Tuple[int, int]] = None
) -> Tuple[int, int]:
    """Output size for a plan/render.

    Explicit sizes are used as given (made even). ``None`` means "1080p-class
    output with the source aspect, never larger than the source".
    """
    if output_size is not None:
        ow, oh = int(output_size[0]), int(output_size[1])
        if ow <= 0 or oh <= 0:
            raise ValueError("output_size must be positive")
        return max(2, ow - ow % 2), max(2, oh - oh % 2)
    fw, fh = max(2, int(frame_size[0])), max(2, int(frame_size[1]))
    oh = min(DEFAULT_OUTPUT_SIZE[1], fh)
    ow = int(round(fw * oh / fh))
    return max(2, ow - ow % 2), max(2, oh - oh % 2)


def base_crop_size(frame_size: Tuple[int, int], output_size: Tuple[int, int]) -> Tuple[float, float]:
    """Largest crop with the OUTPUT aspect that fits the frame (zoom 1.0)."""
    fw, fh = float(frame_size[0]), float(frame_size[1])
    aspect = float(output_size[0]) / float(output_size[1])
    if fw / fh > aspect:
        return fh * aspect, fh
    return fw, fw / aspect


def crop_rects_from_centers(
    centers_x: np.ndarray,
    centers_y: np.ndarray,
    zooms: np.ndarray,
    frame_size: Tuple[int, int],
    output_size: Tuple[int, int],
    size_hold_px: float = 2.0,
) -> np.ndarray:
    """Float camera path -> even integer crop rects ``[x, y, w, h]`` (int32).

    Size and position are rounded separately; the size only changes when the
    float size moved >= ``size_hold_px`` from the held integer size, so it
    never flickers by +-2 px around a rounding boundary. Height is derived
    from width so the aspect stays locked to the output aspect.
    """
    fw, fh = int(frame_size[0]), int(frame_size[1])
    bw, _ = base_crop_size(frame_size, output_size)
    aspect = float(output_size[1]) / float(output_size[0])
    n = len(zooms)
    rects = np.empty((n, 4), dtype=np.int32)
    held_w: Optional[int] = None
    max_w = fw - fw % 2
    max_h = fh - fh % 2
    for i in range(n):
        z = max(1e-6, float(zooms[i]))
        w_f = min(bw / z, float(max_w))
        if held_w is None or abs(w_f - held_w) >= size_hold_px:
            held_w = max(2, min(max_w, _even(w_f)))
        w = held_w
        h = max(2, min(max_h, _even(w * aspect)))
        x = _even(float(centers_x[i]) - w / 2.0)
        y = _even(float(centers_y[i]) - h / 2.0)
        x = min(max(0, x), max(0, fw - w))
        y = min(max(0, y), max(0, fh - h))
        rects[i] = (x, y, w, h)
    return rects


def sendcmd_lines(
    rects: np.ndarray,
    fps: float,
    *,
    time_origin_frames: float = 0.0,
    lead_frames: float = 0.5,
    target: str = "crop",
) -> List[str]:
    """ffmpeg ``sendcmd`` lines for ``rects`` (one per rect *change*).

    Frame ``j`` of ``rects`` is expected at filter time
    ``(j + time_origin_frames) / fps``; its command is scheduled
    ``lead_frames`` earlier so it lands between the previous frame and this
    one regardless of timestamp rounding. Syntax (verified on ffmpeg 6.1):
    ``0.000 crop w 1920, crop h 1080, crop x 100, crop y 50;``.
    """
    lines: List[str] = []
    prev: Optional[Tuple[int, int, int, int]] = None
    for j in range(len(rects)):
        x, y, w, h = (int(v) for v in rects[j])
        cur = (x, y, w, h)
        if cur == prev:
            continue
        t = max(0.0, (j + time_origin_frames - lead_frames) / fps) if j > 0 else 0.0
        lines.append(f"{t:.4f} {target} w {w}, {target} h {h}, {target} x {x}, {target} y {y};")
        prev = cur
    return lines


@dataclass
class CameraPlan:
    start_seconds: float
    fps: float
    frame_size: Tuple[int, int]
    base_zoom: float
    decisions: List[CameraDecision] = field(default_factory=list)
    # v2 fields (optional so hand-built plans keep working).
    output_size: Optional[Tuple[int, int]] = None
    crop_rects: Optional[np.ndarray] = None  # int32 [n, 4] = x, y, w, h (source px, even)
    cuts: List[int] = field(default_factory=list)  # decision indices that are hard cuts
    max_zoom: Optional[float] = None
    style: str = "broadcast"

    def __len__(self) -> int:
        return len(self.decisions)

    # ------------------------------------------------------------------
    def resolved_output_size(self) -> Tuple[int, int]:
        return resolve_output_size(self.frame_size, self.output_size)

    def get_crop_rects(self, output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Integer crop rects ``[x, y, w, h]`` per decision.

        Uses the planner's rects when present (and the aspect matches),
        otherwise derives them from the decisions' float centers and zooms.
        """
        out = resolve_output_size(self.frame_size, output_size or self.output_size)
        if self.crop_rects is not None and len(self.crop_rects) == len(self.decisions):
            if output_size is None or tuple(out) == tuple(self.resolved_output_size()):
                return self.crop_rects
        xs = np.array([d.center_x for d in self.decisions], dtype=np.float64)
        ys = np.array([d.center_y for d in self.decisions], dtype=np.float64)
        zs = np.array([d.zoom for d in self.decisions], dtype=np.float64)
        return crop_rects_from_centers(xs, ys, zs, self.frame_size, out)

    def sendcmd_text(self, time_offset: float = 0.0) -> str:
        """The sendcmd script with times relative to ``start_seconds + time_offset``."""
        rects = self.get_crop_rects()
        origin = -float(time_offset) * self.fps
        return "\n".join(sendcmd_lines(rects, self.fps, time_origin_frames=origin)) + "\n"

    def write_sendcmd(self, path: str, time_offset: float = 0.0) -> str:
        """Write ``camera_crops.txt`` (ffmpeg ``sendcmd`` script, docs/ARTIFACTS.md).

        Times are seconds from the plan start (= the processing window start
        for a full plan), each command half a frame ahead of its frame, and a
        command is only written when the crop changes. Coordinates are even
        integers in source-video pixels.
        """
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(self.sendcmd_text(time_offset))
        return path

    def write_jsonl(
        self,
        path: str,
        transform: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
    ) -> str:
        """Write one JSON object per decision; ``transform`` can enrich rows
        (e.g. adding source-video timestamps) so there is exactly one
        serialization of the training-data format."""
        with open(path, "w", encoding="utf-8") as handle:
            for decision in self.decisions:
                row = decision.to_dict()
                if transform is not None:
                    row = transform(row)
                handle.write(json.dumps(row) + "\n")
        return path

    def summary(self) -> Dict[str, object]:
        focus_counts: Dict[str, int] = {}
        for d in self.decisions:
            focus_counts[d.focus] = focus_counts.get(d.focus, 0) + 1
        zooms = [d.zoom for d in self.decisions]
        return {
            "frames": len(self.decisions),
            "fps": round(self.fps, 3),
            "base_zoom": round(self.base_zoom, 3),
            "max_zoom": round(float(self.max_zoom), 3) if self.max_zoom is not None else None,
            "output_size": list(self.resolved_output_size()),
            "zoom_min": round(float(min(zooms)), 3) if zooms else None,
            "zoom_max": round(float(max(zooms)), 3) if zooms else None,
            "cuts": len(self.cuts),
            "focus_frame_counts": focus_counts,
            "mean_confidence": round(
                float(np.mean([d.confidence for d in self.decisions])) if self.decisions else 0.0, 3
            ),
        }


def slice_plan(plan: CameraPlan, start_seconds: float, end_seconds: float) -> CameraPlan:
    """Cut a sub-plan covering [start_seconds, end_seconds) from a full plan.

    Decisions sit on the plan's fps grid, so slicing is an index range; the
    returned plan is re-indexed and starts at the sliced start time.
    """
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    first = max(0, int(round((start_seconds - plan.start_seconds) * plan.fps)))
    last = min(len(plan.decisions), int(round((end_seconds - plan.start_seconds) * plan.fps)))
    rects = None
    if plan.crop_rects is not None and len(plan.crop_rects) == len(plan.decisions):
        rects = np.array(plan.crop_rects[first:last], dtype=np.int32, copy=True)
    sliced = CameraPlan(
        start_seconds=plan.start_seconds + first / plan.fps,
        fps=plan.fps,
        frame_size=plan.frame_size,
        base_zoom=plan.base_zoom,
        output_size=plan.output_size,
        crop_rects=rects,
        cuts=[c - first for c in plan.cuts if first <= c < last],
        max_zoom=plan.max_zoom,
        style=plan.style,
    )
    for new_index, decision in enumerate(plan.decisions[first:last]):
        sliced.decisions.append(replace(decision, index=new_index))
    return sliced


# ---------------------------------------------------------------------------
# Player lookups
# ---------------------------------------------------------------------------


class _PlayerLookup:
    """Time-binned anonymous player-position lookup (legacy point cloud)."""

    def __init__(self, player_positions: Optional[np.ndarray], bin_s: float) -> None:
        self.bin_s = max(0.1, float(bin_s))
        self.bins: Dict[int, np.ndarray] = {}
        if player_positions is None or len(player_positions) == 0:
            return
        arr = np.asarray(player_positions, dtype=np.float64)
        keys = (arr[:, 0] / self.bin_s).astype(np.int64)
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        arr = arr[order]
        boundaries = np.flatnonzero(np.diff(keys)) + 1
        for chunk_keys, chunk in zip(np.split(keys, boundaries), np.split(arr, boundaries)):
            if len(chunk_keys):
                self.bins[int(chunk_keys[0])] = chunk[:, 1:3]

    def __bool__(self) -> bool:
        return bool(self.bins)

    def positions_near(self, t: float) -> Optional[np.ndarray]:
        key = int(t / self.bin_s)
        chunks = [self.bins[k] for k in (key - 1, key, key + 1) if k in self.bins]
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)

    def centroid(self, t: float, around: Optional[Tuple[float, float]] = None,
                 radius: Optional[float] = None) -> Optional[Tuple[float, float]]:
        positions = self.positions_near(t)
        if positions is None or len(positions) == 0:
            return None
        if around is not None and radius is not None:
            deltas = positions - np.asarray(around, dtype=np.float64)
            mask = np.hypot(deltas[:, 0], deltas[:, 1]) <= radius
            if mask.sum() >= 2:
                positions = positions[mask]
        return float(np.mean(positions[:, 0])), float(np.mean(positions[:, 1]))

    def engaged(self, t: float, around: Tuple[float, float], radius: float,
                count: int) -> Optional[np.ndarray]:
        """Points of players engaged in the play (robust box, no identities)."""
        positions = self.bins.get(int(t / self.bin_s))
        if positions is None or len(positions) == 0:
            return None
        deltas = positions - np.asarray(around, dtype=np.float64)
        mask = np.hypot(deltas[:, 0], deltas[:, 1]) <= radius
        pts = positions[mask]
        if len(pts) < 2:
            return None
        if len(pts) > 6:
            lo = np.percentile(pts, 10, axis=0)
            hi = np.percentile(pts, 90, axis=0)
            pts = np.vstack([lo, hi, pts.mean(axis=0, keepdims=True)])
        return pts[: max(2, int(count) + 3)]


class _TrackGrid:
    """Per-player positions on a 10 Hz grid (from a TrackingResult)."""

    def __init__(self, tracking, t0: float, t1: float, hz: float = 10.0,
                 max_gap_s: float = 0.6) -> None:
        self.hz = float(hz)
        self.g0 = int(math.floor(t0 * hz)) - 1
        g1 = int(math.ceil(t1 * hz)) + 1
        grid_t = np.arange(self.g0, g1 + 1, dtype=np.float64) / hz
        players = [tr for tr in tracking.players.values() if len(tr)]
        self.track_ids = np.array([int(tr.track_id) for tr in players], dtype=np.int64)
        self.teams = np.array([int(tr.team) for tr in players], dtype=np.int64)
        self.pos = np.full((len(grid_t), len(players), 2), np.nan, dtype=np.float32)
        for k, tr in enumerate(players):
            tt = np.asarray(tr.t, dtype=np.float64)
            centers = tr.center_xy().astype(np.float64)
            valid = _valid_by_gap(tt, grid_t, max_gap_s)
            self.pos[valid, k, 0] = np.interp(grid_t[valid], tt, centers[:, 0])
            self.pos[valid, k, 1] = np.interp(grid_t[valid], tt, centers[:, 1])

    def __bool__(self) -> bool:
        return self.pos.shape[1] > 0

    def _row(self, t: float) -> Optional[np.ndarray]:
        g = int(round(t * self.hz)) - self.g0
        if g < 0 or g >= self.pos.shape[0]:
            return None
        return self.pos[g]

    def centroid(self, t: float, around: Optional[Tuple[float, float]] = None,
                 radius: Optional[float] = None) -> Optional[Tuple[float, float]]:
        row = self._row(t)
        if row is None:
            return None
        mask = np.isfinite(row[:, 0]) & (self.teams != 2)
        pts = row[mask].astype(np.float64)
        if len(pts) == 0:
            return None
        if around is not None and radius is not None:
            d = np.hypot(pts[:, 0] - around[0], pts[:, 1] - around[1])
            near = pts[d <= radius]
            if len(near) >= 2:
                pts = near
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def engaged(self, t: float, around: Tuple[float, float], radius: float,
                count: int) -> Optional[np.ndarray]:
        row = self._row(t)
        if row is None:
            return None
        mask = np.isfinite(row[:, 0]) & (self.teams != 2)
        pts = row[mask].astype(np.float64)
        if len(pts) == 0:
            return None
        d = np.hypot(pts[:, 0] - around[0], pts[:, 1] - around[1])
        order = np.argsort(d)
        order = order[d[order] <= radius][: max(1, int(count))]
        if len(order) < 2:
            return None
        return pts[order]


def _valid_by_gap(sample_t: np.ndarray, query_t: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Queries whose nearest sample is within ``max_gap_s``."""
    if len(sample_t) == 0:
        return np.zeros(len(query_t), dtype=bool)
    idx = np.searchsorted(sample_t, query_t)
    left = sample_t[np.clip(idx - 1, 0, len(sample_t) - 1)]
    right = sample_t[np.clip(idx, 0, len(sample_t) - 1)]
    gap = np.minimum(np.abs(query_t - left), np.abs(right - query_t))
    return gap <= max_gap_s


def _segment_lookup(segments: Sequence[GameStateSegment]):
    """Return a stateful function mapping monotonically increasing t -> segment."""
    ordered = sorted(segments, key=lambda seg: seg.start_s)
    idx = 0

    def lookup(t: float) -> Optional[GameStateSegment]:
        nonlocal idx
        while idx + 1 < len(ordered) and t >= ordered[idx].end_s:
            idx += 1
        if not ordered:
            return None
        return ordered[idx]

    return lookup


# ---------------------------------------------------------------------------
# Signal-processing primitives (shared with follow_cam)
# ---------------------------------------------------------------------------


def _zero_phase_smooth(values: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Forward+backward exponential smoothing (zero phase lag). Kept for API
    compatibility; the v2 planner uses :func:`_gaussian_smooth`."""
    if tau <= 0 or len(values) < 3:
        return values.astype(np.float64, copy=True)
    alpha = dt / (tau + dt)
    fwd = np.empty(len(values), dtype=np.float64)
    acc = float(values[0])
    for i in range(len(values)):
        acc += alpha * (float(values[i]) - acc)
        fwd[i] = acc
    out = np.empty(len(values), dtype=np.float64)
    acc = fwd[-1]
    for i in range(len(values) - 1, -1, -1):
        acc += alpha * (fwd[i] - acc)
        out[i] = acc
    return out


def _gaussian_smooth(values: np.ndarray, sigma_frames: float,
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Zero-phase Gaussian low-pass (optionally confidence weighted)."""
    values = np.asarray(values, dtype=np.float64)
    if sigma_frames <= 0.05 or len(values) < 2:
        return values.copy()
    from scipy.ndimage import gaussian_filter1d

    if weights is None:
        return gaussian_filter1d(values, sigma_frames, mode="nearest", truncate=3.0)
    w = np.asarray(weights, dtype=np.float64)
    num = gaussian_filter1d(values * w, sigma_frames, mode="nearest", truncate=3.0)
    den = gaussian_filter1d(w, sigma_frames, mode="nearest", truncate=3.0)
    return num / np.maximum(den, 1e-9)


def _hann_kernel(radius: int) -> np.ndarray:
    k = np.hanning(2 * radius + 3)[1:-1]
    return k / k.sum()


def _lipschitz_below(values: np.ndarray, step: float) -> np.ndarray:
    """Largest function <= values whose per-frame change is <= step."""
    out = np.array(values, dtype=np.float64, copy=True)
    for i in range(1, len(out)):
        cap = out[i - 1] + step
        if out[i] > cap:
            out[i] = cap
    for i in range(len(out) - 2, -1, -1):
        cap = out[i + 1] + step
        if out[i] > cap:
            out[i] = cap
    return out


def _smooth_lower_envelope(values: np.ndarray, step: float, radius: int) -> np.ndarray:
    """Smooth, slope-limited curve guaranteed to stay <= ``values``.

    Lipschitz lower envelope -> erosion (min filter) of the kernel radius ->
    finite Hann average. Averaging an eroded signal over a window no wider
    than the erosion can never exceed the original, so constraints computed
    as an upper bound (max allowed zoom) are honoured exactly.
    """
    if len(values) < 3:
        return np.asarray(values, dtype=np.float64).copy()
    from scipy.ndimage import minimum_filter1d

    z = _lipschitz_below(values, step)
    if radius <= 0:
        return z
    eroded = minimum_filter1d(z, size=2 * radius + 1, mode="nearest")
    padded = np.pad(eroded, radius, mode="edge")
    return np.convolve(padded, _hann_kernel(radius), mode="valid")


def _smooth_upper_envelope(values: np.ndarray, step: float, radius: int) -> np.ndarray:
    return -_smooth_lower_envelope(-np.asarray(values, dtype=np.float64), step, radius)


def _track_limited(tx: np.ndarray, ty: np.ndarray, dt: float, vmax: np.ndarray,
                   amax: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Critically-damped tracker with speed/acceleration limits.

    Lands exactly on the target when that is physically allowed; otherwise
    approaches it along a time-optimal braking curve (no overshoot).
    """
    n = len(tx)
    ox = np.empty(n, dtype=np.float64)
    oy = np.empty(n, dtype=np.float64)
    px, py = float(tx[0]), float(ty[0])
    vx = vy = 0.0
    ox[0], oy[0] = px, py
    hypot = math.hypot
    sqrt = math.sqrt
    for i in range(1, n):
        ux = (tx[i] - tx[i - 1]) / dt
        uy = (ty[i] - ty[i - 1]) / dt
        ex = tx[i] - px
        ey = ty[i] - py
        wx, wy = ex / dt, ey / dt
        e = hypot(ex, ey)
        a_i = float(amax[i])
        if e > 1e-9:
            brake = sqrt(2.0 * a_i * e)
            if hypot(wx - ux, wy - uy) > brake:
                wx = ux + ex / e * brake
                wy = uy + ey / e * brake
        s = hypot(wx, wy)
        v_i = float(vmax[i])
        if s > v_i > 0:
            wx *= v_i / s
            wy *= v_i / s
        dvx, dvy = wx - vx, wy - vy
        dv = hypot(dvx, dvy)
        lim = a_i * dt
        if dv > lim > 0:
            dvx *= lim / dv
            dvy *= lim / dv
        vx += dvx
        vy += dvy
        px += vx * dt
        py += vy * dt
        ox[i], oy[i] = px, py
    return ox, oy


def _limit_motion(x: np.ndarray, y: np.ndarray, dt: float, vmax: np.ndarray,
                  amax: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-phase speed/accel limiter: average of a forward and a backward
    limited tracker. Each obeys both limits, so their average does too, and
    the forward lag cancels the backward lead."""
    if len(x) < 3:
        return np.asarray(x, dtype=np.float64).copy(), np.asarray(y, dtype=np.float64).copy()
    fx, fy = _track_limited(x, y, dt, vmax, amax)
    bx, by = _track_limited(x[::-1], y[::-1], dt, vmax[::-1], amax[::-1])
    return (fx + bx[::-1]) / 2.0, (fy + by[::-1]) / 2.0


def _clamp_centers(cx: np.ndarray, cy: np.ndarray, zooms: np.ndarray,
                   frame_size: Tuple[int, int], base: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    fw, fh = float(frame_size[0]), float(frame_size[1])
    half_w = base[0] / np.maximum(zooms, 1e-6) / 2.0
    half_h = base[1] / np.maximum(zooms, 1e-6) / 2.0
    x = np.minimum(np.maximum(cx, half_w), np.maximum(half_w, fw - half_w))
    y = np.minimum(np.maximum(cy, half_h), np.maximum(half_h, fh - half_h))
    return x, y


def _zoom_hysteresis(required: np.ndarray, fps: float, frac: float, hold_s: float,
                     dwell_s: float) -> Tuple[np.ndarray, int]:
    """Step-wise zoom setpoint with hysteresis + minimum dwell.

    The setpoint only changes when the required zoom has differed by more
    than ``frac`` (same direction) for longer than ``hold_s``; changes are at
    least ``dwell_s`` apart. Offline, the change is applied from the onset
    of the sustained deviation (no reaction lag). Returns (setpoint, changes).
    """
    n = len(required)
    if n == 0:
        return np.zeros(0), 0
    hold_n = max(1, int(round(hold_s * fps)))
    dwell_n = max(1, int(round(dwell_s * fps)))
    log_thr = math.log1p(frac)
    log_req = np.log(np.maximum(required, 1e-6))
    sp = float(np.exp(np.median(log_req[: max(1, min(n, hold_n))])))
    out = np.full(n, sp, dtype=np.float64)
    last_change = -(10 ** 9)
    run_start = -1
    run_dir = 0
    changes = 0
    log_sp = math.log(sp)
    for i in range(n):
        d = log_req[i] - log_sp
        direction = 1 if d > log_thr else (-1 if d < -log_thr else 0)
        if direction == 0:
            run_dir = 0
            continue
        if direction != run_dir:
            run_dir = direction
            run_start = i
        if i - run_start + 1 > hold_n and i - last_change >= dwell_n:
            onset = max(run_start, last_change + dwell_n)
            new_sp = float(np.exp(np.median(log_req[run_start: i + 1])))
            out[onset:] = new_sp
            log_sp = math.log(new_sp)
            last_change = onset
            changes += 1
            run_dir = 0
    return out, changes


def _slew(values: np.ndarray, step: float) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        if delta > step:
            out[i] = out[i - 1] + step
        elif delta < -step:
            out[i] = out[i - 1] - step
    return out


def _smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * (3.0 - 2.0 * s)


def _segments_from_cuts(n: int, cuts: Sequence[int]) -> List[Tuple[int, int]]:
    bounds = [0] + sorted(c for c in cuts if 0 < c < n) + [n]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def plan_camera(
    *,
    ball_track: BallTrack,
    player_positions: Optional[np.ndarray],
    geometry: FieldGeometry,
    segments: Sequence[GameStateSegment],
    start_seconds: float,
    end_seconds: float,
    fps: float,
    frame_size: Tuple[int, int],
    base_zoom: float = 1.6,
    config: Optional[CameraPlannerConfig] = None,
    output_size: Optional[Tuple[int, int]] = None,
    player_tracks=None,
    focus_track_id: Optional[int] = None,
    style: str = "broadcast",
) -> CameraPlan:
    """Plan one camera decision per frame for [start_seconds, end_seconds).

    ``output_size``: the render size. When given, the zoom ceiling is bounded
    so no output pixel is upscaled (4K -> 1080p allows zoom <= 2.0). When
    ``None`` the output is 1080p-class (never larger than the source); for
    sources larger than that output the same no-upscale bound applies, for
    sources that are not (e.g. a 1080p file) the bound would forbid any zoom,
    so only ``config.max_zoom`` applies.
    ``player_tracks``: a ``TrackingResult`` for team-shape framing and
    ``focus_track_id`` (follow one player).
    ``style``: ``broadcast`` | ``tight`` | ``wide``.
    """
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    if fps <= 0:
        raise ValueError("fps must be positive")

    cfg = config or CameraPlannerConfig()
    frame_w, frame_h = int(frame_size[0]), int(frame_size[1])
    out_size = resolve_output_size((frame_w, frame_h), output_size)
    base_w, base_h = base_crop_size((frame_w, frame_h), out_size)
    pixel_bound = min(base_w / out_size[0], base_h / out_size[1])
    if output_size is not None or pixel_bound > 1.0 + 1e-6:
        max_zoom = max(1.0, min(cfg.max_zoom, pixel_bound))
    else:
        max_zoom = max(1.0, cfg.max_zoom)
    min_zoom = max(1.0, min(cfg.min_zoom, max_zoom))
    style_key = str(style or "broadcast").strip().lower()
    style_scale = STYLE_ZOOM_SCALE.get(style_key, 1.0)

    def _cz(z: float) -> float:
        return min(max_zoom, max(min_zoom, float(z)))

    base_zoom = _cz(float(base_zoom) * style_scale)
    frame_count = max(1, int(math.ceil((end_seconds - start_seconds) * fps - 1e-9)))
    dt = 1.0 / fps
    times = start_seconds + np.arange(frame_count, dtype=np.float64) * dt

    lookup_points = _PlayerLookup(player_positions, cfg.player_bin_s)
    grid: Optional[_TrackGrid] = None
    if player_tracks is not None and getattr(player_tracks, "players", None):
        grid = _TrackGrid(player_tracks, start_seconds, end_seconds)
    players = grid if grid else lookup_points
    shape_blend = cfg.team_shape_blend if grid else cfg.action_blend
    seg_at = _segment_lookup(segments)
    action_radius = cfg.action_radius_frac * frame_w

    # Focus player positions on the frame grid.
    focus_xy: Optional[np.ndarray] = None
    focus_valid: Optional[np.ndarray] = None
    if focus_track_id is not None and player_tracks is not None:
        track = player_tracks.players.get(int(focus_track_id))
        if track is not None and len(track):
            tt = np.asarray(track.t, dtype=np.float64)
            centers = track.center_xy().astype(np.float64)
            focus_valid = _valid_by_gap(tt, times, cfg.focus_player_max_gap_s)
            focus_xy = np.column_stack([np.interp(times, tt, centers[:, 0]),
                                        np.interp(times, tt, centers[:, 1])])
        else:
            LOGGER.warning("focus track %s not found; planning a ball-led camera", focus_track_id)

    plan = CameraPlan(
        start_seconds=float(start_seconds),
        fps=float(fps),
        frame_size=(frame_w, frame_h),
        base_zoom=base_zoom,
        output_size=out_size,
        max_zoom=max_zoom,
        style=style_key,
    )

    lost_zoom = _cz(base_zoom * cfg.lost_zoom_scale)
    restart_zoom = _cz(min(base_zoom, cfg.restart_zoom_cap))
    goal_zoom = _cz(base_zoom * cfg.goal_zoom_scale)
    infield_offset = cfg.goal_infield_offset_frac * geometry.width
    margin = cfg.both_in_frame_margin_frac

    def _fit_zoom(points: np.ndarray, cap: float) -> float:
        """Widest-necessary zoom that keeps all points in the crop (+margin)."""
        half_w = (float(points[:, 0].max()) - float(points[:, 0].min())) / 2.0 + margin * base_w
        half_h = (float(points[:, 1].max()) - float(points[:, 1].min())) / 2.0 + margin * base_h
        fit = min(base_w / (2.0 * half_w), base_h / (2.0 * half_h))
        return _cz(min(cap, fit))

    # ------------------------------------------------------------------
    # Phase 1: one RAW aim setpoint + zoom request per frame, with reasons.
    # ------------------------------------------------------------------
    raw = np.empty((frame_count, 2), dtype=np.float64)
    raw_zoom = np.empty(frame_count, dtype=np.float64)
    conf = np.empty(frame_count, dtype=np.float64)
    ball_lost_s = np.empty(frame_count, dtype=np.float64)
    modes: List[Tuple[str, str]] = []
    metas: List[Tuple[str, str, str, float, Optional[Tuple[float, float, str]]]] = []
    keep_idx: List[int] = []
    keep_xy: List[Tuple[float, float]] = []
    in_play = np.zeros(frame_count, dtype=bool)

    deadband_px = cfg.deadband_frac * frame_w
    last_ball_seen_t: Optional[float] = None
    last_ball_xy: Optional[Tuple[float, float]] = None
    prev_key: Optional[Tuple[str, str]] = None
    prev_target: Optional[Tuple[float, float]] = None
    prev_zoom: Optional[float] = None
    trans_n = max(1, int(round(cfg.transition_s * fps)))
    mode_start = 0

    for index in range(frame_count):
        t = float(times[index])
        segment = seg_at(t)
        state = segment.state if segment is not None else STATE_IN_PLAY
        state_reason = segment.reason if segment is not None else ""
        in_play[index] = state == STATE_IN_PLAY

        ball = ball_track.position_at(t)
        if ball is not None:
            last_ball_seen_t = t
            last_ball_xy = (ball[0], ball[1])
        ball_lost_s[index] = (t - last_ball_seen_t) if last_ball_seen_t is not None else float("inf")

        target: Tuple[float, float]
        target_zoom = base_zoom
        focus = "frame_center"
        reason = state_reason or "no signal"
        confidence = 0.2
        keeps: List[Tuple[float, float]] = []
        # Non-ball anchors (goal, exit point) only bind once the eased
        # transition into the new state has finished; the ball binds always.
        ball_keeps: List[Tuple[float, float]] = []
        handled = False

        if focus_xy is not None and focus_valid is not None and focus_valid[index]:
            px, py = float(focus_xy[index, 0]), float(focus_xy[index, 1])
            target = (px, py)
            ball_keeps = [(px, py)]
            target_zoom = base_zoom
            focus = "player"
            confidence = 0.85
            reason = f"following player #{focus_track_id}"
            if ball is not None and math.hypot(ball[0] - px, ball[1] - py) <= cfg.focus_ball_max_dist_frac * frame_w:
                w_b = cfg.focus_ball_weight
                target = (px * (1.0 - w_b) + ball[0] * w_b, py * (1.0 - w_b) + ball[1] * w_b)
                target_zoom = _fit_zoom(np.array([[px, py], [ball[0], ball[1]]]), base_zoom)
                ball_keeps = [(px, py), (ball[0], ball[1])]
                reason += " with the ball in shot"
            handled = True

        if handled:
            pass
        elif state in GOAL_STATES and segment is not None:
            goal = geometry.goal_for_side(segment.side or "left")
            gx, gy = goal.center
            gx += infield_offset if goal.side == "left" else -infield_offset
            target = (gx, gy)
            target_zoom = goal_zoom
            focus = f"goal_{goal.side}"
            confidence = 0.9
            reason = state_reason or f"holding on {goal.side} goal after goal"
            keeps = [goal.center]
            if ball is not None:
                ball_keeps = [(ball[0], ball[1])]
        elif state in SET_PIECE_STATES and segment is not None:
            goal = geometry.goal_for_side(segment.side or "left")
            anchor = last_ball_xy if last_ball_xy is not None else goal.center
            if ball is not None:
                anchor = (ball[0], ball[1])
            target = (
                anchor[0] * 0.45 + goal.center[0] * 0.55,
                anchor[1] * 0.45 + goal.center[1] * 0.55,
            )
            target_zoom = _fit_zoom(np.array([anchor, goal.center], dtype=np.float64), base_zoom)
            focus = "set_piece"
            confidence = 0.85
            keeps = [goal.center]
            if ball is not None:
                ball_keeps = [anchor]
            else:
                keeps.append(anchor)
            if state == STATE_CORNER_SETUP:
                reason = state_reason or (
                    f"corner kick setup - wide framing of the corner and the {goal.side} goal"
                )
            else:
                reason = state_reason or (
                    f"free kick setup - keeping the {goal.side} goal in view during the run-up"
                )
        elif state in RESTART_STATES and segment is not None:
            if state == STATE_RESTART_TOUCHLINE:
                anchor = last_ball_xy or ((geometry.x_min + geometry.x_max) / 2.0,
                                          (geometry.y_min + geometry.y_max) / 2.0)
                target = anchor
                focus = "exit_point"
                confidence = 0.6
                reason = state_reason or "ball out over touchline - holding at exit point"
                keeps = [anchor]
            else:
                goal = geometry.goal_for_side(segment.side or "left")
                gx, gy = goal.center
                gx += infield_offset if goal.side == "left" else -infield_offset
                target = (gx, gy)
                focus = f"goal_{goal.side}"
                confidence = 0.75
                reason = state_reason or (
                    f"ball out near {goal.side} goal - staying on the goal until play restarts"
                )
                keeps = [goal.center]
            target_zoom = restart_zoom
            if ball is not None:
                # The visible ball stays in shot while we wait on the goal.
                ball_keeps = [(ball[0], ball[1])]
        elif ball is not None:
            bx, by, source = ball
            vx, vy = ball_track.velocity_at(t)
            lead_x = bx + vx * cfg.lead_time_s
            lead_y = by + vy * cfg.lead_time_s
            engaged = players.engaged(t, (bx, by), action_radius, cfg.engaged_players) if players else None
            if engaged is not None and shape_blend > 0.0:
                cx_, cy_ = float(engaged[:, 0].mean()), float(engaged[:, 1].mean())
                target = (
                    lead_x * (1.0 - shape_blend) + cx_ * shape_blend,
                    lead_y * (1.0 - shape_blend) + cy_ * shape_blend,
                )
                focus = "ball"
                reason = "following the play (ball lead + engaged players)"
            else:
                target = (lead_x, lead_y)
                focus = "ball_lead" if (abs(vx) + abs(vy)) > 1.0 else "ball"
                reason = "following ball"
            confidence = 0.92 if source == BALL_SOURCE_DETECTED else 0.6
            if source != BALL_SOURCE_DETECTED:
                reason += " (interpolated across a short detection gap)"
            ball_keeps = [(bx, by)]
            speed = math.hypot(vx, vy)
            speed_frac = min(1.0, speed / (cfg.fast_ball_speed_frame_widths_per_s * frame_w))
            target_zoom = _cz(base_zoom * (1.0 - cfg.fast_ball_zoom_out_frac * speed_frac))
            if engaged is not None:
                pts = np.vstack([engaged, [[bx, by]]])
                target_zoom = min(target_zoom, _fit_zoom(pts, base_zoom))
            # Goal-threat framing: aim between ball and goal and zoom so
            # both fit; the shot tightens as the ball closes in.
            threat_r = cfg.threat_zoom_dist_frac * geometry.width
            goal = geometry.left_goal if bx - geometry.x_min < geometry.x_max - bx else geometry.right_goal
            gx, gy = goal.center
            dist = math.hypot(bx - gx, by - gy)
            attacking = (vx < -10.0 if goal.side == "left" else vx > 10.0)
            if dist <= threat_r and (attacking or dist <= 0.55 * threat_r):
                closeness = 1.0 - (dist / threat_r)
                w = cfg.threat_goal_blend_max * closeness
                target = (target[0] * (1.0 - w) + gx * w, target[1] * (1.0 - w) + gy * w)
                target_zoom = _fit_zoom(np.array([[bx, by], [gx, gy]]), base_zoom * cfg.threat_tighten_scale)
                focus = "ball_goal_threat"
                reason = f"attacking the {goal.side} goal - framing ball and goal together"
                confidence = max(confidence, 0.9)
                ball_keeps = [(bx, by), (gx, gy)]
        else:
            recently_seen = (
                last_ball_seen_t is not None and (t - last_ball_seen_t) <= cfg.hold_last_s
            )
            if recently_seen and last_ball_xy is not None:
                target = last_ball_xy
                focus = "hold"
                confidence = 0.5
                reason = "ball just went out of sight - holding last known spot"
                target_zoom = base_zoom
            else:
                centroid = players.centroid(t, around=last_ball_xy, radius=action_radius * 2.0) if players else None
                if centroid is None and players:
                    centroid = players.centroid(t)
                if centroid is not None:
                    target = centroid
                    focus = "action_centroid"
                    confidence = 0.35
                    reason = (state_reason or "ball not visible") + " - following player cluster"
                else:
                    target = (frame_w / 2.0, frame_h / 2.0)
                    focus = "frame_center"
                    confidence = 0.1
                    reason = "no ball and no players visible - centering frame"
                target_zoom = lost_zoom
            if focus_xy is not None:
                reason += f" (player #{focus_track_id} not visible)"

        # Deadband on the SETPOINT: hold the previous aim for sub-threshold
        # changes within the same state/focus. Steps are fine here - every
        # later stage is a low-pass.
        key = (state, "ball" if focus.startswith("ball") else focus)
        if prev_key == key and prev_target is not None:
            if math.hypot(target[0] - prev_target[0], target[1] - prev_target[1]) < deadband_px:
                target = prev_target
            if prev_zoom is not None and abs(target_zoom - prev_zoom) < cfg.zoom_deadband:
                target_zoom = prev_zoom
        if key != prev_key:
            mode_start = index
        prev_key = key
        prev_target = target
        prev_zoom = target_zoom
        if index - mode_start < trans_n and index >= trans_n:
            keeps = []
        keeps = keeps + ball_keeps

        raw[index] = target
        raw_zoom[index] = _cz(target_zoom)
        conf[index] = confidence
        modes.append(key)
        metas.append((state, focus, reason, confidence, ball))
        for kp in keeps:
            keep_idx.append(index)
            keep_xy.append((float(kp[0]), float(kp[1])))

    # ------------------------------------------------------------------
    # Phase 2: transitions (eased) and deliberate hard cuts.
    # ------------------------------------------------------------------
    eased = raw.copy()
    cuts: List[int] = []
    jump_px = cfg.transition_jump_frac * frame_w
    ease_from: Optional[np.ndarray] = None
    ease_start = 0
    for i in range(1, frame_count):
        jump = float(np.hypot(*(raw[i] - raw[i - 1])))
        if modes[i] != modes[i - 1] or jump > jump_px:
            total = float(np.hypot(*(raw[i] - eased[i - 1])))
            if total > cfg.cut_min_jump_frac * frame_w and ball_lost_s[i - 1] > cfg.cut_min_ball_lost_s:
                cuts.append(i)
                ease_from = None
                continue
            ease_from = eased[i - 1].copy()
            ease_start = i
        if ease_from is not None:
            s = (i - ease_start + 1) / float(trans_n)
            if s >= 1.0:
                ease_from = None
            else:
                w = _smoothstep(s)
                eased[i] = ease_from * (1.0 - w) + raw[i] * w

    # ------------------------------------------------------------------
    # Phase 3: zoom setpoint (hysteresis, dwell, slew, zero-phase smooth).
    # ------------------------------------------------------------------
    setpoint, zoom_changes = _zoom_hysteresis(
        raw_zoom, fps, cfg.zoom_hysteresis_frac, cfg.zoom_hysteresis_s, cfg.zoom_min_dwell_s
    )
    zoom_sp = _slew(setpoint, cfg.max_zoom_rate_per_s * dt)
    zoom_sp = _gaussian_smooth(zoom_sp, 0.5 * cfg.zoom_smooth_time_constant_s * fps)
    zoom_sp = np.clip(zoom_sp, min_zoom, max_zoom)

    # ------------------------------------------------------------------
    # Phase 4: pan smoothing + constraint solve (zoom out, never snap).
    # ------------------------------------------------------------------
    k_idx = np.asarray(keep_idx, dtype=np.int64)
    k_xy = np.asarray(keep_xy, dtype=np.float64).reshape(-1, 2)
    weights = np.maximum(conf, 0.05)
    sigma_frames = cfg.smooth_time_constant_s * fps
    zoom_radius = max(1, int(round(cfg.zoom_smooth_time_constant_s * fps)))
    constraint_step = cfg.constraint_zoom_rate_per_s * dt
    seg_bounds = _segments_from_cuts(frame_count, cuts)
    keep_m = cfg.keep_margin_frac

    def _pan_for(zoom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tx, ty = _clamp_centers(eased[:, 0], eased[:, 1], zoom, (frame_w, frame_h), (base_w, base_h))
        cx = np.empty(frame_count)
        cy = np.empty(frame_count)
        crop_w = base_w / zoom
        vmax = cfg.max_pan_speed_crop_frac * crop_w
        amax = cfg.max_pan_accel_crop_frac * crop_w
        for a, b in seg_bounds:
            sx = _gaussian_smooth(tx[a:b], sigma_frames, weights[a:b])
            sy = _gaussian_smooth(ty[a:b], sigma_frames, weights[a:b])
            lx, ly = _limit_motion(sx, sy, dt, vmax[a:b], amax[a:b])
            cx[a:b], cy[a:b] = lx, ly
        return _clamp_centers(cx, cy, zoom, (frame_w, frame_h), (base_w, base_h))

    def _keep_need(cx: np.ndarray, cy: np.ndarray, zoom: np.ndarray):
        """Per-keep: allowed zoom and required pan shift (signed, px)."""
        if len(k_idx) == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)
        z = zoom[k_idx]
        cw = base_w / z
        ch = base_h / z
        mx = keep_m * cw
        my = keep_m * ch
        # Anchors closer to the frame edge than the margin only need the
        # crop to touch that edge.
        px = np.clip(k_xy[:, 0], mx, frame_w - mx)
        py = np.clip(k_xy[:, 1], my, frame_h - my)
        dx = np.abs(px - cx[k_idx])
        dy = np.abs(py - cy[k_idx])
        with np.errstate(divide="ignore"):
            zx = np.where(dx > 1e-6, (0.5 - keep_m) * base_w / dx, np.inf)
            zy = np.where(dy > 1e-6, (0.5 - keep_m) * base_h / dy, np.inf)
        allowed = np.minimum(zx, zy)
        half_x = cw / 2.0 - mx
        half_y = ch / 2.0 - my
        need_x = np.where(px > cx[k_idx] + half_x, px - (cx[k_idx] + half_x),
                          np.where(px < cx[k_idx] - half_x, px - (cx[k_idx] - half_x), 0.0))
        need_y = np.where(py > cy[k_idx] + half_y, py - (cy[k_idx] + half_y),
                          np.where(py < cy[k_idx] - half_y, py - (cy[k_idx] - half_y), 0.0))
        hard = (np.abs(k_xy[:, 0] - cx[k_idx]) > cw / 2.0 + 0.5) | (np.abs(k_xy[:, 1] - cy[k_idx]) > ch / 2.0 + 0.5)
        return allowed, need_x, need_y, hard

    zoom = zoom_sp.copy()
    cx, cy = _pan_for(zoom)
    passes = 0
    for passes in range(1, max(1, cfg.smoothing_passes) + 1):
        allowed, need_x, need_y, _hard = _keep_need(cx, cy, zoom)
        violated = (np.abs(need_x) > 0.5) | (np.abs(need_y) > 0.5)
        if not violated.any():
            break
        envelope = np.full(frame_count, max_zoom, dtype=np.float64)
        np.minimum.at(envelope, k_idx, allowed)
        envelope = np.clip(envelope, min_zoom, max_zoom)
        zoom = np.clip(_smooth_lower_envelope(np.minimum(zoom, envelope), constraint_step, zoom_radius),
                       min_zoom, max_zoom)
        cx, cy = _pan_for(zoom)

    # Residual violations: explicit, rate-limited pan correction.
    allowed, need_x, need_y, hard = _keep_need(cx, cy, zoom)
    residual = int(((np.abs(need_x) > 0.5) | (np.abs(need_y) > 0.5)).sum())
    if residual:
        corr_x = np.zeros(frame_count)
        corr_y = np.zeros(frame_count)
        for need, corr in ((need_x, corr_x), (need_y, corr_y)):
            pos = np.zeros(frame_count)
            neg = np.zeros(frame_count)
            np.maximum.at(pos, k_idx, np.maximum(need, 0.0))
            np.maximum.at(neg, k_idx, np.maximum(-need, 0.0))
            step = 0.5 * cfg.max_pan_speed_crop_frac * (base_w / max_zoom) * dt
            radius = max(1, int(round(0.5 * fps)))
            for a, b in seg_bounds:
                corr[a:b] += _smooth_upper_envelope(pos[a:b], step, radius)
                corr[a:b] -= _smooth_upper_envelope(neg[a:b], step, radius)
        cx, cy = _clamp_centers(cx + corr_x, cy + corr_y, zoom, (frame_w, frame_h), (base_w, base_h))
        _, need_x, need_y, hard = _keep_need(cx, cy, zoom)
    hard_left = int(hard.sum()) if len(hard) else 0

    rects = crop_rects_from_centers(cx, cy, zoom, (frame_w, frame_h), out_size, cfg.size_hold_px)
    plan.crop_rects = rects
    plan.cuts = list(cuts)

    cut_set = set(cuts)
    for index in range(frame_count):
        state, focus, reason, confidence, ball = metas[index]
        if index in cut_set:
            focus = "cut"
            reason = f"hard cut: ball lost > {cfg.cut_min_ball_lost_s:.0f}s and the play moved far ({reason})"
        plan.decisions.append(
            CameraDecision(
                index=index,
                t=float(times[index]),
                center_x=float(cx[index]),
                center_y=float(cy[index]),
                zoom=float(zoom[index]),
                state=state,
                focus=focus,
                reason=reason,
                confidence=float(confidence),
                ball_x=float(ball[0]) if ball is not None else None,
                ball_y=float(ball[1]) if ball is not None else None,
                ball_source=ball[2] if ball is not None else None,
                target_x=float(raw[index, 0]),
                target_y=float(raw[index, 1]),
            )
        )

    LOGGER.info(
        "camera plan built: %s (zoom setpoint changes=%d, constraint passes=%d, residual=%d, hard=%d)",
        plan.summary(), zoom_changes, passes, residual, hard_left,
    )
    return plan
