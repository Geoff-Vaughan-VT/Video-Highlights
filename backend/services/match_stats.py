"""Per-player and team match statistics in metric units.

Inputs are the shared analysis artifacts: a ``TrackingResult`` (all player
tracks with per-track team labels), the cleaned ``BallTrack``, the game-state
``segments`` and ``goal_events`` from ``game_tracking``, and a
``PitchCalibration`` (image px -> pitch metres). Outputs are the
``analysis_player_stats.json`` / ``analysis_team_stats.json`` documents of
``docs/ARTIFACTS.md``.

Pipeline (all numpy-vectorized; Python loops run over tracks, touches and
segments, never over samples):

1. **Kinematics** - each track's foot point is mapped to metres, resampled on
   a uniform ``grid_hz`` grid (only while observed: gaps > ``max_gap_s`` are
   not bridged), smoothed per observed run with a Savitzky-Golay filter, and
   differentiated. Steps implying > ``max_speed_mps`` are clipped.
2. **Touches** - at the ball track's native rate the nearest player to the
   ball is found (running minimum over tracks). A touch is either a *dwell*
   (ball within ``touch_radius_m`` of the same nearest player for
   ``touch_min_samples`` consecutive samples) or a *kick* (the ball's
   direction/speed changes sharply within ``kick_radius_m`` of a player; when
   the ball is occluded at the kick, the kick point is the intersection of
   the incoming and outgoing paths). Per-player refractory period 0.5 s.
3. **Possession chain** - touches in time order. A pass is a touch followed
   by a different player's touch within ``pass_max_gap_s`` with the ball
   moving > ``pass_min_distance_m``; completed when the receiver is a
   teammate. Team possession at ``possession_hz`` over in-play time is the
   team of the last toucher (unknown after a restart/goal or after
   ``possession_timeout_s``; "contested" when both teams have a player
   within ``contested_radius_m`` of the ball - both excluded from the
   denominator and reported).
4. **Shots** - the ball leaves a touch faster than ``shot_min_speed_mps``
   toward the opponent goal and ends in the goal area / crosses the goal-line
   region / produces a goal event. Every goal counts as a shot on target.
5. **Sides** - which goal each team defends is measured per half from the
   median foot x of its players (so both a halftime swap and no swap are
   handled); half-time is ``config.half_time_s``, else a long dead-ball
   segment near the middle, else ``duration / 2``.

Everything is heuristic; ``quality`` in the team document reports label and
ball coverage so the UI can show how much to trust the numbers.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .pitch_calibration import THIRD_NAMES, PitchCalibration
from .tracking_types import TEAM_A, TEAM_B, TEAM_REFEREE, TEAM_UNKNOWN, TrackingResult

LOGGER = logging.getLogger("videohighlights.match_stats")

PLAYER_STATS_FILENAME = "analysis_player_stats.json"
TEAM_STATS_FILENAME = "analysis_team_stats.json"

DEFAULT_TEAM_NAMES = {TEAM_A: "HOME", TEAM_B: "AWAY"}
_INTERRUPT_PREFIXES = ("restart", "goal")
_GOAL_PREFIX = "goal"


@dataclass
class MatchStatsConfig:
    # --- kinematics ---
    grid_hz: float = 10.0
    max_gap_s: float = 1.0
    # Savitzky-Golay (order 2, 0.9 s) ~ the noise reduction of a 0.5 s moving
    # average while keeping accelerations (sprints) sharp.
    smooth_window_s: float = 0.9
    smooth_polyorder: int = 2
    max_speed_mps: float = 12.0
    top_speed_percentile: float = 99.0
    sprint_speed_mps: float = 7.0
    sprint_min_s: float = 1.0
    high_intensity_mps: float = 5.5
    min_track_s: float = 2.0
    heatmap_bins_x: int = 21
    heatmap_bins_y: int = 14
    speed_series_hz: float = 1.0
    # --- ball ---
    ball_max_speed_mps: float = 45.0
    # --- touches ---
    touch_radius_m: float = 1.0
    touch_min_samples: int = 2
    touch_refractory_s: float = 0.5
    kick_radius_m: float = 2.0
    kick_angle_deg: float = 35.0
    kick_speed_gain_mps: float = 4.0
    kick_min_speed_mps: float = 2.0
    kick_velocity_span_s: float = 0.08
    # Raw ball detections within this distance of the filtered track are
    # used for touch detection (the filter lags at kicks).
    raw_ball_gate_m: float = 2.5
    # --- passes ---
    pass_max_gap_s: float = 4.0
    pass_min_distance_m: float = 3.0
    # --- shots ---
    shot_min_speed_mps: float = 10.0
    shot_launch_window_s: float = 0.6
    shot_window_s: float = 3.0
    shot_goal_link_s: float = 3.0
    goal_unlinked_toucher_s: float = 8.0
    goal_mouth_half_width_m: float = 3.66
    on_target_margin_m: float = 1.5
    goal_area_depth_m: float = 5.5
    goal_area_half_width_m: float = 9.16
    penalty_area_depth_m: float = 16.5
    penalty_area_half_width_m: float = 20.16
    goal_line_margin_m: float = 1.0
    # --- possession / sides ---
    possession_hz: float = 2.0
    possession_states: Tuple[str, ...] = ("in_play",)
    possession_timeout_s: float = 10.0
    contested_radius_m: float = 1.5
    half_time_s: Optional[float] = None
    halftime_min_dead_s: float = 180.0
    # --- timeline ---
    timeline_bin_s: float = 60.0
    momentum_ema_alpha: float = 0.4
    momentum_shot_weight: float = 1.0
    momentum_progression_m: float = 15.0  # metres of progression = 1 shot


# ---------------------------------------------------------------------------
# Internal structures
# ---------------------------------------------------------------------------


@dataclass
class _Kin:
    """One track's kinematics on its slice of the global grid."""

    track_id: int
    team: int
    g0: int  # first global grid index
    x: np.ndarray  # smoothed metres, NaN where unobserved
    y: np.ndarray
    obs: np.ndarray  # bool
    speed: np.ndarray  # m/s, NaN where unobserved
    step: np.ndarray  # metres moved since previous grid sample (0 at run starts)
    raw_samples: int = 0


@dataclass
class Touch:
    t: float
    track_id: int
    team: int
    x: float
    y: float
    kind: str  # dwell | kick


@dataclass
class Shot:
    t: float
    team: int
    track_id: Optional[int]
    on_target: bool
    goal: bool
    side: str  # goal side attacked ("left"/"right")
    speed_mps: float
    reason: str


@dataclass
class MatchAnalysis:
    """Shared intermediate results (computed once, used by both documents)."""

    config: MatchStatsConfig
    calibration: PitchCalibration
    duration_s: float
    grid_t: np.ndarray
    state: np.ndarray  # per grid sample, state string (object array)
    kins: Dict[int, _Kin]
    ball_xy: np.ndarray  # [n_grid, 2] metres, NaN when not visible
    ball_visible: np.ndarray
    ball_speed: np.ndarray  # m/s on the grid (NaN when not visible)
    ball_vx: np.ndarray
    touches: List[Touch]
    passes: List[Dict[str, object]]
    shots: List[Shot]
    half_time_s: float
    defends: Dict[int, List[Optional[str]]]  # team -> [side 1st half, side 2nd half]
    goal_attribution: List[Dict[str, object]]
    possession_t: np.ndarray
    possession_team: np.ndarray  # TEAM_A/B, -1 unknown, -2 contested, -3 not in play
    team_label_coverage_pct: float
    ball_coverage_pct: float

    def attack_sign(self, team: int, t: Union[float, np.ndarray]) -> np.ndarray:
        """+1 when ``team`` attacks the right goal (+x) at time ``t``."""
        t = np.asarray(t, dtype=np.float64)
        sides = self.defends.get(int(team)) or [None, None]
        first = 1.0 if sides[0] != "right" else -1.0
        second = 1.0 if sides[1] != "right" else -1.0
        if sides[0] is None and sides[1] is None:
            first = second = 1.0
        return np.where(t < self.half_time_s, first, second)

    def defender_of(self, side: str, t: float) -> Optional[int]:
        period = 0 if t < self.half_time_s else 1
        for team in (TEAM_A, TEAM_B):
            sides = self.defends.get(team)
            if sides and sides[period] == side:
                return team
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start/end (exclusive) indices of True runs."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    d = np.diff(np.concatenate([[False], m, [False]]).astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _label_runs(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start/end (exclusive) of runs of equal, non-zero labels."""
    lab = np.asarray(labels)
    if lab.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    change = np.concatenate([[True], lab[1:] != lab[:-1]])
    starts = np.flatnonzero(change)
    ends = np.concatenate([starts[1:], [lab.size]])
    keep = lab[starts] != 0
    return starts[keep], ends[keep]


def _savgol_runs(values: np.ndarray, mask: np.ndarray, window: int, order: int,
                 labels: Optional[np.ndarray] = None) -> np.ndarray:
    from scipy.signal import savgol_filter

    out = values.copy()
    starts, ends = _label_runs(labels) if labels is not None else _runs(mask)
    for s, e in zip(starts, ends):
        n = int(e - s)
        w = min(window, n if n % 2 == 1 else n - 1)
        if w <= order + 1:
            continue
        out[s:e] = savgol_filter(values[s:e], w, order, mode="interp")
    return out


def _state_per_sample(segments: Optional[Sequence[object]], t: np.ndarray) -> np.ndarray:
    states = np.full(t.shape, "in_play", dtype=object)
    segs = sorted(segments or [], key=lambda s: float(s.start_s))
    if not segs:
        return states
    starts = np.asarray([float(s.start_s) for s in segs])
    names = np.asarray([str(s.state) for s in segs], dtype=object)
    idx = np.searchsorted(starts, t, side="right") - 1
    ok = idx >= 0
    states[ok] = names[idx[ok]]
    # Before the first segment: use the first segment's state.
    states[~ok] = names[0]
    return states


def _is_interrupt(states: np.ndarray) -> np.ndarray:
    return np.asarray([str(s).startswith(_INTERRUPT_PREFIXES) for s in states], dtype=bool)


def _ball_on_grid(ball_track, calib: PitchCalibration, grid_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized ``BallTrack.position_at`` on the grid, mapped to metres."""
    n = grid_t.shape[0]
    xy = np.full((n, 2), np.nan)
    vis = np.zeros(n, dtype=bool)
    if ball_track is None or len(ball_track) == 0:
        return xy, vis
    times = np.asarray(ball_track.times, dtype=np.float64)
    pm = calib.to_pitch(np.column_stack([ball_track.xs, ball_track.ys]))
    good = np.all(np.isfinite(pm), axis=1)
    times, pm = times[good], pm[good]
    if times.size == 0:
        return xy, vis
    tol = float(ball_track.config.detected_time_tolerance_s)
    max_gap = float(ball_track.config.max_interpolation_gap_s)
    idx = np.searchsorted(times, grid_t)
    left = np.clip(idx - 1, 0, times.size - 1)
    right = np.clip(idx, 0, times.size - 1)
    span = times[right] - times[left]
    near = np.minimum(np.abs(grid_t - times[left]), np.abs(times[right] - grid_t))
    inside = (idx > 0) & (idx < times.size)
    vis = np.where(inside, ~((span > max_gap) & (near > tol)), near <= tol)
    xy[:, 0] = np.interp(grid_t, times, pm[:, 0])
    xy[:, 1] = np.interp(grid_t, times, pm[:, 1])
    xy[~vis] = np.nan
    return xy, vis


class _Nearest:
    """Running nearest-player search over tracks (no [tracks x samples] matrix)."""

    def __init__(self, kins: Mapping[int, _Kin], grid_hz: float):
        self.kins = [k for k in kins.values() if k.team != TEAM_REFEREE]
        self.hz = grid_hz

    def query(self, t: np.ndarray, xy: np.ndarray) -> Dict[str, np.ndarray]:
        n = t.shape[0]
        best_d = np.full(n, np.inf)
        best_id = np.full(n, -1, dtype=np.int64)
        best_team = np.full(n, TEAM_UNKNOWN, dtype=np.int64)
        team_d = {TEAM_A: np.full(n, np.inf), TEAM_B: np.full(n, np.inf)}
        if n == 0:
            return {"d": best_d, "id": best_id, "team": best_team, "d0": team_d[TEAM_A], "d1": team_d[TEAM_B]}
        order = np.argsort(t, kind="stable")
        ts = t[order]
        for k in self.kins:
            g_start = k.g0 / self.hz
            g_end = (k.g0 + k.x.size - 1) / self.hz
            a = int(np.searchsorted(ts, g_start - 0.5 / self.hz))
            b = int(np.searchsorted(ts, g_end + 0.5 / self.hz, side="right"))
            if b <= a:
                continue
            q = order[a:b]
            fi = (t[q] * self.hz) - k.g0
            gi = np.clip(np.rint(fi).astype(np.int64), 0, k.x.size - 1)
            ok = k.obs[gi]
            lo = np.clip(np.floor(fi).astype(np.int64), 0, k.x.size - 1)
            hi = np.clip(lo + 1, 0, k.x.size - 1)
            w = np.clip(fi - lo, 0.0, 1.0)
            px = k.x[lo] * (1 - w) + k.x[hi] * w
            py = k.y[lo] * (1 - w) + k.y[hi] * w
            bad = ~np.isfinite(px) | ~np.isfinite(py)
            px = np.where(bad, k.x[gi], px)
            py = np.where(bad, k.y[gi], py)
            d = np.hypot(px - xy[q, 0], py - xy[q, 1])
            d = np.where(ok & np.isfinite(d), d, np.inf)
            better = d < best_d[q]
            qi = q[better]
            best_d[qi] = d[better]
            best_id[qi] = k.track_id
            best_team[qi] = k.team
            if k.team in team_d:
                td = team_d[k.team]
                td[q] = np.minimum(td[q], d)
        return {"d": best_d, "id": best_id, "team": best_team, "d0": team_d[TEAM_A], "d1": team_d[TEAM_B]}


# ---------------------------------------------------------------------------
# Stage 1: kinematics
# ---------------------------------------------------------------------------


def _track_kinematics(track, calib: PitchCalibration, cfg: MatchStatsConfig, n_grid: int) -> Optional[_Kin]:
    if len(track) < 2:
        return None
    hz = cfg.grid_hz
    t = np.asarray(track.t, dtype=np.float64)
    m = calib.to_pitch(track.foot_xy().astype(np.float64))
    good = np.all(np.isfinite(m), axis=1) & np.isfinite(t)
    t, m = t[good], m[good]
    if t.size < 2:
        return None
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    t, m = t[keep], m[keep]
    if t.size < 2:
        return None
    tol = 0.5 / hz
    i0 = max(0, int(math.ceil((t[0] - tol) * hz)))
    i1 = min(n_grid - 1, int(math.floor((t[-1] + tol) * hz)))
    if i1 < i0:
        return None
    g = np.arange(i0, i1 + 1) / hz
    idx = np.searchsorted(t, g, side="right")
    prev = np.clip(idx - 1, 0, t.size - 1)
    nxt = np.clip(idx, 0, t.size - 1)
    inside = (idx > 0) & (idx < t.size)
    obs = (inside & ((t[nxt] - t[prev]) <= cfg.max_gap_s)) | (np.abs(g - t[prev]) <= tol) | (np.abs(t[nxt] - g) <= tol)
    x = np.interp(g, t, m[:, 0])
    y = np.interp(g, t, m[:, 1])
    # Identity switches / glitches: a raw jump faster than twice the speed
    # cap is a break in the track (never integrated, never smoothed across).
    raw_step = np.hypot(np.diff(m[:, 0]), np.diff(m[:, 1]))
    jump = (raw_step > 2.0 * cfg.max_speed_mps * np.diff(t)) & (raw_step > 2.0)
    seg = np.concatenate([[0], np.cumsum(jump)])
    nearest = np.where(np.abs(g - t[prev]) <= np.abs(t[nxt] - g), prev, nxt)
    straddle = seg[prev] != seg[nxt]
    x[straddle] = m[nearest[straddle], 0]
    y[straddle] = m[nearest[straddle], 1]
    labels = np.where(obs, seg[nearest] + 1, 0)
    window = max(cfg.smooth_polyorder + 2, int(round(cfg.smooth_window_s * hz)) | 1)
    x = _savgol_runs(x, obs, window, cfg.smooth_polyorder, labels)
    y = _savgol_runs(y, obs, window, cfg.smooth_polyorder, labels)
    x[~obs] = np.nan
    y[~obs] = np.nan
    step = np.zeros(g.size)
    pair = obs[1:] & obs[:-1] & (labels[1:] == labels[:-1])
    d = np.hypot(np.diff(x), np.diff(y))
    d = np.where(pair & np.isfinite(d), d, 0.0)
    step[1:] = np.minimum(d, cfg.max_speed_mps / hz)
    speed = np.full(g.size, np.nan)
    speed[1:][pair] = step[1:][pair] * hz
    # First sample of each run takes the next sample's speed.
    starts, ends = _label_runs(labels)
    for s0, e0 in zip(starts, ends):
        if e0 - s0 >= 2:
            speed[s0] = speed[s0 + 1]
        else:
            speed[s0] = 0.0
    return _Kin(track_id=int(track.track_id), team=int(track.team), g0=i0, x=x, y=y, obs=obs,
                speed=speed, step=step, raw_samples=int(len(track)))


# ---------------------------------------------------------------------------
# Stage 2: touches
# ---------------------------------------------------------------------------


def _touch_ball_samples(ball_track, raw_ball, calib: PitchCalibration,
                        cfg: MatchStatsConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Ball samples (t, metres) for touch detection.

    The ``BallTrack`` holds alpha-beta *filtered* positions, which lag behind
    the true path exactly where a kick changes its direction. When raw
    detections are available, use the raw detections that agree with the
    filtered track (outliers are still rejected by the track) instead.
    """
    tf = np.asarray(ball_track.times, dtype=np.float64)
    pf = calib.to_pitch(np.column_stack([ball_track.xs, ball_track.ys]))
    good = np.all(np.isfinite(pf), axis=1)
    tf, pf = tf[good], pf[good]
    if raw_ball is None or len(raw_ball) < 3 or tf.size < 3:
        return tf, pf
    tr = np.asarray(raw_ball.t, dtype=np.float64)
    pr = calib.to_pitch(np.column_stack([raw_ball.x, raw_ball.y]).astype(np.float64))
    idx = np.clip(np.searchsorted(tf, tr), 1, tf.size - 1)
    near_t = np.minimum(np.abs(tf[idx - 1] - tr), np.abs(tf[idx] - tr))
    fx = np.interp(tr, tf, pf[:, 0])
    fy = np.interp(tr, tf, pf[:, 1])
    ok = (np.all(np.isfinite(pr), axis=1) & (near_t <= 0.1)
          & (np.hypot(pr[:, 0] - fx, pr[:, 1] - fy) <= cfg.raw_ball_gate_m))
    if ok.sum() < 0.5 * tf.size:
        return tf, pf
    tr, pr = tr[ok], pr[ok]
    keep = np.concatenate([[True], np.diff(tr) > 1e-6])
    return tr[keep], pr[keep]


def _detect_touches(
    ball_track, calib: PitchCalibration, nearest: _Nearest, grid_t: np.ndarray, states: np.ndarray,
    cfg: MatchStatsConfig, raw_ball=None,
) -> List[Touch]:
    if ball_track is None or len(ball_track) < 3:
        return []
    tb, pb = _touch_ball_samples(ball_track, raw_ball, calib, cfg)
    n = tb.size
    if n < 3:
        return []
    near = nearest.query(tb, pb)
    candidates: List[Tuple[float, int, int, float, float, str]] = []

    # --- dwell touches ---
    within = near["d"] <= cfg.touch_radius_m
    same = np.concatenate([[False], near["id"][1:] == near["id"][:-1]])
    contiguous = np.concatenate([[False], np.diff(tb) <= 0.25])
    # Run = consecutive samples within radius of the SAME nearest player.
    run_break = ~(within & np.roll(within, 1) & same & contiguous)
    run_break[0] = True
    run_id = np.cumsum(run_break)
    run_id[~within] = 0
    if np.any(within):
        ids, first_idx, counts = np.unique(run_id[within], return_index=True, return_counts=True)
        within_idx = np.flatnonzero(within)
        for rid, fi, c in zip(ids, first_idx, counts):
            if c < cfg.touch_min_samples:
                continue
            i = int(within_idx[fi])
            candidates.append((float(tb[i]), int(near["id"][i]), int(near["team"][i]),
                               float(pb[i, 0]), float(pb[i, 1]), "dwell"))

    # --- kick touches (sharp direction / speed change near a player) ---
    rate = (n - 1) / max(1e-6, tb[-1] - tb[0])
    k = max(1, int(round(cfg.kick_velocity_span_s * rate)))
    if n > 2 * k:
        i = np.arange(k, n - k)
        dt_in = tb[i] - tb[i - k]
        dt_out = tb[i + k] - tb[i]
        ok = (dt_in > 1e-6) & (dt_out > 1e-6) & (dt_in <= 1.0) & (dt_out <= 1.0)
        v_in = (pb[i] - pb[i - k]) / np.maximum(dt_in, 1e-6)[:, None]
        v_out = (pb[i + k] - pb[i]) / np.maximum(dt_out, 1e-6)[:, None]
        s_in = np.hypot(v_in[:, 0], v_in[:, 1])
        s_out = np.hypot(v_out[:, 0], v_out[:, 1])
        cosang = np.sum(v_in * v_out, axis=1) / np.maximum(s_in * s_out, 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        moving = (s_in >= cfg.kick_min_speed_mps) & (s_out >= cfg.kick_min_speed_mps)
        turn = moving & (ang >= cfg.kick_angle_deg)
        gain = (s_out - s_in) >= cfg.kick_speed_gain_mps
        is_kick = ok & (turn | gain) & (s_out <= cfg.ball_max_speed_mps * 1.5)
        score = np.where(turn, ang / 180.0, 0.0) + np.where(gain, (s_out - s_in) / 20.0, 0.0)
        score = np.where(is_kick, score, -1.0)
        # Non-max suppression within +-k samples.
        cand = np.flatnonzero(is_kick)
        keep: List[int] = []
        suppressed = np.zeros(score.size, dtype=bool)
        for c in cand[np.argsort(-score[cand], kind="stable")].tolist():
            if suppressed[c]:
                continue
            keep.append(c)
            suppressed[max(0, c - k):c + k + 1] = True
        keep.sort()
        if keep:
            keep_arr = np.asarray(keep)
            ki = i[keep_arr]
            kt = tb[ki].copy()
            kxy = pb[ki].copy()
            located = np.zeros(ki.size, dtype=bool)
            # Locate turning kicks at the corner of the ball path: intersect
            # the incoming line (samples before the turn) with the outgoing
            # line (samples after it). Works when the kick itself was
            # occluded too (the usual case: the ball is behind the player).
            for j, c in enumerate(ki):
                if not turn[keep_arr[j]]:
                    continue
                a0, a1, b0, b1 = c - 1 - k, c - 1, c + 1, c + 1 + k
                if a0 < 0 or b1 >= n:
                    continue
                if tb[a1] - tb[a0] > 1.0 or tb[b1] - tb[b0] > 1.0 or tb[b0] - tb[a1] > 1.2:
                    continue
                d_in = pb[a1] - pb[a0]
                d_out = pb[b1] - pb[b0]
                mat = np.column_stack([d_in, d_out])
                if abs(np.linalg.det(mat)) < 1e-6:
                    continue
                s_in, s_out = np.linalg.solve(mat, pb[b0] - pb[a1])
                s_out = -s_out
                corner = pb[a1] + d_in * s_in
                l_in = float(np.hypot(*(corner - pb[a1])))
                l_out = float(np.hypot(*(pb[b0] - corner)))
                chord = float(np.hypot(*(pb[b0] - pb[a1])))
                if s_in < -0.25 or s_out < -0.25 or l_in + l_out > chord + 3.0 * max(1.0, chord):
                    continue
                frac = l_in / max(1e-9, l_in + l_out)
                kt[j] = tb[a1] + (tb[b0] - tb[a1]) * frac
                kxy[j] = corner
                located[j] = True
            res = nearest.query(kt, kxy)
            for j, idx_k in enumerate(ki):
                d_best, id_best, team_best = res["d"][j], res["id"][j], res["team"][j]
                if not located[j]:
                    lo, hi = max(0, idx_k - 1), min(n, idx_k + 2)
                    local = int(np.argmin(near["d"][lo:hi])) + lo
                    if near["d"][local] < d_best:
                        d_best, id_best, team_best = near["d"][local], near["id"][local], near["team"][local]
                if d_best <= cfg.kick_radius_m and id_best >= 0:
                    candidates.append((float(kt[j]), int(id_best), int(team_best),
                                       float(kxy[j, 0]), float(kxy[j, 1]), "kick"))

    if not candidates:
        return []
    # Drop "touches" with the ball off the pitch (in the net, out for a
    # restart and being fetched): those are not part of play.
    cand_xy = np.asarray([[c[3], c[4]] for c in candidates])
    on_pitch = calib.in_bounds(cand_xy, margin_m=1.0)
    candidates = [c for c, ok in zip(candidates, on_pitch) if ok]
    candidates.sort(key=lambda c: c[0])
    touches: List[Touch] = []
    last_by_player: Dict[int, float] = {}
    for t, tid, team, x, y, kind in candidates:
        prev = last_by_player.get(tid)
        if prev is not None and t - prev < cfg.touch_refractory_s:
            continue
        last_by_player[tid] = t
        touches.append(Touch(t=t, track_id=tid, team=team, x=x, y=y, kind=kind))
    return touches


# ---------------------------------------------------------------------------
# Stage 3/4: passes, shots, sides
# ---------------------------------------------------------------------------


def _half_time(duration: float, segments: Optional[Sequence[object]], cfg: MatchStatsConfig) -> float:
    if cfg.half_time_s is not None:
        return float(cfg.half_time_s)
    best = None
    for seg in segments or []:
        if str(seg.state) == "in_play":
            continue
        length = float(seg.end_s) - float(seg.start_s)
        mid = (float(seg.end_s) + float(seg.start_s)) / 2.0
        if length >= cfg.halftime_min_dead_s and 0.3 * duration <= mid <= 0.7 * duration:
            if best is None or length > best[0]:
                best = (length, mid)
    return float(best[1]) if best else duration / 2.0


def _defending_sides(kins: Mapping[int, _Kin], half_t: float, cfg: MatchStatsConfig) -> Dict[int, List[Optional[str]]]:
    hz = cfg.grid_hz
    min_samples = int(10 * hz)
    medians: Dict[int, List[Optional[float]]] = {TEAM_A: [None, None], TEAM_B: [None, None]}
    for team in (TEAM_A, TEAM_B):
        for period in (0, 1):
            vals = []
            for k in kins.values():
                if k.team != team:
                    continue
                gt = (k.g0 + np.arange(k.x.size)) / hz
                sel = k.obs & ((gt < half_t) if period == 0 else (gt >= half_t))
                if sel.any():
                    vals.append(k.x[sel])
            if vals:
                v = np.concatenate(vals)
                if v.size >= min_samples:
                    medians[team][period] = float(np.median(v))
    sides: Dict[int, List[Optional[str]]] = {TEAM_A: [None, None], TEAM_B: [None, None]}
    for period in (0, 1):
        ma, mb = medians[TEAM_A][period], medians[TEAM_B][period]
        if ma is not None and mb is not None:
            left = TEAM_A if ma <= mb else TEAM_B
        elif ma is not None:
            left = TEAM_A if ma <= 0 else TEAM_B
        elif mb is not None:
            left = TEAM_B if mb <= 0 else TEAM_A
        else:
            continue
        sides[left][period] = "left"
        sides[TEAM_B if left == TEAM_A else TEAM_A][period] = "right"
    # Fill a missing half by assuming the usual halftime swap.
    swap = {"left": "right", "right": "left"}
    for team in (TEAM_A, TEAM_B):
        s = sides[team]
        if s[0] is None and s[1] is not None:
            s[0] = swap[s[1]]
        elif s[1] is None and s[0] is not None:
            s[1] = swap[s[0]]
    return sides


def _detect_passes(touches: Sequence[Touch], ball_xy: np.ndarray, states: np.ndarray,
                   cfg: MatchStatsConfig) -> List[Dict[str, object]]:
    passes: List[Dict[str, object]] = []
    if len(touches) < 2:
        return passes
    interrupt = _is_interrupt(states)
    cum_int = np.concatenate([[0], np.cumsum(interrupt)])
    hz = cfg.grid_hz
    n = states.size
    for a, b in zip(touches[:-1], touches[1:]):
        if a.track_id == b.track_id:
            continue
        dt = b.t - a.t
        if dt > cfg.pass_max_gap_s or dt <= 0:
            continue
        dist = math.hypot(b.x - a.x, b.y - a.y)
        if dist <= cfg.pass_min_distance_m:
            continue
        ga = int(np.clip(round(a.t * hz), 0, n - 1))
        gb = int(np.clip(round(b.t * hz), 0, n - 1))
        broken = cum_int[gb + 1] - cum_int[ga] > 0
        completed = (not broken) and a.team in (TEAM_A, TEAM_B) and a.team == b.team
        passes.append({"t": a.t, "t_receive": b.t, "from": a.track_id, "to": b.track_id,
                       "team": a.team, "completed": bool(completed), "distance_m": dist,
                       "interrupted": bool(broken)})
    return passes


def _detect_shots(
    an: "MatchAnalysis", goal_events: Sequence[object], cfg: MatchStatsConfig
) -> Tuple[List[Shot], List[Dict[str, object]]]:
    calib = an.calibration
    hz = cfg.grid_hz
    n = an.grid_t.size
    hl = calib.half_length_m
    touches = an.touches
    goals = []
    for g in goal_events or []:
        side = getattr(g, "side", None) if not isinstance(g, dict) else g.get("side")
        t = getattr(g, "t", None) if not isinstance(g, dict) else g.get("t")
        conf = getattr(g, "confidence", None) if not isinstance(g, dict) else g.get("confidence")
        if side in ("left", "right") and t is not None:
            goals.append({"t": float(t), "side": str(side), "confidence": conf})
    goals.sort(key=lambda g: g["t"])

    attribution: List[Dict[str, object]] = []
    for g in goals:
        defender = an.defender_of(g["side"], g["t"])
        scorer = None if defender is None else (TEAM_B if defender == TEAM_A else TEAM_A)
        g["team"] = scorer
        attribution.append({"t": round(g["t"], 3), "side": g["side"], "team": scorer,
                            "defending_team": defender,
                            "confidence": None if g["confidence"] is None else round(float(g["confidence"]), 3)})

    shots: List[Shot] = []
    used_goals = set()
    touch_t = np.asarray([tc.t for tc in touches]) if touches else np.zeros(0)
    for j, tc in enumerate(touches):
        if tc.team not in (TEAM_A, TEAM_B):
            continue
        sign = float(an.attack_sign(tc.team, tc.t))
        side = "right" if sign > 0 else "left"
        t_next = float(touch_t[j + 1]) if j + 1 < len(touches) else float("inf")
        # The shot may be blocked/saved by the next touch; look until then.
        t_end = min(tc.t + cfg.shot_window_s, t_next if t_next > tc.t + 0.15 else tc.t + cfg.shot_window_s)
        g_a = int(np.clip(math.floor(tc.t * hz), 0, n - 1))
        g_l = int(np.clip(math.ceil((tc.t + cfg.shot_launch_window_s) * hz), 0, n - 1))
        g_e = int(np.clip(math.ceil(t_end * hz), 0, n - 1))
        launch_speed = an.ball_speed[g_a:g_l + 1]
        launch_vx = an.ball_vx[g_a:g_l + 1]
        if not np.any(np.isfinite(launch_speed)):
            continue
        vmax = float(np.nanmax(launch_speed))
        toward = float(np.nanmean(launch_vx * sign)) if np.any(np.isfinite(launch_vx)) else 0.0
        linked_goal = None
        for gi, g in enumerate(goals):
            if gi in used_goals or g["side"] != side:
                continue
            if tc.t <= g["t"] <= t_end + cfg.shot_goal_link_s:
                linked_goal = gi
                break
        if linked_goal is None and (vmax < cfg.shot_min_speed_mps or toward <= 0):
            continue
        path = an.ball_xy[g_a:g_e + 1]
        vis = np.all(np.isfinite(path), axis=1)
        on_target = False
        is_shot = linked_goal is not None
        reason = "goal" if is_shot else ""
        if vis.any():
            px = path[vis, 0] * sign
            py = path[vis, 1]
            crossed = (px >= hl - cfg.goal_line_margin_m) & (np.abs(py) <= cfg.penalty_area_half_width_m)
            last_x, last_y = px[-1], py[-1]
            in_goal_area = (last_x >= hl - cfg.goal_area_depth_m) & (abs(last_y) <= cfg.goal_area_half_width_m + 1.0)
            # Extrapolate a ball that vanished heading for goal.
            y_cross = None
            if crossed.any():
                y_cross = float(py[int(np.argmax(crossed))])
            else:
                vis_idx = np.flatnonzero(vis)
                gl = g_a + int(vis_idx[-1])
                vx = an.ball_vx[gl] * sign if np.isfinite(an.ball_vx[gl]) else 0.0
                if vx > 1.0 and last_x >= hl - cfg.penalty_area_depth_m:
                    t_hit = (hl - last_x) / vx
                    if t_hit <= 1.0:
                        vy = an.ball_xy[min(n - 1, gl), 1] - an.ball_xy[max(0, gl - 1), 1]
                        vy = float(vy * hz) if np.isfinite(vy) else 0.0
                        y_cross = float(last_y + vy * t_hit)
                        if abs(y_cross) <= cfg.penalty_area_half_width_m:
                            crossed = np.asarray([True])
            if not is_shot and (bool(crossed.any()) or bool(in_goal_area)):
                is_shot = True
                reason = "crossed goal-line region" if crossed.any() else "ended in goal area"
            if y_cross is not None and abs(y_cross) <= cfg.goal_mouth_half_width_m + cfg.on_target_margin_m:
                on_target = True
        if not is_shot:
            continue
        goal = linked_goal is not None and goals[linked_goal]["team"] == tc.team
        if linked_goal is not None:
            used_goals.add(linked_goal)
            if not goal:
                # Own goal: the touch was not a shot by the scoring team.
                continue
            on_target = True
        shots.append(Shot(t=tc.t, team=tc.team, track_id=tc.track_id, on_target=bool(on_target), goal=bool(goal),
                          side=side, speed_mps=vmax if math.isfinite(vmax) else 0.0, reason=reason))

    # Goals without a detected shot: credit the last teammate toucher (after
    # the previous goal), or the team.
    for gi, g in enumerate(goals):
        if gi in used_goals or g["team"] is None:
            continue
        shooter = None
        prev_goal_t = goals[gi - 1]["t"] if gi > 0 else -np.inf
        if touches:
            k = int(np.searchsorted(touch_t, g["t"], side="right")) - 1
            while (k >= 0 and g["t"] - touches[k].t <= cfg.goal_unlinked_toucher_s
                   and touches[k].t > prev_goal_t):
                if touches[k].team == g["team"]:
                    shooter = touches[k]
                    break
                k -= 1
        shots.append(Shot(t=shooter.t if shooter else g["t"], team=int(g["team"]),
                          track_id=shooter.track_id if shooter else None, on_target=True, goal=True,
                          side=g["side"], speed_mps=0.0,
                          reason="goal (shot inferred from goal event)"))
    shots.sort(key=lambda s: s.t)
    return shots, attribution


def _possession(an: "MatchAnalysis", cfg: MatchStatsConfig, near: Dict[str, np.ndarray],
                sample_idx: np.ndarray) -> np.ndarray:
    """Possessing team per possession sample (-1 unknown, -2 contested, -3 not in play)."""
    t = an.grid_t[sample_idx]
    out = np.full(t.size, -3, dtype=np.int64)
    in_play = np.isin(an.state[sample_idx].astype(str), np.asarray(cfg.possession_states, dtype=str))
    out[in_play] = -1
    if not an.touches:
        return out
    tt = np.asarray([tc.t for tc in an.touches])
    team = np.asarray([tc.team for tc in an.touches])
    k = np.searchsorted(tt, t, side="right") - 1
    has = k >= 0
    last_t = np.where(has, tt[np.clip(k, 0, None)], -np.inf)
    last_team = np.where(has, team[np.clip(k, 0, None)], TEAM_UNKNOWN)
    # Most recent interruption (restart/goal) at or before each sample.
    interrupt = _is_interrupt(an.state)
    int_idx = np.flatnonzero(interrupt)
    last_int = np.full(t.size, -np.inf)
    if int_idx.size:
        p = np.searchsorted(int_idx, sample_idx, side="right") - 1
        okp = p >= 0
        last_int[okp] = an.grid_t[int_idx[p[okp]]]
    valid = has & (last_t > last_int) & ((t - last_t) <= cfg.possession_timeout_s)
    poss = np.where(valid, last_team, TEAM_UNKNOWN)
    poss = np.where(np.isin(poss, (TEAM_A, TEAM_B)), poss, TEAM_UNKNOWN)
    contested = (near["d0"] <= cfg.contested_radius_m) & (near["d1"] <= cfg.contested_radius_m)
    poss = np.where(contested & (poss != TEAM_UNKNOWN), -2, poss)
    out[in_play] = poss[in_play]
    return out


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze_match(
    tracking: TrackingResult,
    ball_track,
    segments: Optional[Sequence[object]],
    calibration: PitchCalibration,
    *,
    goal_events: Optional[Sequence[object]] = None,
    config: Optional[MatchStatsConfig] = None,
) -> MatchAnalysis:
    """Compute the shared intermediate results for both stats documents."""
    cfg = config or MatchStatsConfig()
    hz = cfg.grid_hz
    duration = float(tracking.duration_s or 0.0)
    for tr in tracking.players.values():
        if len(tr):
            duration = max(duration, float(tr.t[-1]))
    if ball_track is not None and len(ball_track):
        duration = max(duration, float(ball_track.times[-1]))
    n_grid = int(math.floor(duration * hz)) + 1
    grid_t = np.arange(n_grid) / hz
    states = _state_per_sample(segments, grid_t)

    kins: Dict[int, _Kin] = {}
    for tr in tracking.players.values():
        kin = _track_kinematics(tr, calibration, cfg, n_grid)
        if kin is not None:
            kins[kin.track_id] = kin

    ball_xy, ball_vis = _ball_on_grid(ball_track, calibration, grid_t)
    ball_speed = np.full(n_grid, np.nan)
    ball_vx = np.full(n_grid, np.nan)
    if ball_vis.any():
        bx = _savgol_runs(np.nan_to_num(ball_xy[:, 0]), ball_vis, 5, 2)
        by = _savgol_runs(np.nan_to_num(ball_xy[:, 1]), ball_vis, 5, 2)
        pair = ball_vis[1:] & ball_vis[:-1]
        vx = np.diff(bx) * hz
        vy = np.diff(by) * hz
        sp = np.hypot(vx, vy)
        valid = pair & (sp <= cfg.ball_max_speed_mps * 1.5)
        ball_speed[1:][valid] = np.minimum(sp[valid], cfg.ball_max_speed_mps)
        ball_vx[1:][valid] = vx[valid]

    nearest = _Nearest(kins, hz)
    touches = _detect_touches(ball_track, calibration, nearest, grid_t, states, cfg,
                              raw_ball=getattr(tracking, "ball", None))
    half_t = _half_time(duration, segments, cfg)
    defends = _defending_sides(kins, half_t, cfg)

    # Coverage numbers.
    total = sum(int(k.obs.sum()) for k in kins.values() if k.team != TEAM_REFEREE)
    known = sum(int(k.obs.sum()) for k in kins.values() if k.team in (TEAM_A, TEAM_B))
    label_cov = 100.0 * known / total if total else 0.0
    ball_cov = 100.0 * float(ball_vis.mean()) if n_grid else 0.0

    an = MatchAnalysis(
        config=cfg, calibration=calibration, duration_s=duration, grid_t=grid_t, state=states, kins=kins,
        ball_xy=ball_xy, ball_visible=ball_vis, ball_speed=ball_speed, ball_vx=ball_vx, touches=touches,
        passes=[], shots=[], half_time_s=half_t, defends=defends, goal_attribution=[],
        possession_t=np.zeros(0), possession_team=np.zeros(0, dtype=np.int64),
        team_label_coverage_pct=label_cov, ball_coverage_pct=ball_cov,
    )
    an.passes = _detect_passes(touches, ball_xy, states, cfg)
    an.shots, an.goal_attribution = _detect_shots(an, goal_events or [], cfg)

    stride = max(1, int(round(hz / max(1e-3, cfg.possession_hz))))
    sample_idx = np.arange(0, n_grid, stride)
    q_xy = np.where(np.isfinite(ball_xy[sample_idx]), ball_xy[sample_idx], 1e9)
    near = nearest.query(grid_t[sample_idx], q_xy)
    an.possession_t = grid_t[sample_idx]
    an.possession_team = _possession(an, cfg, near, sample_idx)
    LOGGER.info(
        "match analysis: %d tracks, %d touches, %d passes, %d shots, %d goals attributed, "
        "half-time %.1fs, sides %s, ball coverage %.1f%%, team label coverage %.1f%%",
        len(kins), len(touches), len(an.passes), len(an.shots), len(an.goal_attribution), half_t,
        defends, ball_cov, label_cov,
    )
    return an


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _r(v: float, nd: int = 2) -> float:
    v = float(v)
    return round(v, nd) if math.isfinite(v) else 0.0


def _team_names(team_configs) -> Dict[int, Tuple[str, Optional[str]]]:
    out: Dict[int, Tuple[str, Optional[str]]] = {
        TEAM_A: (DEFAULT_TEAM_NAMES[TEAM_A], None), TEAM_B: (DEFAULT_TEAM_NAMES[TEAM_B], None)}
    if team_configs is None:
        return out
    items = team_configs.items() if isinstance(team_configs, Mapping) else enumerate(team_configs)
    for key, cfg in items:
        team = int(key)
        if cfg is None or team not in out:
            continue
        name = getattr(cfg, "name", None) or (cfg.get("name") if isinstance(cfg, dict) else None)
        color = getattr(cfg, "color_hex", None) or (cfg.get("color_hex") if isinstance(cfg, dict) else None)
        out[team] = (str(name or out[team][0]), color)
    return out


def _player_entry(an: MatchAnalysis, kin: _Kin, track, names) -> Dict[str, object]:
    cfg = an.config
    hz = cfg.grid_hz
    calib = an.calibration
    obs = kin.obs
    obs_s = float(obs.sum()) / hz
    gt = (kin.g0 + np.arange(kin.x.size)) / hz
    distance = float(kin.step.sum())
    sp = kin.speed[obs & np.isfinite(kin.speed)]
    top = float(np.percentile(sp, cfg.top_speed_percentile)) if sp.size else 0.0
    sprint_mask = np.nan_to_num(kin.speed, nan=0.0) > cfg.sprint_speed_mps
    s_starts, s_ends = _runs(sprint_mask)
    min_len = int(math.ceil(cfg.sprint_min_s * hz))
    sprints = 0
    sprint_dist = 0.0
    for s, e in zip(s_starts, s_ends):
        if e - s >= min_len:
            sprints += 1
            sprint_dist += float(kin.step[s:e].sum())
    hid = float(kin.step[np.nan_to_num(kin.speed, nan=0.0) > cfg.high_intensity_mps].sum())

    known_team = kin.team in (TEAM_A, TEAM_B)
    sign = an.attack_sign(kin.team, gt) if known_team else np.ones(gt.size)
    xs = kin.x[obs] * sign[obs]
    ys = kin.y[obs]
    thirds = calib.thirds(xs, +1)
    tc = np.bincount(thirds, minlength=3).astype(np.float64)
    thirds_pct = {THIRD_NAMES[i]: _r(100.0 * tc[i] / max(1.0, tc.sum()), 1) for i in range(3)}
    hl, hw = calib.half_length_m, calib.half_width_m
    hist, _, _ = np.histogram2d(
        np.clip(ys, -hw, hw - 1e-6), np.clip(xs, -hl, hl - 1e-6),
        bins=[cfg.heatmap_bins_y, cfg.heatmap_bins_x], range=[[-hw, hw], [-hl, hl]],
    )
    total = hist.sum()
    grid = (hist / total) if total > 0 else hist
    # Speed series: mean speed per 1/speed_series_hz bin, observed bins only.
    bin_s = 1.0 / max(1e-3, cfg.speed_series_hz)
    series: List[Dict[str, float]] = []
    if obs.any():
        b = np.floor(gt[obs] / bin_s).astype(np.int64)
        v = np.nan_to_num(kin.speed[obs], nan=0.0)
        ub, inv = np.unique(b, return_inverse=True)
        sums = np.bincount(inv, weights=v)
        cnts = np.bincount(inv)
        series = [{"t": _r(bb * bin_s, 1), "v": _r(s / c, 2)} for bb, s, c in zip(ub, sums, cnts)]

    touches = [tc_ for tc_ in an.touches if tc_.track_id == kin.track_id]
    passes = [p for p in an.passes if p["from"] == kin.track_id]
    shots = [s for s in an.shots if s.track_id == kin.track_id]
    team_name = names[kin.team][0] if known_team else None
    return {
        "track_id": int(kin.track_id),
        "team": int(kin.team),
        "team_name": team_name,
        "team_unknown": not known_team,
        "team_confidence": _r(getattr(track, "team_confidence", 0.0), 3),
        "label": getattr(track, "label", None),
        "jersey_number": getattr(track, "jersey_number", None),
        "jersey_color_hex": getattr(track, "jersey_color_hex", None),
        "start_s": _r(track.start_s, 2),
        "end_s": _r(track.end_s, 2),
        "minutes_tracked": _r(obs_s / 60.0, 2),
        "distance_m": _r(distance, 1),
        "top_speed_mps": _r(top, 2),
        "avg_speed_mps": _r(distance / obs_s if obs_s > 0 else 0.0, 2),
        "sprints": int(sprints),
        "sprint_distance_m": _r(sprint_dist, 1),
        "high_intensity_distance_m": _r(hid, 1),
        "touches": len(touches),
        "passes_attempted": len(passes),
        "passes_completed": int(sum(1 for p in passes if p["completed"])),
        "shots": len(shots),
        "shots_on_target": int(sum(1 for s in shots if s.on_target)),
        "goals": int(sum(1 for s in shots if s.goal)),
        "time_in_thirds_pct": thirds_pct,
        "heatmap": {"bins_x": cfg.heatmap_bins_x, "bins_y": cfg.heatmap_bins_y,
                    "orientation": "attacking_right" if known_team else "image_left_to_right",
                    "grid": [[round(float(v), 5) for v in row] for row in grid]},
        "speed_series": series,
    }


def compute_player_stats(
    tracking: TrackingResult,
    ball_track,
    segments: Optional[Sequence[object]],
    calibration: PitchCalibration,
    *,
    goal_events: Optional[Sequence[object]] = None,
    team_configs=None,
    config: Optional[MatchStatsConfig] = None,
    analysis: Optional[MatchAnalysis] = None,
) -> Dict[str, object]:
    """``analysis_player_stats.json`` document (see docs/ARTIFACTS.md).

    One entry per track except referees; tracks with an unknown team are
    included with ``team = -1`` and ``team_unknown = true``. Tracks observed
    for less than ``config.min_track_s`` are skipped (counted in
    ``skipped_short_tracks``). Heatmaps are oriented so the player's team
    attacks to the right (``heatmap.orientation``); ``grid`` is
    ``bins_y`` rows (far touchline first) of ``bins_x`` columns and sums to 1.
    """
    an = analysis or analyze_match(tracking, ball_track, segments, calibration,
                                   goal_events=goal_events, config=config)
    names = _team_names(team_configs)
    players = []
    skipped = 0
    for tid, kin in an.kins.items():
        track = tracking.players.get(tid)
        if track is None or kin.team == TEAM_REFEREE:
            continue
        if float(kin.obs.sum()) / an.config.grid_hz < an.config.min_track_s:
            skipped += 1
            continue
        players.append(_player_entry(an, kin, track, names))
    players.sort(key=lambda p: (p["team"] if p["team"] >= 0 else 9, -p["minutes_tracked"]))
    return {
        "generated_at": _now(),
        "pitch_calibration": calibration.to_dict(),
        "focus_player_track_id": tracking.focus_track_id,
        "duration_s": _r(an.duration_s, 2),
        "half_time_s": _r(an.half_time_s, 2),
        "skipped_short_tracks": skipped,
        "players": players,
    }


def compute_team_stats_v2(
    tracking: TrackingResult,
    ball_track,
    segments: Optional[Sequence[object]],
    goal_events: Optional[Sequence[object]],
    calibration: PitchCalibration,
    team_configs=None,
    *,
    set_piece_events: Optional[Sequence[object]] = None,
    config: Optional[MatchStatsConfig] = None,
    analysis: Optional[MatchAnalysis] = None,
) -> Dict[str, object]:
    """``analysis_team_stats.json`` document (see docs/ARTIFACTS.md).

    ``team_configs``: ``[TeamConfig|None, TeamConfig|None]`` or a mapping
    ``{0: ..., 1: ...}`` (name + colour). Colours fall back to the majority
    ``jersey_color_hex`` of each team's tracks. ``set_piece_events``
    (``game_tracking.SetPieceEvent``) feed the corner count.
    """
    an = analysis or analyze_match(tracking, ball_track, segments, calibration,
                                   goal_events=goal_events, config=config)
    cfg = an.config
    hz = cfg.grid_hz
    names = _team_names(team_configs)
    calib = an.calibration
    pos_team = an.possession_team
    pos_t = an.possession_t
    in_play_mask = pos_team != -3
    n0 = int(np.sum(pos_team == TEAM_A))
    n1 = int(np.sum(pos_team == TEAM_B))
    n_known = n0 + n1
    n_in_play = int(in_play_mask.sum())
    pos_dt = 1.0 / max(1e-3, cfg.possession_hz)

    grid_in_play = np.isin(an.state.astype(str), np.asarray(cfg.possession_states, dtype=str))
    ball_ok = grid_in_play & an.ball_visible & np.all(np.isfinite(an.ball_xy), axis=1)
    # Possession on the full grid (step function of the 2 Hz samples).
    gi_pos = np.clip(np.floor(an.grid_t * cfg.possession_hz).astype(np.int64), 0, max(0, pos_team.size - 1))
    grid_pos = pos_team[gi_pos] if pos_team.size else np.full(an.grid_t.size, -3)

    teams_doc: Dict[str, Dict[str, object]] = {}
    for team in (TEAM_A, TEAM_B):
        other = TEAM_B if team == TEAM_A else TEAM_A
        sign = an.attack_sign(team, an.grid_t)
        kins = [k for k in an.kins.values() if k.team == team]
        dist = float(sum(k.step.sum() for k in kins))
        obs_s = float(sum(k.obs.sum() for k in kins)) / hz
        passes = [p for p in an.passes if p["team"] == team]
        completed = sum(1 for p in passes if p["completed"])
        shots = [s for s in an.shots if s.team == team]
        goals = sum(1 for g in an.goal_attribution if g["team"] == team)
        # Territory: where the ball is, in this team's attacking frame.
        if ball_ok.any():
            th = calib.thirds(an.ball_xy[ball_ok, 0] * sign[ball_ok], +1)
            cnt = np.bincount(th, minlength=3).astype(np.float64)
            territory = {THIRD_NAMES[i]: _r(100.0 * cnt[i] / cnt.sum(), 1) for i in range(3)}
        else:
            territory = {name: 0.0 for name in THIRD_NAMES}
        poss_grid = (grid_pos == team) & an.ball_visible & np.isfinite(an.ball_speed)
        ball_speed_mean = float(np.mean(an.ball_speed[poss_grid])) if poss_grid.any() else 0.0
        prog = an.ball_vx * sign
        prog_ok = poss_grid & np.isfinite(prog)
        progression = float(np.mean(prog[prog_ok])) if prog_ok.any() else 0.0
        poss_minutes = float(np.sum(pos_team == team)) * pos_dt / 60.0
        corners = 0
        for sp in set_piece_events or []:
            kind = getattr(sp, "kind", None) if not isinstance(sp, dict) else sp.get("kind")
            if kind != "corner_kick":
                continue
            t_k = getattr(sp, "t_kick", None) if not isinstance(sp, dict) else sp.get("t_kick")
            side = getattr(sp, "side", None) if not isinstance(sp, dict) else sp.get("side")
            if side not in ("left", "right"):
                sx = getattr(sp, "x", None) if not isinstance(sp, dict) else sp.get("x")
                sy = getattr(sp, "y", None) if not isinstance(sp, dict) else sp.get("y")
                if sx is None or sy is None:
                    continue
                pm = calib.to_pitch([float(sx), float(sy)])
                side = "left" if pm[0] < 0 else "right"
            defender = an.defender_of(str(side), float(t_k or 0.0))
            if defender == other:
                corners += 1
        colour = names[team][1]
        if not colour:
            hexes = [tracking.players[k.track_id].jersey_color_hex for k in kins
                     if tracking.players.get(k.track_id) is not None and tracking.players[k.track_id].jersey_color_hex]
            if hexes:
                vals, counts = np.unique(np.asarray(hexes), return_counts=True)
                colour = str(vals[int(np.argmax(counts))])
        sides = an.defends.get(team) or [None, None]
        teams_doc[str(team)] = {
            "name": names[team][0],
            "color_hex": colour,
            "defends_first_half": sides[0],
            "defends_second_half": sides[1],
            # null when no in-play sample had a known possessor (see quality).
            "possession_pct": _r(100.0 * (n0 if team == TEAM_A else n1) / n_known, 1) if n_known else None,
            "passes": len(passes),
            "passes_completed": int(completed),
            "pass_accuracy_pct": _r(100.0 * completed / len(passes), 1) if passes else 0.0,
            "shots": len(shots),
            "shots_on_target": int(sum(1 for s in shots if s.on_target)),
            "goals": int(goals),
            "corners": int(corners),
            "touches": int(sum(1 for t in an.touches if t.team == team)),
            "territory_pct": territory,
            "avg_speed_mps": _r(dist / obs_s if obs_s > 0 else 0.0, 2),
            "distance_m": _r(dist, 1),
            "players_tracked": len(kins),
            "play_speed": {
                "ball_speed_mean_mps": _r(ball_speed_mean, 2),
                "progression_mps": _r(progression, 2),
                "passes_per_minute": _r(len(passes) / poss_minutes, 2) if poss_minutes > 0 else 0.0,
            },
        }

    timeline = _timeline(an, grid_pos)
    unknown = int(np.sum(pos_team == -1))
    contested = int(np.sum(pos_team == -2))
    return {
        "generated_at": _now(),
        "pitch_calibration": calib.to_dict(),
        "duration_s": _r(an.duration_s, 2),
        "half_time_s": _r(an.half_time_s, 2),
        "teams": teams_doc,
        "timeline": timeline,
        "goal_attribution": [
            {**g, "team_name": names[g["team"]][0] if g["team"] in names else None} for g in an.goal_attribution
        ],
        "shots": [
            {"t": _r(s.t, 2), "team": s.team, "track_id": s.track_id, "on_target": s.on_target, "goal": s.goal,
             "side": s.side, "speed_mps": _r(s.speed_mps, 1), "reason": s.reason}
            for s in an.shots
        ],
        "quality": {
            "team_label_coverage_pct": _r(an.team_label_coverage_pct, 1),
            "ball_coverage_pct": _r(an.ball_coverage_pct, 1),
            "in_play_s": _r(n_in_play * pos_dt, 1),
            "possession_known_pct": _r(100.0 * n_known / n_in_play, 1) if n_in_play else 0.0,
            "possession_unknown_pct": _r(100.0 * unknown / n_in_play, 1) if n_in_play else 0.0,
            "possession_contested_pct": _r(100.0 * contested / n_in_play, 1) if n_in_play else 0.0,
            "touches_detected": len(an.touches),
            "calibration_source": calib.source,
            "calibration_confidence": _r(calib.confidence, 2),
        },
    }


def _timeline(an: MatchAnalysis, grid_pos: np.ndarray) -> Dict[str, object]:
    cfg = an.config
    bin_s = float(cfg.timeline_bin_s)
    n_bins = max(1, int(math.ceil(an.duration_s / bin_s)))
    pos_b = np.clip(np.floor(an.possession_t / bin_s).astype(np.int64), 0, n_bins - 1)
    c0 = np.bincount(pos_b[an.possession_team == TEAM_A], minlength=n_bins)[:n_bins]
    c1 = np.bincount(pos_b[an.possession_team == TEAM_B], minlength=n_bins)[:n_bins]
    poss0: List[Optional[float]] = [
        _r(100.0 * a / (a + b), 1) if (a + b) > 0 else None for a, b in zip(c0, c1)
    ]
    shots = {TEAM_A: np.zeros(n_bins, dtype=np.int64), TEAM_B: np.zeros(n_bins, dtype=np.int64)}
    for s in an.shots:
        shots[s.team][min(n_bins - 1, int(s.t // bin_s))] += 1
    # Progression: metres the possessing team moved the ball toward goal.
    grid_b = np.clip(np.floor(an.grid_t / bin_s).astype(np.int64), 0, n_bins - 1)
    prog = {}
    for team in (TEAM_A, TEAM_B):
        v = an.ball_vx * an.attack_sign(team, an.grid_t)
        ok = (grid_pos == team) & np.isfinite(v)
        prog[team] = np.bincount(grid_b[ok], weights=np.clip(v[ok], 0, None) / cfg.grid_hz, minlength=n_bins)[:n_bins]
    raw = (cfg.momentum_shot_weight * (shots[TEAM_A] - shots[TEAM_B])
           + (prog[TEAM_A] - prog[TEAM_B]) / max(1e-6, cfg.momentum_progression_m))
    ema = np.zeros(n_bins)
    acc = 0.0
    for i, r in enumerate(raw):
        acc = cfg.momentum_ema_alpha * float(r) + (1 - cfg.momentum_ema_alpha) * acc
        ema[i] = acc
    peak = float(np.max(np.abs(ema))) if ema.size else 0.0
    momentum = (ema / peak) if peak > 1e-9 else ema
    return {
        "bin_s": int(bin_s) if float(bin_s).is_integer() else bin_s,
        "possession_pct_team0": poss0,
        "momentum": [_r(m, 3) for m in momentum],
        "shots_team0": [int(v) for v in shots[TEAM_A]],
        "shots_team1": [int(v) for v in shots[TEAM_B]],
    }


def compute_match_stats(
    tracking: TrackingResult,
    ball_track,
    segments: Optional[Sequence[object]],
    goal_events: Optional[Sequence[object]],
    calibration: PitchCalibration,
    team_configs=None,
    *,
    set_piece_events: Optional[Sequence[object]] = None,
    config: Optional[MatchStatsConfig] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Both documents from ONE shared analysis (preferred entry point)."""
    an = analyze_match(tracking, ball_track, segments, calibration, goal_events=goal_events, config=config)
    players = compute_player_stats(tracking, ball_track, segments, calibration, goal_events=goal_events,
                                   team_configs=team_configs, analysis=an)
    teams = compute_team_stats_v2(tracking, ball_track, segments, goal_events, calibration, team_configs,
                                  set_piece_events=set_piece_events, analysis=an)
    return players, teams


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def write_stats(out_dir: Union[str, Path], player_stats: Dict[str, object],
                team_stats: Dict[str, object]) -> Tuple[Path, Path]:
    """Write ``analysis_player_stats.json`` and ``analysis_team_stats.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p_path = out / PLAYER_STATS_FILENAME
    t_path = out / TEAM_STATS_FILENAME
    for path, doc in ((p_path, player_stats), (t_path, team_stats)):
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=_json_default, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    return p_path, t_path


def load_stats(out_dir: Union[str, Path]) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    """Read both stats documents (None for a missing file)."""
    out = Path(out_dir)
    docs: List[Optional[Dict[str, object]]] = []
    for name in (PLAYER_STATS_FILENAME, TEAM_STATS_FILENAME):
        path = out / name
        docs.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None)
    return docs[0], docs[1]


def team_stats_for_catalog(team_stats_v2: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    """Map a v2 team document onto ``stat_catalog.BASELINE_STATS`` keys.

    Returns ``{"home": {...}, "away": {...}}`` (home = team "0") with
    ``goals``, ``possession``, ``total_shots``, ``shots_on_target``,
    ``total_passes``, ``pass_accuracy`` and ``corners``. Saves are not
    produced by the stats engine (they come from the event engine) and are
    omitted; a missing team yields an empty dict.
    """
    teams = (team_stats_v2 or {}).get("teams") or {}
    out: Dict[str, Dict[str, object]] = {}
    for key, label in (("0", "home"), ("1", "away")):
        t = teams.get(key) if isinstance(teams, Mapping) else None
        if not isinstance(t, Mapping):
            out[label] = {}
            continue
        out[label] = {
            "goals": t.get("goals"),
            "possession": t.get("possession_pct"),
            "total_shots": t.get("shots"),
            "shots_on_target": t.get("shots_on_target"),
            "total_passes": t.get("passes"),
            "pass_accuracy": t.get("pass_accuracy_pct"),
            "corners": t.get("corners"),
        }
    return out
