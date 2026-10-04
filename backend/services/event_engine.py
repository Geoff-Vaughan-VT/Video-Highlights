"""Event engine: typed match events, excitement ranking and the reel plan.

Consumes the shared ``TrackingResult`` (all players + teams), the cleaned
``BallTrack``, ``FieldGeometry``, the game-state timeline, corroborated goals
(``game_tracking.detect_goal_events`` v2), set pieces and referee cards, and
produces ``analysis_events.json`` (see ``docs/ARTIFACTS.md``):

* ``goal`` - corroborated goals (v2 detector); uncorroborated candidates
  become ``shot``/``chance``.
* ``shot`` - the ball leaves a player fast (> 10 m/s, or > 0.25 frame
  widths/s uncalibrated) heading at a goal and ends near/in the goal or out
  behind the goal line. Missed shots (projected wide) are ``chance`` events
  with ``evidence.shot_attempt = true``.
* ``save`` - an on-target shot that stops/reverses in the goal area next to
  a keeper-like player (a track that lives near that goal line).
* ``chance`` - the attacking team carries the ball into the penalty area.
* ``corner_kick``/``free_kick``/``penalty_kick``/``goal_kick``/``kickoff`` -
  from set pieces; ``yellow_card``/``red_card`` - passthrough.
* ``sprint`` (> 7 m/s for >= 1.5 s; 6 m/s for the focus player),
  ``dribble`` (keeps the ball > 15 m with an opponent within 3 m),
  ``turnover`` (possession changes team in open play), ``foul_candidate``
  (audio peak + play stops + players converge; low confidence).

Distances use ``pitch_calibration.PitchCalibration`` when one is passed
(metres); otherwise pixels are converted with ``field width px / 105 m`` and
ball-speed thresholds are frame-width relative. Every event carries team,
player, confidence, an excitement score in [0, 1], a reason, evidence and
sources. ``plan_reel`` picks events for a target reel length and
``events_to_bookmarks`` produces the legacy bookmark rows.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .broadcast import BroadcastConfig, emotion_end, story_start
from .game_tracking import (
    STATE_IN_PLAY,
    BallTrack,
    FieldGeometry,
    GoalEvent,
    audio_peak_near,
    defending_sides,
    detect_goal_candidates,
    find_audio_peaks,
    iter_player_tracks,
    state_at,
)

LOGGER = logging.getLogger("videohighlights.event_engine")

EVENTS_FILENAME = "analysis_events.json"

EVENT_TYPES = (
    "goal", "shot", "save", "chance", "corner_kick", "free_kick", "penalty_kick", "goal_kick",
    "kickoff", "yellow_card", "red_card", "sprint", "dribble", "turnover", "foul_candidate",
)
SET_PIECE_TYPES = ("corner_kick", "free_kick", "penalty_kick", "goal_kick", "kickoff")

# Reel length presets (seconds).
REEL_PRESETS: Dict[str, float] = {"1min": 60.0, "3min": 180.0, "5min": 300.0, "10min": 600.0}

# Base excitement by type (bonuses are added on top; see _excitement).
BASE_EXCITEMENT: Dict[str, float] = {
    "goal": 1.0,
    "shot_on_target": 0.8,
    "shot": 0.6,
    "save": 0.75,
    "chance": 0.5,
    "yellow_card": 0.6,
    "red_card": 0.6,
    "penalty_kick": 0.5,
    "corner_kick": 0.4,
    "free_kick_threat": 0.4,
    "free_kick": 0.3,
    "goal_kick": 0.3,
    "kickoff": 0.3,
    "dribble": 0.35,
    "foul_candidate": 0.3,
    "sprint": 0.25,
    "turnover": 0.2,
}

DEFAULT_TEAM_NAMES = {0: "HOME", 1: "AWAY"}


@dataclass
class EventEngineConfig:
    grid_hz: float = 10.0
    pitch_length_m: float = 105.0
    # Shots.
    shot_speed_mps: float = 10.0
    shot_speed_frame_widths_per_s: float = 0.25
    shot_heading_change_deg: float = 30.0
    shot_goal_region_frac: float = 0.12  # of field width from the goal line
    shot_max_origin_frac: float = 0.65  # shots start within this of the goal line
    shot_target_margin_goal_heights: float = 2.0  # real-size goal boxes are ~11% of field height
    shot_end_lookahead_s: float = 1.5
    shot_shooter_radius_m: float = 4.0
    shot_kick_accel_ratio: float = 1.35
    shot_goal_link_s: float = 1.5
    # Possession.
    possession_radius_m: float = 2.0
    possession_hold_s: float = 3.0
    # Sprints.
    sprint_speed_mps: float = 7.0
    focus_sprint_speed_mps: float = 6.0
    sprint_min_duration_s: float = 1.5
    speed_baseline_s: float = 0.4
    # Dribbles.
    dribble_min_distance_m: float = 15.0
    dribble_opponent_radius_m: float = 3.0
    dribble_gap_s: float = 1.0
    # Turnovers.
    turnover_min_hold_s: float = 0.2
    # Saves.
    keeper_line_frac: float = 0.15
    keeper_time_frac: float = 0.6
    keeper_window_s: float = 120.0
    save_area_frac: float = 0.12
    save_player_radius_m: float = 4.0
    # Penalty-area chances.
    chance_area_depth_frac: float = 0.157  # 16.5 m / 105 m
    chance_area_half_height_frac: float = 0.30  # 20.15 m / 68 m
    chance_min_speed_mps: float = 2.0
    chance_debounce_s: float = 8.0
    # Foul candidates.
    foul_converge_radius_m: float = 10.0
    foul_stop_window_s: float = 4.0
    # Ranking / merging.
    merge_window_s: float = 3.0
    audio_window_s: float = 4.0
    audio_bonus_max: float = 0.15
    proximity_bonus_max: float = 0.10
    focus_bonus: float = 0.15
    # Clip windows.
    pre_s: float = 4.0
    max_post_s: float = 10.0
    max_lookback_s: float = 15.0


@dataclass
class Event:
    type: str
    t: float
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    team: Optional[int] = None
    team_name: Optional[str] = None
    side: Optional[str] = None
    player_track_id: Optional[int] = None
    secondary_track_id: Optional[int] = None
    confidence: float = 0.5
    excitement: float = 0.0
    reason: str = ""
    evidence: Dict[str, object] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    id: str = ""
    location_px: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "type": self.type,
            "t": round(float(self.t), 3),
            "t_start": None if self.t_start is None else round(float(self.t_start), 3),
            "t_end": None if self.t_end is None else round(float(self.t_end), 3),
            "team": self.team,
            "team_name": self.team_name,
            "side": self.side,
            "player_track_id": self.player_track_id,
            "secondary_track_id": self.secondary_track_id,
            "confidence": round(float(self.confidence), 3),
            "excitement": round(float(self.excitement), 3),
            "reason": self.reason,
            "evidence": _jsonable(self.evidence),
            "sources": list(self.sources),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Event":
        def _opt_int(v: object) -> Optional[int]:
            return None if v is None else int(v)  # type: ignore[arg-type]

        def _opt_float(v: object) -> Optional[float]:
            return None if v is None else float(v)  # type: ignore[arg-type]

        return cls(
            type=str(data.get("type")),
            t=float(data.get("t", 0.0) or 0.0),  # type: ignore[arg-type]
            t_start=_opt_float(data.get("t_start")),
            t_end=_opt_float(data.get("t_end")),
            team=_opt_int(data.get("team")),
            team_name=data.get("team_name"),  # type: ignore[arg-type]
            side=data.get("side"),  # type: ignore[arg-type]
            player_track_id=_opt_int(data.get("player_track_id")),
            secondary_track_id=_opt_int(data.get("secondary_track_id")),
            confidence=float(data.get("confidence", 0.0) or 0.0),  # type: ignore[arg-type]
            excitement=float(data.get("excitement", 0.0) or 0.0),  # type: ignore[arg-type]
            reason=str(data.get("reason") or ""),
            evidence=dict(data.get("evidence") or {}),  # type: ignore[arg-type]
            sources=list(data.get("sources") or []),  # type: ignore[arg-type]
            id=str(data.get("id") or ""),
        )


@dataclass
class EventSet:
    events: List[Event]
    trim_offset_s: float = 0.0
    team_names: Dict[int, str] = field(default_factory=lambda: dict(DEFAULT_TEAM_NAMES))
    duration_s: float = 0.0
    focus_track_id: Optional[int] = None
    calibrated: bool = False
    reel_plan: Optional[Dict[str, object]] = None
    generated_at: str = ""

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def by_type(self, event_type: str) -> List[Event]:
        return [e for e in self.events if e.type == event_type]

    def get(self, event_id: str) -> Optional[Event]:
        return next((e for e in self.events if e.id == event_id), None)

    def summary(self) -> Dict[str, object]:
        counts: Dict[str, int] = {}
        for ev in self.events:
            counts[ev.type] = counts.get(ev.type, 0) + 1
        per_team: Dict[str, Dict[str, int]] = {}
        for team in (0, 1):
            evs = [e for e in self.events if e.team == team]
            shots = [e for e in evs if e.type == "shot" or (e.type == "chance" and e.evidence.get("shot_attempt"))]
            per_team[str(team)] = {
                "goals": sum(1 for e in evs if e.type == "goal"),
                "shots": len(shots),
                "shots_on_target": sum(1 for e in evs if e.type == "shot" and e.evidence.get("on_target")),
                "saves": sum(1 for e in evs if e.type == "save"),
                "chances": sum(1 for e in evs if e.type == "chance"),
                "corners": sum(1 for e in evs if e.type == "corner_kick"),
                "free_kicks": sum(1 for e in evs if e.type == "free_kick"),
                "cards": sum(1 for e in evs if e.type in ("yellow_card", "red_card")),
                "sprints": sum(1 for e in evs if e.type == "sprint"),
                "dribbles": sum(1 for e in evs if e.type == "dribble"),
                "turnovers_won": sum(1 for e in evs if e.type == "turnover"),
            }
        return {"counts_by_type": dict(sorted(counts.items())), "per_team": per_team}

    def to_dict(self, reel_plan: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        plan = reel_plan if reel_plan is not None else self.reel_plan
        return {
            "generated_at": self.generated_at or datetime.now(timezone.utc).isoformat(),
            "trim_offset_seconds": round(float(self.trim_offset_s), 3),
            "duration_s": round(float(self.duration_s), 3),
            "focus_track_id": self.focus_track_id,
            "calibrated": bool(self.calibrated),
            "team_names": {str(k): v for k, v in self.team_names.items()},
            "events": [e.to_dict() for e in self.events],
            "reel_plan": plan or {},
            "summary": self.summary(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating,)):
        f = float(value)
        return round(f, 4) if math.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    return value


def _get(obj: object, key: str, default: object = None) -> object:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


class _Metric:
    """Pixels -> metres via a PitchCalibration, or a field-width scale."""

    def __init__(self, geometry: Optional[FieldGeometry], frame_size: Tuple[int, int],
                 calibration: object, pitch_length_m: float) -> None:
        self.calibration = calibration if calibration is not None and hasattr(calibration, "to_pitch") else None
        width_px = geometry.width if geometry is not None else float(frame_size[0])
        self.px_per_m = max(1e-6, float(width_px) / float(pitch_length_m))
        self.frame_w = float(frame_size[0])

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    def to_m(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if self.calibration is not None:
            try:
                out = np.asarray(self.calibration.to_pitch(xy), dtype=np.float64).reshape(-1, 2)
                return out
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("calibration mapping failed (%s); using pixel scale", exc)
                self.calibration = None
        return xy / self.px_per_m


@dataclass
class _TrackGrid:
    track_id: int
    team: int
    i0: int
    xy_px: np.ndarray  # [k, 2] (NaN in gaps)
    xy_m: np.ndarray  # [k, 2]
    speed_mps: np.ndarray  # [k]

    @property
    def i1(self) -> int:
        return self.i0 + len(self.xy_px)

    def at(self, g: int) -> Optional[np.ndarray]:
        if self.i0 <= g < self.i1:
            v = self.xy_m[g - self.i0]
            if np.all(np.isfinite(v)):
                return v
        return None


def _grid_track(track: object, grid_dt: float, n_grid: int, metric: _Metric,
                baseline_s: float) -> Optional[_TrackGrid]:
    tt = np.asarray(track.t, dtype=np.float64)  # type: ignore[attr-defined]
    if len(tt) == 0:
        return None
    fx = (np.asarray(track.x1, dtype=np.float64) + np.asarray(track.x2, dtype=np.float64)) * 0.5  # type: ignore[attr-defined]
    fy = np.asarray(track.y2, dtype=np.float64)  # type: ignore[attr-defined]
    i0 = max(0, int(math.ceil(tt[0] / grid_dt - 1e-9)))
    i1 = min(n_grid - 1, int(math.floor(tt[-1] / grid_dt + 1e-9)))
    if i1 < i0:
        return None
    gt = np.arange(i0, i1 + 1, dtype=np.float64) * grid_dt
    x = np.interp(gt, tt, fx)
    y = np.interp(gt, tt, fy)
    if len(tt) >= 2:
        idx = np.clip(np.searchsorted(tt, gt), 1, len(tt) - 1)
        span = tt[idx] - tt[idx - 1]
        near = np.minimum(np.abs(gt - tt[idx - 1]), np.abs(tt[idx] - gt))
        bad = (span > 1.0) & (near > 0.25)
        x[bad] = np.nan
        y[bad] = np.nan
    xy_px = np.stack([x, y], axis=1)
    xy_m = metric.to_m(xy_px)
    half = max(1, int(round(baseline_s / grid_dt / 2.0)))
    speed = np.full(len(gt), np.nan)
    if len(gt) > 2 * half:
        d = np.hypot(xy_m[2 * half:, 0] - xy_m[:-2 * half, 0], xy_m[2 * half:, 1] - xy_m[:-2 * half, 1])
        speed[half:len(gt) - half] = d / (2 * half * grid_dt)
    return _TrackGrid(int(track.track_id), int(getattr(track, "team", -1)), i0, xy_px, xy_m, speed)  # type: ignore[attr-defined]


def _ball_on_grid(ball_track: Optional[BallTrack], grid: np.ndarray) -> np.ndarray:
    out = np.full((len(grid), 2), np.nan)
    if ball_track is None or len(ball_track) == 0:
        return out
    times = np.asarray(ball_track.times, dtype=np.float64)
    xs = np.asarray(ball_track.xs, dtype=np.float64)
    ys = np.asarray(ball_track.ys, dtype=np.float64)
    if len(times) == 1:
        sel = np.abs(grid - times[0]) <= ball_track.config.detected_time_tolerance_s
        out[sel] = (xs[0], ys[0])
        return out
    x = np.interp(grid, times, xs)
    y = np.interp(grid, times, ys)
    idx = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
    span = times[idx] - times[idx - 1]
    near = np.minimum(np.abs(grid - times[idx - 1]), np.abs(times[idx] - grid))
    tol = ball_track.config.detected_time_tolerance_s
    valid = ((span <= ball_track.config.max_interpolation_gap_s) | (near <= tol))
    valid &= (grid >= times[0] - tol) & (grid <= times[-1] + tol)
    out[valid, 0] = x[valid]
    out[valid, 1] = y[valid]
    return out


def _goal_line_x(geometry: FieldGeometry, side: str) -> float:
    goal = geometry.goal_for_side(side)
    return goal.x2 if side == "left" else goal.x1


class _Context:
    """Precomputed per-match arrays shared by the detectors."""

    def __init__(self, tracking: object, ball_track: Optional[BallTrack], geometry: FieldGeometry,
                 segments: Sequence[object], cfg: EventEngineConfig, metric: _Metric,
                 duration_s: float, focus_track_id: Optional[int]) -> None:
        self.cfg = cfg
        self.metric = metric
        self.geometry = geometry
        self.ball_track = ball_track
        self.segments = list(segments or [])
        self.tracking = tracking
        self.focus_track_id = focus_track_id
        self.dt = 1.0 / cfg.grid_hz
        self.duration_s = max(duration_s, self.dt)
        self.n = int(math.floor(self.duration_s / self.dt)) + 1
        self.grid = np.arange(self.n, dtype=np.float64) * self.dt
        self.ball_px = _ball_on_grid(ball_track, self.grid)
        self.ball_m = metric.to_m(self.ball_px)
        self.tracks: List[_TrackGrid] = []
        for track in iter_player_tracks(tracking):
            tg = _grid_track(track, self.dt, self.n, metric, cfg.speed_baseline_s)
            if tg is not None:
                self.tracks.append(tg)
        self.by_id = {tg.track_id: tg for tg in self.tracks}
        # Ball speed (m/s) on the grid.
        self.ball_speed = np.full(self.n, np.nan)
        if self.n > 2:
            d = np.hypot(self.ball_m[2:, 0] - self.ball_m[:-2, 0], self.ball_m[2:, 1] - self.ball_m[:-2, 1])
            self.ball_speed[1:-1] = d / (2 * self.dt)
        # Nearest player per team to the ball.
        self.best_d = {k: np.full(self.n, np.inf) for k in (0, 1)}
        self.best_id = {k: np.full(self.n, -1, dtype=np.int64) for k in (0, 1)}
        for tg in self.tracks:
            if tg.team not in (0, 1):
                continue
            seg = self.ball_m[tg.i0:tg.i1]
            d = np.hypot(seg[:, 0] - tg.xy_m[:, 0], seg[:, 1] - tg.xy_m[:, 1])
            d = np.where(np.isfinite(d), d, np.inf)
            cur = self.best_d[tg.team][tg.i0:tg.i1]
            better = d < cur
            cur[better] = d[better]
            self.best_id[tg.team][tg.i0:tg.i1][better] = tg.track_id
        any_team1 = self.best_d[1] < self.best_d[0]
        self.near_d = np.where(any_team1, self.best_d[1], self.best_d[0])
        self.near_id = np.where(any_team1, self.best_id[1], self.best_id[0])
        self.near_team = np.where(any_team1, 1, 0)
        possessed = self.near_d <= cfg.possession_radius_m
        self.poss_id = np.where(possessed, self.near_id, -1)
        self.poss_team = np.where(possessed, self.near_team, -1)
        # Team in control: last possessor's team, held for possession_hold_s.
        self.ctrl_team = np.full(self.n, -1, dtype=np.int64)
        last_team, last_g = -1, -10 ** 9
        hold = int(round(cfg.possession_hold_s / self.dt))
        for g in range(self.n):
            if self.poss_team[g] >= 0:
                last_team, last_g = int(self.poss_team[g]), g
            if last_team >= 0 and g - last_g <= hold:
                self.ctrl_team[g] = last_team
        # Open play mask.
        self.in_play = np.ones(self.n, dtype=bool)
        if self.segments:
            self.in_play[:] = False
            for seg in self.segments:
                state = _get(seg, "state")
                if state != STATE_IN_PLAY:
                    continue
                a = int(max(0, math.floor(float(_get(seg, "start_s", 0.0)) / self.dt)))  # type: ignore[arg-type]
                b = int(min(self.n, math.ceil(float(_get(seg, "end_s", 0.0)) / self.dt) + 1))  # type: ignore[arg-type]
                self.in_play[a:b] = True
        # Which team defends which goal, in 30 s bins.
        self._sides_cache: Dict[int, Optional[Dict[str, int]]] = {}

    # ------------------------------------------------------------------
    def g(self, t: float) -> int:
        return int(min(self.n - 1, max(0, round(t / self.dt))))

    def sides_at(self, t: float) -> Optional[Dict[str, int]]:
        b = int(t // 30.0)
        if b not in self._sides_cache:
            self._sides_cache[b] = defending_sides(self.tracking, b * 30.0 + 15.0, window_s=90.0)
        return self._sides_cache[b]

    def attacking_team(self, side: Optional[str], t: float) -> Optional[int]:
        if side not in ("left", "right"):
            return None
        sides = self.sides_at(t)
        if sides is None:
            return None
        return int(sides["left" if side == "right" else "right"])

    def team_of(self, track_id: Optional[int]) -> Optional[int]:
        if track_id is None:
            return None
        tg = self.by_id.get(int(track_id))
        if tg is None or tg.team not in (0, 1):
            return None
        return int(tg.team)

    def nearest_player(self, t: float, xy_px: Optional[Tuple[float, float]] = None,
                       team: Optional[int] = None, radius_m: float = 1e9,
                       window_s: float = 0.2) -> Optional[Tuple[int, float]]:
        """Nearest player (optionally of ``team``) to the ball or ``xy_px`` around ``t``."""
        g0 = self.g(t - window_s)
        g1 = self.g(t + window_s)
        best: Optional[Tuple[int, float]] = None
        fixed = self.metric.to_m(np.asarray(xy_px, dtype=np.float64))[0] if xy_px is not None else None
        for g in range(g0, g1 + 1):
            target = fixed if fixed is not None else self.ball_m[g]
            if not np.all(np.isfinite(target)):
                continue
            for tg in self.tracks:
                if team is not None and tg.team != team:
                    continue
                if team is None and tg.team not in (0, 1):
                    continue
                p = tg.at(g)
                if p is None:
                    continue
                d = float(np.hypot(p[0] - target[0], p[1] - target[1]))
                if d <= radius_m and (best is None or d < best[1]):
                    best = (tg.track_id, d)
        return best

    def last_possessor(self, team: Optional[int], t: float, lookback_s: float = 5.0) -> Optional[int]:
        """Most recent player of ``team`` (any team when None) in possession before ``t``."""
        g_t = self.g(t)
        for g in range(g_t, max(-1, g_t - int(lookback_s / self.dt)), -1):
            if self.poss_id[g] >= 0 and (team is None or int(self.poss_team[g]) == team):
                return int(self.poss_id[g])
        return None

    def ball_xy_at(self, t: float) -> Optional[Tuple[float, float]]:
        p = self.ball_px[self.g(t)]
        if np.all(np.isfinite(p)):
            return float(p[0]), float(p[1])
        if self.ball_track is not None:
            pos = self.ball_track.position_at(t)
            if pos is not None:
                return pos[0], pos[1]
        return None

    def is_keeper_like(self, track_id: int, side: str, t: float) -> bool:
        tg = self.by_id.get(int(track_id))
        if tg is None:
            return False
        a = max(tg.i0, self.g(t - self.cfg.keeper_window_s))
        b = min(tg.i1, self.g(t + self.cfg.keeper_window_s) + 1)
        if b <= a:
            return False
        xs = tg.xy_px[a - tg.i0:b - tg.i0, 0]
        xs = xs[np.isfinite(xs)]
        if len(xs) == 0:
            return False
        line_x = _goal_line_x(self.geometry, side)
        near = np.abs(xs - line_x) <= self.cfg.keeper_line_frac * self.geometry.width
        return float(np.mean(near)) >= self.cfg.keeper_time_frac


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def _detect_shots(ctx: _Context, goal_events: Sequence[GoalEvent]) -> List[Event]:
    cfg, geometry, bt = ctx.cfg, ctx.geometry, ctx.ball_track
    events: List[Event] = []
    if bt is None or len(bt) < 3:
        return events
    times = np.asarray(bt.times, dtype=np.float64)
    xs = np.asarray(bt.xs, dtype=np.float64)
    ys = np.asarray(bt.ys, dtype=np.float64)
    n = len(times)
    # Velocities (image px/s for direction; speed in the threshold's units).
    vx = np.zeros(n)
    vy = np.zeros(n)
    for i in range(n):
        a = i - 1 if i > 0 and times[i] - times[i - 1] <= 0.25 else i
        b = i + 1 if i + 1 < n and times[i + 1] - times[i] <= 0.25 else i
        if b == a:
            continue
        dt = times[b] - times[a]
        vx[i] = (xs[b] - xs[a]) / dt
        vy[i] = (ys[b] - ys[a]) / dt
    if ctx.metric.calibrated:
        pts_m = ctx.metric.to_m(np.stack([xs, ys], axis=1))
        speed = np.zeros(n)
        for i in range(n):
            a = i - 1 if i > 0 and times[i] - times[i - 1] <= 0.25 else i
            b = i + 1 if i + 1 < n and times[i + 1] - times[i] <= 0.25 else i
            if b != a:
                speed[i] = float(np.hypot(*(pts_m[b] - pts_m[a]))) / (times[b] - times[a])
        threshold = cfg.shot_speed_mps
        unit = "m/s"
    else:
        speed = np.hypot(vx, vy)
        # Without a calibration, scale the metric shot speed by the field's
        # pixel length (the field spans pitch_length_m); the frame-width
        # fraction only acts as a cap so a mis-estimated field cannot make
        # ordinary passes look like shots.
        px_per_m = geometry.width / max(1.0, cfg.pitch_length_m) if geometry.width > 0 else 0.0
        metric_threshold = cfg.shot_speed_mps * px_per_m if px_per_m > 0 else float("inf")
        threshold = min(metric_threshold, cfg.shot_speed_frame_widths_per_s * ctx.metric.frame_w)
        unit = "px/s"
    speed = np.nan_to_num(speed, nan=0.0)
    heading = np.arctan2(vy, vx)
    max_turn = math.radians(cfg.shot_heading_change_deg)

    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if speed[i] < threshold:
            i += 1
            continue
        a = i
        h0 = heading[i]
        j = i + 1
        while j < n and speed[j] >= threshold and times[j] - times[j - 1] <= 0.3:
            turn = abs((heading[j] - h0 + math.pi) % (2 * math.pi) - math.pi)
            if turn > max_turn:
                break
            # A sudden acceleration is a new kick (pass -> shot in the same
            # direction): start a new flight there.
            ref = float(np.min(speed[max(a, j - 3):j]))
            if j - a >= 2 and speed[j] > cfg.shot_kick_accel_ratio * ref:
                break
            j += 1
        runs.append((a, j - 1))
        i = j

    for run_idx, (a, b) in enumerate(runs):
        if b - a < 1:
            continue
        next_start = runs[run_idx + 1][0] if run_idx + 1 < len(runs) else n
        k = min(b, a + 3)
        mvx, mvy = float(np.mean(vx[a:k + 1])), float(np.mean(vy[a:k + 1]))
        if abs(mvx) < 1e-6:
            continue
        side = "right" if mvx > 0 else "left"
        goal = geometry.goal_for_side(side)
        line_x = _goal_line_x(geometry, side)
        toward_field = 1.0 if side == "left" else -1.0  # +x is infield for the left goal
        origin_dist = (xs[a] - line_x) * toward_field
        if origin_dist < 0 or origin_dist > cfg.shot_max_origin_frac * geometry.width:
            continue
        y_proj = ys[a] + mvy / mvx * (line_x - xs[a])
        gy = goal.center[1]
        gh = abs(goal.y2 - goal.y1)
        margin = max(4.0, 0.06 * gh)
        on_target = (goal.y1 - margin) <= y_proj <= (goal.y2 + margin)
        near = abs(y_proj - gy) <= gh / 2.0 + gh * cfg.shot_target_margin_goal_heights
        if not near:
            continue
        # How did the flight end?
        t_run_end = float(times[b])
        # Look a little past the fast part (the ball keeps rolling) but never
        # into the next kick.
        hi = min(next_start, int(np.searchsorted(times, t_run_end + cfg.shot_end_lookahead_s, side="right")))
        flight_x = xs[a:max(hi, b + 1)]
        beyond = bool(np.any((flight_x - line_x) * toward_field < 0))
        end_dist = (xs[b] - line_x) * toward_field
        in_region = (end_dist <= cfg.shot_goal_region_frac * geometry.width
                     and abs(ys[b] - gy) <= gh * 1.5)
        vanished = (b + 1 >= n or times[b + 1] - times[b] > bt.config.max_interpolation_gap_s) and \
            end_dist <= 0.25 * geometry.width
        if not (beyond or in_region or vanished):
            continue

        t_kick = float(times[max(0, a - 1)]) if a > 0 and times[a] - times[a - 1] <= 0.3 else float(times[a])
        linked_goal = next(
            (gev for gev in goal_events
             if gev.side == side and t_kick - 0.5 <= gev.t <= t_run_end + cfg.shot_goal_link_s), None
        )
        outcome = "goal" if linked_goal is not None else ("on_target" if on_target else "wide")
        if beyond and not on_target and linked_goal is None:
            outcome = "wide"

        attacking = ctx.attacking_team(side, t_kick)
        shooter = ctx.nearest_player(t_kick, team=attacking, radius_m=cfg.shot_shooter_radius_m) \
            if attacking is not None else ctx.nearest_player(t_kick, radius_m=cfg.shot_shooter_radius_m)
        shooter_id = shooter[0] if shooter is not None else ctx.last_possessor(attacking, t_kick, 3.0)
        team = attacking if attacking is not None else ctx.team_of(shooter_id)

        evidence: Dict[str, object] = {
            "shot_attempt": True,
            "on_target": bool(on_target or linked_goal is not None),
            "outcome": outcome,
            "speed": round(float(np.max(speed[a:b + 1])), 2),
            "speed_unit": unit,
            "origin_px": [round(float(xs[a]), 1), round(float(ys[a]), 1)],
            "origin_distance_to_goal_line_px": round(float(origin_dist), 1),
            "projected_goal_line_y": round(float(y_proj), 1),
            "ended": "behind_goal_line" if beyond else ("goal_area" if in_region else "vanished_near_goal"),
            "flight_end_s": round(t_run_end, 3),
        }
        if linked_goal is not None:
            evidence["goal_t"] = round(float(linked_goal.t), 3)
        if shooter is not None:
            evidence["shooter_distance_m"] = round(float(shooter[1]), 2)

        # Save: on-target shot stopped/reversed in front of the goal by a keeper.
        if linked_goal is None and on_target and not beyond and end_dist <= cfg.save_area_frac * geometry.width:
            after = slice(b + 1, min(n, b + 6))
            reversed_ = bool(np.any(vx[after] * toward_field > 0)) if b + 1 < n else False
            stopped = bool(np.any(speed[after] < threshold / 3.0)) if b + 1 < n else True
            if reversed_ or stopped:
                keeper = ctx.nearest_player(t_run_end, xy_px=(float(xs[b]), float(ys[b])),
                                            radius_m=cfg.save_player_radius_m)
                if keeper is not None and ctx.is_keeper_like(keeper[0], side, t_run_end):
                    keeper_team = ctx.team_of(keeper[0])
                    defending = None if attacking is None else 1 - attacking
                    if defending is None or keeper_team is None or keeper_team == defending:
                        evidence["outcome"] = "saved"
                        events.append(Event(
                            type="save", t=t_run_end, side=side, player_track_id=keeper[0],
                            secondary_track_id=shooter_id,
                            team=keeper_team if keeper_team is not None else defending,
                            confidence=0.6, reason=f"on-target shot stopped at the {side} goal by a keeper-like player",
                            evidence={"keeper_distance_m": round(float(keeper[1]), 2), "shot_t": round(t_kick, 3),
                                      "reversed": reversed_, "stopped": stopped},
                            sources=["ball_tracking", "motion"], location_px=(float(xs[b]), float(ys[b])),
                        ))

        is_shot = evidence["on_target"] or outcome == "goal"
        confidence = 0.75 if linked_goal is not None else (0.6 if on_target else 0.5)
        events.append(Event(
            type="shot" if is_shot else "chance",
            t=t_kick, side=side, team=team, player_track_id=shooter_id, confidence=confidence,
            reason=(f"shot toward the {side} goal ({outcome.replace('_', ' ')})" if is_shot
                    else f"shot at the {side} goal went wide"),
            evidence=evidence, sources=["ball_tracking", "motion"],
            location_px=(float(xs[a]), float(ys[a])),
        ))
    return events


def _detect_area_entries(ctx: _Context, existing: Sequence[Event]) -> List[Event]:
    cfg, geometry = ctx.cfg, ctx.geometry
    events: List[Event] = []
    last_t = {"left": -1e9, "right": -1e9}
    px = ctx.ball_px
    for side in ("left", "right"):
        goal = geometry.goal_for_side(side)
        line_x = _goal_line_x(geometry, side)
        toward_field = 1.0 if side == "left" else -1.0
        depth = (px[:, 0] - line_x) * toward_field
        inside = (depth >= 0) & (depth <= cfg.chance_area_depth_frac * geometry.width) \
            & (np.abs(px[:, 1] - goal.center[1]) <= cfg.chance_area_half_height_frac * geometry.height)
        inside &= np.isfinite(px[:, 0])
        entries = np.flatnonzero(inside[1:] & ~inside[:-1]) + 1
        for g in entries:
            t = float(ctx.grid[g])
            if t - last_t[side] < cfg.chance_debounce_s:
                continue
            attacking = ctx.attacking_team(side, t)
            ctrl = int(ctx.ctrl_team[g])
            if attacking is None or ctrl != attacking or not ctx.in_play[g]:
                continue
            spd = ctx.ball_speed[max(0, g - 2):g + 3]
            spd = spd[np.isfinite(spd)]
            if len(spd) == 0 or float(np.max(spd)) < cfg.chance_min_speed_mps:
                continue
            if any(e.side == side and e.type in ("shot", "goal", "chance", "save") and -2.0 <= e.t - t <= 6.0
                   for e in existing):
                continue
            last_t[side] = t
            carrier = ctx.nearest_player(t, team=attacking, radius_m=cfg.possession_radius_m * 2.0)
            events.append(Event(
                type="chance", t=t, side=side, team=attacking,
                player_track_id=carrier[0] if carrier else None, confidence=0.5,
                reason=f"attacking team carried the ball into the {side} penalty area",
                evidence={"penalty_area_entry": True, "ball_speed_mps": round(float(np.max(spd)), 2)},
                sources=["ball_tracking", "motion"], location_px=(float(px[g, 0]), float(px[g, 1])),
            ))
    return events


def _detect_sprints(ctx: _Context) -> List[Event]:
    cfg = ctx.cfg
    events: List[Event] = []
    min_len = int(math.ceil(cfg.sprint_min_duration_s / ctx.dt))
    for tg in ctx.tracks:
        is_focus = ctx.focus_track_id is not None and tg.track_id == int(ctx.focus_track_id)
        if tg.team not in (0, 1) and not is_focus:
            continue
        thr = cfg.focus_sprint_speed_mps if is_focus else cfg.sprint_speed_mps
        fast = np.nan_to_num(tg.speed_mps, nan=0.0) > thr
        if not fast.any():
            continue
        edges = np.diff(np.concatenate([[0], fast.astype(np.int8), [0]]))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for s, e in zip(starts, ends):
            if e - s < min_len:
                continue
            seg_speed = tg.speed_mps[s:e]
            peak = int(s + np.nanargmax(seg_speed))
            pts = tg.xy_m[s:e]
            steps = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
            distance = float(np.nansum(steps))
            t0 = (tg.i0 + s) * ctx.dt
            t1 = (tg.i0 + e - 1) * ctx.dt
            events.append(Event(
                type="sprint", t=(tg.i0 + peak) * ctx.dt, t_start=max(0.0, t0 - 1.0), t_end=t1 + 1.0,
                team=tg.team if tg.team in (0, 1) else None, player_track_id=tg.track_id,
                confidence=0.6 if ctx.metric.calibrated else 0.45,
                reason=f"sprint: {float(np.nanmax(seg_speed)):.1f} m/s for {t1 - t0 + ctx.dt:.1f}s",
                evidence={"top_speed_mps": round(float(np.nanmax(seg_speed)), 2),
                          "duration_s": round(t1 - t0 + ctx.dt, 2), "distance_m": round(distance, 1),
                          "threshold_mps": thr, "calibrated": ctx.metric.calibrated},
                sources=["motion"],
                location_px=((float(tg.xy_px[peak, 0]), float(tg.xy_px[peak, 1]))
                             if np.all(np.isfinite(tg.xy_px[peak])) else None),
            ))
    return events


def _detect_possession_events(ctx: _Context) -> List[Event]:
    """Dribbles (long possession spells under pressure) and turnovers."""
    cfg = ctx.cfg
    events: List[Event] = []
    gap = int(round(cfg.dribble_gap_s / ctx.dt))
    # --- dribbles ---
    g = 0
    while g < ctx.n:
        pid = int(ctx.poss_id[g])
        if pid < 0:
            g += 1
            continue
        start = g
        last = g
        h = g + 1
        while h < ctx.n:
            other = int(ctx.poss_id[h])
            if other == pid:
                last = h
            elif other >= 0 or h - last > gap:
                break
            h += 1
        tg = ctx.by_id.get(pid)
        if tg is not None and last > start:
            a, b = max(start, tg.i0), min(last + 1, tg.i1)
            pts = tg.xy_m[a - tg.i0:b - tg.i0]
            dist = float(np.nansum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))) if len(pts) > 1 else 0.0
            opp = 1 - tg.team if tg.team in (0, 1) else None
            if dist >= cfg.dribble_min_distance_m and opp is not None:
                od = ctx.best_d[opp][start:last + 1]
                if len(od) and float(np.min(od)) <= cfg.dribble_opponent_radius_m:
                    gmin = start + int(np.argmin(od))
                    events.append(Event(
                        type="dribble", t=float(ctx.grid[gmin]), t_start=max(0.0, start * ctx.dt - 1.0),
                        t_end=last * ctx.dt + 1.0, team=tg.team, player_track_id=pid,
                        secondary_track_id=int(ctx.best_id[opp][gmin]) if ctx.best_id[opp][gmin] >= 0 else None,
                        confidence=0.5, reason=f"dribble: kept the ball {dist:.0f} m under pressure",
                        evidence={"distance_m": round(dist, 1),
                                  "closest_opponent_m": round(float(np.min(od)), 2),
                                  "duration_s": round((last - start) * ctx.dt, 2)},
                        sources=["ball_tracking", "motion"],
                        location_px=ctx.ball_xy_at(float(ctx.grid[gmin])),
                    ))
        g = max(h, g + 1)
    # --- turnovers ---
    min_hold = max(1, int(round(cfg.turnover_min_hold_s / ctx.dt)))
    last_team, last_pid, last_g = -1, -1, -10 ** 9
    hold_n = int(round(cfg.possession_hold_s / ctx.dt))
    for g in range(ctx.n):
        team = int(ctx.poss_team[g])
        if team < 0:
            continue
        pid = int(ctx.poss_id[g])
        if last_team >= 0 and team != last_team and g - last_g <= hold_n and ctx.in_play[g]:
            held = ctx.poss_team[g:g + min_hold]
            if len(held) and np.all((held == team) | (held < 0)):
                events.append(Event(
                    type="turnover", t=float(ctx.grid[g]), team=team, player_track_id=pid,
                    secondary_track_id=last_pid, confidence=0.5,
                    reason="possession changed team in open play",
                    evidence={"lost_by_team": last_team},
                    sources=["ball_tracking", "motion"], location_px=ctx.ball_xy_at(float(ctx.grid[g])),
                ))
        last_team, last_pid, last_g = team, pid, g
    return events


def _detect_fouls(ctx: _Context, peaks: Sequence[Tuple[float, float]], existing: Sequence[Event],
                  set_pieces: Sequence[Event]) -> List[Event]:
    cfg = ctx.cfg
    events: List[Event] = []
    if not ctx.tracks:
        return events
    for t_p, strength in peaks:
        if any(e.type in ("goal", "shot", "save", "yellow_card", "red_card") and abs(e.t - t_p) <= 4.0
               for e in existing):
            continue
        g_a, g_b = ctx.g(t_p + 1.0), ctx.g(t_p + cfg.foul_stop_window_s)
        stopped_state = float(np.mean(~ctx.in_play[g_a:g_b + 1])) >= 0.5 if g_b >= g_a else False
        spd = ctx.ball_speed[g_a:g_b + 1]
        slow = np.isfinite(spd) & (spd < 1.0)
        stopped_ball = int(np.sum(slow)) * ctx.dt >= 1.5
        if not (stopped_state or stopped_ball):
            continue
        xy = ctx.ball_xy_at(t_p)
        if xy is None:
            continue

        def _count_near(t: float) -> int:
            target = ctx.metric.to_m(np.asarray(xy, dtype=np.float64))[0]
            g = ctx.g(t)
            c = 0
            for tg in ctx.tracks:
                p = tg.at(g)
                if p is not None and np.hypot(p[0] - target[0], p[1] - target[1]) <= cfg.foul_converge_radius_m:
                    c += 1
            return c

        before, after = _count_near(t_p - 1.0), _count_near(t_p + 2.0)
        if after < before + 2:
            continue
        confidence = 0.3 + 0.1 * float(strength)
        follow = next((sp for sp in set_pieces if sp.type in ("free_kick", "penalty_kick")
                       and 0.0 <= sp.t - t_p <= 30.0), None)
        if follow is not None:
            confidence += 0.1
        events.append(Event(
            type="foul_candidate", t=t_p, confidence=min(0.55, confidence),
            reason="crowd reaction, play stopped and players converged",
            evidence={"audio_strength": round(float(strength), 3), "players_near_before": before,
                      "players_near_after": after, "stopped_state": bool(stopped_state),
                      "followed_by": follow.type if follow is not None else None},
            sources=["audio", "motion"], location_px=xy,
        ))
    return events


# ---------------------------------------------------------------------------
# Ranking, merging, windows
# ---------------------------------------------------------------------------


def _merge_same_type(events: List[Event], window_s: float) -> List[Event]:
    def _key(e: Event) -> Tuple[object, ...]:
        if e.type in ("sprint", "dribble"):
            return (e.type, e.player_track_id)
        if e.type in ("goal", "shot", "save", "chance", "corner_kick", "penalty_kick"):
            return (e.type, e.side)
        return (e.type,)

    groups: Dict[Tuple[object, ...], List[Event]] = {}
    for ev in events:
        groups.setdefault(_key(ev), []).append(ev)
    merged: List[Event] = []
    for evs in groups.values():
        evs.sort(key=lambda e: e.t)
        cluster: List[Event] = []
        for ev in evs + [None]:  # type: ignore[list-item]
            if ev is not None and cluster and ev.t - cluster[-1].t <= window_s:
                cluster.append(ev)
                continue
            if cluster:
                best = max(cluster, key=lambda e: (e.confidence, -e.t))
                if len(cluster) > 1:
                    best.evidence["merged_count"] = len(cluster)
                    best.sources = sorted({s for e in cluster for s in e.sources})
                    starts = [e.t_start for e in cluster if e.t_start is not None]
                    ends = [e.t_end for e in cluster if e.t_end is not None]
                    if starts:
                        best.t_start = min(starts)
                    if ends:
                        best.t_end = max(ends)
                merged.append(best)
            cluster = [ev] if ev is not None else []
    return merged


def _base_excitement(ev: Event) -> float:
    if ev.type == "shot":
        return BASE_EXCITEMENT["shot_on_target"] if ev.evidence.get("on_target") else BASE_EXCITEMENT["shot"]
    if ev.type == "free_kick" and ev.side:
        return BASE_EXCITEMENT["free_kick_threat"]
    return BASE_EXCITEMENT.get(ev.type, 0.2)


def _score_events(ctx: _Context, events: List[Event], peaks: Sequence[Tuple[float, float]]) -> None:
    cfg, geometry = ctx.cfg, ctx.geometry
    for ev in events:
        score = _base_excitement(ev)
        peak = audio_peak_near(peaks, ev.t, cfg.audio_window_s)
        if peak is not None:
            score += cfg.audio_bonus_max * float(peak[1])
            ev.evidence.setdefault("audio_peak_s", round(peak[0], 3))
            if "audio" not in ev.sources:
                ev.sources.append("audio")
        loc = ev.location_px or ctx.ball_xy_at(ev.t)
        if loc is not None and geometry is not None:
            d = min(abs(loc[0] - geometry.x_min), abs(geometry.x_max - loc[0]))
            prox = max(0.0, 1.0 - d / (0.35 * geometry.width))
            score += cfg.proximity_bonus_max * prox
        focus = ctx.focus_track_id
        if focus is not None and int(focus) in (ev.player_track_id, ev.secondary_track_id):
            score += cfg.focus_bonus
            ev.evidence["focus_involved"] = True
        ev.excitement = float(min(1.0, max(0.0, score)))


def _assign_windows(ctx: _Context, events: List[Event], envelope: object) -> None:
    cfg = ctx.cfg
    bcfg = BroadcastConfig(max_lookback_s=cfg.max_lookback_s, max_post_goal_s=cfg.max_post_s,
                           max_post_s=min(8.0, cfg.max_post_s))
    for ev in events:
        if ev.type in ("goal", "shot", "save", "chance"):
            start = story_start(ev.t, ctx.ball_track, ctx.segments, ev.side, bcfg) \
                if ctx.ball_track is not None else ev.t - cfg.pre_s
            end = emotion_end(ev.t, envelope, ev.type == "goal", bcfg)  # type: ignore[arg-type]
        elif ev.type in SET_PIECE_TYPES:
            start = ev.t - 3.0
            end = ev.t + 7.0
        else:
            start = ev.t - cfg.pre_s
            end = emotion_end(ev.t, envelope, False, bcfg)  # type: ignore[arg-type]
        if ev.t_start is not None:
            start = min(start, ev.t_start)
        if ev.t_end is not None:
            end = max(end, ev.t_end)
        start = max(0.0, max(start, ev.t - cfg.max_lookback_s))
        end = min(ctx.duration_s, min(end, ev.t + max(cfg.max_post_s, 4.0)))
        ev.t_start = round(min(start, ev.t), 3)
        ev.t_end = round(max(end, min(ctx.duration_s, ev.t + 1.0)), 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_events(
    tracking: object,
    ball_track: Optional[BallTrack],
    geometry: FieldGeometry,
    segments: Optional[Sequence[object]],
    goal_events: Optional[Sequence[object]],
    set_piece_events: Optional[Sequence[object]] = None,
    card_events: Optional[Sequence[object]] = None,
    *,
    calibration: object = None,
    audio_envelope: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    team_names: Optional[Union[Mapping[int, str], Sequence[str]]] = None,
    config: Optional[EventEngineConfig] = None,
    focus_track_id: Optional[int] = None,
    goal_candidates: Optional[Sequence[GoalEvent]] = None,
    duration_s: Optional[float] = None,
) -> EventSet:
    """Detect all typed events for one processing window.

    ``tracking``: ``TrackingResult`` (players with per-track ``team``).
    ``goal_events``: corroborated goals (``GoalEvent`` or dicts with
    ``t``/``side``/``confidence``/``reason``); None runs the v2 detector
    here. ``goal_candidates``: rejected v2 candidates (``verdict``
    shot/chance); None recomputes them. ``set_piece_events``/``card_events``:
    ``SetPieceEvent``/``CardEvent`` objects or their dicts. ``calibration``:
    optional ``pitch_calibration.PitchCalibration`` (metres). All times are
    processing-window seconds.
    """
    cfg = config or EventEngineConfig()
    frame_size = tuple(getattr(tracking, "frame_size", None) or (geometry.frame_size if geometry else (1920, 1080)))
    if duration_s is None:
        duration_s = float(getattr(tracking, "duration_s", 0.0) or 0.0)
        if ball_track is not None and len(ball_track):
            duration_s = max(duration_s, float(ball_track.times[-1]))
    if focus_track_id is None:
        focus_track_id = getattr(tracking, "focus_track_id", None)
    names = dict(DEFAULT_TEAM_NAMES)
    if isinstance(team_names, Mapping):
        names.update({int(k): str(v) for k, v in team_names.items()})
    elif team_names:
        names.update({i: str(v) for i, v in enumerate(list(team_names)[:2])})

    metric = _Metric(geometry, frame_size, calibration, cfg.pitch_length_m)  # type: ignore[arg-type]
    ctx = _Context(tracking, ball_track, geometry, segments or [], cfg, metric, float(duration_s), focus_track_id)
    peaks = find_audio_peaks(audio_envelope)

    goals: List[GoalEvent] = []
    if goal_events is None and ball_track is not None:
        rejected: List[GoalEvent] = []
        from .game_tracking import detect_goal_events as _v2

        goals = _v2(ball_track, geometry, 0.0, ctx.duration_s, player_tracks=tracking,
                    audio_envelope=audio_envelope, candidates_out=rejected)
        if goal_candidates is None:
            goal_candidates = rejected
    else:
        for row in goal_events or []:
            goals.append(GoalEvent(
                t=float(_get(row, "t", 0.0)), side=str(_get(row, "side", "")),  # type: ignore[arg-type]
                confidence=float(_get(row, "confidence", 0.0) or 0.0),  # type: ignore[arg-type]
                reason=str(_get(row, "reason", "") or ""),
                evidence=dict(_get(row, "evidence", {}) or {}),  # type: ignore[arg-type]
            ))
    if goal_candidates is None and ball_track is not None:
        goal_candidates = [c for c in detect_goal_candidates(
            ball_track, geometry, 0.0, ctx.duration_s, player_tracks=tracking, audio_envelope=audio_envelope,
        ) if c.verdict != "goal"]

    events: List[Event] = []
    shots = _detect_shots(ctx, goals)
    events.extend(shots)

    for gev in goals:
        attacking = ctx.attacking_team(gev.side, gev.t)
        linked = [s for s in shots if s.type == "shot" and s.evidence.get("goal_t") is not None
                  and abs(float(s.evidence["goal_t"]) - gev.t) < 1e-3]  # type: ignore[arg-type]
        scorer = linked[0].player_track_id if linked else None
        if scorer is None and attacking is not None:
            scorer = ctx.last_possessor(attacking, gev.t, 5.0)
        sources = ["ball_tracking"]
        if gev.evidence.get("audio_peak_s") is not None:
            sources.append("audio")
        if gev.evidence.get("kickoff_validation") == "players":
            sources.append("vision")
        events.append(Event(
            type="goal", t=gev.t, side=gev.side, team=attacking if attacking is not None else ctx.team_of(scorer),
            player_track_id=scorer, confidence=float(gev.confidence), reason=gev.reason,
            evidence={**gev.evidence, "shot_event_t": round(linked[0].t, 3) if linked else None},
            sources=sources, location_px=ctx.ball_xy_at(gev.t),
        ))

    for cand in goal_candidates or []:
        if any(e.side == cand.side and e.type in ("shot", "chance", "goal") and abs(e.t - cand.t) <= 3.0
               for e in events):
            for e in events:
                if e.side == cand.side and e.type in ("shot", "chance") and abs(e.t - cand.t) <= 3.0:
                    e.evidence["goal_candidate"] = {"t": round(cand.t, 3), "confidence": round(cand.confidence, 3),
                                                    "reason": cand.reason}
            continue
        attacking = ctx.attacking_team(cand.side, cand.t)
        events.append(Event(
            type="shot" if cand.verdict == "shot" else "chance", t=cand.t, side=cand.side, team=attacking,
            confidence=float(cand.confidence), reason=cand.reason,
            evidence={**cand.evidence, "shot_attempt": True, "on_target": cand.verdict == "shot",
                      "from_goal_candidate": True},
            sources=["ball_tracking"], location_px=ctx.ball_xy_at(cand.t),
        ))

    set_piece_out: List[Event] = []
    for sp in set_piece_events or []:
        kind = str(_get(sp, "kind", ""))
        if kind not in SET_PIECE_TYPES:
            continue
        t_kick = float(_get(sp, "t_kick", _get(sp, "t", 0.0)))  # type: ignore[arg-type]
        x, y = _get(sp, "x"), _get(sp, "y")
        xy = (float(x), float(y)) if x is not None and y is not None else None  # type: ignore[arg-type]
        kicker = ctx.nearest_player(t_kick, xy_px=xy, radius_m=4.0, window_s=0.3) if xy else None
        side = _get(sp, "side")
        team = ctx.team_of(kicker[0]) if kicker else None
        if team is None and kind in ("corner_kick", "penalty_kick") and side:
            team = ctx.attacking_team(str(side), t_kick)
        set_piece_out.append(Event(
            type=kind, t=t_kick, side=str(side) if side else None, team=team,
            player_track_id=kicker[0] if kicker else None,
            confidence=0.8 if kind in ("corner_kick", "penalty_kick") else 0.7,
            reason=str(_get(sp, "reason", "") or kind.replace("_", " ")),
            evidence={"t_setup": _get(sp, "t_start"), "location_px": [x, y]},
            sources=["ball_tracking"], location_px=xy,
        ))
    events.extend(set_piece_out)

    for card in card_events or []:
        kind = str(_get(card, "kind", "yellow_card"))
        if kind not in ("yellow_card", "red_card"):
            continue
        x, y = _get(card, "x"), _get(card, "y")
        evidence: Dict[str, object] = {}
        if _get(card, "crop_path"):
            evidence["card_crop_path"] = _get(card, "crop_path")
        events.append(Event(
            type=kind, t=float(_get(card, "t", 0.0)),  # type: ignore[arg-type]
            confidence=float(_get(card, "confidence", 0.6) or 0.6),  # type: ignore[arg-type]
            reason=str(_get(card, "reason", "") or kind.replace("_", " ")),
            evidence=evidence, sources=["vision"],
            location_px=(float(x), float(y)) if x is not None and y is not None else None,  # type: ignore[arg-type]
        ))

    events.extend(_detect_area_entries(ctx, events))
    events.extend(_detect_sprints(ctx))
    events.extend(_detect_possession_events(ctx))
    events.extend(_detect_fouls(ctx, peaks, events, set_piece_out))

    events = _merge_same_type(events, cfg.merge_window_s)
    for ev in events:
        if ev.team is None and ev.player_track_id is not None:
            ev.team = ctx.team_of(ev.player_track_id)
        ev.team_name = names.get(ev.team) if ev.team is not None else None
        state = state_at(ctx.segments, ev.t) if ctx.segments else None
        if state is not None:
            ev.evidence.setdefault("game_state", _get(state, "state"))
        if ev.location_px is not None:
            ev.evidence.setdefault("location_px", [round(float(ev.location_px[0]), 1),
                                                   round(float(ev.location_px[1]), 1)])
    _score_events(ctx, events, peaks)
    _assign_windows(ctx, events, audio_envelope)
    events.sort(key=lambda e: (e.t, e.type))
    for i, ev in enumerate(events, start=1):
        ev.id = f"ev_{i:04d}"

    event_set = EventSet(
        events=events, trim_offset_s=float(getattr(tracking, "trim_offset_s", 0.0) or 0.0),
        team_names=names, duration_s=ctx.duration_s,
        focus_track_id=int(focus_track_id) if focus_track_id is not None else None,
        calibrated=metric.calibrated, generated_at=datetime.now(timezone.utc).isoformat(),
    )
    LOGGER.info("event engine: %d events %s", len(events), event_set.summary()["counts_by_type"])
    return event_set


def plan_reel(
    events: Union[EventSet, Sequence[Event]],
    *,
    target_duration_s: Optional[float] = 300.0,
    pre_s: float = 4.0,
    post_s: float = 6.0,
    must_include: Sequence[str] = ("goal", "red_card"),
    focus_track_id: Optional[int] = None,
    preset: Optional[str] = None,
    min_excitement: float = 0.2,
    exclude_types: Sequence[str] = ("turnover",),
    merge_gap_s: float = 1.0,
    duration_s: Optional[float] = None,
    crossfade_s: float = 0.5,
) -> Dict[str, object]:
    """Pick events for a reel of ``target_duration_s`` (or a ``preset``:
    ``1min``/``3min``/``5min``/``10min``).

    Events are ranked by excitement (+0.15 for the focus player when the
    engine has not already applied it); ``must_include`` types are always
    selected even past the target. Clip windows come from the events'
    story/emotion boundaries (``t_start``/``t_end``), padded to at least
    ``pre_s``/``post_s`` around ``t``; overlapping windows merge into one clip
    holding several events. Clips are chronological.
    """
    evs = list(events.events if isinstance(events, EventSet) else events)
    if preset:
        target_duration_s = REEL_PRESETS.get(str(preset), target_duration_s)
    target = float(target_duration_s) if target_duration_s else float("inf")
    if duration_s is None and isinstance(events, EventSet):
        duration_s = events.duration_s or None
    focus = focus_track_id
    if focus is None and isinstance(events, EventSet):
        focus = events.focus_track_id

    def _window(ev: Event) -> Tuple[float, float]:
        s = ev.t - pre_s if ev.t_start is None else min(ev.t_start, ev.t - pre_s)
        e = ev.t + post_s if ev.t_end is None else max(ev.t_end, ev.t + post_s)
        s = max(0.0, s)
        if duration_s:
            e = min(float(duration_s), e)
        return s, max(e, s + 0.5)

    def _score(ev: Event) -> float:
        score = ev.excitement
        if focus is not None and int(focus) in (ev.player_track_id, ev.secondary_track_id) \
                and not ev.evidence.get("focus_involved"):
            score += 0.15
        return score

    def _union(windows: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        merged: List[Tuple[float, float]] = []
        for s, e in sorted(windows):
            if merged and s <= merged[-1][1] + merge_gap_s:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def _total(windows: List[Tuple[float, float]]) -> float:
        return sum(e - s for s, e in _union(windows))

    must = [e for e in evs if e.type in set(must_include)]
    rest = [e for e in evs if e.type not in set(must_include) and e.type not in set(exclude_types)
            and _score(e) >= min_excitement]
    rest.sort(key=lambda e: (-_score(e), -e.confidence, e.t))
    selected: List[Event] = []
    windows: List[Tuple[float, float]] = []
    for ev in sorted(must, key=lambda e: e.t):
        selected.append(ev)
        windows.append(_window(ev))
    for ev in rest:
        candidate = windows + [_window(ev)]
        if _total(candidate) <= target + 1e-6:
            selected.append(ev)
            windows = candidate
    clips_w = _union(windows)
    clips: List[Dict[str, object]] = []
    for idx, (s, e) in enumerate(clips_w, start=1):
        inside = sorted([ev for ev in selected if s <= ev.t <= e], key=lambda ev: ev.t)
        primary = max(inside, key=lambda ev: (_score(ev), ev.confidence)) if inside else None
        clips.append({
            "index": idx,
            "t_start": round(s, 3),
            "t_end": round(e, 3),
            "duration_s": round(e - s, 3),
            "event_ids": [ev.id for ev in inside],
            "primary_event_id": primary.id if primary else None,
            "primary_type": primary.type if primary else None,
            "side": primary.side if primary else None,
            "occurred_at_s": round(primary.t, 3) if primary else None,
            "excitement": round(_score(primary), 3) if primary else 0.0,
        })
    total = sum(float(c["duration_s"]) for c in clips)  # type: ignore[arg-type]
    plan = {
        "target_duration_s": None if not math.isfinite(target) else round(target, 3),
        "preset": preset,
        "selected_event_ids": [ev.id for ev in sorted(selected, key=lambda ev: ev.t)],
        "total_duration_s": round(total, 3),
        "estimated_reel_duration_s": round(max(0.0, total - crossfade_s * max(0, len(clips) - 1)), 3),
        "focus_track_id": focus,
        "must_include": list(must_include),
        "clips": clips,
    }
    LOGGER.info("reel plan: %d events in %d clips, %.1fs (target %s)",
                len(selected), len(clips), total, plan["target_duration_s"])
    return plan


def reel_clip_specs(reel_plan: Mapping[str, object], event_set: EventSet, clip_paths: Sequence[str],
                    trim_offset_s: float = 0.0) -> List[Dict[str, object]]:
    """``broadcast.build_broadcast_reel`` specs for clips rendered from ``reel_plan['clips']``
    (``clip_paths[i]`` is clip ``i``)."""
    specs: List[Dict[str, object]] = []
    for clip, path in zip(list(reel_plan.get("clips") or []), clip_paths):  # type: ignore[arg-type]
        primary = event_set.get(str(clip.get("primary_event_id"))) if clip.get("primary_event_id") else None
        specs.append({
            "path": path,
            "start_s": float(clip["t_start"]) + trim_offset_s,
            "end_s": float(clip["t_end"]) + trim_offset_s,
            "event_type": primary.type if primary else None,
            "occurred_at_s": (primary.t + trim_offset_s) if primary else None,
            "confidence": primary.confidence if primary else None,
            "title": (primary.type.replace("_", " ") if primary else None),
        })
    return specs


def events_to_bookmarks(
    event_set: Union[EventSet, Sequence[Event]],
    reel_plan: Optional[Mapping[str, object]] = None,
    trim_offset_s: float = 0.0,
) -> List[Dict[str, object]]:
    """Legacy bookmark rows (``analysis_bookmarks.json`` shape), one per event.

    With ``reel_plan`` only reel-selected events are included and their
    ``start_s``/``end_s`` are the enclosing clip's window (several events can
    share a clip - each still gets its own bookmark). Times are in the
    ORIGINAL file's timebase (``+ trim_offset_s``), like the old bookmarks.
    """
    evs = list(event_set.events if isinstance(event_set, EventSet) else event_set)
    clip_of: Dict[str, Mapping[str, object]] = {}
    if reel_plan:
        selected = set(reel_plan.get("selected_event_ids") or [])  # type: ignore[arg-type]
        evs = [e for e in evs if e.id in selected]
        for clip in reel_plan.get("clips") or []:  # type: ignore[union-attr]
            for eid in clip.get("event_ids") or []:
                clip_of[str(eid)] = clip
    else:
        evs = [e for e in evs if e.type not in ("turnover", "sprint")]
    evs.sort(key=lambda e: e.t)
    rows: List[Dict[str, object]] = []
    for idx, ev in enumerate(evs, start=1):
        clip = clip_of.get(ev.id)
        start = float(clip["t_start"]) if clip else float(ev.t_start if ev.t_start is not None else ev.t - 4.0)  # type: ignore[arg-type]
        end = float(clip["t_end"]) if clip else float(ev.t_end if ev.t_end is not None else ev.t + 6.0)  # type: ignore[arg-type]
        signals: Dict[str, object] = {"excitement": round(ev.excitement, 3), "side": ev.side, "reason": ev.reason}
        if ev.type == "goal":
            signals["goal_side"] = ev.side
            signals["goal_reason"] = ev.reason
        elif ev.type in ("yellow_card", "red_card"):
            signals["card_reason"] = ev.reason
            if ev.evidence.get("card_crop_path"):
                signals["card_crop_path"] = ev.evidence["card_crop_path"]
        elif ev.type in SET_PIECE_TYPES:
            signals["set_piece_side"] = ev.side
            signals["set_piece_reason"] = ev.reason
        rows.append({
            "bookmark_id": f"bm_{idx:04d}",
            "index": idx,
            "event_id": ev.id,
            "event_type": ev.type,
            "label": f"{ev.type}_detected",
            "confidence": round(float(ev.confidence), 3),
            "start_s": round(start + trim_offset_s, 3),
            "occurred_at_s": round(ev.t + trim_offset_s, 3),
            "end_s": round(end + trim_offset_s, 3),
            "duration_s": round(max(0.0, end - start), 3),
            "sources": list(ev.sources) or ["ball_tracking"],
            "game_state": ev.evidence.get("game_state"),
            "signals": signals,
            "team": ev.team,
            "team_name": ev.team_name,
            "player_track_id": ev.player_track_id,
            "excitement": round(float(ev.excitement), 3),
            "clip_index": clip.get("index") if clip else None,
        })
    return rows


def write_events(out_dir: Union[str, Path], event_set: EventSet,
                 reel_plan: Optional[Dict[str, object]] = None) -> str:
    """Write ``analysis_events.json`` (ARTIFACTS.md shape); returns the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if reel_plan is not None:
        event_set.reel_plan = reel_plan
    path = out / EVENTS_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_jsonable(event_set.to_dict()), indent=2), encoding="utf-8")
    tmp.replace(path)
    return str(path)


def load_events(out_dir: Union[str, Path]) -> Optional[EventSet]:
    """Read ``analysis_events.json`` back (None when missing)."""
    path = Path(out_dir)
    if path.is_dir():
        path = path / EVENTS_FILENAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    names = {int(k): str(v) for k, v in (data.get("team_names") or {}).items()} or dict(DEFAULT_TEAM_NAMES)
    return EventSet(
        events=[Event.from_dict(e) for e in data.get("events") or []],
        trim_offset_s=float(data.get("trim_offset_seconds", 0.0) or 0.0),
        team_names=names,
        duration_s=float(data.get("duration_s", 0.0) or 0.0),
        focus_track_id=data.get("focus_track_id"),
        calibrated=bool(data.get("calibrated", False)),
        reel_plan=dict(data.get("reel_plan") or {}) or None,
        generated_at=str(data.get("generated_at") or ""),
    )


def run_event_pipeline(
    out_dir: Union[str, Path],
    tracking: object,
    ball_track: Optional[BallTrack],
    geometry: FieldGeometry,
    segments: Sequence[object],
    goal_events: Optional[Sequence[object]],
    set_piece_events: Optional[Sequence[object]] = None,
    card_events: Optional[Sequence[object]] = None,
    *,
    calibration: object = None,
    audio_envelope: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    team_names: Optional[Union[Mapping[int, str], Sequence[str]]] = None,
    reel_target_s: Optional[float] = 300.0,
    reel_preset: Optional[str] = None,
    focus_track_id: Optional[int] = None,
    config: Optional[EventEngineConfig] = None,
) -> Tuple[EventSet, Dict[str, object], List[Dict[str, object]]]:
    """Convenience: detect -> plan -> write; returns (events, reel_plan, bookmarks)."""
    event_set = detect_events(
        tracking, ball_track, geometry, segments, goal_events, set_piece_events, card_events,
        calibration=calibration, audio_envelope=audio_envelope, team_names=team_names,
        config=config, focus_track_id=focus_track_id,
    )
    plan = plan_reel(event_set, target_duration_s=reel_target_s, preset=reel_preset,
                     focus_track_id=focus_track_id)
    write_events(out_dir, event_set, plan)
    bookmarks = events_to_bookmarks(event_set, plan, event_set.trim_offset_s)
    return event_set, plan, bookmarks


__all__ = [
    "EVENTS_FILENAME", "EVENT_TYPES", "REEL_PRESETS", "BASE_EXCITEMENT", "Event", "EventSet",
    "EventEngineConfig", "detect_events", "plan_reel", "reel_clip_specs", "events_to_bookmarks",
    "write_events", "load_events", "run_event_pipeline",
]
