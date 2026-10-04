"""Team identification by uniform color and team-level match stats.

The user configures two teams by name + jersey color (hex, from a color
picker). Player positions collected during tracking are then labeled by
sampling a small torso patch around each position in ~1Hz sampled frames
and matching its dominant color to the nearer team color in HSV space.

From the labeled positions we derive:

* which side each team defends (median x per period - handles the
  halftime swap),
* possession (nearest labeled player to the ball, sampled over in-play
  time),
* territory (share of team presence in each third),
* goal attribution: a goal INTO a side's goal is scored BY the other team.

Everything lands in ``analysis_team_stats.json`` and the goal/card
bookmarks gain a ``team`` field.

v2 (preferred): :func:`assign_track_teams` labels every *track* of a
``TrackingResult`` once (majority vote of torso colour samples spread over
the whole match, auto-detected kits when no colours are given, referees as
``TEAM_REFEREE``); ``backend.services.match_stats`` then computes the stats.
The positional functions below remain for legacy callers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("videohighlights.team_classification")


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for team classification") from exc
    return cv2


def hex_to_bgr(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Invalid hex color: {value!r}")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


@dataclass
class TeamConfig:
    name: str
    color_hex: str  # jersey color, e.g. "#d32f2f"


@dataclass
class TeamClassifierConfig:
    sample_fps: float = 1.0
    patch_half_px: int = 12
    # A patch must be at least this close (HSV distance) to SOME team color
    # to be labeled; otherwise unknown (-1). Hue is circular, weighted high.
    max_color_distance: float = 0.45
    # And clearly closer to one team than the other.
    min_margin: float = 0.08
    # Minimum saturation for a patch to be color-classifiable (grass-green
    # shadows and white kits need the value/hue combination below).
    possession_radius_frac: float = 0.06  # of frame width
    # Deprecated: legacy row cap. It used to stop labelling after the first
    # few minutes of a match; sampling is now capped by frame count
    # (``max_frames``, spread evenly over the whole window) instead.
    max_samples: int = 4000
    max_frames: int = 600
    # Positions labelled per sampled frame (legacy positional API).
    max_positions_per_frame: int = 60

    # --- per-track labelling (assign_track_teams) ---
    # Torso patch inside the player box: rows 25-55% of the box height
    # from the top, central 50% of the width.
    torso_top_frac: float = 0.25
    torso_bottom_frac: float = 0.55
    torso_width_frac: float = 0.5
    min_box_height_px: float = 10.0
    min_torso_pixels: int = 8
    # Skip a torso patch when this fraction of it is covered by a box that is
    # in front (larger y2 = closer to a side-line camera).
    max_occluded_frac: float = 0.3
    # A track needs this many good torso samples to get a team/referee label.
    min_track_samples: int = 2
    # Extra frames (beyond the uniform sample) used to label tracks that the
    # uniform sample never saw.
    extra_frames: int = 200
    # Colour feature space is OpenCV 8-bit Lab with L down-weighted (shadows
    # and sun change lightness far more than chroma).
    lightness_weight: float = 0.5
    # A sample further than this from a team centre is "other" (referee,
    # goalkeeper, occlusion). The effective threshold also adapts to the
    # observed spread of each kit.
    min_outlier_distance: float = 28.0
    outlier_spread_factor: float = 3.0
    # When colours are supplied, how far a cluster centre may be from the
    # supplied colour before we stop trusting the clustering.
    supplied_color_max_distance: float = 90.0
    # Re-assign "other" tracks that live in a goal area to the team defending
    # that goal (goalkeepers usually wear a third colour).
    goalkeeper_by_position: bool = True
    goalkeeper_zone_frac: float = 0.16
    goalkeeper_min_share: float = 0.8


def _hsv_of_bgr(bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
    cv2 = _import_cv2()
    px = np.uint8([[list(bgr)]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
    return float(h), float(s), float(v)


def _color_distance(hsv_a: Tuple[float, float, float], hsv_b: Tuple[float, float, float]) -> float:
    """Perceptual-ish HSV distance in [0, ~1.7]: circular hue + sat + value."""
    dh = abs(hsv_a[0] - hsv_b[0])
    dh = min(dh, 180.0 - dh) / 90.0  # 0..1
    ds = abs(hsv_a[1] - hsv_b[1]) / 255.0
    dv = abs(hsv_a[2] - hsv_b[2]) / 255.0
    # Low-saturation colors (white/black kits) carry no hue information.
    sat_weight = min(hsv_a[1], hsv_b[1]) / 255.0
    return dh * (0.6 + 0.8 * sat_weight) + ds * 0.5 + dv * 0.35


def classify_player_teams(
    video_path: str,
    player_positions: Optional[np.ndarray],
    team_a: TeamConfig,
    team_b: TeamConfig,
    config: Optional[TeamClassifierConfig] = None,
) -> np.ndarray:
    """Label player positions by team: returns (t, x, y, team) rows.

    team: 0 = team_a, 1 = team_b, -1 = unknown (referee, keeper in a third
    color, occluded patch, off-color pixels).
    """
    cfg = config or TeamClassifierConfig()
    if player_positions is None or len(player_positions) == 0:
        return np.empty((0, 4), dtype=np.float32)
    cv2 = _import_cv2()

    target_a = _hsv_of_bgr(hex_to_bgr(team_a.color_hex))
    target_b = _hsv_of_bgr(hex_to_bgr(team_b.color_hex))

    positions = np.asarray(player_positions, dtype=np.float64)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.warning("team classification skipped: cannot open %s", video_path)
        return np.empty((0, 4), dtype=np.float32)

    labeled: List[Tuple[float, float, float, int]] = []
    try:
        t_min, t_max = float(positions[:, 0].min()), float(positions[:, 0].max())
        # Spread a bounded number of frames evenly over the WHOLE window. The
        # old implementation stopped once ``max_samples`` rows were labelled,
        # which silently ignored everything after the first few minutes.
        n_frames = int(min(max(1, cfg.max_frames), math.ceil((t_max - t_min) * max(0.01, cfg.sample_fps)) + 1))
        for t in np.linspace(t_min, t_max, n_frames):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            actual_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if actual_t <= 0.0 and t > 0.5:
                actual_t = float(t)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, w = frame.shape[:2]
            mask = np.abs(positions[:, 0] - actual_t) <= 0.15
            for _, px, py in positions[mask][: cfg.max_positions_per_frame]:
                x0 = int(max(0, px - cfg.patch_half_px))
                x1 = int(min(w, px + cfg.patch_half_px))
                y0 = int(max(0, py - cfg.patch_half_px))
                y1 = int(min(h, py + cfg.patch_half_px))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                patch = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
                # Drop grass pixels (green hue band) before taking the median.
                grass = (patch[:, 0] > 35) & (patch[:, 0] < 85) & (patch[:, 1] > 60)
                kept = patch[~grass]
                if len(kept) < 12:
                    continue
                med = tuple(np.median(kept, axis=0))
                da = _color_distance(med, target_a)
                db = _color_distance(med, target_b)
                if min(da, db) > cfg.max_color_distance or abs(da - db) < cfg.min_margin:
                    team = -1
                else:
                    team = 0 if da < db else 1
                labeled.append((actual_t, float(px), float(py), team))
    finally:
        cap.release()

    result = np.asarray(labeled, dtype=np.float32) if labeled else np.empty((0, 4), dtype=np.float32)
    known = int((result[:, 3] >= 0).sum()) if len(result) else 0
    LOGGER.info(
        "team classification: %d samples, %d labeled (%s vs %s)",
        len(result), known, team_a.name, team_b.name,
    )
    return result


def compute_team_stats(
    labeled_positions: np.ndarray,
    ball_track,
    geometry,
    goal_events: Sequence[object],
    team_a: TeamConfig,
    team_b: TeamConfig,
    duration_s: float,
    config: Optional[TeamClassifierConfig] = None,
) -> Dict[str, object]:
    """Team-level stats + goal attribution from labeled positions."""
    cfg = config or TeamClassifierConfig()
    names = {0: team_a.name, 1: team_b.name}
    stats: Dict[str, object] = {
        "teams": [
            {"team": team_a.name, "color": team_a.color_hex},
            {"team": team_b.name, "color": team_b.color_hex},
        ],
        "label_counts": {},
        "defending_side": {},
        "possession_pct": {},
        "territory_pct": {},
        "goals": {team_a.name: 0, team_b.name: 0},
        "goal_attribution": [],
        "periods": [],
    }
    if labeled_positions is None or len(labeled_positions) == 0:
        stats["note"] = "no labeled player positions - check team colors"
        return stats

    rows = labeled_positions
    known = rows[rows[:, 3] >= 0]
    stats["label_counts"] = {
        team_a.name: int((known[:, 3] == 0).sum()),
        team_b.name: int((known[:, 3] == 1).sum()),
        "unknown": int((rows[:, 3] < 0).sum()),
    }

    # Defending side per period (median x per team; handles halftime swap).
    mid_x = (geometry.x_min + geometry.x_max) / 2.0
    period_edges = [0.0, duration_s / 2.0, duration_s]
    periods: List[Dict[str, object]] = []
    for p0, p1 in zip(period_edges[:-1], period_edges[1:]):
        window = known[(known[:, 0] >= p0) & (known[:, 0] < p1)]
        sides: Dict[str, str] = {}
        for team in (0, 1):
            tp = window[window[:, 3] == team]
            if len(tp) >= 20:
                sides[names[team]] = "left" if float(np.median(tp[:, 1])) < mid_x else "right"
        periods.append({"start_s": round(p0, 1), "end_s": round(p1, 1), "defending": sides})
    stats["periods"] = periods
    if periods and periods[0]["defending"]:
        stats["defending_side"] = periods[0]["defending"]

    def _defender_of(side: str, t: float) -> Optional[str]:
        period = periods[0] if t < duration_s / 2.0 else periods[-1]
        for team_name, team_side in (period.get("defending") or {}).items():
            if team_side == side:
                return team_name
        return None

    # Possession: nearest labeled player to the ball at 1s steps.
    radius = cfg.possession_radius_frac * (geometry.frame_size[0] if hasattr(geometry, "frame_size") else 1920)
    counts = {team_a.name: 0, team_b.name: 0}
    thirds = {team_a.name: [0, 0, 0], team_b.name: [0, 0, 0]}
    third_w = geometry.width / 3.0
    for t in np.arange(0.0, duration_s, 1.0):
        window = known[np.abs(known[:, 0] - t) <= 0.6]
        for team in (0, 1):
            tp = window[window[:, 3] == team]
            if len(tp):
                mean_x = float(np.mean(tp[:, 1]))
                idx = int(min(2, max(0, (mean_x - geometry.x_min) // third_w)))
                thirds[names[team]][idx] += 1
        ball = ball_track.position_at(float(t)) if ball_track is not None else None
        if ball is None or len(window) == 0:
            continue
        dists = np.hypot(window[:, 1] - ball[0], window[:, 2] - ball[1])
        nearest = int(np.argmin(dists))
        if dists[nearest] <= radius * 3.0:
            counts[names[int(window[nearest, 3])]] += 1
    total = sum(counts.values())
    if total:
        stats["possession_pct"] = {k: round(100.0 * v / total, 1) for k, v in counts.items()}
    for team_name, buckets in thirds.items():
        s = sum(buckets)
        if s:
            stats["territory_pct"][team_name] = [round(100.0 * b / s, 1) for b in buckets]

    # Goal attribution: goal INTO a side's goal = scored by the OTHER team.
    for goal in goal_events or []:
        side = getattr(goal, "side", None) or (goal.get("side") if isinstance(goal, dict) else None)
        t = float(getattr(goal, "t", None) or (goal.get("t") if isinstance(goal, dict) else 0.0))
        defender = _defender_of(str(side), t)
        scorer = None
        if defender == team_a.name:
            scorer = team_b.name
        elif defender == team_b.name:
            scorer = team_a.name
        if scorer:
            stats["goals"][scorer] = int(stats["goals"].get(scorer, 0)) + 1
        stats["goal_attribution"].append(
            {"t": round(t, 3), "into_goal": side, "team": scorer, "defending_team": defender}
        )
    return stats


def detect_team_colors(
    video_path: str,
    player_positions: Optional[np.ndarray],
    samples: int = 7,
    patch_half_px: int = 12,
) -> Optional[Tuple[str, str]]:
    """Suggest the two team jersey colors from a handful of frames.

    Samples torso patches around tracked player positions in ``samples``
    frames spread across the match, drops grass pixels, and k-means-clusters
    the remaining colors (k=3: two kits + referee/noise). The two most
    populous, mutually distinct clusters become the suggested hex colors.
    Returns (hex_a, hex_b) or None when there is not enough signal.
    """
    if player_positions is None or len(player_positions) < 40:
        return None
    cv2 = _import_cv2()
    positions = np.asarray(player_positions, dtype=np.float64)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    pixels: List[np.ndarray] = []
    try:
        t_min, t_max = float(positions[:, 0].min()), float(positions[:, 0].max())
        for t in np.linspace(t_min + 1.0, max(t_min + 1.0, t_max - 1.0), samples):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            actual_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            h, w = frame.shape[:2]
            near = positions[np.abs(positions[:, 0] - actual_t) <= 0.2][:25]
            for _, px, py in near:
                x0, x1 = int(max(0, px - patch_half_px)), int(min(w, px + patch_half_px))
                y0, y1 = int(max(0, py - patch_half_px)), int(min(h, py + patch_half_px))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                patch = frame[y0:y1, x0:x1].reshape(-1, 3)
                hsv = cv2.cvtColor(patch.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
                grass = (hsv[:, 0] > 35) & (hsv[:, 0] < 85) & (hsv[:, 1] > 60)
                kept = patch[~grass]
                if len(kept):
                    pixels.append(kept)
    finally:
        cap.release()
    if not pixels:
        return None
    data = np.concatenate(pixels).astype(np.float32)
    if len(data) < 200:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 3, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=3)
    order = np.argsort(counts)[::-1]
    picked: List[np.ndarray] = []
    for idx in order:
        center = centers[idx]
        if picked and float(np.linalg.norm(center - picked[0])) < 60.0:
            continue  # same kit seen twice (shadow/sun) - need a DIFFERENT color
        picked.append(center)
        if len(picked) == 2:
            break
    if len(picked) < 2:
        return None

    def _hex(bgr: np.ndarray) -> str:
        b, g, r = (int(max(0, min(255, round(float(v))))) for v in bgr)
        return f"#{r:02x}{g:02x}{b:02x}"

    hex_a, hex_b = _hex(picked[0]), _hex(picked[1])
    LOGGER.info("auto-detected jersey colors: %s vs %s", hex_a, hex_b)
    return hex_a, hex_b


# ---------------------------------------------------------------------------
# v2: per-TRACK team labelling
# ---------------------------------------------------------------------------
#
# Each tracker identity gets ONE team label from a majority vote over torso
# colour samples taken across the whole match (frames spread evenly, capped by
# frame count). Positions are never labelled individually any more: a track
# is a person, and a person does not change team mid-match.


@dataclass
class TeamAssignmentReport:
    """What :func:`assign_track_teams` did (also stored on the TrackingResult)."""

    frames_planned: int = 0
    frames_read: int = 0
    samples: int = 0
    tracks_total: int = 0
    tracks_team0: int = 0
    tracks_team1: int = 0
    tracks_referee: int = 0
    tracks_unknown: int = 0
    tracks_goalkeeper_by_position: int = 0
    # Share of tracks / of player samples (time-weighted) with a team 0/1
    # label. Referee tracks count as labelled for coverage purposes.
    track_coverage_pct: float = 0.0
    sample_coverage_pct: float = 0.0
    colors_source: str = "none"  # supplied | detected | supplied+adapted | none
    team_colors_hex: Dict[str, Optional[str]] = field(default_factory=dict)
    per_track: Dict[int, Dict[str, object]] = field(default_factory=dict)

    def to_dict(self, include_tracks: bool = False) -> Dict[str, object]:
        out: Dict[str, object] = {
            "frames_planned": self.frames_planned,
            "frames_read": self.frames_read,
            "samples": self.samples,
            "tracks_total": self.tracks_total,
            "tracks_team0": self.tracks_team0,
            "tracks_team1": self.tracks_team1,
            "tracks_referee": self.tracks_referee,
            "tracks_unknown": self.tracks_unknown,
            "tracks_goalkeeper_by_position": self.tracks_goalkeeper_by_position,
            "track_coverage_pct": round(self.track_coverage_pct, 1),
            "sample_coverage_pct": round(self.sample_coverage_pct, 1),
            "colors_source": self.colors_source,
            "team_colors_hex": dict(self.team_colors_hex),
        }
        if include_tracks:
            out["per_track"] = {str(k): v for k, v in self.per_track.items()}
        return out


def _bgr_to_hex(bgr: Sequence[float]) -> str:
    b, g, r = (int(max(0, min(255, round(float(v))))) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def _lab_of_bgr(bgr: Tuple[int, int, int]) -> np.ndarray:
    cv2 = _import_cv2()
    px = np.uint8([[list(bgr)]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0][0].astype(np.float64)


def _bgr_of_lab(lab: Sequence[float]) -> Tuple[float, float, float]:
    cv2 = _import_cv2()
    px = np.uint8([[[int(max(0, min(255, round(float(v))))) for v in lab]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0][0]
    return float(b), float(g), float(r)


def _feature(lab: np.ndarray, cfg: TeamClassifierConfig) -> np.ndarray:
    """Lab (OpenCV 8-bit) -> clustering feature (L down-weighted)."""
    lab = np.asarray(lab, dtype=np.float64)
    out = lab.copy()
    out[..., 0] = lab[..., 0] * cfg.lightness_weight
    return out


def _frames_seeing(track, times: np.ndarray, tol_s: float) -> int:
    """How many of ``times`` fall on a sample of ``track`` (within ``tol_s``)."""
    if len(track) == 0 or times.size == 0:
        return 0
    idx = np.clip(np.searchsorted(track.t, times), 1, max(1, len(track) - 1))
    if len(track) == 1:
        near = np.abs(times - float(track.t[0]))
    else:
        near = np.minimum(np.abs(track.t[idx - 1] - times), np.abs(track.t[idx] - times))
    return int(np.sum(near <= tol_s))


def _plan_sample_times(
    tracking, sample_hz: float, max_frames: int, extra_frames: int, tol_s: float, min_samples: int = 2
) -> Tuple[np.ndarray, int]:
    """Uniform frame times over the whole window + extra frames for tracks it under-samples."""
    tracks = [tr for tr in tracking.players.values() if len(tr)]
    if not tracks:
        return np.zeros(0), 0
    t0 = min(float(tr.t[0]) for tr in tracks)
    t1 = max(float(tr.t[-1]) for tr in tracks)
    n = int(min(max(1, max_frames), max(1, math.ceil((t1 - t0) * max(1e-3, sample_hz)))))
    if n == 1:
        times = np.asarray([(t0 + t1) / 2.0])
    else:
        # Centre the samples inside n equal bins so both ends are covered.
        times = t0 + (np.arange(n) + 0.5) * (t1 - t0) / n
    # Tracks the uniform sample sees fewer than ``min_samples`` times get
    # extra frames spread inside them (longest tracks first; every extra
    # frame also serves the other tracks alive at that moment).
    needy = [tr for tr in tracks if _frames_seeing(tr, times, tol_s) < min_samples]
    needy.sort(key=lambda tr: -tr.duration_s)
    extra: List[float] = []
    for tr in needy:
        if len(extra) >= extra_frames:
            break
        have = _frames_seeing(tr, np.concatenate([times, np.asarray(extra)]), tol_s)
        need = min_samples - have
        if need <= 0:
            continue
        for q in np.linspace(0.0, 1.0, need + 2)[1:-1]:
            if len(extra) >= extra_frames:
                break
            extra.append(float(tr.t[min(len(tr) - 1, int(round(q * (len(tr) - 1))))]))
    all_times = np.unique(np.concatenate([times, np.asarray(extra, dtype=np.float64)]))
    return all_times, len(extra)


def _boxes_at_times(tracking, times: np.ndarray, tol_s: float) -> List[List[Tuple[int, np.ndarray]]]:
    """For each sample time, the (track_id, [x1, y1, x2, y2]) boxes visible."""
    per_frame: List[List[Tuple[int, np.ndarray]]] = [[] for _ in range(len(times))]
    if len(times) == 0:
        return per_frame
    for tr in tracking.players.values():
        n = len(tr)
        if n == 0:
            continue
        idx = np.searchsorted(tr.t, times)
        lo = np.clip(idx - 1, 0, n - 1)
        hi = np.clip(idx, 0, n - 1)
        pick = np.where(np.abs(tr.t[lo] - times) <= np.abs(tr.t[hi] - times), lo, hi)
        ok = np.abs(tr.t[pick] - times) <= tol_s
        for fi in np.flatnonzero(ok):
            j = int(pick[fi])
            per_frame[fi].append(
                (int(tr.track_id), np.asarray([tr.x1[j], tr.y1[j], tr.x2[j], tr.y2[j]], dtype=np.float64))
            )
    return per_frame


def _torso_descriptors(
    frame: np.ndarray,
    boxes: List[Tuple[int, np.ndarray]],
    scale: Tuple[float, float],
    cfg: TeamClassifierConfig,
) -> List[Tuple[int, np.ndarray]]:
    """Median Lab of grass-free torso pixels per (non-occluded) box."""
    cv2 = _import_cv2()
    if not boxes:
        return []
    h, w = frame.shape[:2]
    sx, sy = scale
    arr = np.stack([b for _, b in boxes]) * np.asarray([sx, sy, sx, sy])
    bh = arr[:, 3] - arr[:, 1]
    bw = arr[:, 2] - arr[:, 0]
    tx1 = arr[:, 0] + bw * (1.0 - cfg.torso_width_frac) / 2.0
    tx2 = arr[:, 2] - bw * (1.0 - cfg.torso_width_frac) / 2.0
    ty1 = arr[:, 1] + bh * cfg.torso_top_frac
    ty2 = arr[:, 1] + bh * cfg.torso_bottom_frac
    torso_area = np.maximum(1e-6, (tx2 - tx1) * (ty2 - ty1))
    # Occlusion by boxes in front (larger y2 = nearer a side-line camera).
    ix = np.clip(np.minimum(tx2[:, None], arr[None, :, 2]) - np.maximum(tx1[:, None], arr[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(ty2[:, None], arr[None, :, 3]) - np.maximum(ty1[:, None], arr[None, :, 1]), 0, None)
    inter = ix * iy
    in_front = arr[None, :, 3] > arr[:, None, 3]
    np.fill_diagonal(in_front, False)
    occluded = (inter * in_front).max(axis=1) / torso_area if len(boxes) > 1 else np.zeros(len(boxes))

    out: List[Tuple[int, np.ndarray]] = []
    for k, (track_id, _box) in enumerate(boxes):
        if bh[k] < cfg.min_box_height_px or occluded[k] > cfg.max_occluded_frac:
            continue
        x0, x1 = int(max(0, math.floor(tx1[k]))), int(min(w, math.ceil(tx2[k])))
        y0, y1 = int(max(0, math.floor(ty1[k]))), int(min(h, math.ceil(ty2[k])))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        patch = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        grass = (hsv[:, 0] > 35) & (hsv[:, 0] < 85) & (hsv[:, 1] > 60)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3)[~grass]
        if len(lab) < max(cfg.min_torso_pixels, int(0.25 * len(hsv))):
            continue
        out.append((track_id, np.median(lab.astype(np.float64), axis=0)))
    return out


def _read_sampled_frames(video_path: str, times: np.ndarray):
    """Yield (sample_index, frame) for each time; sequential grab when dense."""
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.warning("team assignment skipped: cannot open %s", video_path)
        return
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3 or not math.isfinite(fps):
            fps = 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = np.round(np.asarray(times) * fps).astype(np.int64)
        if frame_count > 0:
            indices = np.clip(indices, 0, frame_count - 1)
        pos = 0
        seq_limit = max(30, int(round(fps * 1.5)))
        for i, target in enumerate(indices):
            target = int(target)
            if target < pos or target - pos > seq_limit:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
                pos = target
            while pos < target:
                if not cap.grab():
                    break
                pos += 1
            ok, frame = cap.read()
            if not ok:
                continue
            pos = target + 1
            yield i, frame
    finally:
        cap.release()


def _kmeans2(x: np.ndarray, weights: np.ndarray, cap_dist: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Robust weighted 2-means (truncated loss); returns (centres[2,3], labels)."""
    n = len(x)
    rng = np.random.default_rng(seed)
    inits: List[Tuple[int, int]] = []
    d_all = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(d_all)), d_all.shape)
    inits.append((int(i), int(j)))
    for _ in range(12):
        a = int(rng.choice(n, p=weights / weights.sum()))
        p = d_all[a] ** 2 * weights
        if p.sum() <= 0:
            continue
        b = int(rng.choice(n, p=p / p.sum()))
        inits.append((a, b))
    best = None
    for a, b in inits:
        centres = np.stack([x[a], x[b]]).astype(np.float64)
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(30):
            d = np.linalg.norm(x[:, None, :] - centres[None, :, :], axis=2)
            labels = np.argmin(d, axis=1)
            dmin = d[np.arange(n), labels]
            inlier = dmin <= cap_dist
            new = centres.copy()
            for k in range(2):
                m = (labels == k) & inlier
                if not m.any():
                    m = labels == k
                if m.any():
                    new[k] = np.average(x[m], axis=0, weights=weights[m])
            if np.allclose(new, centres, atol=1e-3):
                centres = new
                break
            centres = new
        d = np.linalg.norm(x[:, None, :] - centres[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        loss = float(np.sum(weights * np.minimum(d[np.arange(n), labels], cap_dist) ** 2))
        if np.linalg.norm(centres[0] - centres[1]) < 1e-6:
            loss = float("inf")
        if best is None or loss < best[0]:
            best = (loss, centres, labels)
    assert best is not None
    return best[1], best[2]


def _cluster_tracks(
    feats: np.ndarray, weights: np.ndarray, cfg: TeamClassifierConfig
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """Two kit clusters + "other" (-1) for far tracks. Returns (centres, labels, outlier_threshold)."""
    n = len(feats)
    if n < 2:
        return None, np.full(n, -1, dtype=np.int64), cfg.min_outlier_distance
    cap = cfg.min_outlier_distance * 2.0
    centres, labels = _kmeans2(feats, weights, cap)
    thr = cfg.min_outlier_distance
    for _ in range(3):
        d = np.linalg.norm(feats[:, None, :] - centres[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        dmin = d[np.arange(n), labels]
        spread = float(np.median(dmin)) if n else 0.0
        thr = max(cfg.min_outlier_distance, cfg.outlier_spread_factor * spread)
        inlier = dmin <= thr
        for k in range(2):
            m = inlier & (labels == k)
            if m.any():
                centres[k] = np.average(feats[m], axis=0, weights=weights[m])
    d = np.linalg.norm(feats[:, None, :] - centres[None, :, :], axis=2)
    labels = np.argmin(d, axis=1)
    labels[d[np.arange(n), labels] > thr] = -1
    return centres, labels, thr


def assign_track_teams_with_report(
    video_path: str,
    tracking,
    team_a: Optional[TeamConfig] = None,
    team_b: Optional[TeamConfig] = None,
    *,
    sample_hz: float = 0.5,
    max_frames: int = 400,
    cfg: Optional[TeamClassifierConfig] = None,
) -> Tuple[object, TeamAssignmentReport]:
    """:func:`assign_track_teams` that also returns the :class:`TeamAssignmentReport`."""
    from .tracking_types import TEAM_A, TEAM_B, TEAM_REFEREE, TEAM_UNKNOWN

    cfg = cfg or TeamClassifierConfig()
    report = TeamAssignmentReport(tracks_total=len(tracking.players))
    if not tracking.players:
        return tracking, report

    fps = float(tracking.fps or 25.0) / max(1, int(tracking.vid_stride or 1))
    tol_s = max(0.06, 1.5 / max(1.0, fps))
    times, n_extra = _plan_sample_times(tracking, sample_hz, max_frames, cfg.extra_frames, tol_s,
                                         max(1, cfg.min_track_samples))
    report.frames_planned = int(len(times))
    boxes_per_frame = _boxes_at_times(tracking, times, tol_s)

    samples: Dict[int, List[np.ndarray]] = {}
    for fi, frame in _read_sampled_frames(video_path, times):
        report.frames_read += 1
        fh, fw = frame.shape[:2]
        scale = (fw / max(1.0, float(tracking.frame_width)), fh / max(1.0, float(tracking.frame_height)))
        for track_id, lab in _torso_descriptors(frame, boxes_per_frame[fi], scale, cfg):
            samples.setdefault(track_id, []).append(lab)
    report.samples = int(sum(len(v) for v in samples.values()))

    track_ids = [tid for tid, v in samples.items() if len(v) >= cfg.min_track_samples]
    sample_feats = {tid: _feature(np.stack(samples[tid]), cfg) for tid in samples}
    medians_lab = {tid: np.median(np.stack(samples[tid]), axis=0) for tid in samples}
    track_feats = np.stack([np.median(sample_feats[tid], axis=0) for tid in track_ids]) if track_ids else np.zeros((0, 3))
    weights = np.asarray([min(len(samples[tid]), 50) for tid in track_ids], dtype=np.float64)

    supplied = [None if c is None else _feature(_lab_of_bgr(hex_to_bgr(c.color_hex)), cfg) for c in (team_a, team_b)]
    centres, cluster_labels, thr = _cluster_tracks(track_feats, weights, cfg) if len(track_ids) >= 2 else (
        None, np.full(len(track_ids), -1, dtype=np.int64), cfg.min_outlier_distance)

    refs: Optional[np.ndarray] = None
    if centres is not None:
        # Name clusters: by supplied colours when given, else by population.
        if supplied[0] is not None and supplied[1] is not None:
            keep = float(np.linalg.norm(centres[0] - supplied[0]) + np.linalg.norm(centres[1] - supplied[1]))
            swap = float(np.linalg.norm(centres[1] - supplied[0]) + np.linalg.norm(centres[0] - supplied[1]))
            order = [0, 1] if keep <= swap else [1, 0]
            mapped = centres[order]
            if max(np.linalg.norm(mapped[0] - supplied[0]), np.linalg.norm(mapped[1] - supplied[1])) <= cfg.supplied_color_max_distance:
                refs = mapped
                report.colors_source = "supplied+adapted"
            else:
                refs = np.stack([supplied[0], supplied[1]])
                report.colors_source = "supplied"
        elif supplied[0] is not None or supplied[1] is not None:
            known_k = 0 if supplied[0] is not None else 1
            target = supplied[known_k]
            nearest = int(np.argmin(np.linalg.norm(centres - target, axis=1)))
            pair = [nearest, 1 - nearest] if known_k == 0 else [1 - nearest, nearest]
            refs = centres[pair]
            report.colors_source = "supplied+adapted"
        else:
            pop = [float(weights[cluster_labels == k].sum()) for k in (0, 1)]
            refs = centres[[0, 1] if pop[0] >= pop[1] else [1, 0]]
            report.colors_source = "detected"
    elif supplied[0] is not None and supplied[1] is not None:
        refs = np.stack([supplied[0], supplied[1]])
        report.colors_source = "supplied"
    if report.colors_source == "supplied":
        # Picked colours rarely match the on-camera kit exactly: be lenient.
        thr = max(thr, cfg.supplied_color_max_distance * 0.66)

    if refs is not None:
        report.team_colors_hex = {
            "0": _bgr_to_hex(_bgr_of_lab(_unfeature(refs[0], cfg))),
            "1": _bgr_to_hex(_bgr_of_lab(_unfeature(refs[1], cfg))),
            "0_supplied": team_a.color_hex if team_a else None,
            "1_supplied": team_b.color_hex if team_b else None,
        }

    other_feats: List[np.ndarray] = []
    for track in tracking.players.values():
        tid = int(track.track_id)
        lab_samples = samples.get(tid)
        if lab_samples:
            track.jersey_color_hex = _bgr_to_hex(_bgr_of_lab(medians_lab[tid]))
        if refs is None or not lab_samples or len(lab_samples) < cfg.min_track_samples:
            track.team = TEAM_UNKNOWN
            track.team_confidence = 0.0
            report.per_track[tid] = {"team": TEAM_UNKNOWN, "samples": len(lab_samples or [])}
            continue
        f = sample_feats[tid]
        d = np.linalg.norm(f[:, None, :] - refs[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        votes_other = d[np.arange(len(f)), nearest] > thr
        v0 = int(np.sum((nearest == 0) & ~votes_other))
        v1 = int(np.sum((nearest == 1) & ~votes_other))
        vo = int(np.sum(votes_other))
        total = max(1, v0 + v1 + vo)
        if vo > max(v0, v1):
            track.team = TEAM_REFEREE
            track.team_confidence = round(vo / total, 3)
            other_feats.append(np.median(f, axis=0))
        else:
            track.team = TEAM_A if v0 >= v1 else TEAM_B
            track.team_confidence = round(max(v0, v1) / total, 3)
        report.per_track[tid] = {"team": int(track.team), "votes": [v0, v1, vo],
                                 "confidence": track.team_confidence, "samples": len(f)}

    if other_feats:
        report.team_colors_hex["other"] = _bgr_to_hex(
            _bgr_of_lab(_unfeature(np.median(np.stack(other_feats), axis=0), cfg))
        )
    if cfg.goalkeeper_by_position:
        report.tracks_goalkeeper_by_position = _goalkeepers_by_position(tracking, cfg)

    teams = [int(tr.team) for tr in tracking.players.values()]
    report.tracks_team0 = teams.count(TEAM_A)
    report.tracks_team1 = teams.count(TEAM_B)
    report.tracks_referee = teams.count(TEAM_REFEREE)
    report.tracks_unknown = teams.count(TEAM_UNKNOWN)
    report.track_coverage_pct = 100.0 * (len(teams) - report.tracks_unknown) / max(1, len(teams))
    n_all = sum(len(tr) for tr in tracking.players.values())
    n_lab = sum(len(tr) for tr in tracking.players.values() if int(tr.team) != TEAM_UNKNOWN)
    report.sample_coverage_pct = 100.0 * n_lab / max(1, n_all)
    try:
        tracking.detector["team_assignment"] = report.to_dict()
    except Exception:  # pragma: no cover - detector is a plain dict
        pass
    LOGGER.info(
        "team assignment: %d frames (%d extra), %d samples; tracks team0=%d team1=%d referee=%d unknown=%d "
        "(coverage %.1f%% of tracks, %.1f%% of samples), colours %s %s",
        report.frames_read, n_extra, report.samples, report.tracks_team0, report.tracks_team1,
        report.tracks_referee, report.tracks_unknown, report.track_coverage_pct, report.sample_coverage_pct,
        report.colors_source, report.team_colors_hex,
    )
    return tracking, report


def _unfeature(feat: np.ndarray, cfg: TeamClassifierConfig) -> np.ndarray:
    lab = np.asarray(feat, dtype=np.float64).copy()
    lab[0] = lab[0] / max(1e-6, cfg.lightness_weight)
    return lab


def _goalkeepers_by_position(tracking, cfg: TeamClassifierConfig) -> int:
    """Third-colour tracks that live in one goal area join the team defending it.

    Field ends come from robust percentiles of every foot position; which
    team defends each end comes from the median x of each labelled team
    (over the whole window: a goalkeeper track that survives halftime is
    rare, and the swap is handled by the track ending/restarting).
    """
    from .tracking_types import TEAM_A, TEAM_B, TEAM_REFEREE

    feet = [tr.foot_xy()[:, 0] for tr in tracking.players.values() if len(tr)]
    if not feet:
        return 0
    all_x = np.concatenate(feet)
    x_lo, x_hi = float(np.percentile(all_x, 1.0)), float(np.percentile(all_x, 99.0))
    span = max(1.0, x_hi - x_lo)
    med = {}
    for team in (TEAM_A, TEAM_B):
        xs = [tr.foot_xy()[:, 0] for tr in tracking.players.values() if int(tr.team) == team and len(tr)]
        if xs:
            med[team] = float(np.median(np.concatenate(xs)))
    if len(med) < 2 or abs(med[TEAM_A] - med[TEAM_B]) < 0.05 * span:
        return 0
    left_team = TEAM_A if med[TEAM_A] < med[TEAM_B] else TEAM_B
    right_team = TEAM_B if left_team == TEAM_A else TEAM_A
    zone = cfg.goalkeeper_zone_frac * span
    changed = 0
    for tr in tracking.players.values():
        if int(tr.team) != TEAM_REFEREE or len(tr) < 10:
            continue
        x = tr.foot_xy()[:, 0]
        in_left = float(np.mean(x <= x_lo + zone))
        in_right = float(np.mean(x >= x_hi - zone))
        if in_left >= cfg.goalkeeper_min_share:
            tr.team, changed = left_team, changed + 1
        elif in_right >= cfg.goalkeeper_min_share:
            tr.team, changed = right_team, changed + 1
        else:
            continue
        tr.team_confidence = round(min(float(tr.team_confidence), 0.5), 3)
    return changed


def assign_track_teams(
    video_path: str,
    tracking,
    team_a: Optional[TeamConfig] = None,
    team_b: Optional[TeamConfig] = None,
    *,
    sample_hz: float = 0.5,
    max_frames: int = 400,
    cfg: Optional[TeamClassifierConfig] = None,
):
    """Label every track of a ``TrackingResult`` with a team (in place).

    Frames are sampled uniformly over the WHOLE match (``sample_hz``, capped
    at ``max_frames`` frames spread evenly, plus up to ``cfg.extra_frames``
    frames for tracks the uniform sample misses). For every visible track a
    torso patch (rows 25-55% of the box, central 50% width) is cut, grass
    pixels are removed and the median Lab colour is kept. Each track is then
    labelled by majority vote of its samples against two kit centres:

    * colours supplied: the two kit clusters found in the data are matched to
      the supplied colours (so the labels follow the real on-camera
      appearance); if the clustering disagrees with the supplied colours, the
      supplied colours are used directly;
    * colours not supplied: tracks are clustered into two kits (robust
      2-means on Lab with lightness down-weighted); team 0 is the more
      populous cluster. Detected colours are reported.

    Tracks whose samples are mostly far from both kits become
    ``TEAM_REFEREE`` (referees, linesmen, and - unless they stay in one goal
    area, see ``cfg.goalkeeper_by_position`` - goalkeepers). Tracks with
    fewer than ``cfg.min_track_samples`` good samples stay ``TEAM_UNKNOWN``.

    Sets ``PlayerTrack.team``, ``team_confidence`` and ``jersey_color_hex``;
    stores the :class:`TeamAssignmentReport` summary (coverage, colours) in
    ``tracking.detector["team_assignment"]`` and returns ``tracking``.
    ``video_path`` may be the proxy: boxes are rescaled to the frame size.
    """
    result, _report = assign_track_teams_with_report(
        video_path, tracking, team_a, team_b, sample_hz=sample_hz, max_frames=max_frames, cfg=cfg
    )
    return result
