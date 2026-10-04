"""Player selection and identity stitching.

* :func:`resolve_player_roi_box` / :func:`choose_target_track_id` /
  :func:`stitch_target_track`: legacy helpers working on ``TrackPoint``-like
  objects (``.t``, ``.xy``, ``.bbox``); still used by ``VideoHighlights``.
* :func:`stitch_tracks`: whole-match re-identification. Links tracker
  fragments into identities using motion continuity *and* an appearance
  embedding (torso colour histogram). Candidates are found with a sorted
  start-time index and a bounded gap window, so the cost is
  ``O(n log n + k)`` for ``n`` fragments and ``k`` candidate pairs instead of
  the all-pairs, per-step scan of :func:`stitch_target_track`.
* :func:`select_track_at`: pick the identity under a user-drawn box at a
  given time (the "track this player" picker).
"""

from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("videohighlights.player_focus")


def resolve_player_roi_box(
    raw_roi: Optional[Dict[str, object]],
    frame_width: int,
    frame_height: int,
) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(raw_roi, dict) or frame_width <= 0 or frame_height <= 0:
        return None

    normalized = bool(raw_roi.get("normalized", False))

    if all(key in raw_roi for key in ("x1_norm", "y1_norm", "x2_norm", "y2_norm")):
        normalized = True
        x1 = float(raw_roi.get("x1_norm", 0.0))
        y1 = float(raw_roi.get("y1_norm", 0.0))
        x2 = float(raw_roi.get("x2_norm", 1.0))
        y2 = float(raw_roi.get("y2_norm", 1.0))
    elif all(key in raw_roi for key in ("x", "y", "w", "h")):
        x = float(raw_roi.get("x", 0.0))
        y = float(raw_roi.get("y", 0.0))
        w = float(raw_roi.get("w", 0.0))
        h = float(raw_roi.get("h", 0.0))
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
    else:
        return None

    if normalized:
        x1 *= frame_width
        x2 *= frame_width
        y1 *= frame_height
        y2 *= frame_height

    x1 = min(max(0.0, x1), float(frame_width))
    x2 = min(max(0.0, x2), float(frame_width))
    y1 = min(max(0.0, y1), float(frame_height))
    y2 = min(max(0.0, y2), float(frame_height))

    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return x1, y1, x2, y2


def box_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _box_area(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _point_time(point: object) -> float:
    return float(getattr(point, "t", 0.0))


def _point_center(point: object) -> Optional[Tuple[float, float]]:
    center = getattr(point, "xy", None)
    if center is None:
        return None
    return float(center[0]), float(center[1])


def _point_bbox(point: object) -> Optional[Tuple[float, float, float, float]]:
    bbox = getattr(point, "bbox", None)
    if bbox is None:
        return None
    try:
        return tuple(float(value) for value in bbox)
    except Exception:
        return None


def choose_target_track_id(
    tracks: Dict[int, Iterable[object]],
    user_box: Tuple[float, float, float, float],
    window_t: float = 3.0,
) -> Optional[int]:
    ux = (user_box[0] + user_box[2]) / 2.0
    uy = (user_box[1] + user_box[3]) / 2.0

    best_id: Optional[int] = None
    best_score: Optional[Tuple[float, float, float, float, int]] = None

    for track_id, raw_traj in tracks.items():
        early = [point for point in raw_traj if _point_time(point) <= window_t]
        if not early:
            continue

        ious = []
        inside_hits = 0
        dists = []
        for point in early:
            center = _point_center(point)
            if center is None:
                continue
            cx, cy = center
            dists.append(((cx - ux) ** 2 + (cy - uy) ** 2) ** 0.5)
            if user_box[0] <= cx <= user_box[2] and user_box[1] <= cy <= user_box[3]:
                inside_hits += 1

            bbox = _point_bbox(point)
            if bbox is not None:
                try:
                    ious.append(box_iou(user_box, bbox))
                except Exception:
                    pass

        if not dists:
            continue

        max_iou = max(ious) if ious else 0.0
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        inside_ratio = inside_hits / max(1, len(early))
        mean_dist = sum(dists) / len(dists)
        score = (
            max_iou,
            mean_iou,
            inside_ratio,
            -mean_dist,
            len(early),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_id = int(track_id)

    return best_id


def stitch_target_track(
    tracks: Dict[int, Iterable[object]],
    target_track_id: int,
    *,
    max_gap_seconds: float = 1.0,
    overlap_tolerance_seconds: float = 0.35,
    max_center_distance: float = 140.0,
    min_link_iou: float = 0.02,
) -> Tuple[List[int], List[object]]:
    if int(target_track_id) not in tracks:
        return [], []

    ordered_tracks: Dict[int, List[object]] = {}
    for track_id, raw_traj in tracks.items():
        points = list(raw_traj or [])
        if not points:
            continue
        ordered_tracks[int(track_id)] = sorted(points, key=_point_time)

    if int(target_track_id) not in ordered_tracks:
        return [], []

    stitched_ids = [int(target_track_id)]
    used_ids = {int(target_track_id)}
    stitched = list(ordered_tracks[int(target_track_id)])

    while stitched:
        tail = stitched[-1]
        tail_t = _point_time(tail)
        tail_center = _point_center(tail)
        if tail_center is None:
            break
        tail_bbox = _point_bbox(tail)
        tail_area = _box_area(tail_bbox) if tail_bbox is not None else 0.0
        dynamic_max_distance = max(
            float(max_center_distance),
            (tail_bbox[2] - tail_bbox[0]) * 1.75 if tail_bbox is not None else 0.0,
        )

        best_candidate_id: Optional[int] = None
        best_candidate_points: List[object] = []
        best_candidate_score: Optional[Tuple[float, float, float, float, float, int]] = None

        for track_id, candidate_points in ordered_tracks.items():
            if track_id in used_ids:
                continue

            window_points = [
                point
                for point in candidate_points
                if (tail_t - float(overlap_tolerance_seconds)) <= _point_time(point) <= (tail_t + float(max_gap_seconds))
            ]
            if not window_points:
                continue

            anchor = min(window_points, key=_point_time)
            anchor_center = _point_center(anchor)
            if anchor_center is None:
                continue

            gap = _point_time(anchor) - tail_t
            center_distance = ((anchor_center[0] - tail_center[0]) ** 2 + (anchor_center[1] - tail_center[1]) ** 2) ** 0.5

            anchor_bbox = _point_bbox(anchor)
            link_iou = box_iou(tail_bbox, anchor_bbox) if tail_bbox is not None and anchor_bbox is not None else 0.0
            anchor_area = _box_area(anchor_bbox) if anchor_bbox is not None else 0.0
            size_similarity = 0.0
            if tail_area > 0.0 and anchor_area > 0.0:
                size_similarity = min(tail_area, anchor_area) / max(tail_area, anchor_area)

            if center_distance > dynamic_max_distance and link_iou < float(min_link_iou):
                continue

            append_points = [point for point in candidate_points if _point_time(point) > (tail_t + 1e-6)]
            if not append_points:
                continue

            score = (
                link_iou,
                size_similarity,
                -max(0.0, gap),
                -center_distance,
                -abs(gap),
                len(append_points),
            )
            if best_candidate_score is None or score > best_candidate_score:
                best_candidate_score = score
                best_candidate_id = track_id
                best_candidate_points = append_points

        if best_candidate_id is None or not best_candidate_points:
            break

        used_ids.add(best_candidate_id)
        stitched_ids.append(best_candidate_id)
        stitched.extend(best_candidate_points)

    stitched.sort(key=_point_time)
    return stitched_ids, stitched


# ----------------------------------------------------------------------
# Whole-match identity stitching (motion + appearance)
# ----------------------------------------------------------------------

Box = Tuple[float, float, float, float]


@dataclass
class TrackFragment:
    """Summary of one raw tracker ID used for stitching.

    ``start_vel``/``end_vel`` are box-center velocities (px/s) estimated over
    the first/last ~0.5 s. ``appearance`` is an L1-normalised colour
    histogram (or ``None`` when unavailable). ``team`` (when known, >= 0)
    forbids links across teams.
    """

    track_id: int
    t_start: float
    t_end: float
    start_box: Box
    end_box: Box
    start_vel: Tuple[float, float] = (0.0, 0.0)
    end_vel: Tuple[float, float] = (0.0, 0.0)
    appearance: Optional[np.ndarray] = None
    samples: int = 0
    team: int = -1
    #: appearance from the first / last few samples (preferred for linking,
    #: robust to a fragment drifting onto another player later on)
    start_appearance: Optional[np.ndarray] = None
    end_appearance: Optional[np.ndarray] = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t_end - self.t_start)


@dataclass
class StitchLink:
    """An accepted ``prev -> next`` link with its evidence (for logs/debug)."""

    prev_id: int
    next_id: int
    gap_s: float
    distance_px: float
    radius_px: float
    appearance_sim: Optional[float]
    cost: float


@dataclass
class StitchResult:
    chains: List[List[int]]
    links: List[StitchLink] = field(default_factory=list)
    candidates_considered: int = 0


def appearance_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Bhattacharyya coefficient of two L1-normalised histograms (1 = identical)."""
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape or a.sum() <= 0 or b.sum() <= 0:
        return None
    a = a / a.sum()
    b = b / b.sum()
    return float(np.sqrt(a * b).sum())


def _center(box: Box) -> Tuple[float, float]:
    return (float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5


def _height(box: Box) -> float:
    return max(1.0, float(box[3]) - float(box[1]))


def stitch_tracks(
    fragments: Sequence[TrackFragment],
    *,
    max_gap_s: float = 3.0,
    overlap_tolerance_s: float = 0.2,
    base_radius_heights: float = 1.2,
    speed_heights_per_s: float = 4.0,
    max_extrapolate_s: float = 1.0,
    max_size_ratio: float = 1.8,
    min_appearance_sim: float = 0.55,
    appearance_weight: float = 0.5,
    return_details: bool = False,
):
    """Link tracker fragments into identities.

    A link ``a -> b`` is a candidate when ``b`` starts within
    ``[a.t_end - overlap_tolerance_s, a.t_end + max_gap_s]``, ``b``'s first
    box center lies within a radius of ``a``'s last box center extrapolated
    with ``a``'s velocity (radius grows with the gap: ``h * (base +
    speed * gap)`` where ``h`` is the box height, i.e. a scale-free
    "player heights per second" speed bound), box heights agree within
    ``max_size_ratio``, teams agree when both known, and appearance
    similarity (``a``'s end vs ``b``'s start embedding, falling back to the
    whole-fragment means) is at least ``min_appearance_sim`` when both have
    one. Candidates are accepted greedily by ascending cost
    (normalised distance blended with appearance dissimilarity), each
    fragment getting at most one predecessor and one successor.

    Complexity: fragments are sorted by start time once and each fragment's
    candidates are found with :func:`bisect` over the start times, so the
    work is ``O(n log n + k log k)`` for ``k`` candidate pairs.

    Returns:
        ``List[List[int]]`` chains of track ids in time order (every input id
        appears exactly once), or a :class:`StitchResult` when
        ``return_details`` is true.
    """
    frags = [f for f in fragments if f.samples > 0 or f.t_end >= f.t_start]
    by_start = sorted(frags, key=lambda f: (f.t_start, f.track_id))
    starts = [f.t_start for f in by_start]
    w_app = min(1.0, max(0.0, float(appearance_weight)))

    candidates: List[Tuple[float, int, int, StitchLink]] = []
    considered = 0
    for a in frags:
        lo = bisect.bisect_left(starts, a.t_end - overlap_tolerance_s)
        hi = bisect.bisect_right(starts, a.t_end + max_gap_s)
        if lo >= hi:
            continue
        ax, ay = _center(a.end_box)
        ah = _height(a.end_box)
        for b in by_start[lo:hi]:
            if b.track_id == a.track_id or b.t_end <= a.t_end:
                continue
            considered += 1
            if a.team >= 0 and b.team >= 0 and a.team != b.team:
                continue
            gap = b.t_start - a.t_end
            bh = _height(b.start_box)
            if max(ah, bh) / min(ah, bh) > max_size_ratio:
                continue
            g = min(max(0.0, gap), max_extrapolate_s)
            px = ax + a.end_vel[0] * g
            py = ay + a.end_vel[1] * g
            bx, by = _center(b.start_box)
            dist = math.hypot(bx - px, by - py)
            h = 0.5 * (ah + bh)
            radius = h * (base_radius_heights + speed_heights_per_s * max(0.0, gap))
            if dist > radius:
                continue
            a_app = a.end_appearance if a.end_appearance is not None else a.appearance
            b_app = b.start_appearance if b.start_appearance is not None else b.appearance
            sim = appearance_similarity(a_app, b_app)
            if sim is not None and sim < min_appearance_sim:
                continue
            motion_cost = dist / max(radius, 1e-6)
            if sim is None:
                cost = motion_cost
            else:
                cost = (1.0 - w_app) * motion_cost + w_app * (1.0 - sim) / max(1e-6, 1.0 - min_appearance_sim)
            cost += 0.1 * max(0.0, gap) / max(max_gap_s, 1e-6)
            link = StitchLink(a.track_id, b.track_id, gap, dist, radius, sim, cost)
            candidates.append((cost, a.track_id, b.track_id, link))

    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    successor: Dict[int, int] = {}
    predecessor: Dict[int, int] = {}
    accepted: List[StitchLink] = []
    for _cost, a_id, b_id, link in candidates:
        if a_id in successor or b_id in predecessor:
            continue
        successor[a_id] = b_id
        predecessor[b_id] = a_id
        accepted.append(link)

    chains: List[List[int]] = []
    for f in by_start:
        if f.track_id in predecessor:
            continue
        chain = [f.track_id]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])
        chains.append(chain)
    if accepted:
        logger.debug("stitch_tracks: %d fragments -> %d identities (%d links, %d candidates)",
                     len(frags), len(chains), len(accepted), considered)
    if return_details:
        return StitchResult(chains=chains, links=accepted, candidates_considered=considered)
    return chains


def select_track_at(
    tracks: Mapping[int, object],
    box: Box,
    t: float = 0.0,
    *,
    window_s: float = 0.5,
    max_center_distance_boxes: float = 1.5,
) -> Optional[int]:
    """Pick the track whose box best matches ``box`` near time ``t``.

    ``tracks`` maps id -> object with aligned float arrays ``t, x1, y1, x2,
    y2`` (e.g. :class:`~backend.services.tracking_types.PlayerTrack`). For
    each track the sample nearest ``t`` within ``window_s`` is compared to
    ``box``; ranking is (IoU, center-inside, -center distance). Tracks with no
    overlap whose center is further than ``max_center_distance_boxes`` box
    diagonals from the ROI center are ignored. Returns ``None`` when nothing
    qualifies.
    """
    ux = (box[0] + box[2]) * 0.5
    uy = (box[1] + box[3]) * 0.5
    diag = math.hypot(box[2] - box[0], box[3] - box[1])
    best_id: Optional[int] = None
    best_score: Optional[Tuple[float, float, float]] = None
    for track_id, track in tracks.items():
        ts = np.asarray(getattr(track, "t", ()), dtype=np.float64)
        if ts.size == 0:
            continue
        idx = int(np.searchsorted(ts, t))
        options = [i for i in (idx - 1, idx) if 0 <= i < ts.size]
        i = min(options, key=lambda k: abs(ts[k] - t))
        if abs(ts[i] - t) > window_s:
            continue
        cand = (float(track.x1[i]), float(track.y1[i]), float(track.x2[i]), float(track.y2[i]))
        iou = box_iou(box, cand)
        cx, cy = _center(cand)
        inside = 1.0 if (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]) else 0.0
        dist = math.hypot(cx - ux, cy - uy)
        if iou <= 0.0 and inside == 0.0 and dist > max_center_distance_boxes * max(diag, 1.0):
            continue
        score = (iou, inside, -dist)
        if best_score is None or score > best_score:
            best_score = score
            best_id = int(track_id)
    return best_id
