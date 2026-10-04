"""Batched detection + multi-object tracking over the analysis proxy.

``track_video`` is the single detection pass of a run:

    FrameReader (thread, bounded queue)
        -> detector.detect_batch(frames)         (GPU, batched)
        -> tracker worker (thread): person tracker update per frame,
           raw ball detections, torso-colour appearance samples
        -> post-process: fragments -> stitched identities (motion +
           appearance), drop short identities, rescale to source pixels
        -> TrackingResult (all players + all raw ball detections)

Trackers
--------
``"bytetrack"`` (default) is a compact, dependency-free ByteTrack
(:class:`ByteTrackLite`): Kalman filter in (cx, cy, aspect, h) space,
two-stage high/low score association with Hungarian matching
(``scipy.optimize.linear_sum_assignment``), plus a scale-free
center-distance fallback stage for small fast players whose boxes stop
overlapping between frames. ``"botsort"`` uses ``ultralytics``'
``BOTSORT`` class through a small Results-like shim (GMC off by default:
panoramic cameras are static); it falls back to ``bytetrack`` when the
ultralytics tracker cannot be constructed.

Only PERSON detections go through the tracker. BALL detections are kept
raw (every detection with its confidence and size) in
:class:`~backend.services.tracking_types.BallDetections`.

Memory: rows are appended to a growing float32 table (7 floats per player
sample), so a 90 min x 30 fps x 25 player match is ~115 MB.
"""

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .detectors import BALL, PERSON, Detections, Detector, iou_matrix
from .frame_source import FrameItem, FrameReader, ProxyResult
from .player_focus import (
    TrackFragment,
    resolve_player_roi_box,
    select_track_at,
    stitch_tracks,
)
from .tracking_types import BallDetections, PlayerTrack, TrackingResult

logger = logging.getLogger("videohighlights.tracking_engine")

ProgressCallback = Callable[[str, float, str, Optional[Dict[str, object]]], None]

_PLAYER_COLS = 7  # track_id, frame_index, x1, y1, x2, y2, conf
_BALL_COLS = 6  # frame_index, cx, cy, w, h, conf


# ----------------------------------------------------------------------
# Growing row table
# ----------------------------------------------------------------------


class RowBuffer:
    """Append-only float32 table with amortised O(1) appends."""

    def __init__(self, cols: int, capacity: int = 4096) -> None:
        self.cols = int(cols)
        self._data = np.empty((max(16, int(capacity)), self.cols), dtype=np.float32)
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def _reserve(self, extra: int) -> None:
        need = self._n + extra
        if need <= self._data.shape[0]:
            return
        cap = self._data.shape[0]
        while cap < need:
            cap *= 2
        grown = np.empty((cap, self.cols), dtype=np.float32)
        grown[: self._n] = self._data[: self._n]
        self._data = grown

    def append_rows(self, rows: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=np.float32).reshape(-1, self.cols)
        if not len(rows):
            return
        self._reserve(len(rows))
        self._data[self._n:self._n + len(rows)] = rows
        self._n += len(rows)

    def array(self) -> np.ndarray:
        return self._data[: self._n]

    @property
    def nbytes(self) -> int:
        return int(self._data.nbytes)


# ----------------------------------------------------------------------
# Tracker configuration + ByteTrack-lite
# ----------------------------------------------------------------------


@dataclass
class TrackerConfig:
    """Association thresholds (defaults tuned for YOLO person scores)."""

    track_high_thresh: float = 0.3
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.4
    match_thresh: float = 0.8  # max fused cost (1 - IoU*score) in stage 1
    low_match_iou: float = 0.5  # min IoU for stage 2 (low-score dets)
    unconfirmed_match_thresh: float = 0.7
    center_match_heights: float = 0.6  # stage-3 gate: center distance / box height
    track_buffer_s: float = 1.5  # keep lost tracks this long
    lost_velocity_decay: float = 0.9  # per processed frame while lost
    fuse_score: bool = True
    center_cost_weight: float = 0.5  # stage-1 ranking: + w * center distance / box height
    # Appearance (torso colour) term, evaluated only for ambiguous stage-1
    # pairs (a track with >= 2 gated detections or vice versa); prevents
    # identity swaps between differently dressed players when they cross.
    appearance_weight: float = 0.5
    appearance_update_every: int = 5  # processed frames between EMA updates
    appearance_ema: float = 0.2
    gmc_method: str = "none"  # BoT-SORT only


_STD_POS = 1.0 / 20.0
_STD_VEL = 1.0 / 160.0


def _xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    w = np.maximum(b[:, 2] - b[:, 0], 1e-3)
    h = np.maximum(b[:, 3] - b[:, 1], 1e-3)
    return np.stack([(b[:, 0] + b[:, 2]) * 0.5, (b[:, 1] + b[:, 3]) * 0.5, w / h, h], axis=1)


def _xyah_to_xyxy(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64).reshape(-1, 4)
    h = m[:, 3]
    w = m[:, 2] * h
    return np.stack([m[:, 0] - w / 2, m[:, 1] - h / 2, m[:, 0] + w / 2, m[:, 1] + h / 2], axis=1)


_F = np.eye(8)
for _i in range(4):
    _F[_i, _i + 4] = 1.0
_H = np.eye(4, 8)


def _kalman_init(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = z[3]
    mean = np.concatenate([z, np.zeros(4)])
    std = np.array([2 * _STD_POS * h, 2 * _STD_POS * h, 1e-2, 2 * _STD_POS * h,
                    10 * _STD_VEL * h, 10 * _STD_VEL * h, 1e-5, 10 * _STD_VEL * h])
    return mean, np.diag(std ** 2)


def _kalman_predict(mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Batched predict: mean [n,8], cov [n,8,8]."""
    h = mean[:, 3]
    std = np.stack([_STD_POS * h, _STD_POS * h, np.full_like(h, 1e-2), _STD_POS * h,
                    _STD_VEL * h, _STD_VEL * h, np.full_like(h, 1e-5), _STD_VEL * h], axis=1)
    q = np.zeros_like(cov)
    idx = np.arange(8)
    q[:, idx, idx] = std ** 2
    mean = mean @ _F.T
    cov = _F @ cov @ _F.T + q
    return mean, cov


def _kalman_update(mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Batched update: mean [n,8], cov [n,8,8], z [n,4] (xyah)."""
    h = mean[:, 3]
    std = np.stack([_STD_POS * h, _STD_POS * h, np.full_like(h, 1e-1), _STD_POS * h], axis=1)
    r = np.zeros((len(mean), 4, 4))
    idx = np.arange(4)
    r[:, idx, idx] = std ** 2
    proj_mean = mean[:, :4]
    proj_cov = cov[:, :4, :4] + r
    pht = cov[:, :, :4]  # cov @ H.T
    # K = P H^T S^-1  ->  solve S K^T = (P H^T)^T
    gain = np.linalg.solve(proj_cov, np.transpose(pht, (0, 2, 1)))
    gain = np.transpose(gain, (0, 2, 1))  # [n,8,4]
    innov = z - proj_mean
    mean = mean + np.einsum("nij,nj->ni", gain, innov)
    cov = cov - gain @ proj_cov @ np.transpose(gain, (0, 2, 1))
    return mean, cov


def _center_dist_heights(tracks_xyxy: np.ndarray, dets_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise center distance divided by the mean box height."""
    a = np.asarray(tracks_xyxy, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(dets_xyxy, dtype=np.float64).reshape(-1, 4)
    ac = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], axis=1)
    bc = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
    ah = np.maximum(a[:, 3] - a[:, 1], 1.0)
    bh = np.maximum(b[:, 3] - b[:, 1], 1.0)
    dist = np.linalg.norm(ac[:, None, :] - bc[None, :, :], axis=2)
    return dist / (0.5 * (ah[:, None] + bh[None, :]))


def _linear_assignment(cost: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian matching keeping only pairs with cost <= thresh."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    from scipy.optimize import linear_sum_assignment

    big = 1e6
    c = np.where(cost <= thresh, cost, big)
    rows, cols = linear_sum_assignment(c)
    matches = [(int(r), int(k)) for r, k in zip(rows, cols) if c[r, k] < big]
    mr = {r for r, _ in matches}
    mc = {k for _, k in matches}
    return matches, [i for i in range(n) if i not in mr], [j for j in range(m) if j not in mc]


class _Track:
    __slots__ = ("track_id", "mean", "cov", "state", "hits", "last_frame", "pending", "score", "app", "app_frame")

    TRACKED, LOST, UNCONFIRMED = 0, 1, 2

    def __init__(self, mean: np.ndarray, cov: np.ndarray, frame: int, score: float) -> None:
        self.track_id = 0
        self.mean = mean
        self.cov = cov
        self.state = _Track.UNCONFIRMED
        self.hits = 1
        self.last_frame = frame
        self.pending: List[Tuple[int, np.ndarray, float]] = []
        self.score = score
        self.app: Optional[np.ndarray] = None
        self.app_frame = -10 ** 9


TrackOutput = Tuple[int, np.ndarray, float, int]  # (track_id, xyxy, conf, det_index)


class ByteTrackLite:
    """Compact ByteTrack (Kalman + Hungarian), per-instance IDs.

    :meth:`update` takes the person detections of one processed frame and
    returns ``(track_id, xyxy, conf, det_index)`` for every confirmed track
    matched in that frame, plus (once) the first detection of tracks that
    were confirmed in this frame (``det_index == -1`` and the box from that
    earlier frame is returned via :attr:`backfill`).
    """

    def __init__(self, config: Optional[TrackerConfig] = None, *, frame_rate: float = 30.0) -> None:
        self.cfg = config or TrackerConfig()
        self.max_lost = max(1, int(round(self.cfg.track_buffer_s * max(1.0, frame_rate))))
        self.tracks: List[_Track] = []
        self.frame_id = 0
        self._next_id = 1
        #: rows confirmed retroactively this update: (frame_id, track_id, xyxy, conf)
        self.backfill: List[Tuple[int, int, np.ndarray, float]] = []

    # ------------------------------------------------------------------
    def _boxes(self, tracks: Sequence[_Track]) -> np.ndarray:
        if not tracks:
            return np.zeros((0, 4))
        return _xyah_to_xyxy(np.stack([t.mean[:4] for t in tracks]))

    def _apply(self, pairs: List[Tuple[_Track, int]], xyxy: np.ndarray, conf: np.ndarray) -> None:
        if not pairs:
            return
        means = np.stack([t.mean for t, _ in pairs])
        covs = np.stack([t.cov for t, _ in pairs])
        z = _xyxy_to_xyah(xyxy[[d for _, d in pairs]])
        means, covs = _kalman_update(means, covs, z)
        for k, (trk, d) in enumerate(pairs):
            trk.mean, trk.cov = means[k], covs[k]
            trk.hits += 1
            trk.last_frame = self.frame_id
            trk.score = float(conf[d])

    def _det_hist(self, cache: Dict[int, Optional[np.ndarray]], frame: Optional[np.ndarray], xyxy: np.ndarray,
                  d: int) -> Optional[np.ndarray]:
        if frame is None:
            return None
        if d not in cache:
            cache[d] = torso_histogram(frame, xyxy[d])
        return cache[d]

    def _appearance_cost(self, tracks: Sequence[_Track], det_idx: np.ndarray, gate: np.ndarray,
                         frame: Optional[np.ndarray], xyxy: np.ndarray,
                         cache: Dict[int, Optional[np.ndarray]]) -> np.ndarray:
        """Extra cost ``w * (1 - similarity)`` for ambiguous gated pairs only."""
        extra = np.zeros(gate.shape)
        if frame is None or self.cfg.appearance_weight <= 0 or not gate.size:
            return extra
        amb = gate & ((gate.sum(axis=1) >= 2)[:, None] | (gate.sum(axis=0) >= 2)[None, :])
        if not amb.any():
            return extra
        for i, j in zip(*np.nonzero(amb)):
            t_app = tracks[i].app
            if t_app is None:
                continue
            d_app = self._det_hist(cache, frame, xyxy, int(det_idx[j]))
            if d_app is None:
                continue
            sim = float(np.sqrt(t_app * d_app).sum())
            extra[i, j] = self.cfg.appearance_weight * (1.0 - sim)
        return extra

    def _update_appearance(self, pairs: Sequence[Tuple[_Track, int]], frame: Optional[np.ndarray],
                           xyxy: np.ndarray, cache: Dict[int, Optional[np.ndarray]]) -> None:
        if frame is None or self.cfg.appearance_weight <= 0 or not pairs:
            return
        due = [(t, d) for t, d in pairs
               if t.app is None or self.frame_id - t.app_frame >= self.cfg.appearance_update_every]
        if not due:
            return
        ious = iou_matrix(xyxy, xyxy)
        np.fill_diagonal(ious, 0.0)
        for trk, d in due:
            if ious[d].max(initial=0.0) > 0.05:  # overlapping players: colour would be mixed
                continue
            hist = self._det_hist(cache, frame, xyxy, d)
            if hist is None:
                continue
            a = self.cfg.appearance_ema
            trk.app = hist.astype(np.float64) if trk.app is None else (1 - a) * trk.app + a * hist
            trk.app_frame = self.frame_id

    def update(self, xyxy: np.ndarray, conf: np.ndarray, frame: Optional[np.ndarray] = None) -> List[TrackOutput]:
        """Advance one processed frame.

        ``frame`` (BGR, same pixel space as ``xyxy``) enables the appearance
        term; without it association is motion-only.
        """
        cfg = self.cfg
        hist_cache: Dict[int, Optional[np.ndarray]] = {}
        self.frame_id += 1
        self.backfill = []
        xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
        conf = np.asarray(conf, dtype=np.float64).reshape(-1)
        wh_ok = ((xyxy[:, 2] - xyxy[:, 0]) > 1e-3) & ((xyxy[:, 3] - xyxy[:, 1]) > 1e-3)
        high = np.flatnonzero(wh_ok & (conf >= cfg.track_high_thresh))
        low = np.flatnonzero(wh_ok & (conf > cfg.track_low_thresh) & (conf < cfg.track_high_thresh))

        # Predict every live track.
        if self.tracks:
            for trk in self.tracks:
                if trk.state == _Track.LOST:
                    trk.mean[4:] *= cfg.lost_velocity_decay
                    trk.mean[6] = 0.0
            means = np.stack([t.mean for t in self.tracks])
            covs = np.stack([t.cov for t in self.tracks])
            means, covs = _kalman_predict(means, covs)
            for k, trk in enumerate(self.tracks):
                trk.mean, trk.cov = means[k], covs[k]

        confirmed = [t for t in self.tracks if t.state != _Track.UNCONFIRMED]
        unconfirmed = [t for t in self.tracks if t.state == _Track.UNCONFIRMED]
        matched: Dict[int, int] = {}  # id(track) -> det index

        # Stage 1: confirmed (tracked + lost) vs high-score detections.
        pool_boxes = self._boxes(confirmed)
        cost = 1.0 - iou_matrix(pool_boxes, xyxy[high]).astype(np.float64)
        if cfg.fuse_score and cost.size:
            cost = 1.0 - (1.0 - cost) * conf[high][None, :]
        if cfg.center_cost_weight > 0 and cost.size:
            # Gate on fused IoU, but rank feasible pairs with a center-distance
            # term: IoU of tiny, jittery boxes is a noisy identity cue.
            gate = cost <= cfg.match_thresh
            cost = cost + cfg.center_cost_weight * _center_dist_heights(pool_boxes, xyxy[high])
            cost = cost + self._appearance_cost(confirmed, high, gate, frame, xyxy, hist_cache)
            cost = np.where(gate, cost, 1e6)
            m1, u_trk, u_det = _linear_assignment(cost, 1e5)
        else:
            m1, u_trk, u_det = _linear_assignment(cost, cfg.match_thresh)
        pairs = [(confirmed[i], int(high[j])) for i, j in m1]
        remaining_high = [int(high[j]) for j in u_det]

        # Stage 2: still-tracked leftovers vs low-score detections (IoU only).
        r_tracked = [confirmed[i] for i in u_trk if confirmed[i].state == _Track.TRACKED]
        r_lost = [confirmed[i] for i in u_trk if confirmed[i].state == _Track.LOST]
        if r_tracked and len(low):
            cost2 = 1.0 - iou_matrix(self._boxes(r_tracked), xyxy[low]).astype(np.float64)
            m2, u2, _ = _linear_assignment(cost2, 1.0 - cfg.low_match_iou)
            pairs += [(r_tracked[i], int(low[j])) for i, j in m2]
            r_tracked = [r_tracked[i] for i in u2]

        # Stage 3: center-distance fallback (scale-free, in box heights) for
        # small/fast players whose predicted and detected boxes do not overlap.
        leftovers = r_tracked + r_lost
        if leftovers and remaining_high and cfg.center_match_heights > 0:
            tb = self._boxes(leftovers)
            db = xyxy[remaining_high]
            tc = np.stack([(tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2], axis=1)
            dc = np.stack([(db[:, 0] + db[:, 2]) / 2, (db[:, 1] + db[:, 3]) / 2], axis=1)
            th = np.maximum(tb[:, 3] - tb[:, 1], 1.0)
            dh = np.maximum(db[:, 3] - db[:, 1], 1.0)
            dist = np.linalg.norm(tc[:, None, :] - dc[None, :, :], axis=2) / th[:, None]
            ratio = np.maximum(th[:, None], dh[None, :]) / np.minimum(th[:, None], dh[None, :])
            dist = np.where(ratio <= 1.5, dist, 1e6)
            m3, u3, u3d = _linear_assignment(dist, cfg.center_match_heights)
            pairs += [(leftovers[i], remaining_high[j]) for i, j in m3]
            leftovers = [leftovers[i] for i in u3]
            remaining_high = [remaining_high[j] for j in u3d]

        for trk, d in pairs:
            matched[id(trk)] = d
        self._apply(pairs, xyxy, conf)
        self._update_appearance(pairs, frame, xyxy, hist_cache)
        for trk, _ in pairs:
            trk.state = _Track.TRACKED

        # Unconfirmed tracks vs remaining high detections.
        new_pairs: List[Tuple[_Track, int]] = []
        if unconfirmed:
            cost_u = 1.0 - iou_matrix(self._boxes(unconfirmed), xyxy[remaining_high]).astype(np.float64)
            if cfg.fuse_score and cost_u.size:
                cost_u = 1.0 - (1.0 - cost_u) * conf[remaining_high][None, :]
            mu, uu, ud = _linear_assignment(cost_u, cfg.unconfirmed_match_thresh)
            new_pairs = [(unconfirmed[i], remaining_high[j]) for i, j in mu]
            dead_unconfirmed = {id(unconfirmed[i]) for i in uu}
            remaining_high = [remaining_high[j] for j in ud]
        else:
            dead_unconfirmed = set()
        self._apply(new_pairs, xyxy, conf)
        self._update_appearance(new_pairs, frame, xyxy, hist_cache)
        outputs: List[TrackOutput] = []
        for trk, d in new_pairs:
            trk.state = _Track.TRACKED
            trk.track_id = self._next_id
            self._next_id += 1
            matched[id(trk)] = d
            for frame_id, box, score in trk.pending:
                self.backfill.append((frame_id, trk.track_id, box, score))
            trk.pending = []

        # Lost bookkeeping.
        survivors: List[_Track] = []
        for trk in self.tracks:
            if id(trk) in dead_unconfirmed:
                continue
            if trk.state == _Track.TRACKED and id(trk) not in matched:
                trk.state = _Track.LOST
            if trk.state == _Track.LOST and self.frame_id - trk.last_frame > self.max_lost:
                continue
            survivors.append(trk)

        # New tracks.
        for d in remaining_high:
            if conf[d] < cfg.new_track_thresh:
                continue
            mean, cov = _kalman_init(_xyxy_to_xyah(xyxy[d:d + 1])[0])
            trk = _Track(mean, cov, self.frame_id, float(conf[d]))
            if self.frame_id == 1:
                trk.state = _Track.TRACKED
                trk.track_id = self._next_id
                self._next_id += 1
                matched[id(trk)] = d
            else:
                trk.pending.append((self.frame_id, xyxy[d].copy(), float(conf[d])))
            survivors.append(trk)
        self.tracks = survivors

        for trk in self.tracks:
            d = matched.get(id(trk))
            if d is not None and trk.state == _Track.TRACKED and trk.track_id:
                outputs.append((trk.track_id, xyxy[d].copy(), float(conf[d]), int(d)))
        return outputs

    @property
    def active_count(self) -> int:
        return sum(1 for t in self.tracks if t.state == _Track.TRACKED)


class _BoxesShim:
    """Minimal ``ultralytics.engine.results.Boxes``-like view (numpy)."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: Optional[np.ndarray] = None) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), np.float32) if cls is None else np.asarray(cls, np.float32)

    @property
    def xywh(self) -> np.ndarray:
        b = self.xyxy
        return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]],
                        axis=1) if len(b) else np.zeros((0, 4), np.float32)

    @property
    def data(self) -> np.ndarray:
        return np.column_stack([self.xyxy, self.conf, self.cls]) if len(self.conf) else np.zeros((0, 6), np.float32)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, idx) -> "_BoxesShim":
        return _BoxesShim(self.xyxy[idx], self.conf[idx], self.cls[idx])

    def cpu(self) -> "_BoxesShim":
        return self

    def numpy(self) -> "_BoxesShim":
        return self


class UltralyticsTrackerAdapter:
    """Wrap ``ultralytics.trackers.BOTSORT``/``BYTETracker`` behind :meth:`update`.

    Ultralytics keeps a process-global ID counter, so IDs are remapped to a
    per-instance sequence.
    """

    def __init__(self, kind: str, config: Optional[TrackerConfig] = None, *, frame_rate: float = 30.0) -> None:
        import importlib.util
        import sys

        # ultralytics' matching module pip-installs ``lap`` at import time when
        # it is missing; never trigger a network install from the pipeline.
        if "lap" not in sys.modules and importlib.util.find_spec("lap") is None:
            raise ImportError("ultralytics trackers need the 'lap' package (pip install lap)")
        from ultralytics.trackers import BOTSORT, BYTETracker

        cfg = config or TrackerConfig()
        args = SimpleNamespace(
            tracker_type=kind,
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            track_buffer=max(1, int(round(cfg.track_buffer_s * frame_rate))),
            match_thresh=cfg.match_thresh,
            fuse_score=cfg.fuse_score,
            gmc_method=cfg.gmc_method,
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model="auto",
            device="cpu",
        )
        cls = BOTSORT if kind == "botsort" else BYTETracker
        try:  # ultralytics < 8.4 takes frame_rate
            self._tracker = cls(args, frame_rate=int(round(frame_rate)))
        except TypeError:
            self._tracker = cls(args)
        self.kind = kind
        self._ids: Dict[int, int] = {}
        self._next = 1
        self.backfill: List[Tuple[int, int, np.ndarray, float]] = []

    def update(self, xyxy: np.ndarray, conf: np.ndarray, frame: Optional[np.ndarray] = None) -> List[TrackOutput]:
        self.backfill = []
        res = self._tracker.update(_BoxesShim(xyxy, conf), frame)
        out: List[TrackOutput] = []
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        for row in np.asarray(res).reshape(-1, 8) if len(res) else []:
            raw_id = int(row[4])
            det = int(row[7])
            if raw_id not in self._ids:
                self._ids[raw_id] = self._next
                self._next += 1
            box = xyxy[det] if 0 <= det < len(xyxy) else row[:4]
            out.append((self._ids[raw_id], np.asarray(box, dtype=np.float64), float(row[5]), det))
        return out


def make_tracker(kind: str, config: Optional[TrackerConfig], frame_rate: float):
    """``bytetrack`` -> :class:`ByteTrackLite`; ``botsort`` -> ultralytics BOT-SORT (fallback ByteTrackLite)."""
    kind = (kind or "bytetrack").strip().lower().replace(".yaml", "")
    if kind == "botsort":
        try:
            return UltralyticsTrackerAdapter("botsort", config, frame_rate=frame_rate), "botsort"
        except Exception as exc:
            logger.warning("BoT-SORT unavailable (%s); using ByteTrackLite", exc)
    elif kind not in ("bytetrack", "bytetrack_lite", "native"):
        logger.warning("Unknown tracker %r; using ByteTrackLite", kind)
    return ByteTrackLite(config, frame_rate=frame_rate), "bytetrack"


# ----------------------------------------------------------------------
# Appearance
# ----------------------------------------------------------------------

APPEARANCE_BINS = (8, 4, 4)


def torso_histogram(frame: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    """L1-normalised HSV histogram (8x4x4) of the jersey region of ``box``.

    The torso is taken as 15-50 % of the box height and the central 60 % of
    its width, which excludes most grass, shorts and the head.
    """
    import cv2

    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    bw, bh = x2 - x1, y2 - y1
    if bw < 3 or bh < 6:
        return None
    px1 = int(max(0, math.floor(x1 + 0.2 * bw)))
    px2 = int(min(w_img, math.ceil(x2 - 0.2 * bw)))
    py1 = int(max(0, math.floor(y1 + 0.15 * bh)))
    py2 = int(min(h_img, math.ceil(y1 + 0.5 * bh)))
    if px2 - px1 < 2 or py2 - py1 < 2:
        return None
    patch = frame[py1:py2, px1:px2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(APPEARANCE_BINS), [0, 180, 0, 256, 0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0:
        return None
    return (hist / total).astype(np.float32)


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------


@dataclass
class _EngineState:
    """Everything the tracking worker thread mutates."""

    tracker: object
    fps: float
    max_players: int
    appearance_every: int
    players: RowBuffer = field(default_factory=lambda: RowBuffer(_PLAYER_COLS, 1 << 15))
    ball: RowBuffer = field(default_factory=lambda: RowBuffer(_BALL_COLS, 1 << 12))
    app_sum: Dict[int, np.ndarray] = field(default_factory=dict)
    app_n: Dict[int, int] = field(default_factory=dict)
    app_last: Dict[int, int] = field(default_factory=dict)
    app_head: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    app_tail: Dict[int, Deque[np.ndarray]] = field(default_factory=dict)
    app_window: int = 3
    frames_processed: int = 0
    last_frame_index: int = -1
    track_s: float = 0.0
    frame_of_tracker_step: Dict[int, int] = field(default_factory=dict)

    def process_batch(self, items: Sequence[FrameItem], dets: Sequence[Detections]) -> None:
        t0 = time.perf_counter()
        for item, det in zip(items, dets):
            self._process_frame(item, det)
        self.track_s += time.perf_counter() - t0

    def _process_frame(self, item: FrameItem, det: Detections) -> None:
        fi = int(item.index)
        balls = det.of_class(BALL)
        if len(balls):
            b = balls.xyxy
            rows = np.column_stack([
                np.full(len(b), fi, np.float32), (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                b[:, 2] - b[:, 0], b[:, 3] - b[:, 1], balls.conf,
            ])
            self.ball.append_rows(rows)
        persons = det.of_class(PERSON)
        if len(persons) > self.max_players > 0:
            keep = np.argsort(-persons.conf, kind="stable")[: self.max_players]
            persons = persons.select(np.sort(keep))
        outputs = self.tracker.update(persons.xyxy, persons.conf, item.frame)
        step = getattr(self.tracker, "frame_id", None)
        if step is not None:
            self.frame_of_tracker_step[int(step)] = fi
        rows: List[List[float]] = []
        for frame_step, tid, box, score in getattr(self.tracker, "backfill", []) or []:
            prev_fi = self.frame_of_tracker_step.get(int(frame_step), fi)
            rows.append([tid, prev_fi, box[0], box[1], box[2], box[3], score])
        if step is not None and len(self.frame_of_tracker_step) > 8:
            for old in [k for k in self.frame_of_tracker_step if k < int(step) - 4]:
                self.frame_of_tracker_step.pop(old, None)
        for tid, box, score, _d in outputs:
            rows.append([tid, fi, box[0], box[1], box[2], box[3], score])
        if rows:
            self.players.append_rows(np.asarray(rows, dtype=np.float32))
        if outputs:
            self._sample_appearance(item, outputs)
        self.frames_processed += 1
        self.last_frame_index = fi

    def _sample_appearance(self, item: FrameItem, outputs: Sequence[TrackOutput]) -> None:
        due = [k for k, (tid, _b, _s, _d) in enumerate(outputs)
               if item.index - self.app_last.get(tid, -10 ** 9) >= self.appearance_every]
        if not due:
            return
        boxes = np.stack([np.asarray(o[1], dtype=np.float32) for o in outputs])
        ious = iou_matrix(boxes, boxes)
        np.fill_diagonal(ious, 0.0)
        for k in due:
            if ious[k].max(initial=0.0) > 0.05:  # occluded: try again next frame
                continue
            tid = outputs[k][0]
            hist = torso_histogram(item.frame, outputs[k][1])
            if hist is None:
                continue
            if tid in self.app_sum:
                self.app_sum[tid] += hist
            else:
                self.app_sum[tid] = hist.copy()
            self.app_n[tid] = self.app_n.get(tid, 0) + 1
            self.app_last[tid] = int(item.index)
            head = self.app_head.setdefault(tid, [])
            if len(head) < self.app_window:
                head.append(hist)
            tail = self.app_tail.get(tid)
            if tail is None:
                tail = self.app_tail[tid] = collections.deque(maxlen=self.app_window)
            tail.append(hist)


def _emit(cb: Optional[ProgressCallback], fraction: float, message: str, data: Dict[str, object]) -> None:
    if cb is None:
        return
    try:
        cb("tracking", float(min(1.0, max(0.0, fraction))), message, data)
    except Exception:  # pragma: no cover - callback bugs must not stop tracking
        logger.exception("progress callback failed")


def _velocity(t: np.ndarray, cx: np.ndarray, cy: np.ndarray, *, head: bool, window_s: float = 0.5) -> Tuple[float, float]:
    if len(t) < 2:
        return 0.0, 0.0
    if head:
        j = int(np.searchsorted(t, t[0] + window_s, side="right")) - 1
        i, j = 0, max(1, j)
    else:
        i = int(np.searchsorted(t, t[-1] - window_s, side="left"))
        i, j = min(i, len(t) - 2), len(t) - 1
    dt = float(t[j] - t[i])
    if dt <= 0:
        return 0.0, 0.0
    return float((cx[j] - cx[i]) / dt), float((cy[j] - cy[i]) / dt)


def _resolve_proxy(proxy: Union[ProxyResult, str, Path]) -> Tuple[str, Optional[float], Optional[Tuple[int, int]], float]:
    if isinstance(proxy, ProxyResult):
        src = (proxy.source_width or proxy.width, proxy.source_height or proxy.height)
        return proxy.path, proxy.fps, (int(src[0]), int(src[1])), float(proxy.trim_start_s)
    return str(proxy), None, None, 0.0


def track_video(
    proxy: Union[ProxyResult, str, Path],
    *,
    detector: Detector,
    source_frame_size: Optional[Tuple[int, int]] = None,
    fps: Optional[float] = None,
    stride: int = 1,
    tracker: str = "bytetrack",
    batch_size: Optional[int] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
    focus_roi: Optional[Mapping[str, object]] = None,
    focus_track_id: Optional[int] = None,
    max_players: int = 30,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    trim_offset_s: Optional[float] = None,
    source_video_path: Optional[str] = None,
    tracker_config: Optional[TrackerConfig] = None,
    min_track_s: float = 1.0,
    appearance_interval_s: float = 1.0,
    stitch: bool = True,
    stitch_kwargs: Optional[Dict[str, object]] = None,
    prefetch_frames: Optional[int] = None,
) -> TrackingResult:
    """Detect and track every player (and every raw ball detection) in ``proxy``.

    Args:
        proxy: :class:`ProxyResult` from ``build_proxy`` or a video path.
        detector: any :class:`~backend.services.detectors.Detector`.
        source_frame_size: ``(W, H)`` of the original video; output
            coordinates are scaled to it (default: from ``ProxyResult``, else
            the proxy's own size).
        fps: proxy frame rate (default: from ``ProxyResult``/container).
            ``t = frame_index / fps``.
        stride: process every ``stride``-th frame.
        tracker: ``"bytetrack"`` (built-in) or ``"botsort"`` (ultralytics).
        batch_size: frames per ``detect_batch`` (default ``detector.batch``).
        progress_cb: ``(stage, fraction, message, data)``; ``data`` has
            ``frames_done, frames_total, fps_processing, eta_s, device,
            batch, active_tracks``.
        cancel_event: checked every batch; on cancel the partial result is
            returned with ``timings["cancelled"] = True``.
        focus_roi: user box (``{x1_norm..y2_norm}`` or ``{x, y, w, h}`` with
            ``normalized``; pixel values are SOURCE pixels) plus optional
            ``t`` (seconds, default 0) at which it was drawn.
        focus_track_id: explicit follow target (raw or stitched id).
        max_players: per-frame cap on person detections fed to the tracker
            (highest confidence first).
        start_frame / end_frame: proxy frame window (end exclusive).
        trim_offset_s: proxy t=0 in source seconds (default from ProxyResult).
        min_track_s: identities shorter than this are dropped.
        appearance_interval_s: per-track torso histogram sampling period.
        stitch: link fragments into identities (motion + appearance).
    """
    started = time.monotonic()
    path, proxy_fps, proxy_src_size, proxy_trim = _resolve_proxy(proxy)
    stride = max(1, int(stride))
    batch = max(1, int(batch_size or getattr(detector, "batch", 8) or 8))
    reader = FrameReader(path, stride=stride, start_frame=start_frame, end_frame=end_frame,
                         prefetch=prefetch_frames or max(16, 2 * batch), cancel_event=cancel_event,
                         fps=fps or proxy_fps)
    fps_v = float(reader.fps)
    proxy_w, proxy_h = reader.width, reader.height
    src_w, src_h = source_frame_size or proxy_src_size or (proxy_w, proxy_h)
    sx = float(src_w) / float(proxy_w or src_w)
    sy = float(src_h) / float(proxy_h or src_h)
    frames_total = reader.expected_frames
    det_info = dict(detector.info()) if hasattr(detector, "info") else {"name": type(detector).__name__}
    device_label = str(det_info.get("device", "cpu"))

    trk, tracker_name = make_tracker(tracker, tracker_config, fps_v / stride)
    state = _EngineState(
        tracker=trk,
        fps=fps_v,
        max_players=int(max_players),
        appearance_every=max(1, int(round(appearance_interval_s * fps_v))),
    )
    logger.info("Tracking %s: %dx%d @ %.3f fps, stride=%d, batch=%d, tracker=%s, detector=%s",
                path, proxy_w, proxy_h, fps_v, stride, batch, tracker_name, det_info.get("name"))
    _emit(progress_cb, 0.0, "Tracking started", {
        "frames_done": 0, "frames_total": frames_total, "fps_processing": 0.0, "eta_s": None,
        "device": device_label, "batch": batch, "tracker": tracker_name,
    })

    detect_s = 0.0
    frames_done = 0
    cancelled = False
    last_emit = time.monotonic()
    loop_started = time.monotonic()
    pending: Deque[Future] = collections.deque()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracker")
    try:
        reader.start()
        for items in reader.batches(batch):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            t0 = time.perf_counter()
            dets = detector.detect_batch([it.frame for it in items], frame_indices=[it.index for it in items])
            detect_s += time.perf_counter() - t0
            if len(dets) != len(items):
                raise RuntimeError(f"detector returned {len(dets)} results for {len(items)} frames")
            pending.append(executor.submit(state.process_batch, items, dets))
            while len(pending) > 2:
                pending.popleft().result()
            frames_done += len(items)
            now = time.monotonic()
            if progress_cb is not None and now - last_emit >= 1.0:
                elapsed = now - loop_started
                rate = frames_done / elapsed if elapsed > 0 else 0.0
                remaining = max(0, frames_total - frames_done)
                _emit(progress_cb, frames_done / frames_total if frames_total else 0.0, "Tracking frames", {
                    "frames_done": frames_done, "frames_total": frames_total,
                    "fps_processing": round(rate, 2), "eta_s": round(remaining / rate, 1) if rate > 0 else None,
                    "device": device_label, "batch": batch,
                    "active_tracks": getattr(trk, "active_count", None),
                })
                last_emit = now
        while pending:
            pending.popleft().result()
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
    finally:
        for fut in pending:
            fut.cancel()
        executor.shutdown(wait=True)
        reader.close()

    loop_s = time.monotonic() - loop_started
    t_post = time.perf_counter()
    result, stats = _build_result(
        state, fps=fps_v, sx=sx, sy=sy, src_size=(int(src_w), int(src_h)), stride=stride,
        min_track_s=min_track_s, stitch=stitch, stitch_kwargs=stitch_kwargs or {},
    )
    window_end = state.last_frame_index + 1 if cancelled else max(
        state.last_frame_index + 1, end_frame if end_frame is not None else reader.frame_count)
    result.duration_s = max(0, window_end - max(0, int(start_frame))) / fps_v
    result.trim_offset_s = float(trim_offset_s if trim_offset_s is not None else proxy_trim)
    result.proxy_scale = 1.0 / sx if sx else 1.0
    result.source_video_path = source_video_path
    result.processing_video_path = path

    selection = _select_focus(result, focus_roi=focus_roi, focus_track_id=focus_track_id, stride=stride)
    post_s = time.perf_counter() - t_post
    total_s = time.monotonic() - started
    result.timings = {
        "decode_s": round(reader.decode_s, 4),
        "detect_s": round(detect_s, 4),
        "track_s": round(state.track_s, 4),
        "postprocess_s": round(post_s, 4),
        "loop_s": round(loop_s, 4),
        "total_s": round(total_s, 4),
        "frames": float(state.frames_processed),
        "fps_processing": round(state.frames_processed / loop_s, 3) if loop_s > 0 else 0.0,
        "cancelled": cancelled,  # type: ignore[dict-item]
    }
    result.detector = {
        **det_info,
        "tracker": tracker_name,
        "stride": stride,
        "batch": batch,
        "max_players": int(max_players),
        "proxy_size": [int(proxy_w), int(proxy_h)],
        "tracking_stats": {**stats, "player_rows_bytes": state.players.nbytes},
        "focus_selection": selection,
    }
    _emit(progress_cb, 1.0 if not cancelled else (frames_done / frames_total if frames_total else 0.0),
          "Tracking cancelled" if cancelled else "Tracking complete", {
              "frames_done": frames_done, "frames_total": frames_total,
              "fps_processing": result.timings["fps_processing"], "eta_s": 0.0, "device": device_label,
              "batch": batch, "player_tracks": len(result.players), "ball_detections": len(result.ball),
              "cancelled": cancelled,
          })
    logger.info("Tracking %s: %d frames, %d identities (%d raw tracks), %d ball detections in %.1fs "
                "(detect %.1fs, track %.1fs, decode %.1fs)",
                "cancelled" if cancelled else "done", state.frames_processed, len(result.players),
                stats.get("raw_tracks", 0), len(result.ball), total_s, detect_s, state.track_s, reader.decode_s)
    return result


def _build_result(
    state: _EngineState,
    *,
    fps: float,
    sx: float,
    sy: float,
    src_size: Tuple[int, int],
    stride: int,
    min_track_s: float,
    stitch: bool,
    stitch_kwargs: Dict[str, object],
) -> Tuple[TrackingResult, Dict[str, object]]:
    table = state.players.array()
    stats: Dict[str, object] = {"raw_tracks": 0, "dropped_tiny_fragments": 0, "stitch_links": 0,
                                "dropped_short_tracks": 0, "identities": 0}
    raw: Dict[int, np.ndarray] = {}
    if len(table):
        order = np.lexsort((table[:, 1], table[:, 0]))
        table = table[order]
        ids = table[:, 0].astype(np.int64)
        bounds = np.flatnonzero(np.diff(ids)) + 1
        for chunk in np.split(table, bounds):
            raw[int(chunk[0, 0])] = chunk
    stats["raw_tracks"] = len(raw)

    min_frag_samples = 3
    fragments: List[TrackFragment] = []
    for tid, chunk in raw.items():
        if len(chunk) < min_frag_samples:
            stats["dropped_tiny_fragments"] = int(stats["dropped_tiny_fragments"]) + 1
            continue
        t = chunk[:, 1].astype(np.float64) / fps
        cx = (chunk[:, 2] + chunk[:, 4]) * 0.5
        cy = (chunk[:, 3] + chunk[:, 5]) * 0.5
        app = None
        if tid in state.app_sum and state.app_n.get(tid, 0) > 0:
            app = state.app_sum[tid] / float(state.app_n[tid])
        fragments.append(TrackFragment(
            track_id=tid, t_start=float(t[0]), t_end=float(t[-1]),
            start_box=tuple(float(v) for v in chunk[0, 2:6]),  # type: ignore[arg-type]
            end_box=tuple(float(v) for v in chunk[-1, 2:6]),  # type: ignore[arg-type]
            start_vel=_velocity(t, cx, cy, head=True), end_vel=_velocity(t, cx, cy, head=False),
            appearance=app, samples=len(chunk),
            start_appearance=np.mean(state.app_head[tid], axis=0) if state.app_head.get(tid) else None,
            end_appearance=np.mean(list(state.app_tail[tid]), axis=0) if state.app_tail.get(tid) else None,
        ))

    if stitch and fragments:
        chains = stitch_tracks(fragments, **stitch_kwargs)  # type: ignore[arg-type]
    else:
        chains = [[f.track_id] for f in sorted(fragments, key=lambda f: (f.t_start, f.track_id))]
    stats["stitch_links"] = sum(len(c) - 1 for c in chains)

    players: Dict[int, PlayerTrack] = {}
    for chain in chains:
        block = np.concatenate([raw[i] for i in chain]) if len(chain) > 1 else raw[chain[0]]
        if len(chain) > 1:
            block = block[np.argsort(block[:, 1], kind="stable")]
            _, first = np.unique(block[:, 1], return_index=True)
            block = block[np.sort(first)]
        t = (block[:, 1].astype(np.float64) / fps).astype(np.float32)
        if len(t) < 2 or float(t[-1] - t[0]) < min_track_s:
            stats["dropped_short_tracks"] = int(stats["dropped_short_tracks"]) + 1
            continue
        identity = int(chain[0])
        players[identity] = PlayerTrack(
            track_id=identity,
            t=t,
            x1=(block[:, 2] * sx).astype(np.float32),
            y1=(block[:, 3] * sy).astype(np.float32),
            x2=(block[:, 4] * sx).astype(np.float32),
            y2=(block[:, 5] * sy).astype(np.float32),
            conf=block[:, 6].astype(np.float32),
            source_track_ids=[int(c) for c in chain],
        )
    stats["identities"] = len(players)

    btab = state.ball.array()
    if len(btab):
        btab = btab[np.argsort(btab[:, 0], kind="stable")]
        ball = BallDetections(
            t=(btab[:, 0].astype(np.float64) / fps).astype(np.float32),
            x=(btab[:, 1] * sx).astype(np.float32),
            y=(btab[:, 2] * sy).astype(np.float32),
            w=(btab[:, 3] * sx).astype(np.float32),
            h=(btab[:, 4] * sy).astype(np.float32),
            conf=btab[:, 5].astype(np.float32),
        )
    else:
        ball = BallDetections.empty()
    result = TrackingResult(fps=fps, frame_size=src_size, duration_s=0.0, players=players, ball=ball,
                            vid_stride=stride)
    return result, stats


def select_focus(
    result: TrackingResult,
    *,
    focus_roi: Optional[Mapping[str, object]] = None,
    focus_track_id: Optional[int] = None,
    stride: Optional[int] = None,
) -> Dict[str, object]:
    """Choose the focus player of a (fresh or loaded) :class:`TrackingResult`.

    Sets ``result.focus_track_id`` in place and returns how it was chosen as
    ``{"method": ..., "track_id": ...}`` (plus details per method):

    * ``focus_track_id`` given: that identity (``"track_id"``), or the
      stitched identity whose ``source_track_ids`` contain it
      (``"track_id_stitched"``). An unknown id falls through to the ROI.
    * ``focus_roi`` given (``x1_norm``/``y1_norm``/... or pixel box, plus
      optional ``t``/``time_s``): the track under the box at that time
      (``"roi"``), else the longest track (``"fallback_longest"``).
    * neither: no focus (``"none"``, ``focus_track_id`` reset to None).

    ``stride`` (frames between processed samples) defaults to the result's
    ``vid_stride``; it only widens the ROI matching window. Use this to
    re-target the focus player on reused tracks without re-detecting.
    """
    if stride is None:
        stride = int(getattr(result, "vid_stride", 1) or 1)
    return _select_focus(result, focus_roi=focus_roi, focus_track_id=focus_track_id, stride=max(1, int(stride)))


def _select_focus(
    result: TrackingResult,
    *,
    focus_roi: Optional[Mapping[str, object]],
    focus_track_id: Optional[int],
    stride: int,
) -> Dict[str, object]:
    """Set ``result.focus_track_id``; returns how it was chosen."""
    if focus_track_id is not None:
        fid = int(focus_track_id)
        if fid in result.players:
            result.focus_track_id = fid
            return {"method": "track_id", "requested": fid, "track_id": fid}
        for pid, track in result.players.items():
            if fid in track.source_track_ids:
                result.focus_track_id = int(pid)
                return {"method": "track_id_stitched", "requested": fid, "track_id": int(pid)}
        logger.warning("focus_track_id %s not found among %d identities", fid, len(result.players))
    if focus_roi:
        box = resolve_player_roi_box(dict(focus_roi), result.frame_width, result.frame_height)
        if box is not None:
            try:
                t_roi = float(focus_roi.get("t", focus_roi.get("time_s", 0.0)) or 0.0)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                t_roi = 0.0
            frame_dt = stride / max(result.fps, 1e-6)
            for window in (max(0.5, 2 * frame_dt), 3.0):
                chosen = select_track_at(result.players, box, t_roi, window_s=window)
                if chosen is not None:
                    result.focus_track_id = int(chosen)
                    return {"method": "roi", "t": t_roi, "window_s": window, "box": [round(v, 1) for v in box],
                            "track_id": int(chosen)}
            logger.warning("No track matched the focus ROI at t=%.2fs; using the longest track", t_roi)
        longest = result.longest_track_id()
        result.focus_track_id = longest
        return {"method": "fallback_longest", "track_id": longest}
    result.focus_track_id = None
    return {"method": "none"}


# ----------------------------------------------------------------------
# Legacy adapters (old VideoHighlights.track_video shapes)
# ----------------------------------------------------------------------


@dataclass
class TrackPoint:
    """Field-compatible with ``VideoHighlights.TrackPoint``."""

    t: float
    xy: Tuple[float, float]
    bbox: Optional[Tuple[float, float, float, float]] = None


def track_points(track: PlayerTrack) -> List[TrackPoint]:
    return [
        TrackPoint(t=float(t), xy=(float((a + c) * 0.5), float((b + d) * 0.5)),
                   bbox=(float(a), float(b), float(c), float(d)))
        for t, a, b, c, d in zip(track.t, track.x1, track.y1, track.x2, track.y2)
    ]


def legacy_views(
    result: TrackingResult,
    target_track_id: Optional[int] = None,
    *,
    positions_max_hz: float = 10.0,
) -> Tuple[Dict[int, List[TrackPoint]], List[TrackPoint], np.ndarray]:
    """Shapes the old ``VideoHighlights.track_video`` produced.

    Returns ``({target_id: [TrackPoint]}, ball_trajectory, player_positions)``
    where ``target_id`` is ``target_track_id``, else ``result.focus_track_id``,
    else the longest identity; ``ball_trajectory`` holds every raw ball
    detection; ``player_positions`` is ``(t, cx, cy)`` float32 at <=10 Hz.
    """
    tid = target_track_id if target_track_id is not None else result.focus_track_id
    if tid is None or int(tid) not in result.players:
        tid = result.longest_track_id()
    tracks: Dict[int, List[TrackPoint]] = {}
    if tid is not None:
        tracks[int(tid)] = track_points(result.players[int(tid)])
    b = result.ball
    ball = [
        TrackPoint(t=float(t), xy=(float(x), float(y)),
                   bbox=(float(x - w / 2), float(y - h / 2), float(x + w / 2), float(y + h / 2)))
        for t, x, y, w, h in zip(b.t, b.x, b.y, b.w, b.h)
    ]
    return tracks, ball, result.all_player_positions(max_hz=positions_max_hz)


def legacy_track_video_tuple(
    result: TrackingResult,
    target_track_id: Optional[int] = None,
) -> Tuple[Dict[int, List[TrackPoint]], List[TrackPoint], float, Tuple[int, int], Dict[str, object]]:
    """Exact 5-tuple returned by the old ``VideoHighlights.track_video``."""
    tracks, ball, positions = legacy_views(result, target_track_id)
    if not tracks:
        raise RuntimeError("No player tracks detected in video. Ensure the video contains visible people.")
    tid = next(iter(tracks))
    stitched = list(result.players[tid].source_track_ids) or [tid]
    return (
        tracks,
        ball,
        float(result.fps),
        (result.frame_width, result.frame_height),
        {
            "target_track_id": int(tid),
            "stitched_track_ids": stitched,
            "stitched_track_count": len(stitched),
            "player_positions": positions,
        },
    )


# ----------------------------------------------------------------------
# Evaluation against ground truth (tests / bench)
# ----------------------------------------------------------------------


def identity_metrics(
    result: TrackingResult,
    truth: TrackingResult,
    *,
    iou_thresh: float = 0.5,
    min_coverage: float = 0.6,
    max_switches: int = 1,
    ambiguous_iou: float = 0.5,
) -> Dict[str, object]:
    """CLEAR-MOT style identity evaluation against ground truth.

    Frames where a truth box overlaps another truth box with IoU >=
    ``ambiguous_iou`` (players passing through each other) are skipped for
    that player: identity there is undefined even for a perfect tracker.

    At every predicted sample time, truth boxes and predicted boxes are
    matched one-to-one (Hungarian on IoU, pairs below ``iou_thresh``
    rejected). Per truth player we record the sequence of matched predicted
    identities: ``switches`` counts changes in that sequence and
    ``coverage`` is the share of evaluated frames matched to the dominant
    identity. A truth player is *recovered* when ``coverage >= min_coverage``
    and ``switches <= max_switches``.
    """
    pred_by_t: Dict[int, Tuple[List[int], List[np.ndarray]]] = collections.defaultdict(lambda: ([], []))
    for pid, tr in result.players.items():
        keys = np.round(tr.t.astype(np.float64) * 1000).astype(np.int64)
        boxes = np.stack([tr.x1, tr.y1, tr.x2, tr.y2], axis=1)
        for k, box in zip(keys, boxes):
            entry = pred_by_t[int(k)]
            entry[0].append(int(pid))
            entry[1].append(box)
    truth_by_t: Dict[int, Tuple[List[int], List[np.ndarray]]] = collections.defaultdict(lambda: ([], []))
    for gid, gt in truth.players.items():
        keys = np.round(gt.t.astype(np.float64) * 1000).astype(np.int64)
        boxes = np.stack([gt.x1, gt.y1, gt.x2, gt.y2], axis=1)
        for k, box in zip(keys, boxes):
            entry = truth_by_t[int(k)]
            entry[0].append(int(gid))
            entry[1].append(box)

    seqs: Dict[int, List[int]] = {int(g): [] for g in truth.players}
    evaluated: Dict[int, int] = {int(g): 0 for g in truth.players}
    for key in sorted(pred_by_t):
        if key not in truth_by_t:
            continue
        gids, gboxes = truth_by_t[key]
        pids, pboxes = pred_by_t[key]
        g_arr = np.stack(gboxes)
        self_iou = iou_matrix(g_arr, g_arr)
        np.fill_diagonal(self_iou, 0.0)
        ambiguous = self_iou.max(axis=1) >= ambiguous_iou
        for gi, g in enumerate(gids):
            if not ambiguous[gi]:
                evaluated[g] += 1
        ious = iou_matrix(g_arr, np.stack(pboxes)).astype(np.float64)
        matches, _, _ = _linear_assignment(1.0 - ious, 1.0 - iou_thresh)
        for gi, pj in matches:
            if not ambiguous[gi]:
                seqs[gids[gi]].append(pids[pj])

    per_player: Dict[int, Dict[str, object]] = {}
    for gid, seq in seqs.items():
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        dominant, dom_count = (collections.Counter(seq).most_common(1)[0] if seq else (None, 0))
        n_eval = evaluated[gid]
        coverage = dom_count / n_eval if n_eval else 0.0
        per_player[gid] = {
            "dominant_id": dominant, "coverage": round(coverage, 3), "switches": switches,
            "matched_frames": len(seq), "evaluated_frames": n_eval,
            "recovered": bool(coverage >= min_coverage and switches <= max_switches),
        }
    recovered = sum(1 for v in per_player.values() if v["recovered"])
    return {
        "recovered": recovered,
        "total": len(per_player),
        "recovered_fraction": recovered / len(per_player) if per_player else 0.0,
        "max_switches": max((int(v["switches"]) for v in per_player.values()), default=0),
        "players": per_player,
    }
