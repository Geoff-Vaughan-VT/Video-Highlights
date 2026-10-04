"""Shared tracking data contracts.

Every analysis stage (camera planning, stats, events, UI) consumes the same
``TrackingResult`` so a match is **tracked once and analyzed many times**:
re-running stats, re-planning the camera, or rendering a different player's
follow-cam never requires another detection pass over the source video.

Coordinate conventions
----------------------
* All coordinates are in **source-video pixel space** (the original frame
  size), even when detection ran on a downscaled proxy. Producers must
  rescale before building these objects.
* ``t`` is seconds from the start of the *processing* video (the trimmed
  working window). ``TrackingResult.trim_offset_s`` converts to the original
  file's timebase: ``t_source = t + trim_offset_s``.
* A player's ground position is the **foot point**: bottom-center of the
  bounding box ``((x1 + x2) / 2, y2)``. Use :meth:`PlayerTrack.foot_xy` for
  anything that maps to the pitch (speed, distance, heatmaps, possession).
  Box centers are kept for camera framing only.

Persistence
-----------
``TrackingResult.save(dir)`` writes ``tracks.npz`` (compact float32 arrays,
all players + ball) and ``TrackingResult.load(dir)`` restores it. The
human-readable ``analysis_tracking.json`` manifest written by the pipeline
remains the summary document; ``tracks.npz`` is the full data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TRACKS_FILENAME = "tracks.npz"
TRACKS_META_FILENAME = "tracks_meta.json"

TEAM_UNKNOWN = -1
TEAM_A = 0  # "left"/home team as configured by the user
TEAM_B = 1  # "right"/away team
TEAM_REFEREE = 2


@dataclass
class PlayerTrack:
    """One tracker identity (possibly stitched across several raw IDs).

    Arrays are aligned and sorted by ``t``. ``conf`` may be all-ones when the
    producer does not expose detector confidence.
    """

    track_id: int
    t: np.ndarray  # float32 [n]
    x1: np.ndarray  # float32 [n]
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    conf: np.ndarray  # float32 [n]
    team: int = TEAM_UNKNOWN
    team_confidence: float = 0.0
    jersey_color_hex: Optional[str] = None
    jersey_number: Optional[int] = None
    label: Optional[str] = None  # user-provided name/label
    source_track_ids: List[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def start_s(self) -> float:
        return float(self.t[0]) if len(self) else 0.0

    @property
    def end_s(self) -> float:
        return float(self.t[-1]) if len(self) else 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def center_xy(self) -> np.ndarray:
        """Box centers, shape [n, 2]."""
        return np.stack([(self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5], axis=1)

    def foot_xy(self) -> np.ndarray:
        """Ground contact points (bottom-center of box), shape [n, 2]."""
        return np.stack([(self.x1 + self.x2) * 0.5, self.y2], axis=1)

    def box_heights(self) -> np.ndarray:
        return self.y2 - self.y1

    def index_at(self, t: float, tolerance_s: float = 0.25) -> Optional[int]:
        """Index of the sample nearest ``t`` within ``tolerance_s``."""
        if not len(self):
            return None
        idx = int(np.searchsorted(self.t, t))
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(self)]
        if not candidates:
            return None
        best = min(candidates, key=lambda i: abs(float(self.t[i]) - t))
        if abs(float(self.t[best]) - t) > tolerance_s:
            return None
        return best

    def box_at(self, t: float, tolerance_s: float = 0.25) -> Optional[Tuple[float, float, float, float]]:
        idx = self.index_at(t, tolerance_s)
        if idx is None:
            return None
        return float(self.x1[idx]), float(self.y1[idx]), float(self.x2[idx]), float(self.y2[idx])

    def foot_at(self, t: float, tolerance_s: float = 0.25) -> Optional[Tuple[float, float]]:
        idx = self.index_at(t, tolerance_s)
        if idx is None:
            return None
        return float((self.x1[idx] + self.x2[idx]) * 0.5), float(self.y2[idx])

    def to_samples(self) -> List[Tuple[float, float, float]]:
        """Legacy ``(t, cx, cy)`` tuples used by the follow-cam renderer."""
        centers = self.center_xy()
        return [(float(tt), float(c[0]), float(c[1])) for tt, c in zip(self.t, centers)]

    def summary(self) -> Dict[str, object]:
        return {
            "track_id": int(self.track_id),
            "samples": len(self),
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "team": int(self.team),
            "team_confidence": round(float(self.team_confidence), 3),
            "jersey_color_hex": self.jersey_color_hex,
            "jersey_number": self.jersey_number,
            "label": self.label,
            "source_track_ids": [int(v) for v in self.source_track_ids],
        }


@dataclass
class BallDetections:
    """Raw (unfiltered) ball detections; feed to ``game_tracking.build_ball_track``."""

    t: np.ndarray  # float32 [n]
    x: np.ndarray  # center x
    y: np.ndarray  # center y
    w: np.ndarray  # box width
    h: np.ndarray
    conf: np.ndarray

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def to_samples(self) -> List[Tuple[float, float, float]]:
        return [(float(a), float(b), float(c)) for a, b, c in zip(self.t, self.x, self.y)]

    @staticmethod
    def empty() -> "BallDetections":
        z = np.zeros(0, dtype=np.float32)
        return BallDetections(t=z, x=z.copy(), y=z.copy(), w=z.copy(), h=z.copy(), conf=z.copy())


@dataclass
class TrackingResult:
    """Everything the detector/tracker learned about one processing window."""

    fps: float
    frame_size: Tuple[int, int]  # (W, H) of the SOURCE video
    duration_s: float
    players: Dict[int, PlayerTrack]
    ball: BallDetections
    trim_offset_s: float = 0.0
    vid_stride: int = 1
    # Which player the user asked to follow (after ROI matching/stitching).
    focus_track_id: Optional[int] = None
    # Detection ran at this scale relative to the source (1.0 = full res).
    proxy_scale: float = 1.0
    detector: Dict[str, object] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    source_video_path: Optional[str] = None
    processing_video_path: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def frame_width(self) -> int:
        return int(self.frame_size[0])

    @property
    def frame_height(self) -> int:
        return int(self.frame_size[1])

    def focus_track(self) -> Optional[PlayerTrack]:
        if self.focus_track_id is None:
            return None
        return self.players.get(int(self.focus_track_id))

    def longest_track_id(self) -> Optional[int]:
        if not self.players:
            return None
        return max(self.players, key=lambda k: self.players[k].duration_s)

    def all_player_positions(self, max_hz: float = 10.0) -> np.ndarray:
        """``(t, cx, cy)`` float32 rows for every player, subsampled to ``max_hz``.

        This is the legacy input for field-geometry estimation and the camera
        planner's action-centroid lookup.
        """
        rows: List[np.ndarray] = []
        keep_every = max(1, int(round((self.fps / max(1, self.vid_stride)) / max(0.5, max_hz))))
        for track in self.players.values():
            if not len(track):
                continue
            centers = track.center_xy()[::keep_every]
            ts = track.t[::keep_every]
            rows.append(np.column_stack([ts, centers[:, 0], centers[:, 1]]).astype(np.float32))
        if not rows:
            return np.empty((0, 3), dtype=np.float32)
        out = np.concatenate(rows, axis=0)
        return out[np.argsort(out[:, 0], kind="stable")]

    def tracks_at(self, t: float, tolerance_s: float = 0.25) -> List[Dict[str, object]]:
        """Boxes of every player visible near time ``t`` (UI player picker)."""
        found: List[Dict[str, object]] = []
        for track in self.players.values():
            box = track.box_at(t, tolerance_s)
            if box is None:
                continue
            found.append(
                {
                    "track_id": int(track.track_id),
                    "team": int(track.team),
                    "label": track.label,
                    "jersey_number": track.jersey_number,
                    "x1": round(box[0], 1),
                    "y1": round(box[1], 1),
                    "x2": round(box[2], 1),
                    "y2": round(box[3], 1),
                }
            )
        return found

    def team_of(self, track_id: int) -> int:
        track = self.players.get(int(track_id))
        return int(track.team) if track is not None else TEAM_UNKNOWN

    def summary(self) -> Dict[str, object]:
        return {
            "fps": round(float(self.fps), 3),
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "duration_s": round(float(self.duration_s), 3),
            "trim_offset_s": round(float(self.trim_offset_s), 3),
            "vid_stride": int(self.vid_stride),
            "proxy_scale": round(float(self.proxy_scale), 4),
            "player_track_count": len(self.players),
            "player_samples": int(sum(len(p) for p in self.players.values())),
            "ball_detection_count": len(self.ball),
            "focus_track_id": self.focus_track_id,
            "detector": dict(self.detector),
            "timings": {k: round(float(v), 3) for k, v in self.timings.items()},
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Write ``tracks.npz`` + ``tracks_meta.json`` into ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        # Flatten players into one table: [track_id, t, x1, y1, x2, y2, conf]
        if self.players:
            blocks = []
            for track in self.players.values():
                n = len(track)
                if n == 0:
                    continue
                ids = np.full(n, float(track.track_id), dtype=np.float32)
                blocks.append(
                    np.column_stack([ids, track.t, track.x1, track.y1, track.x2, track.y2, track.conf]).astype(
                        np.float32
                    )
                )
            arrays["players"] = np.concatenate(blocks, axis=0) if blocks else np.empty((0, 7), np.float32)
        else:
            arrays["players"] = np.empty((0, 7), np.float32)
        arrays["ball"] = np.column_stack(
            [self.ball.t, self.ball.x, self.ball.y, self.ball.w, self.ball.h, self.ball.conf]
        ).astype(np.float32) if len(self.ball) else np.empty((0, 6), np.float32)
        np.savez_compressed(directory / TRACKS_FILENAME, **arrays)

        meta = {
            **self.summary(),
            "source_video_path": self.source_video_path,
            "processing_video_path": self.processing_video_path,
            "players": [p.summary() for p in self.players.values()],
        }
        (directory / TRACKS_META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return directory / TRACKS_FILENAME

    @staticmethod
    def exists(directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / TRACKS_FILENAME).is_file() and (directory / TRACKS_META_FILENAME).is_file()

    @classmethod
    def load(cls, directory: str | Path) -> "TrackingResult":
        directory = Path(directory)
        meta = json.loads((directory / TRACKS_META_FILENAME).read_text(encoding="utf-8"))
        data = np.load(directory / TRACKS_FILENAME)
        table = np.asarray(data["players"], dtype=np.float32).reshape(-1, 7)
        player_meta = {int(p["track_id"]): p for p in meta.get("players", [])}
        players: Dict[int, PlayerTrack] = {}
        if table.shape[0]:
            order = np.lexsort((table[:, 1], table[:, 0]))
            table = table[order]
            ids = table[:, 0].astype(np.int64)
            boundaries = np.flatnonzero(np.diff(ids)) + 1
            for chunk in np.split(table, boundaries):
                track_id = int(chunk[0, 0])
                pm = player_meta.get(track_id, {})
                players[track_id] = PlayerTrack(
                    track_id=track_id,
                    t=chunk[:, 1].copy(),
                    x1=chunk[:, 2].copy(),
                    y1=chunk[:, 3].copy(),
                    x2=chunk[:, 4].copy(),
                    y2=chunk[:, 5].copy(),
                    conf=chunk[:, 6].copy(),
                    team=int(pm.get("team", TEAM_UNKNOWN)),
                    team_confidence=float(pm.get("team_confidence", 0.0) or 0.0),
                    jersey_color_hex=pm.get("jersey_color_hex"),
                    jersey_number=pm.get("jersey_number"),
                    label=pm.get("label"),
                    source_track_ids=[int(v) for v in pm.get("source_track_ids", [])],
                )
        ball_table = np.asarray(data["ball"], dtype=np.float32).reshape(-1, 6)
        ball = BallDetections(
            t=ball_table[:, 0].copy(),
            x=ball_table[:, 1].copy(),
            y=ball_table[:, 2].copy(),
            w=ball_table[:, 3].copy(),
            h=ball_table[:, 4].copy(),
            conf=ball_table[:, 5].copy(),
        )
        return cls(
            fps=float(meta["fps"]),
            frame_size=(int(meta["frame_width"]), int(meta["frame_height"])),
            duration_s=float(meta.get("duration_s", 0.0)),
            players=players,
            ball=ball,
            trim_offset_s=float(meta.get("trim_offset_s", 0.0)),
            vid_stride=int(meta.get("vid_stride", 1)),
            focus_track_id=meta.get("focus_track_id"),
            proxy_scale=float(meta.get("proxy_scale", 1.0)),
            detector=dict(meta.get("detector", {})),
            timings=dict(meta.get("timings", {})),
            source_video_path=meta.get("source_video_path"),
            processing_video_path=meta.get("processing_video_path"),
        )


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------


def player_track_from_rows(
    track_id: int,
    rows: Iterable[Sequence[float]],
    **kwargs: object,
) -> PlayerTrack:
    """Build a PlayerTrack from ``(t, x1, y1, x2, y2[, conf])`` rows."""
    arr = np.asarray(list(rows), dtype=np.float32)
    if arr.size == 0:
        z = np.zeros(0, dtype=np.float32)
        return PlayerTrack(track_id=int(track_id), t=z, x1=z.copy(), y1=z.copy(), x2=z.copy(), y2=z.copy(),
                           conf=z.copy(), **kwargs)  # type: ignore[arg-type]
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError("rows must be (t, x1, y1, x2, y2[, conf])")
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    conf = arr[:, 5] if arr.shape[1] >= 6 else np.ones(arr.shape[0], dtype=np.float32)
    return PlayerTrack(
        track_id=int(track_id),
        t=arr[:, 0].copy(),
        x1=arr[:, 1].copy(),
        y1=arr[:, 2].copy(),
        x2=arr[:, 3].copy(),
        y2=arr[:, 4].copy(),
        conf=np.asarray(conf, dtype=np.float32).copy(),
        **kwargs,  # type: ignore[arg-type]
    )


def ball_detections_from_rows(rows: Iterable[Sequence[float]]) -> BallDetections:
    """Build from ``(t, x, y[, w, h, conf])`` rows."""
    arr = np.asarray(list(rows), dtype=np.float32)
    if arr.size == 0:
        return BallDetections.empty()
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("rows must be (t, x, y[, w, h, conf])")
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    n = arr.shape[0]
    w = arr[:, 3] if arr.shape[1] >= 4 else np.zeros(n, np.float32)
    h = arr[:, 4] if arr.shape[1] >= 5 else np.zeros(n, np.float32)
    conf = arr[:, 5] if arr.shape[1] >= 6 else np.ones(n, np.float32)
    return BallDetections(
        t=arr[:, 0].copy(), x=arr[:, 1].copy(), y=arr[:, 2].copy(),
        w=np.asarray(w, np.float32).copy(), h=np.asarray(h, np.float32).copy(),
        conf=np.asarray(conf, np.float32).copy(),
    )
