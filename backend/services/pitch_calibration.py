"""Image-to-pitch calibration (homography) for metric match stats.

A :class:`PitchCalibration` maps **source-video pixels** to **pitch metres**
with a planar homography. Pitch coordinates:

* origin at the centre spot,
* ``x`` along the length: the *left* goal line (as seen in the image) is at
  ``x = -pitch_length_m / 2``, the right goal line at ``+pitch_length_m / 2``,
* ``y`` across the width: the far touchline (top of the image) is at
  ``y = -pitch_width_m / 2``, the near touchline (bottom of the image) at
  ``+pitch_width_m / 2``.

Everything mapped to the pitch uses the **foot point** of a player box
(``PlayerTrack.foot_xy``) and the ball's image position (the ball's height
above the grass is ignored, which is fine while it is on the ground and an
approximation in the air).

Two ways to build one:

* :func:`calibrate_from_corners` (``source="manual"``, confidence 0.95): the
  user clicks the four corners of the playing area. Exact up to click error.
* :func:`calibrate_from_geometry` (``source="auto"``, confidence ~0.5): from
  ``game_tracking.FieldGeometry`` (axis-aligned bounds estimated from where
  players went). A side-line camera sees the pitch in perspective: the far
  touchline is shorter than the near one. With only a bounding rectangle we
  assume the trapezoid of a typical elevated side-line camera: the near
  touchline spans the full estimated width and the far touchline is
  ``far_line_ratio`` (default 0.85) of it, centred. This is an approximation;
  metres from an auto calibration are indicative and the UI should show the
  confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

LOGGER = logging.getLogger("videohighlights.pitch_calibration")

DEFAULT_PITCH_LENGTH_M = 105.0
DEFAULT_PITCH_WIDTH_M = 68.0
# Typical elevated side-line camera: the far touchline appears this fraction
# of the near touchline's length. Only used when the field is known as an
# axis-aligned rectangle (auto calibration).
DEFAULT_FAR_LINE_RATIO = 0.85

AUTO_CONFIDENCE = 0.5
AUTO_DEFAULT_FRAME_CONFIDENCE = 0.25
MANUAL_CONFIDENCE = 0.95

THIRD_DEFENSIVE = 0
THIRD_MIDDLE = 1
THIRD_ATTACKING = 2
THIRD_NAMES = ("defensive", "middle", "attacking")


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - OpenCV is a hard dependency
        raise RuntimeError("OpenCV is required for pitch calibration") from exc
    return cv2


def _pitch_corners_m(length_m: float, width_m: float) -> np.ndarray:
    """TL, TR, BR, BL of the playing area in pitch metres."""
    hl, hw = length_m / 2.0, width_m / 2.0
    return np.asarray([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=np.float64)


def _apply_h(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to [n, 2] points; points behind the horizon -> NaN."""
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    pts = pts.reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    xw = h[0, 0] * pts[:, 0] + h[0, 1] * pts[:, 1] + h[0, 2]
    yw = h[1, 0] * pts[:, 0] + h[1, 1] * pts[:, 1] + h[1, 2]
    w = h[2, 0] * pts[:, 0] + h[2, 1] * pts[:, 1] + h[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.stack([xw / w, yw / w], axis=1)
    # w <= 0 means the point is on/behind the vanishing line: meaningless.
    out[~(w > 1e-12)] = np.nan
    return out[0] if single else out


def _is_convex_quad(corners: np.ndarray) -> bool:
    cross = []
    for i in range(4):
        a, b, c = corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]
        ab, bc = b - a, c - b
        cross.append(ab[0] * bc[1] - ab[1] * bc[0])
    cross = np.asarray(cross)
    return bool(np.all(cross > 1e-6) or np.all(cross < -1e-6))


@dataclass
class PitchCalibration:
    """Homography from source-video pixels to pitch metres (and back)."""

    homography: np.ndarray  # 3x3, image px -> pitch m
    image_corners_px: np.ndarray  # [4, 2] TL, TR, BR, BL of the playing area
    source: str = "auto"  # "auto" | "manual"
    confidence: float = AUTO_CONFIDENCE
    pitch_length_m: float = DEFAULT_PITCH_LENGTH_M
    pitch_width_m: float = DEFAULT_PITCH_WIDTH_M
    notes: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.homography = np.asarray(self.homography, dtype=np.float64).reshape(3, 3)
        self.image_corners_px = np.asarray(self.image_corners_px, dtype=np.float64).reshape(4, 2)

    # ------------------------------------------------------------------
    @property
    def inverse(self) -> np.ndarray:
        """3x3 homography pitch m -> image px."""
        return np.linalg.inv(self.homography)

    @property
    def half_length_m(self) -> float:
        return self.pitch_length_m / 2.0

    @property
    def half_width_m(self) -> float:
        return self.pitch_width_m / 2.0

    def to_pitch(self, points_px: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Image pixels ([2] or [n, 2]) -> pitch metres (NaN beyond the horizon)."""
        return _apply_h(self.homography, np.asarray(points_px, dtype=np.float64))

    def to_image(self, points_m: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Pitch metres ([2] or [n, 2]) -> image pixels."""
        return _apply_h(self.inverse, np.asarray(points_m, dtype=np.float64))

    def in_bounds(self, points_m: Union[Sequence[float], np.ndarray], margin_m: float = 0.0) -> np.ndarray:
        """Boolean mask: pitch-metre points inside the playing area (+ margin)."""
        pts = np.asarray(points_m, dtype=np.float64)
        single = pts.ndim == 1
        pts = pts.reshape(-1, 2)
        with np.errstate(invalid="ignore"):
            ok = (np.abs(pts[:, 0]) <= self.half_length_m + margin_m) & (
                np.abs(pts[:, 1]) <= self.half_width_m + margin_m
            )
        return bool(ok[0]) if single else ok

    def thirds(self, x_m: Union[float, np.ndarray], attacking_direction: Union[int, str]) -> np.ndarray:
        """Third index per x (0 defensive, 1 middle, 2 attacking).

        ``attacking_direction`` is +1/"right" when the team attacks the right
        goal (+x), -1/"left" otherwise.
        """
        sign = _direction_sign(attacking_direction)
        x = np.asarray(x_m, dtype=np.float64) * sign
        third = self.pitch_length_m / 3.0
        edge = self.half_length_m - third  # = length / 6
        out = np.full(x.shape, THIRD_MIDDLE, dtype=np.int8)
        out[x < -edge] = THIRD_DEFENSIVE
        out[x > edge] = THIRD_ATTACKING
        return out

    def clip_to_pitch(self, points_m: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 2).copy()
        pts[:, 0] = np.clip(pts[:, 0], -self.half_length_m, self.half_length_m)
        pts[:, 1] = np.clip(pts[:, 1], -self.half_width_m, self.half_width_m)
        return pts

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "confidence": round(float(self.confidence), 3),
            "pitch_length_m": float(self.pitch_length_m),
            "pitch_width_m": float(self.pitch_width_m),
            "image_corners_px": [[round(float(x), 2), round(float(y), 2)] for x, y in self.image_corners_px],
            "homography": [[float(v) for v in row] for row in self.homography],
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PitchCalibration":
        corners = np.asarray(data["image_corners_px"], dtype=np.float64).reshape(4, 2)
        length = float(data.get("pitch_length_m", DEFAULT_PITCH_LENGTH_M))  # type: ignore[arg-type]
        width = float(data.get("pitch_width_m", DEFAULT_PITCH_WIDTH_M))  # type: ignore[arg-type]
        raw_h = data.get("homography")
        if raw_h is None:
            h = _homography_from_corners(corners, length, width)
        else:
            h = np.asarray(raw_h, dtype=np.float64).reshape(3, 3)
        return cls(
            homography=h,
            image_corners_px=corners,
            source=str(data.get("source", "auto")),
            confidence=float(data.get("confidence", AUTO_CONFIDENCE)),  # type: ignore[arg-type]
            pitch_length_m=length,
            pitch_width_m=width,
            notes=dict(data.get("notes") or {}),  # type: ignore[arg-type]
        )


def _direction_sign(direction: Union[int, float, str]) -> float:
    if isinstance(direction, str):
        return 1.0 if direction.lower() in ("right", "+", "+x", "1") else -1.0
    return 1.0 if float(direction) >= 0 else -1.0


def _homography_from_corners(corners_px: np.ndarray, length_m: float, width_m: float) -> np.ndarray:
    cv2 = _import_cv2()
    src = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
    dst = _pitch_corners_m(length_m, width_m).astype(np.float32)
    return np.asarray(cv2.getPerspectiveTransform(src, dst), dtype=np.float64)


def calibrate_from_corners(
    image_corners_px: Sequence[Sequence[float]],
    pitch_length_m: float = DEFAULT_PITCH_LENGTH_M,
    pitch_width_m: float = DEFAULT_PITCH_WIDTH_M,
    *,
    frame_size: Optional[Tuple[int, int]] = None,
    confidence: float = MANUAL_CONFIDENCE,
) -> PitchCalibration:
    """Manual calibration from the 4 playing-area corners clicked in the image.

    ``image_corners_px`` is TL, TR, BR, BL (far-left, far-right, near-right,
    near-left corner flags) in source pixels. When every value is <= 1 and
    ``frame_size`` is given they are treated as normalized coordinates.
    """
    corners = np.asarray(image_corners_px, dtype=np.float64).reshape(-1, 2)
    if corners.shape[0] != 4:
        raise ValueError("image_corners_px must hold exactly 4 points (TL, TR, BR, BL)")
    if frame_size is not None and float(np.nanmax(np.abs(corners))) <= 1.0:
        corners = corners * np.asarray([float(frame_size[0]), float(frame_size[1])])
    if not np.all(np.isfinite(corners)) or not _is_convex_quad(corners):
        raise ValueError("image_corners_px must form a convex quadrilateral (TL, TR, BR, BL)")
    h = _homography_from_corners(corners, pitch_length_m, pitch_width_m)
    return PitchCalibration(
        homography=h,
        image_corners_px=corners,
        source="manual",
        confidence=float(confidence),
        pitch_length_m=float(pitch_length_m),
        pitch_width_m=float(pitch_width_m),
    )


def calibrate_from_geometry(
    field_geometry,
    frame_size: Optional[Tuple[int, int]] = None,
    pitch_length_m: float = DEFAULT_PITCH_LENGTH_M,
    pitch_width_m: float = DEFAULT_PITCH_WIDTH_M,
    *,
    far_line_ratio: float = DEFAULT_FAR_LINE_RATIO,
) -> PitchCalibration:
    """Auto calibration from ``game_tracking.FieldGeometry`` bounds.

    The near touchline (``y_max``) spans ``[x_min, x_max]``; the far
    touchline (``y_min``) is ``far_line_ratio`` of that length, centred - the
    trapezoid an elevated side-line camera sees. ``far_line_ratio=1.0``
    reproduces a plain rectangle (top-down camera). Confidence is 0.5 for
    player-derived bounds and 0.25 when the geometry is only a frame default.
    """
    if field_geometry is None:
        if frame_size is None:
            raise ValueError("field_geometry or frame_size is required")
        w, h = float(frame_size[0]), float(frame_size[1])
        x_min, x_max, y_min, y_max = w * 0.02, w * 0.98, h * 0.10, h * 0.90
        geo_source = "frame_default"
    else:
        x_min, x_max = float(field_geometry.x_min), float(field_geometry.x_max)
        y_min, y_max = float(field_geometry.y_min), float(field_geometry.y_max)
        geo_source = str(getattr(field_geometry, "source", "estimated"))
    if x_max - x_min < 4 or y_max - y_min < 4:
        raise ValueError("field geometry is degenerate")
    ratio = float(min(1.0, max(0.3, far_line_ratio)))
    cx = (x_min + x_max) / 2.0
    half_far = (x_max - x_min) * ratio / 2.0
    corners = np.asarray(
        [[cx - half_far, y_min], [cx + half_far, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float64
    )
    h = _homography_from_corners(corners, pitch_length_m, pitch_width_m)
    if geo_source == "frame_default":
        confidence = AUTO_DEFAULT_FRAME_CONFIDENCE
    elif "manual" in geo_source:
        confidence = 0.6
    else:
        confidence = AUTO_CONFIDENCE
    calib = PitchCalibration(
        homography=h,
        image_corners_px=corners,
        source="auto",
        confidence=confidence,
        pitch_length_m=float(pitch_length_m),
        pitch_width_m=float(pitch_width_m),
        notes={"geometry_source": geo_source, "far_line_ratio": ratio},
    )
    LOGGER.info(
        "auto pitch calibration from %s bounds x=[%.0f, %.0f] y=[%.0f, %.0f] (far line ratio %.2f, confidence %.2f)",
        geo_source, x_min, x_max, y_min, y_max, ratio, confidence,
    )
    return calib


def foot_positions(tracking, max_hz: float = 10.0) -> np.ndarray:
    """``(t, foot_x, foot_y)`` rows for every non-referee player (``max_hz``).

    Same shape as ``TrackingResult.all_player_positions`` but with foot points
    (where players touch the grass), which is what a pitch calibration needs:
    feed it to ``game_tracking.estimate_field_geometry`` before
    :func:`calibrate_from_geometry` for bounds that sit on the grass rather
    than half a body-height above it.
    """
    from .tracking_types import TEAM_REFEREE

    rows = []
    keep_every = max(1, int(round((tracking.fps / max(1, tracking.vid_stride)) / max(0.5, max_hz))))
    for track in tracking.players.values():
        if not len(track) or int(track.team) == TEAM_REFEREE:
            continue
        foot = track.foot_xy()[::keep_every]
        rows.append(np.column_stack([track.t[::keep_every], foot]).astype(np.float32))
    if not rows:
        return np.empty((0, 3), dtype=np.float32)
    out = np.concatenate(rows, axis=0)
    return out[np.argsort(out[:, 0], kind="stable")]


@dataclass
class _Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    source: str = "estimated"


def estimate_pitch_bounds(
    tracking,
    ball_track=None,
    segments: Optional[Sequence[object]] = None,
    *,
    field_geometry=None,
    player_percentile: float = 0.5,
    ball_percentile: float = 1.0,
) -> Optional[_Bounds]:
    """Playing-area bounds (px) from player FOOT points and the in-play ball.

    Player spread alone underestimates the pitch (outfield players rarely
    reach the goal lines or touchlines, especially in a short clip); the ball
    does reach them. Bounds are the union of robust percentiles of
    non-referee foot points, of the ball (except while it is in a net, i.e.
    during ``goal_*`` segments) and of ``field_geometry`` when supplied.
    """
    xs_lo, xs_hi, ys_lo, ys_hi = [], [], [], []
    feet = foot_positions(tracking)
    sources = []
    if len(feet) >= 50:
        xs_lo.append(np.percentile(feet[:, 1], player_percentile))
        xs_hi.append(np.percentile(feet[:, 1], 100 - player_percentile))
        ys_lo.append(np.percentile(feet[:, 2], player_percentile))
        ys_hi.append(np.percentile(feet[:, 2], 100 - player_percentile))
        sources.append("players")
    if ball_track is not None and len(ball_track) >= 50:
        bt = np.asarray(ball_track.times, dtype=np.float64)
        keep = np.ones(bt.size, dtype=bool)
        # Drop the ball while it sits in a net (goal states). Restart states
        # are kept on purpose: they are judged against an estimated field
        # that is itself too small when it came from player spread.
        for seg in segments or []:
            if str(getattr(seg, "state", "")).startswith("goal"):
                keep &= ~((bt >= float(seg.start_s)) & (bt <= float(seg.end_s)))
        if keep.sum() >= 50:
            bx = np.asarray(ball_track.xs, dtype=np.float64)[keep]
            by = np.asarray(ball_track.ys, dtype=np.float64)[keep]
            xs_lo.append(np.percentile(bx, ball_percentile))
            xs_hi.append(np.percentile(bx, 100 - ball_percentile))
            ys_lo.append(np.percentile(by, ball_percentile))
            ys_hi.append(np.percentile(by, 100 - ball_percentile))
            sources.append("ball")
    if field_geometry is not None and str(getattr(field_geometry, "source", "")) != "frame_default":
        xs_lo.append(float(field_geometry.x_min))
        xs_hi.append(float(field_geometry.x_max))
        ys_lo.append(float(field_geometry.y_min))
        ys_hi.append(float(field_geometry.y_max))
        sources.append("geometry")
    if not sources:
        return None
    w, h = float(tracking.frame_width), float(tracking.frame_height)
    return _Bounds(
        x_min=float(max(0.0, min(xs_lo))), x_max=float(min(w, max(xs_hi))),
        y_min=float(max(0.0, min(ys_lo))), y_max=float(min(h, max(ys_hi))),
        source="+".join(sources),
    )


def calibrate_auto(
    tracking,
    ball_track=None,
    segments: Optional[Sequence[object]] = None,
    *,
    field_geometry=None,
    pitch_length_m: float = DEFAULT_PITCH_LENGTH_M,
    pitch_width_m: float = DEFAULT_PITCH_WIDTH_M,
    far_line_ratio: float = DEFAULT_FAR_LINE_RATIO,
) -> PitchCalibration:
    """Best automatic calibration available: :func:`estimate_pitch_bounds` +
    :func:`calibrate_from_geometry` (falls back to the frame default)."""
    bounds = estimate_pitch_bounds(tracking, ball_track, segments, field_geometry=field_geometry)
    calib = calibrate_from_geometry(bounds, tracking.frame_size, pitch_length_m, pitch_width_m,
                                    far_line_ratio=far_line_ratio)
    return calib
