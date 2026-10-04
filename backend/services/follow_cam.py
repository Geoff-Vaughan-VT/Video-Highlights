"""Legacy follow-cam (constant zoom, follow one player and/or the ball).

The centers are smoothed with the camera planner v2 primitives (zero-phase
Gaussian low-pass + forward/backward speed/acceleration limiter), and clips
are rendered by the ffmpeg-native camera renderer (``camera_render``): the
source is decoded once by ffmpeg, cropped via a ``sendcmd`` script, scaled
to the output size (1080p-class by default, never the 4K source size) and
encoded with audio in the same command - no per-frame Python.
"""

from __future__ import annotations

import logging
import math
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .ffmpeg_tools import ffmpeg_available, ffmpeg_exe
from ..utils import ensure_dir

TrackSample = Tuple[float, float, float]
VideoCenter = Tuple[float, float]
RenderProgressCallback = Callable[[int, int], None]

LOGGER = logging.getLogger("videohighlights.follow_cam")

# Legacy smooth_factor (per-frame exponential factor) -> v2 Gaussian sigma:
# 0.2 (old default) maps to this many seconds; 1.0 means "no smoothing".
_LEGACY_SMOOTH_SIGMA_S = 0.55
_LEGACY_MAX_PAN_SPEED_CROP_FRAC = 1.0
_LEGACY_MAX_PAN_ACCEL_CROP_FRAC = 1.2

# How strongly each camera mode weights the ball vs the player track when
# blending the legacy follow-cam focus point. Shared by the pipeline and the
# API event-clip renderer so the policy cannot drift between them.
CAMERA_MODE_BALL_WEIGHTS = {
    "follow_player": 0.0,
    "follow_action": 0.35,
    "follow_ball": 1.0,
}


def ball_weight_for_mode(camera_mode: str) -> float:
    return CAMERA_MODE_BALL_WEIGHTS.get(str(camera_mode or "").strip().lower(), 0.0)


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video follow-cam rendering") from exc
    return cv2


def _normalize_track(track: Optional[Iterable[TrackSample]]) -> np.ndarray:
    if not track:
        return np.empty((0, 3), dtype=np.float32)

    rows: List[Tuple[float, float, float]] = []
    for item in track:
        try:
            t, x, y = item
            rows.append((float(t), float(x), float(y)))
        except Exception:
            continue

    if not rows:
        return np.empty((0, 3), dtype=np.float32)

    rows.sort(key=lambda value: value[0])
    return np.asarray(rows, dtype=np.float32)


def _interpolate_track_point(track: np.ndarray, t: float, max_gap_seconds: Optional[float] = None) -> Optional[Tuple[float, float]]:
    if track.size == 0:
        return None

    times = track[:, 0]
    idx = int(np.searchsorted(times, t))

    if idx <= 0:
        nearest_gap = abs(float(t) - float(times[0]))
        if max_gap_seconds is not None and nearest_gap > max_gap_seconds:
            return None
        return float(track[0, 1]), float(track[0, 2])

    if idx >= len(track):
        nearest_gap = abs(float(t) - float(times[-1]))
        if max_gap_seconds is not None and nearest_gap > max_gap_seconds:
            return None
        return float(track[-1, 1]), float(track[-1, 2])

    left = track[idx - 1]
    right = track[idx]
    left_gap = abs(float(t) - float(left[0]))
    right_gap = abs(float(right[0]) - float(t))
    nearest_gap = min(left_gap, right_gap)
    if max_gap_seconds is not None and nearest_gap > max_gap_seconds:
        return None

    span = float(right[0] - left[0])
    if span <= 1e-6:
        return float(left[1]), float(left[2])

    alpha = float((t - left[0]) / span)
    x = float(left[1] + (right[1] - left[1]) * alpha)
    y = float(left[2] + (right[2] - left[2]) * alpha)
    return x, y


def _clamp_center(center: Tuple[float, float], frame_size: Tuple[int, int], zoom_factor: float) -> Tuple[float, float]:
    frame_w, frame_h = frame_size
    crop_w = max(2.0, float(frame_w) / max(1.0, float(zoom_factor)))
    crop_h = max(2.0, float(frame_h) / max(1.0, float(zoom_factor)))
    half_w = crop_w / 2.0
    half_h = crop_h / 2.0
    x = min(max(center[0], half_w), max(half_w, float(frame_w) - half_w))
    y = min(max(center[1], half_h), max(half_h, float(frame_h) - half_h))
    return x, y


def build_follow_cam_centers(
    player_track: Sequence[TrackSample],
    ball_track: Optional[Sequence[TrackSample]],
    start_seconds: float,
    end_seconds: float,
    fps: float,
    frame_size: Tuple[int, int],
    zoom_factor: float = 1.6,
    ball_weight: float = 0.0,
    smooth_factor: float = 0.2,
    max_player_gap_seconds: float = 0.75,
    max_ball_gap_seconds: float = 0.35,
) -> List[Tuple[float, float]]:
    """Per-frame crop centers following a player (blended with the ball).

    v2 smoothing: the raw focus points are clamped to the legal crop region,
    low-passed with a zero-phase Gaussian whose width follows the legacy
    ``smooth_factor`` (0.2 -> ~0.55 s, 1.0 -> none), then passed through the
    planner's forward/backward speed/acceleration limiter (1.0 crop widths/s,
    1.2 crop widths/s^2), so there is no lag and no snapping.
    """
    from .camera_planner import _gaussian_smooth, _limit_motion

    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    if fps <= 0:
        raise ValueError("fps must be positive")

    player = _normalize_track(player_track)
    ball = _normalize_track(ball_track)
    frame_w, frame_h = frame_size
    frame_count = max(1, int(math.ceil((end_seconds - start_seconds) * fps)))

    raw = np.empty((frame_count, 2), dtype=np.float64)
    for index in range(frame_count):
        t = float(start_seconds + (index / fps))
        player_point = _interpolate_track_point(player, t, max_gap_seconds=max_player_gap_seconds)
        ball_point = None
        if ball_weight > 0.0:
            ball_point = _interpolate_track_point(ball, t, max_gap_seconds=max_ball_gap_seconds)

        if player_point is not None:
            focus_x, focus_y = player_point
            if ball_point is not None:
                focus_x = (focus_x * (1.0 - ball_weight)) + (ball_point[0] * ball_weight)
                focus_y = (focus_y * (1.0 - ball_weight)) + (ball_point[1] * ball_weight)
        elif ball_point is not None:
            focus_x, focus_y = ball_point
        else:
            focus_x, focus_y = frame_w / 2.0, frame_h / 2.0
        raw[index] = _clamp_center((focus_x, focus_y), frame_size, zoom_factor)

    strength = min(1.5, max(0.0, (1.0 - float(smooth_factor)) / 0.8))
    sigma_frames = _LEGACY_SMOOTH_SIGMA_S * strength * fps
    xs = _gaussian_smooth(raw[:, 0], sigma_frames)
    ys = _gaussian_smooth(raw[:, 1], sigma_frames)
    crop_w = float(frame_w) / max(1.0, float(zoom_factor))
    dt = 1.0 / fps
    vmax = np.full(frame_count, _LEGACY_MAX_PAN_SPEED_CROP_FRAC * crop_w)
    amax = np.full(frame_count, _LEGACY_MAX_PAN_ACCEL_CROP_FRAC * crop_w)
    xs, ys = _limit_motion(xs, ys, dt, vmax, amax)

    centers: List[Tuple[float, float]] = []
    for x, y in zip(xs, ys):
        cx, cy = _clamp_center((float(x), float(y)), frame_size, zoom_factor)
        centers.append((round(cx, 3), round(cy, 3)))
    return centers


def crop_frame_to_center(
    frame: np.ndarray,
    center: Tuple[float, float],
    zoom_factor: float,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    target_w, target_h = output_size or (frame_w, frame_h)
    crop_w = max(2, min(frame_w, int(round(frame_w / max(1.0, float(zoom_factor))))))
    crop_h = max(2, min(frame_h, int(round(frame_h / max(1.0, float(zoom_factor))))))

    clamped_x, clamped_y = _clamp_center(center, (frame_w, frame_h), zoom_factor)
    x1 = int(round(clamped_x - (crop_w / 2.0)))
    y1 = int(round(clamped_y - (crop_h / 2.0)))
    x1 = min(max(0, x1), max(0, frame_w - crop_w))
    y1 = min(max(0, y1), max(0, frame_h - crop_h))
    cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
    if cropped.size == 0:
        cropped = frame
    try:
        cv2 = _import_cv2()
        downscale = cropped.shape[1] >= target_w and cropped.shape[0] >= target_h
        interp = cv2.INTER_AREA if downscale else cv2.INTER_CUBIC
        return cv2.resize(cropped, (target_w, target_h), interpolation=interp)
    except RuntimeError:
        y_idx = np.linspace(0, cropped.shape[0] - 1, target_h).astype(int)
        x_idx = np.linspace(0, cropped.shape[1] - 1, target_w).astype(int)
        return cropped[np.ix_(y_idx, x_idx)]


def _mux_audio(
    source_video_path: str,
    temp_video_path: str,
    output_path: str,
    start_seconds: float,
    end_seconds: float,
) -> bool:
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-to",
        f"{end_seconds:.3f}",
        "-i",
        str(source_video_path),
        "-i",
        str(temp_video_path),
        "-map",
        "1:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0


@lru_cache(maxsize=None)
def _ffmpeg_encoder_available(encoder: str) -> bool:
    # Cached: availability cannot change mid-process, and probing spawns an
    # ffmpeg subprocess that would otherwise run once per rendered clip.
    if not ffmpeg_available():
        return False
    result = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-v", "error", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and encoder in result.stdout


def render_follow_cam_clip(
    video_path: str,
    output_path: str,
    start_seconds: float,
    end_seconds: float,
    player_track: Sequence[TrackSample],
    ball_track: Optional[Sequence[TrackSample]] = None,
    zoom_factor: float = 1.6,
    ball_weight: float = 0.0,
    smooth_factor: float = 0.2,
    include_audio: bool = True,
    progress_callback: Optional[RenderProgressCallback] = None,
    output_size: Optional[Tuple[int, int]] = None,
    encoder: str = "auto",
    hwaccel: Optional[str] = "auto",
    engine: str = "ffmpeg",
) -> str:
    """Render a constant-zoom follow-cam clip of ``[start_seconds, end_seconds)``.

    Times and track samples are in ``video_path``'s timebase. Builds a
    :class:`CameraPlan` from the v2-smoothed centers and renders it with the
    ffmpeg-native renderer (crop via ``sendcmd`` + scale + encode + audio in
    one ffmpeg pass). ``output_size`` defaults to 1080p-class output, never
    larger than the source.
    """
    from .camera_planner import CameraDecision, CameraPlan, resolve_output_size
    from .camera_render import probe_video, render_camera_plan_video

    start_s = max(0.0, float(start_seconds))
    end_s = float(end_seconds)
    if end_s <= start_s:
        raise ValueError("end_seconds must be greater than start_seconds")
    out_file = Path(output_path)
    ensure_dir(str(out_file.parent))

    info = probe_video(video_path)
    if info is None or info.width <= 0 or info.height <= 0:
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(info.fps or 30.0)
    frame_size = (int(info.width), int(info.height))
    zoom = max(1.0, float(zoom_factor))

    centers = build_follow_cam_centers(
        player_track=player_track,
        ball_track=ball_track,
        start_seconds=start_s,
        end_seconds=end_s,
        fps=fps,
        frame_size=frame_size,
        zoom_factor=zoom,
        ball_weight=ball_weight,
        smooth_factor=smooth_factor,
    )
    plan = CameraPlan(
        start_seconds=start_s,
        fps=fps,
        frame_size=frame_size,
        base_zoom=zoom,
        output_size=resolve_output_size(frame_size, output_size),
        max_zoom=zoom,
        style="legacy_follow_cam",
    )
    reason = "legacy follow-cam: player" + (f" blended with ball ({ball_weight:.2f})" if ball_weight > 0 else "")
    for index, (cx, cy) in enumerate(centers):
        plan.decisions.append(
            CameraDecision(index=index, t=start_s + index / fps, center_x=cx, center_y=cy, zoom=zoom,
                           state="in_play", focus="player", reason=reason, confidence=1.0)
        )
    path = render_camera_plan_video(
        video_path=video_path,
        output_path=str(out_file),
        plan=plan,
        include_audio=include_audio,
        progress_callback=progress_callback,
        output_size=plan.output_size,
        encoder=encoder,
        hwaccel=hwaccel,
        engine=engine,
    )
    LOGGER.info("follow-cam clip %.1fs-%.1fs rendered at %dx%d -> %s", start_s, end_s,
                plan.output_size[0], plan.output_size[1], out_file.name)
    return path
