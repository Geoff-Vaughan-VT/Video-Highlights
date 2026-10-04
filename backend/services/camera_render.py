"""Render videos from a :class:`CameraPlan`.

Engines:

* **ffmpeg** (default, production) - the source is decoded once by ffmpeg
  (``-hwaccel cuda``/``videotoolbox`` when usable) and the virtual camera is
  applied inside the filter graph, encoded with h264_nvenc ->
  h264_videotoolbox -> libx264 and with the source audio muxed in the same
  command. No Python runs per frame. Two filter modes (both verified on
  ffmpeg 6.1 against a per-frame Python reference crop):

  - ``crop`` (default): ``sendcmd=f=cmds.txt,crop=W:H:X:Y,null,
    scale=OW:OH:flags=bicubic:threads=1,format=yuv420p`` with ``crop w/h/x/y``
    commands in source pixels (the ``camera_crops.txt`` format; size commands
    only when the size changes). ``crop`` rewrites its output link size on a
    ``w``/``h`` command, so a directly attached ``scale`` never notices the
    new size and ffmpeg 6.1 HANGS; the ``null`` filter keeps the stale link
    size so ``scale`` sees the change and re-initialises once per size step.
  - ``scale`` (automatic fallback): ``sendcmd,scale=w=SW:h=SH:eval=frame,
    crop=OW:OH:X:Y:exact=1,format=yuv420p`` - scale the whole source by
    OW/crop_w, then a fixed-size crop (1 output-pixel pan precision). Correct
    but ~4x slower at steady zoom on 4K (it rescales the full frame).

  Measured on this 4-core sandbox, 150 4K frames, crop+scale to 1080p,
  filter only: steady zoom 1.5 s, zoom changing every frame 1.8 s (decode
  alone 0.65 s); swscale ``threads=1`` matters (default threads: 4.5 s).

  A stalled ffmpeg (no progress for ``stall_timeout_s``) is killed and the
  next encoder / hwaccel / filter-mode combination is tried.

  ``sendcmd`` scans every interval on every frame (O(frames x commands)):
  54k per-frame commands cost ~280 s over a 30-minute render. Long plans are
  therefore rendered in chunks (default 60 s) with their own small command
  scripts, video H.264 + PCM audio in Matroska, then joined with the concat
  demuxer (video stream copy, one AAC encode of the audio). Chunks can run in
  parallel (``workers``).

  ``-ss``/``-to`` are input options (fast keyframe seek, frame-accurate
  output). Filter timestamps start at the seek point, so command times are
  written relative to each chunk's seek position; every command lands half a
  frame before its frame so timestamp rounding cannot shift it by one frame.

* **python** (fallback) - OpenCV decode, crop, INTER_AREA downscale to the
  OUTPUT size (never the source size), piped to an ffmpeg encoder. Used
  automatically for Python-only decorations (``overlay_banner`` or a custom
  ``scorebug_fn``) or when ffmpeg is unavailable.

* **debug wide** - annotated wide review video (Python, opt-in). Renders from
  ``debug_source_path`` (the proxy) when given, at most ``max_debug_width``
  wide.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, IO, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import ensure_dir
from .camera_planner import (
    CameraDecision,
    CameraPlan,
    _even,
    resolve_output_size,
    sendcmd_lines,
)
from .ffmpeg_tools import ffmpeg_available, ffmpeg_exe, ffprobe_exe
from .follow_cam import (
    RenderProgressCallback,
    _ffmpeg_encoder_available,
    _import_cv2,
    _mux_audio,
)
from .game_tracking import FieldGeometry, GoalBox, GOAL_STATES, RESTART_STATES

LOGGER = logging.getLogger("videohighlights.camera_render")

FrameDecorator = Callable[[np.ndarray, CameraDecision], np.ndarray]
ScorebugFn = Callable[[np.ndarray, float], np.ndarray]
ProgressInfoCallback = Callable[[Dict[str, float]], None]

ENCODER_CHAIN: Tuple[str, ...] = ("h264_nvenc", "h264_videotoolbox", "libx264")
HWACCEL_CHAIN: Tuple[str, ...] = ("cuda", "videotoolbox")
DEFAULT_CHUNK_SECONDS = 60.0

_FONT_CANDIDATES: Tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


# ---------------------------------------------------------------------------
# Scorebug
# ---------------------------------------------------------------------------


class RenderCancelled(RuntimeError):
    """The render's ``cancel_event`` was set; ffmpeg was stopped."""


class _RenderStopped(RuntimeError):
    """Internal: another chunk failed, so this one did not (re)start."""


def _scoring_side(goal: Dict[str, object]) -> str:
    """Renderer side convention for one goal: the goal the ball went INTO.

    ``team`` (0 = left label, 1 = right label) wins over ``side`` when known:
    teams swap ends at half time, so the goal side alone cannot say who
    scored. Team 0's goals count as "into the right goal", team 1's as "into
    the left goal"; ``side`` is the fallback when the team is unknown.
    """
    team = goal.get("team")
    if team is not None and not isinstance(team, bool):
        try:
            idx = int(team)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            idx = -1
        if idx in (0, 1):
            return "right" if idx == 0 else "left"
    return str(goal.get("side") or "")


def _safe_team(name: Optional[str], default: str) -> str:
    cleaned = "".join(ch for ch in str(name or default).upper() if ch.isalnum() or ch in " -")
    return (cleaned.strip() or default)[:10]


class ScorebugRenderer:
    """Broadcast scorebug: running score + match clock + GOAL flash.

    Callable as ``fn(frame, t)`` (OpenCV, python engine) and convertible to
    ffmpeg ``drawtext`` filters (ffmpeg engine) so the production render
    stays Python-free.
    """

    def __init__(self, goal_events: Sequence[Dict[str, object]], team_left: str = "HOME",
                 team_right: str = "AWAY", flash_s: float = 4.0) -> None:
        self.goals: List[Tuple[float, str]] = sorted(
            (float(g.get("t", 0.0)), _scoring_side(g)) for g in goal_events
        )
        self.team_left = _safe_team(team_left, "HOME")
        self.team_right = _safe_team(team_right, "AWAY")
        self.flash_s = float(flash_s)

    def score_at(self, t: float) -> Tuple[int, int]:
        # A goal INTO the left goal scores for the right-defending team.
        left = sum(1 for gt, side in self.goals if gt <= t and side == "right")
        right = sum(1 for gt, side in self.goals if gt <= t and side == "left")
        return left, right

    # -- OpenCV ---------------------------------------------------------
    def __call__(self, frame: np.ndarray, t: float) -> np.ndarray:
        cv2 = _import_cv2()
        score_l, score_r = self.score_at(t)
        clock = f"{int(t // 60):02d}:{int(t % 60):02d}"
        text = f"{self.team_left} {score_l} - {score_r} {self.team_right}   {clock}"

        h, w = frame.shape[:2]
        scale = max(0.45, w / 2600.0)
        thick = max(1, w // 900)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        pad = max(6, th // 2)
        x0, y0 = pad, pad
        x1, y1 = min(w, x0 + tw + 2 * pad), min(h, y0 + th + 2 * pad)
        overlay = frame[y0:y1, x0:x1].copy()
        overlay[:] = (20, 20, 20)
        frame[y0:y1, x0:x1] = cv2.addWeighted(overlay, 0.72, frame[y0:y1, x0:x1], 0.28, 0)
        cv2.putText(frame, text, (x0 + pad, y1 - pad), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thick)

        flash = next((gt for gt, _ in self.goals if 0.0 <= t - gt <= self.flash_s), None)
        if flash is not None:
            gtext = f"GOAL!  {int(flash // 60)}'"
            (gw, gh), _ = cv2.getTextSize(gtext, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.6, thick + 1)
            gx, gy = (w - gw) // 2, y1 + gh + pad * 2
            cv2.putText(frame, gtext, (gx + 2, gy + 2), cv2.FONT_HERSHEY_SIMPLEX,
                        scale * 1.6, (0, 0, 0), thick + 2)
            cv2.putText(frame, gtext, (gx, gy), cv2.FONT_HERSHEY_SIMPLEX,
                        scale * 1.6, (60, 60, 240), thick + 1)
        return frame

    # -- ffmpeg drawtext -------------------------------------------------
    def ffmpeg_filters(self, work_dir: Path, tag: str, plan_t0: float, duration: float,
                       output_size: Tuple[int, int], fontfile: Optional[str]) -> List[str]:
        """drawtext filters for a render whose filter time 0 = plan time ``plan_t0``.

        Text lives in small files inside ``work_dir`` (ffmpeg runs with that
        cwd), so no text or path ever needs filtergraph escaping.
        """
        out_w, out_h = output_size
        size = max(12, int(round(out_h * 0.034)))
        pad = max(4, size // 3)
        font = f":fontfile={fontfile}" if fontfile else ""
        filters: List[str] = []
        # Score changes only at goals -> one drawtext per score interval.
        bounds = [-1e9] + [gt for gt, _ in self.goals] + [1e12]
        for k in range(len(bounds) - 1):
            a, b = bounds[k] - plan_t0, bounds[k + 1] - plan_t0
            if b <= 0 or a >= duration:
                continue
            score_l, score_r = self.score_at(max(bounds[k], -1e8) + 1e-6 if k else -1e9)
            name = f"sb_{tag}_{k}.txt"
            clock = (f"%{{eif:floor((t+{plan_t0:.3f})/60):d:2}}:"
                     f"%{{eif:mod(floor(t+{plan_t0:.3f}),60):d:2}}")
            (work_dir / name).write_text(
                f"{self.team_left} {score_l} - {score_r} {self.team_right}   {clock}", encoding="utf-8"
            )
            filters.append(
                f"drawtext=textfile={name}:expansion=normal{font}:fontsize={size}:fontcolor=white"
                f":box=1:boxcolor=black@0.72:boxborderw={pad}:x={pad * 2}:y={pad * 2}"
                f":enable='between(t,{max(a, 0.0):.3f},{min(b, duration + 1.0):.3f})'"
            )
        for k, (gt, _side) in enumerate(self.goals):
            a, b = gt - plan_t0, gt + self.flash_s - plan_t0
            if b <= 0 or a >= duration:
                continue
            name = f"goal_{tag}_{k}.txt"
            (work_dir / name).write_text(f"GOAL!  {int(gt // 60)}'", encoding="utf-8")
            filters.append(
                f"drawtext=textfile={name}:expansion=none{font}:fontsize={int(size * 1.7)}"
                f":fontcolor=0xF03C3C:borderw={max(2, size // 8)}:bordercolor=black"
                f":x=(w-text_w)/2:y={pad * 4 + size * 2}"
                f":enable='between(t,{max(a, 0.0):.3f},{b:.3f})'"
            )
        return filters


def make_scorebug_renderer(
    goal_events: Sequence[Dict[str, object]],
    team_left: str = "HOME",
    team_right: str = "AWAY",
    flash_s: float = 4.0,
) -> ScorebugFn:
    """Broadcast scorebug: running score + match clock + GOAL flash.

    ``goal_events``: dicts with ``t``, ``side`` (which goal the ball
    entered) and optionally ``team`` (scoring team: 0 = ``team_left``,
    1 = ``team_right``) in the same timebase as render decisions. ``team``
    is authoritative (ends swap at half time); without it a goal INTO the
    left goal scores for the right label and vice versa. The returned
    :class:`ScorebugRenderer` is drawn with ffmpeg ``drawtext`` by the
    ffmpeg engine and with OpenCV by the python engine.
    """
    return ScorebugRenderer(goal_events, team_left=team_left, team_right=team_right, flash_s=flash_s)


# ---------------------------------------------------------------------------
# Debug overlays (python)
# ---------------------------------------------------------------------------

_STATE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "in_play": (80, 200, 80),
    "ball_lost": (60, 160, 230),
    "restart_left": (0, 170, 255),
    "restart_right": (0, 170, 255),
    "restart_touchline": (0, 170, 255),
    "goal_left": (60, 60, 240),
    "goal_right": (60, 60, 240),
}

_BALL_TRAIL_SECONDS = 1.5


def _state_color(state: str) -> Tuple[int, int, int]:
    return _STATE_COLORS.get(state, (200, 200, 200))


def _format_clock(t: float) -> str:
    minutes = int(t // 60)
    seconds = t - minutes * 60
    return f"{minutes:02d}:{seconds:04.1f}"


def annotate_wide_frame(
    frame: np.ndarray,
    decision: CameraDecision,
    *,
    ball_trail: Optional[Sequence[Tuple[float, float]]] = None,
    geometry: Optional[FieldGeometry] = None,
    player_marker: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Draw the camera decision explanation onto a wide frame (in place).

    Coordinates in ``decision``/``geometry`` must be in this frame's pixels
    (the debug renderer rescales them when drawing on a proxy).
    """
    cv2 = _import_cv2()
    frame_h, frame_w = frame.shape[:2]
    color = _state_color(decision.state)
    thickness = max(1, frame_w // 640)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, frame_w / 2400.0)

    if geometry is not None:
        cv2.rectangle(
            frame,
            (int(geometry.x_min), int(geometry.y_min)),
            (int(geometry.x_max), int(geometry.y_max)),
            (128, 128, 128),
            thickness,
        )
        for goal in (geometry.left_goal, geometry.right_goal):
            goal_color = (255, 160, 0)
            if decision.state in RESTART_STATES.union(GOAL_STATES) and decision.focus == f"goal_{goal.side}":
                goal_color = (0, 80, 255)
            cv2.rectangle(frame, (int(goal.x1), int(goal.y1)), (int(goal.x2), int(goal.y2)),
                          goal_color, thickness + 1)

    crop_w = frame_w / max(1.0, decision.zoom)
    crop_h = frame_h / max(1.0, decision.zoom)
    x1 = int(round(decision.center_x - crop_w / 2.0))
    y1 = int(round(decision.center_y - crop_h / 2.0))
    cv2.rectangle(frame, (x1, y1), (int(x1 + crop_w), int(y1 + crop_h)), color, thickness + 1)

    cx, cy = int(round(decision.center_x)), int(round(decision.center_y))
    arm = max(10, frame_w // 96)
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), (255, 0, 255), thickness + 1)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), (255, 0, 255), thickness + 1)

    if decision.target_x is not None and decision.target_y is not None:
        cv2.circle(frame, (int(decision.target_x), int(decision.target_y)),
                   max(3, thickness * 2), (255, 255, 0), -1)

    if ball_trail and len(ball_trail) >= 2:
        points = np.asarray(ball_trail, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=thickness)
    if decision.ball_x is not None and decision.ball_y is not None:
        bx, by = int(round(decision.ball_x)), int(round(decision.ball_y))
        radius = max(6, frame_w // 160)
        if decision.ball_source == "detected":
            cv2.circle(frame, (bx, by), radius, (0, 255, 255), -1)
        else:
            cv2.circle(frame, (bx, by), radius, (0, 255, 255), thickness + 1)

    if player_marker is not None:
        px, py = int(round(player_marker[0])), int(round(player_marker[1]))
        cv2.drawMarker(frame, (px, py), (255, 128, 0), cv2.MARKER_TRIANGLE_UP,
                       max(12, frame_w // 120), thickness + 1)

    banner_h = max(34, int(frame_h * 0.075))
    overlay = frame[0:banner_h, :].copy()
    overlay[:] = (20, 20, 20)
    frame[0:banner_h, :] = cv2.addWeighted(overlay, 0.65, frame[0:banner_h, :], 0.35, 0)
    line1 = (
        f"t={_format_clock(decision.t)}  state={decision.state.upper()}  "
        f"focus={decision.focus}  zoom={decision.zoom:.2f}x  conf={decision.confidence:.2f}"
    )
    line2 = f"why: {decision.reason}"
    cv2.putText(frame, line1, (10, int(banner_h * 0.42)), font, font_scale, color, thickness)
    cv2.putText(frame, line2, (10, int(banner_h * 0.85)), font, font_scale, (235, 235, 235), thickness)
    if decision.state in GOAL_STATES:
        cv2.putText(frame, "GOAL!", (frame_w - int(180 * font_scale * 2), int(banner_h * 0.7)),
                    font, font_scale * 1.8, (60, 60, 240), thickness + 2)
    return frame


def annotate_zoomed_banner(frame: np.ndarray, decision: CameraDecision) -> np.ndarray:
    """Small status strip at the bottom of a follow-camera frame."""
    cv2 = _import_cv2()
    frame_h, frame_w = frame.shape[:2]
    banner_h = max(22, int(frame_h * 0.05))
    y0 = frame_h - banner_h
    overlay = frame[y0:frame_h, :].copy()
    overlay[:] = (20, 20, 20)
    frame[y0:frame_h, :] = cv2.addWeighted(overlay, 0.6, frame[y0:frame_h, :], 0.4, 0)
    font_scale = max(0.4, frame_w / 2600.0)
    text = f"{_format_clock(decision.t)} | {decision.state} | {decision.reason}"
    cv2.putText(frame, text[:160], (8, frame_h - int(banner_h * 0.3)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, _state_color(decision.state),
                max(1, frame_w // 800))
    return frame


# ---------------------------------------------------------------------------
# ffmpeg capability probes (cached: they spawn processes)
# ---------------------------------------------------------------------------


def _run_quiet(cmd: List[str], timeout: float = 30.0) -> Tuple[int, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return result.returncode, (result.stdout or "") + (result.stderr or "")


@lru_cache(maxsize=None)
def ffmpeg_filter_available(name: str) -> bool:
    if not ffmpeg_available():
        return False
    code, out = _run_quiet([ffmpeg_exe(), "-hide_banner", "-filters"])
    return code == 0 and any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


@lru_cache(maxsize=None)
def encoder_usable(encoder: str) -> bool:
    """Listed AND actually opens (NVENC is listed in builds without a GPU)."""
    if not _ffmpeg_encoder_available(encoder):
        return False
    code, out = _run_quiet([
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "color=c=black:s=256x144:r=10:d=0.3", "-pix_fmt", "yuv420p",
        "-c:v", encoder, "-f", "null", "-",
    ])
    if code != 0:
        LOGGER.info("encoder %s not usable here: %s", encoder, out.strip()[:200])
    return code == 0


@lru_cache(maxsize=None)
def hwaccel_usable(name: str) -> bool:
    if not ffmpeg_available():
        return False
    code, out = _run_quiet([ffmpeg_exe(), "-hide_banner", "-hwaccels"])
    if code != 0 or name not in out.split():
        return False
    code, out = _run_quiet([
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-init_hw_device", f"{name}=probe",
        "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.04", "-f", "null", "-",
    ])
    if code != 0:
        LOGGER.info("hwaccel %s not usable here: %s", name, out.strip()[:200])
    return code == 0


def select_encoders(encoder: str = "auto") -> List[str]:
    """Encoder fallback chain for ``encoder`` ('auto' or a codec name)."""
    pref = str(encoder or "auto").strip().lower()
    chain = list(ENCODER_CHAIN) if pref == "auto" else [pref] + [e for e in ENCODER_CHAIN if e != pref]
    usable = [e for e in chain if encoder_usable(e)]
    if not usable and _ffmpeg_encoder_available("mpeg4"):
        usable = ["mpeg4"]
    return usable


def select_hwaccel(hwaccel: Optional[str] = "auto") -> Optional[str]:
    pref = str(hwaccel or "none").strip().lower()
    if pref in {"none", "off", "false", "cpu", ""}:
        return None
    candidates = list(HWACCEL_CHAIN) if pref == "auto" else [pref]
    for name in candidates:
        if hwaccel_usable(name):
            return name
    return None


def _encoder_args(encoder: str, output_size: Tuple[int, int], fps: float) -> List[str]:
    pixels = max(1, output_size[0] * output_size[1])
    scale = pixels / (1920.0 * 1080.0)
    gop = str(max(1, int(round(2.0 * fps))))  # 2 s GOP: clean stream-copy cuts later
    if encoder == "h264_nvenc":
        maxrate = max(4, int(round(20 * scale)))
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "21",
                "-b:v", "0", "-maxrate", f"{maxrate}M", "-bufsize", f"{2 * maxrate}M",
                "-profile:v", "high", "-g", gop]
    if encoder == "h264_videotoolbox":
        rate = max(3, int(round(12 * scale)))
        return ["-c:v", "h264_videotoolbox", "-b:v", f"{rate}M", "-maxrate", f"{int(rate * 1.6)}M",
                "-profile:v", "high", "-allow_sw", "1", "-g", gop]
    if encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "faster", "-crf", "20", "-g", gop]
    return ["-c:v", encoder, "-q:v", "3", "-g", gop]


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


def _stream_rotation(stream: Dict[str, object]) -> int:
    """Clockwise display rotation (0..359) from the ``rotate`` tag or display matrix."""
    tags = stream.get("tags") or {}
    if isinstance(tags, dict) and "rotate" in tags:
        try:
            return int(float(tags["rotate"])) % 360  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    for side in stream.get("side_data_list") or []:  # type: ignore[union-attr]
        if isinstance(side, dict) and "rotation" in side:
            try:
                return int(-float(side["rotation"])) % 360
            except (TypeError, ValueError):
                pass
    return 0


@lru_cache(maxsize=64)
def _probe_cached(path: str, mtime: float, size: int) -> Optional[VideoInfo]:
    """Probe in DISPLAY orientation: ffmpeg autorotates on decode, so crop
    rects (and the planner's frame size) live in the rotated pixel space."""
    del mtime, size
    code, out = _run_quiet([
        ffprobe_exe(), "-v", "error", "-show_streams", "-show_format", "-of", "json", path,
    ])
    if code == 0:
        try:
            data = json.loads(out[out.index("{"):])
            streams = data.get("streams", [])
            video = next((s for s in streams if s.get("codec_type") == "video"
                          and not (s.get("disposition") or {}).get("attached_pic")), None)
            if video is not None:
                fps = 0.0
                for key in ("avg_frame_rate", "r_frame_rate"):
                    num, _, den = str(video.get(key, "0/1")).partition("/")
                    try:
                        fps = float(num) / float(den or 1)
                    except (ValueError, ZeroDivisionError):
                        fps = 0.0
                    if fps > 0:
                        break
                width, height = int(video.get("width", 0) or 0), int(video.get("height", 0) or 0)
                if _stream_rotation(video) % 180 == 90:
                    width, height = height, width
                return VideoInfo(
                    width=width, height=height, fps=fps,
                    duration=float(data.get("format", {}).get("duration", 0.0) or 0.0),
                    has_audio=any(s.get("codec_type") == "audio" for s in streams),
                )
        except (ValueError, KeyError, TypeError):
            pass
    try:
        cv2 = _import_cv2()
    except RuntimeError:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        return None
    return VideoInfo(w, h, fps, frames / fps if fps > 0 else 0.0, has_audio=False)


def probe_video(path: str) -> Optional[VideoInfo]:
    p = Path(path)
    if not p.is_file():
        return None
    st = p.stat()
    return _probe_cached(str(p.resolve()), st.st_mtime, st.st_size)


def _find_font() -> Optional[str]:
    explicit = os.getenv("VH_SCOREBUG_FONT", "").strip()
    for candidate in ([explicit] if explicit else []) + list(_FONT_CANDIDATES):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Rect scaling
# ---------------------------------------------------------------------------


def _scale_rects(rects: np.ndarray, plan_size: Tuple[int, int], video_size: Tuple[int, int],
                 output_size: Tuple[int, int]) -> np.ndarray:
    """Map plan-space rects onto a video of a different size (e.g. a proxy)."""
    pw, ph = plan_size
    vw, vh = video_size
    if (pw, ph) == (vw, vh):
        return rects
    sx, sy = vw / float(pw), vh / float(ph)
    aspect = output_size[1] / float(output_size[0])
    out = np.empty_like(rects)
    max_w, max_h = vw - vw % 2, vh - vh % 2
    for i, (x, y, w, h) in enumerate(rects):
        nw = max(2, min(max_w, _even(w * sx)))
        nh = max(2, min(max_h, _even(nw * aspect)))
        cx, cy = (x + w / 2.0) * sx, (y + h / 2.0) * sy
        nx = min(max(0, _even(cx - nw / 2.0)), max(0, vw - nw))
        ny = min(max(0, _even(cy - nh / 2.0)), max(0, vh - nh))
        out[i] = (nx, ny, nw, nh)
    return out


# ---------------------------------------------------------------------------
# ffmpeg engine
# ---------------------------------------------------------------------------


class _Progress:
    """Thread-safe aggregation of per-chunk ffmpeg ``-progress`` output."""

    def __init__(self, total: int, callback: Optional[RenderProgressCallback],
                 info_callback: Optional[ProgressInfoCallback],
                 cancel_event: Optional[threading.Event] = None) -> None:
        self.total = max(1, int(total))
        self.callback = callback
        self.info_callback = info_callback
        self.cancel_event = cancel_event
        #: Set on cancel or on the first chunk failure: nothing new may start.
        self.stop = threading.Event()
        self.done: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.last_emit = 0.0

    def cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def check(self) -> None:
        """Raise when the render must stop (checked before every ffmpeg spawn
        and on every progress line, which kills the running ffmpeg)."""
        if self.cancelled():
            self.stop.set()
            raise RenderCancelled("Render canceled")
        if self.stop.is_set():
            raise _RenderStopped("render stopped after another chunk failed")

    def update(self, key: str, frames: int, final: bool = False) -> None:
        self.check()
        with self.lock:
            self.done[key] = max(self.done.get(key, 0), int(frames))
            written = min(self.total, sum(self.done.values()))
            now = time.monotonic()
            if not final and written < self.total and (now - self.last_emit) < 1.0:
                return
            self.last_emit = now
        elapsed = max(1e-6, now - self.started)
        fps = written / elapsed
        if self.callback is not None:
            self.callback(written, self.total)
        if self.info_callback is not None:
            self.info_callback({
                "frame": written, "total": self.total, "fps": round(fps, 2),
                "eta_s": round((self.total - written) / fps, 1) if fps > 0 else -1.0,
                "elapsed_s": round(elapsed, 2),
            })


def _run_ffmpeg(cmd: List[str], cwd: Path, on_frames: Optional[Callable[[int], None]] = None,
                stall_timeout_s: Optional[float] = None) -> Tuple[int, str]:
    """Run ffmpeg with ``-progress pipe:1``; stderr goes to a temp file so a
    chatty encoder can never block on a full pipe. A watchdog kills the
    process when no progress line arrives for ``stall_timeout_s``."""
    with tempfile.TemporaryFile() as stderr_sink:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=stderr_sink, text=True, bufsize=1)
        last_seen = [time.monotonic()]
        stalled = [False]
        done = threading.Event()

        def _watchdog() -> None:
            while not done.wait(1.0):
                if stall_timeout_s and time.monotonic() - last_seen[0] > stall_timeout_s:
                    stalled[0] = True
                    proc.kill()
                    return

        watcher = threading.Thread(target=_watchdog, daemon=True)
        watcher.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                last_seen[0] = time.monotonic()
                key, _, value = line.strip().partition("=")
                if key == "frame" and on_frames is not None:
                    try:
                        on_frames(int(value))
                    except ValueError:
                        pass
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            done.set()
        code = proc.wait()
        stderr_sink.seek(0)
        stderr = stderr_sink.read().decode("utf-8", errors="ignore")
    if stalled[0]:
        stderr = f"ffmpeg stalled (no progress for {stall_timeout_s:.0f}s) and was killed. " + stderr
        code = code or -9
    return code, stderr


@dataclass
class _Chunk:
    index: int
    first: int
    last: int  # exclusive
    ss: float
    to: float
    origin_frames: float


def _plan_chunks(n: int, fps: float, t0_src: float, chunk_seconds: float) -> List[_Chunk]:
    chunk_n = max(1, int(round(max(1.0, chunk_seconds) * fps)))
    bounds = list(range(0, n, chunk_n)) + [n]
    # Fold a short tail into the previous chunk.
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < chunk_n // 4:
        bounds.pop(-2)
    chunks: List[_Chunk] = []
    half = 0.5 / fps
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        ss = max(0.0, t0_src + a / fps - half)
        to = t0_src + b / fps - half
        origin = (t0_src + a / fps - ss) * fps
        chunks.append(_Chunk(k, a, b, ss, to, origin))
    return chunks


FILTER_MODES: Tuple[str, ...] = ("crop", "scale")


def _scale_mode_params(rects: np.ndarray, centers: np.ndarray, video_size: Tuple[int, int],
                       output_size: Tuple[int, int]) -> np.ndarray:
    """Per-frame ``[scaled_w, scaled_h, crop_x, crop_y]`` for the scale mode.

    The source is scaled by ``s = out_w / crop_w`` (crop_w = the held, even
    integer crop width) and the output window is placed at the planner's
    float center, rounded to whole output pixels.
    """
    vw, vh = video_size
    ow, oh = output_size
    out = np.empty((len(rects), 4), dtype=np.int64)
    for i in range(len(rects)):
        w = max(2, int(rects[i, 2]))
        sc = ow / float(w)
        sw = max(ow, _even(vw * sc))
        sh = max(oh, _even(vh * sc))
        sx, sy = sw / float(vw), sh / float(vh)
        x = int(round(float(centers[i, 0]) * sx - ow / 2.0))
        y = int(round(float(centers[i, 1]) * sy - oh / 2.0))
        out[i] = (sw, sh, min(max(0, x), sw - ow), min(max(0, y), sh - oh))
    return out


def _script_lines(mode: str, rows: np.ndarray, fps: float, origin_frames: float) -> List[str]:
    """Per-chunk sendcmd script: only fields that changed are sent.

    A ``scale``/``crop`` size command re-initialises the scaler even when the
    value is unchanged, so size commands are only emitted on real changes and
    pure pans send ``crop x``/``crop y`` only.
    """
    size_target = "crop" if mode == "crop" else "scale"
    names = (f"{size_target} w", f"{size_target} h", "crop x", "crop y")
    if mode == "crop":
        rows = np.asarray(rows)[:, [2, 3, 0, 1]]  # x,y,w,h -> w,h,x,y
    lines: List[str] = []
    prev: Optional[Tuple[int, ...]] = None
    for j in range(len(rows)):
        cur = tuple(int(v) for v in rows[j])
        if cur == prev:
            continue
        t = max(0.0, (j + origin_frames - 0.5) / fps) if j > 0 else 0.0
        cmds = [f"{name} {value}" for k, (name, value) in enumerate(zip(names, cur))
                if prev is None or prev[k] != value]
        lines.append(f"{t:.4f} " + ", ".join(cmds) + ";")
        prev = cur
    return lines


@lru_cache(maxsize=None)
def _scale_threads_option() -> str:
    """``:threads=1`` when supported. swscale's slice threading rebuilds a
    thread pool on every re-init (each zoom step): measured 4.5 s -> 1.8 s
    for 150 4K frames with a changing crop size, and faster at steady zoom
    too. Decode/encode threads provide the parallelism."""
    if not ffmpeg_available():
        return ""
    code, _ = _run_quiet([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                          "color=s=64x36:d=0.1", "-vf", "scale=32:18:threads=1", "-f", "null", "-"])
    return ":threads=1" if code == 0 else ""


def _build_filter(mode: str, script: str, first: Sequence[int], output_size: Tuple[int, int],
                  extra: Sequence[str]) -> str:
    out_w, out_h = output_size
    a, b, c, d = (int(v) for v in first)
    if mode == "crop":
        parts = [
            f"sendcmd=f={script}",
            f"crop=w={c}:h={d}:x={a}:y={b}",
            # null keeps the pre-command link size so scale notices size
            # changes (without it ffmpeg 6.1 hangs).
            "null",
            f"scale={out_w}:{out_h}:flags=bicubic{_scale_threads_option()}",
            "format=yuv420p",
        ]
    else:
        parts = [
            f"sendcmd=f={script}",
            f"scale=w={a}:h={b}:flags=bicubic:eval=frame{_scale_threads_option()}",
            f"crop=w={out_w}:h={out_h}:x={c}:y={d}:exact=1",
            "format=yuv420p",
        ]
    parts.extend(extra)
    # The seek point is half a frame before the first frame; rebase so the
    # first frame is t=0 (otherwise the CFR muxer pads a duplicate frame).
    parts.append("setpts=PTS-STARTPTS")
    return ",".join(parts)


def _render_ffmpeg(
    *,
    video_path: str,
    output_path: Path,
    plan: CameraPlan,
    output_size: Tuple[int, int],
    include_audio: bool,
    scorebug: Optional[ScorebugRenderer],
    encoder: str,
    hwaccel: Optional[str],
    source_time_offset: float,
    chunk_seconds: float,
    workers: Optional[int],
    progress: _Progress,
    filter_mode: str = "crop",
    stall_timeout_s: Optional[float] = 120.0,
) -> Tuple[int, str]:
    info = probe_video(video_path)
    if info is None or info.width <= 0:
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(plan.fps)
    n = len(plan.decisions)
    video_size = (info.width, info.height)
    rects = _scale_rects(plan.get_crop_rects(), plan.frame_size, video_size, output_size)
    vs = info.width / float(plan.frame_size[0])
    vsy = info.height / float(plan.frame_size[1])
    centers = np.array([(d.center_x * vs, d.center_y * vsy) for d in plan.decisions], dtype=np.float64)
    rows_by_mode = {
        "crop": rects,
        "scale": _scale_mode_params(rects, centers, video_size, output_size),
    }
    t0_src = float(source_time_offset) + float(plan.start_seconds)
    chunks = _plan_chunks(n, fps, t0_src, chunk_seconds)
    single = len(chunks) == 1
    with_audio = include_audio and info.has_audio
    encoders = select_encoders(encoder)
    if not encoders:
        raise RuntimeError("no usable H.264 encoder in this ffmpeg build")
    primary = filter_mode if filter_mode in FILTER_MODES else "crop"
    modes = [primary] + [m for m in FILTER_MODES if m != primary]
    hw_options = [hwaccel, None] if hwaccel else [None]
    attempts = [(mode, enc, hw) for mode in modes for enc in encoders for hw in hw_options]

    src = str(Path(video_path).resolve())
    out_dir = output_path.parent
    work = Path(tempfile.mkdtemp(prefix=f".{output_path.stem}_render_", dir=str(out_dir)))
    font_name: Optional[str] = None
    if scorebug is not None:
        if not ffmpeg_filter_available("drawtext"):
            LOGGER.warning("ffmpeg has no drawtext filter; rendering without the scorebug")
            scorebug = None
        else:
            font = _find_font()
            if font is not None:
                font_name = "scorebug_font" + Path(font).suffix.lower()
                shutil.copyfile(font, work / font_name)
    try:
        def _chunk_cmd(chunk: _Chunk, mode: str, enc: str, hw: Optional[str]) -> Tuple[List[str], Path]:
            script = f"cmds_{chunk.index:04d}_{mode}.txt"
            rows = rows_by_mode[mode][chunk.first:chunk.last]
            lines = _script_lines(mode, rows, fps, chunk.origin_frames)
            (work / script).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            extra: List[str] = []
            if scorebug is not None:
                plan_t0 = chunk.ss - float(source_time_offset)
                extra = scorebug.ffmpeg_filters(work, f"{chunk.index:04d}", plan_t0,
                                                chunk.to - chunk.ss, output_size, font_name)
            vf = _build_filter(mode, script, rows[0], output_size, extra)
            target = (work / "single.mp4") if single else (work / f"chunk_{chunk.index:04d}.mkv")
            cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
                   "-progress", "pipe:1", "-nostats"]
            if hw:
                cmd += ["-hwaccel", hw]
            if chunk.ss > 0:
                cmd += ["-ss", f"{chunk.ss:.6f}"]
            cmd += ["-to", f"{chunk.to:.6f}", "-i", src, "-vf", vf, "-map", "0:v:0"]
            if with_audio:
                # Drop the same half frame of audio so A/V stay aligned.
                lead = max(0.0, chunk.origin_frames / fps)
                cmd += ["-map", "0:a:0?", "-af", f"atrim=start={lead:.6f},asetpts=PTS-STARTPTS"]
                cmd += ["-c:a", "aac", "-b:a", "160k"] if single else ["-c:a", "pcm_s16le"]
            else:
                cmd += ["-an"]
            cmd += _encoder_args(enc, output_size, fps)
            if single:
                cmd += ["-movflags", "+faststart"]
            cmd += [target.name]
            return cmd, target

        def _render_chunk(chunk: _Chunk, combos) -> Tuple[str, str, Optional[str]]:
            last_err = ""
            progress.check()
            for mode, enc, hw in combos:
                progress.check()  # never spawn ffmpeg after a cancel / failure
                cmd, target = _chunk_cmd(chunk, mode, enc, hw)
                key = f"c{chunk.index}"
                code, err = _run_ffmpeg(cmd, work, lambda f, k=key: progress.update(k, f),
                                        stall_timeout_s=stall_timeout_s)
                if code == 0 and target.exists() and target.stat().st_size > 0:
                    progress.update(key, chunk.last - chunk.first, final=True)
                    return mode, enc, hw
                last_err = err.strip()[-600:]
                LOGGER.warning("chunk %d render (mode=%s, %s, hwaccel=%s) failed: %s",
                               chunk.index, mode, enc, hw, last_err)
                target.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg camera render failed for {video_path}: {last_err}")

        def _render_parallel(rest: List[_Chunk], combos, n_workers: int) -> None:
            """Chunks on a pool. The first failure (or a cancel) sets the stop
            flag in the failing worker: queued chunks are dropped
            (``cancel_futures``) or refuse to start, and running ffmpegs are
            killed at their next progress line."""

            def _guarded(chunk: _Chunk):
                try:
                    return _render_chunk(chunk, combos)
                except BaseException:
                    progress.stop.set()
                    raise

            pool = ThreadPoolExecutor(max_workers=n_workers)
            errors: List[BaseException] = []
            try:
                futures = [pool.submit(_guarded, c) for c in rest]
                for future in futures:
                    # Not as_completed: futures cancelled by shutdown() never
                    # notify its waiters; result() does see the cancel.
                    try:
                        future.result()
                    except CancelledError:
                        continue
                    except BaseException as exc:  # noqa: BLE001
                        errors.append(exc)
                        pool.shutdown(wait=False, cancel_futures=True)
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
            if progress.cancelled():
                raise RenderCancelled("Render canceled")
            real = [e for e in errors if not isinstance(e, (_RenderStopped, RenderCancelled))]
            if real or errors:
                raise (real or errors)[0]

        # The first chunk picks the working mode/encoder/hwaccel combination.
        chosen = _render_chunk(chunks[0], attempts)
        rest = chunks[1:]
        if rest:
            combos = [chosen] + [c for c in attempts if c != chosen]
            n_workers = max(1, int(workers or _default_workers(chosen[1], chosen[2])))
            if n_workers == 1:
                for chunk in rest:
                    _render_chunk(chunk, combos)
            else:
                _render_parallel(rest, combos, n_workers)

        progress.check()
        final_tmp = out_dir / f".{output_path.stem}.partial{output_path.suffix or '.mp4'}"
        if single:
            os.replace(work / "single.mp4", final_tmp)
        else:
            listing = "".join(
                f"file 'chunk_{c.index:04d}.mkv'\nduration {(c.last - c.first) / fps:.6f}\n" for c in chunks
            )
            (work / "chunks.txt").write_text(listing, encoding="utf-8", newline="\n")
            cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
                   "-f", "concat", "-safe", "0", "-i", "chunks.txt", "-map", "0:v:0"]
            cmd += (["-map", "0:a:0?", "-c:a", "aac", "-b:a", "160k"] if with_audio else ["-an"])
            cmd += ["-c:v", "copy", "-movflags", "+faststart", str(final_tmp.resolve())]
            code, err = _run_ffmpeg(cmd, work, stall_timeout_s=stall_timeout_s)
            if code != 0 or not final_tmp.exists():
                raise RuntimeError(f"ffmpeg concat failed: {err.strip()[-600:]}")
        os.replace(final_tmp, output_path)
        mode, enc, hw = chosen
        return n, f"{enc}{'+' + hw if hw else ''} ({mode} mode, {len(chunks)} chunk(s))"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _default_workers(encoder: str, hwaccel: Optional[str]) -> int:
    cpus = os.cpu_count() or 2
    if encoder in {"h264_nvenc", "h264_videotoolbox"} or hwaccel:
        return 2
    return max(1, min(4, cpus // 6))


# ---------------------------------------------------------------------------
# python engine (fallback) + debug wide
# ---------------------------------------------------------------------------


def _open_pipe_writer(output_path: Path, frame_size: Tuple[int, int], fps: float,
                      encoder: str, stderr_sink: IO[bytes]) -> subprocess.Popen:
    frame_w, frame_h = frame_size
    cmd = [
        ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{frame_w}x{frame_h}", "-r", f"{fps:.6f}",
        "-i", "pipe:0", "-an", *_encoder_args(encoder, (frame_w, frame_h), fps),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_sink)


def _encode_frames(frames, temp_file: Path, size: Tuple[int, int], fps: float, encoder: str,
                   on_frame: Callable[[int], None]) -> Tuple[int, str]:
    """Encode an iterator of BGR frames; ffmpeg pipe first, OpenCV writer last."""
    cv2 = _import_cv2()
    encoders = select_encoders(encoder) if ffmpeg_available() else []
    if encoders:
        enc = encoders[0]
        with tempfile.TemporaryFile() as stderr_sink:
            process = _open_pipe_writer(temp_file, size, fps, enc, stderr_sink)
            written = 0
            try:
                for frame in frames:
                    assert process.stdin is not None
                    process.stdin.write(np.ascontiguousarray(frame).tobytes())
                    written += 1
                    on_frame(written)
            except BrokenPipeError:
                pass
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
            code = process.wait()
            stderr_sink.seek(0)
            err = stderr_sink.read().decode("utf-8", errors="ignore")
        if code == 0 and written > 0 and temp_file.exists():
            return written, enc
        raise RuntimeError(f"python-engine encode with {enc} failed: {err.strip()[-400:]}")
    writer = cv2.VideoWriter(str(temp_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {temp_file}")
    written = 0
    try:
        for frame in frames:
            writer.write(frame)
            written += 1
            on_frame(written)
    finally:
        writer.release()
    return written, "mp4v"


def _open_capture(path: str, seek_s: float):
    cv2 = _import_cv2()
    if not Path(path).is_file():
        raise RuntimeError(f"Could not open video: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    if seek_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, seek_s * 1000.0)
    return cap


def _finish_with_audio(source: str, temp_file: Path, out_file: Path, start_s: float, written: int,
                       fps: float, include_audio: bool) -> str:
    end_s = start_s + written / fps
    if include_audio and ffmpeg_available():
        if _mux_audio(str(source), str(temp_file), str(out_file), start_s, end_s):
            temp_file.unlink(missing_ok=True)
            return str(out_file.resolve())
    os.replace(temp_file, out_file)
    return str(out_file.resolve())


def _render_python(*, video_path: str, out_file: Path, plan: CameraPlan, output_size: Tuple[int, int],
                   include_audio: bool, overlay_banner: bool, scorebug_fn: Optional[ScorebugFn],
                   encoder: str, source_time_offset: float, progress: _Progress) -> Tuple[str, str]:
    cv2 = _import_cv2()
    start_src = float(source_time_offset) + plan.start_seconds
    cap = _open_capture(video_path, start_src)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or plan.frame_size[0])
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or plan.frame_size[1])
    rects = _scale_rects(plan.get_crop_rects(), plan.frame_size, (vw, vh), output_size)
    out_w, out_h = output_size

    def _frames():
        for i, decision in enumerate(plan.decisions):
            ok, frame = cap.read()
            if not ok:
                break
            x, y, w, h = (int(v) for v in rects[i])
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                crop = frame
            interp = cv2.INTER_AREA if (crop.shape[1] >= out_w and crop.shape[0] >= out_h) else cv2.INTER_CUBIC
            out = cv2.resize(crop, (out_w, out_h), interpolation=interp)
            if overlay_banner:
                annotate_zoomed_banner(out, decision)
            if scorebug_fn is not None:
                out = scorebug_fn(out, decision.t)
            yield out

    temp_file = out_file.with_name(f"{out_file.stem}_temp_video.mp4")
    temp_file.unlink(missing_ok=True)
    try:
        written, enc = _encode_frames(_frames(), temp_file, (out_w, out_h), plan.fps, encoder,
                                      lambda k: progress.update("py", k))
    finally:
        cap.release()
    if written <= 0 or not temp_file.exists() or temp_file.stat().st_size <= 0:
        temp_file.unlink(missing_ok=True)
        raise RuntimeError(f"Camera-plan render produced no frames for: {video_path}")
    progress.update("py", written, final=True)
    return _finish_with_audio(video_path, temp_file, out_file, start_src, written, plan.fps, include_audio), enc


def _scale_geometry(geometry: FieldGeometry, s: float) -> FieldGeometry:
    def _goal(g: GoalBox) -> GoalBox:
        return GoalBox(side=g.side, x1=g.x1 * s, y1=g.y1 * s, x2=g.x2 * s, y2=g.y2 * s)

    return FieldGeometry(
        x_min=geometry.x_min * s, x_max=geometry.x_max * s, y_min=geometry.y_min * s,
        y_max=geometry.y_max * s, left_goal=_goal(geometry.left_goal), right_goal=_goal(geometry.right_goal),
        frame_size=(int(round(geometry.frame_size[0] * s)), int(round(geometry.frame_size[1] * s))),
        source=geometry.source,
    )


def _scale_decision(d: CameraDecision, s: float) -> CameraDecision:
    if abs(s - 1.0) < 1e-9:
        return d
    return CameraDecision(
        index=d.index, t=d.t, center_x=d.center_x * s, center_y=d.center_y * s, zoom=d.zoom,
        state=d.state, focus=d.focus, reason=d.reason, confidence=d.confidence,
        ball_x=d.ball_x * s if d.ball_x is not None else None,
        ball_y=d.ball_y * s if d.ball_y is not None else None,
        ball_source=d.ball_source,
        target_x=d.target_x * s if d.target_x is not None else None,
        target_y=d.target_y * s if d.target_y is not None else None,
    )


def _render_debug_wide(*, source: str, out_file: Path, plan: CameraPlan, include_audio: bool,
                       geometry: Optional[FieldGeometry], max_debug_width: int, encoder: str,
                       seek_s: float, progress: _Progress) -> Tuple[str, str]:
    cv2 = _import_cv2()
    cap = _open_capture(source, seek_s)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or plan.frame_size[0])
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or plan.frame_size[1])
    out_w = min(vw, int(max_debug_width))
    out_h = int(round(vh * out_w / float(vw)))
    out_w -= out_w % 2
    out_h -= out_h % 2
    s = out_w / float(plan.frame_size[0])
    geo = _scale_geometry(geometry, s) if geometry is not None else None
    trail_frames = max(1, int(_BALL_TRAIL_SECONDS * plan.fps))

    def _frames():
        trail: List[Tuple[float, float]] = []
        for decision in plan.decisions:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            d = _scale_decision(decision, s)
            if d.ball_x is not None and d.ball_y is not None:
                trail.append((d.ball_x, d.ball_y))
            if len(trail) > trail_frames:
                del trail[:-trail_frames]
            yield annotate_wide_frame(frame, d, ball_trail=trail, geometry=geo)

    temp_file = out_file.with_name(f"{out_file.stem}_temp_video.mp4")
    temp_file.unlink(missing_ok=True)
    try:
        written, enc = _encode_frames(_frames(), temp_file, (out_w, out_h), plan.fps, encoder,
                                      lambda k: progress.update("debug", k))
    finally:
        cap.release()
    if written <= 0 or not temp_file.exists() or temp_file.stat().st_size <= 0:
        temp_file.unlink(missing_ok=True)
        raise RuntimeError(f"Camera-plan debug render produced no frames for: {source}")
    progress.update("debug", written, final=True)
    return _finish_with_audio(source, temp_file, out_file, seek_s, written, plan.fps, include_audio), enc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_camera_plan_video(
    *,
    video_path: str,
    output_path: str,
    plan: CameraPlan,
    include_audio: bool = True,
    debug_wide: bool = False,
    overlay_banner: bool = False,
    scorebug_fn: Optional[ScorebugFn] = None,
    geometry: Optional[FieldGeometry] = None,
    max_debug_width: int = 1280,
    progress_callback: Optional[RenderProgressCallback] = None,
    output_size: Optional[Tuple[int, int]] = None,
    encoder: str = "auto",
    hwaccel: Optional[str] = "auto",
    engine: str = "ffmpeg",
    source_time_offset: float = 0.0,
    debug_source_path: Optional[str] = None,
    debug_source_time_offset: float = 0.0,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    workers: Optional[int] = None,
    progress_info_callback: Optional[ProgressInfoCallback] = None,
    filter_mode: str = "crop",
    stall_timeout_s: Optional[float] = 120.0,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """Render ``plan`` to ``output_path`` and return its absolute path.

    ``video_path`` is decoded from ``source_time_offset + plan.start_seconds``
    (pass the original 4K file plus ``trim_offset_seconds`` to render
    straight from the source). If its frame size differs from
    ``plan.frame_size`` the crop rects are rescaled.
    ``output_size``: defaults to ``plan.output_size`` or 1080p-class (never
    larger than the source). ``encoder``: ``auto`` (h264_nvenc ->
    h264_videotoolbox -> libx264) or a codec name. ``hwaccel``: ``auto`` |
    ``cuda`` | ``videotoolbox`` | ``none``. ``engine``: ``ffmpeg`` | ``python``.
    ``debug_wide=True`` renders the annotated wide review video from
    ``debug_source_path`` (proxy, seeking ``debug_source_time_offset +
    plan.start_seconds``) or ``video_path``, at most ``max_debug_width`` wide.
    ``progress_callback(written, total)``; ``progress_info_callback`` gets
    ``{frame, total, fps, eta_s, elapsed_s}``. ``filter_mode``: ``crop``
    (default) or ``scale`` (see module docstring; the other mode is the
    automatic fallback). ``stall_timeout_s``: kill ffmpeg after this long
    without progress and try the next combination. ``cancel_event``: when
    set, the running ffmpeg is killed, no further chunk starts and
    :class:`RenderCancelled` is raised.
    """
    if not plan.decisions:
        raise RuntimeError("camera plan is empty")
    out_file = Path(output_path)
    ensure_dir(str(out_file.parent))
    started = time.monotonic()
    progress = _Progress(len(plan.decisions), progress_callback, progress_info_callback,
                         cancel_event=cancel_event)
    progress.check()

    if debug_wide:
        if debug_source_path:
            source, seek = debug_source_path, float(debug_source_time_offset) + plan.start_seconds
        else:
            source, seek = video_path, float(source_time_offset) + plan.start_seconds
        path, enc = _render_debug_wide(source=source, out_file=out_file, plan=plan,
                                       include_audio=include_audio, geometry=geometry,
                                       max_debug_width=max_debug_width, encoder=encoder,
                                       seek_s=seek, progress=progress)
        LOGGER.info("camera-plan debug render: %d frames with %s in %.1fs -> %s",
                    len(plan.decisions), enc, time.monotonic() - started, out_file.name)
        return path

    if not Path(video_path).is_file():
        raise RuntimeError(f"Could not open video: {video_path}")
    info = probe_video(video_path)
    frame_size = (info.width, info.height) if info is not None and info.width > 0 else plan.frame_size
    out_size = resolve_output_size(plan.frame_size, output_size or plan.output_size)
    if output_size is None and plan.output_size is None:
        out_size = resolve_output_size(frame_size, None)

    use_python = str(engine or "ffmpeg").strip().lower() == "python"
    scorebug: Optional[ScorebugRenderer] = None
    if scorebug_fn is not None:
        if isinstance(scorebug_fn, ScorebugRenderer):
            scorebug = scorebug_fn
        else:
            use_python = True
            LOGGER.info("custom scorebug callable needs per-frame Python; using the python engine")
    if overlay_banner:
        use_python = True
    if not use_python and not ffmpeg_available():
        LOGGER.warning("ffmpeg not found; falling back to the python render engine")
        use_python = True

    if use_python:
        path, enc = _render_python(video_path=video_path, out_file=out_file, plan=plan, output_size=out_size,
                                   include_audio=include_audio, overlay_banner=overlay_banner,
                                   scorebug_fn=scorebug_fn, encoder=encoder,
                                   source_time_offset=source_time_offset, progress=progress)
        LOGGER.info("camera-plan render (python engine): %d frames %dx%d with %s in %.1fs -> %s",
                    len(plan.decisions), out_size[0], out_size[1], enc, time.monotonic() - started,
                    out_file.name)
        return path

    frames, enc = _render_ffmpeg(
        video_path=video_path, output_path=out_file, plan=plan, output_size=out_size,
        include_audio=include_audio, scorebug=scorebug, encoder=encoder,
        hwaccel=select_hwaccel(hwaccel), source_time_offset=source_time_offset,
        chunk_seconds=chunk_seconds, workers=workers, progress=progress,
        filter_mode=filter_mode, stall_timeout_s=stall_timeout_s,
    )
    elapsed = time.monotonic() - started
    LOGGER.info("camera-plan render (ffmpeg engine): %d frames %dx%d with %s in %.1fs (%.1f fps) -> %s",
                frames, out_size[0], out_size[1], enc, elapsed, frames / max(elapsed, 1e-6), out_file.name)
    return str(out_file.resolve())
