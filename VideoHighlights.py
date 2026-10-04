"""
Soccer Highlight Agent (v2, proxy-first)
----------------------------------------

A local pipeline that turns one panoramic match recording into a game-camera
movie, highlight clips, a broadcast reel, events and player/team stats:

  1. proxy      ONE ffmpeg pass over the source -> proxy_<H>p.mp4 +
                audio_analysis.wav + thumbs/ (backend.services.frame_source)
  2. tracking   batched detection + multi-object tracking on the proxy
                (backend.services.tracking_engine) -> tracks.npz
  3. teams      per-track kit colours on proxy frames
  4. analysis   ball track, pitch calibration, goals, states, set pieces,
                cards, stats (analysis_*_stats.json), events
                (analysis_events.json) and the reel plan
  5. camera     camera planner v2 -> camera_decisions.jsonl, camera_crops.txt,
                camera_quality.json
  6. render     ffmpeg crop+scale straight from the source (the second and
                last decode of the source) -> full_follow_ball_zoom.mp4
  7. clips/reel cut from the finished movie -> highlight_NN.mp4,
                highlights_reel.mp4

Analysis-only runs stop after step 5 (one decode of the source). Re-runs
with ``reuse_tracking_from`` skip steps 1-2 entirely.

Usage examples:
    python VideoHighlights.py --video match.mp4 --out ./run --camera-mode follow_ball --render-full-follow-cam
    python VideoHighlights.py --video match.mp4 --profile fast --trim-start 45:00 --trim-end 1:30:00
    python VideoHighlights.py --video match.mp4 --out ./run2 --reuse-tracking ./run \\
        --camera-mode follow_player --focus-track-id 7 --render-full-follow-cam

Profiles (fast / balanced / quality) pick proxy height, detector, imgsz,
stride and output height; see docs/ARTIFACTS.md.
"""

import os
import sys
import argparse
import csv
import json
import logging
import threading
import inspect
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
import cv2
from tqdm import tqdm

from backend.services.follow_cam import ball_weight_for_mode, render_follow_cam_clip
from backend.services import camera_planner as _cp
from backend.services import camera_render as _cr
from backend.services import event_engine as _ee
from backend.services import frame_source as _fs
from backend.services import match_stats as _ms
from backend.services import pitch_calibration as _pc
from backend.services import tracking_engine as _te
from backend.services.tracking_types import TrackingResult
from backend.services.game_tracking import (
    FieldGeometry,  # noqa: F401  (legacy re-export)
    analyze_game_states,
    build_ball_track,
    detect_goal_events,
    detect_set_pieces,
    estimate_field_geometry,
    overlay_set_piece_states,
    summarize_states,
)
from backend.services.card_detection import detect_card_events, stopped_play_windows
from backend.services.team_classification import TeamConfig, assign_track_teams
from backend.services.match_report import generate_match_report
from backend.services.broadcast import build_broadcast_reel, compute_audio_envelope
from backend.services.event_clip_renderer import cut_clip_from_rendered, render_clip_ffmpeg
from backend.services.perf_profiles import resolve_job_config, resolve_model_path

try:  # camera smoothness report (camera workstream)
    from backend.services import camera_quality as _camera_quality
except Exception:  # pragma: no cover
    _camera_quality = None  # type: ignore[assignment]

# Legacy names kept importable for callers of the old module surface.
CameraPlan = _cp.CameraPlan
plan_camera = _cp.plan_camera
slice_plan = _cp.slice_plan
make_scorebug_renderer = _cr.make_scorebug_renderer
render_camera_plan_video = _cr.render_camera_plan_video

LOGGER = logging.getLogger("videohighlights")

_LOGGING_LOCK = threading.Lock()


class _CurrentStdout:
    """File-like proxy that always writes to the *current* sys.stdout.

    The GUI redirects sys.stdout to a fresh StringIO per run; binding the
    console handler to this proxy (instead of a snapshot of sys.stdout)
    keeps log output flowing to whatever stdout is active.
    """

    def write(self, text: str) -> int:
        return sys.stdout.write(text)

    def flush(self) -> None:
        try:
            sys.stdout.flush()
        except Exception:
            pass


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> Optional[logging.Handler]:
    """Configure pipeline logging for one run.

    ``debug=True`` prints every debug-level diagnostic to the console.
    ``log_file`` additionally captures the full DEBUG stream (with
    timestamps) regardless of the console level - useful for reviewing a run
    and for building training datasets.

    The console handler is created once and only its level is adjusted, and
    the per-run file handler is RETURNED so the caller can detach it when the
    run finishes (see :func:`teardown_run_logging`). Never removes handlers
    it did not create - concurrent jobs in one process (API worker threads)
    must not strip each other's log files mid-run.
    """
    with _LOGGING_LOCK:
        root = logging.getLogger("videohighlights")
        root.setLevel(logging.DEBUG)
        root.propagate = False

        console = next(
            (h for h in root.handlers if getattr(h, "_vh_console", False)), None
        )
        if console is None:
            console = logging.StreamHandler(_CurrentStdout())
            console._vh_console = True  # type: ignore[attr-defined]
            console.setFormatter(logging.Formatter("[%(levelname).1s] %(message)s"))
            root.addHandler(console)
        console.setLevel(logging.DEBUG if debug else logging.INFO)

        file_handler: Optional[logging.Handler] = None
        if log_file:
            log_dir = os.path.dirname(os.path.abspath(log_file))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            )
            root.addHandler(file_handler)
        return file_handler


def teardown_run_logging(file_handler: Optional[logging.Handler]) -> None:
    """Detach and close a run's file handler (prevents fd leaks and, on
    Windows, lingering locks on the output directory)."""
    if file_handler is None:
        return
    with _LOGGING_LOCK:
        root = logging.getLogger("videohighlights")
        root.removeHandler(file_handler)
        try:
            file_handler.close()
        except Exception:
            pass


# --- moviepy is only used by the optional legacy spotlight overlay ---


def _import_moviepy():
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips  # type: ignore
    except ImportError:
        try:
            # moviepy 2.x has a different import structure
            from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
            from moviepy.video.compositing.CompositeVideoClip import (  # type: ignore
                concatenate_videoclips,
            )
        except ImportError as exc:
            raise RuntimeError(
                "moviepy is required for clip export. Install with: pip install moviepy"
            ) from exc
    return VideoFileClip, concatenate_videoclips


@dataclass
class TrackPoint:
    t: float  # seconds
    xy: Tuple[float, float]  # center x,y in pixels
    bbox: Optional[Tuple[float, float, float, float]] = None


FOLLOW_CAM_MODES = {"wide", "follow_player", "follow_action", "follow_ball"}
ProgressCallback = Callable[[str, float, str, Optional[Dict[str, object]]], None]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def emit_progress(
    progress_callback: Optional[ProgressCallback],
    stage: str,
    progress: float,
    message: str,
    data: Optional[Dict[str, object]] = None,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(stage, max(0.0, min(1.0, float(progress))), message, data or {})
    except Exception as exc:
        print(f"[warn] Progress callback failed: {exc}")


def parse_time(time_str: str) -> float:
    """Parse time string to seconds. Supports formats: seconds (123), MM:SS (12:30), HH:MM:SS (1:23:45)"""
    if not time_str:
        return 0.0

    time_str = time_str.strip()

    # Try parsing as plain seconds first
    try:
        return float(time_str)
    except ValueError:
        pass

    # Parse as time format (MM:SS or HH:MM:SS)
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS
        try:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}. Use MM:SS, HH:MM:SS, or seconds")
    elif len(parts) == 3:  # HH:MM:SS
        try:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}. Use MM:SS, HH:MM:SS, or seconds")
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use MM:SS, HH:MM:SS, or seconds")


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"




def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2]."""
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = areaA + areaB - inter
    return float(inter / union) if union > 0 else 0.0


def robust_threshold(series: np.ndarray, k: float = 3.0) -> float:
    """Median + k * MAD as a robust outlier/highlight threshold."""
    if len(series) == 0:
        return float('inf')
    med = np.median(series)
    mad = np.median(np.abs(series - med)) + 1e-9
    return med + k * mad


def merge_intervals(intervals: List[Tuple[float, float]], min_gap: float = 0.75) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s - last_e <= min_gap:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def _interval_overlap_seconds(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _pick_event_type(
    has_speed_signal: bool,
    has_audio_signal: bool,
    requested_targets: List[str],
) -> str:
    allowed_types = {
        "goal",
        "shot",
        "corner_kick",
        "penalty_kick",
        "free_kick",
        "goal_kick",
        "kickoff",
        "foul",
        "save",
    }
    targets = [item for item in requested_targets if item in allowed_types]
    if len(set(targets)) == 1:
        return targets[0]

    if has_speed_signal and has_audio_signal:
        for candidate in ("goal", "shot", "penalty_kick", "save"):
            if candidate in targets:
                return candidate
        return "goal"

    if has_speed_signal:
        for candidate in ("shot", "goal", "penalty_kick", "corner_kick", "free_kick", "save"):
            if candidate in targets:
                return candidate
        return "shot"

    if has_audio_signal:
        for candidate in ("foul", "kickoff", "goal", "corner_kick"):
            if candidate in targets:
                return candidate
        return "foul"

    if targets:
        return targets[0]
    return "shot"


def build_analysis_bookmarks(
    original_intervals: List[Tuple[float, float]],
    speed_intervals: List[Tuple[float, float]],
    audio_intervals: List[Tuple[float, float]],
    requested_targets: List[str],
    live_manifest_path: Optional[str] = None,
    live_manifest_context: Optional[Dict[str, object]] = None,
    goal_events: Optional[List[Dict[str, object]]] = None,
    game_states: Optional[List[Dict[str, object]]] = None,
    card_events: Optional[List[Dict[str, object]]] = None,
    set_piece_events: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    """Build the bookmark table for a run.

    ``goal_events`` (dicts with ``t``/``side``/``confidence``/``reason`` in the
    same timebase as ``original_intervals``) upgrade any overlapping interval
    to a confirmed ``goal`` bookmark. ``game_states`` (dicts with
    ``start_s``/``end_s``/``state``) tag each bookmark with the game state at
    its center so tags explain what the game was doing.
    """
    bookmarks: List[Dict[str, object]] = []
    context = dict(live_manifest_context or {})
    goal_rows = list(goal_events or [])
    state_rows = list(game_states or [])
    card_rows = list(card_events or [])
    set_piece_rows = list(set_piece_events or [])

    def _row_within(rows: List[Dict[str, object]], key: str, start_s: float, end_s: float) -> Optional[Dict[str, object]]:
        for row in rows:
            if start_s <= float(row.get(key, -1.0)) <= end_s:
                return row
        return None

    def _state_label_at(t: float) -> Optional[str]:
        for row in state_rows:
            if float(row.get("start_s", 0.0)) <= t < float(row.get("end_s", 0.0)):
                return str(row.get("state"))
        return None

    def _goal_within(start_s: float, end_s: float) -> Optional[Dict[str, object]]:
        for row in goal_rows:
            if start_s <= float(row.get("t", -1.0)) <= end_s:
                return row
        return None

    def _write_live_manifest() -> None:
        if not live_manifest_path:
            return
        payload = dict(context)
        payload["bookmarks"] = list(bookmarks)
        payload["stats"] = dict(payload.get("stats", {}))
        payload["stats"]["bookmark_count"] = len(bookmarks)
        try:
            with open(live_manifest_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            pass

    _write_live_manifest()
    for idx, (start_s, end_s) in enumerate(original_intervals, start=1):
        speed_overlap = sum(_interval_overlap_seconds((start_s, end_s), interval) for interval in speed_intervals)
        audio_overlap = sum(_interval_overlap_seconds((start_s, end_s), interval) for interval in audio_intervals)
        has_speed_signal = speed_overlap > 0.0
        has_audio_signal = audio_overlap > 0.0

        event_type = _pick_event_type(has_speed_signal, has_audio_signal, requested_targets)
        confidence = 0.45
        if has_speed_signal:
            confidence += 0.22
        if has_audio_signal:
            confidence += 0.18
        if requested_targets and event_type in requested_targets:
            confidence += 0.08
        confidence = float(min(0.99, round(confidence, 3)))

        sources: List[str] = []
        if has_speed_signal:
            sources.append("motion")
        if has_audio_signal:
            sources.append("audio")
        if not sources:
            sources.append("motion")

        occurred_at_s = (start_s + end_s) / 2.0
        duration_s = max(0.0, end_s - start_s)

        label = f"{event_type}_candidate"
        signals: Dict[str, object] = {
            "speed_overlap_s": round(speed_overlap, 3),
            "audio_overlap_s": round(audio_overlap, 3),
        }

        # Ball-tracking goal detection overrides the heuristic event type:
        # a flagged goal inside this interval makes it a goal bookmark. A
        # goal-only interval (no motion/audio overlap) must not carry the
        # default "motion" source label - its evidence is the ball track.
        goal_row = _goal_within(start_s, end_s)
        if goal_row is not None and not has_speed_signal and not has_audio_signal:
            sources = []
        if goal_row is not None:
            event_type = "goal"
            label = "goal_detected"
            occurred_at_s = float(goal_row.get("t", occurred_at_s))
            confidence = float(min(0.99, max(confidence, float(goal_row.get("confidence", 0.0)))))
            if "ball_tracking" not in sources:
                sources.append("ball_tracking")
            signals["goal_side"] = goal_row.get("side")
            signals["goal_reason"] = goal_row.get("reason")
            if has_audio_signal:
                # Crowd noise corroborating a detected goal.
                confidence = float(min(0.99, confidence + 0.05))

        # Referee cards outrank set pieces; both defer to detected goals.
        if goal_row is None:
            card_row = _row_within(card_rows, "t", start_s, end_s)
            if card_row is not None:
                event_type = str(card_row.get("kind") or "yellow_card")
                label = f"{event_type}_detected"
                occurred_at_s = float(card_row.get("t", occurred_at_s))
                confidence = float(min(0.99, max(confidence, float(card_row.get("confidence", 0.0)))))
                if not has_speed_signal and not has_audio_signal:
                    sources = []
                if "vision" not in sources:
                    sources.append("vision")
                signals["card_reason"] = card_row.get("reason")
                if card_row.get("crop_path"):
                    signals["card_crop_path"] = card_row.get("crop_path")
            else:
                sp_row = _row_within(set_piece_rows, "t_kick", start_s, end_s)
                if sp_row is not None and sp_row.get("kind") in {
                    "corner_kick", "free_kick", "penalty_kick", "goal_kick", "kickoff",
                }:
                    event_type = str(sp_row["kind"])
                    label = f"{event_type}_detected"
                    occurred_at_s = float(sp_row.get("t_kick", occurred_at_s))
                    confidence = float(min(0.99, max(confidence, 0.8)))
                    if not has_speed_signal and not has_audio_signal:
                        sources = []
                    if "ball_tracking" not in sources:
                        sources.append("ball_tracking")
                    signals["set_piece_side"] = sp_row.get("side")
                    signals["set_piece_reason"] = sp_row.get("reason")

        if not sources:
            sources.append("motion")

        game_state = _state_label_at(occurred_at_s)

        bookmarks.append(
            {
                "bookmark_id": f"bm_{idx:04d}",
                "index": idx,
                "event_type": event_type,
                "label": label,
                "confidence": confidence,
                "start_s": round(start_s, 3),
                "occurred_at_s": round(occurred_at_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(duration_s, 3),
                "sources": sources,
                "game_state": game_state,
                "signals": signals,
            }
        )
        _write_live_manifest()
    return bookmarks


def write_analysis_bookmark_files(
    output_dir: str,
    manifest: Dict[str, object],
) -> Tuple[str, str]:
    json_path = os.path.join(output_dir, "analysis_bookmarks.json")
    csv_path = os.path.join(output_dir, "analysis_bookmarks.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    bookmarks = list(manifest.get("bookmarks", []) or [])
    field_names = [
        "bookmark_id",
        "index",
        "event_type",
        "label",
        "confidence",
        "start_s",
        "occurred_at_s",
        "end_s",
        "duration_s",
        "sources",
        "game_state",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for item in bookmarks:
            row = {name: item.get(name) for name in field_names}
            sources = row.get("sources", [])
            if isinstance(sources, list):
                row["sources"] = ",".join(str(source) for source in sources)
            writer.writerow(row)

    return json_path, csv_path


def write_tracking_manifest(
    output_dir: str,
    payload: Dict[str, object],
) -> str:
    tracking_path = os.path.join(output_dir, "analysis_tracking.json")
    with open(tracking_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return tracking_path


def detect_review_candidate_intervals(
    times: np.ndarray,
    speed: np.ndarray,
    direction_changes: np.ndarray,
    pre: float,
    post: float,
    max_candidates: int = 3,
) -> List[Tuple[float, float]]:
    if len(times) == 0 or len(speed) == 0:
        return []

    speed_norm = (speed - speed.min()) / (speed.max() - speed.min() + 1e-9)
    if len(direction_changes) == 0:
        direction_norm = np.zeros_like(speed_norm)
    else:
        direction_norm = direction_changes / (np.pi + 1e-9)
        if len(direction_norm) != len(speed_norm):
            if len(direction_norm) > len(speed_norm):
                direction_norm = direction_norm[:len(speed_norm)]
            else:
                direction_norm = np.pad(direction_norm, (0, len(speed_norm) - len(direction_norm)), "constant")

    score = (0.75 * speed_norm) + (0.25 * direction_norm)
    if not np.any(score > 0.0):
        return []

    order = np.argsort(score)[::-1]
    selected_times: List[float] = []
    min_separation = max(8.0, float(pre + post))
    for idx in order:
        t = float(times[int(idx)])
        if all(abs(t - existing) >= min_separation for existing in selected_times):
            selected_times.append(t)
        if len(selected_times) >= max(1, int(max_candidates)):
            break

    intervals = [(max(0.0, t - pre), t + post) for t in sorted(selected_times)]
    merged = merge_intervals(intervals, min_gap=1.0)
    LOGGER.debug(f"Review candidate fallback intervals: {len(merged)}")
    return merged




def _trajectory_to_samples(traj: List[TrackPoint], time_offset_seconds: float = 0.0) -> List[Tuple[float, float, float]]:
    return [
        (float(point.t + time_offset_seconds), float(point.xy[0]), float(point.xy[1]))
        for point in traj
    ]


def _trajectory_to_manifest_points(
    traj: List[TrackPoint],
    trim_offset: float = 0.0,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for point in traj:
        item: Dict[str, float] = {
            "t": round(float(point.t + trim_offset), 3),
            "x": round(float(point.xy[0]), 3),
            "y": round(float(point.xy[1]), 3),
        }
        if point.bbox is not None:
            item.update(
                {
                    "x1": round(float(point.bbox[0]), 3),
                    "y1": round(float(point.bbox[1]), 3),
                    "x2": round(float(point.bbox[2]), 3),
                    "y2": round(float(point.bbox[3]), 3),
                }
            )
        rows.append(item)
    return rows



def write_full_follow_cam_video(
    video_path: str,
    interval: Tuple[float, float],
    out_dir: str,
    player_traj: List[TrackPoint],
    ball_traj: List[TrackPoint],
    camera_mode: str = "follow_action",
    zoom_factor: float = 1.6,
    track_time_offset_seconds: float = 0.0,
    progress_callback: Optional[ProgressCallback] = None,
) -> Optional[str]:
    start_s, end_s = interval
    if end_s <= start_s:
        print(f"[warn] Full follow-cam interval is empty: {start_s:.2f}s - {end_s:.2f}s")
        return None

    safe_mode = str(camera_mode or "follow_action").strip().lower()
    out_path = os.path.join(out_dir, f"full_{safe_mode}_zoom.mp4")
    ball_weight = ball_weight_for_mode(safe_mode)
    duration_s = float(end_s - start_s)
    print(
        f"[follow-cam] Rendering full zoom movie: {format_time(start_s)} - {format_time(end_s)} "
        f"({duration_s:.1f}s), mode={safe_mode}, zoom={zoom_factor:.2f}x"
    )

    def _render_progress(written_frames: int, total_frames: int) -> None:
        if total_frames <= 0:
            return
        fraction = min(1.0, max(0.0, float(written_frames) / float(total_frames)))
        emit_progress(
            progress_callback,
            "rendering_full_zoom",
            0.955 + (0.03 * fraction),
            "Rendering full zoom movie",
            {
                "written_frames": int(written_frames),
                "total_frames": int(total_frames),
                "camera_mode": safe_mode,
                "zoom_factor": round(float(zoom_factor), 3),
                "output_path": out_path,
            },
        )

    try:
        return render_follow_cam_clip(
            video_path=video_path,
            output_path=out_path,
            start_seconds=start_s,
            end_seconds=end_s,
            player_track=_trajectory_to_samples(player_traj, time_offset_seconds=track_time_offset_seconds),
            ball_track=_trajectory_to_samples(ball_traj, time_offset_seconds=track_time_offset_seconds),
            zoom_factor=zoom_factor,
            ball_weight=ball_weight,
            smooth_factor=0.24,
            include_audio=True,
            progress_callback=_render_progress,
        )
    except Exception as ex:
        print(f"[warn] Failed to render full follow-cam movie ({start_s:.1f}s - {end_s:.1f}s): {ex}")
        return None


def draw_single_spotlight_overlay(video_path: str, traj: List[TrackPoint], interval: Tuple[float, float],
                                   clip_num: int, out_dir: str, radius: int = 35) -> Optional[str]:
    """Draw spotlight overlay for a single clip (used for parallel processing)"""
    s, e = interval
    t_arr = np.array([p.t for p in traj])
    xy_arr = np.array([p.xy for p in traj])

    def pos_at(t: float) -> Tuple[int, int]:
        # nearest neighbor in time
        idx = int(np.argmin(np.abs(t_arr - t)))
        x, y = xy_arr[idx]
        return int(round(x)), int(round(y))

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for overlay {clip_num}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Seek to start
        seek_success = cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000.0)
        if not seek_success:
            print(f"[warn] Failed to seek to {s}s for overlay {clip_num}, starting from beginning")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        temp_path = os.path.join(out_dir, f"highlight_{clip_num:02d}_spotlight_temp.mp4")
        out_path = os.path.join(out_dir, f"highlight_{clip_num:02d}_spotlight.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"[warn] Could not open writer for {temp_path}, trying alternate codec...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        frames_needed = int((e - s) * fps)
        for _ in range(frames_needed):
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            cx, cy = pos_at(t)
            # Draw soft circle
            cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), radius + 6, (0, 0, 0), 2)
            writer.write(frame)

        writer.release()
        cap.release()

        # Add audio using moviepy
        try:
            VideoFileClip, _ = _import_moviepy()
            with VideoFileClip(video_path) as source_clip:
                with VideoFileClip(temp_path) as video_only:
                    # Extract audio from the same time interval
                    try:
                        audio_subclip = source_clip.subclip(s, e)
                    except AttributeError:
                        audio_subclip = source_clip.subclipped(s, e)
                    audio_clip = audio_subclip.audio
                    if audio_clip is not None:
                        final_clip = video_only.set_audio(audio_clip)
                        final_clip.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
                        final_clip.close()
                    else:
                        # No audio in source, just rename temp file
                        os.rename(temp_path, out_path)
            # Clean up temp file if it still exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return out_path
        except Exception as e:
            print(f"[warn] Could not add audio to overlay {clip_num}: {e}. Using video-only version.")
            if os.path.exists(temp_path):
                os.rename(temp_path, out_path)
            return out_path
    except Exception as ex:
        print(f"[warn] Failed to create overlay for clip {clip_num}: {ex}")
        return None


def draw_spotlight_overlay(video_path: str, traj: List[TrackPoint], intervals: List[Tuple[float, float]],
                           out_dir: str, radius: int = 35, max_workers: Optional[int] = None):
    """Draw spotlight overlays using parallel processing"""
    if max_workers is None:
        # Use up to 50% of CPU count for overlays (memory intensive)
        max_workers = max(2, int(multiprocessing.cpu_count() * 0.5))

    print(f"[performance] Rendering {len(intervals)} overlays using {max_workers} parallel workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all overlay rendering tasks
        future_to_clip = {
            executor.submit(draw_single_spotlight_overlay, video_path, traj, interval, k, out_dir, radius): k
            for k, interval in enumerate(intervals, start=1)
        }

        # Collect results as they complete with progress bar
        with tqdm(total=len(intervals), desc="Rendering overlays", unit="clip") as pbar:
            for future in as_completed(future_to_clip):
                clip_num = future_to_clip[future]
                try:
                    result = future.result()
                    if not result:
                        print(f"[warn] Failed to render overlay for clip {clip_num}")
                except Exception as exc:
                    print(f"[warn] Overlay {clip_num} generated an exception: {exc}")
                pbar.update(1)





# ---------------------------------------------------------------------------
# v2 pipeline: progress, cancel, helpers
# ---------------------------------------------------------------------------


class PipelineCancelled(RuntimeError):
    """Raised inside the pipeline when the run's cancel event is set."""


#: Stage order and weights of overall progress (docs/ARTIFACTS.md progress.json).
STAGE_PLAN: Tuple[Tuple[str, float], ...] = (
    ("proxy", 0.10),
    ("tracking", 0.42),
    ("teams", 0.03),
    ("analysis", 0.08),
    ("camera_plan", 0.02),
    ("render", 0.25),
    ("clips_reel", 0.10),
)
ANALYSIS_ONLY_STAGES = ("proxy", "tracking", "teams", "analysis", "camera_plan")


class ProgressTracker:
    """Maps per-stage fractions to overall progress and publishes it.

    Writes ``progress.json`` (atomically, at least every 2 s while the run is
    alive thanks to a heartbeat thread) and forwards every update to the
    legacy ``progress_callback(sub_stage, progress, message, data)`` so the
    API worker keeps its job log / cancel polling cadence.
    """

    def __init__(
        self,
        run_dir: str,
        *,
        analysis_only: bool = False,
        legacy_callback: Optional[ProgressCallback] = None,
        device: Optional[str] = None,
        write_interval_s: float = 2.0,
        heartbeat: bool = True,
    ) -> None:
        stages = [(n, w) for n, w in STAGE_PLAN if not analysis_only or n in ANALYSIS_ONLY_STAGES]
        total = sum(w for _, w in stages) or 1.0
        self.stages: List[Tuple[str, float]] = [(n, w / total) for n, w in stages]
        self.stage_names = [n for n, _ in self.stages]
        self.path = os.path.join(run_dir, "progress.json")
        self.legacy_callback = legacy_callback
        self.device = device
        self.write_interval_s = max(0.2, float(write_interval_s))
        self._lock = threading.RLock()
        self.started = time.monotonic()
        self.stage = "initializing"
        self.stage_index = -1
        self.stage_fraction = 0.0
        self.stage_started = self.started
        self.message = "Initializing"
        self.fps: Optional[float] = None
        self.stage_eta: Optional[float] = None
        self.stage_timings: Dict[str, float] = {}
        self.cancelled = False
        self.final_status: Optional[str] = None
        self._last_data: Dict[str, object] = {}
        self._last_write = 0.0
        self._last_legacy = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if heartbeat:
            self._thread = threading.Thread(target=self._heartbeat, name="vh-progress", daemon=True)
            self._thread.start()

    # -- computed values -------------------------------------------------
    def _weight(self, idx: int) -> float:
        return self.stages[idx][1] if 0 <= idx < len(self.stages) else 0.0

    def _done_weight(self) -> float:
        if self.stage_index < 0:
            return 0.0
        base = sum(w for _, w in self.stages[: self.stage_index])
        return base + self._weight(self.stage_index) * self.stage_fraction

    @property
    def progress(self) -> float:
        if self.final_status == "completed":
            return 1.0
        return max(0.0, min(1.0, self._done_weight()))

    def eta_s(self) -> Optional[float]:
        if self.final_status:
            return 0.0
        now = time.monotonic()
        done = self._done_weight()
        elapsed = now - self.started
        frac = self.stage_fraction
        cur_w = self._weight(self.stage_index)
        stage_rem: Optional[float] = None
        if self.stage_eta is not None and self.stage_eta >= 0:
            stage_rem = float(self.stage_eta)
        elif frac > 0.02:
            stage_el = now - self.stage_started
            stage_rem = stage_el * (1.0 - frac) / frac
        if done < 0.02:
            return round(stage_rem, 1) if stage_rem is not None and self.stage_index == len(self.stages) - 1 else None
        rate = elapsed / done  # seconds per unit of weight measured so far
        if stage_rem is None:
            stage_rem = cur_w * (1.0 - frac) * rate
        future_w = sum(w for _, w in self.stages[self.stage_index + 1:])
        return round(max(0.0, stage_rem + future_w * rate), 1)

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            idx = self.stage_index
            return {
                "stage": self.stage,
                "stage_index": idx,
                "stage_count": len(self.stages),
                "stages": list(self.stage_names),
                "stage_timings": {k: round(v, 3) for k, v in self.stage_timings.items()},
                "progress": round(self.progress, 4),
                "stage_progress": round(1.0 if self.final_status == "completed" else self.stage_fraction, 4),
                "eta_s": self.eta_s(),
                "elapsed_s": round(time.monotonic() - self.started, 2),
                "fps_processing": round(self.fps, 2) if self.fps is not None else None,
                "message": self.message,
                "device": self.device,
                "cancelled": bool(self.cancelled),
                "status": self.final_status or "running",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    # -- mutation --------------------------------------------------------
    def _close_stage(self) -> None:
        if 0 <= self.stage_index < len(self.stages) and self.stage in self.stage_names:
            self.stage_timings[self.stage] = self.stage_timings.get(self.stage, 0.0) + (
                time.monotonic() - self.stage_started)

    def start_stage(self, name: str, message: str = "") -> None:
        with self._lock:
            if self.stage == name and self.stage_index >= 0:
                self.message = message or self.message
            else:
                self._close_stage()
                if name in self.stage_names:
                    self.stage_index = self.stage_names.index(name)
                self.stage = name
                self.stage_fraction = 0.0
                self.stage_started = time.monotonic()
                self.fps = None
                self.stage_eta = None
                self.message = message or name
        self._publish(force=True)

    def update(self, fraction: Optional[float] = None, message: Optional[str] = None,
               data: Optional[Dict[str, object]] = None) -> None:
        with self._lock:
            if fraction is not None:
                try:
                    f = max(0.0, min(1.0, float(fraction)))
                except (TypeError, ValueError):
                    f = self.stage_fraction
                self.stage_fraction = max(self.stage_fraction, f)
            if message:
                self.message = str(message)
            if data:
                fps = data.get("fps_processing", data.get("fps"))
                if isinstance(fps, (int, float)) and fps > 0:
                    self.fps = float(fps)
                eta = data.get("eta_s")
                self.stage_eta = float(eta) if isinstance(eta, (int, float)) and eta >= 0 else None
                self._last_data = {k: v for k, v in data.items() if isinstance(v, (int, float, str, bool)) or v is None}
        self._publish(data=data)

    def module_callback(self, stage_name: str) -> ProgressCallback:
        """``(stage, fraction, message, data)`` callback for a module call."""

        def _cb(_stage: str, fraction: float, message: str, data: Optional[Dict[str, object]] = None) -> None:
            self.update(fraction, message, dict(data or {}))

        return _cb

    def finish(self, status: str, message: str) -> None:
        with self._lock:
            self._close_stage()
            self.final_status = status
            self.cancelled = status == "canceled"
            if status == "completed":
                self.stage_fraction = 1.0
            self.stage = status
            self.message = message
        self._publish(force=True)
        self.close()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None

    # -- publishing ------------------------------------------------------
    def _write(self) -> None:
        snap = self.snapshot()
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(snap, handle, indent=2)
            os.replace(tmp, self.path)
        except OSError as exc:  # pragma: no cover - disk issues must not kill a run
            LOGGER.debug("progress.json write failed: %s", exc)
        self._last_write = time.monotonic()

    def _legacy(self, data: Optional[Dict[str, object]] = None) -> None:
        if self.legacy_callback is None:
            return
        self._last_legacy = time.monotonic()
        payload: Dict[str, object] = dict(self._last_data)
        payload.update({k: v for k, v in (data or {}).items()
                        if isinstance(v, (int, float, str, bool)) or v is None})
        payload.update({"stage_progress": round(self.stage_fraction, 4), "eta_s": self.eta_s(),
                        "stage_index": self.stage_index, "stage_count": len(self.stages)})
        stage = self.stage if not self.final_status else ("canceled" if self.cancelled else self.final_status)
        emit_progress(self.legacy_callback, stage, self.progress, self.message, payload)

    def _publish(self, force: bool = False, data: Optional[Dict[str, object]] = None) -> None:
        now = time.monotonic()
        if force or now - self._last_write >= min(1.0, self.write_interval_s):
            self._write()
        self._legacy(data)

    def _heartbeat(self) -> None:
        while not self._stop.wait(0.5):
            now = time.monotonic()
            try:
                if now - self._last_write >= self.write_interval_s:
                    self._write()
                if now - self._last_legacy >= self.write_interval_s:
                    self._legacy()
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("progress heartbeat failed: %s", exc)


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise PipelineCancelled("Run canceled")


def _even(value: float) -> int:
    v = int(round(float(value)))
    return max(2, v - (v % 2))


def _tracker_kind(tracker_config: Optional[str]) -> str:
    return "botsort" if "botsort" in str(tracker_config or "").lower() else "bytetrack"


def _resolve_reuse_dir(reuse: Optional[str]) -> Optional[str]:
    """``reuse_tracking_from`` may be a run directory or a job id under the output root."""
    if not reuse:
        return None
    raw = str(reuse).strip()
    candidates = [raw]
    try:
        from backend.config import settings as _settings

        candidates.append(os.path.join(_settings.output_root, raw))
    except Exception:
        candidates.append(os.path.join("outputs", raw))
    for cand in candidates:
        if cand and os.path.isdir(cand) and TrackingResult.exists(cand):
            return os.path.abspath(cand)
    return None


def _find_proxy_file(run_dir: str, tracking: Optional[TrackingResult]) -> Optional[str]:
    if tracking is not None and tracking.processing_video_path and os.path.isfile(tracking.processing_video_path) \
            and os.path.basename(tracking.processing_video_path).startswith("proxy_"):
        return tracking.processing_video_path
    try:
        names = sorted(n for n in os.listdir(run_dir) if n.startswith("proxy_") and n.endswith(".mp4"))
    except OSError:
        return None
    return os.path.join(run_dir, names[-1]) if names else None


def _link_or_copy(src: str, dst: str) -> None:
    import shutil

    if os.path.abspath(src) == os.path.abspath(dst):
        return
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _normalize_corners(corners: object, frame_size: Tuple[int, int]) -> Optional[List[List[float]]]:
    try:
        pts = [[float(p[0]), float(p[1])] for p in list(corners)]  # type: ignore[arg-type]
    except Exception:
        try:
            pts = [[float(p["x"]), float(p["y"])] for p in list(corners)]  # type: ignore[index,arg-type]
        except Exception:
            return None
    if len(pts) != 4:
        return None
    if all(0.0 <= v <= 1.0 for p in pts for v in p):
        pts = [[p[0] * frame_size[0], p[1] * frame_size[1]] for p in pts]
    return pts


def _interactive_roi_from_proxy(proxy_path: str) -> Optional[Dict[str, float]]:
    """Legacy ``--select``: draw a box on the first PROXY frame (never the source)."""
    cap = cv2.VideoCapture(proxy_path)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    try:
        box = cv2.selectROI("Select your player (ENTER to confirm)", frame, showCrosshair=True)
        cv2.destroyAllWindows()
    except Exception as exc:
        print(f"[warn] Interactive selection unavailable ({exc}); using the longest track")
        return None
    x, y, w, h = [float(v) for v in box]
    if w <= 0 or h <= 0:
        return None
    fh, fw = frame.shape[:2]
    return {"x1_norm": x / fw, "y1_norm": y / fh, "x2_norm": (x + w) / fw, "y2_norm": (y + h) / fh}


def _apply_focus(tracking: TrackingResult, focus_track_id: Optional[int],
                 player_roi: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Set ``tracking.focus_track_id`` from an explicit id or ROI (reused tracks)."""
    if focus_track_id is None and not player_roi:
        return {"method": "unchanged", "track_id": tracking.focus_track_id}
    selector = getattr(_te, "_select_focus", None)
    if selector is not None:
        try:
            return selector(tracking, focus_roi=player_roi, focus_track_id=focus_track_id,
                            stride=max(1, int(tracking.vid_stride or 1)))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("focus selection failed: %s", exc)
    if focus_track_id is not None and int(focus_track_id) in tracking.players:
        tracking.focus_track_id = int(focus_track_id)
    return {"method": "track_id", "track_id": tracking.focus_track_id}


def _event_location_evidence(event_set: object, ball_track: object, calibration: object) -> None:
    """Add ``evidence.ball_xy`` (source px) and ``evidence.pitch_xy_m`` for the shot map."""
    for ev in getattr(event_set, "events", []) or []:
        ev_dict = ev.evidence if isinstance(getattr(ev, "evidence", None), dict) else None
        if ev_dict is None:
            continue
        xy: Optional[Tuple[float, float]] = None
        if "ball_xy" in ev_dict and ev_dict["ball_xy"]:
            try:
                bx, by = ev_dict["ball_xy"][:2]  # type: ignore[index]
                xy = (float(bx), float(by))
            except Exception:
                xy = None
        if xy is None and ball_track is not None:
            try:
                pos = ball_track.position_at(float(ev.t))  # type: ignore[attr-defined]
            except Exception:
                pos = None
            if pos is not None:
                xy = (float(pos[0]), float(pos[1]))
        if xy is None and getattr(ev, "location_px", None):
            xy = (float(ev.location_px[0]), float(ev.location_px[1]))
        if xy is None:
            continue
        ev_dict["ball_xy"] = [round(xy[0], 1), round(xy[1], 1)]
        if calibration is not None and "pitch_xy_m" not in ev_dict:
            try:
                pm = np.asarray(calibration.to_pitch([xy[0], xy[1]]), dtype=float).reshape(-1)  # type: ignore[attr-defined]
                if pm.size >= 2 and np.all(np.isfinite(pm[:2])):
                    ev_dict["pitch_xy_m"] = [round(float(pm[0]), 2), round(float(pm[1]), 2)]
            except Exception:
                pass


def _enforce_min_clip(plan: Dict[str, object], min_clip_s: float, duration_s: float) -> None:
    for clip in plan.get("clips") or []:  # type: ignore[union-attr]
        s, e = float(clip["t_start"]), float(clip["t_end"])
        if e - s < min_clip_s:
            e = min(duration_s, s + min_clip_s) if duration_s > 0 else s + min_clip_s
            if e - s < min_clip_s:
                s = max(0.0, e - min_clip_s)
            clip["t_start"], clip["t_end"] = round(s, 3), round(e, 3)
            clip["duration_s"] = round(e - s, 3)
    clips = list(plan.get("clips") or [])  # type: ignore[arg-type]
    plan["total_duration_s"] = round(sum(float(c["duration_s"]) for c in clips), 3)


DetectorFactory = Callable[[Dict[str, object]], object]

# Keys resolved through backend.services.perf_profiles (profile defaults).
_PROFILE_KWARGS = (
    "proxy_height", "inference_imgsz", "vid_stride", "yolo_model", "batch_size",
    "output_height", "debug_video", "ball_tiles", "tracker_config",
)


def process_video_highlights(
    video_path: str,
    output_dir: str,
    select_player: bool = False,
    pre_seconds: float = 2.0,
    post_seconds: float = 6.0,
    min_clip_duration: float = 4.0,
    no_audio: bool = False,
    overlay: bool = False,
    trim_start: Optional[float] = None,
    trim_end: Optional[float] = None,
    threads: Optional[int] = None,
    require_gpu: bool = False,
    speed_sensitivity: float = 2.0,
    audio_sensitivity: float = 2.0,
    focus_event_types: Optional[List[str]] = None,
    model_version: Optional[str] = None,
    analysis_only: bool = False,
    camera_mode: str = "wide",
    zoom_factor: float = 1.6,
    render_full_follow_cam: bool = False,
    player_roi: Optional[Dict[str, float]] = None,
    yolo_model: Optional[str] = None,
    tracker_config: Optional[str] = None,
    inference_imgsz: Optional[int] = None,
    detection_conf: float = 0.18,
    vid_stride: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
    debug: bool = False,
    log_file: Optional[str] = None,
    debug_video: Optional[bool] = None,
    dump_training_data: bool = False,
    goal_box_left: Optional[Dict[str, float]] = None,
    goal_box_right: Optional[Dict[str, float]] = None,
    detect_cards: bool = True,
    broadcast_reel: bool = True,
    scorebug: bool = True,
    team_left: str = "HOME",
    team_right: str = "AWAY",
    team_left_color: Optional[str] = None,
    team_right_color: Optional[str] = None,
    auto_detect_team_colors: bool = False,
    llm_report: bool = True,
    *,
    profile: Optional[str] = None,
    proxy_height: Optional[int] = None,
    output_height: Optional[int] = None,
    batch_size: Optional[object] = None,
    ball_tiles: Optional[bool] = None,
    use_tensorrt: Optional[bool] = None,
    device: Optional[str] = None,
    camera_style: Optional[str] = None,
    focus_track_id: Optional[int] = None,
    reuse_tracking_from: Optional[str] = None,
    pitch_corners: Optional[List[List[float]]] = None,
    reel_minutes: Optional[float] = None,
    reel_preset: Optional[str] = None,
    player_spotlight_reel: bool = False,
    cancel_event: Optional[threading.Event] = None,
    detector_factory: Optional[DetectorFactory] = None,
) -> bool:
    """Public pipeline entry point used by the CLI, GUI, and API worker.

    v2 (proxy-first): the source is decoded once by ``build_proxy`` (proxy +
    analysis audio + thumbnails), every analysis stage reads the proxy, and
    the source is decoded once more by the final ffmpeg render. Profile keys
    (``profile``, ``proxy_height``, ``inference_imgsz``, ``vid_stride``,
    ``yolo_model``, ``batch_size``, ``output_height``, ``debug_video``,
    ``ball_tiles``, ``tracker_config``) left as ``None`` come from the
    profile (``perf_profiles.resolve_job_config``).

    ``reuse_tracking_from``: a previous run directory or job id; its
    ``tracks.npz`` (and proxy) are reused, so no detection runs.
    ``cancel_event``: set it to stop the run (returns False, progress.json
    ``cancelled: true``). ``detector_factory(config) -> Detector`` replaces
    the YOLO detector (tests inject a ground-truth detector).

    Configures per-run logging, then delegates to the implementation; the
    run's log-file handler is always detached and closed on exit.
    """
    run_log_handler = setup_logging(debug=debug, log_file=log_file)
    try:
        return _process_video_highlights_impl(**{k: v for k, v in locals().items() if k != "run_log_handler"})
    finally:
        teardown_run_logging(run_log_handler)


def _preflight_dependencies(camera_mode: str, analysis_only: bool, no_audio: bool,
                            need_detector: bool = True) -> Optional[str]:
    """Check heavy dependencies BEFORE any expensive work.

    Returns an error message when the run cannot possibly succeed.
    """
    import importlib.util
    import shutil as _shutil

    if need_detector and importlib.util.find_spec("ultralytics") is None:
        return (
            "ultralytics is not installed - player/ball tracking cannot run. "
            "Install with: pip install ultralytics"
        )
    try:
        from backend.services.ffmpeg_tools import ffmpeg_exe

        ffmpeg_ok = bool(ffmpeg_exe())
    except Exception:
        ffmpeg_ok = _shutil.which("ffmpeg") is not None
    if not ffmpeg_ok:
        return "ffmpeg is not installed - the proxy pass and renders need it (https://ffmpeg.org)."
    return None


def _process_video_highlights_impl(
    video_path: str,
    output_dir: str,
    select_player: bool = False,
    pre_seconds: float = 2.0,
    post_seconds: float = 6.0,
    min_clip_duration: float = 4.0,
    no_audio: bool = False,
    overlay: bool = False,
    trim_start: Optional[float] = None,
    trim_end: Optional[float] = None,
    threads: Optional[int] = None,
    require_gpu: bool = False,
    speed_sensitivity: float = 2.0,
    audio_sensitivity: float = 2.0,
    focus_event_types: Optional[List[str]] = None,
    model_version: Optional[str] = None,
    analysis_only: bool = False,
    camera_mode: str = "wide",
    zoom_factor: float = 1.6,
    render_full_follow_cam: bool = False,
    player_roi: Optional[Dict[str, float]] = None,
    yolo_model: Optional[str] = None,
    tracker_config: Optional[str] = None,
    inference_imgsz: Optional[int] = None,
    detection_conf: float = 0.18,
    vid_stride: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
    debug: bool = False,
    log_file: Optional[str] = None,
    debug_video: Optional[bool] = None,
    dump_training_data: bool = False,
    goal_box_left: Optional[Dict[str, float]] = None,
    goal_box_right: Optional[Dict[str, float]] = None,
    detect_cards: bool = True,
    broadcast_reel: bool = True,
    scorebug: bool = True,
    team_left: str = "HOME",
    team_right: str = "AWAY",
    team_left_color: Optional[str] = None,
    team_right_color: Optional[str] = None,
    auto_detect_team_colors: bool = False,
    llm_report: bool = True,
    profile: Optional[str] = None,
    proxy_height: Optional[int] = None,
    output_height: Optional[int] = None,
    batch_size: Optional[object] = None,
    ball_tiles: Optional[bool] = None,
    use_tensorrt: Optional[bool] = None,
    device: Optional[str] = None,
    camera_style: Optional[str] = None,
    focus_track_id: Optional[int] = None,
    reuse_tracking_from: Optional[str] = None,
    pitch_corners: Optional[List[List[float]]] = None,
    reel_minutes: Optional[float] = None,
    reel_preset: Optional[str] = None,
    player_spotlight_reel: bool = False,
    cancel_event: Optional[threading.Event] = None,
    detector_factory: Optional[DetectorFactory] = None,
) -> bool:
    """Proxy-first v2 pipeline (see :func:`process_video_highlights`)."""
    emit_progress(progress_callback, "initializing", 0.0, "Validating runtime and configuration")
    if require_gpu:
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        if not cuda_ok:
            print("ERROR: GPU acceleration is required but no CUDA-capable GPU was detected.")
            emit_progress(progress_callback, "failed", 1.0, "GPU is required but CUDA is not available")
            return False

    video_path = os.path.abspath(os.path.expanduser(video_path))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}")
        emit_progress(progress_callback, "failed", 1.0, "Source video file was not found")
        return False

    camera_mode = str(camera_mode or "wide").strip().lower()
    if camera_mode not in FOLLOW_CAM_MODES:
        print(f"Error: Unsupported camera_mode '{camera_mode}'. Valid options: {', '.join(sorted(FOLLOW_CAM_MODES))}")
        emit_progress(progress_callback, "failed", 1.0, "Unsupported camera mode")
        return False
    zoom_factor = max(1.0, float(zoom_factor or 1.0))

    try:
        cfg = resolve_job_config({
            "profile": profile,
            "proxy_height": proxy_height,
            "inference_imgsz": inference_imgsz,
            "vid_stride": vid_stride,
            "yolo_model": yolo_model,
            "batch_size": batch_size,
            "output_height": output_height,
            "debug_video": debug_video,
            "ball_tiles": ball_tiles,
            "tracker_config": tracker_config,
        })
    except ValueError as exc:
        print(f"Error: {exc}")
        emit_progress(progress_callback, "failed", 1.0, str(exc))
        return False
    eff_proxy_h = int(cfg["proxy_height"])
    eff_imgsz = int(cfg["inference_imgsz"])
    eff_stride = max(1, int(cfg["vid_stride"]))
    eff_model = str(cfg["yolo_model"])
    eff_batch = None if str(cfg.get("batch_size", "auto")).lower() == "auto" else max(1, int(cfg["batch_size"]))
    eff_out_h = _even(int(cfg["output_height"]))
    eff_debug_video = bool(cfg.get("debug_video"))
    eff_ball_tiles = bool(cfg.get("ball_tiles"))
    eff_tracker = str(cfg.get("tracker_config") or "bytetrack.yaml")

    reuse_dir = _resolve_reuse_dir(reuse_tracking_from)
    if reuse_tracking_from and reuse_dir is None:
        print(f"[warn] reuse_tracking_from={reuse_tracking_from!r}: no tracks.npz found; tracking will run")

    preflight_error = _preflight_dependencies(
        camera_mode, analysis_only, no_audio,
        need_detector=detector_factory is None and reuse_dir is None,
    )
    if preflight_error:
        print(f"ERROR: {preflight_error}")
        emit_progress(progress_callback, "failed", 1.0, preflight_error)
        return False

    requested_targets: List[str] = [str(i).strip().lower() for i in (focus_event_types or []) if str(i).strip()]
    print(f"\nProcessing video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Profile: {cfg['profile']} | proxy {eff_proxy_h}p | detector {eff_model} imgsz={eff_imgsz} "
          f"stride={eff_stride} batch={eff_batch or 'auto'} | tracker {_tracker_kind(eff_tracker)} | "
          f"output {eff_out_h}p")
    print(f"Camera mode: {camera_mode} (style {camera_style or 'broadcast'}, zoom {zoom_factor:.2f}x) | "
          f"full movie: {'yes' if render_full_follow_cam else 'no'} | analysis-only: {'yes' if analysis_only else 'no'}")
    if trim_start is not None or trim_end is not None:
        print(f"Trim range: {format_time(trim_start or 0)} to {format_time(trim_end) if trim_end else 'end'}")
    if reuse_dir:
        print(f"Reusing tracking from: {reuse_dir}")
    print()

    ensure_dir(output_dir)
    device_label = str(device or "auto")
    try:
        from backend.services.device import select_device

        device_label = select_device(str(device) if device else "auto").torch_device
    except Exception:
        pass
    tracker = ProgressTracker(output_dir, analysis_only=analysis_only, legacy_callback=progress_callback,
                              device=device_label)
    timings: Dict[str, float] = {}
    run_started = time.monotonic()

    def _timed(name: str, started: float) -> None:
        timings[name] = round(timings.get(name, 0.0) + (time.monotonic() - started), 3)

    try:
        _check_cancel(cancel_event)
        original_video = video_path

        # ------------------------------------------------------------------
        # 1. Proxy (the one analysis decode of the source) or reuse
        # ------------------------------------------------------------------
        tracker.start_stage("proxy", "Building analysis proxy")
        t0 = time.monotonic()
        tracking: Optional[TrackingResult] = None
        proxy = None
        if reuse_dir:
            tracking = TrackingResult.load(reuse_dir)
            reuse_proxy = _find_proxy_file(reuse_dir, tracking)
            if reuse_proxy:
                proxy = _fs.proxy_result_from_file(reuse_proxy, source_size=tuple(tracking.frame_size),
                                                   trim_start_s=float(tracking.trim_offset_s))
                if os.path.abspath(reuse_dir) != output_dir:
                    local_proxy = os.path.join(output_dir, os.path.basename(reuse_proxy))
                    _link_or_copy(reuse_proxy, local_proxy)
                    proxy.path = local_proxy
                    for extra in (_fs.AUDIO_ANALYSIS_FILENAME,):
                        src_extra = os.path.join(reuse_dir, extra)
                        if os.path.isfile(src_extra):
                            _link_or_copy(src_extra, os.path.join(output_dir, extra))
                    thumbs_src = os.path.join(reuse_dir, _fs.THUMBS_DIRNAME)
                    thumbs_dst = os.path.join(output_dir, _fs.THUMBS_DIRNAME)
                    if os.path.isdir(thumbs_src) and not os.path.isdir(thumbs_dst):
                        import shutil

                        shutil.copytree(thumbs_src, thumbs_dst)
                audio_local = os.path.join(output_dir, _fs.AUDIO_ANALYSIS_FILENAME)
                proxy.audio_path = audio_local if os.path.isfile(audio_local) else None
                proxy.thumbs_dir = os.path.join(output_dir, _fs.THUMBS_DIRNAME)
                tracker.update(1.0, "Reusing proxy from previous run")
                if trim_start is not None and abs(float(trim_start) - float(tracking.trim_offset_s)) > 0.05:
                    print(f"[warn] trim_start {trim_start} differs from the reused tracking window "
                          f"({tracking.trim_offset_s:.2f}s); using the reused window")
        if proxy is None:
            proxy = _fs.build_proxy(
                video_path, output_dir, height=eff_proxy_h, trim_start=trim_start, trim_end=trim_end,
                thumbs=True, audio_wav=True, progress_cb=tracker.module_callback("proxy"),
                cancel_event=cancel_event,
            )
        _timed("proxy", t0)
        processing_video = proxy.path
        trim_offset = float(proxy.trim_start_s)
        print(f"[1/6] Proxy: {proxy.path} ({proxy.width}x{proxy.height} @ {proxy.fps:.3f} fps, "
              f"scale {proxy.scale:.3f}) in {timings['proxy']:.1f}s")
        _check_cancel(cancel_event)

        # ------------------------------------------------------------------
        # 2. Detection + tracking on the proxy (skipped when reused)
        # ------------------------------------------------------------------
        tracker.start_stage("tracking", "Detecting and tracking players and ball")
        t0 = time.monotonic()
        roi = dict(player_roi) if isinstance(player_roi, dict) and player_roi else None
        if select_player and roi is None:
            roi = _interactive_roi_from_proxy(proxy.path)
        if tracking is None:
            effective = dict(cfg)
            effective.update({
                "inference_imgsz": eff_imgsz, "vid_stride": eff_stride, "yolo_model": eff_model,
                "batch_size": eff_batch, "ball_tiles": eff_ball_tiles, "detection_conf": float(detection_conf),
                "use_tensorrt": use_tensorrt, "device": device, "proxy": proxy.as_dict(),
            })
            if detector_factory is not None:
                detector = detector_factory(effective)
            else:
                from backend.services.detectors import build_detector

                detector = build_detector(
                    resolve_model_path(eff_model), imgsz=eff_imgsz, conf=float(detection_conf),
                    device=device or None, batch=eff_batch, use_tensorrt=use_tensorrt, ball_tiles=eff_ball_tiles,
                )
            tracking = _te.track_video(
                proxy, detector=detector, stride=eff_stride, tracker=_tracker_kind(eff_tracker),
                batch_size=eff_batch, progress_cb=tracker.module_callback("tracking"), cancel_event=cancel_event,
                focus_roi=roi, focus_track_id=focus_track_id, source_video_path=original_video,
            )
            if (tracking.timings or {}).get("cancelled"):
                raise PipelineCancelled("Run canceled during tracking")
            tracking.vid_stride = eff_stride
        else:
            selection = _apply_focus(tracking, focus_track_id, roi)
            tracking.detector = {**dict(tracking.detector or {}), "focus_selection": selection,
                                 "reused_from": reuse_dir}
            tracker.update(1.0, "Reused tracks from previous run")
        tracking.source_video_path = original_video
        tracking.processing_video_path = processing_video
        _timed("tracking", t0)
        print(f"[2/6] Tracking: {len(tracking.players)} player identities, {len(tracking.ball)} ball detections "
              f"({'reused' if reuse_dir else str(timings.get('tracking', 0.0)) + 's'})")
        _check_cancel(cancel_event)

        # ------------------------------------------------------------------
        # 3. Teams (per-track kit colours, from proxy frames)
        # ------------------------------------------------------------------
        tracker.start_stage("teams", "Assigning teams by kit colour")
        t0 = time.monotonic()
        cfg_a = TeamConfig(name=team_left, color_hex=team_left_color) if team_left_color else None
        cfg_b = TeamConfig(name=team_right, color_hex=team_right_color) if team_right_color else None
        already = reuse_dir is not None and any(p.team in (0, 1) for p in tracking.players.values())
        if not already:
            try:
                assign_track_teams(proxy.path, tracking, cfg_a, cfg_b)
            except Exception as team_exc:
                print(f"[warn] Team assignment failed: {team_exc}")
        team_report = dict((tracking.detector or {}).get("team_assignment") or {})
        detected_colors = dict(team_report.get("team_colors_hex") or {})
        team_left_color = team_left_color or detected_colors.get("0")
        team_right_color = team_right_color or detected_colors.get("1")
        if focus_track_id is None and tracking.focus_track_id is None and camera_mode in {"follow_player", "follow_action"}:
            tracking.focus_track_id = tracking.longest_track_id()
            print(f"[warn] No focus player given; following the longest track #{tracking.focus_track_id}")
        tracking.save(output_dir)
        tracker.update(1.0, "Teams assigned")
        _timed("teams", t0)
        _check_cancel(cancel_event)

        legacy_tracks: Dict[int, List[TrackPoint]] = {}
        legacy_ball: List[TrackPoint] = []
        legacy_target_id: Optional[int] = None
        try:
            legacy_tracks, legacy_ball, _fps, _wh, legacy_sel = _te.legacy_track_video_tuple(tracking)
            legacy_target_id = int(legacy_sel["target_track_id"])
        except Exception as legacy_exc:
            LOGGER.debug("legacy track views unavailable: %s", legacy_exc)
        traj: List[TrackPoint] = list(legacy_tracks.get(legacy_target_id, [])) if legacy_target_id is not None else []

        # ------------------------------------------------------------------
        # 4. Analysis: ball, pitch, goals, states, cards, stats, events
        # ------------------------------------------------------------------
        tracker.start_stage("analysis", "Analysing ball, pitch, goals, stats and events")
        t0 = time.monotonic()
        W, H = int(tracking.frame_size[0]), int(tracking.frame_size[1])
        fps = float(tracking.fps or proxy.fps or 25.0)
        analysis_end_s = float(tracking.duration_s or proxy.duration_s or 0.0)
        if analysis_end_s <= 0:
            analysis_end_s = max(float(tracking.ball.t[-1]) if len(tracking.ball) else 0.0, 1.0)
        player_positions = tracking.all_player_positions()
        ball_track = build_ball_track(tracking.ball.to_samples(), (W, H))
        tracker.update(0.1, "Ball track built")

        calibration = None
        corners = _normalize_corners(pitch_corners, (W, H)) if pitch_corners else None
        if corners is not None:
            try:
                calibration = _pc.calibrate_from_corners(corners, frame_size=(W, H))
            except Exception as calib_exc:
                print(f"[warn] Manual pitch corners rejected ({calib_exc}); using auto calibration")
        field_geometry = estimate_field_geometry(
            player_positions, (W, H), goal_box_left=goal_box_left, goal_box_right=goal_box_right,
            calibration=calibration,
        )
        if calibration is None:
            try:
                calibration = _pc.calibrate_auto(tracking, ball_track, None, field_geometry=field_geometry)
                if float(getattr(calibration, "confidence", 0.0) or 0.0) >= 0.45:
                    # Player-only geometry under-detects goals: pin the field
                    # rectangle (and goal mouths) to the calibrated pitch.
                    field_geometry = estimate_field_geometry(
                        player_positions, (W, H), goal_box_left=goal_box_left, goal_box_right=goal_box_right,
                        calibration=calibration,
                    )
            except Exception as calib_exc:
                print(f"[warn] Auto pitch calibration failed: {calib_exc}")
                calibration = None
        tracker.update(0.25, "Pitch geometry estimated")

        audio_envelope = None
        if not no_audio:
            try:
                audio_envelope = compute_audio_envelope(processing_video, audio_path=proxy.audio_path)
            except Exception as audio_exc:
                print(f"[warn] Audio analysis failed: {audio_exc}")
        goal_candidates: List[object] = []
        goal_events = detect_goal_events(
            ball_track, field_geometry, 0.0, analysis_end_s, player_tracks=tracking,
            audio_envelope=audio_envelope, candidates_out=goal_candidates,
        )
        game_segments = analyze_game_states(ball_track, field_geometry, 0.0, analysis_end_s, goal_events=goal_events)
        set_piece_events = detect_set_pieces(ball_track, field_geometry, 0.0, analysis_end_s)
        game_segments = overlay_set_piece_states(game_segments, set_piece_events)
        tracker.update(0.45, "Goals, game states and set pieces detected")
        _check_cancel(cancel_event)

        card_events: List[object] = []
        if detect_cards:
            try:
                card_events = detect_card_events(
                    processing_video, stopped_play_windows(game_segments),
                    debug_dir=os.path.join(output_dir, "card_crops"), ball_track=ball_track,
                    coord_scale=float(proxy.scale or 1.0),
                )
            except Exception as card_exc:
                print(f"[warn] Card detection failed: {card_exc}")
        tracker.update(0.55, "Card scan complete")
        _check_cancel(cancel_event)

        team_cfgs = [
            {"name": team_left, "color_hex": team_left_color},
            {"name": team_right, "color_hex": team_right_color},
        ]
        player_stats_doc: Dict[str, object] = {}
        team_stats_doc: Dict[str, object] = {}
        player_stats_path = team_stats_path = None
        try:
            player_stats_doc, team_stats_doc = _ms.compute_match_stats(
                tracking, ball_track, game_segments, goal_events, calibration, team_cfgs,
                set_piece_events=set_piece_events,
            )
            p_path, t_path = _ms.write_stats(output_dir, player_stats_doc, team_stats_doc)
            player_stats_path, team_stats_path = str(p_path), str(t_path)
        except Exception as stats_exc:
            print(f"[warn] Match stats failed: {stats_exc}")
            LOGGER.debug("match stats traceback", exc_info=True)
        tracker.update(0.7, "Player and team stats computed")

        event_set = _ee.detect_events(
            tracking, ball_track, field_geometry, game_segments, goal_events, set_piece_events, card_events,
            calibration=calibration, audio_envelope=audio_envelope,
            team_names={0: team_left, 1: team_right}, focus_track_id=tracking.focus_track_id,
            goal_candidates=goal_candidates, duration_s=analysis_end_s,
        )
        event_set.trim_offset_s = trim_offset
        _event_location_evidence(event_set, ball_track, calibration)
        must_include = ["goal", "red_card"] + [t for t in requested_targets if t in _ee.EVENT_TYPES]
        reel_target = float(reel_minutes) * 60.0 if reel_minutes else 300.0
        preset = reel_preset if reel_preset in _ee.REEL_PRESETS else None
        if player_spotlight_reel and tracking.focus_track_id is not None:
            fid = int(tracking.focus_track_id)
            involved = [e for e in event_set.events if fid in (e.player_track_id, e.secondary_track_id)]
            reel_plan = _ee.plan_reel(
                involved, target_duration_s=reel_target, preset=preset, focus_track_id=fid,
                must_include=(), pre_s=max(float(pre_seconds), 4.0), post_s=max(float(post_seconds), 6.0),
                duration_s=analysis_end_s, min_excitement=0.0, exclude_types=(),
            )
            reel_plan["spotlight_track_id"] = fid
            if not reel_plan.get("clips"):
                print(f"[warn] No events involve player #{fid}; the spotlight reel will be empty")
        else:
            reel_plan = _ee.plan_reel(
                event_set, target_duration_s=reel_target, preset=preset,
                focus_track_id=tracking.focus_track_id, must_include=tuple(dict.fromkeys(must_include)),
                pre_s=max(float(pre_seconds), 4.0), post_s=max(float(post_seconds), 6.0),
                duration_s=analysis_end_s,
            )
        _enforce_min_clip(reel_plan, float(min_clip_duration or 0.0), analysis_end_s)
        events_path = _ee.write_events(output_dir, event_set, reel_plan)
        bookmarks = _ee.events_to_bookmarks(event_set, reel_plan, trim_offset)
        tracker.update(0.85, "Events detected and reel planned")

        # -- manifests (shapes unchanged for API/UI consumers) --
        state_summary = summarize_states(game_segments)
        ball_coverage = ball_track.coverage_fraction(0.0, analysis_end_s)
        ball_track_stats = {
            **{key: float(value) for key, value in ball_track.stats.items()},
            "coverage_fraction": round(ball_coverage, 4),
        }
        attribution = list((team_stats_doc or {}).get("goal_attribution") or [])

        def _goal_team(goal_t: float) -> Optional[int]:
            best = None
            for row in attribution:
                try:
                    dt = abs(float(row.get("t", -1e9)) - goal_t)
                except (TypeError, ValueError):
                    continue
                if dt <= 1.5 and (best is None or dt < best[0]):
                    best = (dt, row.get("team"))
            return best[1] if best else None

        original_goal_events = [
            {**goal.to_dict(), "t": round(goal.t + trim_offset, 3), "team": _goal_team(goal.t)}
            for goal in goal_events
        ]
        original_set_pieces = [
            {**sp.to_dict(), "t_start": round(sp.t_start + trim_offset, 3), "t_kick": round(sp.t_kick + trim_offset, 3)}
            for sp in set_piece_events
        ]
        original_card_events = [{**card.to_dict(), "t": round(card.t + trim_offset, 3)} for card in card_events]
        original_game_states = [
            {**seg.to_dict(), "start_s": round(seg.start_s + trim_offset, 3), "end_s": round(seg.end_s + trim_offset, 3)}
            for seg in game_segments
        ]
        game_states_path = os.path.join(output_dir, "analysis_game_states.json")
        with open(game_states_path, "w", encoding="utf-8") as handle:
            json.dump({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "trim_offset_seconds": round(trim_offset, 3),
                "field_geometry": field_geometry.to_dict(),
                "pitch_calibration": calibration.to_dict() if calibration is not None else None,
                "ball_track_stats": ball_track_stats,
                "state_summary_s": state_summary,
                "segments": original_game_states,
                "goal_events": original_goal_events,
                "goal_candidates": [
                    {**c.to_dict(), "t": round(c.t + trim_offset, 3)} for c in goal_candidates
                    if hasattr(c, "to_dict")
                ],
                "set_piece_events": original_set_pieces,
                "card_events": original_card_events,
            }, handle, indent=2, default=str)

        tracking_manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_video_path": original_video,
            "processing_video_path": processing_video,
            "output_dir": output_dir,
            "trim_offset_seconds": round(trim_offset, 3),
            "tracks_file": "tracks.npz",
            "proxy": proxy.as_dict(),
            "profile": cfg["profile"],
            "camera": {
                "mode": camera_mode,
                "style": camera_style or "broadcast",
                "zoom_factor": round(zoom_factor, 3),
                "render_full_follow_cam": bool(render_full_follow_cam),
                "output_height": eff_out_h,
            },
            "detector": {
                "yolo_model": eff_model,
                "tracker_config": eff_tracker,
                "inference_imgsz": eff_imgsz,
                "detection_conf": round(float(detection_conf), 3),
                "vid_stride": eff_stride,
                "batch_size": eff_batch or "auto",
                "info": {k: v for k, v in (tracking.detector or {}).items()
                         if isinstance(v, (str, int, float, bool)) or v is None},
                "reused_from": reuse_dir,
            },
            "selection": {
                "manual_select_popup": bool(select_player),
                "player_roi": dict(roi or {}),
                "focus_track_id": tracking.focus_track_id,
                "stitched_track_ids": list(tracking.players[legacy_target_id].source_track_ids or [legacy_target_id])
                if legacy_target_id is not None and legacy_target_id in tracking.players else [],
            },
            "video": {"fps": round(fps, 3), "frame_width": W, "frame_height": H},
            "tracking": {
                "target_track_id": legacy_target_id,
                "focus_track_id": tracking.focus_track_id,
                "player_track_count": len(tracking.players),
                "target_track": _trajectory_to_manifest_points(traj, trim_offset=trim_offset),
                "ball_track": _trajectory_to_manifest_points(legacy_ball, trim_offset=trim_offset),
            },
            "game_analysis": {
                "field_geometry": field_geometry.to_dict(),
                "ball_track_stats": ball_track_stats,
                "state_summary_s": state_summary,
                "goal_events": original_goal_events,
            },
        }
        tracking_manifest_path = write_tracking_manifest(output_dir, tracking_manifest)

        manifest_path = os.path.join(output_dir, "analysis_bookmarks.json")
        settings_block = {
            "profile": cfg["profile"],
            "pre_seconds": pre_seconds,
            "post_seconds": post_seconds,
            "min_clip_duration": min_clip_duration,
            "speed_sensitivity": speed_sensitivity,
            "audio_sensitivity": audio_sensitivity,
            "camera_mode": camera_mode,
            "camera_style": camera_style or "broadcast",
            "zoom_factor": round(zoom_factor, 3),
            "render_full_follow_cam": bool(render_full_follow_cam),
            "no_audio": no_audio,
            "overlay": overlay,
            "threads": threads,
            "yolo_model": eff_model,
            "tracker_config": eff_tracker,
            "inference_imgsz": eff_imgsz,
            "detection_conf": round(float(detection_conf), 3),
            "vid_stride": eff_stride,
            "proxy_height": eff_proxy_h,
            "output_height": eff_out_h,
            "debug": bool(debug),
            "debug_video": eff_debug_video,
            "dump_training_data": bool(dump_training_data),
            "goal_box_left": dict(goal_box_left or {}),
            "goal_box_right": dict(goal_box_right or {}),
            "focus_track_id": tracking.focus_track_id,
            "reuse_tracking_from": reuse_dir,
            "reel_target_s": reel_plan.get("target_duration_s"),
            "player_spotlight_reel": bool(player_spotlight_reel),
        }
        manifest: Dict[str, object] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "video_path": original_video,
            "processing_video_path": processing_video,
            "output_dir": output_dir,
            "analysis_only": analysis_only,
            "model_version": model_version,
            "focus_event_types": requested_targets,
            "trim_offset_seconds": round(trim_offset, 3),
            "tracking_manifest_path": tracking_manifest_path,
            "events_path": events_path,
            "player_stats_path": player_stats_path,
            "team_stats_path": team_stats_path,
            "settings": settings_block,
            "stats": {
                "event_count": len(event_set.events),
                "event_counts_by_type": event_set.summary().get("counts_by_type", {}),
                "reel_clip_count": len(reel_plan.get("clips") or []),
                "goal_event_count": len(goal_events),
                "goal_candidate_count": len(goal_candidates),
                "bookmark_count": len(bookmarks),
                "duration_s": round(analysis_end_s, 3),
            },
            "goal_events": original_goal_events,
            "game_states_path": game_states_path,
            "bookmarks": bookmarks,
        }
        manifest_path, csv_path = write_analysis_bookmark_files(output_dir, manifest)
        if llm_report:
            try:
                report_md = generate_match_report({
                    "goal_events": original_goal_events,
                    "card_events": original_card_events,
                    "set_piece_events": original_set_pieces,
                    "state_summary_s": state_summary,
                    "team_stats": team_stats_doc,
                    "player_stats": [
                        {k: p.get(k) for k in ("track_id", "team", "team_name", "label", "jersey_number",
                                                "minutes_tracked", "distance_m", "top_speed_mps", "sprints",
                                                "touches", "passes_attempted", "passes_completed", "shots")}
                        for p in list(player_stats_doc.get("players") or [])[:30] if isinstance(p, dict)
                    ],
                    "events_summary": event_set.summary(),
                    "events": [e.to_dict() for e in sorted(event_set.events, key=lambda e: -e.excitement)[:40]],
                    "ball_coverage_fraction": round(ball_coverage, 3),
                    "bookmarks": bookmarks[:60],
                })
                with open(os.path.join(output_dir, "match_report.md"), "w", encoding="utf-8") as handle:
                    handle.write(report_md)
            except Exception as report_exc:
                print(f"[warn] Match report failed: {report_exc}")
        _timed("analysis", t0)
        print(f"[3/6] Analysis: {len(goal_events)} goal(s), {len(event_set.events)} events, "
              f"{len(reel_plan.get('clips') or [])} reel clip(s); ball visible {ball_coverage * 100:.1f}% "
              f"({timings['analysis']:.1f}s)")
        print(f"[analysis] Events: {events_path}")
        print(f"[analysis] Bookmarks: {manifest_path} / {csv_path}")
        _check_cancel(cancel_event)

        # ------------------------------------------------------------------
        # 5. Camera plan (follow modes; cheap, always written for them)
        # ------------------------------------------------------------------
        tracker.start_stage("camera_plan", "Planning the game camera")
        t0 = time.monotonic()
        follow_mode = camera_mode != "wide"
        out_w = _even(eff_out_h * 16.0 / 9.0)
        render_size = (out_w, eff_out_h)
        # Explicit output size bounds max zoom so no output pixel is upscaled;
        # sources that are not much larger than the output (720p/1080p files)
        # would then get no zoom at all, so plan those at source scale and
        # let the renderer scale to the requested height.
        plan_size: Optional[Tuple[int, int]] = render_size if H >= 1.5 * eff_out_h else None
        camera_plan = None
        camera_quality: Optional[Dict[str, object]] = None
        plan_focus: Optional[int] = None
        if follow_mode or eff_debug_video or dump_training_data:
            plan_focus = tracking.focus_track_id if camera_mode in {"follow_player", "follow_action"} else None
            planner_cfg = None
            if camera_mode == "follow_action":
                try:
                    planner_cfg = _cp.CameraPlannerConfig(focus_ball_weight=0.6)
                except Exception:
                    planner_cfg = None
            camera_plan = _cp.plan_camera(
                ball_track=ball_track, player_positions=player_positions, geometry=field_geometry,
                segments=game_segments, start_seconds=0.0, end_seconds=analysis_end_s, fps=fps,
                frame_size=(W, H), base_zoom=zoom_factor, config=planner_cfg, output_size=plan_size,
                player_tracks=tracking, focus_track_id=plan_focus, style=camera_style or "broadcast",
            )
            camera_plan.write_jsonl(
                os.path.join(output_dir, "camera_decisions.jsonl"),
                transform=lambda row: {**row, "t_source": round(float(row["t"]) + trim_offset, 3)},
            )
            camera_plan.write_sendcmd(os.path.join(output_dir, "camera_crops.txt"))
            if _camera_quality is not None:
                try:
                    _camera_quality.write_plan_quality(
                        camera_plan, os.path.join(output_dir, "camera_quality.json"),
                        ball_track=ball_track, segments=game_segments,
                    )
                    with open(os.path.join(output_dir, "camera_quality.json"), "r", encoding="utf-8") as handle:
                        camera_quality = json.load(handle)
                except Exception as quality_exc:
                    print(f"[warn] Camera quality report failed: {quality_exc}")
            print(f"[camera] Plan summary: {camera_plan.summary()}")
        if dump_training_data:
            ball_csv_path = os.path.join(output_dir, "ball_track.csv")
            with open(ball_csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["t", "t_source", "x", "y"])
                writer.writeheader()
                for row in ball_track.to_rows():
                    writer.writerow({**row, "t_source": round(row["t"] + trim_offset, 3)})
        tracker.update(1.0, "Camera planned")
        _timed("camera_plan", t0)
        _check_cancel(cancel_event)

        def _render_info_cb(info: Dict[str, object]) -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise PipelineCancelled("Run canceled during render")
            total = float(info.get("total") or 0) or 1.0
            tracker.update(float(info.get("frame") or 0) / total, None,
                           {"fps_processing": info.get("fps"), "eta_s": info.get("eta_s"),
                            "frames_done": info.get("frame"), "frames_total": info.get("total")})

        if eff_debug_video and camera_plan is not None:
            try:
                _cr.render_camera_plan_video(
                    video_path=original_video, output_path=os.path.join(output_dir, "debug_camera_wide.mp4"),
                    plan=camera_plan, include_audio=True, debug_wide=True, geometry=field_geometry,
                    debug_source_path=processing_video, debug_source_time_offset=0.0,
                    source_time_offset=trim_offset,
                )
            except PipelineCancelled:
                raise
            except Exception as exc:
                print(f"[warn] Debug video render failed: {exc}")

        result_summary = {
            "proxy": proxy.as_dict(),
            "timings": timings,
            "camera_quality": camera_quality,
            "events_path": events_path,
            "player_stats_path": player_stats_path,
            "team_stats_path": team_stats_path,
        }
        if analysis_only:
            _write_run_summary(output_dir, {**result_summary, "analysis_only": True,
                                            "total_s": round(time.monotonic() - run_started, 3)})
            print("[analysis] Analysis-only run complete. Skipped rendering.")
            tracker.finish("completed", "Analysis-only run complete")
            emit_progress(progress_callback, "completed", 0.99, "Analysis-only run complete",
                          {"timings": dict(timings)})
            return True

        # ------------------------------------------------------------------
        # 6. Final render: the second (and last) decode of the source
        # ------------------------------------------------------------------
        tracker.start_stage("render", "Rendering the game-camera movie")
        t0 = time.monotonic()
        scorebug_fn = None
        if scorebug and follow_mode:
            scorebug_fn = _cr.make_scorebug_renderer(
                [{"t": g.t, "side": g.side} for g in goal_events], team_left=team_left, team_right=team_right,
            )
        full_follow_cam_path: Optional[str] = None
        if render_full_follow_cam and follow_mode and camera_plan is not None:
            plan_supports_focus = "focus_track_id" in inspect.signature(_cp.plan_camera).parameters
            if camera_mode == "follow_action" and not plan_supports_focus and traj:
                full_follow_cam_path = write_full_follow_cam_video(
                    original_video, (trim_offset, trim_offset + analysis_end_s), output_dir, traj, legacy_ball,
                    camera_mode=camera_mode, zoom_factor=zoom_factor, track_time_offset_seconds=trim_offset,
                    progress_callback=progress_callback,
                )
            else:
                full_follow_cam_path = _cr.render_camera_plan_video(
                    video_path=original_video, output_path=os.path.join(output_dir, "full_follow_ball_zoom.mp4"),
                    plan=camera_plan, include_audio=True, scorebug_fn=scorebug_fn, geometry=field_geometry,
                    output_size=render_size, source_time_offset=trim_offset, encoder="auto", hwaccel="auto",
                    engine="ffmpeg", progress_info_callback=_render_info_cb,
                )
            print(f"[4/6] Full game-camera movie: {full_follow_cam_path}")
        elif render_full_follow_cam and not follow_mode:
            print("[warn] Full follow-cam movie requested, but camera mode is wide. Skipping full zoom export.")
        tracker.update(1.0, "Movie rendered" if full_follow_cam_path else "No full movie requested")
        _timed("render", t0)
        _check_cancel(cancel_event)

        # ------------------------------------------------------------------
        # 7. Highlight clips + reel (cut from the movie when it exists)
        # ------------------------------------------------------------------
        tracker.start_stage("clips_reel", "Cutting highlight clips")
        t0 = time.monotonic()
        plan_clips = list(reel_plan.get("clips") or [])
        rendered: List[Tuple[Dict[str, object], str]] = []
        for n, clip in enumerate(plan_clips, start=1):
            _check_cancel(cancel_event)
            s, e = float(clip["t_start"]), float(clip["t_end"])
            out_path = os.path.join(output_dir, f"highlight_{n:02d}.mp4")
            try:
                if full_follow_cam_path and os.path.isfile(full_follow_cam_path):
                    path = cut_clip_from_rendered(full_follow_cam_path, s, e, out_path)
                elif follow_mode and camera_plan is not None:
                    path = _cr.render_camera_plan_video(
                        video_path=original_video, output_path=out_path, plan=_cp.slice_plan(camera_plan, s, e),
                        include_audio=True, scorebug_fn=scorebug_fn, geometry=field_geometry,
                        output_size=render_size, source_time_offset=trim_offset,
                    )
                else:
                    path = render_clip_ffmpeg(original_video, out_path, s + trim_offset, e + trim_offset)
                rendered.append((clip, path))
            except PipelineCancelled:
                raise
            except Exception as clip_exc:
                print(f"[warn] Highlight clip {n} ({s:.1f}s - {e:.1f}s) failed: {clip_exc}")
            tracker.update(0.7 * n / max(1, len(plan_clips)), f"Cut clip {n}/{len(plan_clips)}")
        clip_paths = [p for _, p in rendered]
        reel_path: Optional[str] = None
        if clip_paths and broadcast_reel:
            try:
                specs = _ee.reel_clip_specs({"clips": [c for c, _ in rendered]}, event_set, clip_paths, trim_offset)
                reel_path = build_broadcast_reel(specs, os.path.join(output_dir, "highlights_reel.mp4"))
                if reel_path:
                    print(f"[reel] {'Player spotlight' if player_spotlight_reel else 'Broadcast'} reel: {reel_path}")
            except Exception as reel_exc:
                print(f"[warn] Reel failed: {reel_exc}")
        _timed("clips_reel", t0)

        if overlay and traj:
            original_intervals = [(float(c["t_start"]) + trim_offset, float(c["t_end"]) + trim_offset)
                                  for c, _ in rendered]
            overlay_workers = min(2, threads) if threads else None
            try:
                draw_spotlight_overlay(original_video, traj, original_intervals, output_dir,
                                       max_workers=overlay_workers)
            except Exception as overlay_exc:
                print(f"[warn] Spotlight overlays failed: {overlay_exc}")

        _write_run_summary(output_dir, {
            **result_summary, "analysis_only": False, "full_follow_cam_path": full_follow_cam_path,
            "clip_paths": clip_paths, "reel_path": reel_path, "total_s": round(time.monotonic() - run_started, 3),
        })
        print(f"[6/6] {len(clip_paths)} clip(s){' + reel' if reel_path else ''} in {output_dir} "
              f"(total {time.monotonic() - run_started:.1f}s; stages {timings})")
        if plan_clips and not clip_paths and not full_follow_cam_path:
            print("[error] Every planned highlight clip failed to render.")
            tracker.finish("failed", "Highlight clip rendering failed")
            emit_progress(progress_callback, "failed", 0.98, "Highlight clip rendering failed")
            return False
        if not plan_clips:
            print("No highlight-worthy events were found. Bookmark table generated for manual review.")
        tracker.finish("completed", "Run complete")
        emit_progress(progress_callback, "completed", 0.99, "Run complete", {"clip_count": len(clip_paths)})
        return True

    except (PipelineCancelled, _fs.ProxyCancelled) as cancel_exc:
        print(f"[cancel] {cancel_exc}")
        tracker.finish("canceled", "Run canceled")
        emit_progress(progress_callback, "canceled", 1.0, "Run canceled", {"cancelled": True})
        return False
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()
        tracker.finish("failed", f"Processing failed: {e}")
        emit_progress(progress_callback, "failed", 1.0, "Processing failed", {"error": str(e)})
        return False
    finally:
        tracker.close()


def _write_run_summary(output_dir: str, payload: Dict[str, object]) -> None:
    """``run_summary.json``: proxy info, stage timings, camera quality, artifact paths."""
    try:
        with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), **payload}, handle, indent=2,
                      default=str)
    except OSError as exc:  # pragma: no cover
        LOGGER.debug("run_summary.json write failed: %s", exc)


def main():
    ap = argparse.ArgumentParser(description="Soccer highlight generator (proxy-first v2 pipeline)")
    ap.add_argument("--video", help="Input video path (iPhone recording)")
    ap.add_argument("--out", help="Output directory for highlights")
    ap.add_argument("--select", action="store_true", help="Interactively select your player's box on first frame")
    ap.add_argument("--pre", type=float, default=2.0, help="Seconds before event")
    ap.add_argument("--post", type=float, default=6.0, help="Seconds after event")
    ap.add_argument("--min-clip", type=float, default=4.0, help="Minimum clip duration (after merging)")
    ap.add_argument("--no-audio", action="store_true", help="Disable audio-based peak detection")
    ap.add_argument("--overlay", action="store_true", help="Render spotlight overlay clips (slower)")
    ap.add_argument("--trim-start", type=str, help="Trim video start time (format: MM:SS or HH:MM:SS or seconds)")
    ap.add_argument("--trim-end", type=str, help="Trim video end time (format: MM:SS or HH:MM:SS or seconds)")
    ap.add_argument("--threads", type=int, default=None, help="Number of parallel threads for clip writing (default: auto, max 4)")
    ap.add_argument("--speed-sensitivity", type=float, default=2.0, help="Speed detection sensitivity (lower = more sensitive, default: 2.0, old default was 3.0)")
    ap.add_argument("--audio-sensitivity", type=float, default=2.0, help="Audio peak detection sensitivity (lower = more sensitive, default: 2.0, old default was 3.0)")
    ap.add_argument("--analysis-only", action="store_true", help="Run detection/bookmark analysis without writing highlight clips")
    ap.add_argument("--camera-mode", choices=sorted(FOLLOW_CAM_MODES), default="wide", help="Video framing mode for rendered clips (follow_ball = game-centric camera that tracks the ball)")
    ap.add_argument("--zoom-factor", type=float, default=1.6, help="Zoom factor for follow camera modes")
    ap.add_argument("--render-full-follow-cam", action="store_true", help="Also render one continuous zoomed follow-cam movie for the processed window/full source")
    ap.add_argument("--debug", action="store_true", help="Print full debug diagnostics to the console")
    ap.add_argument("--log-file", type=str, default=None, help="Write a full DEBUG log (with timestamps) to this file")
    ap.add_argument("--debug-video", action="store_true", help="Render debug_camera_wide.mp4: annotated wide video showing the center of the game, crop box, ball trail, and why each camera decision was made")
    ap.add_argument("--dump-training-data", action="store_true", help="Write camera_decisions.jsonl and ball_track.csv for tuning/training")
    ap.add_argument("--goal-box-left", type=str, default=None, help="Manual left goal box as x1,y1,x2,y2 (normalized 0-1 or pixels); overrides auto estimate")
    ap.add_argument("--goal-box-right", type=str, default=None, help="Manual right goal box as x1,y1,x2,y2 (normalized 0-1 or pixels); overrides auto estimate")
    ap.add_argument("--no-card-detection", action="store_true", help="Disable yellow/red card flagging (enabled by default)")
    ap.add_argument("--no-broadcast-reel", action="store_true", help="Skip the broadcast reel (clips only)")
    ap.add_argument("--team-left", type=str, default="HOME", help="Name of the team defending the LEFT goal (scorebug)")
    ap.add_argument("--team-right", type=str, default="AWAY", help="Name of the team defending the RIGHT goal (scorebug)")
    ap.add_argument("--no-scorebug", action="store_true", help="Disable the score + clock overlay on follow_ball renders")
    ap.add_argument("--team-left-color", type=str, default=None, help="Jersey hex color of the team defending the LEFT goal at kickoff (e.g. #d32f2f) - enables team stats and goal attribution")
    ap.add_argument("--team-right-color", type=str, default=None, help="Jersey hex color of the team defending the RIGHT goal at kickoff")
    ap.add_argument("--auto-team-colors", action="store_true", help="Auto-detect the two jersey colors from sampled frames when colors are not provided")
    ap.add_argument("--no-llm-report", action="store_true", help="Skip the LLM match report (enabled when VH_LLM_PROVIDER is configured)")
    ap.add_argument("--fast", action="store_true", help="Shortcut for --profile fast")
    ap.add_argument("--profile", choices=["fast", "balanced", "quality"], default=None, help="Processing profile (default: balanced)")
    ap.add_argument("--camera-style", choices=["broadcast", "tight", "wide"], default=None, help="Camera framing style for follow modes")
    ap.add_argument("--output-height", type=int, default=None, help="Rendered movie height (profile default 1080)")
    ap.add_argument("--proxy-height", type=int, default=None, help="Analysis proxy height (profile default)")
    ap.add_argument("--yolo-model", type=str, default=None, help="Detector weights (profile default, resolved via VH_MODEL_DIR)")
    ap.add_argument("--imgsz", type=int, default=None, help="Inference image size (profile default)")
    ap.add_argument("--stride", type=int, default=None, help="Analyze every Nth proxy frame (profile default)")
    ap.add_argument("--device", type=str, default=None, help="auto | cuda | mps | cpu")
    ap.add_argument("--focus-track-id", type=int, default=None, help="Track id to follow (follow_player / follow_action)")
    ap.add_argument("--reuse-tracking", type=str, default=None, help="Run directory (or job id) whose tracks.npz and proxy are reused - no detection pass")
    ap.add_argument("--reel-minutes", type=float, default=None, help="Target highlight reel length in minutes (default 5)")
    args = ap.parse_args()

    def _parse_goal_box(raw: Optional[str], flag: str) -> Optional[Dict[str, float]]:
        if not raw:
            return None
        parts = [item.strip() for item in raw.split(",")]
        if len(parts) != 4:
            print(f"Error: {flag} expects x1,y1,x2,y2 (got: {raw})")
            sys.exit(1)
        try:
            x1, y1, x2, y2 = (float(item) for item in parts)
        except ValueError:
            print(f"Error: {flag} values must be numbers (got: {raw})")
            sys.exit(1)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    goal_box_left = _parse_goal_box(args.goal_box_left, "--goal-box-left")
    goal_box_right = _parse_goal_box(args.goal_box_right, "--goal-box-right")

    # Interactive mode if video or output not provided
    if not args.video:
        print("\n=== Video Highlights Generator ===\n")
        args.video = input("Enter the path to your video file: ").strip()
        if not args.video:
            print("Error: Video path is required")
            sys.exit(1)

    if not args.out:
        default_out = "./highlights_output"
        out_input = input(f"Enter output directory (default: {default_out}): ").strip()
        args.out = out_input if out_input else default_out

    # Validate and normalize paths
    args.video = os.path.abspath(os.path.expanduser(args.video))
    args.out = os.path.abspath(os.path.expanduser(args.out))

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if not os.path.isfile(args.video):
        print(f"Error: Path is not a file: {args.video}")
        sys.exit(1)

    # Validate video file extension
    valid_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.MP4', '.MOV', '.AVI', '.MKV', '.M4V'}
    if not any(args.video.endswith(ext) for ext in valid_extensions):
        print(f"Warning: File extension may not be a valid video format: {args.video}")

    # Ask about player selection if not already set
    if not args.select and sys.stdin.isatty():
        select_input = input("Do you want to manually select your player on the first frame? (y/N): ").strip().lower()
        args.select = select_input in ['y', 'yes']

    # Ask about overlay if not already set
    if not args.overlay and sys.stdin.isatty():
        overlay_input = input("Do you want to render spotlight overlay clips? (slower) (y/N): ").strip().lower()
        args.overlay = overlay_input in ['y', 'yes']

    # Ask about analysis-only mode if not already set
    if not args.analysis_only and sys.stdin.isatty():
        analysis_input = input("Run analysis-only mode (bookmarks table, no clip rendering)? (y/N): ").strip().lower()
        args.analysis_only = analysis_input in ['y', 'yes']

    # Ask about trimming if not already set
    trim_start_seconds = None
    trim_end_seconds = None

    if args.trim_start:
        try:
            trim_start_seconds = parse_time(args.trim_start)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    if args.trim_end:
        try:
            trim_end_seconds = parse_time(args.trim_end)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Interactive trim prompts
    if not args.trim_start and not args.trim_end and sys.stdin.isatty():
        trim_input = input("Do you want to trim the video to a specific time range? (y/N): ").strip().lower()
        if trim_input in ['y', 'yes']:
            start_input = input("Enter start time (MM:SS, HH:MM:SS, or seconds) [press Enter for beginning]: ").strip()
            if start_input:
                try:
                    trim_start_seconds = parse_time(start_input)
                except ValueError as e:
                    print(f"Error: {e}")
                    sys.exit(1)

            end_input = input("Enter end time (MM:SS, HH:MM:SS, or seconds) [press Enter for end]: ").strip()
            if end_input:
                try:
                    trim_end_seconds = parse_time(end_input)
                except ValueError as e:
                    print(f"Error: {e}")
                    sys.exit(1)

    # Call the core processing function
    success = process_video_highlights(
        video_path=args.video,
        output_dir=args.out,
        select_player=args.select,
        pre_seconds=args.pre,
        post_seconds=args.post,
        min_clip_duration=args.min_clip,
        no_audio=args.no_audio,
        overlay=args.overlay,
        trim_start=trim_start_seconds,
        trim_end=trim_end_seconds,
        threads=args.threads,
        require_gpu=False,  # CLI doesn't require GPU by default
        speed_sensitivity=args.speed_sensitivity,
        audio_sensitivity=args.audio_sensitivity,
        analysis_only=args.analysis_only,
        camera_mode=args.camera_mode,
        zoom_factor=args.zoom_factor,
        render_full_follow_cam=args.render_full_follow_cam,
        debug=args.debug,
        log_file=args.log_file,
        dump_training_data=args.dump_training_data,
        goal_box_left=goal_box_left,
        goal_box_right=goal_box_right,
        detect_cards=not args.no_card_detection,
        broadcast_reel=not args.no_broadcast_reel,
        scorebug=not args.no_scorebug,
        team_left=args.team_left,
        team_right=args.team_right,
        team_left_color=args.team_left_color,
        team_right_color=args.team_right_color,
        auto_detect_team_colors=args.auto_team_colors,
        llm_report=not args.no_llm_report,
        profile=args.profile or ("fast" if args.fast else None),
        camera_style=args.camera_style,
        output_height=args.output_height,
        proxy_height=args.proxy_height,
        yolo_model=args.yolo_model,
        inference_imgsz=args.imgsz,
        vid_stride=args.stride,
        device=args.device,
        focus_track_id=args.focus_track_id,
        reuse_tracking_from=args.reuse_tracking,
        reel_minutes=args.reel_minutes,
        debug_video=True if args.debug_video else None,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
