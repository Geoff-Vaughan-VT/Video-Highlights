"""Referee card (yellow/red) flagging.

Cards are shown while play is stopped, so this module scans ONLY the
stopped-play windows of a match (ball lost / restart waits / free-kick
setups) - a small fraction of the footage - for the visual signature of a
raised card: a small, strongly saturated yellow or red patch, roughly
card-shaped (taller than wide), that persists across consecutive sampled
frames and is isolated (not part of a large jersey/banner region).

Every flagged event carries a confidence, a human-readable reason, and an
optional saved crop image so detections can be reviewed and used as training
data. This is an honest heuristic: it will miss some cards and needs review
for low-confidence hits - which is exactly why the crops are saved.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("videohighlights.card_detection")


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for card detection") from exc
    return cv2


@dataclass
class CardDetectionConfig:
    # Frames per second sampled inside each stopped-play window.
    sample_fps: float = 6.0
    # HSV thresholds (OpenCV ranges: H 0-179, S/V 0-255).
    yellow_h_range: Tuple[int, int] = (20, 35)
    red_h_max: int = 8
    red_h_min: int = 172
    min_saturation: int = 120
    min_value_yellow: int = 140
    min_value_red: int = 100
    # Card blob geometry, relative to the frame.
    min_area_frac: float = 2e-5
    max_area_frac: float = 1.5e-3
    # Bounding-box aspect (width/height): cards are held portrait, so a
    # 1:1 blob (the ball!) must not pass.
    aspect_range: Tuple[float, float] = (0.35, 0.92)
    # Contour must fill most of its bounding box: a rectangle fills ~0.95,
    # a circle only ~0.78 - this also rejects balls.
    min_fill_ratio: float = 0.82
    # The blob must persist this long (tracked across sampled frames).
    min_persistence_s: float = 0.5
    # ...within this pixel radius between consecutive samples. Deliberately
    # tight: a held card wobbles a few pixels per sample; a walking player's
    # bib moves ~15px per sample and must NOT chain into a track.
    track_radius_px: float = 12.0
    # A held-up card is stationary; reject tracks that drift further than
    # this from where they first appeared (walking players, flags).
    max_track_drift_px: float = 60.0
    # Ignore blobs this close to the known ball position (yellow balls!).
    ball_exclusion_radius_px: float = 50.0
    # Isolation: the same-color area in a 3x-sized neighborhood must not
    # dwarf the blob (rejects jerseys, bibs, banners).
    max_neighborhood_ratio: float = 3.0
    # Merge detections on the same color within this many seconds.
    merge_window_s: float = 10.0
    # Do not scan more than this much total footage (safety cap).
    max_total_scan_s: float = 900.0
    min_confidence: float = 0.55


@dataclass
class CardEvent:
    t: float
    kind: str  # yellow_card | red_card
    confidence: float
    x: float
    y: float
    reason: str
    crop_path: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "t": round(self.t, 3),
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "reason": self.reason,
            "crop_path": self.crop_path,
        }


def _color_masks(hsv: np.ndarray, cfg: CardDetectionConfig) -> Dict[str, np.ndarray]:
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = (
        (h >= cfg.yellow_h_range[0]) & (h <= cfg.yellow_h_range[1])
        & (s >= cfg.min_saturation) & (v >= cfg.min_value_yellow)
    )
    red = (
        ((h <= cfg.red_h_max) | (h >= cfg.red_h_min))
        & (s >= cfg.min_saturation) & (v >= cfg.min_value_red)
    )
    return {"yellow_card": yellow.astype(np.uint8), "red_card": red.astype(np.uint8)}


def _card_blobs(mask: np.ndarray, frame_area: float, cfg: CardDetectionConfig,
                cv2) -> List[Tuple[float, float, float, Tuple[int, int, int, int]]]:
    """Return (cx, cy, quality, bbox) for card-shaped blobs in a color mask."""
    blobs: List[Tuple[float, float, float, Tuple[int, int, int, int]]] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        frac = area / frame_area
        if not (cfg.min_area_frac <= frac <= cfg.max_area_frac):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0 or w <= 0:
            continue
        aspect = w / h
        if not (cfg.aspect_range[0] <= aspect <= cfg.aspect_range[1]):
            continue
        fill = area / float(w * h)
        if fill < cfg.min_fill_ratio:
            continue
        # Isolation check: same-color pixels in the surrounding region.
        pad_w, pad_h = w * 2, h * 2
        y0, y1 = max(0, y - pad_h), min(mask.shape[0], y + h + pad_h)
        x0, x1 = max(0, x - pad_w), min(mask.shape[1], x + w + pad_w)
        neighborhood = float(mask[y0:y1, x0:x1].sum())
        if neighborhood > cfg.max_neighborhood_ratio * area:
            continue
        quality = min(1.0, fill) * min(1.0, area / (cfg.min_area_frac * frame_area * 4.0))
        blobs.append((x + w / 2.0, y + h / 2.0, quality, (x, y, w, h)))
    return blobs


def detect_card_events(
    video_path: str,
    windows: Sequence[Tuple[float, float]],
    config: Optional[CardDetectionConfig] = None,
    debug_dir: Optional[str] = None,
    ball_track: Optional[object] = None,
    coord_scale: float = 1.0,
) -> List[CardEvent]:
    """Scan stopped-play windows of ``video_path`` for raised cards.

    ``ball_track`` (anything with ``position_at(t)``) excludes blobs at the
    ball's known position - a yellow ball is the classic false positive.

    ``coord_scale`` = ``video pixels / ball_track pixels``: the pipeline scans
    the analysis proxy (``ProxyResult.scale``, e.g. 0.5 for a 4K source and a
    1080p proxy) while ball positions are in source pixels. Ball positions
    are scaled into the video's pixel space and returned card ``x``/``y`` are
    scaled back to ball-track (source) pixels.
    """
    scale = float(coord_scale) if coord_scale and coord_scale > 0 else 1.0
    cfg = config or CardDetectionConfig()
    cv2 = _import_cv2()
    events: List[CardEvent] = []
    if not windows:
        return events

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.warning("card detection skipped: cannot open %s", video_path)
        return events

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    step = 1.0 / max(0.5, cfg.sample_fps)
    needed_hits = max(2, int(round(cfg.min_persistence_s * cfg.sample_fps)))
    scanned_s = 0.0

    # Per-color track state across sampled frames: list of
    # {x, y, hits, first_t, best_quality, best_frame, best_bbox}
    try:
        for win_start, win_end in windows:
            if scanned_s >= cfg.max_total_scan_s:
                LOGGER.info("card detection scan cap reached (%.0fs)", cfg.max_total_scan_s)
                break
            tracks: Dict[str, List[Dict[str, object]]] = {"yellow_card": [], "red_card": []}
            t = max(0.0, float(win_start))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            while t <= float(win_end) and scanned_s < cfg.max_total_scan_s:
                ok, frame = cap.read()
                if not ok:
                    break
                actual_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if actual_t > float(win_end) + step:
                    break
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame_area = float(frame.shape[0] * frame.shape[1])
                masks = _color_masks(hsv, cfg)
                ball_pos = None
                if ball_track is not None:
                    try:
                        ball_pos = ball_track.position_at(actual_t)  # type: ignore[attr-defined]
                        if ball_pos is not None and scale != 1.0:
                            ball_pos = (float(ball_pos[0]) * scale, float(ball_pos[1]) * scale)
                    except Exception:
                        ball_pos = None
                for kind, mask in masks.items():
                    blobs = _card_blobs(mask, frame_area, cfg, cv2)
                    if ball_pos is not None:
                        blobs = [
                            b for b in blobs
                            if math.hypot(b[0] - ball_pos[0], b[1] - ball_pos[1])
                            > cfg.ball_exclusion_radius_px
                        ]
                    still_alive: List[Dict[str, object]] = []
                    for track in tracks[kind]:
                        matched = None
                        for blob in blobs:
                            if math.hypot(blob[0] - track["x"], blob[1] - track["y"]) <= cfg.track_radius_px:
                                matched = blob
                                break
                        if matched is not None:
                            blobs.remove(matched)
                            track["x"], track["y"] = matched[0], matched[1]
                            drift = math.hypot(
                                matched[0] - float(track["first_x"]),
                                matched[1] - float(track["first_y"]),
                            )
                            track["max_drift"] = max(float(track.get("max_drift", 0.0)), drift)
                            if drift > cfg.max_track_drift_px:
                                # Moving object (player, flag) - not a held card.
                                continue
                            track["hits"] = int(track["hits"]) + 1
                            if matched[2] > float(track["best_quality"]):
                                track["best_quality"] = matched[2]
                                track["best_frame"] = frame.copy()
                                track["best_bbox"] = matched[3]
                                track["best_t"] = actual_t
                            still_alive.append(track)
                        elif int(track["hits"]) >= needed_hits:
                            finalized = _finalize_track(track, kind, cfg, debug_dir, cv2)
                            if finalized is not None:
                                events.append(finalized)
                        # else: short-lived blob, drop silently
                    for blob in blobs:
                        still_alive.append({
                            "x": blob[0], "y": blob[1], "hits": 1,
                            "first_x": blob[0], "first_y": blob[1],
                            "first_t": actual_t, "best_t": actual_t,
                            "best_quality": blob[2], "best_frame": frame.copy(),
                            "best_bbox": blob[3],
                        })
                    tracks[kind] = still_alive
                t += step
                scanned_s += step
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            # Window ended: flush persistent tracks.
            for kind, kind_tracks in tracks.items():
                for track in kind_tracks:
                    if int(track["hits"]) >= needed_hits:
                        finalized = _finalize_track(track, kind, cfg, debug_dir, cv2)
                        if finalized is not None:
                            events.append(finalized)
    finally:
        cap.release()

    # Merge nearby same-color events and apply the confidence floor.
    merged: List[CardEvent] = []
    for event in sorted(events, key=lambda e: e.t):
        if merged and merged[-1].kind == event.kind and (event.t - merged[-1].t) <= cfg.merge_window_s:
            if event.confidence > merged[-1].confidence:
                merged[-1] = event
            continue
        merged.append(event)
    kept = [e for e in merged if e.confidence >= cfg.min_confidence]
    if scale != 1.0:
        for event in kept:
            event.x, event.y = event.x / scale, event.y / scale
    for event in kept:
        LOGGER.info("card flagged: %s at t=%.1fs confidence=%.2f (%s)",
                    event.kind, event.t, event.confidence, event.reason)
    return kept


def _finalize_track(track: Dict[str, object], kind: str, cfg: CardDetectionConfig,
                    debug_dir: Optional[str], cv2) -> Optional[CardEvent]:
    # A held-up card is stationary; any track that wandered is a moving
    # object (player bib, flag), no matter how many hits it collected.
    if float(track.get("max_drift", 0.0)) > cfg.max_track_drift_px * 0.75:
        return None
    hits = int(track["hits"])
    confidence = min(0.95, 0.45 + 0.08 * hits + 0.2 * float(track["best_quality"]))
    crop_path = None
    frame = track.get("best_frame")
    bbox = track.get("best_bbox")
    if debug_dir and frame is not None and bbox is not None:
        x, y, w, h = bbox  # type: ignore[misc]
        pad = max(w, h) * 3
        y0, y1 = max(0, y - pad), min(frame.shape[0], y + h + pad)  # type: ignore[union-attr]
        x0, x1 = max(0, x - pad), min(frame.shape[1], x + w + pad)  # type: ignore[union-attr]
        crop_path = os.path.join(debug_dir, f"{kind}_{float(track['best_t']):.1f}s.png")
        try:
            cv2.imwrite(crop_path, frame[y0:y1, x0:x1])  # type: ignore[index]
        except Exception:
            crop_path = None
    color = "yellow" if kind == "yellow_card" else "red"
    return CardEvent(
        t=float(track["first_t"]),
        kind=kind,
        confidence=confidence,
        x=float(track["x"]),
        y=float(track["y"]),
        reason=(
            f"raised {color} card signature: small saturated {color} patch persisted "
            f"across {hits} sampled frames during stopped play"
        ),
        crop_path=crop_path,
    )


def stopped_play_windows(
    segments: Sequence[object],
    pad_s: float = 2.0,
    min_window_s: float = 1.0,
) -> List[Tuple[float, float]]:
    """Extract merged stopped-play windows from game-state segments.

    Accepts GameStateSegment objects or dicts with start_s/end_s/state.
    """
    stopped_states = {
        "ball_lost", "restart_left", "restart_right", "restart_touchline",
        "free_kick_setup", "corner_kick_setup",
    }
    raw: List[Tuple[float, float]] = []
    for seg in segments:
        state = getattr(seg, "state", None) or (seg.get("state") if isinstance(seg, dict) else None)
        if state not in stopped_states:
            continue
        start = getattr(seg, "start_s", None)
        end = getattr(seg, "end_s", None)
        if start is None and isinstance(seg, dict):
            start, end = seg.get("start_s"), seg.get("end_s")
        if start is None or end is None:
            continue
        raw.append((max(0.0, float(start) - pad_s), float(end) + pad_s))
    raw.sort()
    merged: List[Tuple[float, float]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [(s, e) for s, e in merged if e - s >= min_window_s]
