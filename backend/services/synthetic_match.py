"""Deterministic synthetic soccer footage with ground truth.

Used by tests, the benchmark harness, and CI to exercise the full pipeline
(detection, tracking, camera planning, rendering, stats, events) without
real match video or a GPU. The scene is a static wide "panoramic" camera
view like an xBotGo/Veo recording: a green pitch, two goals at the left and
right edges, two teams in distinct kit colors, a referee, and a ball that
moves between players with occasional shots and goals.

Everything is seeded, so the same ``SyntheticMatchSpec`` always produces the
same frames and the same ground truth. Ground truth is returned as a
``TrackingResult`` (player boxes + ball detections per frame) plus the list
of scripted goal timestamps, so each stage can be evaluated against truth.

Rendering uses OpenCV only (``mp4v`` writer; ``libx264`` via ffmpeg when
``prefer_ffmpeg=True`` and ffmpeg is available).
"""

from __future__ import annotations

import math
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tracking_types import (
    TEAM_A,
    TEAM_B,
    TEAM_REFEREE,
    BallDetections,
    PlayerTrack,
    TrackingResult,
    ball_detections_from_rows,
    player_track_from_rows,
)


@dataclass
class SyntheticMatchSpec:
    width: int = 1280
    height: int = 720
    fps: float = 25.0
    duration_s: float = 20.0
    players_per_team: int = 5
    seed: int = 7
    # Kit colors in BGR.
    team_a_bgr: Tuple[int, int, int] = (40, 40, 220)   # red
    team_b_bgr: Tuple[int, int, int] = (220, 200, 40)  # cyan/yellow-ish
    referee_bgr: Tuple[int, int, int] = (20, 20, 20)
    grass_bgr: Tuple[int, int, int] = (60, 140, 60)
    ball_radius_px: int = 6
    # Scripted goals: (t_shot_start_s, side) where side is "left"/"right".
    goals: Sequence[Tuple[float, str]] = field(default_factory=lambda: [(8.0, "right"), (15.0, "left")])
    # Pitch margins (fraction of frame).
    margin_x: float = 0.06
    margin_y: float = 0.12
    # Player body size at this resolution.
    player_w_px: int = 18
    player_h_px: int = 44


@dataclass
class SyntheticGroundTruth:
    spec: SyntheticMatchSpec
    tracking: TrackingResult
    goal_times_s: List[Tuple[float, str]]
    pitch_bounds_px: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    goal_boxes_px: Dict[str, Tuple[float, float, float, float]]

    def ball_xy_at(self, t: float) -> Optional[Tuple[float, float]]:
        b = self.tracking.ball
        if not len(b):
            return None
        idx = int(np.argmin(np.abs(b.t - t)))
        return float(b.x[idx]), float(b.y[idx])


def _pitch_geometry(spec: SyntheticMatchSpec):
    x_min = spec.width * spec.margin_x
    x_max = spec.width * (1.0 - spec.margin_x)
    y_min = spec.height * spec.margin_y
    y_max = spec.height * (1.0 - spec.margin_y)
    goal_h = (y_max - y_min) * 0.3
    goal_d = (x_max - x_min) * 0.04
    cy = (y_min + y_max) / 2.0
    goals = {
        "left": (x_min - goal_d, cy - goal_h / 2, x_min, cy + goal_h / 2),
        "right": (x_max, cy - goal_h / 2, x_max + goal_d, cy + goal_h / 2),
    }
    return (x_min, y_min, x_max, y_max), goals


def _simulate(spec: SyntheticMatchSpec):
    """Return per-frame player positions, ball positions and ball visibility."""
    rng = random.Random(spec.seed)
    (x_min, y_min, x_max, y_max), goals = _pitch_geometry(spec)
    n_frames = int(round(spec.duration_s * spec.fps))
    dt = 1.0 / spec.fps

    # Players: team A defends left, team B defends right. Each has a home
    # position and jogs around it with smooth random drift.
    players = []
    for team in (TEAM_A, TEAM_B):
        for k in range(spec.players_per_team):
            frac = (k + 1) / (spec.players_per_team + 1)
            if team == TEAM_A:
                hx = x_min + (x_max - x_min) * (0.18 + 0.32 * frac)
            else:
                hx = x_min + (x_max - x_min) * (0.50 + 0.32 * frac)
            hy = y_min + (y_max - y_min) * (0.15 + 0.7 * ((k * 0.37 + 0.11) % 1.0))
            players.append({"team": team, "home": (hx, hy), "pos": [hx, hy], "vel": [0.0, 0.0],
                            "phase": rng.random() * 6.283})
    players.append({"team": TEAM_REFEREE, "home": ((x_min + x_max) / 2, (y_min + y_max) / 2),
                    "pos": [(x_min + x_max) / 2, (y_min + y_max) / 2 + 40], "vel": [0.0, 0.0],
                    "phase": 1.0})

    # Ball: passes between random players; scripted shots into goals.
    goal_script = sorted((float(t), side) for t, side in spec.goals)
    ball = [ (x_min + x_max) / 2.0, (y_min + y_max) / 2.0 ]
    ball_target_player = 0
    ball_vel = [0.0, 0.0]
    ball_visible = True
    ball_hidden_until = -1.0
    next_pass_t = 1.0
    goal_state: Optional[Dict[str, object]] = None
    goal_events: List[Tuple[float, str]] = []

    pos_frames = np.zeros((n_frames, len(players), 2), dtype=np.float32)
    ball_frames = np.zeros((n_frames, 2), dtype=np.float32)
    ball_vis = np.zeros(n_frames, dtype=bool)

    for f in range(n_frames):
        t = f * dt
        # --- players ---
        for i, p in enumerate(players):
            hx, hy = p["home"]
            # Spring toward home + sinusoidal wander + attraction to ball for
            # the two nearest players (so there is "action" around the ball).
            ax = (hx - p["pos"][0]) * 0.8 + 60 * math.sin(t * 0.7 + p["phase"])
            ay = (hy - p["pos"][1]) * 0.8 + 40 * math.cos(t * 0.9 + p["phase"])
            dxb = ball[0] - p["pos"][0]
            dyb = ball[1] - p["pos"][1]
            dist_b = math.hypot(dxb, dyb)
            if dist_b < 260 and p["team"] != TEAM_REFEREE:
                ax += dxb * 1.2
                ay += dyb * 1.2
            p["vel"][0] = p["vel"][0] * 0.9 + ax * dt
            p["vel"][1] = p["vel"][1] * 0.9 + ay * dt
            speed = math.hypot(*p["vel"])
            max_speed = 170.0  # px/s
            if speed > max_speed:
                p["vel"][0] *= max_speed / speed
                p["vel"][1] *= max_speed / speed
            p["pos"][0] = min(max(x_min + 5, p["pos"][0] + p["vel"][0] * dt), x_max - 5)
            p["pos"][1] = min(max(y_min + 5, p["pos"][1] + p["vel"][1] * dt), y_max - 5)
            pos_frames[f, i] = p["pos"]

        # --- ball ---
        if goal_state is None and goal_script and t >= goal_script[0][0]:
            t0, side = goal_script.pop(0)
            gx1, gy1, gx2, gy2 = goals[side]
            target = ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0)
            goal_state = {"side": side, "target": target, "phase": "shot", "t0": t}
            d = math.hypot(target[0] - ball[0], target[1] - ball[1])
            shot_speed = 900.0
            ball_vel = [(target[0] - ball[0]) / d * shot_speed, (target[1] - ball[1]) / d * shot_speed]

        if goal_state is not None:
            phase = goal_state["phase"]
            if phase == "shot":
                ball[0] += ball_vel[0] * dt
                ball[1] += ball_vel[1] * dt
                gx1, gy1, gx2, gy2 = goals[goal_state["side"]]
                inside = gx1 <= ball[0] <= gx2 and gy1 <= ball[1] <= gy2
                past = ball[0] < gx1 if goal_state["side"] == "left" else ball[0] > gx2
                if inside or past:
                    goal_events.append((t, goal_state["side"]))
                    goal_state["phase"] = "in_net"
                    goal_state["t_goal"] = t
                    ball[0] = min(max(ball[0], gx1 + 4), gx2 - 4)
                    ball_vel = [0.0, 0.0]
            elif phase == "in_net":
                if t - goal_state["t_goal"] > 2.0:
                    ball_visible = False
                    goal_state["phase"] = "reset"
            elif phase == "reset":
                if t - goal_state["t_goal"] > 4.0:
                    ball = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0]
                    ball_visible = True
                    goal_state = None
                    next_pass_t = t + 1.0
        else:
            # Passing play: fly toward the target player, then pick another.
            tp = players[ball_target_player]["pos"]
            dx, dy = tp[0] - ball[0], tp[1] - ball[1]
            d = math.hypot(dx, dy)
            if d < 14 or t >= next_pass_t:
                # choose a new teammate-ish target
                candidates = [i for i, p in enumerate(players) if p["team"] != TEAM_REFEREE and i != ball_target_player]
                ball_target_player = rng.choice(candidates)
                tp = players[ball_target_player]["pos"]
                dx, dy = tp[0] - ball[0], tp[1] - ball[1]
                d = max(1.0, math.hypot(dx, dy))
                pass_speed = 420.0 + rng.random() * 200.0
                ball_vel = [dx / d * pass_speed, dy / d * pass_speed]
                next_pass_t = t + d / pass_speed + 0.6 + rng.random() * 0.8
                # occasional brief occlusion of the ball (behind a player)
                if rng.random() < 0.15:
                    ball_hidden_until = t + 0.3
            ball[0] += ball_vel[0] * dt
            ball[1] += ball_vel[1] * dt
            ball[0] = min(max(x_min + 2, ball[0]), x_max - 2)
            ball[1] = min(max(y_min + 2, ball[1]), y_max - 2)

        ball_frames[f] = ball
        ball_vis[f] = ball_visible and t >= ball_hidden_until

    return pos_frames, ball_frames, ball_vis, goal_events, players


def _draw_frame(canvas: np.ndarray, spec: SyntheticMatchSpec, positions: np.ndarray,
                players_meta, ball_xy: np.ndarray, ball_visible: bool, cv2) -> None:
    (x_min, y_min, x_max, y_max), goals = _pitch_geometry(spec)
    canvas[:] = spec.grass_bgr
    # Pitch lines
    white = (235, 235, 235)
    cv2.rectangle(canvas, (int(x_min), int(y_min)), (int(x_max), int(y_max)), white, 2)
    cx = int((x_min + x_max) / 2)
    cv2.line(canvas, (cx, int(y_min)), (cx, int(y_max)), white, 2)
    cv2.circle(canvas, (cx, int((y_min + y_max) / 2)), int((y_max - y_min) * 0.12), white, 2)
    for box in goals.values():
        cv2.rectangle(canvas, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (200, 200, 200), 2)
    # Players (torso rectangle + head), drawn far-to-near by y for occlusion
    order = np.argsort(positions[:, 1])
    hw, hh = spec.player_w_px // 2, spec.player_h_px
    for i in order:
        px, py = positions[i]
        team = players_meta[i]["team"]
        color = spec.team_a_bgr if team == TEAM_A else spec.team_b_bgr if team == TEAM_B else spec.referee_bgr
        x1, y1, x2, y2 = int(px - hw), int(py - hh), int(px + hw), int(py)
        # legs/shorts
        cv2.rectangle(canvas, (x1, int(py - hh * 0.45)), (x2, y2), (30, 30, 30), -1)
        # jersey
        cv2.rectangle(canvas, (x1, int(py - hh * 0.85)), (x2, int(py - hh * 0.45)), color, -1)
        # head
        cv2.circle(canvas, (int(px), int(py - hh * 0.93)), max(3, hw // 2), (190, 160, 140), -1)
    if ball_visible:
        cv2.circle(canvas, (int(ball_xy[0]), int(ball_xy[1])), spec.ball_radius_px, (255, 255, 255), -1)
        cv2.circle(canvas, (int(ball_xy[0]), int(ball_xy[1])), spec.ball_radius_px, (0, 0, 0), 1)


def generate_synthetic_match(
    output_path: str | Path,
    spec: Optional[SyntheticMatchSpec] = None,
    *,
    prefer_ffmpeg: bool = False,
) -> SyntheticGroundTruth:
    """Render the synthetic match to ``output_path`` and return ground truth."""
    import cv2  # local import keeps module import cheap

    spec = spec or SyntheticMatchSpec()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions, balls, ball_vis, goal_events, players_meta = _simulate(spec)
    n_frames = positions.shape[0]

    canvas = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
    proc = None
    writer = None
    if prefer_ffmpeg:
        try:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{spec.width}x{spec.height}", "-r", f"{spec.fps:.4f}", "-i", "pipe:0",
                "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={spec.duration_s}",
                "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-movflags", "+faststart", str(output_path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            proc = None
    if proc is None:
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), spec.fps, (spec.width, spec.height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer for {output_path}")

    try:
        for f in range(n_frames):
            _draw_frame(canvas, spec, positions[f], players_meta, balls[f], bool(ball_vis[f]), cv2)
            if proc is not None and proc.stdin is not None:
                proc.stdin.write(canvas.tobytes())
            elif writer is not None:
                writer.write(canvas)
    finally:
        if proc is not None and proc.stdin is not None:
            proc.stdin.close()
            proc.wait()
        if writer is not None:
            writer.release()

    # Ground-truth tracking result in source pixel space.
    dt = 1.0 / spec.fps
    hw, hh = spec.player_w_px / 2.0, float(spec.player_h_px)
    players: Dict[int, PlayerTrack] = {}
    for i, meta in enumerate(players_meta):
        rows = []
        for f in range(n_frames):
            px, py = positions[f, i]
            rows.append((f * dt, px - hw, py - hh, px + hw, py, 1.0))
        players[i + 1] = player_track_from_rows(i + 1, rows, team=int(meta["team"]), team_confidence=1.0)
    ball_rows = [
        (f * dt, balls[f, 0], balls[f, 1], spec.ball_radius_px * 2, spec.ball_radius_px * 2, 1.0)
        for f in range(n_frames) if ball_vis[f]
    ]
    tracking = TrackingResult(
        fps=spec.fps,
        frame_size=(spec.width, spec.height),
        duration_s=n_frames * dt,
        players=players,
        ball=ball_detections_from_rows(ball_rows) if ball_rows else BallDetections.empty(),
        focus_track_id=1,
        detector={"name": "synthetic_ground_truth"},
        source_video_path=str(output_path),
        processing_video_path=str(output_path),
    )
    bounds, goal_boxes = _pitch_geometry(spec)
    return SyntheticGroundTruth(
        spec=spec,
        tracking=tracking,
        goal_times_s=[(round(t, 3), side) for t, side in goal_events],
        pitch_bounds_px=bounds,
        goal_boxes_px=goal_boxes,
    )
