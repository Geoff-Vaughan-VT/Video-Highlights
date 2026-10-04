from __future__ import annotations

import numpy as np
import pytest

from backend.services.broadcast import (
    BroadcastConfig,
    emotion_end,
    refine_intervals,
    story_start,
)
from backend.services.camera_planner import CameraPlannerConfig, plan_camera
from backend.services.game_tracking import (
    GameStateSegment,
    STATE_IN_PLAY,
    build_ball_track,
    estimate_field_geometry,
)

FRAME = (1920, 1080)


def _geometry():
    rng = np.random.default_rng(3)
    n = 3000
    cloud = np.stack(
        [rng.uniform(0, 60, n), rng.uniform(100, 1820, n), rng.uniform(200, 880, n)],
        axis=1,
    )
    return estimate_field_geometry(cloud, FRAME)


def _attack_track(t_turn=20.0, t_goal=28.0):
    """Ball drifts AWAY from the left goal, then turns and attacks it."""
    hz = 10.0
    rows = []
    for i in range(int(t_turn * hz)):
        t = i / hz
        rows.append((t, 600.0 + 20.0 * t, 500.0))  # moving right (away from left goal)
    x0 = 600.0 + 20.0 * t_turn
    for i in range(int((t_goal - t_turn) * hz)):
        t = t_turn + i / hz
        f = (t - t_turn) / (t_goal - t_turn)
        rows.append((t, x0 - (x0 - 160.0) * f, 500.0))  # attacking left
    return build_ball_track(rows, FRAME)


def test_story_start_walks_back_to_attack_origin() -> None:
    track = _attack_track(t_turn=20.0, t_goal=28.0)
    segments = [GameStateSegment(0.0, 30.0, STATE_IN_PLAY, reason="in play")]

    start = story_start(28.0, track, segments, "left")

    # The move began at ~20s (direction turn); with 1.5s preroll the clip
    # starts just before it - NOT at 28-2s and NOT at the 15s lookback cap.
    assert 17.5 <= start <= 21.0, f"story start {start}"


def test_story_start_respects_dead_ball_boundary() -> None:
    track = _attack_track(t_turn=20.0, t_goal=28.0)
    segments = [
        GameStateSegment(0.0, 24.0, "restart_left", side="left", reason="goal kick wait"),
        GameStateSegment(24.0, 30.0, STATE_IN_PLAY, reason="in play"),
    ]
    start = story_start(28.0, track, segments, "left")
    # Cannot start before play resumed at 24s (minus preroll).
    assert start >= 24.0 - BroadcastConfig().preroll_s - 1e-6
    assert start < 26.0


def test_emotion_end_waits_for_crowd_decay() -> None:
    times = np.arange(0.0, 40.0, 0.1)
    rms = np.full_like(times, 0.05)
    # Crowd erupts at t=10 and decays until t=19.
    surge = (times >= 10.0) & (times <= 19.0)
    rms[surge] = 0.5 - 0.05 * (times[surge] - 10.0)

    end = emotion_end(10.0, (times, rms), is_goal=True)

    cfg = BroadcastConfig()
    assert 15.0 <= end <= 10.0 + cfg.max_post_goal_s
    # And well past the minimum post window.
    assert end > 10.0 + cfg.min_post_s


def test_emotion_end_caps_and_handles_missing_audio() -> None:
    cfg = BroadcastConfig()
    assert emotion_end(10.0, None, is_goal=False) == 10.0 + cfg.max_post_s
    # Crowd never decays -> capped.
    times = np.arange(0.0, 60.0, 0.1)
    rms = np.where(times >= 10.0, 0.5, 0.05)
    assert emotion_end(10.0, (times, rms), is_goal=True) == 10.0 + cfg.max_post_goal_s


def test_refine_intervals_extends_goal_clip_and_merges_overlaps() -> None:
    track = _attack_track(t_turn=20.0, t_goal=28.0)
    segments = [GameStateSegment(0.0, 60.0, STATE_IN_PLAY, reason="in play")]
    times = np.arange(0.0, 60.0, 0.1)
    rms = np.full_like(times, 0.05)
    surge = (times >= 28.0) & (times <= 36.0)
    rms[surge] = 0.5 - 0.055 * (times[surge] - 28.0)

    refined = refine_intervals(
        intervals=[(26.0, 32.0), (33.0, 38.0)],
        event_rows=[{"t": 28.0, "event_type": "goal", "side": "left"}],
        ball_track=track,
        segments=segments,
        envelope=(times, rms),
        duration_s=60.0,
    )

    # Goal start walked back toward the attack origin and the two intervals
    # merged after the ending grew.
    assert len(refined) == 1
    assert refined[0][0] <= 21.0
    assert refined[0][1] >= 33.0


def test_camera_deadband_holds_aim_for_jitter() -> None:
    geometry = _geometry()
    hz = 15.0
    rng = np.random.default_rng(5)
    # Ball jitters within a few pixels of a fixed spot (dribbling in place).
    rows = [(i / hz, 800.0 + rng.uniform(-6, 6), 500.0 + rng.uniform(-6, 6))
            for i in range(int(6 * hz))]
    track = build_ball_track(rows, FRAME)
    segments = [GameStateSegment(0.0, 6.0, STATE_IN_PLAY, reason="in play")]

    plan = plan_camera(
        ball_track=track, player_positions=None, geometry=geometry,
        segments=segments, start_seconds=0.0, end_seconds=6.0, fps=30.0,
        frame_size=FRAME, base_zoom=1.8, config=CameraPlannerConfig(),
    )

    late = [d for d in plan.decisions if d.t >= 2.0]
    xs = np.array([d.center_x for d in late])
    ys = np.array([d.center_y for d in late])
    # The camera settles and rests: total wander stays within a few pixels.
    assert xs.max() - xs.min() < 8.0, f"camera hunted: {xs.max() - xs.min():.1f}px"
    assert ys.max() - ys.min() < 8.0
    zooms = np.array([d.zoom for d in late])
    assert zooms.max() - zooms.min() < 0.06


# ---------------------------------------------------------------------------
# Reel building (ffmpeg only - no moviepy)
# ---------------------------------------------------------------------------

import shutil
import subprocess

ffmpeg_required = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                     reason="ffmpeg/ffprobe not installed")


def _lavfi_clip(path, seconds, audio=True, size="160x90", rate=25, freq=440):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return str(path)


def _probe(path):
    from backend.services.event_clip_renderer import probe_media

    return probe_media(str(path))


@ffmpeg_required
def test_reel_crossfades_three_clips_including_one_without_audio(tmp_path) -> None:
    from backend.services.broadcast import build_broadcast_reel

    clips = [
        _lavfi_clip(tmp_path / "a.mp4", 3.0),
        _lavfi_clip(tmp_path / "b.mp4", 2.5, audio=False),
        _lavfi_clip(tmp_path / "c.mp4", 3.0, size="320x180", freq=660),  # different size: scaled
    ]
    out = tmp_path / "highlights_reel.mp4"
    result = build_broadcast_reel(
        [{"path": p, "start_s": 10.0 * i, "end_s": 10.0 * i + 3} for i, p in enumerate(clips)],
        str(out), crossfade_s=0.5,
    )
    assert result == str(out)
    info = _probe(out)
    assert info["has_video"] and info["has_audio"]
    assert (info["width"], info["height"]) == (160, 90)
    assert info["duration"] == pytest.approx(3.0 + 2.5 + 3.0 - 2 * 0.5, abs=0.2)


@ffmpeg_required
def test_reel_goal_replay_is_slow_motion(tmp_path) -> None:
    from backend.services.broadcast import BroadcastConfig, build_broadcast_reel

    goal = _lavfi_clip(tmp_path / "goal.mp4", 6.0)
    other = _lavfi_clip(tmp_path / "other.mp4", 4.0)
    out = tmp_path / "reel.mp4"
    specs = [
        {"path": goal, "start_s": 100.0, "end_s": 106.0, "event_type": "goal", "occurred_at_s": 104.0,
         "confidence": 0.95},
        {"path": other, "start_s": 200.0, "end_s": 204.0, "event_type": None, "confidence": 0.6},
    ]
    assert build_broadcast_reel(specs, str(out)) == str(out)
    cfg = BroadcastConfig()
    replay = (cfg.replay_pre_s + cfg.replay_post_s) / cfg.replay_speed  # 5 s at 0.5x = 10 s
    assert _probe(out)["duration"] == pytest.approx(6.0 + replay + 4.0 - 2 * 0.5, abs=0.25)
    # Without replays the goal clip plays once.
    plain = tmp_path / "plain.mp4"
    assert build_broadcast_reel(specs, str(plain), slowmo_replays=False) == str(plain)
    assert _probe(plain)["duration"] == pytest.approx(6.0 + 4.0 - 0.5, abs=0.2)


@ffmpeg_required
def test_reel_uses_concat_fades_for_many_clips(tmp_path) -> None:
    from backend.services.broadcast import BroadcastConfig, build_broadcast_reel

    clips = [_lavfi_clip(tmp_path / f"c{i}.mp4", 1.5, audio=(i % 2 == 0)) for i in range(4)]
    out = tmp_path / "reel.mp4"
    result = build_broadcast_reel([{"path": p} for p in clips], str(out), BroadcastConfig(max_xfade_clips=3))
    assert result == str(out)
    info = _probe(out)
    assert info["has_audio"]
    assert info["duration"] == pytest.approx(4 * 1.5, abs=0.25)  # faded, not overlapped


@ffmpeg_required
def test_reel_title_cards_and_cold_open(tmp_path) -> None:
    from backend.services.broadcast import BroadcastConfig, build_broadcast_reel

    clips = [_lavfi_clip(tmp_path / f"c{i}.mp4", 3.0) for i in range(2)]
    out = tmp_path / "reel.mp4"
    specs = [{"path": clips[0], "start_s": 0.0, "end_s": 3.0, "event_type": "shot", "confidence": 0.9,
              "occurred_at_s": 1.5},
             {"path": clips[1], "start_s": 5.0, "end_s": 8.0, "event_type": "corner_kick"}]
    assert build_broadcast_reel(specs, str(out), title_cards=True, cold_open=True) == str(out)
    cfg = BroadcastConfig()
    expected = cfg.cold_open_s + 2 * cfg.title_card_s + 2 * 3.0 - 4 * 0.5
    assert _probe(out)["duration"] == pytest.approx(expected, abs=0.25)


def test_build_broadcast_reel_empty_specs_returns_none(tmp_path) -> None:
    from backend.services.broadcast import build_broadcast_reel

    assert build_broadcast_reel([], str(tmp_path / "reel.mp4")) is None
    assert build_broadcast_reel([{"path": str(tmp_path / "missing.mp4")}], str(tmp_path / "r.mp4")) is None


# ---------------------------------------------------------------------------
# Audio envelope (no librosa)
# ---------------------------------------------------------------------------


def _write_wav(path, seconds=6.0, sr=16000, loud_at=(3.0, 4.0)):
    from scipy.io import wavfile

    t = np.arange(int(seconds * sr)) / sr
    y = 0.02 * np.sin(2 * np.pi * 220 * t)
    loud = (t >= loud_at[0]) & (t < loud_at[1])
    y[loud] = 0.5 * np.sin(2 * np.pi * 440 * t[loud])
    wavfile.write(str(path), sr, (y * 32767).astype(np.int16))


def test_audio_envelope_from_wav_finds_loud_section(tmp_path) -> None:
    from backend.services.broadcast import compute_audio_envelope

    wav = tmp_path / "audio_analysis.wav"
    _write_wav(wav)
    env = compute_audio_envelope(str(wav))
    assert env is not None
    times, rms = env
    assert len(times) == len(rms) and times[1] - times[0] == pytest.approx(0.1)
    assert times[-1] == pytest.approx(5.9, abs=0.15)
    loud = rms[(times > 3.2) & (times < 3.8)]
    quiet = rms[(times > 0.5) & (times < 2.5)]
    assert loud.min() > 10 * quiet.max()


def test_audio_envelope_prefers_audio_analysis_wav_next_to_video(tmp_path) -> None:
    from backend.services.broadcast import compute_audio_envelope

    _write_wav(tmp_path / "audio_analysis.wav", loud_at=(1.0, 2.0))
    video = tmp_path / "proxy_1080p.mp4"
    video.write_bytes(b"not a real video")  # must not be decoded
    env = compute_audio_envelope(str(video))
    assert env is not None
    times, rms = env
    assert times[int(np.argmax(rms))] == pytest.approx(1.5, abs=0.6)
    # Explicit audio_path wins.
    other = tmp_path / "other.wav"
    _write_wav(other, loud_at=(4.0, 5.0))
    times2, rms2 = compute_audio_envelope(str(video), audio_path=str(other))
    assert times2[int(np.argmax(rms2))] == pytest.approx(4.5, abs=0.6)


@ffmpeg_required
def test_audio_envelope_decodes_video_audio(tmp_path) -> None:
    from backend.services.broadcast import compute_audio_envelope

    clip = _lavfi_clip(tmp_path / "clip.mp4", 2.0)
    env = compute_audio_envelope(clip)
    assert env is not None and env[0][-1] == pytest.approx(2.0, abs=0.2)
    silent = _lavfi_clip(tmp_path / "silent.mp4", 1.0, audio=False)
    assert compute_audio_envelope(silent) is None
