from __future__ import annotations

import numpy as np
import pytest

from backend.services.camera_planner import CameraDecision, CameraPlan
from backend.services.camera_render import (
    annotate_wide_frame,
    annotate_zoomed_banner,
    render_camera_plan_video,
)
from backend.services.game_tracking import estimate_field_geometry

cv2 = pytest.importorskip("cv2")

WIDTH, HEIGHT = 128, 96
FPS = 5.0


def _write_source_video(path, frames=15):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    try:
        for index in range(frames):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            # Keep the ball inside the legal crop-center region for zoom=2.0
            # (x in [32, 96]) so the planned center is not clamped away.
            x = 40 + index * 3
            cv2.circle(frame, (x, HEIGHT // 2), 4, (0, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def _plan(frames=15, zoom=2.0, debug_state="in_play"):
    plan = CameraPlan(start_seconds=0.0, fps=FPS, frame_size=(WIDTH, HEIGHT), base_zoom=zoom)
    for index in range(frames):
        t = index / FPS
        x = 40.0 + index * 3.0
        plan.decisions.append(
            CameraDecision(
                index=index,
                t=t,
                center_x=x,
                center_y=HEIGHT / 2.0,
                zoom=zoom,
                state=debug_state,
                focus="ball",
                reason="following ball",
                confidence=0.9,
                ball_x=x,
                ball_y=HEIGHT / 2.0,
                ball_source="detected",
                target_x=x,
                target_y=HEIGHT / 2.0,
            )
        )
    return plan


def test_render_camera_plan_video_zoomed_output(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "zoomed.mp4"
    _write_source_video(source)

    rendered = render_camera_plan_video(
        video_path=str(source),
        output_path=str(output),
        plan=_plan(),
        include_audio=False,
    )

    assert rendered == str(output.resolve())
    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()
    assert ok
    assert frame_count >= 14
    # Ball was centered by the crop: bright pixels near frame center.
    center = first[HEIGHT // 2 - 8 : HEIGHT // 2 + 8, WIDTH // 2 - 8 : WIDTH // 2 + 8]
    assert float(center[:, :, 1].mean()) > 20.0


def test_render_camera_plan_video_debug_wide(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "debug.mp4"
    _write_source_video(source)
    geometry = estimate_field_geometry(None, (WIDTH, HEIGHT))

    rendered = render_camera_plan_video(
        video_path=str(source),
        output_path=str(output),
        plan=_plan(debug_state="restart_right"),
        include_audio=False,
        debug_wide=True,
        geometry=geometry,
    )

    assert rendered == str(output.resolve())
    cap = cv2.VideoCapture(str(output))
    ok, first = cap.read()
    cap.release()
    assert ok
    # The banner darkens/annotates the top strip; it must not be all black
    # and must differ from the raw source frame (which was black up top).
    banner = first[0:10, :]
    assert float(banner.mean()) > 1.0


def test_annotate_wide_frame_draws_overlay_and_banner() -> None:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    geometry = estimate_field_geometry(None, (WIDTH, HEIGHT))
    decision = _plan(frames=1).decisions[0]

    out = annotate_wide_frame(
        frame,
        decision,
        ball_trail=[(5.0, 40.0), (10.0, 45.0), (15.0, 48.0)],
        geometry=geometry,
    )

    assert out is frame
    assert float(frame.mean()) > 0.5  # something was drawn
    # Crosshair arm at the camera center in magenta (sample just outside the
    # yellow ball/target markers that sit on the exact center pixel).
    cx, cy = int(decision.center_x), int(decision.center_y)
    assert frame[cy, cx + 8, 0] > 100 and frame[cy, cx + 8, 2] > 100


def test_annotate_zoomed_banner_writes_reason_strip() -> None:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    decision = _plan(frames=1).decisions[0]
    annotate_zoomed_banner(frame, decision)
    assert float(frame[-int(HEIGHT * 0.05) :, :].mean()) > 0.5


def test_render_fails_cleanly_on_missing_video(tmp_path) -> None:
    with pytest.raises(RuntimeError):
        render_camera_plan_video(
            video_path=str(tmp_path / "missing.mp4"),
            output_path=str(tmp_path / "out.mp4"),
            plan=_plan(),
            include_audio=False,
        )


def test_scorebug_renders_score_clock_and_goal_flash() -> None:
    from backend.services.camera_render import make_scorebug_renderer

    fn = make_scorebug_renderer(
        [{"t": 100.0, "side": "left"}, {"t": 200.0, "side": "right"}],
        team_left="LIONS", team_right="HAWKS",
    )
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    out = fn(frame, 102.0)  # 2s after a goal into the LEFT goal
    assert out is frame
    # Scorebug strip drawn top-left...
    assert float(frame[8:30, 8:200].mean()) > 2.0
    # ...and the GOAL flash is red-dominant somewhere mid-frame.
    band = frame[30:120, :]
    assert int((band[:, :, 2].astype(int) - band[:, :, 1]).max()) > 100

    # Score changes over time: before any goal, after 1, after 2.
    for t, expect in ((50.0, (0, 0)), (150.0, (0, 1)), (250.0, (1, 1))):
        f2 = np.zeros((360, 640, 3), dtype=np.uint8)
        fn(f2, t)  # smoke: no exception; score text differs per t


# ---------------------------------------------------------------------------
# ffmpeg-native engine
# ---------------------------------------------------------------------------

import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

from backend.services.camera_render import ScorebugRenderer, make_scorebug_renderer, probe_video  # noqa: E402

QW, QH, QFPS = 640, 360, 25.0
# BGR means of the quadrants as decoded (lavfi red / green / blue / white).
RED, GREEN, BLUE, WHITE = (0, 0, 255), (0, 128, 0), (255, 0, 0), (255, 255, 255)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def quadrant_source(tmp_path_factory):
    """8 s, 25 fps, 640x360: red | green / blue | white quadrants + 440 Hz audio."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("quad") / "quadrants.mp4"
    colors = ["red", "green", "blue", "white"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for c in colors:
        cmd += ["-f", "lavfi", "-i", f"color=c={c}:s=320x180:r=25:d=8"]
    cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=8",
            "-filter_complex", "[0][1][2][3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]",
            "-map", "[v]", "-map", "4:a", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
            "-pix_fmt", "yuv420p", "-g", "25", "-c:a", "aac", str(path)]
    subprocess.run(cmd, check=True)
    return path


def _quadrant_plan(start=1.0, seconds=4.0, jump_at=3.0, output_size=(320, 180)):
    """Zoom 2 crop on the red quadrant, jumping to the white one at ``jump_at``."""
    plan = CameraPlan(start_seconds=start, fps=QFPS, frame_size=(QW, QH), base_zoom=2.0,
                      output_size=output_size)
    for i in range(int(round(seconds * QFPS))):
        t = start + i / QFPS
        cx, cy = (160.0, 90.0) if t < jump_at - 1e-9 else (480.0, 270.0)
        plan.decisions.append(CameraDecision(index=i, t=t, center_x=cx, center_y=cy, zoom=2.0,
                                             state="in_play", focus="ball", reason="test", confidence=1.0))
    return plan


def _frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _center_mean(frame):
    h, w = frame.shape[:2]
    return frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4].reshape(-1, 3).mean(axis=0)


def _is(color, mean, tol=40):
    return all(abs(float(m) - c) <= tol for m, c in zip(mean, color))


def _has_audio(path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True).stdout
    return "audio" in out.split()


@needs_ffmpeg
@pytest.mark.parametrize("filter_mode", ["crop", "scale"])
def test_ffmpeg_render_crop_timing_size_and_audio(tmp_path, quadrant_source, filter_mode) -> None:
    plan = _quadrant_plan()
    output = tmp_path / f"follow_{filter_mode}.mp4"
    calls = []
    started = time.monotonic()
    rendered = render_camera_plan_video(
        video_path=str(quadrant_source), output_path=str(output), plan=plan, include_audio=True,
        progress_callback=lambda done, total: calls.append((done, total)), filter_mode=filter_mode,
    )
    elapsed = time.monotonic() - started
    assert rendered == str(output.resolve())
    info = probe_video(str(output))
    assert (info.width, info.height) == (320, 180)
    assert info.has_audio
    frames = _frames(output)
    assert abs(len(frames) - len(plan)) <= 1
    # Plan t=3.0 s is output t=2.0 s (frame 50): -ss before -i + sendcmd
    # times relative to the seek point must line up exactly.
    for k in (0, 25, 48, 49):
        assert _is(RED, _center_mean(frames[k])), (k, _center_mean(frames[k]))
    for k in (50, 51, 75, len(frames) - 1):
        assert _is(WHITE, _center_mean(frames[k])), (k, _center_mean(frames[k]))
    assert calls and calls[-1] == (len(plan), len(plan))
    assert elapsed < 30.0
    assert not list(tmp_path.glob(".follow_*"))  # work dir cleaned up


@needs_ffmpeg
def test_ffmpeg_render_chunked_concat_keeps_timing_and_audio(tmp_path, quadrant_source) -> None:
    # 6 s plan rendered as 1 s chunks + concat; crop jumps at plan t=4.2 s
    # (inside a chunk) and the zoom changes size at t=2.0 s.
    plan = _quadrant_plan(start=0.5, seconds=6.0, jump_at=4.2)
    for d in plan.decisions:
        if d.t < 2.0:
            d.zoom, d.center_x, d.center_y = 1.0, 320.0, 180.0  # whole frame
    infos = []
    output = tmp_path / "chunked.mp4"
    render_camera_plan_video(
        video_path=str(quadrant_source), output_path=str(output), plan=plan, include_audio=True,
        chunk_seconds=1.0, workers=2, progress_info_callback=infos.append,
    )
    frames = _frames(output)
    assert abs(len(frames) - len(plan)) <= 1
    assert _probe_duration(output, "a") == pytest.approx(len(plan) / QFPS, abs=0.1)
    jump_full = next(i for i, d in enumerate(plan.decisions) if d.t >= 2.0)
    jump_white = next(i for i, d in enumerate(plan.decisions) if d.center_x > 400.0)
    mixed = np.mean([RED, GREEN, BLUE, WHITE], axis=0)
    assert np.abs(frames[0].reshape(-1, 3).mean(axis=0) - mixed).max() < 30
    assert np.abs(frames[jump_full - 1].reshape(-1, 3).mean(axis=0) - mixed).max() < 30
    assert _is(RED, _center_mean(frames[jump_full]))
    assert _is(RED, _center_mean(frames[jump_white - 1]))
    assert _is(WHITE, _center_mean(frames[jump_white]))
    assert infos and {"frame", "total", "fps", "eta_s"} <= set(infos[-1])


def _probe_duration(path, stream: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", f"{stream}:0", "-show_entries",
                          "stream=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return float(out.strip().splitlines()[0])


@needs_ffmpeg
def test_ffmpeg_render_with_source_offset_and_rescaled_plan(tmp_path, quadrant_source) -> None:
    # Plan made in "4K" coordinates (2x the file) and in the processing
    # timebase that starts 1.0 s into the file (trim offset).
    plan = CameraPlan(start_seconds=0.0, fps=QFPS, frame_size=(QW * 2, QH * 2), base_zoom=2.0)
    for i in range(50):
        t = i / QFPS
        cx, cy = (960.0, 180.0) if t < 1.0 else (320.0, 540.0)  # green, then blue
        plan.decisions.append(CameraDecision(index=i, t=t, center_x=cx, center_y=cy, zoom=2.0,
                                             state="in_play", focus="ball", reason="t", confidence=1.0))
    output = tmp_path / "offset.mp4"
    render_camera_plan_video(video_path=str(quadrant_source), output_path=str(output), plan=plan,
                             include_audio=False, source_time_offset=1.0, output_size=(320, 180))
    frames = _frames(output)
    assert len(frames) == 50
    assert _is(GREEN, _center_mean(frames[24])) and _is(BLUE, _center_mean(frames[25]))
    assert not _has_audio(output)


@needs_ffmpeg
def test_ffmpeg_render_draws_scorebug_with_drawtext(tmp_path, quadrant_source) -> None:
    plan = _quadrant_plan(start=1.0, seconds=2.0, jump_at=99.0)
    plain = tmp_path / "plain.mp4"
    with_bug = tmp_path / "scorebug.mp4"
    render_camera_plan_video(video_path=str(quadrant_source), output_path=str(plain), plan=plan,
                             include_audio=False)
    scorebug = make_scorebug_renderer([{"t": 1.5, "side": "left"}], team_left="LIONS", team_right="HAWKS")
    assert isinstance(scorebug, ScorebugRenderer)
    render_camera_plan_video(video_path=str(quadrant_source), output_path=str(with_bug), plan=plan,
                             include_audio=False, scorebug_fn=scorebug)
    a, b = _frames(plain), _frames(with_bug)
    assert len(a) == len(b) == len(plan)
    # Scorebug box top-left darkens the red crop; the goal flash appears
    # only after the goal (plan t=1.5 -> frame 12).
    top_left = (slice(0, 30), slice(0, 160))
    assert np.abs(a[5][top_left].astype(int) - b[5][top_left].astype(int)).mean() > 20
    flash = (slice(40, 90), slice(60, 260))
    before = np.abs(a[8][flash].astype(int) - b[8][flash].astype(int)).mean()
    after = np.abs(a[20][flash].astype(int) - b[20][flash].astype(int)).mean()
    assert after > before + 5


@needs_ffmpeg
def test_python_engine_writes_output_size_with_area_downscale(tmp_path, quadrant_source) -> None:
    plan = _quadrant_plan()
    output = tmp_path / "python.mp4"
    render_camera_plan_video(video_path=str(quadrant_source), output_path=str(output), plan=plan,
                             include_audio=True, engine="python", output_size=(256, 144))
    info = probe_video(str(output))
    assert (info.width, info.height) == (256, 144)
    assert info.has_audio
    frames = _frames(output)
    assert abs(len(frames) - len(plan)) <= 1
    assert _is(RED, _center_mean(frames[40])) and _is(WHITE, _center_mean(frames[60]))


@needs_ffmpeg
def test_debug_wide_renders_from_proxy_source(tmp_path, quadrant_source) -> None:
    proxy = tmp_path / "proxy.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(quadrant_source),
                    "-vf", "scale=320:180", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                    str(proxy)], check=True)
    plan = _quadrant_plan()  # plan in 640x360 source coordinates
    output = tmp_path / "debug.mp4"
    render_camera_plan_video(video_path=str(quadrant_source), output_path=str(output), plan=plan,
                             include_audio=False, debug_wide=True, debug_source_path=str(proxy),
                             max_debug_width=1280)
    info = probe_video(str(output))
    assert (info.width, info.height) == (320, 180)  # proxy size, not the source size
    frames = _frames(output)
    assert abs(len(frames) - len(plan)) <= 1
    # Crop rectangle drawn in scaled coordinates: the red quadrant's crop box
    # outline (state colour green-ish) sits inside the top-left quadrant.
    assert float(frames[10][0:20, :].mean()) > 1.0


# ---------------------------------------------------------------------------
# Scorebug attribution, rotated sources, cancel
# ---------------------------------------------------------------------------


def test_scorebug_credits_scoring_team_across_half_time_swap() -> None:
    # HOME (team 0, left label) attacks the right goal in the first half and
    # the LEFT goal after the half-time swap; AWAY (team 1) the opposite.
    goals = [
        {"t": 600.0, "side": "right", "team": 0},   # 1H: HOME into the right goal
        {"t": 3300.0, "side": "left", "team": 0},   # 2H: HOME into the left goal
        {"t": 4000.0, "side": "right", "team": 1},  # 2H: AWAY into the right goal
    ]
    bug = make_scorebug_renderer(goals, team_left="HOME", team_right="AWAY")
    assert bug.score_at(0.0) == (0, 0)
    assert bug.score_at(700.0) == (1, 0)
    assert bug.score_at(3400.0) == (2, 0)  # side-only attribution would say (1, 1)
    assert bug.score_at(4100.0) == (2, 1)
    # Unknown team: the goal side decides (into the left goal -> right label).
    legacy = make_scorebug_renderer([{"t": 10.0, "side": "left", "team": None},
                                     {"t": 20.0, "side": "right"}])
    assert legacy.score_at(30.0) == (1, 1) and legacy.score_at(15.0) == (0, 1)
    # The drawtext text files follow the same score.
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as work:
        bug.ffmpeg_filters(_P(work), "t", 3350.0, 10.0, (640, 360), None)
        texts = [p.read_text() for p in sorted(_P(work).glob("sb_t_*.txt"))]
    assert any("HOME 2 - 0 AWAY" in t for t in texts), texts


@needs_ffmpeg
def test_ffmpeg_render_handles_rotated_source(tmp_path, quadrant_source) -> None:
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "90",
                    "-i", str(quadrant_source), "-c", "copy", str(rotated)], check=True)
    info = probe_video(str(rotated))
    assert (info.width, info.height) == (QH, QW)  # display orientation (portrait)

    # Plan in display pixels (360x640), zoom 2 around a point in the lower half.
    dw, dh = QH, QW
    plan = CameraPlan(start_seconds=1.0, fps=QFPS, frame_size=(dw, dh), base_zoom=2.0, output_size=(160, 90))
    for i in range(25):
        plan.decisions.append(CameraDecision(index=i, t=1.0 + i / QFPS, center_x=dw * 0.25, center_y=dh * 0.75,
                                             zoom=2.0, state="in_play", focus="ball", reason="t", confidence=1.0))
    output = tmp_path / "rotated_follow.mp4"
    render_camera_plan_video(video_path=str(rotated), output_path=str(output), plan=plan, include_audio=False)
    out_info = probe_video(str(output))
    assert (out_info.width, out_info.height) == (160, 90)
    frames = _frames(output)
    assert abs(len(frames) - len(plan)) <= 1

    # Expected content: the same crop of the autorotated (display) frame.
    cap = cv2.VideoCapture(str(rotated))
    ok, display = cap.read()
    cap.release()
    assert ok and display.shape[:2] == (dh, dw)
    x, y, w, h = (int(v) for v in plan.get_crop_rects()[0])
    expected = display[y:y + h, x:x + w].reshape(-1, 3).mean(axis=0)
    got = frames[0].reshape(-1, 3).mean(axis=0)
    assert np.abs(expected - got).max() < 30, (expected, got)


@needs_ffmpeg
def test_chunked_render_cancel_stops_queued_chunks(tmp_path, quadrant_source, monkeypatch) -> None:
    import threading

    from backend.services import camera_render as cr

    plan = _quadrant_plan(start=0.5, seconds=6.0, jump_at=99.0)
    cancel = threading.Event()
    spawns: list = []
    lock = threading.Lock()
    original_popen = subprocess.Popen

    class CountingPopen(original_popen):  # type: ignore[misc, valid-type]
        def __init__(self, args, *a, **kw):
            if isinstance(args, (list, tuple)) and "-progress" in [str(x) for x in args]:
                with lock:
                    spawns.append((time.monotonic(), cancel.is_set()))
            super().__init__(args, *a, **kw)

    first_chunk = int(round(1.0 * QFPS))
    cancelled_at: list = []

    def on_info(info) -> None:
        if not cancel.is_set() and info["frame"] >= first_chunk:
            cancelled_at.append(time.monotonic())
            cancel.set()

    monkeypatch.setattr(subprocess, "Popen", CountingPopen)
    with pytest.raises(cr.RenderCancelled):
        render_camera_plan_video(video_path=str(quadrant_source), output_path=str(tmp_path / "c.mp4"), plan=plan,
                                 include_audio=True, chunk_seconds=1.0, workers=2, progress_info_callback=on_info,
                                 cancel_event=cancel)
    returned = time.monotonic()
    assert cancelled_at, "the first chunk never completed"
    assert len(spawns) == 1, spawns  # only the first chunk's ffmpeg ever ran
    assert not any(after for _, after in spawns)
    assert returned - cancelled_at[0] < 2.0
    assert not (tmp_path / "c.mp4").exists()
    assert not list(tmp_path.glob(".c_render_*"))  # work dir cleaned up

    # Already-set cancel: nothing is spawned at all.
    spawns.clear()
    with pytest.raises(cr.RenderCancelled):
        render_camera_plan_video(video_path=str(quadrant_source), output_path=str(tmp_path / "d.mp4"), plan=plan,
                                 include_audio=False, chunk_seconds=1.0, workers=2, cancel_event=cancel)
    assert spawns == []


@needs_ffmpeg
def test_chunked_render_failure_stops_queued_chunks(tmp_path, quadrant_source, monkeypatch) -> None:
    import threading

    from backend.services import camera_render as cr

    plan = _quadrant_plan(start=0.5, seconds=6.0, jump_at=99.0)
    original = cr._run_ffmpeg
    calls: list = []
    lock = threading.Lock()

    def flaky(cmd, cwd, on_frames=None, stall_timeout_s=None):
        target = str(cmd[-1])
        with lock:
            calls.append(target)
        if target.startswith("chunk_0002"):
            return 1, "simulated encoder crash"
        return original(cmd, cwd, on_frames, stall_timeout_s=stall_timeout_s)

    monkeypatch.setattr(cr, "_run_ffmpeg", flaky)
    with pytest.raises(RuntimeError, match="simulated encoder crash"):
        render_camera_plan_video(video_path=str(quadrant_source), output_path=str(tmp_path / "f.mp4"), plan=plan,
                                 include_audio=False, chunk_seconds=1.0, workers=1)
    # Sequential: chunks after the failing one (3..5) never start; the failing
    # chunk tries its fallback combinations only.
    assert not any(c.startswith(("chunk_0003", "chunk_0004", "chunk_0005")) for c in calls), calls
