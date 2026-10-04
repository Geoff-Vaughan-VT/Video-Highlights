from __future__ import annotations

import numpy as np
import pytest

from backend.services.follow_cam import build_follow_cam_centers, crop_frame_to_center, render_follow_cam_clip


def test_build_follow_cam_centers_tracks_player_motion() -> None:
    player_track = [
        (0.0, 120.0, 80.0),
        (1.0, 280.0, 80.0),
    ]

    centers = build_follow_cam_centers(
        player_track=player_track,
        ball_track=None,
        start_seconds=0.0,
        end_seconds=1.0,
        fps=2.0,
        frame_size=(400, 200),
        zoom_factor=2.0,
        smooth_factor=1.0,
    )

    assert len(centers) == 2
    assert centers[0][0] < centers[1][0]
    assert centers[0][1] == centers[1][1] == 80.0


def test_build_follow_cam_centers_follow_action_blends_ball_position() -> None:
    player_track = [
        (0.0, 180.0, 100.0),
        (1.0, 180.0, 100.0),
    ]
    ball_track = [
        (0.0, 300.0, 100.0),
        (1.0, 300.0, 100.0),
    ]

    wide_centers = build_follow_cam_centers(
        player_track=player_track,
        ball_track=ball_track,
        start_seconds=0.0,
        end_seconds=1.0,
        fps=1.0,
        frame_size=(400, 200),
        zoom_factor=1.6,
        ball_weight=0.0,
        smooth_factor=1.0,
    )
    action_centers = build_follow_cam_centers(
        player_track=player_track,
        ball_track=ball_track,
        start_seconds=0.0,
        end_seconds=1.0,
        fps=1.0,
        frame_size=(400, 200),
        zoom_factor=1.6,
        ball_weight=0.5,
        smooth_factor=1.0,
    )

    assert action_centers[0][0] > wide_centers[0][0]


def test_crop_frame_to_center_preserves_output_size() -> None:
    frame = np.arange(80 * 40 * 3, dtype=np.uint8).reshape((40, 80, 3))
    cropped = crop_frame_to_center(frame, center=(60.0, 20.0), zoom_factor=2.0)

    assert cropped.shape == frame.shape


def test_build_follow_cam_centers_recenters_when_player_track_goes_stale() -> None:
    player_track = [
        (0.0, 80.0, 100.0),
    ]

    centers = build_follow_cam_centers(
        player_track=player_track,
        ball_track=None,
        start_seconds=0.0,
        end_seconds=2.1,
        fps=1.0,
        frame_size=(400, 200),
        zoom_factor=2.0,
        smooth_factor=1.0,
        max_player_gap_seconds=0.25,
    )

    assert centers[0] == (100.0, 100.0)
    assert centers[1][0] > centers[0][0]
    # v2 smoothing (zero-phase low-pass + speed/accel limiter) reaches the
    # frame center and holds there; the legacy per-frame max_step crawl that
    # made this strictly increasing is gone (it was a source of judder).
    assert centers[2][0] >= centers[1][0]
    assert centers[2][0] <= 200.0
    assert centers[1][1] == centers[2][1] == 100.0


def test_render_follow_cam_clip_writes_zoomed_video(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "zoomed.mp4"
    fps = 5.0
    width, height = 64, 48
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()

    player_track = []
    try:
        for frame_index in range(12):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 16 + (frame_index * 3)
            y = 24
            cv2.rectangle(frame, (x - 3, y - 3), (x + 3, y + 3), (0, 255, 0), thickness=-1)
            writer.write(frame)
            player_track.append((frame_index / fps, float(x), float(y)))
    finally:
        writer.release()

    rendered = render_follow_cam_clip(
        video_path=str(source_path),
        output_path=str(output_path),
        start_seconds=0.0,
        end_seconds=2.0,
        player_track=player_track,
        ball_track=None,
        zoom_factor=2.0,
        include_audio=False,
    )

    assert rendered == str(output_path.resolve())
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    cap = cv2.VideoCapture(str(output_path))
    ok, first_frame = cap.read()
    cap.release()

    assert ok
    center_region = first_frame[height // 2 - 4 : height // 2 + 5, width // 2 - 4 : width // 2 + 5]
    assert float(center_region[:, :, 1].mean()) > 60.0


def test_build_follow_cam_centers_v2_is_smooth_and_speed_limited() -> None:
    # Player track teleports across the pitch (ID switch): v1 chased it with
    # a per-frame max step; v2 glides within the pan speed/accel limits.
    fps = 25.0
    track = [(i / fps, 300.0 if i < 50 else 1600.0, 540.0) for i in range(150)]
    centers = build_follow_cam_centers(
        player_track=track, ball_track=None, start_seconds=0.0, end_seconds=6.0, fps=fps,
        frame_size=(1920, 1080), zoom_factor=1.6, smooth_factor=0.24,
    )
    xs = np.array([c[0] for c in centers])
    crop_w = 1920 / 1.6
    speed = np.abs(np.diff(xs)) * fps
    accel = np.abs(np.diff(xs, 2)) * fps * fps
    assert speed.max() <= 1.0 * crop_w * 1.01
    assert accel.max() <= 1.2 * crop_w * 1.05
    assert xs[-1] == pytest.approx(1920 - crop_w / 2, abs=2.0)  # arrived (clamped at the edge)


def test_render_follow_cam_clip_ffmpeg_output_size_and_audio(tmp_path) -> None:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    from backend.services.camera_render import probe_video

    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=s=640x360:r=25:d=5", "-f", "lavfi", "-i", "sine=frequency=330:duration=5",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(source)],
        check=True,
    )
    output = tmp_path / "clip.mp4"
    track = [(t / 10.0, 200.0 + 40.0 * t / 10.0, 180.0) for t in range(50)]
    rendered = render_follow_cam_clip(
        video_path=str(source), output_path=str(output), start_seconds=1.0, end_seconds=3.0,
        player_track=track, zoom_factor=1.6, include_audio=True, output_size=(320, 180),
    )
    assert rendered == str(output.resolve())
    info = probe_video(str(output))
    assert (info.width, info.height) == (320, 180)
    assert info.has_audio
    assert info.duration == pytest.approx(2.0, abs=0.15)
