"""Legacy spotlight overlay: drawn on the proxy with source-pixel tracks scaled
and in the video's own timebase (no trim misalignment)."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


def test_overlay_scales_source_pixels_and_uses_window_time(tmp_path) -> None:
    import VideoHighlights as vh

    proxy = tmp_path / "proxy.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=black:s=320x180:r=25:d=3", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(proxy)], check=True)
    # Track in SOURCE pixels (640x360, proxy scale 0.5) and window seconds:
    # left side before t=1.5, right side after.
    traj = [vh.TrackPoint(t=k / 25.0, xy=(100.0, 180.0) if k / 25.0 < 1.5 else (540.0, 180.0))
            for k in range(75)]
    out = vh.draw_single_spotlight_overlay(str(proxy), traj, (1.6, 2.4), 1, str(tmp_path), radius=12,
                                           xy_scale=0.5)
    assert out is not None
    cap = cv2.VideoCapture(out)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (180, 320)
    bright = np.argwhere(frame.max(axis=2) > 128)
    assert bright.size, "no spotlight drawn"
    cx = float(bright[:, 1].mean())
    # Window t=1.6 -> the right-side point, at x = 540 * 0.5 = 270 proxy px.
    assert abs(cx - 270.0) < 8.0, cx
    assert abs(float(bright[:, 0].mean()) - 90.0) < 8.0
