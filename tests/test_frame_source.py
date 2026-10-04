from __future__ import annotations

import shutil
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from backend.services.frame_source import (
    FrameReader,
    ProxyCancelled,
    build_proxy,
    probe_video,
    proxy_result_from_file,
    read_frames_at,
    snap_fps,
)
from backend.services.synthetic_match import SyntheticMatchSpec, generate_synthetic_match

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not installed")

SPEC = SyntheticMatchSpec(width=1280, height=720, fps=25.0, duration_s=6.0, seed=3, goals=[(2.5, "right")])


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("frame_source")
    src = root / "match.mp4"
    generate_synthetic_match(src, SPEC, prefer_ffmpeg=True)  # H.264 + AAC audio
    events = []
    proxy = build_proxy(src, root / "run", height=360, trim_start=1.0, trim_end=5.0,
                        progress_cb=lambda *args: events.append(args))
    return {"root": root, "src": src, "proxy": proxy, "events": events}


def test_probe_video_reports_stream_facts(media) -> None:
    info = probe_video(media["src"])
    assert (info.width, info.height) == (1280, 720)
    assert info.fps == pytest.approx(25.0)
    assert info.duration_s == pytest.approx(6.0, abs=0.1)
    assert info.frame_count == 150
    assert info.codec == "h264"
    assert info.has_audio and info.audio_codec == "aac"
    assert not info.is_vfr
    assert info.rotation == 0
    with pytest.raises(FileNotFoundError):
        probe_video(media["root"] / "missing.mp4")


def test_build_proxy_outputs_cfr_trimmed_proxy_audio_and_thumbs(media) -> None:
    proxy = media["proxy"]
    path = Path(proxy.path)
    assert path.name == "proxy_360p.mp4" and path.is_file()
    assert (proxy.width, proxy.height) == (640, 360)
    assert proxy.scale == pytest.approx(0.5)
    assert (proxy.source_width, proxy.source_height) == (1280, 720)
    assert proxy.trim_start_s == pytest.approx(1.0)
    assert proxy.encoder_used in {"libx264", "h264_nvenc", "h264_videotoolbox"}

    info = probe_video(path)
    assert info.fps == pytest.approx(25.0)
    assert info.r_frame_rate == pytest.approx(info.avg_frame_rate)  # constant frame rate
    assert not info.is_vfr
    assert info.duration_s == pytest.approx(4.0, abs=0.1)
    assert info.frame_count == 100  # frame index == t * fps
    assert info.has_audio and info.audio_codec == "aac"
    assert info.pix_fmt == "yuv420p"

    with wave.open(str(proxy.audio_path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000
        assert wav.getsampwidth() == 2
        assert wav.getnframes() / 16000.0 == pytest.approx(4.0, abs=0.15)

    thumbs = sorted(Path(proxy.thumbs_dir).glob("*.jpg"))
    assert [t.name for t in thumbs][:1] == ["0000.jpg"]


def test_build_proxy_reports_progress_with_eta(media) -> None:
    events = media["events"]
    assert events, "progress callback never called"
    stages = {e[0] for e in events}
    assert stages == {"proxy"}
    fractions = [e[1] for e in events]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert any("eta_s" in (e[3] or {}) for e in events)


def test_trimmed_proxy_starts_at_trim_point(media) -> None:
    # The synthetic match is deterministic: proxy frame 0 == source frame 25 (t=1.0 s).
    src_frames = read_frames_at(media["src"], [25])
    proxy_frames = read_frames_at(media["proxy"].path, [0])
    import cv2

    src_small = cv2.resize(src_frames[25], (640, 360), interpolation=cv2.INTER_AREA).astype(np.float32)
    diff_same = float(np.abs(src_small - proxy_frames[0].astype(np.float32)).mean())
    other = cv2.resize(read_frames_at(media["src"], [60])[60], (640, 360),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    diff_other = float(np.abs(other - proxy_frames[0].astype(np.float32)).mean())
    assert diff_same < diff_other


def test_build_proxy_without_audio_writes_silent_wav(tmp_path: Path) -> None:
    src = tmp_path / "noaudio.mp4"
    generate_synthetic_match(src, SyntheticMatchSpec(width=320, height=180, duration_s=2.0, player_w_px=6,
                                                     player_h_px=14, ball_radius_px=2, goals=[]))
    proxy = build_proxy(src, tmp_path / "run", height=1080, thumbs=False, hwaccel="none")
    assert (proxy.width, proxy.height) == (320, 180)  # never upscales
    assert proxy.scale == pytest.approx(1.0)
    assert not proxy.has_audio
    assert proxy.thumbs_dir is None
    with wave.open(str(proxy.audio_path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnframes() / 16000.0 == pytest.approx(2.0, abs=0.15)


def test_vfr_source_becomes_cfr_proxy(tmp_path: Path) -> None:
    src = tmp_path / "vfr.mp4"
    # 30 fps for 2 s, then 15 fps for 2 s: r_frame_rate 30, avg 22.5
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc=size=320x240:rate=30:duration=3", "-vf",
         "setpts='if(lt(N,60),N/30,2+(N-60)/15)/TB'", "-fps_mode", "vfr", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    info = probe_video(src)
    assert info.is_vfr
    proxy = build_proxy(src, tmp_path / "run", height=240, thumbs=False)
    out = probe_video(proxy.path)
    assert not out.is_vfr
    assert out.fps == pytest.approx(proxy.fps)
    assert out.frame_count == pytest.approx(out.duration_s * out.fps, abs=1.5)


def test_build_proxy_cancel_raises(tmp_path: Path, media) -> None:
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ProxyCancelled):
        build_proxy(media["src"], tmp_path / "cancelled", height=360, cancel_event=cancel)


def test_build_proxy_rejects_bad_trim(tmp_path: Path, media) -> None:
    with pytest.raises(ValueError):
        build_proxy(media["src"], tmp_path / "bad", trim_start=4.0, trim_end=2.0)


def test_snap_fps() -> None:
    assert snap_fps(29.95) == pytest.approx(29.97)
    assert snap_fps(59.7) == pytest.approx(59.94)
    assert snap_fps(22.5) == pytest.approx(22.5)


def test_proxy_result_from_file(media) -> None:
    reuse = proxy_result_from_file(media["proxy"].path, source_size=(1280, 720), trim_start_s=1.0)
    assert reuse.scale == pytest.approx(0.5)
    assert reuse.fps == pytest.approx(25.0)
    assert reuse.frame_count == 100
    # The analysis wav and thumbnails written next to the proxy are found again.
    assert reuse.has_audio
    assert reuse.audio_path == media["proxy"].audio_path
    assert Path(reuse.audio_path).name == "audio_analysis.wav"
    assert reuse.thumbs_dir == media["proxy"].thumbs_dir
    assert Path(reuse.thumbs_dir).is_dir()


def test_proxy_result_from_file_without_sidecars(tmp_path: Path, media) -> None:
    lone = tmp_path / "proxy_copy.mp4"
    shutil.copyfile(media["proxy"].path, lone)
    reuse = proxy_result_from_file(lone, source_size=(1280, 720))
    assert reuse.audio_path is None
    assert reuse.thumbs_dir is None


# ----------------------------------------------------------------------
# FrameReader
# ----------------------------------------------------------------------


def test_frame_reader_yields_all_frames_in_order(media) -> None:
    with FrameReader(media["proxy"].path, prefetch=4) as reader:
        assert reader.expected_frames == 100
        items = list(reader)
    assert [it.index for it in items] == list(range(100))
    assert items[10].t == pytest.approx(10 / 25.0)
    assert items[0].frame.shape == (360, 640, 3)
    assert items[0].frame.dtype == np.uint8


def test_frame_reader_stride_window_and_batches(media) -> None:
    with FrameReader(media["proxy"].path, stride=3, start_frame=5, end_frame=40) as reader:
        assert reader.expected_frames == 12
        batches = list(reader.batches(5))
    indices = [it.index for b in batches for it in b]
    assert indices == list(range(5, 40, 3))
    assert [len(b) for b in batches] == [5, 5, 2]
    assert batches[0][1].t == pytest.approx(8 / 25.0)


def test_frame_reader_seek_far_start_matches_sequential(media) -> None:
    with FrameReader(media["proxy"].path, start_frame=150, end_frame=152) as reader:
        assert list(reader) == []  # beyond the end: nothing, no hang
    with FrameReader(media["proxy"].path, start_frame=90, end_frame=92) as reader:
        items = list(reader)
    assert [it.index for it in items] == [90, 91]
    ref = read_frames_at(media["proxy"].path, [90])[90]
    assert np.array_equal(items[0].frame, ref)


def test_frame_reader_cancel_stops_early(media) -> None:
    cancel = threading.Event()
    seen = []
    with FrameReader(media["proxy"].path, prefetch=2, cancel_event=cancel) as reader:
        for item in reader:
            seen.append(item.index)
            if len(seen) == 7:
                cancel.set()
    assert 7 <= len(seen) <= 10
    assert seen == list(range(len(seen)))


def test_frame_reader_close_without_consuming(media) -> None:
    reader = FrameReader(media["proxy"].path, prefetch=2).start()
    reader.close()  # producer blocked on a full queue must still exit
    assert reader._thread is not None and not reader._thread.is_alive()
