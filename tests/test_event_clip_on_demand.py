from __future__ import annotations

from pathlib import Path
from uuid import uuid4


def _create_match(client, source_video_path: str) -> str:
    response = client.post(
        "/v1/matches",
        json={
            "name": "Clip Match",
            "source_video_path": source_video_path,
            "metadata": {},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def _create_event(client, match_id: str) -> str:
    event_id = f"evt_clip_{uuid4().hex[:10]}"
    response = client.put(
        f"/v1/matches/{match_id}/events/{event_id}",
        json={
            "event_type": "goal",
            "status": "auto_detected",
            "confidence": 0.91,
            "period": "1H",
            "occurred_at_ms": 20000,
            "start_ms": 19000,
            "end_ms": 22500,
            "frame_index": 100,
            "source": {"detector": "test"},
            "location": {},
            "participants": [],
            "evidence": {},
            "explanations": [],
        },
    )
    assert response.status_code == 200, response.text
    return event_id


def test_event_clip_on_demand_create_and_cache(client, tmp_path: Path, monkeypatch) -> None:
    source_video = tmp_path / "source_video.mp4"
    source_video.write_bytes(b"fake-video")
    match_id = _create_match(client, str(source_video))
    event_id = _create_event(client, match_id)

    render_calls = []

    def _fake_render(video_path, output_path, start_seconds, end_seconds, include_audio, prefer_gpu):
        render_calls.append(
            {
                "video_path": video_path,
                "output_path": output_path,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "include_audio": include_audio,
                "prefer_gpu": prefer_gpu,
            }
        )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-clip-binary")
        return str(out)

    monkeypatch.setattr("backend.routers.events.render_clip_ffmpeg", _fake_render)

    payload = {
        "pre_seconds": 1.5,
        "post_seconds": 5.0,
        "anchor": "event_window",
        "include_audio": True,
        "prefer_gpu": False,
        "force_rebuild": False,
    }
    first = client.post(f"/v1/matches/{match_id}/events/{event_id}/clip-on-demand", json=payload)
    assert first.status_code == 200, first.text
    first_payload = first.json()
    assert first_payload["reused_existing"] is False
    assert first_payload["event_id"] == event_id
    assert first_payload["asset_id"]
    assert first_payload["path"]
    assert first_payload["download_url"]

    second = client.post(f"/v1/matches/{match_id}/events/{event_id}/clip-on-demand", json=payload)
    assert second.status_code == 200, second.text
    second_payload = second.json()
    assert second_payload["reused_existing"] is True
    assert second_payload["asset_id"] == first_payload["asset_id"]
    assert len(render_calls) == 1

    match = client.get(f"/v1/matches/{match_id}")
    assert match.status_code == 200, match.text
    metadata = match.json().get("metadata") or {}
    generated = list(metadata.get("generated_clips", []))
    assert generated
    assert any(item.get("event_id") == event_id for item in generated)



# ---------------------------------------------------------------------------
# ffmpeg cutting: input seeking (-ss before -i) and real cuts
# ---------------------------------------------------------------------------

import shutil
import subprocess

import pytest

ffmpeg_required = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                     reason="ffmpeg/ffprobe not installed")


def test_build_command_seeks_on_input() -> None:
    from backend.services.event_clip_renderer import build_command

    cmd = build_command("in.mp4", "out.mp4", 12.5, 20.0)
    assert cmd.index("-ss") < cmd.index("-i"), "seek must precede -i (input seeking)"
    assert cmd[cmd.index("-ss") + 1] == "12.500"
    assert "-to" not in cmd
    assert cmd[cmd.index("-t") + 1] == "7.500"
    assert cmd.index("-t") > cmd.index("-i")
    assert "0:a:0?" in cmd  # sources without audio still work
    copy = build_command("in.mp4", "out.mp4", 1.0, 2.0, copy=True, include_audio=False)
    assert copy.index("-ss") < copy.index("-i")
    assert "copy" in copy and "make_zero" in copy and "-an" in copy
    with pytest.raises(ValueError):
        build_command("in.mp4", "out.mp4", 5.0, 5.0)


def _ramp_video(path, seconds=6, fps=10):
    """Luma = 40 * t: the first frame of a cut tells where decoding started."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         f"nullsrc=size=64x64:rate={fps}:duration={seconds},format=gray,geq=lum='clip(40*T,0,255)'",
         "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-g", "100", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )
    return str(path)


def _first_frame_luma(path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    assert ok
    return float(frame.mean())


@ffmpeg_required
def test_render_clip_cuts_exact_window(tmp_path) -> None:
    from backend.services.event_clip_renderer import probe_media, render_clip_ffmpeg

    src = _ramp_video(tmp_path / "src.mp4")
    out = render_clip_ffmpeg(src, str(tmp_path / "clip.mp4"), 2.0, 4.0, include_audio=True, prefer_gpu=False)
    info = probe_media(out)
    assert info["duration"] == pytest.approx(2.0, abs=0.15)
    assert info["has_audio"]
    # Frame-accurate start even though the only keyframe is at 0 s.
    assert _first_frame_luma(out) == pytest.approx(80.0, abs=8.0)


@ffmpeg_required
def test_cut_clip_from_rendered_reencode_and_copy(tmp_path) -> None:
    from backend.services.event_clip_renderer import concat_clips_ffmpeg, cut_clip_from_rendered, probe_media

    src = _ramp_video(tmp_path / "movie.mp4")
    exact = cut_clip_from_rendered(src, 3.0, 4.5, str(tmp_path / "exact.mp4"))
    assert probe_media(exact)["duration"] == pytest.approx(1.5, abs=0.15)
    assert _first_frame_luma(exact) == pytest.approx(120.0, abs=8.0)
    fast = cut_clip_from_rendered(src, 1.0, 2.0, str(tmp_path / "copy.mp4"), reencode=False)
    assert probe_media(fast)["duration"] > 0.5
    joined = concat_clips_ffmpeg([exact, exact], str(tmp_path / "joined.mp4"), copy=True)
    assert probe_media(joined)["duration"] == pytest.approx(3.0, abs=0.2)


def test_encoder_selection_is_cached_and_falls_back() -> None:
    from backend.services import event_clip_renderer as ecr

    first = ecr.select_h264_encoder(prefer_gpu=True)
    assert first[0] in ("h264_nvenc", "h264_videotoolbox", "libx264", "mpeg4")
    hits = ecr._encoder_works.cache_info().hits
    assert ecr.select_h264_encoder(prefer_gpu=True) == first
    assert ecr._encoder_works.cache_info().hits > hits
    assert ecr.select_h264_encoder(prefer_gpu=False, requested="no_such_encoder")[0] in ("libx264", "mpeg4")
