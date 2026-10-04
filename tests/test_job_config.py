"""Typed JobConfig validation, profile resolution and path restrictions."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.schemas import JobConfig, validate_job_config


@pytest.fixture(autouse=True)
def _no_model_family(monkeypatch):
    monkeypatch.delenv("VH_MODEL_FAMILY", raising=False)
    monkeypatch.delenv("VH_MEDIA_ROOTS", raising=False)
    # Production default: API configs cannot choose output_dir.
    monkeypatch.setenv("VH_ALLOW_OUTPUT_DIR_OVERRIDE", "0")


def _match(client: TestClient, tmp_path: Path) -> str:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake-video")
    resp = client.post("/v1/matches", json={"name": "Config Match", "source_video_path": str(source), "metadata": {}})
    assert resp.status_code == 201, resp.text
    return resp.json()["match_id"]


def _create(client: TestClient, match_id: str, config: dict):
    return client.post(f"/v1/matches/{match_id}/jobs", json={"config": config})


def test_profiles_resolve_and_explicit_keys_win() -> None:
    fast = validate_job_config({"profile": "fast"})
    assert (fast["proxy_height"], fast["inference_imgsz"], fast["vid_stride"], fast["yolo_model"]) == (720, 960, 2, "yolov8n.pt")
    balanced = validate_job_config({})
    assert balanced["profile"] == "balanced" and balanced["yolo_model"] == "yolov8s.pt"
    quality = validate_job_config({"profile": "QUALITY", "output_height": 1080, "camera_mode": "follow_ball"})
    assert quality["profile"] == "quality"
    assert quality["output_height"] == 1080 and quality["profile_overrides"] == ["output_height"]
    assert quality["yolo_model"] == "yolov8m.pt" and quality["ball_tiles"] is True
    assert quality["camera_mode"] == "follow_ball"


def test_legacy_keys_pass_through_and_none_means_unset() -> None:
    cfg = validate_job_config({"model_version": "event-v1", "focus_event_types": ["goal"], "output_dir": None,
                               "notify_email": "coach@example.com", "inference_imgsz": None})
    assert cfg["model_version"] == "event-v1"
    assert cfg["notify_email"] == "coach@example.com"
    assert "output_dir" not in cfg
    assert cfg["inference_imgsz"] == 1280


@pytest.mark.parametrize(
    "bad",
    [
        {"camera_mode": "drone"},
        {"profile": "turbo"},
        {"camera_style": "fisheye"},
        {"select_player": True},
        {"team_left_color": "red"},
        {"trim_start": "10:00", "trim_end": "05:00"},
        {"trim_start": "abc"},
        {"batch_size": "lots"},
        {"vid_stride": 0},
        {"zoom_factor": 0.5},
        {"pitch_corners": [[0, 0], [1, 0], [1, 1]]},
        {"reel_preset": "2min"},
        {"device": "tpu"},
    ],
)
def test_invalid_configs_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        validate_job_config(bad)


def test_trim_accepts_clock_strings() -> None:
    cfg = JobConfig.model_validate({"trim_start": "00:01:30", "trim_end": "1:00:00"})
    assert cfg.trim_start == "00:01:30"


def test_create_job_rejects_invalid_config_with_400(client: TestClient, tmp_path: Path) -> None:
    match_id = _match(client, tmp_path)
    for bad in ({"camera_mode": "drone"}, {"select_player": True}, {"profile": "turbo"}):
        resp = _create(client, match_id, bad)
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"]["message"].startswith("Invalid job config")


def test_create_job_stores_resolved_config(client: TestClient, tmp_path: Path) -> None:
    match_id = _match(client, tmp_path)
    resp = _create(client, match_id, {"profile": "fast", "camera_mode": "follow_ball", "debug_video": True})
    assert resp.status_code == 201, resp.text
    config = resp.json()["config"]
    assert config["proxy_height"] == 720 and config["yolo_model"] == "yolov8n.pt"
    assert config["debug_video"] is True and config["profile_overrides"] == ["debug_video"]
    assert "output_dir" not in config


def test_output_dir_is_rejected_unless_test_override(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    match_id = _match(client, tmp_path)
    for value in (str(tmp_path / "out"), "/etc/vh_out", "./outputs/job_other"):
        resp = _create(client, match_id, {"output_dir": value})
        assert resp.status_code == 400, resp.text
        assert "output_dir" in resp.json()["error"]["message"]
    with pytest.raises(ValueError):
        validate_job_config({"output_dir": str(tmp_path / "out")})
    # null means unset and stays accepted (the Studio UI sends output_dir: null).
    assert "output_dir" not in validate_job_config({"output_dir": None})

    monkeypatch.setenv("VH_ALLOW_OUTPUT_DIR_OVERRIDE", "1")
    ok = _create(client, match_id, {"output_dir": str(tmp_path / "out")})
    assert ok.status_code == 201, ok.text
    assert ok.json()["config"]["output_dir"] == str((tmp_path / "out").resolve())


def test_media_roots_restrict_paths(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    from backend.config import settings

    allowed = tmp_path / "media"
    allowed.mkdir()
    (allowed / "game.mp4").write_bytes(b"x")
    match_id = _match(client, tmp_path)
    # Without VH_MEDIA_ROOTS any path is accepted except the output root.
    monkeypatch.setattr(settings, "output_root", str(tmp_path / "outputs"))
    (tmp_path / "outputs" / "job_x").mkdir(parents=True)
    (tmp_path / "outputs" / "job_x" / "full_follow_ball_zoom.mp4").write_bytes(b"x")
    in_outputs = _create(client, match_id, {"video_path": str(tmp_path / "outputs" / "job_x" / "full_follow_ball_zoom.mp4")})
    assert in_outputs.status_code == 400, in_outputs.text
    assert _create(client, match_id, {"video_path": str(tmp_path / "elsewhere.mp4")}).status_code == 201

    monkeypatch.setenv("VH_MEDIA_ROOTS", str(allowed))
    bad_video = _create(client, match_id, {"video_path": str(tmp_path / "elsewhere.mp4")})
    assert bad_video.status_code == 400
    ok = _create(client, match_id, {"video_path": str(allowed / "game.mp4")})
    assert ok.status_code == 201, ok.text
    assert ok.json()["config"]["video_path"] == str((allowed / "game.mp4").resolve())
    # Even with the test override, output_dir must stay inside the allowed roots.
    monkeypatch.setenv("VH_ALLOW_OUTPUT_DIR_OVERRIDE", "1")
    with pytest.raises(ValueError):
        validate_job_config({"output_dir": str(tmp_path.parent / "nope")})


def test_rerun_validates_and_applies_new_profile(client: TestClient, tmp_path: Path) -> None:
    match_id = _match(client, tmp_path)
    parent = _create(client, match_id, {"profile": "fast", "inference_imgsz": 1536})
    assert parent.status_code == 201
    parent_id = parent.json()["job_id"]

    bad = client.post(f"/v1/jobs/{parent_id}/rerun", json={"config_overrides": {"camera_mode": "nope"}})
    assert bad.status_code == 400

    rerun = client.post(f"/v1/jobs/{parent_id}/rerun", json={"config_overrides": {"profile": "quality"}})
    assert rerun.status_code == 201, rerun.text
    config = rerun.json()["config"]
    assert config["profile"] == "quality"
    assert config["proxy_height"] == 1080 and config["yolo_model"] == "yolov8m.pt"  # profile-derived keys follow
    assert config["inference_imgsz"] == 1536  # the parent's explicit choice survives


def test_track_this_player_rerun_never_writes_into_source_run(client: TestClient, tmp_path: Path) -> None:
    match_id = _match(client, tmp_path)
    parent = _create(client, match_id, {"profile": "fast"})
    parent_id = parent.json()["job_id"]
    rerun = client.post(f"/v1/jobs/{parent_id}/rerun", json={"config_overrides": {
        "camera_mode": "follow_player", "focus_track_id": 7, "reuse_tracking_from_job": parent_id,
        "render_full_follow_cam": True,
    }})
    assert rerun.status_code == 201, rerun.text
    config = rerun.json()["config"]
    assert config["focus_track_id"] == 7 and config["reuse_tracking_from_job"] == parent_id
    assert "output_dir" not in config

    missing = client.post(f"/v1/jobs/{parent_id}/rerun", json={"config_overrides": {"reuse_tracking_from_job": "job_nope"}})
    assert missing.status_code == 400
    for bad in ("./other", "../job_x", "/abs/run", "job.x", "a" * 65):
        resp = client.post(f"/v1/jobs/{parent_id}/rerun", json={"config_overrides": {"reuse_tracking_from_job": bad}})
        assert resp.status_code == 400, (bad, resp.text)
        assert _create(client, match_id, {"reuse_tracking_from_job": bad}).status_code == 400, bad
    assert _create(client, match_id, {"reuse_tracking_from_job": "job_nope"}).status_code == 400
    ok = _create(client, match_id, {"reuse_tracking_from_job": parent_id, "camera_mode": "follow_player"})
    assert ok.status_code == 201, ok.text

    # Tracks are per source video: another match's job cannot be reused.
    other_match = _match(client, tmp_path)
    assert _create(client, other_match, {"reuse_tracking_from_job": parent_id}).status_code == 400


def test_event_types_include_cards() -> None:
    from typing import get_args

    from backend.schemas import EventType

    assert {"yellow_card", "red_card"} <= set(get_args(EventType))


def test_no_yolo26_defaults_in_pipeline_entry_points() -> None:
    import inspect

    import VideoHighlights

    root = Path(__file__).resolve().parents[1]
    for rel in ("VideoHighlights.py", "backend/services/job_runner.py", "VideoHighlightsGUI.py"):
        assert "yolo26s.pt" not in (root / rel).read_text(encoding="utf-8"), rel
    assert inspect.signature(VideoHighlights.process_video_highlights).parameters["yolo_model"].default is None
