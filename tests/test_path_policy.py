"""Path policy, reuse-tracking resolution and job-runner hardening.

* ``output_dir`` from the API is ignored at run time / rejected at create:
  runs always go to ``<output_root>/<job_id>``.
* ``reuse_tracking_from_job`` must be a job id of the same tenant whose run
  folder (inside the output root) holds tracks; no bare-path fallback.
* ``VH_MEDIA_ROOTS`` (plus the upload storage root, never the output root)
  applies to job video paths and to match source paths.
* Failed jobs carry the engine's reason; heartbeat logs are rate limited;
  ``cancel_requested`` jobs cannot be deleted; malformed ROI / goal boxes /
  model paths are rejected by the API.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from backend import database
from backend.config import settings
from backend.models import JobLogEntry, Match, ProcessingJob, Tenant


@pytest.fixture(autouse=True)
def _policy_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VH_ALLOW_OUTPUT_DIR_OVERRIDE", "0")
    monkeypatch.delenv("VH_MEDIA_ROOTS", raising=False)
    monkeypatch.delenv("VH_MODEL_FAMILY", raising=False)
    monkeypatch.setattr(settings, "output_root", str(tmp_path / "outputs"))
    monkeypatch.setattr(settings, "local_storage_root", str(tmp_path / "storage"))


def _install_fake_pipeline(monkeypatch, behaviour: Optional[Callable[..., bool]] = None,
                           captured: Optional[Dict[str, object]] = None) -> None:
    fake = types.ModuleType("VideoHighlights")

    def parse_time(value: str) -> float:
        return float(value)

    def process_video_highlights(**kwargs) -> bool:
        if captured is not None:
            captured.update(kwargs)
        out = Path(str(kwargs["output_dir"]))
        out.mkdir(parents=True, exist_ok=True)
        if behaviour is not None:
            return behaviour(**kwargs)
        (out / "analysis_bookmarks.json").write_text(json.dumps({"bookmarks": []}), encoding="utf-8")
        return True

    fake.parse_time = parse_time
    fake.process_video_highlights = process_video_highlights
    monkeypatch.setitem(sys.modules, "VideoHighlights", fake)


def _source(tmp_path: Path, name: str = "source.mp4") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-video")
    return path


def _match(client: TestClient, source: Path | str) -> str:
    resp = client.post("/v1/matches", json={"name": "Policy", "source_video_path": str(source), "metadata": {}})
    assert resp.status_code == 201, resp.text
    return resp.json()["match_id"]


def _job(client: TestClient, match_id: str, config: Optional[dict] = None) -> str:
    resp = client.post(f"/v1/matches/{match_id}/jobs", json={"config": config or {}})
    assert resp.status_code == 201, resp.text
    return resp.json()["job_id"]


def _run_once(client: TestClient, job_id: str) -> dict:
    resp = client.post("/v1/jobs/worker/run-once")
    assert resp.status_code == 200 and resp.json()["job_id"] == job_id, resp.text
    return client.get(f"/v1/jobs/{job_id}").json()


def _db_update(model, row_id: str, **fields) -> None:
    with Session(database.engine) as session:
        row = session.get(model, row_id)
        for key, value in fields.items():
            setattr(row, key, value)
        session.add(row)
        session.commit()


def _write_tracks(run_dir: Path, trim_offset: float = 0.0, duration: float = 5.0) -> None:
    import numpy as np

    from backend.services.tracking_types import BallDetections, TrackingResult

    empty = np.zeros(0, dtype=np.float32)
    tracking = TrackingResult(fps=25.0, frame_size=(640, 360), duration_s=duration, players={},
                              ball=BallDetections(empty, empty, empty, empty, empty, empty),
                              trim_offset_s=trim_offset)
    tracking.save(str(run_dir))


# ---------------------------------------------------------------------------
# output_dir
# ---------------------------------------------------------------------------


def test_stored_output_dir_is_ignored_by_the_runner(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    captured: Dict[str, object] = {}
    _install_fake_pipeline(monkeypatch, captured=captured)
    match_id = _match(client, _source(tmp_path))
    job_id = _job(client, match_id)
    evil = tmp_path / "elsewhere"
    # A legacy row (or a direct DB write) carrying an output_dir.
    _db_update(ProcessingJob, job_id, config_json={"output_dir": str(evil), "profile": "fast"})
    payload = _run_once(client, job_id)
    assert payload["status"] == "completed", payload.get("error_message")
    assert Path(str(captured["output_dir"])) == tmp_path / "outputs" / job_id
    assert Path(payload["result"]["output_dir"]) == (tmp_path / "outputs" / job_id).resolve()
    assert not evil.exists()


# ---------------------------------------------------------------------------
# reuse_tracking_from_job
# ---------------------------------------------------------------------------


def test_reuse_resolves_source_run_inside_output_root(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    captured: Dict[str, object] = {}
    _install_fake_pipeline(monkeypatch, captured=captured)
    match_id = _match(client, _source(tmp_path))
    parent = _job(client, match_id)
    _write_tracks(tmp_path / "outputs" / parent)
    child = _job(client, match_id, {"reuse_tracking_from_job": parent, "camera_mode": "follow_player"})
    payload = _run_once(client, parent)  # parent first (queue order)
    assert payload["status"] == "completed"
    payload = _run_once(client, child)
    assert payload["status"] == "completed", payload.get("error_message")
    assert captured["reuse_tracking_from"] == str((tmp_path / "outputs" / parent).resolve())
    assert Path(str(captured["output_dir"])) == tmp_path / "outputs" / child


def test_reuse_never_falls_back_to_a_bare_path(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    _install_fake_pipeline(monkeypatch)
    match_id = _match(client, _source(tmp_path))
    job_id = _job(client, match_id)
    # Tracks exist under <output_root>/<id> but there is NO job row for it.
    _write_tracks(tmp_path / "outputs" / "job_ghost")
    _db_update(ProcessingJob, job_id, config_json={"reuse_tracking_from_job": "job_ghost"})
    payload = _run_once(client, job_id)
    assert payload["status"] == "failed"
    assert "reuse_tracking_from_job not found: job_ghost" in payload["error_message"]


def test_reuse_rejects_path_like_ids_and_out_of_root_dirs(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    _install_fake_pipeline(monkeypatch)
    match_id = _match(client, _source(tmp_path))
    parent = _job(client, match_id)
    outside = tmp_path / "outside_run"
    _write_tracks(outside)
    # The source's recorded output_dir points outside the output root.
    _db_update(ProcessingJob, parent, status="completed", result_json={"output_dir": str(outside)})
    child = _job(client, match_id, {"reuse_tracking_from_job": parent})
    payload = _run_once(client, child)
    assert payload["status"] == "failed"
    assert "no reusable tracks" in payload["error_message"]

    for bad in ("../outside_run", str(outside), "./job_x"):
        job_id = _job(client, match_id)
        _db_update(ProcessingJob, job_id, config_json={"reuse_tracking_from_job": bad})
        payload = _run_once(client, job_id)
        assert payload["status"] == "failed", bad
        assert "must be a job id" in payload["error_message"], payload["error_message"]


def test_reuse_from_another_tenant_is_rejected(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    _install_fake_pipeline(monkeypatch)
    match_id = _match(client, _source(tmp_path))
    with Session(database.engine) as session:
        other = Tenant(slug="other-club", name="Other club")
        session.add(other)
        session.commit()
        other_match = Match(tenant_id=other.id, name="Theirs", source_video_path=str(_source(tmp_path, "b.mp4")))
        session.add(other_match)
        session.commit()
        foreign = ProcessingJob(tenant_id=other.id, match_id=other_match.id, status="completed")
        session.add(foreign)
        session.commit()
        foreign_id = foreign.id
    _write_tracks(tmp_path / "outputs" / foreign_id)

    resp = client.post(f"/v1/matches/{match_id}/jobs", json={"config": {"reuse_tracking_from_job": foreign_id}})
    assert resp.status_code == 400
    parent = _job(client, match_id)
    resp = client.post(f"/v1/jobs/{parent}/rerun", json={"config_overrides": {"reuse_tracking_from_job": foreign_id}})
    assert resp.status_code == 400
    # Even if such a config reached the queue, the runner refuses it.
    _db_update(ProcessingJob, parent, config_json={"reuse_tracking_from_job": foreign_id})
    payload = _run_once(client, parent)
    assert payload["status"] == "failed"
    assert f"reuse_tracking_from_job not found: {foreign_id}" in payload["error_message"]


# ---------------------------------------------------------------------------
# Media roots: job video paths and match sources
# ---------------------------------------------------------------------------


def test_runner_rejects_sources_outside_media_roots_and_in_output_root(client: TestClient, tmp_path: Path,
                                                                       monkeypatch) -> None:
    _install_fake_pipeline(monkeypatch)
    match_id = _match(client, _source(tmp_path))
    run_dir = tmp_path / "outputs" / "job_someone_else"
    run_dir.mkdir(parents=True)
    (run_dir / "full_follow_ball_zoom.mp4").write_bytes(b"movie")
    job_id = _job(client, match_id)
    _db_update(Match, match_id, source_video_path=str(run_dir / "full_follow_ball_zoom.mp4"))
    payload = _run_once(client, job_id)
    assert payload["status"] == "failed"
    assert "output root" in payload["error_message"]

    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setenv("VH_MEDIA_ROOTS", str(media))
    _db_update(Match, match_id, source_video_path=str(_source(tmp_path, "outside.mp4")))
    job_id = _job(client, match_id)
    payload = _run_once(client, job_id)
    assert payload["status"] == "failed"
    assert "VH_MEDIA_ROOTS" in payload["error_message"]

    _db_update(Match, match_id, source_video_path=str(_source(media, "inside.mp4")))
    job_id = _job(client, match_id)
    assert _run_once(client, job_id)["status"] == "completed"


def test_match_source_paths_follow_media_roots(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "media"
    media.mkdir()
    inside = _source(media, "game.mp4")
    outside = _source(tmp_path, "other.mp4")
    (tmp_path / "outputs" / "job_a").mkdir(parents=True)
    in_outputs = tmp_path / "outputs" / "job_a" / "proxy_720p.mp4"
    in_outputs.write_bytes(b"proxy")

    # No VH_MEDIA_ROOTS: permissive, except the output root.
    assert client.post("/v1/matches", json={"source_video_path": str(outside)}).status_code == 201
    assert client.post("/v1/matches", json={"source_video_path": str(in_outputs)}).status_code == 400

    monkeypatch.setenv("VH_MEDIA_ROOTS", str(media))
    created = client.post("/v1/matches", json={"source_video_path": str(outside)})
    assert created.status_code == 400, created.text
    link = client.post("/v1/matches", json={"source_video_path": "https://www.youtube.com/watch?v=abc"})
    assert link.status_code == 201, link.text
    match_id = _match(client, inside)
    stored = _source(tmp_path / "storage", "upload.mp4")
    assert client.patch(f"/v1/matches/{match_id}", json={"source_video_path": str(stored)}).status_code == 200

    patched = client.patch(f"/v1/matches/{match_id}", json={"source_video_path": str(outside)})
    assert patched.status_code == 400
    sneaky = client.patch(f"/v1/matches/{match_id}", json={"metadata": {"assets": [{"asset_id": "a", "path": str(outside)}]}})
    assert sneaky.status_code == 400
    registered = client.post(f"/v1/matches/{match_id}/assets/register-local", json={"path": str(outside)})
    assert registered.status_code == 400
    inspected = client.post("/v1/matches/assets/inspect-local", json={"path": str(outside)}).json()
    assert inspected["ok"] is False and inspected["code"] == "outside_media_roots"
    ok = client.post(f"/v1/matches/{match_id}/assets/register-local", json={"path": str(inside)})
    assert ok.status_code == 201, ok.text
    assert client.get(f"/v1/matches/{match_id}").json()["source_video_path"] == str(inside)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("roi", [
    {"x1_norm": 0.6, "y1_norm": 0.2, "x2_norm": 0.4, "y2_norm": 0.5},
    {"x1_norm": 0.1, "y1_norm": 0.2, "x2_norm": 1.4, "y2_norm": 0.5},
    {"x": 10, "y": 10, "w": 0, "h": 20},
    {"x": "left", "y": 10, "w": 5, "h": 20},
    {"x1": 1, "y1": 2},
    {"x1_norm": 0.1, "y1_norm": 0.2, "x2_norm": 0.4, "y2_norm": 0.5, "t": -3},
])
def test_malformed_player_roi_is_rejected_at_the_api(client: TestClient, tmp_path: Path, roi: dict) -> None:
    match_id = _match(client, _source(tmp_path))
    resp = client.post(f"/v1/matches/{match_id}/jobs", json={"config": {"player_roi": roi}})
    assert resp.status_code == 400, resp.text
    assert "player_roi" in resp.json()["error"]["message"]


def test_valid_roi_goal_boxes_and_models(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    from backend.schemas import validate_job_config

    cfg = validate_job_config({
        "player_roi": {"x1_norm": 0.1, "y1_norm": 0.2, "x2_norm": 0.4, "y2_norm": 0.5, "t": 12.5},
        "goal_box_left": {"x1": 0, "y1": 100, "x2": 60, "y2": 200},
        "yolo_model": "yolov8n.pt",
    })
    assert cfg["player_roi"]["t"] == 12.5 and cfg["goal_box_left"]["x2"] == 60.0
    assert validate_job_config({"player_roi": {"x": 5, "y": 6, "w": 40, "h": 80}})["player_roi"]["w"] == 40.0
    assert "player_roi" not in validate_job_config({"player_roi": {}})
    for bad in ({"goal_box_right": {"x1": 50, "y1": 0, "x2": 10, "y2": 30}},
                {"goal_box_left": {"x1": 0, "y1": 0}},
                {"pitch_corners": [[0, 0], [0, 0], [1, 1], [0, 1]]},
                {"pitch_corners": [[0, 0], [1, 0], [1, 1], [-5, 1]]}):
        with pytest.raises(ValueError):
            validate_job_config(bad)

    models = tmp_path / "models"
    models.mkdir()
    (models / "custom.pt").write_bytes(b"w")
    monkeypatch.setenv("VH_MODEL_DIR", str(models))
    assert validate_job_config({"yolo_model": str(models / "custom.pt")})["yolo_model"] == str((models / "custom.pt").resolve())
    assert validate_job_config({"yolo_model": "sub/custom.pt"})["yolo_model"] == str((models / "sub" / "custom.pt").resolve())
    for bad in ("/etc/passwd", "../secrets.pt", str(tmp_path / "custom.pt"), "a/../../x.pt"):
        with pytest.raises(ValueError):
            validate_job_config({"yolo_model": bad})
    with pytest.raises(ValueError):
        validate_job_config({"tracker_config": "/etc/botsort.yaml"})
    match_id = _match(client, _source(tmp_path))
    assert client.post(f"/v1/matches/{match_id}/jobs", json={"config": {"yolo_model": "/etc/passwd"}}).status_code == 400


# ---------------------------------------------------------------------------
# Failure reasons, heartbeat logs, delete
# ---------------------------------------------------------------------------


def test_failed_job_keeps_the_engine_reason(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    def fail(**kwargs) -> bool:
        cb = kwargs["progress_callback"]
        cb("tracking", 0.3, "Detecting", {})
        cb("failed", 1.0, "Processing failed: CUDA out of memory", {})
        cb("failed", 1.0, "Processing failed", {"error": "CUDA out of memory while batching 16 frames"})
        return False

    _install_fake_pipeline(monkeypatch, behaviour=fail)
    match_id = _match(client, _source(tmp_path))
    job_id = _job(client, match_id)
    payload = _run_once(client, job_id)
    assert payload["status"] == "failed"
    assert "CUDA out of memory while batching 16 frames" in payload["error_message"]
    diag = client.get(f"/v1/jobs/{job_id}/diagnostics").json()
    assert "CUDA out of memory" in diag["summary"]

    # No callback reason: progress.json's failure message is used.
    def fail_quietly(**kwargs) -> bool:
        out = Path(str(kwargs["output_dir"]))
        (out / "progress.json").write_text(json.dumps({"stage": "failed", "status": "failed",
                                                       "message": "Processing failed: disk full"}))
        return False

    _install_fake_pipeline(monkeypatch, behaviour=fail_quietly)
    job_id = _job(client, match_id)
    assert "disk full" in _run_once(client, job_id)["error_message"]


def test_unchanged_heartbeats_are_rate_limited(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    from backend.services import job_runner

    clock = {"now": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    monkeypatch.setattr(job_runner, "_utcnow", lambda: clock["now"])
    progress_seen: List[float] = []

    def chatty(**kwargs) -> bool:
        cb = kwargs["progress_callback"]
        for _ in range(100):  # 100 identical heartbeats, 2.1 s apart = 210 s
            clock["now"] += timedelta(seconds=2.1)
            cb("tracking", 0.30, "Detecting and tracking players and ball", {})
        for i in range(20):  # repeated >= 0.98 callbacks with the same message
            clock["now"] += timedelta(seconds=0.5)
            cb("clips_reel", 0.985, "Cutting highlight clips", {})
        with Session(database.engine) as session:
            progress_seen.append(float(session.get(ProcessingJob, kwargs["output_dir"].split("/")[-1]).progress))
        (Path(str(kwargs["output_dir"])) / "analysis_bookmarks.json").write_text("{}")
        return True

    _install_fake_pipeline(monkeypatch, behaviour=chatty)
    match_id = _match(client, _source(tmp_path))
    job_id = _job(client, match_id)
    assert _run_once(client, job_id)["status"] == "completed"
    with Session(database.engine) as session:
        rows = list(session.exec(select(JobLogEntry).where(JobLogEntry.job_id == job_id)
                                 .where(JobLogEntry.stage == "processing_video")))
    heartbeat = [r for r in rows if r.message == "Detecting and tracking players and ball"]
    final = [r for r in rows if r.message == "Cutting highlight clips"]
    assert 6 <= len(heartbeat) <= 9, len(heartbeat)  # first + one per 30 s of 210 s
    assert len(final) == 1
    assert progress_seen and progress_seen[0] >= 0.98  # job.progress still advanced


def test_cancel_requested_job_cannot_be_deleted(client: TestClient, tmp_path: Path) -> None:
    match_id = _match(client, _source(tmp_path))
    job_id = _job(client, match_id)
    _db_update(ProcessingJob, job_id, status="cancel_requested", cancel_requested=True)
    resp = client.delete(f"/v1/jobs/{job_id}")
    assert resp.status_code == 409, resp.text
    _db_update(ProcessingJob, job_id, status="canceled")
    assert client.delete(f"/v1/jobs/{job_id}").status_code == 200
