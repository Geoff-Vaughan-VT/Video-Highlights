"""End-to-end v2 pipeline on a synthetic match (CPU, ground-truth detector).

Proves the proxy-first architecture: every v2 artifact is produced, both
scripted goals are found, and the SOURCE video is decoded exactly twice
(proxy pass + final render), once in analysis-only mode and once (render
only) when tracking is reused. Also drives a job through the API worker to
prove the DB event sync and the cancel plumbing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List

import pytest

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")

DURATION_S = 14.0
GOALS = [(4.0, "right"), (9.5, "left")]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    from backend.services.synthetic_match import SyntheticMatchSpec, generate_synthetic_match

    work = tmp_path_factory.mktemp("e2e")
    source = work / "source.mp4"
    gt = generate_synthetic_match(source, SyntheticMatchSpec(duration_s=DURATION_S, goals=GOALS), prefer_ffmpeg=True)
    x0, y0, x1, y1 = gt.pitch_bounds_px
    return {
        "work": work,
        "source": str(source),
        "gt": gt,
        # The user-supplied pitch corners (TL, TR, BR, BL) the UI collects.
        "pitch_corners": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    }


def _detector_factory(gt, delay_s: float = 0.0):
    from backend.services.detectors import GroundTruthDetector

    def factory(_config: Dict[str, object]):
        det = GroundTruthDetector(gt.tracking, seed=1)
        if delay_s <= 0:
            return det
        inner = det.detect_batch

        def slow(frames, *, frame_indices):
            time.sleep(delay_s)
            return inner(frames, frame_indices=frame_indices)

        det.detect_batch = slow  # type: ignore[method-assign]
        return det

    return factory


class DecodeRecorder:
    """Counts decodes of the source: ffmpeg runs with ``-i <source>`` and
    ``cv2.VideoCapture(<source>)`` (forbidden), labelled by the active stage."""

    def __init__(self, source: str) -> None:
        self.source = os.path.realpath(source)
        self.stage = "other"
        self.decodes: List[str] = []
        self.cv2_opens: List[str] = []
        self.lock = threading.Lock()

    def is_source(self, value: object) -> bool:
        try:
            return os.path.realpath(str(value)) == self.source
        except Exception:
            return False

    def record_argv(self, argv) -> None:
        if not isinstance(argv, (list, tuple)) or not argv:
            return
        exe = os.path.basename(str(argv[0])).lower()
        if not exe.startswith("ffmpeg"):
            return
        args = [str(a) for a in argv]
        for i, token in enumerate(args[:-1]):
            if token == "-i" and self.is_source(args[i + 1]):
                with self.lock:
                    self.decodes.append(self.stage)


@pytest.fixture
def decode_recorder(monkeypatch, synthetic):
    import cv2

    from backend.services import camera_render, frame_source

    rec = DecodeRecorder(synthetic["source"])
    original_popen = subprocess.Popen

    class RecordingPopen(original_popen):  # type: ignore[misc, valid-type]
        def __init__(self, args, *a, **kw):
            rec.record_argv(args)
            super().__init__(args, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", RecordingPopen)

    original_capture = cv2.VideoCapture

    def guarded_capture(*args, **kwargs):
        if args and rec.is_source(args[0]):
            rec.cv2_opens.append(rec.stage)
            raise AssertionError(f"cv2.VideoCapture opened the SOURCE during stage {rec.stage}")
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(cv2, "VideoCapture", guarded_capture)

    def staged(name, fn):
        def wrapper(*args, **kwargs):
            previous, rec.stage = rec.stage, name
            try:
                return fn(*args, **kwargs)
            finally:
                rec.stage = previous

        return wrapper

    monkeypatch.setattr(frame_source, "build_proxy", staged("build_proxy", frame_source.build_proxy))
    monkeypatch.setattr(camera_render, "render_camera_plan_video",
                        staged("render", camera_render.render_camera_plan_video))
    return rec


def _run(synthetic, out_dir: Path, **overrides) -> bool:
    import VideoHighlights

    kwargs = dict(
        profile="fast",
        camera_mode="follow_ball",
        render_full_follow_cam=True,
        broadcast_reel=True,
        detect_cards=False,
        llm_report=False,
        pitch_corners=synthetic["pitch_corners"],
        detector_factory=_detector_factory(synthetic["gt"]),
    )
    kwargs.update(overrides)
    return VideoHighlights.process_video_highlights(synthetic["source"], str(out_dir), **kwargs)


def _probe(path: Path):
    from backend.services.frame_source import probe_video

    return probe_video(path)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_run_dir(synthetic):
    return synthetic["work"] / "run_full"


def test_full_run_produces_every_artifact_with_two_source_decodes(synthetic, decode_recorder, full_run_dir):
    started = time.monotonic()
    assert _run(synthetic, full_run_dir) is True
    elapsed = time.monotonic() - started

    run = full_run_dir
    for name in (
        "proxy_720p.mp4", "audio_analysis.wav", "tracks.npz", "tracks_meta.json", "analysis_events.json",
        "analysis_player_stats.json", "analysis_team_stats.json", "analysis_bookmarks.json",
        "analysis_bookmarks.csv", "analysis_game_states.json", "analysis_tracking.json",
        "camera_decisions.jsonl", "camera_crops.txt", "camera_quality.json", "progress.json",
        "full_follow_ball_zoom.mp4", "highlights_reel.mp4",
    ):
        assert (run / name).exists(), f"missing artifact {name}"
    assert any((run / "thumbs").glob("*.jpg"))
    assert sorted(run.glob("highlight_*.mp4")), "no highlight clips"

    # Exactly two decodes of the source: the proxy pass and the final render.
    assert decode_recorder.decodes == ["build_proxy", "render"], decode_recorder.decodes
    assert decode_recorder.cv2_opens == []

    # Final movie: requested output height, duration of the processing window.
    movie = _probe(run / "full_follow_ball_zoom.mp4")
    assert movie.height == 1080
    assert abs(movie.duration_s - DURATION_S) < 0.6

    progress = json.loads((run / "progress.json").read_text())
    assert progress["progress"] == 1.0 and progress["stage"] == "completed"
    assert progress["cancelled"] is False
    assert progress["stages"] == ["proxy", "tracking", "teams", "analysis", "camera_plan", "render", "clips_reel"]
    assert set(progress["stage_timings"]) == set(progress["stages"])
    for key in ("stage_index", "stage_count", "stage_progress", "eta_s", "elapsed_s", "fps_processing",
                "message", "device"):
        assert key in progress

    # Both scripted goals, no false goals.
    events = json.loads((run / "analysis_events.json").read_text())
    goals = sorted(e["t"] for e in events["events"] if e["type"] == "goal")
    truth = sorted(t for t, _side in synthetic["gt"].goal_times_s)
    assert len(goals) == len(truth) == 2
    for found, expected in zip(goals, truth):
        assert abs(found - expected) < 1.5
    assert all("ball_xy" in e["evidence"] for e in events["events"] if e["type"] == "goal")
    assert events["reel_plan"]["clips"]

    # Legacy manifests keep their shape; tracking manifest points at v2 files.
    bookmarks = json.loads((run / "analysis_bookmarks.json").read_text())
    for key in ("settings", "stats", "goal_events", "game_states_path", "bookmarks", "trim_offset_seconds"):
        assert key in bookmarks
    assert any(b["event_type"] == "goal" for b in bookmarks["bookmarks"])
    tracking_manifest = json.loads((run / "analysis_tracking.json").read_text())
    assert tracking_manifest["tracks_file"] == "tracks.npz"
    assert tracking_manifest["proxy"]["height"] == 720
    teams = json.loads((run / "analysis_team_stats.json").read_text())
    assert set(teams["teams"]) == {"0", "1"}
    summary = json.loads((run / "run_summary.json").read_text())
    print(f"\n[e2e] full run {elapsed:.1f}s, stage timings: {summary['timings']}")
    assert elapsed < 120


def test_analysis_only_decodes_source_once(synthetic, decode_recorder, tmp_path):
    out = tmp_path / "analysis_only"
    assert _run(synthetic, out, analysis_only=True) is True
    assert decode_recorder.decodes == ["build_proxy"], decode_recorder.decodes
    assert decode_recorder.cv2_opens == []
    progress = json.loads((out / "progress.json").read_text())
    assert progress["progress"] == 1.0
    assert progress["stages"] == ["proxy", "tracking", "teams", "analysis", "camera_plan"]
    assert (out / "analysis_events.json").exists() and (out / "tracks.npz").exists()
    assert not list(out.glob("*.mp4"))[1:]  # only the proxy
    assert not (out / "full_follow_ball_zoom.mp4").exists()


def test_reuse_tracking_rerenders_follow_player_without_tracking(synthetic, decode_recorder, full_run_dir,
                                                                  tmp_path, monkeypatch):
    from backend.services import tracking_engine
    from backend.services.tracking_types import TrackingResult

    if not TrackingResult.exists(full_run_dir):
        pytest.skip("full run did not complete")

    def no_tracking(*_a, **_k):
        raise AssertionError("track_video must not run when tracking is reused")

    monkeypatch.setattr(tracking_engine, "track_video", no_tracking)
    tracks = TrackingResult.load(full_run_dir)
    focus = max(tracks.players, key=lambda tid: len(tracks.players[tid]))
    out = tmp_path / "rerender"
    assert _run(synthetic, out, camera_mode="follow_player", focus_track_id=focus,
                reuse_tracking_from=str(full_run_dir), broadcast_reel=False,
                detector_factory=None) is True
    # Only the render decodes the source; the proxy is reused, not rebuilt.
    assert decode_recorder.decodes == ["render"], decode_recorder.decodes
    assert (out / "full_follow_ball_zoom.mp4").exists()
    assert (out / "proxy_720p.mp4").exists()
    meta = json.loads((out / "tracks_meta.json").read_text())
    assert meta["focus_track_id"] == focus
    decisions = [json.loads(line) for line in (out / "camera_decisions.jsonl").read_text().splitlines()[:400]]
    assert any(d["focus"] == "player" for d in decisions)


def test_cancel_event_stops_run_and_marks_progress(synthetic, tmp_path):
    import VideoHighlights

    cancel = threading.Event()
    calls: List[str] = []

    def legacy_cb(stage, progress, message, data):  # noqa: ANN001
        calls.append(stage)
        if stage == "tracking":
            cancel.set()

    out = tmp_path / "canceled"
    ok = VideoHighlights.process_video_highlights(
        synthetic["source"], str(out), profile="fast", camera_mode="follow_ball", detect_cards=False,
        llm_report=False, progress_callback=legacy_cb, cancel_event=cancel,
        detector_factory=_detector_factory(synthetic["gt"], delay_s=0.05),
    )
    assert ok is False
    progress = json.loads((out / "progress.json").read_text())
    assert progress["cancelled"] is True and progress["stage"] == "canceled"
    assert "canceled" in calls
    assert not (out / "analysis_events.json").exists()


# ---------------------------------------------------------------------------
# Through the API worker (DB sync + cancel)
# ---------------------------------------------------------------------------


def _create_match(client, source: str) -> str:
    resp = client.post("/v1/matches", json={"name": "Synthetic", "source_video_path": source, "metadata": {}})
    assert resp.status_code == 201, resp.text
    return resp.json()["match_id"]


def test_api_job_syncs_v2_events(client, synthetic, tmp_path, monkeypatch):
    from backend.config import settings
    from backend.services import job_runner

    monkeypatch.setattr(settings, "output_root", str(tmp_path / "outputs"))
    monkeypatch.setattr(job_runner, "DETECTOR_FACTORY_OVERRIDE", _detector_factory(synthetic["gt"]))
    match_id = _create_match(client, synthetic["source"])
    job = client.post(f"/v1/matches/{match_id}/jobs", json={"config": {
        "profile": "fast", "analysis_only": True, "camera_mode": "follow_ball", "detect_cards": False,
        "llm_report": False, "pitch_corners": synthetic["pitch_corners"], "team_left": "RED", "team_right": "BLUE",
    }})
    assert job.status_code == 201, job.text
    job_id = job.json()["job_id"]
    assert job.json()["config"]["proxy_height"] == 720  # resolved profile stored

    run = client.post("/v1/jobs/worker/run-once")
    assert run.status_code == 200 and run.json()["job_id"] == job_id
    payload = client.get(f"/v1/jobs/{job_id}").json()
    assert payload["status"] == "completed", payload.get("error_message")
    result = payload["result"]
    for key in ("events_path", "player_stats_path", "team_stats_path", "proxy_path", "timings"):
        assert result.get(key), key
    assert Path(result["output_dir"]) == (tmp_path / "outputs" / job_id).resolve()

    events_doc = json.loads(Path(result["events_path"]).read_text())
    items = client.get(f"/v1/matches/{match_id}/events?job_id={job_id}&limit=200").json()["items"]
    assert len(items) == len(events_doc["events"])
    by_id = {item["source"]["analysis_event_id"]: item for item in items}
    goals = [e for e in events_doc["events"] if e["type"] == "goal"]
    assert len(goals) == 2
    for goal in goals:
        row = by_id[goal["id"]]
        assert row["event_type"] == "goal"
        assert row["team_id"] == {0: "home", 1: "away"}.get(goal["team"])
        if goal.get("player_track_id") is not None:
            assert row["participants"][0]["track_id"] == goal["player_track_id"]

    stats = client.get(f"/v1/matches/{match_id}/stats?job_id={job_id}")
    assert stats.status_code == 200, stats.text
    by_key = {s["key"]: s for s in stats.json()["stats"]}
    assert by_key["possession"]["method"] == "tracking_v2" and by_key["possession"]["available"]
    assert by_key["goals"]["method"] == "tracking_v2"


def test_api_job_cancel_marks_job_canceled(client, synthetic, tmp_path, monkeypatch):
    from backend.config import settings
    from backend.services import job_runner

    monkeypatch.setattr(settings, "output_root", str(tmp_path / "outputs"))
    monkeypatch.setattr(job_runner, "DETECTOR_FACTORY_OVERRIDE", _detector_factory(synthetic["gt"], delay_s=0.3))
    monkeypatch.setattr(job_runner, "CANCEL_POLL_S", 0.2)
    match_id = _create_match(client, synthetic["source"])
    monkeypatch.setattr(settings, "job_execution_mode", "inline")
    job = client.post(f"/v1/matches/{match_id}/jobs", json={"config": {
        "profile": "fast", "camera_mode": "follow_ball", "render_full_follow_cam": True,
        "detect_cards": False, "llm_report": False,
    }})
    assert job.status_code == 201, job.text
    job_id = job.json()["job_id"]
    progress_path = tmp_path / "outputs" / job_id / "progress.json"

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if progress_path.exists() and json.loads(progress_path.read_text()).get("stage") == "tracking":
            break
        time.sleep(0.1)
    else:
        pytest.fail("job never reached the tracking stage")
    resp = client.post(f"/v1/jobs/{job_id}/cancel")
    assert resp.status_code == 200, resp.text

    deadline = time.monotonic() + 30
    status = None
    while time.monotonic() < deadline:
        status = client.get(f"/v1/jobs/{job_id}").json()["status"]
        if status == "canceled":
            break
        time.sleep(0.2)
    assert status == "canceled"
    progress = json.loads(progress_path.read_text())
    assert progress["cancelled"] is True
    assert not (tmp_path / "outputs" / job_id / "full_follow_ball_zoom.mp4").exists()
