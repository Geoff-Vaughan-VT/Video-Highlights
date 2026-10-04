"""Studio API: run summary, player picker, labels, ranges, traversal, progress."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.config import settings
from backend.routers import studio
from backend.services.synthetic_match import SyntheticMatchSpec, generate_synthetic_match

RUN_ID = "run_test"


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tiny_mp4(path: Path) -> None:
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "testsrc=size=160x90:rate=10:duration=1", "-pix_fmt", "yuv420p", str(path)],
            check=True, timeout=60,
        )
    if not path.exists() or path.stat().st_size == 0:  # pragma: no cover - ffmpeg missing
        path.write_bytes(bytes(range(256)) * 64)


@pytest.fixture()
def run_root(tmp_path: Path):
    previous = settings.output_root
    root = tmp_path / "outputs"
    run = root / RUN_ID
    run.mkdir(parents=True)
    settings.output_root = str(root)
    studio.clear_track_cache()

    spec = SyntheticMatchSpec(width=320, height=180, fps=10.0, duration_s=3.0, players_per_team=2,
                              goals=[(1.0, "right")], player_w_px=6, player_h_px=14)
    truth = generate_synthetic_match(tmp_path / "synthetic.mp4", spec)
    truth.tracking.save(run)

    _tiny_mp4(run / "proxy_1080p.mp4")
    shutil.copy(run / "proxy_1080p.mp4", run / "full_follow_ball_zoom.mp4")
    (run / "thumbs").mkdir()
    (run / "thumbs" / "0001.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    (run / "secret.exe").write_bytes(b"nope")
    (root / "outside.json").write_text("{}", encoding="utf-8")

    _write(run / "analysis_events.json", {
        "generated_at": "2026-10-01T10:00:00Z",
        "trim_offset_seconds": 12.0,
        "events": [
            {"id": "ev_0001", "type": "goal", "t": 1.2, "t_start": 0.5, "t_end": 2.5, "team": 0,
             "team_name": "HOME", "player_track_id": 1, "confidence": 0.9, "excitement": 0.95,
             "reason": "ball crossed the line", "sources": ["ball_tracking", "audio"]},
            {"id": "ev_0002", "type": "shot", "t": 2.0, "t_start": 1.5, "t_end": 2.6, "team": 1,
             "team_name": "AWAY", "player_track_id": 3, "confidence": 0.7, "excitement": 0.6},
        ],
        "reel_plan": {"target_duration_s": 60, "selected_event_ids": ["ev_0001"], "total_duration_s": 2.0},
    })
    _write(run / "analysis_team_stats.json", {
        "teams": {"0": {"name": "HOME", "color_hex": "#d32f2f", "goals": 1, "possession_pct": 55.0},
                  "1": {"name": "AWAY", "color_hex": "#1976d2", "goals": 0, "possession_pct": 45.0}},
        "timeline": {"bin_s": 60, "possession_pct_team0": [55.0], "momentum": [0.2]},
        "quality": {"team_label_coverage_pct": 90.0, "ball_coverage_pct": 70.0},
    })
    _write(run / "analysis_player_stats.json", {
        "pitch_calibration": {"source": "auto", "confidence": 0.5},
        "players": [{"track_id": 1, "team": 0, "distance_m": 120.0, "top_speed_mps": 7.5,
                     "speed_series": [{"t": 0.0, "v": 1.0}, {"t": 1.0, "v": 2.0}]}],
    })
    _write(run / "analysis_bookmarks.json", {"generated_at": "2026-10-01T10:00:00Z", "bookmarks": []})
    yield run
    settings.output_root = previous
    studio.clear_track_cache()


def test_run_summary_includes_v2_artifacts(client, run_root):
    response = client.get(f"/v1/studio/runs/{RUN_ID}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["events_source"] == "analysis_events"
    assert [e["id"] for e in body["events"]] == ["ev_0001", "ev_0002"]
    assert body["videos"]["proxy"] == "proxy_1080p.mp4"
    assert body["videos"]["movie"] == "full_follow_ball_zoom.mp4"
    assert body["thumbs"] == [{"name": "0001.jpg", "t": 0.0}]
    assert body["tracks_available"] is True
    assert body["trim_offset_seconds"] == 12.0
    assert body["team_names"] == ["HOME", "AWAY"]
    assert body["score"] == {"0": 1, "1": 0}
    # speed series is stripped from the summary and served per player
    player = body["player_stats"]["players"][0]
    assert "speed_series" not in player and player["has_speed_series"] is True
    detail = client.get(f"/v1/studio/runs/{RUN_ID}/player-stats/1").json()
    assert len(detail["player"]["speed_series"]) == 2

    listing = client.get("/v1/studio/runs").json()
    assert listing["runs"][0]["run_id"] == RUN_ID
    assert "events" not in listing["runs"][0]


def test_summary_and_library_link_jobs(client, run_root):
    match = client.post("/v1/matches", json={"name": "Cup final", "home_team_name": "HOME",
                                              "away_team_name": "AWAY", "source_video_path": "/tmp/x.mp4"}).json()
    job = client.post(f"/v1/matches/{match['match_id']}/jobs",
                      json={"config": {"output_dir": str(run_root), "profile": "fast"}}).json()
    body = client.get(f"/v1/studio/runs/{RUN_ID}").json()
    assert body["job_id"] == job["job_id"]
    assert body["match_id"] == match["match_id"]

    library = client.get("/v1/studio/library").json()
    entry = next(m for m in library["matches"] if m["match_id"] == match["match_id"])
    assert entry["latest_job"]["job_id"] == job["job_id"]
    assert entry["job_count"] == 1
    assert not any(r["run_id"] == RUN_ID for r in library["unlinked_runs"])

    jobs = client.get("/v1/studio/jobs?limit=10").json()
    assert jobs["items"][0]["match_name"] == "Cup final"
    assert jobs["items"][0]["config"]["profile"] == "fast"
    assert jobs["active_count"] == 1

    # Materialize an analysis event so feedback / clip endpoints can use it.
    made = client.post(f"/v1/studio/runs/{RUN_ID}/events/ev_0002/db-event").json()
    assert made["created"] is True
    again = client.post(f"/v1/studio/runs/{RUN_ID}/events/ev_0002/db-event").json()
    assert again == {**made, "created": False}
    event = client.get(f"/v1/matches/{match['match_id']}/events/{made['db_event_id']}").json()
    assert event["occurred_at_ms"] == 14000  # window t 2.0 + trim 12.0
    assert event["team_id"] == "away"
    feedback = client.post(f"/v1/matches/{match['match_id']}/events/{made['db_event_id']}/feedback",
                           json={"feedback_type": "false_positive"})
    assert feedback.status_code == 201


def test_tracks_at_time(client, run_root):
    response = client.get(f"/v1/studio/runs/{RUN_ID}/tracks", params={"t": 1.0})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["frame_width"] == 320 and body["frame_height"] == 180
    assert body["t_source"] == pytest.approx(1.0)
    ids = {p["track_id"] for p in body["players"]}
    assert ids == {1, 2, 3, 4, 5}  # 2 + 2 players + referee
    first = body["players"][0]
    for key in ("x1", "y1", "x2", "y2", "team"):
        assert key in first
    assert client.get(f"/v1/studio/runs/{RUN_ID}/tracks", params={"t": 99.0}).json()["players"] == []
    assert client.get(f"/v1/studio/runs/{RUN_ID}/tracks").status_code == 400  # t required


def test_player_labels_roundtrip(client, run_root):
    assert client.get(f"/v1/studio/runs/{RUN_ID}/player-labels").json()["labels"] == {}
    put = client.put(f"/v1/studio/runs/{RUN_ID}/player-labels",
                     json={"labels": {"1": {"name": "Sam Kerr", "number": "20"}, "2": {"name": "  "}}})
    assert put.status_code == 200, put.text
    labels = client.get(f"/v1/studio/runs/{RUN_ID}/player-labels").json()["labels"]
    assert labels == {"1": {"name": "Sam Kerr", "number": "20"}}  # empty label dropped
    assert (run_root / "player_labels.json").exists()
    tracked = client.get(f"/v1/studio/runs/{RUN_ID}/tracks", params={"t": 1.0}).json()["players"]
    one = next(p for p in tracked if p["track_id"] == 1)
    assert one["name"] == "Sam Kerr" and one["number"] == "20"
    bad = client.put(f"/v1/studio/runs/{RUN_ID}/player-labels", json={"labels": {"abc": {"name": "x"}}})
    assert bad.status_code == 400


def test_range_requests(client, run_root):
    url = f"/v1/studio/runs/{RUN_ID}/file/proxy_1080p.mp4"
    size = (run_root / "proxy_1080p.mp4").stat().st_size
    full = client.get(url)
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"
    assert "max-age" in full.headers["cache-control"]
    assert len(full.content) == size

    part = client.get(url, headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 0-99/{size}"
    assert part.content == full.content[:100]

    tail = client.get(url, headers={"Range": "bytes=-10"})
    assert tail.status_code == 206 and tail.content == full.content[-10:]
    open_ended = client.get(url, headers={"Range": f"bytes={size - 5}-"})
    assert open_ended.status_code == 206 and len(open_ended.content) == 5

    unsatisfiable = client.get(url, headers={"Range": f"bytes={size + 10}-"})
    assert unsatisfiable.status_code == 416
    assert unsatisfiable.headers["content-range"] == f"bytes */{size}"

    head = client.head(url, headers={"Range": "bytes=10-19"})
    assert head.status_code == 206 and head.headers["content-length"] == "10"

    etag = full.headers["etag"]
    assert client.get(url, headers={"If-None-Match": etag}).status_code == 304
    thumb = client.get(f"/v1/studio/runs/{RUN_ID}/thumb/0001.jpg")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"


def test_path_traversal_rejected(client, run_root):
    for path in (
        "/v1/studio/runs/../file/outside.json",
        "/v1/studio/runs/..%2F/file/outside.json",
        f"/v1/studio/runs/{RUN_ID}/file/..%2Foutside.json",
        f"/v1/studio/runs/{RUN_ID}/file/secret.exe",
        f"/v1/studio/runs/{RUN_ID}/thumb/..%2F..%2Foutside.json",
        f"/v1/studio/runs/{RUN_ID}/thumb/0001.exe",
        "/v1/studio/runs/%2E%2E/tracks?t=1",
        "/v1/studio/runs/does_not_exist/player-labels",
    ):
        response = client.get(path)
        assert response.status_code in {400, 404, 405}, (path, response.status_code)
        assert b"{}" != response.content


def _set_status(job_id: str, status: str) -> None:
    from sqlmodel import Session

    from backend import database
    from backend.models import ProcessingJob

    with Session(database.engine) as session:
        row = session.get(ProcessingJob, job_id)
        row.status = status
        session.add(row)
        session.commit()


def test_progress_endpoint(client, run_root):
    match = client.post("/v1/matches", json={"name": "M", "source_video_path": "/tmp/x.mp4"}).json()
    job = client.post(f"/v1/matches/{match['match_id']}/jobs", json={"config": {"output_dir": str(run_root)}}).json()

    derived = client.get(f"/v1/studio/jobs/{job['job_id']}/progress").json()
    assert derived["source"] == "job_log"
    assert derived["status"] == "queued"
    assert derived["run_id"] == RUN_ID

    _write(run_root / "progress.json", {"stage": "tracking", "stage_index": 2, "stage_count": 6, "progress": 0.4,
                                        "stage_progress": 0.5, "eta_s": 300, "elapsed_s": 200,
                                        "fps_processing": 41.0, "message": "detecting", "device": "cuda:0",
                                        "cancelled": False})
    # A queued job never trusts progress.json (a rerun may share the folder).
    assert client.get(f"/v1/studio/jobs/{job['job_id']}/progress").json()["source"] == "job_log"
    _set_status(job["job_id"], "running")
    live = client.get(f"/v1/studio/jobs/{job['job_id']}/progress").json()
    assert live["source"] == "progress_file"
    assert live["stage"] == "tracking" and live["eta_s"] == 300 and live["device"] == "cuda:0"

    listed = client.get("/v1/studio/jobs").json()["items"][0]
    assert listed["live"]["stage"] == "tracking"
    assert client.get("/v1/studio/jobs/job_missing/progress").status_code == 404


def test_system_info(client, run_root):
    body = client.get("/v1/studio/system").json()
    assert body["output_root"] == str(Path(settings.output_root).resolve())
    assert "encoders" in body["ffmpeg"]


CORNERS = [[0.05, 0.1], [0.95, 0.1], [0.98, 0.9], [0.02, 0.9]]


def test_calibration_roundtrip(client, run_root):
    url = f"/v1/studio/runs/{RUN_ID}/calibration"
    assert client.get(url).json() == {"run_id": RUN_ID, "calibration": None}
    assert client.get(f"/v1/studio/runs/{RUN_ID}").json()["calibration"] is None

    put = client.put(url, json={
        "pitch_corners": CORNERS,
        "goal_box_left": {"x1": 0.06, "y1": 0.55, "x2": 0.0, "y2": 0.45, "normalized": True},
        "goal_box_right": None,
        "frame_width": 999, "frame_height": 999,  # tracks_meta wins
        "normalized": True,
    })
    assert put.status_code == 200, put.text
    saved = put.json()["calibration"]
    assert saved["pitch_corners"] == CORNERS
    assert saved["goal_box_left"] == {"x1": 0.0, "y1": 0.45, "x2": 0.06, "y2": 0.55, "normalized": True}
    assert saved["goal_box_right"] is None
    assert (saved["frame_width"], saved["frame_height"]) == (320, 180)
    assert saved["normalized"] is True and saved["updated_at"]
    assert (run_root / "calibration.json").exists()

    assert client.get(url).json()["calibration"] == saved
    assert client.get(f"/v1/studio/runs/{RUN_ID}").json()["calibration"] == saved

    # Pixel input is converted with the run's frame size and stored normalized.
    pixels = [[x * 320, y * 180] for x, y in CORNERS]
    put_px = client.put(url, json={"pitch_corners": pixels, "normalized": False,
                                   "goal_box_right": {"x1": 304, "y1": 81, "x2": 320, "y2": 99}})
    assert put_px.status_code == 200, put_px.text
    stored = put_px.json()["calibration"]
    assert [v for p in stored["pitch_corners"] for v in p] == pytest.approx([v for p in CORNERS for v in p])
    right = stored["goal_box_right"]
    assert right.pop("normalized") is True
    assert right == pytest.approx({"x1": 0.95, "y1": 0.45, "x2": 1.0, "y2": 0.55})

    deleted = client.delete(url)
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True
    assert client.get(url).json()["calibration"] is None


def test_calibration_validation(client, run_root):
    url = f"/v1/studio/runs/{RUN_ID}/calibration"
    bad_bodies = [
        {"pitch_corners": CORNERS[:3]},  # 3 corners
        {"pitch_corners": CORNERS + [[0.5, 0.5]]},  # 5 corners
        {"pitch_corners": [[0.1, 0.1, 0.2], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]},
        {"pitch_corners": "nope"},
        {},
        {"pitch_corners": [[0.1, 0.1], [1.2, 0.1], [0.9, 0.9], [0.1, 0.9]]},  # > 1 while normalized
        {"pitch_corners": [[-0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]},
        {"pitch_corners": [[0.1, 0.1], ["x", 0.1], [0.9, 0.9], [0.1, 0.9]]},
        {"pitch_corners": [[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]]},  # wrong order (self-crossing)
        {"pitch_corners": CORNERS, "goal_box_left": {"x1": 0.1, "y1": 0.2}},
        {"pitch_corners": CORNERS, "goal_box_left": {"x1": 0.1, "y1": 0.2, "x2": 1.5, "y2": 0.4}},
        {"pitch_corners": CORNERS, "goal_box_right": {"x1": 0.5, "y1": 0.5, "x2": 0.5, "y2": 0.5}},
        {"pitch_corners": [[0, 0], [400, 0], [320, 180], [0, 180]], "normalized": False},  # outside frame
    ]
    for body in bad_bodies:
        response = client.put(url, json=body)
        assert response.status_code == 400, (body, response.status_code, response.text)
    assert not (run_root / "calibration.json").exists()


def test_calibration_path_safety(client, run_root):
    body = {"pitch_corners": CORNERS}
    for run_id in ("..", "..%2F..", "%2E%2E", "run_test%2F..%2F..", "does_not_exist", "a b"):
        for method in ("get", "put", "delete"):
            kwargs = {"json": body} if method == "put" else {}
            response = getattr(client, method)(f"/v1/studio/runs/{run_id}/calibration", **kwargs)
            assert response.status_code in {400, 404, 405}, (run_id, method, response.status_code)
    assert not (run_root.parent / "calibration.json").exists()
    assert not (run_root.parent.parent / "calibration.json").exists()
