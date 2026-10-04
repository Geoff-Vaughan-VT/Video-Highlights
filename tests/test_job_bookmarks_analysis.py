from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient


def _install_fake_videohighlights(monkeypatch, bookmark_count: int = 2, captured: dict | None = None) -> None:
    fake = types.ModuleType("VideoHighlights")

    def parse_time(value: str) -> float:
        return float(value)

    def process_video_highlights(**kwargs) -> bool:
        if captured is not None:
            captured.update(kwargs)
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        bookmarks = []
        for idx in range(bookmark_count):
            start_s = 10.0 + (idx * 20.0)
            end_s = start_s + 6.0
            bookmarks.append(
                {
                    "bookmark_id": f"bm_{idx + 1:04d}",
                    "index": idx + 1,
                    "event_type": "goal" if idx == 0 else "corner_kick",
                    "label": "auto_candidate",
                    "confidence": 0.8,
                    "start_s": start_s,
                    "occurred_at_s": start_s + 2.5,
                    "end_s": end_s,
                    "duration_s": 6.0,
                    "sources": ["motion", "audio"],
                    "signals": {"speed_overlap_s": 1.2, "audio_overlap_s": 0.5},
                }
            )

        manifest = {
            "analysis_only": bool(kwargs.get("analysis_only", False)),
            "bookmarks": bookmarks,
            "stats": {"bookmark_count": bookmark_count},
        }
        (output_dir / "analysis_bookmarks.json").write_text(json.dumps(manifest), encoding="utf-8")
        (output_dir / "analysis_bookmarks.csv").write_text(
            "bookmark_id,index,event_type,label,confidence,start_s,occurred_at_s,end_s,duration_s,sources\n",
            encoding="utf-8",
        )
        return True

    fake.parse_time = parse_time
    fake.process_video_highlights = process_video_highlights
    monkeypatch.setitem(sys.modules, "VideoHighlights", fake)


def _default_output_root(monkeypatch, tmp_path: Path) -> None:
    """Runs go to <output_root>/<job_id> (no output_dir override)."""
    from backend.config import settings

    monkeypatch.setenv("VH_ALLOW_OUTPUT_DIR_OVERRIDE", "0")
    monkeypatch.setattr(settings, "output_root", str(tmp_path / "outputs"))


def test_analysis_only_job_persists_bookmarks_and_events(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    _install_fake_videohighlights(monkeypatch, bookmark_count=2)

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake-video")
    _default_output_root(monkeypatch, tmp_path)

    match = client.post(
        "/v1/matches",
        json={
            "name": "Bookmark Match",
            "source_video_path": str(source_video),
            "metadata": {},
        },
    )
    assert match.status_code == 201, match.text
    match_id = match.json()["match_id"]

    job = client.post(
        f"/v1/matches/{match_id}/jobs",
        json={
            "config": {
                "analysis_only": True,
                "model_version": "event-v1",
                "focus_event_types": ["goal", "corner_kick"],
            }
        },
    )
    assert job.status_code == 201, job.text
    job_id = job.json()["job_id"]

    run_once = client.post("/v1/jobs/worker/run-once")
    assert run_once.status_code == 200, run_once.text
    assert run_once.json()["job_id"] == job_id

    done = client.get(f"/v1/jobs/{job_id}")
    assert done.status_code == 200, done.text
    payload = done.json()
    assert payload["status"] == "completed"
    assert payload["result"]["analysis_only"] is True
    assert payload["result"]["bookmarks_count"] == 2
    assert len(payload["result"]["bookmarks"]) == 2
    assert payload["result"]["artifact_count"] == 0

    events = client.get(f"/v1/matches/{match_id}/events?job_id={job_id}&limit=50")
    assert events.status_code == 200, events.text
    items = events.json()["items"]
    assert len(items) == 2
    assert all(item["job_id"] == job_id for item in items)


def test_job_runner_passes_trim_window_to_pipeline(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}
    _install_fake_videohighlights(monkeypatch, bookmark_count=1, captured=captured)

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake-video")
    _default_output_root(monkeypatch, tmp_path)

    match = client.post(
        "/v1/matches",
        json={
            "name": "Trim Window Match",
            "source_video_path": str(source_video),
            "metadata": {},
        },
    )
    assert match.status_code == 201, match.text
    match_id = match.json()["match_id"]

    job = client.post(
        f"/v1/matches/{match_id}/jobs",
        json={
            "config": {
                "analysis_only": True,
                "trim_start": 15.0,
                "trim_end": 135.0,
                "model_version": "event-v1",
            }
        },
    )
    assert job.status_code == 201, job.text
    job_id = job.json()["job_id"]

    run_once = client.post("/v1/jobs/worker/run-once")
    assert run_once.status_code == 200, run_once.text
    assert run_once.json()["job_id"] == job_id

    assert Path(captured["output_dir"]) == tmp_path / "outputs" / job_id
    assert captured["trim_start"] == 15.0
    assert captured["trim_end"] == 135.0
