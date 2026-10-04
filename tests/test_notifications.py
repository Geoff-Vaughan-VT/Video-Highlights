from __future__ import annotations

from fastapi.testclient import TestClient

from backend.config import settings
from backend.database import session_scope
from backend.models import ProcessingJob
from backend.services.notifications import notify_job_terminal_state


def _create_match(client: TestClient, notify_email: str | None = None) -> str:
    metadata = {"notify_email": notify_email} if notify_email else {}
    response = client.post(
        "/v1/matches",
        json={"name": "Notify Match", "source_video_path": "C:/tmp/nonexistent.mp4", "metadata": metadata},
    )
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def _create_job(client: TestClient, match_id: str, config: dict | None = None) -> str:
    response = client.post(f"/v1/matches/{match_id}/jobs", json={"config": config or {}})
    assert response.status_code == 201, response.text
    return response.json()["job_id"]


def _complete_job_and_notify(job_id: str, status: str = "completed") -> None:
    with session_scope() as session:
        job = session.get(ProcessingJob, job_id)
        job.status = status
        if status == "failed":
            job.error_message = "boom"
        session.add(job)
        notify_job_terminal_state(session, job)


def test_completion_notification_sent_console(client: TestClient) -> None:
    match_id = _create_match(client, notify_email="coach@example.com")
    job_id = _create_job(client, match_id)
    _complete_job_and_notify(job_id)

    response = client.get(f"/v1/jobs/{job_id}/notifications")
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert len(items) == 1
    entry = items[0]
    assert entry["status"] == "sent"
    assert entry["backend"] == "console"
    assert entry["recipient"] == "coach@example.com"
    assert "ready" in entry["subject"]


def test_job_config_email_overrides_match_email(client: TestClient) -> None:
    match_id = _create_match(client, notify_email="coach@example.com")
    job_id = _create_job(client, match_id, config={"notify_email": "override@example.com"})
    _complete_job_and_notify(job_id, status="failed")

    items = client.get(f"/v1/jobs/{job_id}/notifications").json()["items"]
    assert len(items) == 1
    assert items[0]["recipient"] == "override@example.com"
    assert "failed" in items[0]["subject"]


def test_notification_skipped_without_recipient(client: TestClient) -> None:
    match_id = _create_match(client)
    job_id = _create_job(client, match_id)
    _complete_job_and_notify(job_id)

    items = client.get(f"/v1/jobs/{job_id}/notifications").json()["items"]
    assert len(items) == 1
    assert items[0]["status"] == "skipped"
    assert items[0]["recipient"] is None


def test_notification_disabled_backend_records_nothing(client: TestClient) -> None:
    match_id = _create_match(client, notify_email="coach@example.com")
    job_id = _create_job(client, match_id)
    original = settings.notify_backend
    settings.notify_backend = "disabled"
    try:
        _complete_job_and_notify(job_id)
    finally:
        settings.notify_backend = original

    items = client.get(f"/v1/jobs/{job_id}/notifications").json()["items"]
    assert items == []


def test_non_terminal_status_does_not_notify(client: TestClient) -> None:
    match_id = _create_match(client, notify_email="coach@example.com")
    job_id = _create_job(client, match_id)
    _complete_job_and_notify(job_id, status="running")

    items = client.get(f"/v1/jobs/{job_id}/notifications").json()["items"]
    assert items == []
