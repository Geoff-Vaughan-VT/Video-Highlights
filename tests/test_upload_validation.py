from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.database import session_scope
from backend.models import Tenant


def _create_match(client: TestClient) -> str:
    response = client.post(
        "/v1/matches",
        json={"name": "Upload Rules Match", "source_video_path": "", "metadata": {}},
    )
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def test_upload_policy_endpoint(client: TestClient) -> None:
    response = client.get("/v1/matches/upload-policy")
    assert response.status_code == 200, response.text
    policy = response.json()
    assert policy["max_upload_gb"] == settings.upload_max_gb
    assert policy["extended_upload_enabled"] is False
    assert policy["extended_max_upload_gb"] == settings.upload_extended_max_gb
    assert ".mp4" in policy["allowed_extensions"]
    assert policy["processing_sla_hours"] == [
        settings.processing_sla_hours_min,
        settings.processing_sla_hours_max,
    ]


def test_upload_rejects_unsupported_extension(client: TestClient) -> None:
    match_id = _create_match(client)
    response = client.post(
        f"/v1/matches/{match_id}/assets/upload",
        files={"file": ("notes.txt", b"not-a-video", "text/plain")},
    )
    assert response.status_code == 400
    assert "unsupported video format" in response.json()["error"]["message"].lower()


def test_upload_rejects_oversize_file(client: TestClient) -> None:
    match_id = _create_match(client)
    original = settings.upload_max_gb
    settings.upload_max_gb = 1e-6  # ~1 KB cap
    try:
        response = client.post(
            f"/v1/matches/{match_id}/assets/upload",
            files={"file": ("big.mp4", b"x" * 4096, "video/mp4")},
        )
        assert response.status_code == 413
        message = response.json()["error"]["message"]
        assert "upload limit" in message
        assert "paid add-on" in message
    finally:
        settings.upload_max_gb = original


def test_extended_entitlement_raises_cap(client: TestClient) -> None:
    match_id = _create_match(client)

    match = client.get(f"/v1/matches/{match_id}").json()
    tenant_id = match["tenant_id"]
    with session_scope() as session:
        tenant = session.get(Tenant, tenant_id)
        metadata = dict(tenant.metadata_json or {})
        metadata["entitlements"] = {"extended_uploads": True}
        tenant.metadata_json = metadata
        session.add(tenant)

    policy = client.get("/v1/matches/upload-policy").json()
    assert policy["extended_upload_enabled"] is True
    assert policy["max_upload_gb"] == settings.upload_extended_max_gb

    original = settings.upload_max_gb
    settings.upload_max_gb = 1e-6  # standard cap would reject this upload
    try:
        response = client.post(
            f"/v1/matches/{match_id}/assets/upload",
            files={"file": ("big.mp4", b"x" * 4096, "video/mp4")},
        )
        assert response.status_code == 201, response.text
    finally:
        settings.upload_max_gb = original


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)
def test_upload_rejects_video_shorter_than_minimum(client: TestClient, tmp_path) -> None:
    import subprocess

    short_video = tmp_path / "short.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=green:s=64x64:d=2",
            "-pix_fmt", "yuv420p", str(short_video),
        ],
        check=True,
        capture_output=True,
    )

    match_id = _create_match(client)
    original = settings.upload_min_duration_seconds
    settings.upload_min_duration_seconds = 1800.0
    try:
        response = client.post(
            f"/v1/matches/{match_id}/assets/upload",
            files={"file": ("short.mp4", short_video.read_bytes(), "video/mp4")},
        )
        assert response.status_code == 400
        assert "at least 30 minutes" in response.json()["error"]["message"]
    finally:
        settings.upload_min_duration_seconds = original


def test_upload_records_probe_metadata_when_available(client: TestClient) -> None:
    match_id = _create_match(client)
    response = client.post(
        f"/v1/matches/{match_id}/assets/upload",
        files={"file": ("clip.mp4", b"fake-binary-content", "video/mp4")},
    )
    # Fake bytes are not probe-able; the upload must still succeed with the
    # min-duration gate disabled (duration unknown -> allowed).
    assert response.status_code == 201, response.text
    assert response.json()["size_bytes"] > 0
