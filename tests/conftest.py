from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.database import reset_db, set_engine
from backend.main import app

# API jobs always write to <output_root>/<job_id> (backend/schemas.py
# output_dir_override_allowed). Several suites seed run folders through
# config.output_dir, which needs the test-only override; the path-policy
# tests (tests/test_job_config.py, tests/test_path_policy.py) switch it off.
os.environ.setdefault("VH_ALLOW_OUTPUT_DIR_OVERRIDE", "1")


@pytest.fixture(scope="function")
def isolated_db(tmp_path: Path) -> Generator[str, None, None]:
    db_path = tmp_path / "test_api.db"
    set_engine(f"sqlite:///{db_path}")
    reset_db()
    yield str(db_path)


@pytest.fixture(scope="function")
def client(isolated_db: str) -> Generator[TestClient, None, None]:
    settings.job_execution_mode = "queue"
    settings.auth_required = False
    settings.skip_user_management = False
    settings.base_tenant_slug = "sandbox"
    settings.base_tenant_name = "Sandbox Tenant"
    settings.test_mode = False
    settings.log_level = "INFO"
    settings.job_log_detail = "basic"
    settings.persist_job_logs = True
    settings.auth_tokens_raw = ""
    settings.auth_bootstrap_key = None
    settings.jwt_secret = None
    settings.jwt_issuer = "video-highlights"
    settings.jwt_audience = None
    settings.upload_max_gb = 0.0
    settings.upload_extended_max_gb = 0.0
    settings.upload_min_duration_seconds = 0.0
    settings.notify_backend = "console"
    settings.clear_api_token_cache()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def auth_client(isolated_db: str) -> Generator[TestClient, None, None]:
    settings.job_execution_mode = "queue"
    settings.auth_required = True
    settings.skip_user_management = False
    settings.base_tenant_slug = "sandbox"
    settings.base_tenant_name = "Sandbox Tenant"
    settings.test_mode = False
    settings.log_level = "INFO"
    settings.job_log_detail = "basic"
    settings.persist_job_logs = True
    settings.auth_tokens_raw = "admin-token:admin,coach-token:coach,analyst-token:analyst,parent-token:parent,system-token:system"
    settings.auth_bootstrap_key = None
    settings.jwt_secret = "test-jwt-secret-32-char-minimum-key"
    settings.jwt_issuer = "video-highlights"
    settings.jwt_audience = None
    settings.upload_max_gb = 0.0
    settings.upload_extended_max_gb = 0.0
    settings.upload_min_duration_seconds = 0.0
    settings.notify_backend = "console"
    settings.clear_api_token_cache()
    with TestClient(app) as test_client:
        yield test_client

    settings.auth_required = False
    settings.skip_user_management = False
    settings.test_mode = False
    settings.log_level = "INFO"
    settings.job_log_detail = "basic"
    settings.persist_job_logs = True
    settings.auth_tokens_raw = ""
    settings.auth_bootstrap_key = None
    settings.jwt_secret = None
    settings.clear_api_token_cache()
