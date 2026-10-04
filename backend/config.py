from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict


class Settings:
    api_title: str = "Video Highlights API"
    api_version: str = "0.1.0"
    db_url: str = os.getenv("VH_DB_URL", "sqlite:///./video_highlights_v1.db")
    test_mode: bool = os.getenv("VH_TEST_MODE", "false").lower() in {"1", "true", "yes"}
    job_max_workers: int = int(os.getenv("VH_JOB_MAX_WORKERS", "2"))
    output_root: str = os.getenv("VH_OUTPUT_ROOT", "./outputs")
    llm_provider: str = os.getenv("VH_LLM_PROVIDER", "none").lower()
    llm_model: str = os.getenv("VH_LLM_MODEL", "gpt-4o-mini")
    llm_base_url: str | None = os.getenv("VH_LLM_BASE_URL")
    llm_api_key: str | None = os.getenv("VH_LLM_API_KEY")
    llm_timeout_seconds: float = float(os.getenv("VH_LLM_TIMEOUT_SECONDS", "20"))
    llm_keep_alive: str = os.getenv("VH_LLM_KEEP_ALIVE", "0")
    rnnoise_model_path: str | None = os.getenv("VH_RNNOISE_MODEL_PATH")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    job_execution_mode: str = os.getenv("VH_JOB_EXECUTION_MODE", "inline").lower()
    storage_backend: str = os.getenv("VH_STORAGE_BACKEND", "local").lower()
    local_storage_root: str = os.getenv("VH_LOCAL_STORAGE_ROOT", "./storage")
    auth_required: bool = os.getenv("VH_AUTH_REQUIRED", "false").lower() in {"1", "true", "yes"}
    auth_tokens_raw: str = os.getenv("VH_API_TOKENS", "")
    auth_bootstrap_key: str | None = os.getenv("VH_AUTH_BOOTSTRAP_KEY")
    jwt_secret: str | None = os.getenv("VH_JWT_SECRET")
    jwt_algorithm: str = os.getenv("VH_JWT_ALGORITHM", "HS256")
    jwt_issuer: str = os.getenv("VH_JWT_ISSUER", "video-highlights")
    jwt_audience: str | None = os.getenv("VH_JWT_AUDIENCE")
    jwt_default_exp_minutes: int = int(os.getenv("VH_JWT_DEFAULT_EXP_MINUTES", "120"))
    s3_endpoint_url: str | None = os.getenv("VH_S3_ENDPOINT_URL")
    s3_bucket: str | None = os.getenv("VH_S3_BUCKET")
    s3_region: str | None = os.getenv("VH_S3_REGION")
    s3_access_key_id: str | None = os.getenv("VH_S3_ACCESS_KEY_ID")
    s3_secret_access_key: str | None = os.getenv("VH_S3_SECRET_ACCESS_KEY")
    s3_key_prefix: str = os.getenv("VH_S3_KEY_PREFIX", "video-highlights")
    skip_user_management: bool = os.getenv("VH_SKIP_USER_MANAGEMENT", "false").lower() in {"1", "true", "yes"}
    base_tenant_slug: str = os.getenv("VH_BASE_TENANT_SLUG", "sandbox")
    base_tenant_name: str = os.getenv("VH_BASE_TENANT_NAME", "Sandbox Tenant")
    log_level: str = os.getenv("VH_LOG_LEVEL", "DEBUG" if test_mode else "INFO").upper()
    job_log_detail: str = os.getenv("VH_JOB_LOG_DETAIL", "extreme" if test_mode else "basic").lower()
    persist_job_logs: bool = os.getenv("VH_PERSIST_JOB_LOGS", "true").lower() in {"1", "true", "yes"}
    # Upload policy (FR-INGEST-05..07): standard cap, entitlement-gated extended cap,
    # and an optional minimum-duration gate (0 disables; set 1800 for the 30-minute product rule).
    upload_max_gb: float = float(os.getenv("VH_UPLOAD_MAX_GB", "3"))
    upload_extended_max_gb: float = float(os.getenv("VH_UPLOAD_EXTENDED_MAX_GB", "8"))
    upload_min_duration_seconds: float = float(os.getenv("VH_UPLOAD_MIN_DURATION_SECONDS", "0"))
    # Customer-facing processing turnaround target (NFR-SLA-01), surfaced in the UI.
    processing_sla_hours_min: int = int(os.getenv("VH_PROCESSING_SLA_HOURS_MIN", "4"))
    processing_sla_hours_max: int = int(os.getenv("VH_PROCESSING_SLA_HOURS_MAX", "6"))
    # Completion notifications (FR-NOTIFY-01): console logs by default, smtp for real email.
    notify_backend: str = os.getenv("VH_NOTIFY_BACKEND", "console").lower()  # console|smtp|disabled
    smtp_host: str | None = os.getenv("VH_SMTP_HOST")
    smtp_port: int = int(os.getenv("VH_SMTP_PORT", "587"))
    smtp_username: str | None = os.getenv("VH_SMTP_USERNAME")
    smtp_password: str | None = os.getenv("VH_SMTP_PASSWORD")
    smtp_from: str = os.getenv("VH_SMTP_FROM", "no-reply@video-highlights.local")
    smtp_starttls: bool = os.getenv("VH_SMTP_STARTTLS", "true").lower() in {"1", "true", "yes"}

    @property
    @lru_cache(maxsize=1)
    def api_tokens(self) -> Dict[str, str]:
        """
        Parse VH_API_TOKENS env in format:
            token1:admin,token2:coach,token3:analyst
        Returns {token: role}.
        """
        mapping: Dict[str, str] = {}
        if not self.auth_tokens_raw:
            return mapping
        chunks = [chunk.strip() for chunk in self.auth_tokens_raw.split(",") if chunk.strip()]
        for chunk in chunks:
            if ":" not in chunk:
                continue
            token, role = chunk.split(":", 1)
            token = token.strip()
            role = role.strip().lower()
            if token and role:
                mapping[token] = role
        return mapping

    def clear_api_token_cache(self) -> None:
        type(self).api_tokens.fget.cache_clear()


settings = Settings()
