"""Job completion notifications (FR-NOTIFY-01).

Best-effort email notification when a processing job reaches a terminal
state. Recipients come from the job config (``notify_email``) or the match
metadata. Backends: ``console`` (default, logs only), ``smtp`` (real email),
``disabled``. Every attempt is recorded in ``notification_logs`` regardless
of backend so the UI can show delivery state. Never raises: a notification
failure must not fail the job.
"""

from __future__ import annotations

import logging
from email.message import EmailMessage
from typing import Optional

from sqlmodel import Session

from ..config import settings
from ..models import Match, NotificationLog, ProcessingJob

logger = logging.getLogger("videohighlights.notifications")

TERMINAL_STATUSES = {"completed", "failed"}


def _resolve_recipient(job: ProcessingJob, match: Optional[Match]) -> Optional[str]:
    config_email = str((job.config_json or {}).get("notify_email") or "").strip()
    if config_email:
        return config_email
    if match is not None:
        match_email = str((match.metadata_json or {}).get("notify_email") or "").strip()
        if match_email:
            return match_email
    return None


def _build_message(job: ProcessingJob, match: Optional[Match]) -> tuple[str, str]:
    match_name = (match.name if match else None) or job.match_id
    if job.status == "completed":
        subject = f"Your match analysis is ready: {match_name}"
        body = (
            f"Good news! The analysis for '{match_name}' finished successfully.\n\n"
            f"Job: {job.id}\n"
            f"Open the match in the Studio portal to review the highlights, "
            f"stats, and share links."
        )
    else:
        subject = f"Match analysis {job.status}: {match_name}"
        body = (
            f"The analysis for '{match_name}' ended with status '{job.status}'.\n\n"
            f"Job: {job.id}\n"
            f"Error: {job.error_message or 'unknown'}\n\n"
            f"You can retry the run from the Studio portal."
        )
    return subject, body


def _send_smtp(recipient: str, subject: str, body: str) -> None:
    import smtplib

    if not settings.smtp_host:
        raise RuntimeError("VH_SMTP_HOST is not configured")
    message = EmailMessage()
    message["From"] = settings.smtp_from
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as client:
        if settings.smtp_starttls:
            client.starttls()
        if settings.smtp_username and settings.smtp_password:
            client.login(settings.smtp_username, settings.smtp_password)
        client.send_message(message)


def send_notification(
    session: Session,
    recipient: Optional[str],
    subject: str,
    body: str,
    tenant_id: Optional[str] = None,
    match_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> NotificationLog:
    """Deliver (or record) a one-off email and log the attempt.

    Always returns a log row — callers use its ``status`` to report what
    happened. Never raises: a delivery failure must not fail the caller.
    """
    entry = NotificationLog(
        tenant_id=tenant_id,
        match_id=match_id,
        job_id=job_id,
        channel="email",
        backend=settings.notify_backend,
        recipient=recipient,
        subject=subject,
        body=body,
    )
    if settings.notify_backend == "disabled":
        entry.status = "skipped"
        entry.error_message = "Notifications are disabled"
    elif not recipient:
        entry.status = "skipped"
        entry.error_message = "No recipient email address"
    elif settings.notify_backend == "console":
        entry.status = "sent"
        logger.info("[notify] to=%s subject=%s", recipient, subject)
    elif settings.notify_backend == "smtp":
        try:
            _send_smtp(recipient, subject, body)
            entry.status = "sent"
        except Exception as exc:
            entry.status = "failed"
            entry.error_message = str(exc)
            logger.warning("[notify] smtp delivery failed to=%s: %s", recipient, exc)
    else:
        entry.status = "failed"
        entry.error_message = f"Unknown notify backend: {settings.notify_backend}"
    session.add(entry)
    return entry


def notify_job_terminal_state(session: Session, job: ProcessingJob) -> Optional[NotificationLog]:
    """Record (and, where configured, deliver) a terminal-state notification."""
    try:
        if settings.notify_backend == "disabled" or job.status not in TERMINAL_STATUSES:
            return None
        match = session.get(Match, job.match_id)
        recipient = _resolve_recipient(job, match)
        subject, body = _build_message(job, match)
        entry = NotificationLog(
            tenant_id=job.tenant_id,
            match_id=job.match_id,
            job_id=job.id,
            channel="email",
            backend=settings.notify_backend,
            recipient=recipient,
            subject=subject,
            body=body,
        )
        if not recipient:
            entry.status = "skipped"
            entry.error_message = "No notify_email configured on the job or match"
        elif settings.notify_backend == "console":
            entry.status = "sent"
            logger.info("[notify] to=%s subject=%s", recipient, subject)
        elif settings.notify_backend == "smtp":
            try:
                _send_smtp(recipient, subject, body)
                entry.status = "sent"
            except Exception as exc:
                entry.status = "failed"
                entry.error_message = str(exc)
                logger.warning("[notify] smtp delivery failed to=%s: %s", recipient, exc)
        else:
            entry.status = "failed"
            entry.error_message = f"Unknown notify backend: {settings.notify_backend}"
        session.add(entry)
        return entry
    except Exception:
        logger.exception("Notification handling failed for job %s", job.id)
        return None
