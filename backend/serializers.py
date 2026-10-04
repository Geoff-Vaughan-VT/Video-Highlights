from __future__ import annotations

from .models import (
    Event,
    EventFeedback,
    JobLogEntry,
    Match,
    NotificationLog,
    ProcessingJob,
    RosterEntry,
    RosterTemplate,
    Tenant,
    TenantMembership,
    TrainingFeedbackBatch,
    TrainingRun,
    UserAccount,
)
from .schemas import (
    EventRead,
    FeedbackBatchRead,
    FeedbackRead,
    JobRead,
    JobLogRead,
    MatchRead,
    NotificationRead,
    RosterEntryRead,
    RosterTemplateEntry,
    RosterTemplateRead,
    TenantMembershipRead,
    TenantRead,
    TrainingRunRead,
    UserAccountRead,
)


def match_to_read(match: Match) -> MatchRead:
    return MatchRead(
        match_id=match.id,
        tenant_id=match.tenant_id,
        name=match.name,
        home_team_name=match.home_team_name,
        away_team_name=match.away_team_name,
        match_date=match.match_date,
        source_video_path=match.source_video_path,
        metadata=match.metadata_json or {},
        created_at=match.created_at,
        updated_at=match.updated_at,
    )


def job_to_read(job: ProcessingJob) -> JobRead:
    return JobRead(
        job_id=job.id,
        tenant_id=job.tenant_id,
        match_id=job.match_id,
        status=job.status,
        cancel_requested=job.cancel_requested,
        stage=job.stage,
        progress=job.progress,
        config=job.config_json or {},
        result=job.result_json or {},
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


def job_log_to_read(item: JobLogEntry) -> JobLogRead:
    return JobLogRead(
        log_id=item.id,
        job_id=item.job_id,
        tenant_id=item.tenant_id,
        level=item.level,
        detail_level=item.detail_level,
        stage=item.stage,
        message=item.message,
        data=item.data_json or {},
        created_at=item.created_at,
    )


def event_to_read(event: Event) -> EventRead:
    return EventRead(
        event_id=event.id,
        tenant_id=event.tenant_id,
        match_id=event.match_id,
        job_id=event.job_id,
        event_type=event.event_type,
        status=event.status,
        confidence=event.confidence,
        period=event.period,
        occurred_at_ms=event.occurred_at_ms,
        start_ms=event.start_ms,
        end_ms=event.end_ms,
        frame_index=event.frame_index,
        team_id=event.team_id,
        player_id=event.player_id,
        jersey_number=event.jersey_number,
        source=event.source_json or {},
        location=event.location_json or {},
        participants=event.participants_json or [],
        evidence=event.evidence_json or {},
        explanations=event.explanations_json or [],
        created_at=event.created_at,
        updated_at=event.updated_at,
    )


def roster_to_read(entry: RosterEntry) -> RosterEntryRead:
    return RosterEntryRead(
        roster_entry_id=entry.id,
        tenant_id=entry.tenant_id,
        match_id=entry.match_id,
        player_name=entry.player_name,
        jersey_number=entry.jersey_number,
        position=entry.position,
        email=entry.email,
        team_side=entry.team_side,
        metadata=entry.metadata_json or {},
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


def roster_template_to_read(template: RosterTemplate) -> RosterTemplateRead:
    entries = []
    for raw in list(template.entries_json or []):
        try:
            entries.append(RosterTemplateEntry(**raw))
        except Exception:
            continue
    return RosterTemplateRead(
        template_id=template.id,
        tenant_id=template.tenant_id,
        name=template.name,
        description=template.description,
        entries=entries,
        entry_count=len(entries),
        created_at=template.created_at,
        updated_at=template.updated_at,
    )


def notification_to_read(entry: NotificationLog) -> NotificationRead:
    return NotificationRead(
        notification_id=entry.id,
        tenant_id=entry.tenant_id,
        match_id=entry.match_id,
        job_id=entry.job_id,
        channel=entry.channel,
        backend=entry.backend,
        recipient=entry.recipient,
        subject=entry.subject,
        status=entry.status,
        error_message=entry.error_message,
        created_at=entry.created_at,
    )


def feedback_to_read(feedback: EventFeedback) -> FeedbackRead:
    return FeedbackRead(
        feedback_id=feedback.id,
        tenant_id=feedback.tenant_id,
        match_id=feedback.match_id,
        event_id=feedback.event_id,
        feedback_type=feedback.feedback_type,
        status=feedback.status,
        severity=feedback.severity,
        comment=feedback.comment,
        submitted_by={
            "user_id": feedback.submitted_by_user_id,
            "role": feedback.submitted_by_role,
        },
        correction=feedback.correction_json or {},
        evidence=feedback.evidence_json or [],
        review=feedback.review_json or {},
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
    )


def batch_to_read(batch: TrainingFeedbackBatch) -> FeedbackBatchRead:
    return FeedbackBatchRead(
        batch_id=batch.id,
        tenant_id=batch.tenant_id,
        item_count=batch.item_count,
        created_at=batch.created_at,
    )


def training_run_to_read(run: TrainingRun) -> TrainingRunRead:
    return TrainingRunRead(
        run_id=run.id,
        tenant_id=run.tenant_id,
        status=run.status,
        candidate_model_version=run.candidate_model_version,
        metrics=run.metrics_json or {},
        gates_passed=run.gates_passed,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def tenant_to_read(tenant: Tenant) -> TenantRead:
    return TenantRead(
        tenant_id=tenant.id,
        slug=tenant.slug,
        name=tenant.name,
        status=tenant.status,
        metadata=tenant.metadata_json or {},
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


def user_to_read(user: UserAccount) -> UserAccountRead:
    return UserAccountRead(
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        status=user.status,
        is_global_admin=user.is_global_admin,
        metadata=user.metadata_json or {},
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


def membership_to_read(membership: TenantMembership) -> TenantMembershipRead:
    return TenantMembershipRead(
        membership_id=membership.id,
        tenant_id=membership.tenant_id,
        user_id=membership.user_id,
        role=membership.role,
        status=membership.status,
        metadata=membership.metadata_json or {},
        created_at=membership.created_at,
        updated_at=membership.updated_at,
    )
