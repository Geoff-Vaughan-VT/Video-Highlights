from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..database import get_session
from ..models import EventFeedback, ModelPromotionDecision, ModelVersion, TrainingFeedbackBatch, TrainingRun
from ..schemas import (
    FeedbackBatchCreate,
    FeedbackBatchRead,
    ModelVersionRead,
    TrainingRunCreate,
    TrainingRunPromoteRequest,
    TrainingRunRead,
)
from ..serializers import batch_to_read, training_run_to_read
from ..services.job_runner import job_runner
from ..tenant import TenantContext, get_tenant_context
from ..utils import utcnow

router = APIRouter(prefix="/training", tags=["training"])


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ISO date: {value}") from exc
    # Stored timestamps are timezone-aware (UTC); a date-only or naive input
    # is interpreted as UTC so the comparison is valid on every backend.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@router.post("/feedback-batches", response_model=FeedbackBatchRead, status_code=201)
def create_feedback_batch(
    payload: FeedbackBatchCreate,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles("admin", "analyst", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> FeedbackBatchRead:
    from_date = _parse_iso(payload.from_date)
    to_date = _parse_iso(payload.to_date)

    stmt = (
        select(EventFeedback)
        .where(EventFeedback.status == payload.feedback_status)
        .where(EventFeedback.tenant_id == tenant.tenant_id)
    )
    if payload.match_ids:
        stmt = stmt.where(EventFeedback.match_id.in_(payload.match_ids))
    if payload.feedback_types:
        stmt = stmt.where(EventFeedback.feedback_type.in_(payload.feedback_types))
    if from_date:
        stmt = stmt.where(EventFeedback.created_at >= from_date)
    if to_date:
        stmt = stmt.where(EventFeedback.created_at <= to_date)

    items = list(session.exec(stmt))
    batch = TrainingFeedbackBatch(
        tenant_id=tenant.tenant_id,
        criteria_json=payload.model_dump(exclude_none=True),
        item_count=len(items),
        created_by_user_id=payload.created_by_user_id or user.user_id,
    )
    session.add(batch)
    session.commit()
    session.refresh(batch)
    return batch_to_read(batch)


@router.post("/runs", response_model=TrainingRunRead, status_code=202)
def create_training_run(
    payload: TrainingRunCreate,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> TrainingRunRead:
    training_config = dict(payload.training_config or {})
    training_kind = str(training_config.get("kind") or training_config.get("training_type") or "").strip().lower()
    batch = None
    if payload.batch_id:
        batch = session.get(TrainingFeedbackBatch, payload.batch_id)
        if not batch or batch.tenant_id != tenant.tenant_id:
            raise HTTPException(status_code=404, detail=f"Feedback batch not found: {payload.batch_id}")
    elif training_kind != "ultralytics_yolo":
        raise HTTPException(status_code=400, detail="batch_id is required unless training_config.kind is 'ultralytics_yolo'")

    notes_payload = {
        "notes": payload.notes,
        "training_config": training_config,
    }
    run = TrainingRun(
        tenant_id=tenant.tenant_id,
        batch_id=batch.id if batch else None,
        target_model=payload.target_model,
        notes=json.dumps(notes_payload, sort_keys=True),
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    job_runner.submit_training_run(run.id)
    return training_run_to_read(run)


@router.get("/runs/{run_id}", response_model=TrainingRunRead)
def get_training_run(
    run_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> TrainingRunRead:
    run = session.get(TrainingRun, run_id)
    if not run or run.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Training run not found: {run_id}")
    return training_run_to_read(run)


@router.post("/runs/{run_id}/promote", response_model=ModelVersionRead)
def promote_training_run(
    run_id: str,
    payload: TrainingRunPromoteRequest,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles("admin", "analyst", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> ModelVersionRead:
    run = session.get(TrainingRun, run_id)
    if not run or run.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Training run not found: {run_id}")
    if run.status != "completed":
        raise HTTPException(status_code=409, detail="Training run is not completed")
    if not run.candidate_model_version:
        raise HTTPException(status_code=409, detail="Training run has no candidate model version")

    decision = ModelPromotionDecision(
        tenant_id=tenant.tenant_id,
        run_id=run.id,
        target_model=run.target_model,
        candidate_model_version=run.candidate_model_version,
        decision=payload.decision,
        decided_by_user_id=user.user_id,
        reason=payload.reason,
        force=payload.force,
    )
    session.add(decision)

    if payload.decision == "rejected":
        session.commit()
        raise HTTPException(status_code=409, detail="Candidate model promotion rejected")

    if not run.gates_passed and not payload.force:
        session.commit()
        raise HTTPException(
            status_code=409,
            detail="Candidate model did not pass gates (set force=true to override)",
        )

    existing = session.exec(
        select(ModelVersion)
        .where(ModelVersion.tenant_id == tenant.tenant_id)
        .where(ModelVersion.run_id == run.id)
        .where(ModelVersion.promoted == True)  # noqa: E712
    ).first()
    if existing:
        return ModelVersionRead(
            model_id=existing.id,
            tenant_id=existing.tenant_id,
            target_model=existing.target_model,
            version=existing.version,
            run_id=existing.run_id,
            promoted=existing.promoted,
            promoted_by_user_id=existing.promoted_by_user_id,
            promoted_at=existing.promoted_at,
            metrics=existing.metrics_json or {},
            notes=existing.notes,
            created_at=existing.created_at,
        )

    model_version = ModelVersion(
        tenant_id=tenant.tenant_id,
        target_model=run.target_model,
        version=run.candidate_model_version,
        run_id=run.id,
        promoted=True,
        promoted_by_user_id=user.user_id,
        promoted_at=utcnow(),
        metrics_json=run.metrics_json or {},
        notes=payload.notes,
    )
    session.add(model_version)
    session.commit()
    session.refresh(model_version)

    return ModelVersionRead(
        model_id=model_version.id,
        tenant_id=model_version.tenant_id,
        target_model=model_version.target_model,
        version=model_version.version,
        run_id=model_version.run_id,
        promoted=model_version.promoted,
        promoted_by_user_id=model_version.promoted_by_user_id,
        promoted_at=model_version.promoted_at,
        metrics=model_version.metrics_json or {},
        notes=model_version.notes,
        created_at=model_version.created_at,
    )


@router.get("/models", response_model=list[ModelVersionRead])
def list_model_versions(
    target_model: str | None = None,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> list[ModelVersionRead]:
    stmt = select(ModelVersion).where(ModelVersion.tenant_id == tenant.tenant_id)
    if target_model:
        stmt = stmt.where(ModelVersion.target_model == target_model)
    stmt = stmt.order_by(ModelVersion.created_at.desc())
    rows = list(session.exec(stmt))
    return [
        ModelVersionRead(
            model_id=item.id,
            tenant_id=item.tenant_id,
            target_model=item.target_model,
            version=item.version,
            run_id=item.run_id,
            promoted=item.promoted,
            promoted_by_user_id=item.promoted_by_user_id,
            promoted_at=item.promoted_at,
            metrics=item.metrics_json or {},
            notes=item.notes,
            created_at=item.created_at,
        )
        for item in rows
    ]
