from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class CursorPage(BaseModel):
    items: List[Any]
    next_cursor: Optional[str] = None


EventType = Literal[
    "goal",
    "shot",
    "corner_kick",
    "penalty_kick",
    "free_kick",
    "goal_kick",
    "kickoff",
    "foul",
    "save",
]

EventStatus = Literal["auto_detected", "confirmed", "corrected", "rejected"]
PeriodType = Literal["1H", "2H", "ET1", "ET2", "PK"]
FeedbackType = Literal[
    "false_positive",
    "missed_event",
    "wrong_timestamp",
    "wrong_event_type",
    "wrong_player",
    "wrong_team",
    "duplicate_event",
    "confidence_miscalibrated",
]
FeedbackStatus = Literal["pending_review", "approved", "rejected", "needs_more_info", "merged"]
SeverityType = Literal["low", "medium", "high", "critical"]
ReviewerRole = Literal["coach", "analyst", "admin", "tenant_admin", "parent", "system"]
TenantStatus = Literal["active", "suspended", "archived"]
UserStatus = Literal["active", "disabled", "invited"]
MembershipRole = Literal["tenant_admin", "coach", "analyst", "parent", "player", "system"]
MembershipStatus = Literal["active", "invited", "disabled"]


class MatchCreate(BaseModel):
    tenant_id: Optional[str] = None
    name: Optional[str] = None
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None
    match_date: Optional[str] = None
    source_video_path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MatchPatch(BaseModel):
    tenant_id: Optional[str] = None
    name: Optional[str] = None
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None
    match_date: Optional[str] = None
    source_video_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MatchLocalAssetRegister(BaseModel):
    path: str
    set_as_source: bool = True


class MatchRead(BaseModel):
    match_id: str
    tenant_id: Optional[str] = None
    name: Optional[str] = None
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None
    match_date: Optional[str] = None
    source_video_path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class JobCreate(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)


class JobRerunRequest(BaseModel):
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None


class JobRead(BaseModel):
    job_id: str
    tenant_id: Optional[str] = None
    match_id: str
    status: str
    cancel_requested: bool = False
    stage: Optional[str] = None
    progress: float = 0.0
    config: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobLogRead(BaseModel):
    log_id: str
    job_id: str
    tenant_id: Optional[str] = None
    level: str
    detail_level: str
    stage: Optional[str] = None
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class Participant(BaseModel):
    team_id: Optional[str] = None
    player_id: Optional[str] = None
    jersey_number: Optional[str] = None
    role: Optional[str] = None


class SignalExplanation(BaseModel):
    signal: str
    value: float


class EventSource(BaseModel):
    detector: Optional[str] = None
    detector_version: Optional[str] = None
    tracker_version: Optional[str] = None
    follow_cam_version: Optional[str] = None
    camera_mode: Optional[str] = None
    zoom_factor: Optional[float] = None


class EventLocation(BaseModel):
    x_norm: Optional[float] = None
    y_norm: Optional[float] = None
    zone: Optional[str] = None


class EventEvidence(BaseModel):
    source_asset_id: Optional[str] = None
    follow_cam_asset_id: Optional[str] = None
    evidence_clip_asset_id: Optional[str] = None
    thumbnail_asset_id: Optional[str] = None
    analysis_manifest_path: Optional[str] = None
    tracking_manifest_path: Optional[str] = None
    bookmark_id: Optional[str] = None


class EventUpsert(BaseModel):
    event_type: EventType
    status: EventStatus = "auto_detected"
    confidence: float = 0.0
    period: Optional[PeriodType] = None
    occurred_at_ms: int = 0
    start_ms: int = 0
    end_ms: int = 0
    frame_index: int = 0
    team_id: Optional[str] = None
    player_id: Optional[str] = None
    jersey_number: Optional[str] = None
    source: EventSource = Field(default_factory=EventSource)
    location: EventLocation = Field(default_factory=EventLocation)
    participants: List[Participant] = Field(default_factory=list)
    evidence: EventEvidence = Field(default_factory=EventEvidence)
    explanations: List[SignalExplanation] = Field(default_factory=list)
    job_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_times(self) -> "EventUpsert":
        if self.start_ms > self.occurred_at_ms or self.occurred_at_ms > self.end_ms:
            raise ValueError("Must satisfy start_ms <= occurred_at_ms <= end_ms")
        return self


class EventPatch(BaseModel):
    event_type: Optional[EventType] = None
    status: Optional[EventStatus] = None
    confidence: Optional[float] = None
    period: Optional[PeriodType] = None
    occurred_at_ms: Optional[int] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    frame_index: Optional[int] = None
    team_id: Optional[str] = None
    player_id: Optional[str] = None
    jersey_number: Optional[str] = None
    source: Optional[EventSource] = None
    location: Optional[EventLocation] = None
    participants: Optional[List[Participant]] = None
    evidence: Optional[EventEvidence] = None
    explanations: Optional[List[SignalExplanation]] = None


class EventRead(BaseModel):
    event_id: str
    tenant_id: Optional[str] = None
    match_id: str
    job_id: Optional[str] = None
    event_type: str
    status: str
    confidence: float
    period: Optional[str] = None
    occurred_at_ms: int
    start_ms: int
    end_ms: int
    frame_index: int
    team_id: Optional[str] = None
    player_id: Optional[str] = None
    jersey_number: Optional[str] = None
    source: Dict[str, Any] = Field(default_factory=dict)
    location: Dict[str, Any] = Field(default_factory=dict)
    participants: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    explanations: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class EventClipRequest(BaseModel):
    pre_seconds: float = Field(default=2.0, ge=0.0, le=120.0)
    post_seconds: float = Field(default=8.0, ge=0.0, le=300.0)
    anchor: Literal["occurred_at", "event_window"] = "event_window"
    include_audio: bool = True
    prefer_gpu: bool = True
    force_rebuild: bool = False
    expires_seconds: int = Field(default=3600, ge=60, le=86400)


class EventClipRead(BaseModel):
    clip_id: str
    match_id: str
    event_id: str
    asset_id: str
    path: str
    download_url: str
    start_ms: int
    end_ms: int
    duration_ms: int
    include_audio: bool = True
    anchor: str
    reused_existing: bool = False


class HighlightExportRequest(BaseModel):
    event_ids: List[str] = Field(default_factory=list, min_length=1)
    pre_seconds: float = Field(default=1.5, ge=0.0, le=120.0)
    post_seconds: float = Field(default=5.0, ge=0.0, le=300.0)
    anchor: Literal["occurred_at", "event_window"] = "event_window"
    include_audio: bool = True
    prefer_gpu: bool = True
    title: Optional[str] = None
    expires_seconds: int = Field(default=3600, ge=60, le=86400)


class HighlightExportRead(BaseModel):
    export_id: str
    match_id: str
    event_ids: List[str] = Field(default_factory=list)
    clip_count: int = 0
    asset_id: str
    path: str
    download_url: str
    duration_ms: int = 0
    created_at: str


class AudioEditRead(BaseModel):
    audio_edit_id: str
    match_id: str
    asset_id: str
    path: str
    download_url: str
    mode: str
    cleanup_profile: str = "none"
    size_bytes: int = 0
    created_at: str


RosterTeamSide = Literal["home", "away"]


class RosterEntryCreate(BaseModel):
    player_name: str
    jersey_number: str
    position: Optional[str] = None
    email: Optional[str] = None
    team_side: RosterTeamSide = "home"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RosterEntryPatch(BaseModel):
    player_name: Optional[str] = None
    jersey_number: Optional[str] = None
    position: Optional[str] = None
    email: Optional[str] = None
    team_side: Optional[RosterTeamSide] = None
    metadata: Optional[Dict[str, Any]] = None


class RosterEntryRead(BaseModel):
    roster_entry_id: str
    tenant_id: Optional[str] = None
    match_id: str
    player_name: str
    jersey_number: str
    position: Optional[str] = None
    email: Optional[str] = None
    team_side: str = "home"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class RosterImportRequest(BaseModel):
    csv_text: str
    team_side: RosterTeamSide = "home"
    replace_existing: bool = False


class RosterImportError(BaseModel):
    line: int
    issue: str


class RosterImportResult(BaseModel):
    created: int = 0
    updated: int = 0
    skipped: int = 0
    errors: List[RosterImportError] = Field(default_factory=list)
    entries: List[RosterEntryRead] = Field(default_factory=list)


class EventAssignRequest(BaseModel):
    roster_entry_id: Optional[str] = None  # null clears the assignment


class StatValue(BaseModel):
    key: str
    label: str
    unit: Literal["count", "percent"] = "count"
    available: bool = False
    reason: Optional[str] = None
    method: Optional[str] = None
    home: Optional[float] = None
    away: Optional[float] = None
    unattributed: Optional[float] = None
    total: Optional[float] = None
    raw: Dict[str, Any] = Field(default_factory=dict)
    event_ids: List[str] = Field(default_factory=list)


class MatchStatsRead(BaseModel):
    match_id: str
    job_id: Optional[str] = None
    teams: Dict[str, Optional[str]] = Field(default_factory=dict)
    generated_at: str
    analysis: Dict[str, Any] = Field(default_factory=dict)
    stats: List[StatValue] = Field(default_factory=list)


ShareScope = Literal["match", "highlight", "player_card"]


class ShareLinkCreate(BaseModel):
    scope: ShareScope = "match"
    event_id: Optional[str] = None
    roster_entry_id: Optional[str] = None
    label: Optional[str] = None
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)


class ShareLinkRead(BaseModel):
    share_id: str
    token: str
    url_path: str
    tenant_id: Optional[str] = None
    match_id: str
    scope: str
    event_id: Optional[str] = None
    roster_entry_id: Optional[str] = None
    label: Optional[str] = None
    revoked: bool = False
    expires_at: Optional[datetime] = None
    view_count: int = 0
    created_at: datetime


class RosterTemplateEntry(BaseModel):
    player_name: str
    jersey_number: str
    position: Optional[str] = None
    email: Optional[str] = None
    team_side: RosterTeamSide = "home"


class RosterTemplateCreate(BaseModel):
    name: str
    description: Optional[str] = None
    entries: List[RosterTemplateEntry] = Field(default_factory=list)


class RosterTemplateFromMatch(BaseModel):
    name: str
    description: Optional[str] = None
    team_side: Optional[RosterTeamSide] = None


class RosterTemplateApply(BaseModel):
    team_side: Optional[RosterTeamSide] = None
    replace_existing: bool = False


class RosterTemplateRead(BaseModel):
    template_id: str
    tenant_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    entries: List[RosterTemplateEntry] = Field(default_factory=list)
    entry_count: int = 0
    created_at: datetime
    updated_at: datetime


class PlayerCardStat(BaseModel):
    key: str
    label: str
    count: int = 0


class PlayerCardRead(BaseModel):
    match_id: str
    roster_entry_id: str
    player_name: str
    jersey_number: str
    position: Optional[str] = None
    team_side: str = "home"
    team_name: Optional[str] = None
    match_name: Optional[str] = None
    match_date: Optional[str] = None
    highlight_count: int = 0
    stats: List[PlayerCardStat] = Field(default_factory=list)
    highlights: List[Dict[str, Any]] = Field(default_factory=list)
    share_url_path: Optional[str] = None


class RoutingResult(BaseModel):
    match_id: str
    routed: int = 0
    already_routed: int = 0
    unmatched_jersey_numbers: List[str] = Field(default_factory=list)
    unassigned_remaining: int = 0
    roster_size: int = 0


class PlayerCardSendResult(BaseModel):
    match_id: str
    sent: int = 0
    skipped: int = 0
    details: List[Dict[str, Any]] = Field(default_factory=list)


class NotificationRead(BaseModel):
    notification_id: str
    tenant_id: Optional[str] = None
    match_id: Optional[str] = None
    job_id: Optional[str] = None
    channel: str
    backend: str
    recipient: Optional[str] = None
    subject: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime


class UploadPolicyRead(BaseModel):
    max_upload_bytes: int
    max_upload_gb: float
    extended_max_upload_bytes: int
    extended_max_upload_gb: float
    extended_upload_enabled: bool = False
    min_duration_seconds: float = 0.0
    allowed_extensions: List[str] = Field(default_factory=list)
    processing_sla_hours: List[int] = Field(default_factory=list)


class FeedbackSubmittedBy(BaseModel):
    user_id: Optional[str] = None
    role: Optional[ReviewerRole] = None


class FeedbackEvidenceItem(BaseModel):
    asset_id: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    note: Optional[str] = None


class FeedbackCorrection(BaseModel):
    expected_event_type: Optional[EventType] = None
    corrected_occurred_at_ms: Optional[int] = None
    corrected_start_ms: Optional[int] = None
    corrected_end_ms: Optional[int] = None
    corrected_team_id: Optional[str] = None
    corrected_player_id: Optional[str] = None
    corrected_jersey_number: Optional[str] = None


class FeedbackCreate(BaseModel):
    feedback_type: FeedbackType
    status: FeedbackStatus = "pending_review"
    severity: SeverityType = "medium"
    comment: Optional[str] = None
    submitted_by: FeedbackSubmittedBy = Field(default_factory=FeedbackSubmittedBy)
    correction: FeedbackCorrection = Field(default_factory=FeedbackCorrection)
    evidence: List[FeedbackEvidenceItem] = Field(default_factory=list)


class FeedbackReviewRequest(BaseModel):
    review_decision: FeedbackStatus
    review_note: Optional[str] = None


class FeedbackRead(BaseModel):
    feedback_id: str
    tenant_id: Optional[str] = None
    match_id: str
    event_id: Optional[str] = None
    feedback_type: str
    status: str
    severity: str
    comment: Optional[str] = None
    submitted_by: Dict[str, Any] = Field(default_factory=dict)
    correction: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    review: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class FeedbackBatchCreate(BaseModel):
    match_ids: List[str] = Field(default_factory=list)
    feedback_status: FeedbackStatus = "approved"
    feedback_types: List[FeedbackType] = Field(default_factory=list)
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    created_by_user_id: Optional[str] = None


class FeedbackBatchRead(BaseModel):
    batch_id: str
    tenant_id: Optional[str] = None
    item_count: int
    created_at: datetime


class TrainingRunCreate(BaseModel):
    batch_id: Optional[str] = None
    target_model: str = "event-v0"
    training_config: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


class TrainingRunRead(BaseModel):
    run_id: str
    tenant_id: Optional[str] = None
    status: str
    candidate_model_version: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    gates_passed: bool = False
    created_at: datetime
    updated_at: datetime


class TrainingRunPromoteRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    reason: Optional[str] = None
    notes: Optional[str] = None
    force: bool = False


class ModelVersionRead(BaseModel):
    model_id: str
    tenant_id: Optional[str] = None
    target_model: str
    version: str
    run_id: Optional[str] = None
    promoted: bool
    promoted_by_user_id: Optional[str] = None
    promoted_at: Optional[datetime] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None
    created_at: datetime


class AgentQueryRequest(BaseModel):
    query: str
    include_event_limit: int = 50


class AgentQueryResponse(BaseModel):
    provider: str
    model: Optional[str] = None
    answer: str
    referenced_event_ids: List[str] = Field(default_factory=list)


class AgentExplainRequest(BaseModel):
    question: Optional[str] = None


class AuthTokenIssueRequest(BaseModel):
    user_id: str
    role: ReviewerRole
    tenant_id: Optional[str] = None
    is_global_admin: bool = False
    expires_in_minutes: Optional[int] = Field(default=None, ge=1, le=10080)


class AuthTokenIssueResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: str
    issued_for_user_id: str
    issued_for_role: ReviewerRole
    issued_for_tenant_id: Optional[str] = None
    issued_for_is_global_admin: bool = False


class AuthMeResponse(BaseModel):
    user_id: str
    role: str
    tenant_id: Optional[str] = None
    tenant_role: Optional[str] = None
    is_global_admin: bool = False
    auth_source: str


class TenantCreate(BaseModel):
    slug: str
    name: str
    status: TenantStatus = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TenantPatch(BaseModel):
    slug: Optional[str] = None
    name: Optional[str] = None
    status: Optional[TenantStatus] = None
    metadata: Optional[Dict[str, Any]] = None


class TenantRead(BaseModel):
    tenant_id: str
    slug: str
    name: str
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class UserAccountCreate(BaseModel):
    user_id: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    status: UserStatus = "active"
    is_global_admin: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserAccountPatch(BaseModel):
    email: Optional[str] = None
    display_name: Optional[str] = None
    status: Optional[UserStatus] = None
    is_global_admin: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class UserAccountRead(BaseModel):
    user_id: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    status: str
    is_global_admin: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class TenantMembershipCreate(BaseModel):
    user_id: str
    role: MembershipRole
    status: MembershipStatus = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TenantMembershipPatch(BaseModel):
    role: Optional[MembershipRole] = None
    status: Optional[MembershipStatus] = None
    metadata: Optional[Dict[str, Any]] = None


class TenantMembershipRead(BaseModel):
    membership_id: str
    tenant_id: str
    user_id: str
    role: str
    status: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class GlobalAdminSummaryRead(BaseModel):
    tenant_count: int
    user_count: int
    membership_count: int
    match_count: int
    job_count: int
    event_count: int
    feedback_count: int
    training_run_count: int


class TenantAdminSummaryRead(BaseModel):
    tenant_id: str
    tenant_slug: str
    tenant_name: str
    user_count: int
    membership_count: int
    match_count: int
    job_count: int
    event_count: int
    feedback_count: int
    training_run_count: int


class TenantUserRead(BaseModel):
    membership_id: str
    tenant_id: str
    user_id: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    user_status: str
    role: str
    membership_status: str
    is_global_admin: bool = False
    created_at: datetime
    updated_at: datetime


class TenantAdminUserCreate(BaseModel):
    user_id: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    user_status: UserStatus = "active"
    role: MembershipRole = "coach"
    membership_status: MembershipStatus = "active"
    user_metadata: Dict[str, Any] = Field(default_factory=dict)
    membership_metadata: Dict[str, Any] = Field(default_factory=dict)


class TenantAdminUserPatch(BaseModel):
    email: Optional[str] = None
    display_name: Optional[str] = None
    user_status: Optional[UserStatus] = None
    role: Optional[MembershipRole] = None
    membership_status: Optional[MembershipStatus] = None
    user_metadata: Optional[Dict[str, Any]] = None
    membership_metadata: Optional[Dict[str, Any]] = None
