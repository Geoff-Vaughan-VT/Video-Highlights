from __future__ import annotations

from datetime import datetime
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    "yellow_card",
    "red_card",
    # v2 event engine types (analysis_events.json)
    "chance",
    "sprint",
    "dribble",
    "turnover",
    "foul_candidate",
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


CameraMode = Literal["wide", "follow_ball", "follow_player", "follow_action"]
CameraStyle = Literal["broadcast", "tight", "wide"]
ReelPreset = Literal["1min", "3min", "5min", "10min"]

_JOB_LOGGER = logging.getLogger("videohighlights.job_config")
_HEX_COLOR = re.compile(r"^#?[0-9a-fA-F]{6}$")
_DEVICE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")


_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_URL_LIKE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def output_dir_override_allowed() -> bool:
    """``VH_ALLOW_OUTPUT_DIR_OVERRIDE=1`` lets a job config choose ``output_dir``.

    Off by default: API jobs always write to ``<output_root>/<job_id>``. The
    override exists for test suites that seed run folders; it is read on
    every call so tests can toggle it.
    """
    return os.getenv("VH_ALLOW_OUTPUT_DIR_OVERRIDE", "").strip().lower() in {"1", "true", "yes", "on"}


def is_valid_job_id(value: object) -> bool:
    """Job ids are ``job_<hex>``; accept ``[A-Za-z0-9_-]{1,64}`` (no dots, no separators)."""
    return isinstance(value, str) and bool(_JOB_ID.match(value))


def media_roots() -> List[Path]:
    """Allowed media roots from ``VH_MEDIA_ROOTS`` (comma/os.pathsep separated)."""
    raw = os.getenv("VH_MEDIA_ROOTS", "").strip()
    if not raw:
        return []
    parts = [p.strip() for chunk in raw.split(",") for p in chunk.split(os.pathsep) if p.strip()]
    return [Path(p).expanduser().resolve() for p in parts]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def check_media_path(value: str, *, field: str = "video_path") -> Path:
    """Resolve a source-video path and enforce the media-root policy.

    * Never inside ``output_root`` (run folders hold other jobs' proxies and
      movies; reading them as a "source" would leak them across tenants).
    * With ``VH_MEDIA_ROOTS`` set: must be inside one of those roots or the
      local upload storage root.
    * Without it: any other path is accepted (local single-user installs).

    Raises ``ValueError`` with a user-facing message.
    """
    from .config import settings

    text = str(value or "")
    if not text.strip() or "\x00" in text:
        raise ValueError(f"{field} is empty or invalid")
    resolved = Path(text.strip()).expanduser().resolve()
    storage_root = Path(settings.local_storage_root).expanduser().resolve()
    output_root = Path(settings.output_root).expanduser().resolve()
    if _is_within(resolved, storage_root):
        return resolved
    if _is_within(resolved, output_root):
        raise ValueError(f"{field} must not point inside the output root ({output_root})")
    roots = media_roots()
    if roots and not any(_is_within(resolved, root) for root in roots):
        raise ValueError(f"{field} must be under a VH_MEDIA_ROOTS directory or the upload storage root")
    return resolved


def is_url_like(value: object) -> bool:
    """True for ``scheme://...`` sources (links), which are not filesystem paths."""
    return isinstance(value, str) and bool(_URL_LIKE.match(value.strip()))


def _check_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(str(value))
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number") from None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_player_roi(value: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """``{x1_norm, y1_norm, x2_norm, y2_norm}`` (0..1) or ``{x, y, w, h}``
    (pixels, or 0..1 with ``normalized: true``), plus optional ``t`` /
    ``time_s`` / ``window_s`` seconds."""
    if not value:
        return None
    roi = dict(value)
    norm_keys = ("x1_norm", "y1_norm", "x2_norm", "y2_norm")
    box_keys = ("x", "y", "w", "h")
    if all(k in roi for k in norm_keys):
        x1, y1, x2, y2 = (_check_number(roi[k], f"player_roi.{k}") for k in norm_keys)
        if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
            raise ValueError("player_roi *_norm values must be between 0 and 1")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("player_roi needs x2_norm > x1_norm and y2_norm > y1_norm")
        roi.update({"x1_norm": x1, "y1_norm": y1, "x2_norm": x2, "y2_norm": y2})
    elif all(k in roi for k in box_keys):
        x, y, w, h = (_check_number(roi[k], f"player_roi.{k}") for k in box_keys)
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError("player_roi needs x, y >= 0 and w, h > 0")
        if roi.get("normalized") and (x + w > 1.0 + 1e-6 or y + h > 1.0 + 1e-6):
            raise ValueError("normalized player_roi must fit inside 0..1")
        roi.update({"x": x, "y": y, "w": w, "h": h})
    else:
        raise ValueError("player_roi must have x1_norm/y1_norm/x2_norm/y2_norm or x/y/w/h")
    for key in ("t", "time_s", "window_s"):
        if roi.get(key) is not None:
            number = _check_number(roi[key], f"player_roi.{key}")
            if number < 0:
                raise ValueError(f"player_roi.{key} must be >= 0")
            roi[key] = number
    if "normalized" in roi and not isinstance(roi["normalized"], bool):
        raise ValueError("player_roi.normalized must be true or false")
    return roi


def _validate_goal_box(value: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """``{x1, y1, x2, y2}`` in source pixels (or all <= 1.0: normalized)."""
    if not value:
        return None
    box = dict(value)
    try:
        x1, y1, x2, y2 = (_check_number(box[k], f"{name}.{k}") for k in ("x1", "y1", "x2", "y2"))
    except KeyError:
        raise ValueError(f"{name} must have x1, y1, x2, y2") from None
    if min(x1, y1, x2, y2) < 0:
        raise ValueError(f"{name} coordinates must be >= 0")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} needs x2 > x1 and y2 > y1")
    box.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return box


def _parse_seconds(value: Union[float, int, str, None]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"invalid time {value!r}; use seconds, MM:SS or HH:MM:SS")
    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"invalid time {value!r}; use seconds, MM:SS or HH:MM:SS") from exc
    total = 0.0
    for n in nums:
        total = total * 60.0 + n
    return total


class JobConfig(BaseModel):
    """Typed processing-job configuration (unknown legacy keys are kept).

    Profile keys (``proxy_height``, ``inference_imgsz``, ``vid_stride``,
    ``yolo_model``, ``batch_size``, ``output_height``, ``debug_video``,
    ``ball_tiles``, ``tracker_config``) left unset come from ``profile`` via
    :func:`resolve_job_config`.
    """

    model_config = ConfigDict(extra="allow")

    profile: Optional[str] = None
    camera_mode: Optional[CameraMode] = None
    camera_style: Optional[CameraStyle] = None
    zoom_factor: Optional[float] = Field(default=None, ge=1.0, le=4.0)
    render_full_follow_cam: Optional[bool] = None
    output_height: Optional[int] = Field(default=None, ge=240, le=2160)
    proxy_height: Optional[int] = Field(default=None, ge=240, le=2160)
    inference_imgsz: Optional[int] = Field(default=None, ge=160, le=4096)
    vid_stride: Optional[int] = Field(default=None, ge=1, le=30)
    yolo_model: Optional[str] = None
    tracker_config: Optional[str] = None
    detection_conf: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    batch_size: Optional[Union[int, str]] = None
    use_tensorrt: Optional[bool] = None
    ball_tiles: Optional[bool] = None
    device: Optional[str] = None
    focus_track_id: Optional[int] = Field(default=None, ge=0)
    player_roi: Optional[Dict[str, Any]] = None
    goal_box_left: Optional[Dict[str, Any]] = None
    goal_box_right: Optional[Dict[str, Any]] = None
    reuse_tracking_from_job: Optional[str] = None
    pitch_corners: Optional[List[List[float]]] = None
    reel_minutes: Optional[float] = Field(default=None, gt=0.0, le=120.0)
    reel_preset: Optional[ReelPreset] = None
    player_spotlight_reel: Optional[bool] = None
    team_left: Optional[str] = Field(default=None, max_length=80)
    team_right: Optional[str] = Field(default=None, max_length=80)
    team_left_color: Optional[str] = None
    team_right_color: Optional[str] = None
    auto_detect_team_colors: Optional[bool] = None
    detect_cards: Optional[bool] = None
    broadcast_reel: Optional[bool] = None
    scorebug: Optional[bool] = None
    debug_video: Optional[bool] = None
    dump_training_data: Optional[bool] = None
    llm_report: Optional[bool] = None
    trim_start: Optional[Union[float, str]] = None
    trim_end: Optional[Union[float, str]] = None
    analysis_only: Optional[bool] = None
    pre_seconds: Optional[float] = Field(default=None, ge=0.0, le=120.0)
    post_seconds: Optional[float] = Field(default=None, ge=0.0, le=120.0)
    min_clip_duration: Optional[float] = Field(default=None, ge=0.0, le=600.0)
    no_audio: Optional[bool] = None
    overlay: Optional[bool] = None
    threads: Optional[int] = Field(default=None, ge=1, le=64)
    require_gpu: Optional[bool] = None
    select_player: Optional[bool] = None
    video_path: Optional[str] = None
    output_dir: Optional[str] = None

    @field_validator("profile")
    @classmethod
    def _check_profile(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        from .services.perf_profiles import PROFILES

        key = str(value).strip().lower()
        if key not in PROFILES:
            raise ValueError(f"unknown profile {value!r}; expected one of {', '.join(PROFILES)}")
        return key

    @field_validator("batch_size")
    @classmethod
    def _check_batch(cls, value: Optional[Union[int, str]]) -> Optional[Union[int, str]]:
        if value is None:
            return None
        if isinstance(value, str):
            if value.strip().lower() == "auto":
                return "auto"
            if not value.strip().isdigit():
                raise ValueError("batch_size must be 'auto' or a positive integer")
            value = int(value.strip())
        if int(value) < 1 or int(value) > 256:
            raise ValueError("batch_size must be between 1 and 256")
        return int(value)

    @field_validator("device")
    @classmethod
    def _check_device(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        if not _DEVICE.match(str(value).strip().lower()):
            raise ValueError("device must be auto, cpu, mps, cuda or cuda:N")
        return str(value).strip().lower()

    @field_validator("team_left_color", "team_right_color")
    @classmethod
    def _check_color(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        text = str(value).strip()
        if not _HEX_COLOR.match(text):
            raise ValueError("colors must be hex like #d32f2f")
        return text if text.startswith("#") else f"#{text}"

    @field_validator("pitch_corners")
    @classmethod
    def _check_corners(cls, value: Optional[List[List[float]]]) -> Optional[List[List[float]]]:
        if value is None:
            return None
        if len(value) != 4 or any(len(p) != 2 for p in value):
            raise ValueError("pitch_corners must be 4 [x, y] points (TL, TR, BR, BL)")
        points = [[_check_number(p[0], "pitch_corners"), _check_number(p[1], "pitch_corners")] for p in value]
        if any(v < 0 for p in points for v in p):
            raise ValueError("pitch_corners coordinates must be >= 0")
        if len({(round(p[0], 3), round(p[1], 3)) for p in points}) != 4:
            raise ValueError("pitch_corners must be 4 distinct points")
        return points

    @field_validator("player_roi")
    @classmethod
    def _check_roi(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        return _validate_player_roi(value)

    @field_validator("goal_box_left", "goal_box_right")
    @classmethod
    def _check_goal_box(cls, value: Optional[Dict[str, Any]], info) -> Optional[Dict[str, Any]]:  # noqa: ANN001
        if value is None:
            return None
        return _validate_goal_box(value, info.field_name)

    @field_validator("reuse_tracking_from_job")
    @classmethod
    def _check_reuse(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        text = str(value).strip()
        if not is_valid_job_id(text):
            raise ValueError("reuse_tracking_from_job must be a job id (letters, digits, '_' or '-')")
        return text

    @field_validator("yolo_model")
    @classmethod
    def _check_model(cls, value: Optional[str]) -> Optional[str]:
        """Stock/bare weight names, or a file inside the model directory
        (``VH_MODEL_DIR``); arbitrary filesystem paths are rejected."""
        if value is None or not str(value).strip():
            return None
        text = str(value).strip()
        if "\x00" in text:
            raise ValueError("invalid model name")
        if _MODEL_NAME.match(text) and ".." not in text:
            return text
        from .services.perf_profiles import model_dir

        root = model_dir().expanduser().resolve()
        candidate = Path(text).expanduser()
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if not _is_within(resolved, root):
            raise ValueError("yolo_model must be a stock model name or a file inside VH_MODEL_DIR")
        return str(resolved)

    @field_validator("tracker_config")
    @classmethod
    def _check_tracker(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        text = str(value).strip()
        if not _MODEL_NAME.match(text) or ".." in text:
            raise ValueError("tracker_config must be a tracker name like bytetrack.yaml or botsort.yaml")
        return text

    @field_validator("select_player")
    @classmethod
    def _check_select(cls, value: Optional[bool]) -> Optional[bool]:
        if value:
            raise ValueError(
                "select_player opens an interactive window and cannot run on the API worker; "
                "use player_roi or focus_track_id instead"
            )
        return value

    @field_validator("video_path")
    @classmethod
    def _check_video_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        resolved = check_media_path(str(value), field="video_path")
        if media_roots() and not resolved.is_file():
            raise ValueError(f"video_path does not exist: {resolved}")
        if not media_roots():
            from .config import settings

            storage = Path(settings.local_storage_root).expanduser().resolve()
            if not _is_within(resolved, storage):
                _JOB_LOGGER.warning(
                    "video_path %s is outside the storage root; set VH_MEDIA_ROOTS to restrict job paths", resolved,
                )
        return str(resolved)

    @field_validator("output_dir")
    @classmethod
    def _check_output_dir(cls, value: Optional[str]) -> Optional[str]:
        """API jobs always write to ``<output_root>/<job_id>``; ``output_dir``
        is only accepted with ``VH_ALLOW_OUTPUT_DIR_OVERRIDE=1`` (tests)."""
        if value is None or not str(value).strip():
            return None
        if not output_dir_override_allowed():
            raise ValueError("output_dir is not accepted; runs are always written to <output_root>/<job_id>")
        from .config import settings

        if "\x00" in str(value):
            raise ValueError("invalid path")
        resolved = Path(str(value)).expanduser().resolve()
        allowed = [Path(settings.output_root).expanduser().resolve(),
                   Path(settings.local_storage_root).expanduser().resolve()] + media_roots()
        if media_roots() and not any(_is_within(resolved, root) for root in allowed):
            raise ValueError("output_dir must be under the output root or a VH_MEDIA_ROOTS directory")
        return str(resolved)

    @model_validator(mode="after")
    def _check_window(self) -> "JobConfig":
        start = _parse_seconds(self.trim_start)
        end = _parse_seconds(self.trim_end)
        if start is not None and start < 0:
            raise ValueError("trim_start must be >= 0")
        if end is not None and end <= 0:
            raise ValueError("trim_end must be > 0")
        if start is not None and end is not None and end <= start:
            raise ValueError("trim_end must be greater than trim_start")
        return self


def validate_job_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate ``raw`` as :class:`JobConfig` and lay the profile defaults under it.

    Returns the stored config: explicit keys (including unknown legacy keys)
    plus the resolved profile keys, ``profile`` and ``profile_overrides``.
    Raises ``ValueError`` / ``pydantic.ValidationError`` on invalid input.
    """
    from .services.perf_profiles import resolve_job_config

    model = JobConfig.model_validate(dict(raw or {}))
    explicit = {k: v for k, v in model.model_dump().items() if v is not None}
    for key in ("output_dir", "video_path"):
        if key in (raw or {}) and (raw or {}).get(key) is None:
            explicit.pop(key, None)
    return resolve_job_config(explicit, strict=True)


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
    value: Union[float, str, bool, None] = None


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
