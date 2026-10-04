"""Ingest source registry and per-source stat coverage (FR-SOURCE-01/02/03).

Raw uploaded files give the analysis pipeline full pixel access, so every
statistic is computable. Link-based sources are progressively more limited:
we do not own the file, the provider controls resolution and access, and
terms change over time. This module is the single source of truth for which
statistics each source can support, so the API can disclose expected
coverage at submission time and the stat catalog can mark the rest
unavailable rather than reporting a misleading zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse

# Every baseline stat key, in catalog order.
ALL_STAT_KEYS = [
    "goals", "assists", "possession", "total_shots", "shots_on_target", "saves",
    "offsides", "total_passes", "pass_accuracy", "key_passes", "duels", "fouls",
    "corners", "free_kicks", "penalties",
]

# Statistics that need sustained full-field pixel access to compute. Link
# sources that re-encode, crop, or restrict downloads lose these first.
_AGGREGATE_STATS = ["total_passes", "pass_accuracy", "key_passes", "duels", "possession"]

RAW_UPLOAD = "raw_upload"
LOCAL_PATH = "local_path"


@dataclass
class SourceType:
    key: str
    label: str
    kind: str  # file | link
    hostnames: List[str] = field(default_factory=list)
    unsupported_stats: List[str] = field(default_factory=list)
    premium_stats_supported: bool = False
    notes: str = ""

    @property
    def supported_stats(self) -> List[str]:
        blocked = set(self.unsupported_stats)
        return [key for key in ALL_STAT_KEYS if key not in blocked]

    def supports(self, stat_key: str) -> bool:
        return stat_key not in set(self.unsupported_stats)


SOURCE_TYPES: Dict[str, SourceType] = {
    RAW_UPLOAD: SourceType(
        key=RAW_UPLOAD,
        label="Raw video upload",
        kind="file",
        premium_stats_supported=True,
        notes="Best results: full stat catalog plus premium off-the-ball and individual statistics.",
    ),
    LOCAL_PATH: SourceType(
        key=LOCAL_PATH,
        label="Local file on the server",
        kind="file",
        premium_stats_supported=True,
        notes="Same coverage as a raw upload — the worker reads the original file in place.",
    ),
    "youtube": SourceType(
        key="youtube",
        label="YouTube",
        kind="link",
        hostnames=["youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"],
        unsupported_stats=_AGGREGATE_STATS,
        notes=(
            "Works well for goals, shots, and set pieces. We do not own the file, so aggregate "
            "passing, duel, and possession stats are not available, and coverage can change when "
            "the provider changes access terms."
        ),
    ),
    "vimeo": SourceType(
        key="vimeo",
        label="Vimeo",
        kind="link",
        hostnames=["vimeo.com", "player.vimeo.com"],
        unsupported_stats=_AGGREGATE_STATS,
        notes="Event detection is reliable; aggregate passing and possession stats need the original file.",
    ),
    "veo": SourceType(
        key="veo",
        label="VEO",
        kind="link",
        hostnames=["veo.co", "app.veo.co"],
        unsupported_stats=["total_passes", "pass_accuracy", "key_passes"],
        notes="Panoramic capture keeps the full field in frame, so possession and duels usually survive.",
    ),
    "hudl": SourceType(
        key="hudl",
        label="Hudl",
        kind="link",
        hostnames=["hudl.com", "www.hudl.com", "fan.hudl.com"],
        unsupported_stats=_AGGREGATE_STATS,
        notes="Playback restrictions limit sustained tracking needed for aggregate stats.",
    ),
    "pixellot": SourceType(
        key="pixellot",
        label="Pixellot",
        kind="link",
        hostnames=["pixellot.tv", "www.pixellot.tv"],
        unsupported_stats=["total_passes", "pass_accuracy", "key_passes"],
        notes="Wide-field automated capture retains most team-level context.",
    ),
    "xbotgo": SourceType(
        key="xbotgo",
        label="XbotGo",
        kind="link",
        hostnames=["xbotgo.com", "www.xbotgo.com"],
        unsupported_stats=_AGGREGATE_STATS,
        notes="Auto-tracked framing follows the ball, so off-ball aggregate stats are not computable.",
    ),
    "nbc_sports_engine": SourceType(
        key="nbc_sports_engine",
        label="NBC Sports Engine",
        kind="link",
        hostnames=["sportsengine.com", "www.sportsengine.com", "nbcsportsengine.com"],
        unsupported_stats=_AGGREGATE_STATS,
        notes="Broadcast-style feeds crop the field, which removes off-ball coverage.",
    ),
}

DEFAULT_LINK_SOURCE = SourceType(
    key="other_link",
    label="Other link",
    kind="link",
    unsupported_stats=_AGGREGATE_STATS,
    notes="Unrecognized provider — expect event detection only, with no aggregate team statistics.",
)


def get_source_type(key: Optional[str]) -> SourceType:
    """Resolve a stored source key. Unknown keys fall back to raw upload
    (the permissive default) so existing matches keep their coverage."""
    if not key:
        return SOURCE_TYPES[RAW_UPLOAD]
    if key in SOURCE_TYPES:
        return SOURCE_TYPES[key]
    if key == DEFAULT_LINK_SOURCE.key:
        return DEFAULT_LINK_SOURCE
    return SOURCE_TYPES[RAW_UPLOAD]


def detect_source_from_url(url: str) -> Optional[SourceType]:
    """Map a pasted URL to a known provider. Returns None for non-URLs."""
    value = str(url or "").strip()
    if not value or "://" not in value:
        return None
    try:
        host = (urlparse(value).hostname or "").lower()
    except ValueError:
        return None
    if not host:
        return None
    for source in SOURCE_TYPES.values():
        if host in source.hostnames:
            return source
    return DEFAULT_LINK_SOURCE


def source_to_dict(source: SourceType) -> Dict[str, object]:
    return {
        "key": source.key,
        "label": source.label,
        "kind": source.kind,
        "supported_stats": source.supported_stats,
        "unsupported_stats": list(source.unsupported_stats),
        "supported_stat_count": len(source.supported_stats),
        "total_stat_count": len(ALL_STAT_KEYS),
        "premium_stats_supported": source.premium_stats_supported,
        "notes": source.notes,
    }


def list_source_types() -> List[Dict[str, object]]:
    ordered = [SOURCE_TYPES[RAW_UPLOAD], SOURCE_TYPES[LOCAL_PATH]]
    ordered += [source for key, source in SOURCE_TYPES.items() if key not in {RAW_UPLOAD, LOCAL_PATH}]
    return [source_to_dict(source) for source in ordered]
