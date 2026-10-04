"""LLM match report: turn a run's analysis into a coach-readable summary.

Uses the same provider settings as the copilot agent (VH_LLM_PROVIDER =
ollama | openai | openai-compatible, VH_LLM_BASE_URL, VH_LLM_MODEL). YOLO
remains the perception engine - the LLM only narrates and evaluates the
structured outputs, and suggests tuning when detections look suspect.
Falls back to a deterministic template when no LLM is configured.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from ..config import settings

LOGGER = logging.getLogger("videohighlights.match_report")

_PROMPT = """You are an assistant coach and video analyst. Using ONLY the JSON
below (produced by a YOLO-based soccer tracking pipeline), write a concise
match report in Markdown:

1. **Match summary** - score with team names if attributed, headline moments.
2. **Timeline** - goals, cards, notable set pieces with MM:SS timestamps.
3. **Team analysis** - possession, territory, what it suggests.
4. **Data quality review** - call out anything suspicious for a human to
   verify (low-confidence goals, cards without corroboration, poor ball
   coverage) and suggest concrete pipeline tuning (sensitivities, goal boxes,
   team colors) when warranted.

Be factual; never invent events that are not in the data.

DATA:
{data}
"""


def _fallback_report(summary: Dict[str, object]) -> str:
    lines = ["# Match Report (no LLM configured)", ""]
    goals = summary.get("goal_events") or []
    lines.append(f"- Goals flagged: {len(goals)}")
    for g in goals:
        team = f" ({g.get('team')})" if g.get("team") else ""
        lines.append(f"  - {g.get('t', 0):.0f}s into the {g.get('side')} goal{team}, confidence {g.get('confidence')}")
    cards = summary.get("card_events") or []
    lines.append(f"- Cards flagged: {len(cards)}")
    ts = summary.get("team_stats") or {}
    if ts.get("possession_pct"):
        lines.append(f"- Possession: {ts['possession_pct']}")
    teams = ts.get("teams") if isinstance(ts.get("teams"), dict) else {}
    for key in sorted(teams):
        team = teams[key] if isinstance(teams[key], dict) else {}
        lines.append(
            f"- {team.get('name') or 'Team ' + str(key)}: possession {team.get('possession_pct')}%, "
            f"shots {team.get('shots')} ({team.get('shots_on_target')} on target), passes {team.get('passes')}, "
            f"goals {team.get('goals')}"
        )
    counts = (summary.get("events_summary") or {}).get("counts_by_type") or {}
    if counts:
        lines.append("- Events: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    lines.append("")
    lines.append("Set VH_LLM_PROVIDER=ollama and VH_LLM_MODEL to enable narrated reports.")
    return chr(10).join(lines)


def generate_match_report(summary: Dict[str, object]) -> str:
    """Return a Markdown report; deterministic fallback without an LLM."""
    provider = (settings.llm_provider or "none").lower()
    if provider in {"none", ""}:
        return _fallback_report(summary)
    try:
        from openai import OpenAI

        base_url = settings.llm_base_url
        if provider == "ollama":
            # Ollama serves the OpenAI-compatible API under /v1; accept the
            # base URL with or without the suffix (the agent settings use it
            # without).
            base_url = (base_url or "http://localhost:11434").rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
        client = OpenAI(
            api_key=settings.llm_api_key or settings.openai_api_key or "ollama",
            base_url=base_url if provider != "openai" else None,
            timeout=max(30.0, float(settings.llm_timeout_seconds or 30.0)),
        )
        payload = json.dumps(summary, default=str)[:24000]
        response = client.chat.completions.create(
            model=settings.llm_model or "llama3.1",
            messages=[{"role": "user", "content": _PROMPT.format(data=payload)}],
            temperature=0.3,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or _fallback_report(summary)
    except Exception as exc:
        LOGGER.warning("LLM match report failed (%s); using fallback", exc)
        return _fallback_report(summary)
