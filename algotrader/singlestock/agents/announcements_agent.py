"""LLM-backed announcements agent.

Pulls recent SEC filings (8-K, 10-Q, 10-K, Form 4 insider) for the
symbol, checks earnings proximity from the seeded calendar, and asks
Claude to assess material-event impact.

Earnings-within-N-days returns direction="none" with a blackout in the
final thesis — this agent itself doesn't enforce, just reports.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import structlog

from algotrader.singlestock.feeds.earnings_calendar import next_earnings
from algotrader.singlestock.feeds.sec_edgar import Filing, SECEdgarClient
from algotrader.singlestock.llm_client import LLMClient
from algotrader.singlestock.thesis import AnnouncementsThesis, Direction

logger = structlog.get_logger()


class AnnouncementsAgent:
    def __init__(
        self,
        edgar_client: SECEdgarClient,
        llm_client: LLMClient,
        model: str = "claude-sonnet-4-6",
        lookback_days: int = 30,
        fetch_excerpts: bool = True,
    ) -> None:
        self._edgar = edgar_client
        self._llm = llm_client
        self._model = model
        self._lookback_days = lookback_days
        self._fetch_excerpts = fetch_excerpts
        self._log = logger.bind(component="announcements_agent")
        self._system_prompt: str | None = None

    def _load_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = self._llm.load_prompt("announcement_assessment.md")
        return self._system_prompt

    def assess(self, symbol: str) -> AnnouncementsThesis:
        filings = self._edgar.fetch_recent_filings(
            symbol,
            lookback_days=self._lookback_days,
            max_filings=20,
        )
        material = [f for f in filings if f.is_material]
        insiders = [f for f in filings if f.is_insider]

        nxt = next_earnings(symbol, date.today())
        days_until_earnings = nxt.days_away if nxt is not None else None

        # If no material filings AND no insider activity AND no LLM,
        # short-circuit with a calm baseline.
        if not material and not insiders and not self._llm.available:
            return AnnouncementsThesis(
                direction=Direction.NONE,
                material_event_score=0.0,
                recent_filings=[],
                earnings_within_days=days_until_earnings,
                summary="no recent material filings or insider activity",
            )

        # Optionally enrich material filings with short excerpts.
        excerpts: dict[str, str] = {}
        if self._fetch_excerpts and self._llm.available:
            for f in material[:3]:
                excerpts[f.accession] = self._edgar.fetch_filing_summary(f, symbol)

        payload = {
            "symbol": symbol,
            "next_earnings_date": nxt.earnings_date.isoformat() if nxt else None,
            "days_until_earnings": days_until_earnings,
            "filings": [
                {
                    "form": f.form,
                    "filed_date": f.filed_date.isoformat(),
                    "items": f.items,
                    "summary_excerpt": excerpts.get(f.accession, "")[:2000],
                }
                for f in material[:10]
            ],
            "recent_insider_activity": [
                {
                    "date": f.filed_date.isoformat(),
                    "form": f.form,
                }
                for f in insiders[:10]
            ],
        }

        if not self._llm.available:
            # Deterministic fallback: presence of any material filing
            # in the lookback raises score moderately; we stay neutral
            # on direction unless earnings is very close.
            score = 0.0
            if material:
                score = 0.3
            if insiders:
                score = max(score, 0.2)
            return AnnouncementsThesis(
                direction=Direction.NONE,
                material_event_score=score,
                recent_filings=[
                    f"{f.form} on {f.filed_date.isoformat()}" for f in material[:5]
                ],
                earnings_within_days=days_until_earnings,
                summary="LLM unavailable; heuristic baseline",
            )

        result = self._llm.call_json(
            model=self._model,
            system_prompt=self._load_system_prompt(),
            user_payload=payload,
            max_tokens=900,
            agent_name="announcements",
        )
        if result is None:
            return AnnouncementsThesis(
                direction=Direction.NONE,
                material_event_score=0.0,
                recent_filings=[],
                earnings_within_days=days_until_earnings,
                summary="LLM call failed; cash-default",
            )

        return AnnouncementsThesis(
            direction=_parse_direction(result.get("direction")),
            material_event_score=_clip01(result.get("material_event_score", 0.0)),
            recent_filings=list(result.get("recent_filings", []) or [])[:8],
            earnings_within_days=_int_or_none(result.get("earnings_within_days", days_until_earnings)),
            summary=str(result.get("summary", "") or "")[:600],
        )


def _parse_direction(raw: Any) -> Direction:
    s = str(raw or "none").lower().strip()
    if s == "long":
        return Direction.LONG
    if s == "short":
        return Direction.SHORT
    return Direction.NONE


def _clip01(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


def _int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
