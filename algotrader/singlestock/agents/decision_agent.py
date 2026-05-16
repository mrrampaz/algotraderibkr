"""LLM-backed decision agent.

Synthesizes the four upstream agent outputs (news, announcements,
market context, technicals) into a final TradeThesis. Applies hard
blackout rules locally before invoking the LLM so we save a call on
obvious cash days (e.g. earnings within 2 days).

Falls back to a deterministic combiner when the LLM is unavailable —
typically yielding direction="none" since cross-agent agreement is
hard to score without judgment.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytz
import structlog

from algotrader.singlestock.llm_client import LLMClient
from algotrader.singlestock.thesis import (
    AnnouncementsThesis,
    Direction,
    MarketContext,
    NewsThesis,
    TechnicalContext,
    TradeThesis,
)

logger = structlog.get_logger()
ET = pytz.timezone("America/New_York")


class DecisionAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "claude-opus-4-7",
        min_conviction: float = 0.65,
        earnings_blackout_days: int = 2,
    ) -> None:
        self._llm = llm_client
        self._model = model
        self._min_conviction = min_conviction
        self._earnings_blackout_days = earnings_blackout_days
        self._log = logger.bind(component="decision_agent")
        self._system_prompt: str | None = None

    def _load_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = self._llm.load_prompt("trade_decision.md")
        return self._system_prompt

    def decide(
        self,
        symbol: str,
        news: NewsThesis,
        announcements: AnnouncementsThesis,
        market: MarketContext,
        technicals: TechnicalContext | None,
    ) -> TradeThesis:
        now = datetime.now(ET)

        # Hard blackouts checked locally — save an LLM call when obvious.
        if (
            announcements.earnings_within_days is not None
            and announcements.earnings_within_days <= self._earnings_blackout_days
        ):
            return TradeThesis(
                symbol=symbol,
                direction=Direction.NONE,
                conviction=0.0,
                timestamp=now,
                rationale=(
                    f"earnings in {announcements.earnings_within_days} day(s); "
                    "single-stock blackout to avoid event-day vol crush."
                ),
                blackout_reason="earnings_proximity",
                news=news,
                announcements=announcements,
                market=market,
                technicals=technicals,
            )

        if technicals is None:
            return TradeThesis(
                symbol=symbol,
                direction=Direction.NONE,
                conviction=0.0,
                timestamp=now,
                rationale="technical signals unavailable (insufficient bars)",
                blackout_reason="thin_evidence",
                news=news,
                announcements=announcements,
                market=market,
                technicals=technicals,
            )

        # LLM call.
        if self._llm.available:
            payload = self._build_payload(symbol, news, announcements, market, technicals)
            result = self._llm.call_json(
                model=self._model,
                system_prompt=self._load_system_prompt(),
                user_payload=payload,
                max_tokens=1200,
                agent_name="decision",
            )
            if result is not None:
                return self._thesis_from_llm(
                    symbol=symbol,
                    now=now,
                    result=result,
                    news=news,
                    announcements=announcements,
                    market=market,
                    technicals=technicals,
                )

        # Deterministic fallback — cash-default unless strong agreement.
        return self._deterministic_fallback(
            symbol=symbol,
            now=now,
            news=news,
            announcements=announcements,
            market=market,
            technicals=technicals,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_payload(
        self,
        symbol: str,
        news: NewsThesis,
        announcements: AnnouncementsThesis,
        market: MarketContext,
        technicals: TechnicalContext,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "current_price": technicals.current_price,
            "atr_14": technicals.atr_14,
            "news": {
                "direction": news.direction.value,
                "confidence": news.confidence,
                "key_catalysts": news.key_catalysts,
                "avg_sentiment": news.avg_sentiment,
                "summary": news.summary,
            },
            "announcements": {
                "direction": announcements.direction.value,
                "material_event_score": announcements.material_event_score,
                "recent_filings": announcements.recent_filings,
                "earnings_within_days": announcements.earnings_within_days,
                "summary": announcements.summary,
            },
            "market": {
                "beta_vs_spy": market.beta_vs_spy,
                "spy_trend": market.spy_trend,
                "vix_level": market.vix_level,
                "regime": market.regime,
                "market_aligned": market.market_aligned,
            },
            "technicals": {
                "direction": technicals.direction.value,
                "vwap": technicals.vwap,
                "rsi_14": technicals.rsi_14,
                "breakout_level_up": technicals.breakout_level_up,
                "breakdown_level_down": technicals.breakdown_level_down,
                "current_price": technicals.current_price,
                "gap_pct": technicals.gap_pct,
            },
        }

    def _thesis_from_llm(
        self,
        symbol: str,
        now: datetime,
        result: dict[str, Any],
        news: NewsThesis,
        announcements: AnnouncementsThesis,
        market: MarketContext,
        technicals: TechnicalContext,
    ) -> TradeThesis:
        direction = _parse_direction(result.get("direction"))
        conviction = _clip01(result.get("conviction", 0.0))
        blackout = result.get("blackout_reason")
        if blackout in (None, "null", ""):
            blackout = None

        ez = result.get("entry_zone")
        if isinstance(ez, (list, tuple)) and len(ez) == 2:
            try:
                entry_zone = (float(ez[0]), float(ez[1]))
            except (TypeError, ValueError):
                entry_zone = None
        else:
            entry_zone = None

        return TradeThesis(
            symbol=symbol,
            direction=direction if blackout is None else Direction.NONE,
            conviction=conviction if blackout is None else 0.0,
            timestamp=now,
            entry_zone=entry_zone,
            stop_price=_optional_float(result.get("stop_price")),
            target_price=_optional_float(result.get("target_price")),
            rationale=str(result.get("rationale", "") or "")[:1000],
            blackout_reason=blackout,
            news=news,
            announcements=announcements,
            market=market,
            technicals=technicals,
            metadata={"source": "llm"},
        )

    def _deterministic_fallback(
        self,
        symbol: str,
        now: datetime,
        news: NewsThesis,
        announcements: AnnouncementsThesis,
        market: MarketContext,
        technicals: TechnicalContext,
    ) -> TradeThesis:
        # Tally directions from the four sub-agents.
        votes = {Direction.LONG: 0, Direction.SHORT: 0}
        for d in (news.direction, announcements.direction, technicals.direction):
            if d in votes:
                votes[d] += 1
        # Market alignment counts as a half-vote toward SPY trend direction.
        if market.market_aligned and market.spy_trend == "up":
            votes[Direction.LONG] += 1
        elif market.market_aligned and market.spy_trend == "down":
            votes[Direction.SHORT] += 1

        if votes[Direction.LONG] >= 3 and votes[Direction.SHORT] == 0:
            direction = Direction.LONG
        elif votes[Direction.SHORT] >= 3 and votes[Direction.LONG] == 0:
            direction = Direction.SHORT
        else:
            return TradeThesis(
                symbol=symbol,
                direction=Direction.NONE,
                conviction=0.0,
                timestamp=now,
                rationale="deterministic fallback: insufficient agent agreement",
                blackout_reason="mixed_signals",
                news=news,
                announcements=announcements,
                market=market,
                technicals=technicals,
                metadata={"source": "deterministic_fallback"},
            )

        atr = max(0.01, technicals.atr_14)
        price = technicals.current_price
        if direction == Direction.LONG:
            stop = price - atr
            target = price + 2 * atr
        else:
            stop = price + atr
            target = price - 2 * atr

        # Conviction = weighted average of sub-signals, capped at 0.75
        # since LLM is unavailable.
        conviction = min(
            0.75,
            0.30 * news.confidence
            + 0.30 * announcements.material_event_score
            + 0.40 * (0.6 if technicals.direction == direction else 0.2),
        )

        return TradeThesis(
            symbol=symbol,
            direction=direction if conviction >= self._min_conviction else Direction.NONE,
            conviction=conviction,
            timestamp=now,
            entry_zone=(price * 0.999, price * 1.002) if direction == Direction.LONG else (price * 0.998, price * 1.001),
            stop_price=round(stop, 2),
            target_price=round(target, 2),
            rationale=(
                f"deterministic fallback: {votes[direction]} votes for {direction.value}, "
                f"conviction={conviction:.2f}"
            ),
            blackout_reason=None if conviction >= self._min_conviction else "thin_evidence",
            news=news,
            announcements=announcements,
            market=market,
            technicals=technicals,
            metadata={"source": "deterministic_fallback", "votes": {k.value: v for k, v in votes.items()}},
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


def _optional_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
