"""Tests for the DecisionAgent — covers earnings blackout, LLM fallback,
deterministic voting, and conviction-threshold filtering.
"""

from __future__ import annotations

from datetime import datetime, timezone

from algotrader.singlestock.agents.decision_agent import DecisionAgent
from algotrader.singlestock.thesis import (
    AnnouncementsThesis,
    Direction,
    MarketContext,
    NewsThesis,
    TechnicalContext,
)


class _StubLLM:
    """Test double — pretend the API is unavailable, no calls happen."""

    def __init__(self, available: bool = False, result: dict | None = None) -> None:
        self.available = available
        self._result = result
        self.calls: list[dict] = []

    @staticmethod
    def load_prompt(name: str) -> str:
        return "stub"

    def call_json(self, **kwargs) -> dict | None:
        self.calls.append(kwargs)
        return self._result


def _make_inputs(
    news_dir: Direction = Direction.LONG,
    ann_dir: Direction = Direction.LONG,
    tech_dir: Direction = Direction.LONG,
    earnings_in: int | None = None,
):
    news = NewsThesis(direction=news_dir, confidence=0.7, summary="x")
    ann = AnnouncementsThesis(
        direction=ann_dir,
        material_event_score=0.6,
        earnings_within_days=earnings_in,
    )
    mkt = MarketContext(
        beta_vs_spy=1.2,
        spy_trend="up",
        vix_level=14.0,
        regime="trending_up",
        market_aligned=True,
    )
    tech = TechnicalContext(
        direction=tech_dir,
        vwap=195.0,
        rsi_14=58.0,
        atr_14=3.0,
        breakout_level_up=198.0,
        breakdown_level_down=185.0,
        current_price=196.0,
        gap_pct=0.2,
    )
    return news, ann, mkt, tech


def test_earnings_blackout_short_circuits() -> None:
    agent = DecisionAgent(_StubLLM(available=False), earnings_blackout_days=2)
    news, ann, mkt, tech = _make_inputs(earnings_in=1)
    t = agent.decide("AAPL", news, ann, mkt, tech)
    assert t.direction == Direction.NONE
    assert t.blackout_reason == "earnings_proximity"
    assert t.conviction == 0.0


def test_deterministic_fallback_long_with_strong_consensus() -> None:
    llm = _StubLLM(available=False)
    agent = DecisionAgent(llm, min_conviction=0.55)
    news, ann, mkt, tech = _make_inputs()
    t = agent.decide("AAPL", news, ann, mkt, tech)
    # 3+ agent agreement → long, conviction crosses min
    assert t.direction == Direction.LONG
    assert t.conviction >= 0.55
    assert t.metadata["source"] == "deterministic_fallback"
    assert llm.calls == []  # No LLM call made — fallback only


def test_deterministic_fallback_mixed_signals() -> None:
    llm = _StubLLM(available=False)
    agent = DecisionAgent(llm)
    news, ann, mkt, tech = _make_inputs(news_dir=Direction.SHORT, tech_dir=Direction.LONG)
    t = agent.decide("AAPL", news, ann, mkt, tech)
    assert t.direction == Direction.NONE
    assert t.blackout_reason == "mixed_signals"


def test_llm_path_used_when_available() -> None:
    llm = _StubLLM(
        available=True,
        result={
            "direction": "long",
            "conviction": 0.82,
            "entry_zone": [195.8, 196.5],
            "stop_price": 193.0,
            "target_price": 202.0,
            "rationale": "all four agents bullish",
            "blackout_reason": None,
        },
    )
    agent = DecisionAgent(llm)
    news, ann, mkt, tech = _make_inputs()
    t = agent.decide("AAPL", news, ann, mkt, tech)
    assert t.direction == Direction.LONG
    assert t.conviction == 0.82
    assert t.entry_zone == (195.8, 196.5)
    assert t.metadata["source"] == "llm"
    assert len(llm.calls) == 1


def test_llm_blackout_response_zeros_conviction() -> None:
    llm = _StubLLM(
        available=True,
        result={
            "direction": "long",
            "conviction": 0.7,
            "blackout_reason": "thin_evidence",
        },
    )
    agent = DecisionAgent(llm)
    news, ann, mkt, tech = _make_inputs()
    t = agent.decide("AAPL", news, ann, mkt, tech)
    assert t.direction == Direction.NONE
    assert t.conviction == 0.0
    assert t.blackout_reason == "thin_evidence"


def test_missing_technicals_returns_blackout() -> None:
    agent = DecisionAgent(_StubLLM(available=False))
    news = NewsThesis(direction=Direction.LONG, confidence=0.5)
    ann = AnnouncementsThesis(direction=Direction.LONG, material_event_score=0.3)
    mkt = MarketContext(1.0, "up", 15.0, "trending_up", True)
    t = agent.decide("AAPL", news, ann, mkt, technicals=None)
    assert t.direction == Direction.NONE
    assert t.blackout_reason == "thin_evidence"
