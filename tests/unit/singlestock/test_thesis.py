"""Tests for the TradeThesis dataclass and Direction enum."""

from __future__ import annotations

from datetime import datetime, timezone

from algotrader.singlestock.thesis import (
    AnnouncementsThesis,
    Direction,
    MarketContext,
    NewsThesis,
    TechnicalContext,
    TradeThesis,
)


def test_thesis_approved_property() -> None:
    t = TradeThesis(
        symbol="AAPL",
        direction=Direction.LONG,
        conviction=0.70,
        timestamp=datetime.now(timezone.utc),
    )
    assert t.approved

    t2 = TradeThesis(
        symbol="AAPL",
        direction=Direction.NONE,
        conviction=0.0,
        timestamp=datetime.now(timezone.utc),
    )
    assert not t2.approved

    t3 = TradeThesis(
        symbol="AAPL",
        direction=Direction.LONG,
        conviction=0.70,
        timestamp=datetime.now(timezone.utc),
        blackout_reason="earnings_proximity",
    )
    assert not t3.approved


def test_thesis_to_dict_round_trip() -> None:
    news = NewsThesis(direction=Direction.LONG, confidence=0.6, summary="bullish")
    ann = AnnouncementsThesis(
        direction=Direction.NONE,
        material_event_score=0.1,
        earnings_within_days=10,
    )
    mkt = MarketContext(
        beta_vs_spy=1.15,
        spy_trend="up",
        vix_level=14.5,
        regime="trending_up",
        market_aligned=True,
    )
    tech = TechnicalContext(
        direction=Direction.LONG,
        vwap=195.0,
        rsi_14=58.0,
        atr_14=3.10,
        breakout_level_up=198.5,
        breakdown_level_down=185.0,
        current_price=196.2,
        gap_pct=0.4,
    )
    t = TradeThesis(
        symbol="AAPL",
        direction=Direction.LONG,
        conviction=0.74,
        timestamp=datetime(2026, 5, 15, 13, 35, tzinfo=timezone.utc),
        entry_zone=(195.8, 196.6),
        stop_price=193.10,
        target_price=202.40,
        rationale="news + market both bullish, tech confirms",
        news=news,
        announcements=ann,
        market=mkt,
        technicals=tech,
    )
    d = t.to_dict()
    assert d["symbol"] == "AAPL"
    assert d["direction"] == "long"
    assert d["conviction"] == 0.74
    assert d["entry_zone"] == [195.8, 196.6]
    assert d["news"]["direction"] == "long"
    assert d["technicals"]["vwap"] == 195.0
