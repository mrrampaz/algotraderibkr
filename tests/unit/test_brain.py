from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytz

from algotrader.core.models import MarketRegime, OrderSide, Position, RegimeType
from algotrader.strategies.base import OpportunityAssessment
from algotrader.strategy_selector.brain import DailyBrain
from algotrader.strategy_selector.candidate import CandidateType, TradeCandidate


@dataclass
class _FakeTrade:
    realized_pnl: float
    side: OrderSide
    entry_time: datetime | None = None
    exit_time: datetime | None = None


class _FakeTradeJournal:
    def __init__(self, trades: dict[tuple[str, str], list[_FakeTrade]] | None = None) -> None:
        self._trades = trades or {}
        self._summaries: dict[str, dict] = {}

    def set_summary(self, strategy_name: str, total_trades: int, total_pnl: float) -> None:
        self._summaries[strategy_name] = {
            "strategy": strategy_name,
            "days": 1,
            "total_trades": total_trades,
            "wins": 0,
            "losses": total_trades,
            "win_rate": 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / total_trades if total_trades else 0.0,
        }

    def get_trades(
        self,
        strategy_name: str | None = None,
        symbol: str | None = None,
        **kwargs,
    ) -> list[_FakeTrade]:
        if strategy_name is None or symbol is None:
            return []
        return list(self._trades.get((strategy_name, symbol), []))

    def get_strategy_summary(self, strategy_name: str, days: int = 30) -> dict:
        return self._summaries.get(
            strategy_name,
            {
                "strategy": strategy_name,
                "days": days,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
            },
        )


class _FakeCalendar:
    def __init__(self, event_day: bool = False) -> None:
        self._event_day = event_day

    def is_event_day(self, target_date=None) -> bool:
        return self._event_day


def _regime(regime_type: RegimeType = RegimeType.TRENDING_UP, vix: float = 18.0) -> MarketRegime:
    return MarketRegime(
        regime_type=regime_type,
        vix_level=vix,
        confidence=0.9,
        timestamp=datetime.now(pytz.UTC),
    )


def _regime_config() -> dict:
    return {
        "regime_weights": {
            "trending_up": {
                "momentum": 0.9,
                "pairs_trading": 0.6,
                "options_premium": 0.6,
                "vwap_reversion": 0.2,
            },
            "ranging": {
                "vwap_reversion": 0.9,
                "pairs_trading": 0.9,
                "momentum": 0.2,
            },
        },
        "candidate_bonuses": {
            "trending_up": {"momentum": 0.10},
            "ranging": {"pairs_trading": 0.10},
        },
    }


def _brain(**kwargs) -> DailyBrain:
    return DailyBrain(
        total_capital=100_000.0,
        regime_config=_regime_config(),
        trade_journal=kwargs.pop("trade_journal", _FakeTradeJournal()),
        event_calendar=kwargs.pop("event_calendar", _FakeCalendar()),
        min_confidence=kwargs.pop("min_confidence", 0.60),
        min_risk_reward=kwargs.pop("min_risk_reward", 1.5),
        min_edge_pct=kwargs.pop("min_edge_pct", 0.3),
        max_daily_trades=kwargs.pop("max_daily_trades", 5),
        max_capital_per_trade_pct=kwargs.pop("max_capital_per_trade_pct", 20.0),
        max_daily_risk_pct=kwargs.pop("max_daily_risk_pct", 2.0),
        cash_is_default=kwargs.pop("cash_is_default", True),
        regime_mismatch_penalty=kwargs.pop("regime_mismatch_penalty", 0.5),
        correlation_penalty=kwargs.pop("correlation_penalty", 0.3),
        recent_loss_cooldown_hours=kwargs.pop("recent_loss_cooldown_hours", 4),
        midday_confidence_multiplier=kwargs.pop("midday_confidence_multiplier", 1.2),
        midday_pnl_stop_pct=kwargs.pop("midday_pnl_stop_pct", -1.0),
    )


def _candidate(
    *,
    symbol: str,
    strategy_name: str = "momentum",
    confidence: float = 0.8,
    rr: float = 2.0,
    edge: float = 0.8,
    regime_fit: float = 0.9,
    risk_dollars: float = 300.0,
    direction: str = "long",
    entry_price: float = 100.0,
    stop_price: float = 95.0,
    target_price: float = 110.0,
    candidate_type: CandidateType = CandidateType.LONG_EQUITY,
    suggested_qty: int = 0,
    metadata: dict | None = None,
) -> TradeCandidate:
    return TradeCandidate(
        strategy_name=strategy_name,
        candidate_type=candidate_type,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_dollars=risk_dollars,
        suggested_qty=suggested_qty,
        risk_reward_ratio=rr,
        confidence=confidence,
        edge_estimate_pct=edge,
        regime_fit=regime_fit,
        catalyst="breakout_volume_2.0x",
        metadata=metadata or {},
    )


def test_no_candidates_returns_cash_day() -> None:
    brain = _brain()
    decision = brain.decide(
        regime=_regime(),
        assessments={"momentum": OpportunityAssessment()},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.is_cash_day
    assert decision.num_trades == 0
    assert decision.cash_pct == 100.0
    assert "100% cash" in decision.reasoning


def test_one_high_quality_candidate_selected_with_allocation() -> None:
    brain = _brain()
    cand = _candidate(symbol="AAPL", suggested_qty=50, risk_dollars=500.0)
    decision = brain.decide(
        regime=_regime(),
        assessments={"momentum": OpportunityAssessment(candidates=[cand])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 1
    selected = decision.selected_trades[0]
    assert selected.candidate.symbol == "AAPL"
    assert selected.allocated_capital == 5000.0
    assert selected.position_size == 50


def test_only_one_candidate_passes_thresholds() -> None:
    brain = _brain()
    good = _candidate(symbol="GOOD")
    low_conf = _candidate(symbol="LOWC", confidence=0.5)
    poor_rr = _candidate(symbol="LOWR", rr=1.1)
    low_edge = _candidate(symbol="LOWE", edge=0.1)

    decision = brain.decide(
        regime=_regime(),
        assessments={
            "momentum": OpportunityAssessment(
                candidates=[good, low_conf, poor_rr, low_edge],
            )
        },
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 1
    assert decision.selected_trades[0].candidate.symbol == "GOOD"
    reasons = {rej.reason for rej in decision.rejected_trades}
    assert "low_confidence" in reasons
    assert "poor_rr" in reasons
    assert "insufficient_edge" in reasons


def test_multiple_good_candidates_ranked_and_limited_by_risk() -> None:
    brain = _brain(max_daily_trades=5, max_daily_risk_pct=1.0)
    c1 = _candidate(symbol="AAA", confidence=0.90, risk_dollars=400.0)
    c2 = _candidate(symbol="BBB", confidence=0.85, risk_dollars=400.0)
    c3 = _candidate(symbol="CCC", confidence=0.80, risk_dollars=400.0)

    decision = brain.decide(
        regime=_regime(),
        assessments={"momentum": OpportunityAssessment(candidates=[c3, c1, c2])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert [s.candidate.symbol for s in decision.selected_trades] == ["AAA", "BBB"]
    rejected = {(r.candidate.symbol, r.reason) for r in decision.rejected_trades}
    assert ("CCC", "daily_risk_limit") in rejected


def test_all_below_min_confidence_is_cash_day() -> None:
    brain = _brain(min_confidence=0.70)
    c1 = _candidate(symbol="AAA", confidence=0.5)
    c2 = _candidate(symbol="BBB", confidence=0.6)

    decision = brain.decide(
        regime=_regime(),
        assessments={"momentum": OpportunityAssessment(candidates=[c1, c2])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.is_cash_day
    assert all(r.reason == "low_confidence" for r in decision.rejected_trades)


def test_correlated_candidates_reject_second_one() -> None:
    brain = _brain()
    c1 = _candidate(symbol="AAPL", metadata={"sector": "tech"})
    c2 = _candidate(symbol="MSFT", metadata={"sector": "tech"})

    decision = brain.decide(
        regime=_regime(),
        assessments={"momentum": OpportunityAssessment(candidates=[c1, c2])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 1
    reasons = {(r.candidate.symbol, r.reason) for r in decision.rejected_trades}
    assert ("MSFT", "correlated_with_selected") in reasons or ("AAPL", "correlated_with_selected") in reasons


def test_daily_risk_limit_rejects_additional_candidates() -> None:
    brain = _brain(max_daily_risk_pct=1.0)
    c1 = _candidate(symbol="AAA", risk_dollars=700.0)
    c2 = _candidate(symbol="BBB", risk_dollars=700.0)

    decision = brain.decide(
        regime=_regime(),
        assessments={"momentum": OpportunityAssessment(candidates=[c1, c2])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 1
    assert any(r.reason == "daily_risk_limit" for r in decision.rejected_trades)


def test_synthetic_candidate_fallback_from_old_assessment() -> None:
    brain = _brain()
    assessment = OpportunityAssessment(
        num_candidates=2,
        avg_risk_reward=2.2,
        confidence=0.85,
        estimated_edge_pct=0.8,
        details=[{"symbol": "AAPL"}],
    )
    decision = brain.decide(
        regime=_regime(),
        assessments={"momentum": assessment},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 1
    assert decision.selected_trades[0].candidate.metadata.get("synthetic") is True
    assert decision.selected_trades[0].candidate.symbol == "AAPL"


def test_midday_review_uses_higher_confidence_threshold() -> None:
    brain = _brain(min_confidence=0.60, midday_confidence_multiplier=1.2)
    c1 = _candidate(symbol="AAPL", confidence=0.65)

    decision = brain.review_midday(
        regime=_regime(),
        assessments={"momentum": OpportunityAssessment(candidates=[c1])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 0
    assert any(r.reason == "low_confidence" for r in decision.rejected_trades)


def test_regime_mismatch_penalty_lowers_selection_priority() -> None:
    brain = _brain(max_daily_trades=1)
    strong_fit = _candidate(symbol="FIT", regime_fit=0.9, confidence=0.8)
    weak_fit = _candidate(symbol="MISFIT", regime_fit=0.1, confidence=0.8)

    decision = brain.decide(
        regime=_regime(),
        assessments={"momentum": OpportunityAssessment(candidates=[weak_fit, strong_fit])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 1
    assert decision.selected_trades[0].candidate.symbol == "FIT"
