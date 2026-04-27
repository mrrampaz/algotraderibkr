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


class _FakeLogger:
    def __init__(self) -> None:
        self.infos: list[tuple[str, dict]] = []
        self.warnings: list[tuple[str, dict]] = []

    def info(self, event: str, **kwargs) -> None:
        self.infos.append((event, kwargs))

    def warning(self, event: str, **kwargs) -> None:
        self.warnings.append((event, kwargs))

    def debug(self, event: str, **kwargs) -> None:
        pass

    def exception(self, event: str, **kwargs) -> None:
        pass


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
        strategy_threshold_overrides=kwargs.pop("strategy_threshold_overrides", None),
        max_daily_trades=kwargs.pop("max_daily_trades", 5),
        max_capital_per_trade_pct=kwargs.pop("max_capital_per_trade_pct", 20.0),
        max_daily_risk_pct=kwargs.pop("max_daily_risk_pct", 2.0),
        cash_is_default=kwargs.pop("cash_is_default", True),
        regime_mismatch_penalty=kwargs.pop("regime_mismatch_penalty", 0.5),
        correlation_penalty=kwargs.pop("correlation_penalty", 0.3),
        recent_loss_cooldown_hours=kwargs.pop("recent_loss_cooldown_hours", 4),
        midday_confidence_multiplier=kwargs.pop("midday_confidence_multiplier", 1.2),
        midday_pnl_stop_pct=kwargs.pop("midday_pnl_stop_pct", -1.0),
        adaptive_sizing=kwargs.pop("adaptive_sizing", False),
        adaptive_risk_tiers=kwargs.pop("adaptive_risk_tiers", None),
        drawdown_governor=kwargs.pop("drawdown_governor", None),
        max_contracts_hard_cap=kwargs.pop("max_contracts_hard_cap", 10),
        max_overnight_exposure_pct=kwargs.pop("max_overnight_exposure_pct", 40.0),
        recent_win_rate_lookback_trades=kwargs.pop("recent_win_rate_lookback_trades", 15),
        recent_win_rate_fallback=kwargs.pop("recent_win_rate_fallback", 0.80),
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
    options_structure: str = "",
    contracts: int = 0,
    credit_received: float = 0.0,
    max_loss: float = 0.0,
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
        options_structure=options_structure,
        contracts=contracts,
        credit_received=credit_received,
        max_loss=max_loss,
        metadata=metadata or {},
    )


def _option_candidate(
    *,
    symbol: str,
    confidence: float = 0.80,
    edge: float = 1.0,
    risk_dollars: float = 2800.0,
    contracts: int = 8,
    credit_received: float = 700.0,
) -> TradeCandidate:
    return _candidate(
        symbol=symbol,
        strategy_name="options_premium",
        candidate_type=CandidateType.CREDIT_SPREAD,
        direction="neutral",
        entry_price=500.0,
        stop_price=495.0,
        target_price=500.0,
        risk_dollars=risk_dollars,
        confidence=confidence,
        rr=0.43,
        edge=edge,
        options_structure="put_spread",
        contracts=contracts,
        suggested_qty=contracts,
        credit_received=credit_received,
        max_loss=risk_dollars,
        metadata={"win_rate": 0.80},
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


def test_commission_gate_rejects_low_edge_dollar_candidates() -> None:
    brain = _brain()
    low_commission_edge = _candidate(symbol="THIN", risk_dollars=50.0, edge=0.8)
    good = _candidate(symbol="GOOD", risk_dollars=600.0, edge=0.8)

    decision = brain.decide(
        regime=_regime(),
        assessments={"momentum": OpportunityAssessment(candidates=[low_commission_edge, good])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 1
    assert decision.selected_trades[0].candidate.symbol == "GOOD"
    reasons = {(r.candidate.symbol, r.reason) for r in decision.rejected_trades}
    assert ("THIN", "commission_exceeds_edge") in reasons


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


def test_adaptive_risk_tier_single_strategy_boosts_to_hard_cap() -> None:
    brain = _brain(adaptive_sizing=True, max_daily_risk_pct=6.0)
    limits = brain._compute_adaptive_risk_limits(
        active_strategy_count=1,
        recent_win_rate=0.90,
        current_drawdown_pct=0.0,
    )

    assert limits["max_daily_risk_pct"] == 6.0
    assert limits["max_contracts"] == 10


def test_adaptive_risk_tier_moderate_drawdown_reduces_risk_and_contracts() -> None:
    brain = _brain(adaptive_sizing=True, max_daily_risk_pct=6.0)
    limits = brain._compute_adaptive_risk_limits(
        active_strategy_count=1,
        recent_win_rate=0.90,
        current_drawdown_pct=2.0,
    )

    assert limits["max_daily_risk_pct"] == 4.2
    assert limits["max_contracts"] == 9


def test_adaptive_risk_tier_severe_drawdown_halves_and_clamps() -> None:
    brain = _brain(adaptive_sizing=True, max_daily_risk_pct=6.0)
    limits = brain._compute_adaptive_risk_limits(
        active_strategy_count=1,
        recent_win_rate=0.90,
        current_drawdown_pct=4.0,
    )

    assert limits["max_daily_risk_pct"] == 3.0
    assert limits["max_contracts"] == 5


def test_decide_applies_adaptive_options_contract_cap() -> None:
    brain = _brain(
        adaptive_sizing=True,
        max_daily_risk_pct=6.0,
        max_capital_per_trade_pct=25.0,
    )
    brain._get_recent_win_rate = lambda lookback_trades=15: 0.90  # type: ignore[method-assign]
    brain._get_current_drawdown = lambda: 0.0  # type: ignore[method-assign]

    options = _candidate(
        symbol="SPY",
        strategy_name="options_premium",
        candidate_type=CandidateType.CREDIT_SPREAD,
        direction="neutral",
        entry_price=500.0,
        stop_price=495.0,
        target_price=500.0,
        risk_dollars=5700.0,
        confidence=0.80,
        rr=0.35,
        edge=0.9,
        options_structure="put_spread",
        contracts=12,
        suggested_qty=12,
        credit_received=1200.0,
        max_loss=5700.0,
        metadata={"win_rate": 0.90},
    )

    decision = brain.decide(
        regime=_regime(RegimeType.HIGH_VOL, vix=24.0),
        assessments={"options_premium": OpportunityAssessment(candidates=[options])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 1
    selected = decision.selected_trades[0]
    assert selected.position_size == 10
    assert selected.risk_amount == 4750.0


def test_adaptive_options_contract_cap_does_not_upsize_candidate() -> None:
    brain = _brain(
        adaptive_sizing=True,
        max_daily_risk_pct=6.0,
        max_capital_per_trade_pct=25.0,
    )
    brain._get_recent_win_rate = lambda lookback_trades=15: 0.90  # type: ignore[method-assign]
    brain._get_current_drawdown = lambda: 0.0  # type: ignore[method-assign]

    options = _option_candidate(
        symbol="SPY",
        risk_dollars=2800.0,
        contracts=8,
        credit_received=800.0,
    )

    decision = brain.decide(
        regime=_regime(RegimeType.HIGH_VOL, vix=24.0),
        assessments={"options_premium": OpportunityAssessment(candidates=[options])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 1
    selected = decision.selected_trades[0]
    assert selected.position_size == 8
    assert selected.risk_amount == 2800.0


def test_brain_accepts_one_when_two_exceed_daily_cap() -> None:
    """When 2 candidates would exceed daily risk cap, Brain accepts the higher-EV one."""
    brain = _brain(
        adaptive_sizing=False,
        max_daily_risk_pct=6.0,
        max_capital_per_trade_pct=25.0,
    )
    spy = _option_candidate(
        symbol="SPY",
        confidence=0.85,
        edge=1.2,
        risk_dollars=3500.0,
        contracts=10,
        credit_received=900.0,
    )
    qqq = _option_candidate(
        symbol="QQQ",
        confidence=0.80,
        edge=1.0,
        risk_dollars=3500.0,
        contracts=10,
        credit_received=900.0,
    )

    decision = brain.decide(
        regime=_regime(RegimeType.HIGH_VOL, vix=24.0),
        assessments={"options_premium": OpportunityAssessment(candidates=[qqq, spy])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert [s.candidate.symbol for s in decision.selected_trades] == ["SPY"]
    rejected = {(r.candidate.symbol, r.reason) for r in decision.rejected_trades}
    assert ("QQQ", "daily_risk_limit") in rejected


def test_brain_logs_actionable_diagnostics_on_full_rejection() -> None:
    """When all candidates are risk-rejected, log includes per-candidate risk and cap."""
    brain = _brain(
        adaptive_sizing=False,
        max_daily_risk_pct=2.0,
        max_capital_per_trade_pct=25.0,
    )
    fake_log = _FakeLogger()
    brain._log = fake_log  # type: ignore[assignment]
    spy = _option_candidate(symbol="SPY", risk_dollars=3000.0, contracts=10, credit_received=900.0)
    qqq = _option_candidate(symbol="QQQ", risk_dollars=3000.0, contracts=10, credit_received=900.0)

    decision = brain.decide(
        regime=_regime(RegimeType.HIGH_VOL, vix=24.0),
        assessments={"options_premium": OpportunityAssessment(candidates=[spy, qqq])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.is_cash_day
    event, payload = next(item for item in fake_log.warnings if item[0] == "brain_risk_limit_blocked_all")
    assert event == "brain_risk_limit_blocked_all"
    assert payload["daily_risk_cap_pct"] == 2.0
    assert payload["total_requested_risk_pct"] == 6.0
    assert payload["individual_risk_pcts"] == [3.0, 3.0]
    assert payload["daily_risk_rejections"] == 2


def test_paired_spreads_fit_within_daily_risk_at_3pct_per_position() -> None:
    """SPY + QQQ paired spreads at 3% each fit under 6% daily cap."""
    brain = _brain(
        adaptive_sizing=False,
        max_daily_risk_pct=6.0,
        max_capital_per_trade_pct=25.0,
    )
    spy = _option_candidate(symbol="SPY", risk_dollars=3000.0, contracts=8, credit_received=800.0)
    qqq = _option_candidate(symbol="QQQ", risk_dollars=3000.0, contracts=8, credit_received=800.0)

    decision = brain.decide(
        regime=_regime(RegimeType.HIGH_VOL, vix=24.0),
        assessments={"options_premium": OpportunityAssessment(candidates=[spy, qqq])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert [s.candidate.symbol for s in decision.selected_trades] == ["SPY", "QQQ"]
    assert decision.total_risk_pct == 6.0


def test_options_only_brain_decision() -> None:
    """Brain decision log does not mention disabled strategies."""
    brain = _brain(
        adaptive_sizing=True,
        max_daily_risk_pct=6.0,
        max_capital_per_trade_pct=25.0,
    )
    brain._get_recent_win_rate = lambda lookback_trades=15: 0.85  # type: ignore[method-assign]
    brain._get_current_drawdown = lambda: 0.0  # type: ignore[method-assign]
    options = _candidate(
        symbol="SPY",
        strategy_name="options_premium",
        candidate_type=CandidateType.CREDIT_SPREAD,
        direction="neutral",
        entry_price=500.0,
        stop_price=495.0,
        target_price=500.0,
        risk_dollars=3200.0,
        confidence=0.80,
        rr=0.35,
        edge=1.0,
        options_structure="put_spread",
        contracts=8,
        suggested_qty=8,
        credit_received=800.0,
        max_loss=3200.0,
        metadata={"win_rate": 0.85},
    )

    decision = brain.decide(
        regime=_regime(RegimeType.HIGH_VOL, vix=24.0),
        assessments={"options_premium": OpportunityAssessment(candidates=[options])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 1
    assert decision.selected_trades[0].candidate.strategy_name == "options_premium"
    parked = [
        "momentum",
        "vwap_reversion",
        "gap_reversal",
        "pairs_trading",
        "sector_rotation",
        "event_driven",
    ]
    assert all(name not in decision.reasoning for name in parked)


def test_increased_sizing_respects_overnight_cap() -> None:
    """10-contract spread sizing still respects 40% overnight exposure cap."""
    brain = _brain(
        adaptive_sizing=True,
        max_daily_risk_pct=6.0,
        max_capital_per_trade_pct=25.0,
        max_overnight_exposure_pct=40.0,
    )
    brain._get_recent_win_rate = lambda lookback_trades=15: 0.85  # type: ignore[method-assign]
    brain._get_current_drawdown = lambda: 0.0  # type: ignore[method-assign]
    existing = Position(
        symbol="AAPL",
        qty=100.0,
        side=OrderSide.BUY,
        avg_entry_price=370.0,
        market_value=37_000.0,
    )
    options = _candidate(
        symbol="SPY",
        strategy_name="options_premium",
        candidate_type=CandidateType.CREDIT_SPREAD,
        direction="neutral",
        entry_price=500.0,
        stop_price=495.0,
        target_price=500.0,
        risk_dollars=3200.0,
        confidence=0.80,
        rr=0.35,
        edge=1.0,
        options_structure="put_spread",
        contracts=8,
        suggested_qty=8,
        credit_received=800.0,
        max_loss=3200.0,
        metadata={"win_rate": 0.85},
    )

    decision = brain.decide(
        regime=_regime(RegimeType.HIGH_VOL, vix=24.0),
        assessments={"options_premium": OpportunityAssessment(candidates=[options])},
        current_positions=[existing],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 1
    selected = decision.selected_trades[0]
    assert selected.position_size == 7
    assert selected.allocated_capital == 2800.0
    assert existing.market_value + selected.allocated_capital <= 40_000.0


def test_adaptive_sizing_scales_down_in_drawdown() -> None:
    """Effective max contracts reduces when drawdown exceeds threshold."""
    brain = _brain(adaptive_sizing=True, max_daily_risk_pct=6.0)
    no_drawdown = brain._compute_adaptive_risk_limits(
        active_strategy_count=1,
        recent_win_rate=0.85,
        current_drawdown_pct=0.0,
    )
    moderate_drawdown = brain._compute_adaptive_risk_limits(
        active_strategy_count=1,
        recent_win_rate=0.85,
        current_drawdown_pct=2.0,
    )

    assert no_drawdown["max_contracts"] == 10
    assert moderate_drawdown["max_contracts"] == 9
    assert moderate_drawdown["max_daily_risk_pct"] < no_drawdown["max_daily_risk_pct"]


def test_brain_strategy_specific_thresholds() -> None:
    brain = _brain(
        strategy_threshold_overrides={
            "momentum": {"min_confidence": 0.40, "min_rr": 2.0, "min_edge": 0.15},
            "gap_reversal": {"min_confidence": 0.45, "min_rr": 0.5, "min_edge": 0.10},
        }
    )

    momentum = _candidate(symbol="MOMO", strategy_name="momentum")
    gap = _candidate(symbol="GAP", strategy_name="gap_reversal")

    assert brain._required_confidence(momentum, 0.60) == 0.40
    assert brain._required_rr(momentum) == 2.0
    assert brain._required_edge(momentum) == 0.15

    assert brain._required_confidence(gap, 0.60) == 0.45
    assert brain._required_rr(gap) == 0.5
    assert brain._required_edge(gap) == 0.10


def test_momentum_passes_brain_at_lower_threshold() -> None:
    brain = _brain(
        min_confidence=0.60,
        strategy_threshold_overrides={
            "momentum": {"min_confidence": 0.40, "min_rr": 2.0, "min_edge": 0.15},
        },
    )
    momentum = _candidate(
        symbol="MOMO",
        strategy_name="momentum",
        confidence=0.45,
        rr=2.1,
        edge=1.0,
        risk_dollars=500.0,
    )

    decision = brain.decide(
        regime=_regime(),
        assessments={"momentum": OpportunityAssessment(candidates=[momentum])},
        current_positions=[],
        daily_pnl=0.0,
    )

    assert decision.num_trades == 1
    assert decision.selected_trades[0].candidate.symbol == "MOMO"
