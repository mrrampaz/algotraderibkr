"""Score strategies against current regime and market conditions.

Scoring formula (multiplicative EV):
  opp_quality = 0.3*(candidates/3) + 0.3*(rr/3) + 0.4*confidence  (0 if no opps)
  modifiers = vix_mod + perf_mod + time_mod + event_mod
  total_score = clamp(base_weight * opp_quality * (1 + modifiers), 0.0, 1.0)
  is_active = total_score >= min_activation_score
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytz
import structlog

from algotrader.core.models import MarketRegime
from algotrader.intelligence.calendar.events import EventCalendar
from algotrader.strategies.base import OpportunityAssessment
from algotrader.strategy_selector.candidate import TradeCandidate
from algotrader.tracking.journal import TradeJournal

logger = structlog.get_logger()


@dataclass
class StrategyScore:
    """Scored strategy with breakdown."""

    strategy_name: str
    total_score: float          # Final score 0.0-1.0
    base_weight: float          # From regime_weights matrix
    vix_modifier: float         # From VIX level
    performance_modifier: float  # From recent P&L
    time_modifier: float        # From time of day
    event_modifier: float       # From event proximity
    opportunity_score: float    # From OpportunityAssessment quality
    is_active: bool             # True if score >= min_activation_score
    assessment: OpportunityAssessment | None = None


@dataclass
class CandidateScore:
    """Brain-style score for a single trade candidate."""

    candidate: TradeCandidate
    score: float
    rejected: bool = False
    reason: str = ""


class StrategyScorer:
    """Score strategies against current regime and market conditions.

    Uses a regime-strategy weight matrix (from regimes.yaml) as the base score,
    then applies modifiers for VIX level, recent performance, time of day,
    and event proximity.
    """

    def __init__(
        self,
        regime_config: dict,
        trade_journal: TradeJournal,
        event_calendar: EventCalendar,
        total_capital: float = 60000.0,
    ) -> None:
        self._log = logger.bind(component="strategy_scorer")
        self._trade_journal = trade_journal
        self._event_calendar = event_calendar
        self._total_capital = total_capital

        # Parse config
        self._regime_weights: dict[str, dict[str, float]] = regime_config.get("regime_weights", {})
        self._min_activation_score: float = regime_config.get("min_activation_score", 0.35)

        modifiers = regime_config.get("modifiers", {})
        self._vix_high_boost: dict[str, float] = modifiers.get("vix_high_boost", {})
        self._vix_low_penalty: dict[str, float] = modifiers.get("vix_low_penalty", {})
        self._morning_session: dict[str, float] = modifiers.get("morning_session", {})
        self._afternoon_session: dict[str, float] = modifiers.get("afternoon_session", {})
        self._pre_event_day: dict[str, float] = modifiers.get("pre_event_day", {})

        self._perf_lookback_days: int = modifiers.get("performance_lookback_days", 5)
        self._perf_boost_per_pct: float = modifiers.get("performance_boost_per_pct", 0.02)
        self._perf_penalty_per_pct: float = modifiers.get("performance_penalty_per_pct", 0.03)
        self._max_perf_modifier: float = modifiers.get("max_performance_modifier", 0.15)

        self._log.info(
            "scorer_initialized",
            regimes=list(self._regime_weights.keys()),
            min_score=self._min_activation_score,
        )

    def score_strategies(
        self,
        strategy_names: list[str],
        regime: MarketRegime,
        vix_level: float | None = None,
        assessments: dict[str, OpportunityAssessment] | None = None,
    ) -> list[StrategyScore]:
        """Score all strategies. Returns sorted by score (highest first)."""
        scores = [
            self.score_single(
                name, regime, vix_level,
                assessment=assessments.get(name) if assessments else None,
            )
            for name in strategy_names
        ]
        scores.sort(key=lambda s: s.total_score, reverse=True)

        active_count = sum(1 for s in scores if s.is_active)
        self._log.info(
            "strategies_scored",
            regime=regime.regime_type.value,
            total=len(scores),
            active=active_count,
            top_strategy=scores[0].strategy_name if scores else "none",
            top_score=round(scores[0].total_score, 3) if scores else 0,
        )

        return scores

    def score_single(
        self,
        strategy_name: str,
        regime: MarketRegime,
        vix_level: float | None = None,
        assessment: OpportunityAssessment | None = None,
    ) -> StrategyScore:
        """Score a single strategy using multiplicative EV formula."""
        # 1. Base weight from regime matrix
        regime_key = regime.regime_type.value
        regime_weights = self._regime_weights.get(regime_key, {})
        base_weight = regime_weights.get(strategy_name, 0.0)

        # 2. Opportunity quality from assessment
        if assessment and assessment.has_opportunities:
            opp_quality = (
                0.3 * min(assessment.num_candidates / 3.0, 1.0)
                + 0.3 * min(assessment.avg_risk_reward / 3.0, 1.0)
                + 0.4 * assessment.confidence
            )
        else:
            opp_quality = 0.0  # Zero opportunities = zero score

        # 3. VIX modifier
        vix_mod = self._calc_vix_modifier(strategy_name, vix_level)

        # 4. Performance modifier
        perf_mod = self._calc_performance_modifier(strategy_name)

        # 5. Time-of-day modifier
        time_mod = self._calc_time_modifier(strategy_name)

        # 6. Event proximity modifier
        event_mod = self._calc_event_modifier(strategy_name)

        # 7. Multiplicative EV score (clamped 0.0-1.0)
        modifiers_sum = vix_mod + perf_mod + time_mod + event_mod
        total = max(0.0, min(1.0, base_weight * opp_quality * (1 + modifiers_sum)))
        is_active = total >= self._min_activation_score

        return StrategyScore(
            strategy_name=strategy_name,
            total_score=round(total, 4),
            base_weight=base_weight,
            vix_modifier=round(vix_mod, 4),
            performance_modifier=round(perf_mod, 4),
            time_modifier=round(time_mod, 4),
            event_modifier=round(event_mod, 4),
            opportunity_score=round(opp_quality, 4),
            is_active=is_active,
            assessment=assessment,
        )

    def _calc_vix_modifier(self, strategy_name: str, vix_level: float | None) -> float:
        """Calculate VIX-based score modifier."""
        if vix_level is None:
            return 0.0
        if vix_level >= 25:
            return self._vix_high_boost.get(strategy_name, 0.0)
        elif vix_level <= 13:
            return self._vix_low_penalty.get(strategy_name, 0.0)
        return 0.0

    def _calc_performance_modifier(self, strategy_name: str) -> float:
        """Calculate performance-based score modifier from recent trade history."""
        try:
            summary = self._trade_journal.get_strategy_summary(
                strategy_name, days=self._perf_lookback_days
            )
        except Exception:
            self._log.debug("performance_lookup_failed", strategy=strategy_name)
            return 0.0

        if summary["total_trades"] < 3:
            return 0.0  # Not enough data

        total_pnl = summary["total_pnl"]
        if self._total_capital <= 0:
            return 0.0

        return_pct = total_pnl / self._total_capital * 100

        if return_pct > 0:
            mod = min(return_pct * self._perf_boost_per_pct, self._max_perf_modifier)
        else:
            mod = max(return_pct * self._perf_penalty_per_pct, -self._max_perf_modifier)

        return mod

    def _calc_time_modifier(self, strategy_name: str) -> float:
        """Calculate time-of-day score modifier."""
        et_now = datetime.now(pytz.timezone("America/New_York"))
        hour, minute = et_now.hour, et_now.minute
        time_decimal = hour + minute / 60.0

        # Morning session: 9:30-11:00 AM
        if 9.5 <= time_decimal < 11.0:
            return self._morning_session.get(strategy_name, 0.0)
        # Afternoon session: 2:00-3:30 PM
        elif 14.0 <= time_decimal < 15.5:
            return self._afternoon_session.get(strategy_name, 0.0)

        return 0.0

    def _calc_event_modifier(self, strategy_name: str) -> float:
        """Calculate event proximity score modifier."""
        et_now = datetime.now(pytz.timezone("America/New_York"))
        tomorrow = (et_now + timedelta(days=1)).date()

        # Check if tomorrow has a high-impact event
        tomorrow_events = self._event_calendar.get_events_for_date(tomorrow)
        has_high_impact_tomorrow = any(e.impact >= 3 for e in tomorrow_events)

        if has_high_impact_tomorrow:
            return self._pre_event_day.get(strategy_name, 0.0)

        return 0.0


def score_candidate_brain_style(
    candidate: TradeCandidate,
    regime_mismatch_penalty: float = 0.5,
    correlation_penalty: float = 0.3,
    correlation: float = 0.0,
    losing_strategy_today: bool = False,
    strong_catalyst: bool = False,
    elevated_vix_premium_selling: bool = False,
) -> float:
    """Shared brain-style candidate score formula used by DailyBrain."""
    score = (
        candidate.confidence * 0.35
        + min(candidate.risk_reward_ratio / 4.0, 1.0) * 0.25
        + min(candidate.edge_estimate_pct / 2.0, 1.0) * 0.25
        + candidate.regime_fit * 0.15
    )

    if candidate.regime_fit < 0.35:
        score *= regime_mismatch_penalty
    if correlation > 0:
        score *= max(0.0, 1.0 - (correlation_penalty * correlation))
    if losing_strategy_today:
        score *= 0.7
    if strong_catalyst:
        score *= 1.15
    if elevated_vix_premium_selling:
        score *= 1.1

    return round(max(0.0, min(1.0, score)), 4)
