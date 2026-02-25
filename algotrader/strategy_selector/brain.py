"""Daily Brain decision engine for concentrated trade selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pytz
import structlog

from algotrader.core.models import MarketRegime, Position, RegimeType
from algotrader.intelligence.calendar.events import EventCalendar
from algotrader.strategies.base import OpportunityAssessment
from algotrader.strategy_selector.candidate import CandidateType, TradeCandidate
from algotrader.tracking.journal import TradeJournal

logger = structlog.get_logger()


@dataclass
class TradeSelection:
    """A selected trade with capital allocation."""

    candidate: TradeCandidate
    allocated_capital: float
    position_size: int
    risk_amount: float
    brain_score: float


@dataclass
class TradeRejection:
    """A rejected candidate with reason."""

    candidate: TradeCandidate
    reason: str
    brain_score: float = 0.0


@dataclass
class BrainDecision:
    """The Brain's complete decision for a trading session."""

    timestamp: datetime
    regime: MarketRegime
    selected_trades: list[TradeSelection]
    rejected_trades: list[TradeRejection]
    cash_pct: float
    total_risk_pct: float
    reasoning: str

    @property
    def is_cash_day(self) -> bool:
        return len(self.selected_trades) == 0

    @property
    def num_trades(self) -> int:
        return len(self.selected_trades)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime.model_dump(),
            "selected_trades": [
                {
                    "candidate": _candidate_to_dict(sel.candidate),
                    "allocated_capital": round(sel.allocated_capital, 2),
                    "position_size": sel.position_size,
                    "risk_amount": round(sel.risk_amount, 2),
                    "brain_score": round(sel.brain_score, 4),
                }
                for sel in self.selected_trades
            ],
            "rejected_trades": [
                {
                    "candidate": _candidate_to_dict(rej.candidate),
                    "reason": rej.reason,
                    "brain_score": round(rej.brain_score, 4),
                }
                for rej in self.rejected_trades
            ],
            "cash_pct": round(self.cash_pct, 2),
            "total_risk_pct": round(self.total_risk_pct, 4),
            "reasoning": self.reasoning,
            "is_cash_day": self.is_cash_day,
            "num_trades": self.num_trades,
        }


def _candidate_to_dict(candidate: TradeCandidate) -> dict[str, Any]:
    return {
        "strategy_name": candidate.strategy_name,
        "candidate_type": candidate.candidate_type.value,
        "symbol": candidate.symbol,
        "direction": candidate.direction,
        "entry_price": candidate.entry_price,
        "stop_price": candidate.stop_price,
        "target_price": candidate.target_price,
        "risk_dollars": candidate.risk_dollars,
        "suggested_qty": candidate.suggested_qty,
        "risk_reward_ratio": candidate.risk_reward_ratio,
        "confidence": candidate.confidence,
        "edge_estimate_pct": candidate.edge_estimate_pct,
        "regime_fit": candidate.regime_fit,
        "catalyst": candidate.catalyst,
        "time_horizon_minutes": candidate.time_horizon_minutes,
        "expiry_time": candidate.expiry_time.isoformat() if candidate.expiry_time else None,
        "options_structure": candidate.options_structure,
        "short_strike": candidate.short_strike,
        "long_strike": candidate.long_strike,
        "contracts": candidate.contracts,
        "credit_received": candidate.credit_received,
        "max_loss": candidate.max_loss,
        "symbol_b": candidate.symbol_b,
        "hedge_ratio": candidate.hedge_ratio,
        "z_score": candidate.z_score,
        "timestamp": candidate.timestamp.isoformat(),
        "metadata": candidate.metadata,
        "expected_value": round(candidate.expected_value, 6),
    }


class DailyBrain:
    """Daily decision engine that selects the highest-conviction trades."""

    def __init__(
        self,
        total_capital: float,
        regime_config: dict | None = None,
        trade_journal: TradeJournal | None = None,
        event_calendar: EventCalendar | None = None,
        min_confidence: float = 0.60,
        min_risk_reward: float = 1.5,
        min_edge_pct: float = 0.3,
        max_daily_trades: int = 5,
        max_capital_per_trade_pct: float = 20.0,
        max_daily_risk_pct: float = 2.0,
        cash_is_default: bool = True,
        regime_mismatch_penalty: float = 0.5,
        correlation_penalty: float = 0.3,
        recent_loss_cooldown_hours: int = 4,
        midday_confidence_multiplier: float = 1.2,
        midday_pnl_stop_pct: float = -1.0,
    ) -> None:
        self._log = logger.bind(component="daily_brain")
        self._total_capital = total_capital
        self._trade_journal = trade_journal or TradeJournal()
        self._event_calendar = event_calendar or EventCalendar()
        regime_config = regime_config or {}

        self._min_confidence = min_confidence
        self._min_risk_reward = min_risk_reward
        self._min_edge_pct = min_edge_pct
        self._max_daily_trades = max_daily_trades
        self._max_capital_per_trade_pct = max_capital_per_trade_pct
        self._max_daily_risk_pct = max_daily_risk_pct
        self._cash_is_default = cash_is_default
        self._regime_mismatch_penalty = regime_mismatch_penalty
        self._correlation_penalty = correlation_penalty
        self._recent_loss_cooldown_hours = recent_loss_cooldown_hours
        self._midday_conf_multiplier = midday_confidence_multiplier
        self._midday_pnl_stop_pct = midday_pnl_stop_pct

        self._regime_weights = regime_config.get("regime_weights", {})
        self._candidate_bonuses = regime_config.get("candidate_bonuses", {})

        self._log.info(
            "daily_brain_initialized",
            min_confidence=min_confidence,
            min_rr=min_risk_reward,
            min_edge=min_edge_pct,
            max_daily_trades=max_daily_trades,
            max_daily_risk_pct=max_daily_risk_pct,
        )

    def decide(
        self,
        regime: MarketRegime | None,
        assessments: dict[str, OpportunityAssessment],
        current_positions: list[Position],
        daily_pnl: float,
        vix_level: float | None = None,
    ) -> BrainDecision:
        """Make primary daily trading decisions."""
        normalized_regime = self._normalize_regime(regime)
        return self._decide_impl(
            regime=normalized_regime,
            assessments=assessments,
            current_positions=current_positions,
            daily_pnl=daily_pnl,
            vix_level=vix_level,
            min_confidence=self._min_confidence,
        )

    def review_midday(
        self,
        regime: MarketRegime | None,
        assessments: dict[str, OpportunityAssessment],
        current_positions: list[Position],
        daily_pnl: float,
        vix_level: float | None = None,
    ) -> BrainDecision:
        """Run midday review with stricter entry criteria."""
        normalized_regime = self._normalize_regime(regime)
        pnl_pct = (daily_pnl / self._total_capital * 100.0) if self._total_capital > 0 else 0.0
        close_recommendations: list[TradeRejection] = []

        for pos in current_positions:
            synthetic = self._candidate_from_position(pos)
            should_close, reason = self.should_close_early(synthetic, pos, normalized_regime)
            if should_close:
                close_recommendations.append(
                    TradeRejection(candidate=synthetic, reason=f"close_early:{reason}")
                )

        if pnl_pct <= self._midday_pnl_stop_pct:
            reasoning = (
                f"Midday review halted: daily P&L {pnl_pct:.2f}% <= stop "
                f"{self._midday_pnl_stop_pct:.2f}%. No new entries."
            )
            return BrainDecision(
                timestamp=datetime.now(pytz.UTC),
                regime=normalized_regime,
                selected_trades=[],
                rejected_trades=close_recommendations,
                cash_pct=100.0,
                total_risk_pct=0.0,
                reasoning=reasoning,
            )

        decision = self._decide_impl(
            regime=normalized_regime,
            assessments=assessments,
            current_positions=current_positions,
            daily_pnl=daily_pnl,
            vix_level=vix_level,
            min_confidence=min(1.0, self._min_confidence * self._midday_conf_multiplier),
        )
        if close_recommendations:
            decision.rejected_trades.extend(close_recommendations)
            decision.reasoning = (
                f"{decision.reasoning} Close recommendations: "
                f"{', '.join(r.candidate.symbol for r in close_recommendations)}."
            )
        return decision

    def should_close_early(
        self,
        candidate: TradeCandidate,
        position: Position,
        regime: MarketRegime | None,
    ) -> tuple[bool, str]:
        """Return whether a position should be closed before natural exit."""
        normalized_regime = self._normalize_regime(regime)
        regime_fit = candidate.regime_fit
        if regime_fit <= 0:
            regime_fit = self._get_regime_fit(candidate.strategy_name, normalized_regime)

        if regime_fit < 0.25:
            return True, "regime_mismatch"

        if position.unrealized_pnl_pct <= -1.0:
            return True, "losing_position"

        if self._event_calendar.is_event_day():
            return True, "high_impact_event_day"

        return False, ""

    def _decide_impl(
        self,
        regime: MarketRegime,
        assessments: dict[str, OpportunityAssessment],
        current_positions: list[Position],
        daily_pnl: float,
        vix_level: float | None,
        min_confidence: float,
    ) -> BrainDecision:
        all_candidates, classic_fallback = self._collect_candidates(regime, assessments)
        rejected: list[TradeRejection] = []
        scored: list[tuple[TradeCandidate, float]] = []

        for candidate in all_candidates:
            reason = self._reject_reason(
                candidate=candidate,
                min_confidence=min_confidence,
                current_positions=current_positions,
            )
            if reason:
                rejected.append(TradeRejection(candidate=candidate, reason=reason))
                continue

            score = self._score_candidate(
                candidate=candidate,
                regime=regime,
                current_positions=current_positions,
                daily_pnl=daily_pnl,
                vix_level=vix_level,
            )
            scored.append((candidate, score))

        scored.sort(key=lambda item: item[1], reverse=True)

        selected: list[TradeSelection] = []
        total_risk_dollars = 0.0
        max_daily_risk_dollars = self._total_capital * (self._max_daily_risk_pct / 100.0)
        max_trade_capital = self._total_capital * (self._max_capital_per_trade_pct / 100.0)

        for candidate, score in scored:
            if len(selected) >= self._max_daily_trades:
                break

            risk_amount, position_size, allocated_capital = self._allocation_for_candidate(candidate)

            if total_risk_dollars + risk_amount > max_daily_risk_dollars:
                rejected.append(
                    TradeRejection(candidate=candidate, reason="daily_risk_limit", brain_score=score)
                )
                continue

            if allocated_capital > max_trade_capital:
                rejected.append(
                    TradeRejection(candidate=candidate, reason="single_trade_too_large", brain_score=score)
                )
                continue

            corr_with_selected = self._compute_correlation(
                candidate, [s.candidate for s in selected],
            )
            if corr_with_selected >= 0.6:
                rejected.append(
                    TradeRejection(
                        candidate=candidate,
                        reason="correlated_with_selected",
                        brain_score=score,
                    )
                )
                continue

            selected.append(
                TradeSelection(
                    candidate=candidate,
                    allocated_capital=allocated_capital,
                    position_size=position_size,
                    risk_amount=risk_amount,
                    brain_score=score,
                )
            )
            total_risk_dollars += risk_amount

        total_allocated = sum(s.allocated_capital for s in selected)
        cash_pct = 100.0
        if self._total_capital > 0:
            cash_pct = max(0.0, 100.0 - (total_allocated / self._total_capital * 100.0))

        reasoning = self._build_reasoning(
            regime=regime,
            all_candidates=all_candidates,
            selected=selected,
            rejected=rejected,
            vix_level=vix_level,
            classic_fallback=classic_fallback,
        )

        if not selected and self._cash_is_default:
            self._log.info("brain_cash_day", reason=reasoning)

        total_risk_pct = (total_risk_dollars / self._total_capital * 100.0) if self._total_capital > 0 else 0.0
        decision = BrainDecision(
            timestamp=datetime.now(pytz.UTC),
            regime=regime,
            selected_trades=selected,
            rejected_trades=rejected,
            cash_pct=round(cash_pct, 2),
            total_risk_pct=round(total_risk_pct, 4),
            reasoning=reasoning,
        )
        return decision

    def _collect_candidates(
        self,
        regime: MarketRegime,
        assessments: dict[str, OpportunityAssessment],
    ) -> tuple[list[TradeCandidate], list[str]]:
        candidates: list[TradeCandidate] = []
        classic_fallback: list[str] = []

        for strategy_name, assessment in assessments.items():
            if assessment.candidates:
                for candidate in assessment.candidates:
                    if candidate.regime_fit <= 0:
                        candidate.regime_fit = self._get_regime_fit(strategy_name, regime)
                    candidates.append(candidate)
                continue

            if not assessment.has_opportunities:
                continue

            classic_fallback.append(strategy_name)
            synthetic = TradeCandidate(
                strategy_name=strategy_name,
                candidate_type=self._infer_type(strategy_name),
                symbol=self._infer_primary_symbol(strategy_name, assessment),
                direction=self._infer_direction(strategy_name, assessment),
                entry_price=0.0,
                stop_price=0.0,
                target_price=0.0,
                risk_dollars=0.0,
                confidence=assessment.confidence,
                risk_reward_ratio=assessment.avg_risk_reward,
                edge_estimate_pct=assessment.estimated_edge_pct,
                regime_fit=self._get_regime_fit(strategy_name, regime),
                catalyst=f"aggregate_{strategy_name}",
                metadata={"synthetic": True, "details": assessment.details},
            )
            candidates.append(synthetic)

        return candidates, classic_fallback

    def _reject_reason(
        self,
        candidate: TradeCandidate,
        min_confidence: float,
        current_positions: list[Position],
    ) -> str | None:
        if candidate.confidence < min_confidence:
            return "low_confidence"
        effective_rr = self._effective_rr(candidate)
        if effective_rr < self._required_rr(candidate):
            return "poor_rr"
        if candidate.edge_estimate_pct < self._min_edge_pct:
            return "insufficient_edge"
        if candidate.is_expired:
            return "expired"
        if self._already_positioned(candidate, current_positions):
            return "already_positioned"
        if self._in_recent_loss_cooldown(candidate):
            return "cooldown"
        return None

    def _score_candidate(
        self,
        candidate: TradeCandidate,
        regime: MarketRegime,
        current_positions: list[Position],
        daily_pnl: float,
        vix_level: float | None,
    ) -> float:
        effective_confidence = self._apply_candidate_bonus(
            candidate.confidence,
            candidate.strategy_name,
            regime,
        )
        effective_rr = self._effective_rr(candidate)
        rr_divisor = 2.0 if candidate.is_options else 4.0
        rr_component = min(effective_rr / rr_divisor, 1.0) * 0.25
        base_score = (
            effective_confidence * 0.35
            + rr_component
            + min(candidate.edge_estimate_pct / 2.0, 1.0) * 0.25
            + candidate.regime_fit * 0.15
        )

        if candidate.regime_fit < 0.35:
            base_score *= self._regime_mismatch_penalty

        if regime.regime_type == RegimeType.EVENT_DAY and candidate.strategy_name == "options_premium":
            base_score *= 0.7

        corr = self._compute_correlation(candidate, current_positions)
        if corr > 0:
            base_score *= max(0.0, 1.0 - self._correlation_penalty * corr)

        if self._strategy_losing_today(candidate.strategy_name):
            base_score *= 0.7

        if self._is_strong_catalyst(candidate.catalyst):
            base_score *= 1.15

        if vix_level is not None and vix_level >= 20 and candidate.is_options:
            base_score *= 1.1

        if daily_pnl < 0:
            dd_penalty = max(0.75, 1.0 - abs(daily_pnl) / max(self._total_capital, 1))
            base_score *= dd_penalty

        return round(max(0.0, min(1.0, base_score)), 4)

    def _required_rr(self, candidate: TradeCandidate) -> float:
        if candidate.is_options:
            return max(0.5, self._min_risk_reward * 0.35)
        return self._min_risk_reward

    def _effective_rr(self, candidate: TradeCandidate) -> float:
        rr = max(0.0, candidate.risk_reward_ratio)
        if not candidate.is_options:
            return rr

        max_loss = candidate.max_loss
        credit = candidate.credit_received
        if max_loss <= 0 or credit <= 0:
            return rr

        raw_win_rate = candidate.metadata.get("win_rate") if candidate.metadata else None
        if isinstance(raw_win_rate, (int, float)):
            win_rate = float(raw_win_rate)
        else:
            # Fallback to confidence if strategy doesn't pass explicit win rate.
            win_rate = min(0.9, max(0.5, candidate.confidence))

        win_rate = min(0.99, max(0.01, win_rate))
        loss_rate = max(1e-6, 1.0 - win_rate)
        adjusted_rr = (win_rate * credit) / (loss_rate * max_loss)
        return max(0.0, adjusted_rr)

    def _allocation_for_candidate(self, candidate: TradeCandidate) -> tuple[float, int, float]:
        max_trade_capital = self._total_capital * (self._max_capital_per_trade_pct / 100.0)
        risk_amount = self._compute_trade_risk(candidate)

        if candidate.suggested_qty > 0:
            position_size = candidate.suggested_qty
        elif candidate.entry_price > 0:
            risk_per_unit = abs(candidate.entry_price - candidate.stop_price)
            if risk_per_unit > 0 and risk_amount > 0:
                position_size = max(1, int(risk_amount / risk_per_unit))
            else:
                position_size = max(1, int((max_trade_capital * 0.5) / candidate.entry_price))
        else:
            position_size = max(1, candidate.contracts) if candidate.is_options else 1

        if candidate.entry_price > 0:
            allocated_capital = candidate.entry_price * position_size
        elif candidate.is_options and candidate.max_loss > 0:
            allocated_capital = candidate.max_loss * max(candidate.contracts, 1)
        else:
            allocated_capital = max(risk_amount, max_trade_capital * 0.25)

        allocated_capital = min(allocated_capital, max_trade_capital)
        risk_amount = min(risk_amount, allocated_capital)
        return float(risk_amount), int(position_size), float(allocated_capital)

    def _compute_trade_risk(self, candidate: TradeCandidate) -> float:
        if candidate.risk_dollars > 0:
            return candidate.risk_dollars

        if candidate.is_options and candidate.max_loss > 0:
            contracts = max(candidate.contracts, 1)
            return candidate.max_loss * contracts

        risk_per_unit = abs(candidate.entry_price - candidate.stop_price)
        if risk_per_unit > 0:
            if candidate.suggested_qty > 0:
                return risk_per_unit * candidate.suggested_qty

            fallback_qty = 1
            if candidate.entry_price > 0 and self._total_capital > 0:
                target_notional = self._total_capital * min(
                    self._max_capital_per_trade_pct / 100.0,
                    0.05,
                )
                fallback_qty = max(1, int(target_notional / candidate.entry_price))
            return risk_per_unit * fallback_qty

        return self._total_capital * 0.0025

    def _already_positioned(
        self,
        candidate: TradeCandidate,
        current_positions: list[Position],
    ) -> bool:
        candidate_symbols = {candidate.symbol}
        if candidate.symbol_b:
            candidate_symbols.add(candidate.symbol_b)
        if "/" in candidate.symbol:
            candidate_symbols.update(part.strip() for part in candidate.symbol.split("/"))

        for pos in current_positions:
            if pos.symbol not in candidate_symbols:
                continue
            if not pos.strategy_name:
                return True

            owners = {s.strip() for s in pos.strategy_name.split("+") if s.strip()}
            if candidate.strategy_name not in owners:
                return True
        return False

    def _in_recent_loss_cooldown(self, candidate: TradeCandidate) -> bool:
        try:
            trades = self._trade_journal.get_trades(
                strategy_name=candidate.strategy_name,
                symbol=candidate.symbol,
                limit=30,
            )
        except Exception:
            return False

        now = datetime.now(pytz.UTC)
        cooldown_delta = timedelta(hours=self._recent_loss_cooldown_hours)
        for trade in trades:
            if trade.realized_pnl >= 0:
                continue
            if not self._same_direction(candidate, trade.side.value):
                continue
            trade_time = trade.exit_time or trade.entry_time
            if trade_time is None:
                continue
            if trade_time.tzinfo is None:
                trade_time = trade_time.replace(tzinfo=pytz.UTC)
            if now - trade_time <= cooldown_delta:
                return True
        return False

    def _same_direction(self, candidate: TradeCandidate, side_value: str) -> bool:
        if candidate.direction == "long":
            return side_value == "buy"
        if candidate.direction == "short":
            return side_value == "sell"
        return True

    def _strategy_losing_today(self, strategy_name: str) -> bool:
        try:
            summary = self._trade_journal.get_strategy_summary(strategy_name, days=1)
            return summary["total_trades"] > 0 and summary["total_pnl"] < 0
        except Exception:
            return False

    def _is_strong_catalyst(self, catalyst: str) -> bool:
        text = (catalyst or "").lower()
        strong_tokens = ("news", "volume", "breakout", "event", "earnings", "spike")
        return any(token in text for token in strong_tokens)

    def _apply_candidate_bonus(
        self,
        confidence: float,
        strategy_name: str,
        regime: MarketRegime,
    ) -> float:
        regime_key = regime.regime_type.value
        regime_bonus = self._candidate_bonuses.get(regime_key, {})
        bonus = regime_bonus.get(strategy_name, 0.0)
        return max(0.0, min(1.0, confidence + bonus))

    def _infer_type(self, strategy_name: str) -> CandidateType:
        mapping = {
            "pairs_trading": CandidateType.PAIRS,
            "options_premium": CandidateType.CREDIT_SPREAD,
            "sector_rotation": CandidateType.SECTOR_LONG_SHORT,
            "event_driven": CandidateType.EVENT_DIRECTIONAL,
            "momentum": CandidateType.LONG_EQUITY,
            "gap_reversal": CandidateType.LONG_EQUITY,
            "vwap_reversion": CandidateType.LONG_EQUITY,
        }
        return mapping.get(strategy_name, CandidateType.LONG_EQUITY)

    def _infer_primary_symbol(
        self,
        strategy_name: str,
        assessment: OpportunityAssessment,
    ) -> str:
        if assessment.details:
            first = assessment.details[0]
            symbol = first.get("symbol")
            if symbol:
                return str(symbol)
            pair_id = first.get("pair")
            if pair_id:
                return str(pair_id).replace("_", "/")
            event_symbol = first.get("instrument")
            if event_symbol:
                return str(event_symbol)

        defaults = {
            "options_premium": "SPY",
            "pairs_trading": "XOM/CVX",
            "event_driven": "SPY",
            "sector_rotation": "XLK",
            "vwap_reversion": "SPY",
            "gap_reversal": "SPY",
            "momentum": "SPY",
        }
        return defaults.get(strategy_name, "SPY")

    def _infer_direction(
        self,
        strategy_name: str,
        assessment: OpportunityAssessment,
    ) -> str:
        if assessment.details:
            first = assessment.details[0]
            side = first.get("side") or first.get("direction")
            if isinstance(side, str):
                side_l = side.lower()
                if side_l in ("long", "short", "neutral"):
                    return side_l

        if strategy_name == "options_premium":
            return "neutral"
        return "long"

    def _get_regime_fit(self, strategy_name: str, regime: MarketRegime) -> float:
        regime_key = regime.regime_type.value
        return float(self._regime_weights.get(regime_key, {}).get(strategy_name, 0.0))

    def _normalize_regime(self, regime: MarketRegime | None) -> MarketRegime:
        """Provide a safe default regime when callers pass None."""
        if regime is not None:
            return regime
        return MarketRegime(
            regime_type=RegimeType.RANGING,
            confidence=0.0,
            timestamp=datetime.now(pytz.UTC),
        )

    def _candidate_from_position(self, position: Position) -> TradeCandidate:
        strategy_name = position.strategy_name.split("+")[0] if position.strategy_name else "unknown"
        direction = "long" if position.side.value == "buy" else "short"
        ctype = CandidateType.LONG_EQUITY if direction == "long" else CandidateType.SHORT_EQUITY
        return TradeCandidate(
            strategy_name=strategy_name,
            candidate_type=ctype,
            symbol=position.symbol,
            direction=direction,
            entry_price=position.avg_entry_price,
            stop_price=position.avg_entry_price,
            target_price=position.avg_entry_price,
            risk_dollars=abs(position.unrealized_pnl),
            confidence=1.0,
            risk_reward_ratio=1.0,
            edge_estimate_pct=0.0,
            regime_fit=0.0,
            catalyst="open_position",
            metadata={"from_position": True},
        )

    def _compute_correlation(
        self,
        candidate: TradeCandidate,
        existing: list[TradeCandidate | Position],
    ) -> float:
        """Estimate heuristic correlation against existing exposure."""
        if not existing:
            return 0.0

        max_corr = -1.0
        for item in existing:
            if isinstance(item, Position):
                item_symbol = item.symbol
                item_direction = "long" if item.side.value == "buy" else "short"
                item_type = CandidateType.LONG_EQUITY
                item_sector = ""
            else:
                item_symbol = item.symbol
                item_direction = item.direction
                item_type = item.candidate_type
                item_sector = str(item.metadata.get("sector", ""))

            if candidate.symbol == item_symbol:
                max_corr = max(max_corr, 1.0)
                continue

            if (
                candidate.metadata.get("sector")
                and item_sector
                and candidate.metadata.get("sector") == item_sector
            ):
                max_corr = max(max_corr, 0.6)
                continue

            if (
                candidate.candidate_type in (CandidateType.LONG_EQUITY, CandidateType.SHORT_EQUITY)
                and item_type in (CandidateType.LONG_EQUITY, CandidateType.SHORT_EQUITY)
            ):
                if candidate.direction == item_direction:
                    max_corr = max(max_corr, 0.4)
                else:
                    max_corr = max(max_corr, -0.2)
                continue

            if (
                candidate.is_options
                and item_type in (CandidateType.LONG_EQUITY, CandidateType.SHORT_EQUITY)
            ) or (
                item_type in (CandidateType.CREDIT_SPREAD, CandidateType.IRON_CONDOR)
                and candidate.candidate_type in (CandidateType.LONG_EQUITY, CandidateType.SHORT_EQUITY)
            ):
                max_corr = max(max_corr, 0.2)
                continue

            if (
                candidate.candidate_type == CandidateType.PAIRS
                and item_type
                in (
                    CandidateType.LONG_EQUITY,
                    CandidateType.SHORT_EQUITY,
                    CandidateType.SECTOR_LONG_SHORT,
                    CandidateType.EVENT_DIRECTIONAL,
                )
            ):
                max_corr = max(max_corr, 0.1)
                continue

            if (
                item_type == CandidateType.PAIRS
                and candidate.candidate_type
                in (
                    CandidateType.LONG_EQUITY,
                    CandidateType.SHORT_EQUITY,
                    CandidateType.SECTOR_LONG_SHORT,
                    CandidateType.EVENT_DIRECTIONAL,
                )
            ):
                max_corr = max(max_corr, 0.1)
                continue

            max_corr = max(max_corr, 0.0)

        return max_corr if max_corr >= 0 else 0.0

    def _build_reasoning(
        self,
        regime: MarketRegime,
        all_candidates: list[TradeCandidate],
        selected: list[TradeSelection],
        rejected: list[TradeRejection],
        vix_level: float | None,
        classic_fallback: list[str],
    ) -> str:
        regime_txt = regime.regime_type.value
        vix_txt = f", VIX {vix_level:.1f}" if vix_level is not None else ""
        summary = [
            f"{regime_txt} regime{vix_txt}. Evaluated {len(all_candidates)} candidates.",
        ]

        if classic_fallback:
            summary.append(
                "Fallback synthetic candidates used for: "
                + ", ".join(sorted(classic_fallback))
                + "."
            )

        if selected:
            picks = ", ".join(
                f"{s.candidate.symbol} ({s.candidate.strategy_name}, score={s.brain_score:.2f})"
                for s in selected
            )
            summary.append(f"Selected {len(selected)}: {picks}.")
        else:
            summary.append("None met thresholds. Sitting 100% cash. No trade is better than a bad trade.")

        if rejected:
            counts: dict[str, int] = {}
            for rej in rejected:
                counts[rej.reason] = counts.get(rej.reason, 0) + 1
            rejected_txt = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            summary.append(f"Rejected {len(rejected)}: {rejected_txt}.")

        deployed = sum(s.allocated_capital for s in selected)
        deployed_pct = (deployed / self._total_capital * 100.0) if self._total_capital > 0 else 0.0
        summary.append(f"Deploying {deployed_pct:.1f}% of capital, {100 - deployed_pct:.1f}% cash.")
        return " ".join(summary)
