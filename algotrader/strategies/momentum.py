"""Momentum / Breakout strategy.

Trades confirmed breakouts from consolidation ranges with volume
confirmation. Holds positions with trailing stops until they trigger.
Uses BreakoutScanner for candidate detection.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytz
import structlog

from algotrader.core.config import StrategyConfig
from algotrader.core.events import EventBus
from algotrader.core.models import (
    MarketRegime, Order, OrderSide, OrderType, Signal, SignalDirection,
    TimeInForce,
)
from algotrader.data.provider import DataProvider
from algotrader.execution.executor import Executor
from algotrader.strategies.base import OpportunityAssessment, StrategyBase
from algotrader.strategies.registry import register_strategy
from algotrader.strategy_selector.candidate import CandidateType, TradeCandidate

logger = structlog.get_logger()

ET = pytz.timezone("America/New_York")


@dataclass
class MomentumTrade:
    """Internal state for a single momentum/breakout trade."""

    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    stop_price: float
    initial_stop: float  # Never changes (for R calculation)
    target_price: float | None
    atr: float
    breakout_level: float
    volume_ratio: float
    entry_time: datetime
    capital_used: float = 0.0
    trailing_active: bool = False
    trail_stop: float = 0.0
    highest_price: float = 0.0  # For trailing (longs)
    lowest_price: float = 999999.0  # For trailing (shorts)
    trade_id: str = ""
    bracket_stop_order_id: str = ""
    bracket_tp_order_id: str = ""
    is_bracket: bool = False


@register_strategy("momentum")
class MomentumStrategy(StrategyBase):
    """Momentum / Breakout strategy.

    Scans for breakouts every N minutes, enters on confirmed breaks
    with volume, manages with trailing stops.
    """

    def __init__(
        self,
        name: str,
        config: StrategyConfig,
        data_provider: DataProvider,
        executor: Executor,
        event_bus: EventBus,
    ) -> None:
        super().__init__(name, config, data_provider, executor, event_bus)

        params = config.params
        self._min_volume_ratio = params.get("min_volume_ratio", 1.5)
        self._min_consolidation_days = params.get("min_consolidation_days", 3)
        self._max_spread_pct = params.get("max_spread_pct", 0.2)

        # Entry
        self._entry_buffer_pct = params.get("entry_buffer_pct", 0.1)

        # Stop
        self._stop_atr_mult = params.get("stop_atr_mult", 1.5)

        # Trailing
        self._use_trailing = params.get("use_trailing_stop", True)
        self._trail_atr_mult = params.get("trail_atr_mult", 2.0)
        self._trail_activation_rr = params.get("trail_activation_rr", 1.0)
        self._fixed_target_rr = params.get("fixed_target_rr", 3.0)

        # Time
        self._close_by_hour = params.get("close_by_hour", 15)
        self._close_by_minute = params.get("close_by_minute", 30)
        self._scan_interval_minutes = params.get("scan_interval_minutes", 15)
        self._bar_timeframe = params.get("bar_timeframe", "5Min")

        # Regime filter
        self._allowed_regimes = params.get(
            "allowed_regimes",
            ["trending_up", "trending_down", "high_vol"],
        )

        # Internal state
        self._trades: dict[str, MomentumTrade] = {}
        self._last_scan_time: datetime | None = None

    def warm_up(self) -> None:
        """Initialize — no heavy warm-up needed, scanner runs on demand."""
        self._log.info("warming_up")
        self._warmed_up = True

    def run_cycle(self, regime: MarketRegime | None = None) -> list[Signal]:
        """Run one cycle: manage positions, scan for breakouts."""
        self._check_day_reset()
        self._last_cycle_time = datetime.now(pytz.UTC)
        signals: list[Signal] = []

        et_now = datetime.now(ET)

        # 1. Manage existing positions FIRST (always, regardless of regime)
        signals.extend(self._manage_positions(et_now))

        # 2. Check regime filter (only gates new entries)
        if regime and regime.regime_type.value not in self._allowed_regimes:
            return signals

        # 3. Scan for new entries if scan interval elapsed
        if self._should_scan(et_now) and len(self._trades) < self.config.max_positions:
            signals.extend(self._scan_entries(et_now))

        return signals

    def _should_scan(self, et_now: datetime) -> bool:
        """Check if enough time has passed since last scan."""
        if self._last_scan_time is None:
            return True
        elapsed = (datetime.now(pytz.UTC) - self._last_scan_time).total_seconds() / 60
        return elapsed >= self._scan_interval_minutes

    def _manage_positions(self, et_now: datetime) -> list[Signal]:
        """Manage open momentum trades."""
        signals: list[Signal] = []

        for symbol, trade in list(self._trades.items()):
            current_price = self._get_current_price(symbol)
            if current_price is None:
                continue

            close_reason = ""

            # 1. Time limit
            if (et_now.hour > self._close_by_hour or
                    (et_now.hour == self._close_by_hour and et_now.minute >= self._close_by_minute)):
                close_reason = "time_limit"

            # 2. Initial stop check
            elif not trade.trailing_active:
                if trade.direction == "long" and current_price <= trade.stop_price:
                    close_reason = f"stop_hit: {current_price:.2f} <= {trade.stop_price:.2f}"
                elif trade.direction == "short" and current_price >= trade.stop_price:
                    close_reason = f"stop_hit: {current_price:.2f} >= {trade.stop_price:.2f}"

            # 3. Fixed target (when not trailing)
            if not close_reason and not self._use_trailing and trade.target_price:
                if trade.direction == "long" and current_price >= trade.target_price:
                    close_reason = "target_hit"
                elif trade.direction == "short" and current_price <= trade.target_price:
                    close_reason = "target_hit"

            # 4. Trailing stop management
            if not close_reason and self._use_trailing:
                close_reason = self._update_trailing(trade, current_price)

            if close_reason:
                self._close_trade(symbol, trade, close_reason)
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    direction=SignalDirection.CLOSE,
                    reason=close_reason,
                    metadata={"breakout_level": trade.breakout_level},
                    timestamp=datetime.now(pytz.UTC),
                ))

        return signals

    def _update_trailing(self, trade: MomentumTrade, current_price: float) -> str:
        """Update trailing stop. Returns close reason if trail hit, else empty."""
        risk_amount = abs(trade.entry_price - trade.initial_stop)
        if risk_amount <= 0:
            return ""

        if trade.direction == "long":
            profit = current_price - trade.entry_price
            trade.highest_price = max(trade.highest_price, current_price)

            # Activate trailing after 1R profit
            if not trade.trailing_active and profit >= risk_amount * self._trail_activation_rr:
                trade.trailing_active = True
                trade.trail_stop = current_price - trade.atr * self._trail_atr_mult
                self._log.info("trail_activated", symbol=trade.symbol, trail=trade.trail_stop)

            if trade.trailing_active:
                # Only move trail UP for longs
                new_trail = trade.highest_price - trade.atr * self._trail_atr_mult
                if new_trail > trade.trail_stop:
                    trade.trail_stop = new_trail
                    self._update_bracket_stop(trade, new_trail)
                if current_price <= trade.trail_stop:
                    return f"trail_stop: {current_price:.2f} <= {trade.trail_stop:.2f}"

        elif trade.direction == "short":
            profit = trade.entry_price - current_price
            trade.lowest_price = min(trade.lowest_price, current_price)

            if not trade.trailing_active and profit >= risk_amount * self._trail_activation_rr:
                trade.trailing_active = True
                trade.trail_stop = current_price + trade.atr * self._trail_atr_mult
                self._log.info("trail_activated", symbol=trade.symbol, trail=trade.trail_stop)

            if trade.trailing_active:
                # Only move trail DOWN for shorts
                new_trail = trade.lowest_price + trade.atr * self._trail_atr_mult
                if new_trail < trade.trail_stop:
                    trade.trail_stop = new_trail
                    self._update_bracket_stop(trade, new_trail)
                if current_price >= trade.trail_stop:
                    return f"trail_stop: {current_price:.2f} >= {trade.trail_stop:.2f}"

        return ""

    def _update_bracket_stop(self, trade: MomentumTrade, new_stop: float) -> None:
        """Update the broker-side bracket stop order for trailing."""
        if trade.is_bracket and trade.bracket_stop_order_id:
            updated = self.executor.replace_stop_order(
                trade.bracket_stop_order_id, new_stop,
            )
            if updated:
                trade.bracket_stop_order_id = updated.id

    def _scan_entries(self, et_now: datetime) -> list[Signal]:
        """Run breakout scanner and enter on confirmed breaks."""
        signals: list[Signal] = []
        self._last_scan_time = datetime.now(pytz.UTC)

        try:
            from algotrader.intelligence.scanners.breakout_scanner import BreakoutScanner
            scanner = BreakoutScanner(
                data_provider=self.data_provider,
                min_volume_ratio=self._min_volume_ratio,
                min_consolidation_days=self._min_consolidation_days,
            )
            breakouts = scanner.scan()
        except Exception:
            self._log.exception("breakout_scan_failed")
            return signals

        for bo in breakouts:
            if bo.symbol in self._trades:
                continue
            if len(self._trades) >= self.config.max_positions:
                break

            # Filter by volume
            if bo.volume_ratio < self._min_volume_ratio:
                continue
            if bo.consolidation_days < self._min_consolidation_days:
                continue

            # Check spread
            quote = self.data_provider.get_quote(bo.symbol)
            if quote and quote.is_valid() and quote.spread_pct * 100 > self._max_spread_pct:
                continue

            # Determine direction
            if bo.breakout_type == "resistance_break":
                direction = "long"
                side = OrderSide.BUY
                entry_price = bo.current_price
                stop_price = entry_price - bo.atr * self._stop_atr_mult
                risk = entry_price - stop_price
                target_price = entry_price + risk * self._fixed_target_rr if not self._use_trailing else None
            elif bo.breakout_type == "support_break":
                direction = "short"
                side = OrderSide.SELL
                entry_price = bo.current_price
                stop_price = entry_price + bo.atr * self._stop_atr_mult
                risk = stop_price - entry_price
                target_price = entry_price - risk * self._fixed_target_rr if not self._use_trailing else None
            else:
                continue

            if risk <= 0:
                continue

            # Size position
            risk_amount = self.available_capital * 0.0035
            shares = int(risk_amount / risk)
            max_value = self._total_capital * 0.05
            max_shares = int(max_value / entry_price) if entry_price > 0 else 0
            shares = min(shares, max_shares)

            if shares <= 0:
                continue

            capital_needed = shares * entry_price
            if not self.reserve_capital(capital_needed):
                continue

            client_id = f"momo_{bo.symbol}_{str(uuid.uuid4())[:8]}"
            order = self.executor.submit_order(
                symbol=bo.symbol,
                qty=shares,
                side=side,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_id,
                bracket_stop_price=stop_price,
                bracket_take_profit_price=target_price if not self._use_trailing else None,
            )

            if not order:
                self.release_capital(capital_needed)
                continue

            trade = MomentumTrade(
                symbol=bo.symbol,
                direction=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                initial_stop=stop_price,
                target_price=target_price,
                atr=bo.atr,
                breakout_level=bo.breakout_price,
                volume_ratio=bo.volume_ratio,
                entry_time=datetime.now(pytz.UTC),
                capital_used=capital_needed,
                highest_price=entry_price,
                lowest_price=entry_price,
                trade_id=str(uuid.uuid4()),
                bracket_stop_order_id=order.stop_order_id,
                bracket_tp_order_id=order.tp_order_id,
                is_bracket=order.is_bracket,
            )
            self._trades[bo.symbol] = trade

            self._log.info(
                "momentum_entry",
                symbol=bo.symbol,
                direction=direction,
                entry=entry_price,
                stop=stop_price,
                breakout_level=bo.breakout_price,
                volume_ratio=bo.volume_ratio,
                atr=bo.atr,
                shares=shares,
            )

            signals.append(Signal(
                strategy_name=self.name,
                symbol=bo.symbol,
                direction=SignalDirection.LONG if direction == "long" else SignalDirection.SHORT,
                reason=f"breakout: {bo.breakout_type} vol={bo.volume_ratio:.1f}x",
                metadata={
                    "breakout_type": bo.breakout_type,
                    "volume_ratio": bo.volume_ratio,
                    "consolidation_days": bo.consolidation_days,
                },
                timestamp=datetime.now(pytz.UTC),
            ))

        return signals

    def _close_trade(self, symbol: str, trade: MomentumTrade, reason: str) -> None:
        """Close a momentum trade with position exit safety."""
        broker_pos = self.executor.get_position(symbol)
        pnl = float(broker_pos.unrealized_pnl) if broker_pos else 0.0
        exit_price = float(broker_pos.current_price) if broker_pos else trade.entry_price

        close_success = self.executor.close_position(symbol)
        if not close_success:
            self._log.error("close_failed", symbol=symbol)
            return  # Keep in tracking, retry next cycle

        self._trades.pop(symbol, None)
        self.release_capital(trade.capital_used)

        side = OrderSide.BUY if trade.direction == "long" else OrderSide.SELL
        qty = trade.capital_used / trade.entry_price if trade.entry_price > 0 else 0
        self.record_trade(
            pnl,
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            entry_time=trade.entry_time,
            entry_reason=f"breakout: level={trade.breakout_level:.2f} vol={trade.volume_ratio:.1f}x",
            exit_reason=reason,
            metadata={"breakout_level": trade.breakout_level, "atr": trade.atr},
        )

        self._log.info(
            "momentum_exit",
            symbol=symbol,
            reason=reason,
            pnl=round(pnl, 2),
            direction=trade.direction,
            breakout_level=trade.breakout_level,
        )

    def _get_current_price(self, symbol: str) -> float | None:
        """Get current price from snapshot."""
        try:
            snap = self.data_provider.get_snapshot(symbol)
            if snap and snap.latest_trade_price:
                return snap.latest_trade_price
        except Exception:
            pass
        return None

    def assess_opportunities(self, regime: MarketRegime | None = None) -> OpportunityAssessment:
        """Assess breakout opportunities and emit concrete TradeCandidates."""
        self._log.info("assess_start", strategy=self.name)
        raw_scanned = 0

        def _complete(assessment: OpportunityAssessment) -> OpportunityAssessment:
            self._log.info(
                "assess_complete",
                strategy=self.name,
                num_candidates=len(assessment.candidates),
                num_raw_scanned=raw_scanned,
            )
            return assessment

        try:
            # Regime gate
            if regime and regime.regime_type.value not in self._allowed_regimes:
                return _complete(OpportunityAssessment())

            et_now = datetime.now(ET)
            if et_now.hour >= 15:
                # Momentum setups are not actionable for fresh entries after 3 PM ET.
                return _complete(OpportunityAssessment())

            from algotrader.intelligence.scanners.breakout_scanner import BreakoutScanner
            scanner = BreakoutScanner(
                data_provider=self.data_provider,
                min_volume_ratio=self._min_volume_ratio,
                min_consolidation_days=self._min_consolidation_days,
            )
            breakouts = scanner.scan()
            raw_scanned = len(breakouts)

            viable: list[Any] = []
            trade_candidates: list[TradeCandidate] = []
            rr_values: list[float] = []
            confidence_values: list[float] = []
            edge_values: list[float] = []
            expiry_time = et_now.replace(hour=15, minute=0, second=0, microsecond=0).astimezone(pytz.UTC)

            for bo in breakouts:
                if bo.symbol in self._trades:
                    continue
                if bo.volume_ratio < self._min_volume_ratio:
                    continue
                if bo.consolidation_days < self._min_consolidation_days:
                    continue

                quote = self.data_provider.get_quote(bo.symbol)
                if quote and quote.is_valid() and quote.spread_pct * 100 > self._max_spread_pct:
                    continue

                if bo.breakout_type == "resistance_break":
                    direction = "long"
                    entry_price = float(bo.current_price)
                    stop_price = entry_price - float(bo.atr) * self._stop_atr_mult
                    risk_per_share = entry_price - stop_price
                    target_price = entry_price + (risk_per_share * max(2.0, self._fixed_target_rr))
                    candidate_type = CandidateType.LONG_EQUITY
                elif bo.breakout_type == "support_break":
                    direction = "short"
                    entry_price = float(bo.current_price)
                    stop_price = entry_price + float(bo.atr) * self._stop_atr_mult
                    risk_per_share = stop_price - entry_price
                    target_price = entry_price - (risk_per_share * max(2.0, self._fixed_target_rr))
                    candidate_type = CandidateType.SHORT_EQUITY
                else:
                    continue

                if risk_per_share <= 0 or entry_price <= 0:
                    continue

                rr_ratio = abs(target_price - entry_price) / risk_per_share

                # Confidence calibration: volume quality + regime suitability + breakout quality.
                if bo.volume_ratio >= 2.0:
                    confidence = 0.35
                elif bo.volume_ratio >= 1.5:
                    confidence = 0.25
                else:
                    confidence = 0.10

                confidence += 0.15  # Scanner hit with consolidation and breakout structure.

                regime_fit = 0.5
                if regime:
                    regime_type = regime.regime_type.value
                    if regime_type in ("trending_up", "trending_down"):
                        regime_fit = 0.8
                        confidence += 0.20
                    elif regime_type == "high_vol":
                        regime_fit = 0.3
                        confidence += 0.05
                    elif regime_type == "ranging":
                        regime_fit = 0.2
                    else:
                        regime_fit = 0.5
                        confidence += 0.10

                    if direction == "long" and regime_type == "trending_up":
                        confidence += 0.10
                    elif direction == "short" and regime_type == "trending_down":
                        confidence += 0.10

                confidence = min(1.0, max(0.0, confidence))

                # Conservative edge model from breakout win-rate assumptions.
                est_win_rate = 0.48 if bo.volume_ratio >= 2.0 else 0.43
                expectancy_r = est_win_rate * rr_ratio - (1.0 - est_win_rate)
                risk_pct = (risk_per_share / entry_price) * 100.0
                edge_pct = max(0.0, expectancy_r * risk_pct)

                risk_budget = self.available_capital * 0.005
                suggested_qty = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
                risk_dollars = risk_per_share * max(1, suggested_qty)

                candidate = TradeCandidate(
                    strategy_name=self.name,
                    candidate_type=candidate_type,
                    symbol=bo.symbol,
                    direction=direction,
                    entry_price=round(entry_price, 2),
                    stop_price=round(stop_price, 2),
                    target_price=round(target_price, 2),
                    risk_dollars=round(risk_dollars, 2),
                    suggested_qty=suggested_qty,
                    risk_reward_ratio=round(rr_ratio, 2),
                    confidence=round(confidence, 2),
                    edge_estimate_pct=round(edge_pct, 2),
                    regime_fit=round(regime_fit, 2),
                    catalyst=f"breakout_vol_{bo.volume_ratio:.1f}x",
                    time_horizon_minutes=180,
                    expiry_time=expiry_time,
                    metadata={
                        "breakout_level": round(float(bo.breakout_price), 4),
                        "breakout_type": bo.breakout_type,
                        "volume_ratio": round(float(bo.volume_ratio), 2),
                        "atr": round(float(bo.atr), 4),
                        "consolidation_days": int(bo.consolidation_days),
                    },
                )

                viable.append(bo)
                trade_candidates.append(candidate)
                rr_values.append(rr_ratio)
                confidence_values.append(confidence)
                edge_values.append(edge_pct)

            if not viable:
                return _complete(OpportunityAssessment())

            trade_candidates.sort(key=lambda c: c.expected_value, reverse=True)
            trade_candidates = trade_candidates[:3]

            avg_vol = sum(bo.volume_ratio for bo in viable) / len(viable)
            avg_rr = (sum(rr_values) / len(rr_values)) if rr_values else self._fixed_target_rr
            avg_conf = (sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0
            avg_edge = (sum(edge_values) / len(edge_values)) if edge_values else round(0.3 * avg_vol, 2)

            self._log.debug(
                "momentum_assess_trade_candidates",
                total_viable=len(viable),
                emitted=len(trade_candidates),
                symbols=[c.symbol for c in trade_candidates],
            )

            return _complete(OpportunityAssessment(
                num_candidates=len(viable),
                avg_risk_reward=round(max(0.0, avg_rr), 2),
                confidence=round(max(0.0, min(1.0, avg_conf)), 2),
                estimated_daily_trades=min(len(viable), self.config.max_positions),
                estimated_edge_pct=round(max(0.0, avg_edge), 2),
                details=[
                    {
                        "symbol": bo.symbol,
                        "type": bo.breakout_type,
                        "volume_ratio": round(bo.volume_ratio, 1),
                        "consolidation_days": bo.consolidation_days,
                        "breakout_level": round(float(bo.breakout_price), 2),
                        "atr": round(float(bo.atr), 2),
                    }
                    for bo in viable[:5]
                ],
                candidates=trade_candidates,
            ))
        except Exception as exc:
            self._log.error("assess_failed", strategy=self.name, error=str(exc), exc_info=True)
            return _complete(OpportunityAssessment())

    def _get_state(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        base = super()._get_state()
        base["trades"] = {}
        for symbol, trade in self._trades.items():
            base["trades"][symbol] = {
                "symbol": trade.symbol,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "stop_price": trade.stop_price,
                "initial_stop": trade.initial_stop,
                "target_price": trade.target_price,
                "atr": trade.atr,
                "breakout_level": trade.breakout_level,
                "volume_ratio": trade.volume_ratio,
                "entry_time": trade.entry_time.isoformat(),
                "capital_used": trade.capital_used,
                "trailing_active": trade.trailing_active,
                "trail_stop": trade.trail_stop,
                "highest_price": trade.highest_price,
                "lowest_price": trade.lowest_price,
                "trade_id": trade.trade_id,
                "bracket_stop_order_id": trade.bracket_stop_order_id,
                "bracket_tp_order_id": trade.bracket_tp_order_id,
                "is_bracket": trade.is_bracket,
            }
        return base

    def _restore_state(self, state_data: dict[str, Any]) -> None:
        """Restore state from persistence."""
        super()._restore_state(state_data)
        for symbol, saved in state_data.get("trades", {}).items():
            self._trades[symbol] = MomentumTrade(
                symbol=saved["symbol"],
                direction=saved["direction"],
                entry_price=saved["entry_price"],
                stop_price=saved["stop_price"],
                initial_stop=saved["initial_stop"],
                target_price=saved.get("target_price"),
                atr=saved["atr"],
                breakout_level=saved["breakout_level"],
                volume_ratio=saved["volume_ratio"],
                entry_time=datetime.fromisoformat(saved["entry_time"]),
                capital_used=saved.get("capital_used", 0.0),
                trailing_active=saved.get("trailing_active", False),
                trail_stop=saved.get("trail_stop", 0.0),
                highest_price=saved.get("highest_price", 0.0),
                lowest_price=saved.get("lowest_price", 999999.0),
                trade_id=saved.get("trade_id", ""),
                bracket_stop_order_id=saved.get("bracket_stop_order_id", ""),
                bracket_tp_order_id=saved.get("bracket_tp_order_id", ""),
                is_bracket=saved.get("is_bracket", False),
            )
