"""Event-Driven strategy.

Trades around high-impact macro events (FOMC, CPI) and earnings.
Two modes:
- Pre-event: Reduce exposure or position for expected vol expansion
- Post-event: Trade the confirmed directional move after announcement
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
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

logger = structlog.get_logger()

ET = pytz.timezone("America/New_York")

# Standard event times in ET
EVENT_TIMES = {
    "fomc": (14, 0),   # 2:00 PM ET
    "cpi": (8, 30),    # 8:30 AM ET
    "ppi": (8, 30),
    "jobs": (8, 30),
    "gdp": (8, 30),
    "retail_sales": (8, 30),
    "pce": (8, 30),
}


@dataclass
class EventTrade:
    """Internal state for a single event-driven trade."""

    event_type: str  # "fomc", "cpi", etc.
    event_time: datetime  # When the announcement happens
    phase: str  # "pre_event", "post_event"
    symbol: str
    direction: str  # "long", "short", "neutral"
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    pre_event_price: float = 0.0  # Price before announcement
    post_event_move_pct: float = 0.0
    entry_time: datetime | None = None
    capital_used: float = 0.0
    trade_id: str = ""
    bracket_stop_order_id: str = ""
    bracket_tp_order_id: str = ""
    is_bracket: bool = False


@register_strategy("event_driven")
class EventDrivenStrategy(StrategyBase):
    """Event-driven trading around FOMC, CPI, and other macro events.

    Pre-event: Captures pre-event price snapshot.
    Post-event: Trades confirmed directional move after announcement.
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
        self._trade_fomc = params.get("trade_fomc", True)
        self._trade_cpi = params.get("trade_cpi", True)
        self._trade_earnings = params.get("trade_earnings", False)

        # Pre-event
        self._pre_event_entry_hours = params.get("pre_event_entry_hours", 2)
        self._pre_event_strategy = params.get("pre_event_strategy", "straddle_direction")

        # Post-event
        self._post_event_wait_minutes = params.get("post_event_wait_minutes", 5)
        self._post_event_entry_window = params.get("post_event_entry_window", 30)
        self._post_event_move_threshold = params.get("post_event_move_threshold", 0.3)

        # Sizing
        self._position_size_pct = params.get("position_size_pct", 1.0)
        self._max_loss_pct = params.get("max_loss_pct", 0.3)

        # Targets
        self._target_rr = params.get("target_rr", 2.0)
        self._time_limit_hours = params.get("time_limit_hours", 2)

        # Instruments
        self._instruments = params.get("instruments", ["SPY", "QQQ"])

        # Internal state
        self._trades: dict[str, EventTrade] = {}
        self._today_events: list[dict] = []
        self._pre_event_prices: dict[str, float] = {}  # symbol -> price before event
        self._events_checked_today: bool = False

    def warm_up(self) -> None:
        """Check today's event calendar."""
        self._log.info("warming_up")
        self._check_today_events()
        self._warmed_up = True

    def _check_today_events(self) -> None:
        """Query event calendar for today's events."""
        try:
            from algotrader.intelligence.calendar.events import EventCalendar
            calendar = EventCalendar()
            today_events = calendar.get_events_for_date()

            self._today_events = []
            for event in today_events:
                event_type = event.event_type.value

                # Filter by configured event types
                if event_type == "fomc" and not self._trade_fomc:
                    continue
                if event_type == "cpi" and not self._trade_cpi:
                    continue
                if event_type == "earnings" and not self._trade_earnings:
                    continue

                # Only trade high-impact events
                if event.impact < 2:
                    continue

                self._today_events.append({
                    "type": event_type,
                    "time": event.time,
                    "impact": event.impact,
                    "description": event.description,
                })

            self._events_checked_today = True
            if self._today_events:
                self._log.info("events_today", events=self._today_events)
            else:
                self._log.info("no_tradable_events_today")
        except Exception:
            self._log.exception("event_check_failed")

    def run_cycle(self, regime: MarketRegime | None = None) -> list[Signal]:
        """Run one cycle: check event timing, manage positions."""
        self._check_day_reset()
        self._last_cycle_time = datetime.now(pytz.UTC)
        signals: list[Signal] = []

        # Check events if not done today
        if not self._events_checked_today:
            self._check_today_events()

        if not self._today_events:
            return signals

        et_now = datetime.now(ET)

        # 1. Manage existing positions FIRST
        signals.extend(self._manage_positions(et_now))

        # 2. Check event timing for new entries
        if len(self._trades) < self.config.max_positions:
            signals.extend(self._check_event_entries(et_now))

        return signals

    def _check_event_entries(self, et_now: datetime) -> list[Signal]:
        """Check if we should enter based on event timing."""
        signals: list[Signal] = []

        for event in self._today_events:
            event_type = event["type"]
            event_time_str = event.get("time", "")

            # Parse event time
            event_time = self._parse_event_time(event_time_str, et_now)
            if event_time is None:
                continue

            # Pre-event phase: capture prices before announcement
            pre_event_start = event_time - timedelta(hours=self._pre_event_entry_hours)
            if pre_event_start <= et_now < event_time:
                self._capture_pre_event_prices()

            # Post-event phase: trade confirmed move
            post_event_start = event_time + timedelta(minutes=self._post_event_wait_minutes)
            post_event_end = event_time + timedelta(minutes=self._post_event_entry_window)

            if post_event_start <= et_now <= post_event_end:
                for symbol in self._instruments:
                    trade_key = f"{event_type}_{symbol}"
                    if trade_key in self._trades:
                        continue

                    signal = self._try_post_event_entry(
                        symbol, event_type, event_time, et_now,
                    )
                    if signal:
                        signals.append(signal)

        return signals

    def _try_post_event_entry(
        self, symbol: str, event_type: str, event_time: datetime, et_now: datetime,
    ) -> Signal | None:
        """Attempt a post-event directional entry."""
        current_price = self._get_current_price(symbol)
        if current_price is None or current_price <= 0:
            return None

        pre_event_price = self._pre_event_prices.get(symbol)
        if pre_event_price is None or pre_event_price <= 0:
            # Fall back to getting recent price history
            pre_event_price = self._get_pre_event_price(symbol, event_time)
            if pre_event_price is None:
                return None

        # Calculate post-event move
        move_pct = ((current_price - pre_event_price) / pre_event_price) * 100

        # Check if move exceeds threshold
        if abs(move_pct) < self._post_event_move_threshold:
            self._log.debug(
                "event_move_insufficient",
                symbol=symbol,
                event=event_type,
                move_pct=round(move_pct, 3),
                threshold=self._post_event_move_threshold,
            )
            return None

        # Determine direction based on confirmed move
        if move_pct > 0:
            direction = "long"
            side = OrderSide.BUY
        else:
            direction = "short"
            side = OrderSide.SELL

        # Calculate stop and target
        stop_distance = current_price * (self._max_loss_pct / 100)
        if direction == "long":
            # Use post-event low as potential stop reference
            stop_price = current_price - stop_distance
            target_price = current_price + stop_distance * self._target_rr
        else:
            stop_price = current_price + stop_distance
            target_price = current_price - stop_distance * self._target_rr

        # Size position — small for event trades
        position_value = self._total_capital * (self._position_size_pct / 100)
        shares = int(position_value / current_price)
        if shares <= 0:
            return None

        capital_needed = shares * current_price
        if not self.reserve_capital(capital_needed):
            return None

        client_id = f"event_{event_type}_{symbol}_{str(uuid.uuid4())[:8]}"
        order = self.executor.submit_order(
            symbol=symbol,
            qty=shares,
            side=side,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_id,
            bracket_stop_price=stop_price,
            bracket_take_profit_price=target_price,
        )

        if not order:
            self.release_capital(capital_needed)
            return None

        trade_key = f"{event_type}_{symbol}"
        trade = EventTrade(
            event_type=event_type,
            event_time=event_time,
            phase="post_event",
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            pre_event_price=pre_event_price,
            post_event_move_pct=move_pct,
            entry_time=datetime.now(pytz.UTC),
            capital_used=capital_needed,
            trade_id=str(uuid.uuid4()),
            bracket_stop_order_id=order.stop_order_id,
            bracket_tp_order_id=order.tp_order_id,
            is_bracket=order.is_bracket,
        )
        self._trades[trade_key] = trade

        self._log.info(
            "event_entry",
            event=event_type,
            symbol=symbol,
            direction=direction,
            move_pct=round(move_pct, 2),
            entry=current_price,
            stop=stop_price,
            target=target_price,
            pre_event_price=pre_event_price,
            shares=shares,
        )

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            direction=SignalDirection.LONG if direction == "long" else SignalDirection.SHORT,
            reason=f"post_{event_type}: move={move_pct:+.2f}%",
            metadata={
                "event_type": event_type,
                "move_pct": move_pct,
                "pre_event_price": pre_event_price,
            },
            timestamp=datetime.now(pytz.UTC),
        )

    def _manage_positions(self, et_now: datetime) -> list[Signal]:
        """Manage open event trades."""
        signals: list[Signal] = []

        for trade_key, trade in list(self._trades.items()):
            if trade.entry_price <= 0:
                continue

            current_price = self._get_current_price(trade.symbol)
            if current_price is None:
                continue

            close_reason = ""

            # 1. Time limit
            if trade.entry_time:
                hours_held = (datetime.now(pytz.UTC) - trade.entry_time).total_seconds() / 3600
                if hours_held >= self._time_limit_hours:
                    close_reason = f"time_limit: held {hours_held:.1f}h"

            # 2. Stop loss
            if not close_reason:
                if trade.direction == "long" and current_price <= trade.stop_price:
                    close_reason = f"stop_hit: {current_price:.2f} <= {trade.stop_price:.2f}"
                elif trade.direction == "short" and current_price >= trade.stop_price:
                    close_reason = f"stop_hit: {current_price:.2f} >= {trade.stop_price:.2f}"

            # 3. Target hit
            if not close_reason:
                if trade.direction == "long" and current_price >= trade.target_price:
                    close_reason = f"target_hit: {current_price:.2f} >= {trade.target_price:.2f}"
                elif trade.direction == "short" and current_price <= trade.target_price:
                    close_reason = f"target_hit: {current_price:.2f} <= {trade.target_price:.2f}"

            if close_reason:
                self._close_trade(trade_key, trade, close_reason)
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=trade.symbol,
                    direction=SignalDirection.CLOSE,
                    reason=close_reason,
                    metadata={"event_type": trade.event_type, "phase": trade.phase},
                    timestamp=datetime.now(pytz.UTC),
                ))

        return signals

    def _close_trade(self, trade_key: str, trade: EventTrade, reason: str) -> None:
        """Close an event trade with position exit safety."""
        broker_pos = self.executor.get_position(trade.symbol)
        pnl = float(broker_pos.unrealized_pnl) if broker_pos else 0.0
        exit_price = float(broker_pos.current_price) if broker_pos else trade.entry_price

        close_success = self.executor.close_position(trade.symbol)
        if not close_success:
            self._log.error("close_failed", symbol=trade.symbol)
            return  # Keep in tracking, retry next cycle

        self._trades.pop(trade_key, None)
        self.release_capital(trade.capital_used)

        side = OrderSide.BUY if trade.direction == "long" else OrderSide.SELL
        qty = trade.capital_used / trade.entry_price if trade.entry_price > 0 else 0
        self.record_trade(
            pnl,
            symbol=trade.symbol,
            side=side,
            qty=qty,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            entry_time=trade.entry_time,
            entry_reason=f"post_{trade.event_type}: move={trade.post_event_move_pct:+.2f}%",
            exit_reason=reason,
            metadata={
                "event_type": trade.event_type,
                "phase": trade.phase,
                "pre_event_price": trade.pre_event_price,
                "move_pct": trade.post_event_move_pct,
            },
        )

        self._log.info(
            "event_exit",
            event=trade.event_type,
            symbol=trade.symbol,
            reason=reason,
            pnl=round(pnl, 2),
            direction=trade.direction,
            move_pct=round(trade.post_event_move_pct, 2),
        )

    def _capture_pre_event_prices(self) -> None:
        """Capture current prices as pre-event reference points."""
        for symbol in self._instruments:
            if symbol not in self._pre_event_prices:
                price = self._get_current_price(symbol)
                if price:
                    self._pre_event_prices[symbol] = price
                    self._log.info("pre_event_price_captured", symbol=symbol, price=price)

    def _get_pre_event_price(self, symbol: str, event_time: datetime) -> float | None:
        """Get the price from before the event using historical bars."""
        try:
            bars = self.data_provider.get_bars(symbol, "5Min", 20)
            if bars.empty:
                return None
            # Use the price from earliest available bar as approximation
            return float(bars["close"].iloc[0])
        except Exception:
            return None

    def _parse_event_time(self, time_str: str, et_now: datetime) -> datetime | None:
        """Parse event time string (HH:MM) into today's datetime in ET."""
        if not time_str:
            return None
        try:
            parts = time_str.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            return et_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except (ValueError, IndexError):
            return None

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
        """Assess event-driven opportunities from today's calendar."""
        self._log.info("assess_start", strategy=self.name)
        raw_scanned = len(self._today_events)
        regime_type = regime.regime_type.value if regime else "none"

        def _complete(assessment: OpportunityAssessment) -> OpportunityAssessment:
            self._log.info(
                "assess_complete",
                strategy=self.name,
                num_candidates=len(assessment.candidates),
                num_raw_scanned=raw_scanned,
            )
            return assessment

        try:
            from algotrader.strategy_selector.candidate import CandidateType, TradeCandidate

            self._log.info(
                "event_assess_context",
                regime=regime_type,
                tracked_events=len(self._today_events),
                instruments=self._instruments,
            )
            filter_counts = {
                "no_events_today": 0,
                "already_traded_event": 0,
                "outside_event_window": 0,
                "missing_price_data": 0,
                "missing_event_time": 0,
                "candidate_built": 0,
            }
            if not self._events_checked_today:
                self._check_today_events()
                raw_scanned = len(self._today_events)

            if not self._today_events:
                filter_counts["no_events_today"] = 1
                self._log.info(
                    "event_scan_result",
                    regime=regime_type,
                    final_candidates=0,
                    filter_counts=filter_counts,
                )
                return _complete(OpportunityAssessment())

            # Count actionable events (not already traded)
            actionable = []
            for event in self._today_events:
                event_type = event["type"]
                # Check if any instrument for this event is already traded
                already_traded = any(
                    f"{event_type}_{sym}" in self._trades
                    for sym in self._instruments
                )
                if not already_traded:
                    actionable.append(event)
                else:
                    filter_counts["already_traded_event"] += 1

            if not actionable:
                self._log.info(
                    "event_scan_result",
                    regime=regime_type,
                    final_candidates=0,
                    filter_counts=filter_counts,
                )
                return _complete(OpportunityAssessment())

            et_now = datetime.now(ET)
            trade_candidates: list[TradeCandidate] = []
            details: list[dict] = []

            for event in actionable:
                event_type = event["type"]
                event_time = self._parse_event_time(event.get("time", ""), et_now)
                if event_time is None and event_type in EVENT_TIMES:
                    hh, mm = EVENT_TIMES[event_type]
                    event_time = et_now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                if event_time is None:
                    filter_counts["missing_event_time"] += 1
                    continue

                is_post_event = et_now >= event_time
                hours_to_event = (event_time - et_now).total_seconds() / 3600.0
                if not is_post_event and hours_to_event > self._pre_event_entry_hours:
                    filter_counts["outside_event_window"] += 1
                    continue

                for symbol in self._instruments:
                    trade_key = f"{event_type}_{symbol}"
                    if trade_key in self._trades:
                        continue

                    current_price = self._get_current_price(symbol)
                    if current_price is None or current_price <= 0:
                        filter_counts["missing_price_data"] += 1
                        continue

                    atr = None
                    bars = self.data_provider.get_bars(symbol, "5Min", 30)
                    if not bars.empty and len(bars) >= 15:
                        highs = bars["high"].values
                        lows = bars["low"].values
                        closes = bars["close"].values
                        tr_values = []
                        for i in range(1, len(bars)):
                            tr_values.append(max(
                                highs[i] - lows[i],
                                abs(highs[i] - closes[i - 1]),
                                abs(lows[i] - closes[i - 1]),
                            ))
                        if tr_values:
                            atr = float(sum(tr_values[-14:]) / len(tr_values[-14:]))
                    if not atr or atr <= 0:
                        atr = current_price * 0.005

                    if is_post_event:
                        pre_event_price = self._pre_event_prices.get(symbol) or self._get_pre_event_price(symbol, event_time)
                        if pre_event_price and pre_event_price > 0:
                            move_pct = ((current_price - pre_event_price) / pre_event_price) * 100
                        else:
                            move_pct = 0.0
                        direction = "long" if move_pct >= 0 else "short"
                        base_conf = 0.35 if event_type in {"fomc", "cpi", "ppi", "jobs", "gdp", "retail_sales", "pce"} else 0.30
                        confidence = base_conf + (0.25 if abs(move_pct) >= self._post_event_move_threshold * 1.5 else 0.05)
                        expiry_time = (event_time + timedelta(hours=4)).astimezone(pytz.UTC)
                        catalyst = f"post_{event_type}_{direction}"
                        win_rate = 0.52
                    else:
                        pre_event_price = self._pre_event_prices.get(symbol) or self._get_pre_event_price(symbol, event_time)
                        if pre_event_price and pre_event_price > 0 and current_price >= pre_event_price:
                            direction = "long"
                        else:
                            direction = "short"
                        base_conf = 0.35 if event_type in {"fomc", "cpi", "ppi", "jobs", "gdp", "retail_sales", "pce"} else 0.30
                        confidence = min(0.40, base_conf + 0.05)
                        expiry_time = event_time.astimezone(pytz.UTC)
                        catalyst = f"pre_{event_type}_{direction}"
                        win_rate = 0.45

                    if direction == "long":
                        stop_price = current_price - atr
                        target_price = current_price + 2.0 * atr
                    else:
                        stop_price = current_price + atr
                        target_price = current_price - 2.0 * atr

                    risk_per_share = abs(current_price - stop_price)
                    reward_per_share = abs(target_price - current_price)
                    if risk_per_share <= 0:
                        continue
                    rr_ratio = reward_per_share / risk_per_share

                    regime_fit = 1.0 if regime and regime.regime_type.value == "event_day" else 0.0
                    confidence = min(1.0, max(0.0, confidence))
                    if not is_post_event:
                        confidence = min(confidence, 0.40)

                    risk_pct = (risk_per_share / current_price) * 100.0 if current_price > 0 else 0.0
                    edge_pct = max(0.0, (win_rate * rr_ratio - (1.0 - win_rate)) * risk_pct)
                    risk_budget = self.available_capital * 0.004
                    suggested_qty = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0

                    trade_candidates.append(TradeCandidate(
                        strategy_name=self.name,
                        candidate_type=CandidateType.EVENT_DIRECTIONAL,
                        symbol=symbol,
                        direction=direction,
                        entry_price=round(current_price, 2),
                        stop_price=round(stop_price, 2),
                        target_price=round(target_price, 2),
                        risk_dollars=round(risk_per_share * max(1, suggested_qty), 2),
                        suggested_qty=suggested_qty,
                        risk_reward_ratio=round(rr_ratio, 2),
                        confidence=round(confidence, 2),
                        edge_estimate_pct=round(edge_pct, 2),
                        regime_fit=round(regime_fit, 2),
                        catalyst=catalyst,
                        time_horizon_minutes=240 if is_post_event else int(max(30, hours_to_event * 60)),
                        expiry_time=expiry_time,
                        metadata={
                            "event_type": event_type,
                            "is_post_event": is_post_event,
                            "event_time": event_time.isoformat(),
                        },
                    ))
                    details.append(
                        {
                            "event": event_type,
                            "symbol": symbol,
                            "is_post_event": is_post_event,
                            "time": event.get("time", ""),
                            "impact": event.get("impact", 2),
                        }
                    )
                    filter_counts["candidate_built"] += 1

            if not trade_candidates:
                self._log.info(
                    "event_scan_result",
                    regime=regime_type,
                    final_candidates=0,
                    filter_counts=filter_counts,
                )
                return _complete(OpportunityAssessment())

            trade_candidates.sort(key=lambda c: c.expected_value, reverse=True)
            top_candidates = trade_candidates[:3]
            avg_rr = sum(c.risk_reward_ratio for c in trade_candidates) / len(trade_candidates)
            avg_conf = sum(c.confidence for c in trade_candidates) / len(trade_candidates)
            avg_edge = sum(c.edge_estimate_pct for c in trade_candidates) / len(trade_candidates)
            self._log.info(
                "event_scan_result",
                regime=regime_type,
                final_candidates=len(top_candidates),
                filter_counts=filter_counts,
            )

            return _complete(OpportunityAssessment(
                num_candidates=len(trade_candidates),
                avg_risk_reward=round(max(0.0, avg_rr), 2),
                confidence=round(max(0.0, min(1.0, avg_conf)), 2),
                estimated_daily_trades=min(
                    len(trade_candidates),
                    self.config.max_positions - len(self._trades),
                ),
                estimated_edge_pct=round(max(0.0, avg_edge), 2),
                details=details[:5],
                candidates=top_candidates,
            ))
        except Exception as exc:
            self._log.error("assess_failed", strategy=self.name, error=str(exc), exc_info=True)
            return _complete(OpportunityAssessment())

    def _get_state(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        base = super()._get_state()
        base["trades"] = {}
        for key, trade in self._trades.items():
            base["trades"][key] = {
                "event_type": trade.event_type,
                "event_time": trade.event_time.isoformat(),
                "phase": trade.phase,
                "symbol": trade.symbol,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
                "pre_event_price": trade.pre_event_price,
                "post_event_move_pct": trade.post_event_move_pct,
                "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
                "capital_used": trade.capital_used,
                "trade_id": trade.trade_id,
                "bracket_stop_order_id": trade.bracket_stop_order_id,
                "bracket_tp_order_id": trade.bracket_tp_order_id,
                "is_bracket": trade.is_bracket,
            }
        base["pre_event_prices"] = self._pre_event_prices
        return base

    def _restore_state(self, state_data: dict[str, Any]) -> None:
        """Restore state from persistence."""
        super()._restore_state(state_data)
        self._pre_event_prices = state_data.get("pre_event_prices", {})
        for key, saved in state_data.get("trades", {}).items():
            self._trades[key] = EventTrade(
                event_type=saved["event_type"],
                event_time=datetime.fromisoformat(saved["event_time"]),
                phase=saved["phase"],
                symbol=saved["symbol"],
                direction=saved["direction"],
                entry_price=saved.get("entry_price", 0.0),
                stop_price=saved.get("stop_price", 0.0),
                target_price=saved.get("target_price", 0.0),
                pre_event_price=saved.get("pre_event_price", 0.0),
                post_event_move_pct=saved.get("post_event_move_pct", 0.0),
                entry_time=datetime.fromisoformat(saved["entry_time"]) if saved.get("entry_time") else None,
                capital_used=saved.get("capital_used", 0.0),
                trade_id=saved.get("trade_id", ""),
                bracket_stop_order_id=saved.get("bracket_stop_order_id", ""),
                bracket_tp_order_id=saved.get("bracket_tp_order_id", ""),
                is_bracket=saved.get("is_bracket", False),
            )
