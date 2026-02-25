"""Gap & Reversal strategy.

Trades stocks that gap significantly at the open. Two sub-strategies:
- Gap-and-Go: Continue in gap direction when catalyst (news) confirms
- Gap Fade: Fade the gap when no catalyst and gap starts to fill

Uses GapScanner for candidates and AlpacaNewsClient for catalyst detection.
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

logger = structlog.get_logger()

ET = pytz.timezone("America/New_York")


@dataclass
class GapTrade:
    """Internal state for a single gap trade."""

    symbol: str
    trade_type: str  # "gap_go" or "gap_fade"
    direction: str  # "long" or "short"
    entry_price: float
    stop_price: float
    target_price: float
    gap_pct: float
    prev_close: float
    gap_open: float
    entry_time: datetime
    capital_used: float = 0.0
    first_candle_high: float | None = None
    first_candle_low: float | None = None
    waiting_for_entry: bool = True  # True if monitoring but not yet entered
    trade_id: str = ""
    bracket_stop_order_id: str = ""
    bracket_tp_order_id: str = ""
    is_bracket: bool = False


@register_strategy("gap_reversal")
class GapReversalStrategy(StrategyBase):
    """Gap & Reversal strategy: gap-and-go + gap fade.

    Pre-market: GapScanner provides gappers.
    At open: classify each gap (catalyst → go, no catalyst → fade).
    Manage positions with stops, targets, and time limits.
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
        self._min_gap_pct = params.get("min_gap_pct", 2.0)
        self._max_gap_pct = params.get("max_gap_pct", 8.0)
        self._min_volume = params.get("min_volume", 500_000)

        # Gap-and-Go params
        self._go_entry_bar = params.get("go_entry_bar", 5)
        self._go_stop_atr_mult = params.get("go_stop_atr_mult", 1.0)
        self._go_target_rr = params.get("go_target_rr", 2.0)

        # Gap Fade params
        self._fade_wait_minutes = params.get("fade_wait_minutes", 15)
        self._fade_fill_target_pct = params.get("fade_fill_target_pct", 50.0)
        self._fade_stop_pct = params.get("fade_stop_pct", 0.5)

        # Time management
        self._close_by_hour = params.get("close_by_hour", 11)
        self._close_by_minute = params.get("close_by_minute", 0)

        # Regime filter
        self._allowed_regimes = params.get(
            "allowed_regimes",
            ["trending_up", "trending_down", "ranging", "low_vol"],
        )

        # Internal state
        self._trades: dict[str, GapTrade] = {}
        self._gap_candidates: list[dict] = []  # Populated from GapScanner
        self._candidates_loaded: bool = False

    def warm_up(self) -> None:
        """Load gap candidates from scanner during pre-market."""
        self._log.info("warming_up")
        self._warmed_up = True

    def set_gap_candidates(self, candidates: list[dict]) -> None:
        """Set gap candidates from the GapScanner results.

        Called by the orchestrator after pre-market gap scan.
        Each dict should have: symbol, gap_pct, prev_close, current_price, direction
        """
        self._gap_candidates = candidates
        self._candidates_loaded = True
        self._log.info("gap_candidates_set", count=len(candidates))

    def _past_entry_cutoff(self, et_now: datetime) -> bool:
        """Return True if it's past the 11 AM ET cutoff for new entries."""
        return (et_now.hour > self._close_by_hour or
                (et_now.hour == self._close_by_hour and
                 et_now.minute >= self._close_by_minute))

    def run_cycle(self, regime: MarketRegime | None = None) -> list[Signal]:
        """Run one cycle: manage positions, scan for new gap entries."""
        self._check_day_reset()
        self._last_cycle_time = datetime.now(pytz.UTC)
        signals: list[Signal] = []

        et_now = datetime.now(ET)

        # 1. Manage existing positions FIRST (always, regardless of regime)
        signals.extend(self._manage_positions(et_now))

        # 2. Check regime filter (only gates new entries)
        if regime and regime.regime_type.value not in self._allowed_regimes:
            return signals

        # 3. Scan for new entries (if under max_positions and within entry window)
        if len(self._trades) < self.config.max_positions and not self._past_entry_cutoff(et_now):
            signals.extend(self._scan_entries(et_now))

        return signals

    def _manage_positions(self, et_now: datetime) -> list[Signal]:
        """Manage open gap trades: time limits, stops, targets, trailing."""
        signals: list[Signal] = []
        closed_symbols: list[str] = []

        for symbol, trade in list(self._trades.items()):
            if trade.waiting_for_entry:
                continue  # Not yet entered — handle in _scan_entries

            current_price = self._get_current_price(symbol)
            if current_price is None:
                continue

            close_reason = ""

            # 1. Time limit
            if (et_now.hour > self._close_by_hour or
                    (et_now.hour == self._close_by_hour and et_now.minute >= self._close_by_minute)):
                close_reason = "time_limit"

            # 2. Stop hit
            elif trade.direction == "long" and current_price <= trade.stop_price:
                close_reason = f"stop_hit: {current_price:.2f} <= {trade.stop_price:.2f}"
            elif trade.direction == "short" and current_price >= trade.stop_price:
                close_reason = f"stop_hit: {current_price:.2f} >= {trade.stop_price:.2f}"

            # 3. Target hit
            elif trade.direction == "long" and current_price >= trade.target_price:
                close_reason = f"target_hit: {current_price:.2f} >= {trade.target_price:.2f}"
            elif trade.direction == "short" and current_price <= trade.target_price:
                close_reason = f"target_hit: {current_price:.2f} <= {trade.target_price:.2f}"

            # 4. Trail stop on winners (once 50% to target, move stop to breakeven)
            else:
                self._update_trailing_stop(trade, current_price)

            if close_reason:
                self._close_trade(symbol, trade, close_reason)
                closed_symbols.append(symbol)
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    direction=SignalDirection.CLOSE,
                    reason=close_reason,
                    metadata={"trade_type": trade.trade_type, "gap_pct": trade.gap_pct},
                    timestamp=datetime.now(pytz.UTC),
                ))

        return signals

    def _scan_entries(self, et_now: datetime) -> list[Signal]:
        """Scan gap candidates for new entries."""
        signals: list[Signal] = []

        if not self._candidates_loaded:
            # Try loading from gap scanner on the fly
            self._load_gap_candidates()

        for candidate in self._gap_candidates:
            symbol = candidate.get("symbol", "")
            if not symbol or symbol in self._trades:
                continue
            if len(self._trades) >= self.config.max_positions:
                break
            # Skip symbols already held by another strategy (broker is authoritative)
            if self.executor.get_position(symbol) is not None:
                continue

            gap_pct = candidate.get("gap_pct", 0.0)
            prev_close = candidate.get("prev_close", 0.0)
            gap_open = candidate.get("current_price", 0.0)

            if abs(gap_pct) < self._min_gap_pct or abs(gap_pct) > self._max_gap_pct:
                continue

            # Classify: catalyst check via news sentiment
            has_catalyst = self._check_catalyst(symbol)

            if has_catalyst:
                signal = self._try_gap_go_entry(symbol, gap_pct, prev_close, gap_open, et_now)
            else:
                signal = self._try_gap_fade_entry(symbol, gap_pct, prev_close, gap_open, et_now)

            if signal:
                signals.append(signal)

        return signals

    def _try_gap_go_entry(
        self, symbol: str, gap_pct: float, prev_close: float, gap_open: float, et_now: datetime,
    ) -> Signal | None:
        """Attempt a gap-and-go entry using first N-min candle breakout."""
        # Get recent intraday bars to find first candle
        bars = self.data_provider.get_bars(symbol, f"{self._go_entry_bar}Min", 5)
        if bars.empty or len(bars) < 1:
            return None

        first_high = float(bars["high"].iloc[0])
        first_low = float(bars["low"].iloc[0])
        current_price = float(bars["close"].iloc[-1])

        gap_up = gap_pct > 0

        # Gap UP: buy on break above first candle high
        if gap_up and current_price > first_high:
            entry_price = current_price
            stop_price = first_low
            risk = entry_price - stop_price
            if risk <= 0:
                return None
            target_price = entry_price + risk * self._go_target_rr
            direction = "long"
            side = OrderSide.BUY

        # Gap DOWN: short on break below first candle low
        elif not gap_up and current_price < first_low:
            entry_price = current_price
            stop_price = first_high
            risk = stop_price - entry_price
            if risk <= 0:
                return None
            target_price = entry_price - risk * self._go_target_rr
            direction = "short"
            side = OrderSide.SELL
        else:
            return None

        # Round prices to valid tick (avoid sub-penny bracket rejections)
        stop_price = round(stop_price, 2)
        target_price = round(target_price, 2)

        return self._execute_entry(
            symbol=symbol,
            trade_type="gap_go",
            direction=direction,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            gap_pct=gap_pct,
            prev_close=prev_close,
            gap_open=gap_open,
            first_candle_high=first_high,
            first_candle_low=first_low,
        )

    def _try_gap_fade_entry(
        self, symbol: str, gap_pct: float, prev_close: float, gap_open: float, et_now: datetime,
    ) -> Signal | None:
        """Attempt a gap fade entry after waiting period."""
        # Check that enough time has passed since open
        market_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
        minutes_since_open = (et_now - market_open).total_seconds() / 60

        if minutes_since_open < self._fade_wait_minutes:
            return None

        current_price = self._get_current_price(symbol)
        if current_price is None:
            return None

        gap_up = gap_pct > 0

        # Confirm gap is starting to fill (price moved >20% toward prev close)
        total_gap = abs(gap_open - prev_close)
        if total_gap <= 0:
            return None
        fill_pct = abs(gap_open - current_price) / total_gap * 100

        if fill_pct < 20:
            return None  # Gap not filling yet

        # Calculate target: 50% gap fill
        fill_target = prev_close + (gap_open - prev_close) * (1 - self._fade_fill_target_pct / 100)

        if gap_up:
            # Gap UP fade: short
            direction = "short"
            side = OrderSide.SELL
            entry_price = current_price
            stop_price = gap_open * (1 + self._fade_stop_pct / 100)
            target_price = fill_target
            # Stop must be above entry for a short
            if stop_price <= entry_price:
                return None
        else:
            # Gap DOWN fade: long
            direction = "long"
            side = OrderSide.BUY
            entry_price = current_price
            stop_price = gap_open * (1 - self._fade_stop_pct / 100)
            target_price = fill_target
            # Stop must be below entry for a long
            if stop_price >= entry_price:
                return None

        # Round prices to valid tick (avoid sub-penny rejections)
        stop_price = round(stop_price, 2)
        target_price = round(target_price, 2)

        return self._execute_entry(
            symbol=symbol,
            trade_type="gap_fade",
            direction=direction,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            gap_pct=gap_pct,
            prev_close=prev_close,
            gap_open=gap_open,
        )

    def _execute_entry(
        self,
        symbol: str,
        trade_type: str,
        direction: str,
        side: OrderSide,
        entry_price: float,
        stop_price: float,
        target_price: float,
        gap_pct: float,
        prev_close: float,
        gap_open: float,
        first_candle_high: float | None = None,
        first_candle_low: float | None = None,
    ) -> Signal | None:
        """Execute an entry order and track the trade."""
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return None

        # Calculate shares using risk-based sizing
        risk_amount = self.available_capital * 0.0035  # 0.35% risk
        shares = int(risk_amount / stop_distance)

        # Cap at max position value (5% of capital)
        max_value = self._total_capital * 0.05
        max_shares = int(max_value / entry_price) if entry_price > 0 else 0
        shares = min(shares, max_shares)

        if shares <= 0:
            return None

        capital_needed = shares * entry_price
        if not self.reserve_capital(capital_needed):
            return None

        client_id = f"gap_{trade_type}_{symbol}_{str(uuid.uuid4())[:8]}"
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

        trade = GapTrade(
            symbol=symbol,
            trade_type=trade_type,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            gap_pct=gap_pct,
            prev_close=prev_close,
            gap_open=gap_open,
            entry_time=datetime.now(pytz.UTC),
            capital_used=capital_needed,
            first_candle_high=first_candle_high,
            first_candle_low=first_candle_low,
            waiting_for_entry=False,
            trade_id=str(uuid.uuid4()),
            bracket_stop_order_id=order.stop_order_id,
            bracket_tp_order_id=order.tp_order_id,
            is_bracket=order.is_bracket,
        )
        self._trades[symbol] = trade

        self._log.info(
            "gap_entry",
            symbol=symbol,
            type=trade_type,
            direction=direction,
            entry=entry_price,
            stop=stop_price,
            target=target_price,
            gap_pct=gap_pct,
            shares=shares,
        )

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            direction=SignalDirection.LONG if direction == "long" else SignalDirection.SHORT,
            reason=f"{trade_type}: gap {gap_pct:+.1f}%",
            metadata={"trade_type": trade_type, "gap_pct": gap_pct},
            timestamp=datetime.now(pytz.UTC),
        )

    def _close_trade(self, symbol: str, trade: GapTrade, reason: str) -> None:
        """Close a gap trade with position exit safety."""
        # Get P&L from broker before closing
        broker_pos = self.executor.get_position(symbol)
        pnl = float(broker_pos.unrealized_pnl) if broker_pos else 0.0
        exit_price = float(broker_pos.current_price) if broker_pos else trade.entry_price

        close_success = self.executor.close_position(symbol)
        if not close_success:
            self._log.error("close_failed", symbol=symbol)
            return  # Keep in tracking, retry next cycle

        # Only after confirmed close
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
            entry_reason=f"{trade.trade_type}: gap {trade.gap_pct:+.1f}%",
            exit_reason=reason,
            metadata={"trade_type": trade.trade_type, "gap_pct": trade.gap_pct},
        )

        self._log.info(
            "gap_exit",
            symbol=symbol,
            type=trade.trade_type,
            reason=reason,
            pnl=round(pnl, 2),
            gap_pct=trade.gap_pct,
        )

    def _update_trailing_stop(self, trade: GapTrade, current_price: float) -> None:
        """Move stop to breakeven once 50% to target."""
        if trade.direction == "long":
            distance_to_target = trade.target_price - trade.entry_price
            current_profit = current_price - trade.entry_price
            if distance_to_target > 0 and current_profit >= distance_to_target * 0.5:
                new_stop = max(trade.stop_price, trade.entry_price)
                if new_stop > trade.stop_price:
                    trade.stop_price = new_stop
                    self._update_bracket_stop(trade, new_stop)
        elif trade.direction == "short":
            distance_to_target = trade.entry_price - trade.target_price
            current_profit = trade.entry_price - current_price
            if distance_to_target > 0 and current_profit >= distance_to_target * 0.5:
                new_stop = min(trade.stop_price, trade.entry_price)
                if new_stop < trade.stop_price:
                    trade.stop_price = new_stop
                    self._update_bracket_stop(trade, new_stop)

    def _update_bracket_stop(self, trade: GapTrade, new_stop: float) -> None:
        """Update the broker-side bracket stop order."""
        if trade.is_bracket and trade.bracket_stop_order_id:
            updated = self.executor.replace_stop_order(
                trade.bracket_stop_order_id, new_stop,
            )
            if updated:
                trade.bracket_stop_order_id = updated.id

    def _check_catalyst(self, symbol: str) -> bool:
        """Check if a symbol has a news catalyst."""
        try:
            news = self.data_provider.get_news(symbols=[symbol], limit=5)
            if not news:
                return False
            # Simple check: any recent news with sentiment != 0
            for item in news:
                if item.sentiment_score and abs(item.sentiment_score) > 0.2:
                    return True
            return False
        except Exception:
            return False

    def _get_current_price(self, symbol: str) -> float | None:
        """Get current price from snapshot."""
        try:
            snap = self.data_provider.get_snapshot(symbol)
            if snap and snap.latest_trade_price:
                return snap.latest_trade_price
        except Exception:
            pass
        return None

    def _load_gap_candidates(self) -> None:
        """Load gap candidates from GapScanner on the fly."""
        try:
            from algotrader.intelligence.scanners.gap_scanner import GapScanner
            scanner = GapScanner(
                data_provider=self.data_provider,
                min_gap_pct=self._min_gap_pct,
                min_volume=self._min_volume,
            )
            results = scanner.scan()
            self._gap_candidates = [
                {
                    "symbol": r.symbol,
                    "gap_pct": r.gap_pct,
                    "prev_close": r.prev_close,
                    "current_price": r.current_price,
                    "direction": r.direction,
                }
                for r in results
            ]
            self._candidates_loaded = True
        except Exception:
            self._log.exception("gap_candidates_load_failed")

    def assess_opportunities(self, regime: MarketRegime | None = None) -> OpportunityAssessment:
        """Assess gap trading opportunities from scanner candidates."""
        self._log.info("assess_start", strategy=self.name)
        raw_scanned = len(self._gap_candidates)

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

            # Regime gate
            if regime and regime.regime_type.value not in self._allowed_regimes:
                return _complete(OpportunityAssessment())

            if not self._gap_candidates:
                return _complete(OpportunityAssessment())

            et_now = datetime.now(ET)
            expiry_time = et_now.replace(hour=11, minute=0, second=0, microsecond=0).astimezone(pytz.UTC)

            qualified = []
            details: list[dict] = []
            trade_candidates: list[TradeCandidate] = []
            for c in self._gap_candidates:
                gap_pct = abs(c.get("gap_pct", 0.0))
                if not (self._min_gap_pct <= gap_pct <= self._max_gap_pct):
                    continue

                symbol = c.get("symbol", "")
                if not symbol:
                    continue

                gap_signed = float(c.get("gap_pct", 0.0))
                prev_close = float(c.get("prev_close", 0.0))
                gap_open = float(c.get("current_price", 0.0))
                current_price = self._get_current_price(symbol) or gap_open
                if current_price <= 0 or gap_open <= 0:
                    continue

                bars = self.data_provider.get_bars(symbol, "5Min", 3)
                first_high = float(bars["high"].iloc[0]) if not bars.empty else max(current_price, gap_open)
                first_low = float(bars["low"].iloc[0]) if not bars.empty else min(current_price, gap_open)
                has_catalyst = self._check_catalyst(symbol)

                if has_catalyst:
                    trade_type = "go"
                    if gap_signed > 0:
                        direction = "long"
                        entry_price = max(first_high, current_price)
                        stop_price = min(first_low, gap_open * (1 - self._fade_stop_pct / 100))
                        target_price = entry_price + abs(entry_price - stop_price) * 2.0
                        candidate_type = CandidateType.LONG_EQUITY
                    else:
                        direction = "short"
                        entry_price = min(first_low, current_price)
                        stop_price = max(first_high, gap_open * (1 + self._fade_stop_pct / 100))
                        target_price = entry_price - abs(stop_price - entry_price) * 2.0
                        candidate_type = CandidateType.SHORT_EQUITY
                else:
                    trade_type = "fade"
                    fill_target = prev_close + (gap_open - prev_close) * (1 - self._fade_fill_target_pct / 100)
                    if gap_signed > 0:
                        direction = "short"
                        entry_price = current_price
                        stop_price = gap_open * (1 + self._fade_stop_pct / 100)
                        target_price = fill_target
                        candidate_type = CandidateType.SHORT_EQUITY
                    else:
                        direction = "long"
                        entry_price = current_price
                        stop_price = gap_open * (1 - self._fade_stop_pct / 100)
                        target_price = fill_target
                        candidate_type = CandidateType.LONG_EQUITY

                risk_per_share = abs(entry_price - stop_price)
                reward_per_share = abs(target_price - entry_price)
                if risk_per_share <= 0:
                    continue
                rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0

                pre_market_volume = float(c.get("pre_market_volume", 0.0))
                confidence = 0.10
                if abs(gap_signed) > 4.0:
                    confidence += 0.15
                else:
                    confidence += 0.10

                if pre_market_volume > 1_000_000:
                    confidence += 0.15
                elif pre_market_volume > 500_000:
                    confidence += 0.10

                confidence += 0.15 if has_catalyst else 0.10
                if first_high > first_low:
                    confidence += 0.10

                if et_now.hour < 10:
                    confidence += 0.10
                elif et_now.hour == 10 and et_now.minute <= 30:
                    confidence += 0.05

                regime_fit = 0.6
                if regime:
                    regime_type = regime.regime_type.value
                    if regime_type == "high_vol":
                        regime_fit = 0.25
                    elif regime_type in ("trending_up", "trending_down"):
                        regime_fit = 0.65
                    elif regime_type == "ranging":
                        regime_fit = 0.55
                    else:
                        regime_fit = 0.5

                confidence = min(1.0, max(0.0, confidence))
                win_rate = 0.47 if has_catalyst else 0.52
                risk_pct = (risk_per_share / entry_price) * 100.0 if entry_price > 0 else 0.0
                edge_pct = max(0.0, (win_rate * rr_ratio - (1.0 - win_rate)) * risk_pct)
                risk_budget = self.available_capital * 0.005
                suggested_qty = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0

                trade_candidates.append(TradeCandidate(
                    strategy_name=self.name,
                    candidate_type=candidate_type,
                    symbol=symbol,
                    direction=direction,
                    entry_price=round(entry_price, 2),
                    stop_price=round(stop_price, 2),
                    target_price=round(target_price, 2),
                    risk_dollars=round(risk_per_share * max(1, suggested_qty), 2),
                    suggested_qty=suggested_qty,
                    risk_reward_ratio=round(rr_ratio, 2),
                    confidence=round(confidence, 2),
                    edge_estimate_pct=round(edge_pct, 2),
                    regime_fit=round(regime_fit, 2),
                    catalyst=(
                        f"gap_{'up' if gap_signed > 0 else 'down'}_{abs(gap_signed):.1f}pct_"
                        f"{trade_type}_{'catalyst' if has_catalyst else 'no_catalyst'}"
                    ),
                    time_horizon_minutes=90,
                    expiry_time=expiry_time,
                    metadata={
                        "gap_pct": round(gap_signed, 2),
                        "trade_type": trade_type,
                        "first_high": round(first_high, 2),
                        "first_low": round(first_low, 2),
                        "pre_market_volume": pre_market_volume,
                    },
                ))
                details.append(
                    {
                        "symbol": symbol,
                        "gap_pct": round(gap_signed, 2),
                        "trade_type": trade_type,
                        "direction": direction,
                        "has_catalyst": has_catalyst,
                    }
                )
                qualified.append(c)

            if not qualified:
                return _complete(OpportunityAssessment())

            if not trade_candidates:
                return _complete(OpportunityAssessment(
                    num_candidates=0,
                    confidence=0.0,
                    details=[{"reason": "qualified_but_no_tradecandidate"}],
                    candidates=[],
                ))

            trade_candidates.sort(key=lambda c: c.expected_value, reverse=True)
            top_candidates = trade_candidates[:3]
            avg_rr = sum(c.risk_reward_ratio for c in trade_candidates) / len(trade_candidates)
            avg_conf = sum(c.confidence for c in trade_candidates) / len(trade_candidates)
            avg_edge = sum(c.edge_estimate_pct for c in trade_candidates) / len(trade_candidates)

            return _complete(OpportunityAssessment(
                num_candidates=len(trade_candidates),
                avg_risk_reward=round(max(0.0, avg_rr), 2),
                confidence=round(max(0.0, min(1.0, avg_conf)), 2),
                estimated_daily_trades=min(len(trade_candidates), self.config.max_positions),
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
        for symbol, trade in self._trades.items():
            base["trades"][symbol] = {
                "symbol": trade.symbol,
                "trade_type": trade.trade_type,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
                "gap_pct": trade.gap_pct,
                "prev_close": trade.prev_close,
                "gap_open": trade.gap_open,
                "entry_time": trade.entry_time.isoformat(),
                "capital_used": trade.capital_used,
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
            self._trades[symbol] = GapTrade(
                symbol=saved["symbol"],
                trade_type=saved["trade_type"],
                direction=saved["direction"],
                entry_price=saved["entry_price"],
                stop_price=saved["stop_price"],
                target_price=saved["target_price"],
                gap_pct=saved["gap_pct"],
                prev_close=saved["prev_close"],
                gap_open=saved["gap_open"],
                entry_time=datetime.fromisoformat(saved["entry_time"]),
                capital_used=saved.get("capital_used", 0.0),
                waiting_for_entry=False,
                trade_id=saved.get("trade_id", ""),
                bracket_stop_order_id=saved.get("bracket_stop_order_id", ""),
                bracket_tp_order_id=saved.get("bracket_tp_order_id", ""),
                is_bracket=saved.get("is_bracket", False),
            )
