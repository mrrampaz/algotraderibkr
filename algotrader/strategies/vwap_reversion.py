"""VWAP Mean Reversion strategy.

Trades large-cap stocks that deviate significantly from VWAP back toward
VWAP. Only active in ranging/low-vol regimes where mean reversion is
most reliable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
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


def calculate_vwap_zscore(bars: pd.DataFrame) -> tuple[float, float]:
    """Calculate VWAP and z-score from intraday bars.

    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    Z-score = (current_price - VWAP) / std(price - VWAP)
    """
    if bars.empty or len(bars) < 5:
        return 0.0, 0.0

    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_vol = bars["volume"].cumsum()
    cum_pv = (typical_price * bars["volume"]).cumsum()

    # Avoid division by zero
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    vwap = vwap.ffill()

    deviation = bars["close"] - vwap
    std_dev = deviation.rolling(20, min_periods=5).std()

    current_vwap = float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else 0.0
    if pd.isna(std_dev.iloc[-1]) or std_dev.iloc[-1] <= 0:
        return current_vwap, 0.0

    current_z = float(deviation.iloc[-1] / std_dev.iloc[-1])
    return current_vwap, current_z


@dataclass
class VWAPTrade:
    """Internal state for a single VWAP reversion trade."""

    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    entry_vwap: float
    entry_z_score: float
    stop_price: float
    target_price: float
    entry_time: datetime
    capital_used: float = 0.0
    trade_id: str = ""


@register_strategy("vwap_reversion")
class VWAPReversionStrategy(StrategyBase):
    """VWAP mean reversion for large-cap stocks.

    Entry: when z-score from VWAP exceeds threshold (price far from VWAP).
    Exit: when z-score reverts, stop loss, profit target, or time limit.
    Only active in ranging/low-vol regimes.
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
        self._universe = params.get("universe", [
            "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "JPM", "V",
        ])

        # Entry
        self._min_z_score = params.get("min_z_score", 2.0)
        self._min_volume_ratio = params.get("min_volume_ratio", 1.0)
        self._max_spread_pct = params.get("max_spread_pct", 0.15)

        # Exit
        self._exit_z_score = params.get("exit_z_score", 0.5)
        self._stop_pct = params.get("stop_pct", 0.5)
        self._target_pct = params.get("target_pct", 1.0)

        # Time
        self._earliest_entry_hour = params.get("earliest_entry_hour", 10)
        self._earliest_entry_minute = params.get("earliest_entry_minute", 0)
        self._close_by_hour = params.get("close_by_hour", 15)
        self._close_by_minute = params.get("close_by_minute", 30)
        self._bar_timeframe = params.get("bar_timeframe", "5Min")
        self._vwap_lookback_bars = params.get("vwap_lookback_bars", 78)

        # Regime filter — ONLY ranging/low_vol
        self._allowed_regimes = params.get("allowed_regimes", ["ranging", "low_vol"])

        # Internal state
        self._trades: dict[str, VWAPTrade] = {}

    def warm_up(self) -> None:
        """No heavy warm-up needed; VWAP calculated from intraday bars."""
        self._log.info("warming_up", universe_size=len(self._universe))
        self._warmed_up = True

    def run_cycle(self, regime: MarketRegime | None = None) -> list[Signal]:
        """Run one cycle: manage positions, scan for VWAP deviations."""
        self._check_day_reset()
        self._last_cycle_time = datetime.now(pytz.UTC)
        signals: list[Signal] = []

        et_now = datetime.now(ET)

        # 1. Manage existing positions FIRST (always, regardless of regime)
        signals.extend(self._manage_positions(et_now))

        # 2. Check regime filter (only gates new entries)
        if regime and regime.regime_type.value not in self._allowed_regimes:
            return signals

        # 3. Check time window for new entries
        if (et_now.hour < self._earliest_entry_hour or
                (et_now.hour == self._earliest_entry_hour and et_now.minute < self._earliest_entry_minute)):
            return signals

        # 4. Scan for new entries
        if len(self._trades) < self.config.max_positions:
            signals.extend(self._scan_entries(et_now))

        return signals

    def _manage_positions(self, et_now: datetime) -> list[Signal]:
        """Manage open VWAP reversion trades."""
        signals: list[Signal] = []

        for symbol, trade in list(self._trades.items()):
            close_reason = ""

            # Get current price and VWAP z-score
            current_price = self._get_current_price(symbol)
            if current_price is None:
                continue

            bars = self.data_provider.get_bars(symbol, self._bar_timeframe, self._vwap_lookback_bars)
            _, current_z = calculate_vwap_zscore(bars) if not bars.empty else (0.0, 0.0)

            # 1. Z-score reverted — profit target
            if abs(current_z) <= self._exit_z_score:
                close_reason = f"z_reverted: |{current_z:.2f}| <= {self._exit_z_score}"

            # 2. Stop loss
            elif trade.direction == "long" and current_price <= trade.stop_price:
                close_reason = f"stop_hit: {current_price:.2f} <= {trade.stop_price:.2f}"
            elif trade.direction == "short" and current_price >= trade.stop_price:
                close_reason = f"stop_hit: {current_price:.2f} >= {trade.stop_price:.2f}"

            # 3. Profit target by percentage
            elif trade.direction == "long" and current_price >= trade.target_price:
                close_reason = f"target_hit: {current_price:.2f} >= {trade.target_price:.2f}"
            elif trade.direction == "short" and current_price <= trade.target_price:
                close_reason = f"target_hit: {current_price:.2f} <= {trade.target_price:.2f}"

            # 4. Time limit
            elif (et_now.hour > self._close_by_hour or
                    (et_now.hour == self._close_by_hour and et_now.minute >= self._close_by_minute)):
                close_reason = "time_limit"

            if close_reason:
                self._close_trade(symbol, trade, close_reason)
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    direction=SignalDirection.CLOSE,
                    reason=close_reason,
                    metadata={"entry_z": trade.entry_z_score, "current_z": current_z},
                    timestamp=datetime.now(pytz.UTC),
                ))

        return signals

    def _scan_entries(self, et_now: datetime) -> list[Signal]:
        """Scan universe for VWAP deviation entries."""
        signals: list[Signal] = []

        for symbol in self._universe:
            if symbol in self._trades:
                continue
            if len(self._trades) >= self.config.max_positions:
                break

            # Get intraday bars for VWAP calculation
            bars = self.data_provider.get_bars(symbol, self._bar_timeframe, self._vwap_lookback_bars)
            if bars.empty or len(bars) < 10:
                continue

            current_vwap, z_score = calculate_vwap_zscore(bars)
            if current_vwap <= 0:
                continue

            # Check z-score threshold
            if abs(z_score) < self._min_z_score:
                continue

            # Get current price
            current_price = float(bars["close"].iloc[-1])
            if current_price <= 0:
                continue

            # Validate spread
            quote = self.data_provider.get_quote(symbol)
            if quote and quote.is_valid() and quote.spread_pct * 100 > self._max_spread_pct:
                continue

            # Determine direction
            if z_score > self._min_z_score:
                # Price far above VWAP → SHORT (expect reversion down)
                direction = "short"
                side = OrderSide.SELL
                stop_price = current_price * (1 + self._stop_pct / 100)
                target_price = current_price * (1 - self._target_pct / 100)
            elif z_score < -self._min_z_score:
                # Price far below VWAP → LONG (expect reversion up)
                direction = "long"
                side = OrderSide.BUY
                stop_price = current_price * (1 - self._stop_pct / 100)
                target_price = current_price * (1 + self._target_pct / 100)
            else:
                continue

            # Size position
            stop_distance = abs(current_price - stop_price)
            if stop_distance <= 0:
                continue

            risk_amount = self.available_capital * 0.0035
            shares = int(risk_amount / stop_distance)
            max_value = self._total_capital * 0.05
            max_shares = int(max_value / current_price) if current_price > 0 else 0
            shares = min(shares, max_shares)

            if shares <= 0:
                continue

            capital_needed = shares * current_price
            if not self.reserve_capital(capital_needed):
                continue

            client_id = f"vwap_{symbol}_{str(uuid.uuid4())[:8]}"
            order = self.executor.submit_order(
                symbol=symbol,
                qty=shares,
                side=side,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_id,
            )

            if not order:
                self.release_capital(capital_needed)
                continue

            trade = VWAPTrade(
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                entry_vwap=current_vwap,
                entry_z_score=z_score,
                stop_price=stop_price,
                target_price=target_price,
                entry_time=datetime.now(pytz.UTC),
                capital_used=capital_needed,
                trade_id=str(uuid.uuid4()),
            )
            self._trades[symbol] = trade

            self._log.info(
                "vwap_entry",
                symbol=symbol,
                direction=direction,
                z_score=round(z_score, 2),
                vwap=round(current_vwap, 2),
                price=current_price,
                stop=stop_price,
                target=target_price,
                shares=shares,
            )

            signals.append(Signal(
                strategy_name=self.name,
                symbol=symbol,
                direction=SignalDirection.LONG if direction == "long" else SignalDirection.SHORT,
                reason=f"vwap_deviation: z={z_score:.2f}",
                metadata={"z_score": z_score, "vwap": current_vwap},
                timestamp=datetime.now(pytz.UTC),
            ))

        return signals

    def _close_trade(self, symbol: str, trade: VWAPTrade, reason: str) -> None:
        """Close a VWAP trade with position exit safety."""
        broker_pos = self.executor.get_position(symbol)
        pnl = float(broker_pos.unrealized_pnl) if broker_pos else 0.0

        close_success = self.executor.close_position(symbol)
        if not close_success:
            self._log.error("close_failed", symbol=symbol)
            return  # Keep in tracking, retry next cycle

        self._trades.pop(symbol, None)
        self.release_capital(trade.capital_used)
        self.record_trade(pnl)

        self._log.info(
            "vwap_exit",
            symbol=symbol,
            reason=reason,
            pnl=round(pnl, 2),
            direction=trade.direction,
            entry_z=round(trade.entry_z_score, 2),
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
        """Assess VWAP deviation opportunities across universe."""
        try:
            # Regime gate — only active in ranging/low_vol
            if regime and regime.regime_type.value not in self._allowed_regimes:
                return OpportunityAssessment()

            candidates = []
            for symbol in self._universe:
                if symbol in self._trades:
                    continue
                try:
                    bars = self.data_provider.get_bars(
                        symbol, self._bar_timeframe, self._vwap_lookback_bars,
                    )
                    if bars.empty or len(bars) < 10:
                        continue
                    _, z_score = calculate_vwap_zscore(bars)
                    if abs(z_score) >= self._min_z_score:
                        candidates.append({"symbol": symbol, "z_score": round(z_score, 2)})
                except Exception:
                    continue

            if not candidates:
                return OpportunityAssessment()

            avg_z = sum(abs(c["z_score"]) for c in candidates) / len(candidates)
            confidence = min(1.0, len(candidates) * 0.12 + (avg_z - self._min_z_score) * 0.15)

            return OpportunityAssessment(
                num_candidates=len(candidates),
                avg_risk_reward=self._target_pct / self._stop_pct if self._stop_pct > 0 else 2.0,
                confidence=round(max(0.0, confidence), 2),
                estimated_daily_trades=min(len(candidates), self.config.max_positions),
                estimated_edge_pct=round(0.15 * avg_z, 2),
                details=candidates[:5],
            )
        except Exception:
            self._log.debug("assess_opportunities_failed")
            return OpportunityAssessment()

    def _get_state(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        base = super()._get_state()
        base["trades"] = {}
        for symbol, trade in self._trades.items():
            base["trades"][symbol] = {
                "symbol": trade.symbol,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "entry_vwap": trade.entry_vwap,
                "entry_z_score": trade.entry_z_score,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
                "entry_time": trade.entry_time.isoformat(),
                "capital_used": trade.capital_used,
                "trade_id": trade.trade_id,
            }
        return base

    def _restore_state(self, state_data: dict[str, Any]) -> None:
        """Restore state from persistence."""
        super()._restore_state(state_data)
        for symbol, saved in state_data.get("trades", {}).items():
            self._trades[symbol] = VWAPTrade(
                symbol=saved["symbol"],
                direction=saved["direction"],
                entry_price=saved["entry_price"],
                entry_vwap=saved["entry_vwap"],
                entry_z_score=saved["entry_z_score"],
                stop_price=saved["stop_price"],
                target_price=saved["target_price"],
                entry_time=datetime.fromisoformat(saved["entry_time"]),
                capital_used=saved.get("capital_used", 0.0),
                trade_id=saved.get("trade_id", ""),
            )
