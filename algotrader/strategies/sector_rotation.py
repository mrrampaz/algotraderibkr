"""Sector Rotation strategy.

Trades relative strength divergences between sector ETFs. Long the
strongest sectors, short the weakest relative to SPY.
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
from algotrader.strategies.base import StrategyBase
from algotrader.strategies.registry import register_strategy

logger = structlog.get_logger()

ET = pytz.timezone("America/New_York")


def calculate_relative_strength(
    sector_bars, benchmark_bars, lookback: int = 5,
) -> float | None:
    """Calculate relative strength of a sector vs benchmark.

    RS = (sector N-day return) - (benchmark N-day return)
    Positive RS = sector outperforming.
    Negative RS = sector underperforming.
    """
    if sector_bars.empty or benchmark_bars.empty:
        return None
    if len(sector_bars) < lookback or len(benchmark_bars) < lookback:
        return None

    sector_return = (
        float(sector_bars["close"].iloc[-1]) / float(sector_bars["close"].iloc[-lookback]) - 1
    ) * 100
    bench_return = (
        float(benchmark_bars["close"].iloc[-1]) / float(benchmark_bars["close"].iloc[-lookback]) - 1
    ) * 100

    return sector_return - bench_return


@dataclass
class SectorTrade:
    """Internal state for a single sector rotation trade."""

    symbol: str
    sector_name: str
    direction: str  # "long" or "short"
    entry_price: float
    entry_rs: float  # Relative strength at entry
    stop_price: float
    entry_time: datetime
    capital_used: float = 0.0
    last_rebalance: datetime | None = None
    trade_id: str = ""


@register_strategy("sector_rotation")
class SectorRotationStrategy(StrategyBase):
    """Sector rotation: long strongest, short weakest sectors vs SPY.

    Rebalances every N hours based on rolling relative strength.
    Equal-weight positions across active sectors.
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
        self._sectors: dict[str, str] = params.get("sectors", {
            "XLK": "Technology",
            "XLF": "Financials",
            "XLE": "Energy",
            "XLV": "Healthcare",
            "XLI": "Industrials",
            "XLP": "Consumer Staples",
            "XLU": "Utilities",
            "XLB": "Materials",
            "XLRE": "Real Estate",
            "XLC": "Communication",
            "XLY": "Consumer Discretionary",
        })
        self._benchmark = params.get("benchmark", "SPY")

        # Relative strength
        self._rs_lookback_days = params.get("rs_lookback_days", 5)
        self._rs_long_threshold = params.get("rs_long_threshold", 1.5)
        self._rs_short_threshold = params.get("rs_short_threshold", -1.5)
        self._min_divergence_pct = params.get("min_divergence_pct", 2.0)

        # Position management
        self._rebalance_interval_hours = params.get("rebalance_interval_hours", 4)
        self._stop_pct = params.get("stop_pct", 2.0)

        # Regime filter
        self._allowed_regimes = params.get(
            "allowed_regimes",
            ["trending_up", "trending_down", "ranging"],
        )

        # Internal state
        self._trades: dict[str, SectorTrade] = {}
        self._last_rebalance: datetime | None = None
        self._rs_scores: dict[str, float] = {}

    def warm_up(self) -> None:
        """Calculate initial relative strength scores."""
        self._log.info("warming_up", sectors=len(self._sectors))
        self._calculate_all_rs()
        self._warmed_up = True

    def run_cycle(self, regime: MarketRegime | None = None) -> list[Signal]:
        """Run one cycle: rebalance if interval elapsed."""
        self._check_day_reset()
        self._last_cycle_time = datetime.now(pytz.UTC)
        signals: list[Signal] = []

        # 1. Check regime filter
        if regime and regime.regime_type.value not in self._allowed_regimes:
            return signals

        # 2. Check if it's time to rebalance
        if not self._should_rebalance():
            return signals

        # 3. Calculate all RS scores
        self._calculate_all_rs()

        # 4. Manage existing positions
        signals.extend(self._manage_positions())

        # 5. Open new positions
        signals.extend(self._rebalance())

        self._last_rebalance = datetime.now(pytz.UTC)
        return signals

    def _should_rebalance(self) -> bool:
        """Check if enough time has passed since last rebalance."""
        if self._last_rebalance is None:
            return True
        elapsed = (datetime.now(pytz.UTC) - self._last_rebalance).total_seconds() / 3600
        return elapsed >= self._rebalance_interval_hours

    def _calculate_all_rs(self) -> None:
        """Calculate relative strength for all sectors vs benchmark."""
        benchmark_bars = self.data_provider.get_bars(
            self._benchmark, "1Day", self._rs_lookback_days + 5,
        )
        if benchmark_bars.empty:
            self._log.warning("benchmark_bars_empty")
            return

        self._rs_scores.clear()

        for symbol, sector_name in self._sectors.items():
            try:
                sector_bars = self.data_provider.get_bars(
                    symbol, "1Day", self._rs_lookback_days + 5,
                )
                rs = calculate_relative_strength(
                    sector_bars, benchmark_bars, self._rs_lookback_days,
                )
                if rs is not None:
                    self._rs_scores[symbol] = round(rs, 3)
            except Exception:
                self._log.debug("rs_calc_failed", symbol=symbol)

        if self._rs_scores:
            sorted_rs = sorted(self._rs_scores.items(), key=lambda x: x[1], reverse=True)
            self._log.info(
                "rs_scores_updated",
                strongest=f"{sorted_rs[0][0]} {sorted_rs[0][1]:+.2f}%" if sorted_rs else "none",
                weakest=f"{sorted_rs[-1][0]} {sorted_rs[-1][1]:+.2f}%" if sorted_rs else "none",
                count=len(sorted_rs),
            )

    def _manage_positions(self) -> list[Signal]:
        """Check existing positions for stop/rebalance exits."""
        signals: list[Signal] = []

        for symbol, trade in list(self._trades.items()):
            close_reason = ""

            # Get current price
            current_price = self._get_current_price(symbol)
            if current_price is None:
                continue

            current_rs = self._rs_scores.get(symbol, 0.0)

            # 1. Stop loss
            if trade.direction == "long" and current_price <= trade.stop_price:
                close_reason = f"stop_hit: {current_price:.2f} <= {trade.stop_price:.2f}"
            elif trade.direction == "short" and current_price >= trade.stop_price:
                close_reason = f"stop_hit: {current_price:.2f} >= {trade.stop_price:.2f}"

            # 2. RS reversal: long position lost relative strength
            elif trade.direction == "long" and current_rs < 0:
                close_reason = f"rs_reversal: RS={current_rs:+.2f}% dropped below 0"

            # 3. RS reversal: short position gained relative strength
            elif trade.direction == "short" and current_rs > 0:
                close_reason = f"rs_reversal: RS={current_rs:+.2f}% rose above 0"

            if close_reason:
                self._close_trade(symbol, trade, close_reason)
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    direction=SignalDirection.CLOSE,
                    reason=close_reason,
                    metadata={"sector": trade.sector_name, "rs": current_rs},
                    timestamp=datetime.now(pytz.UTC),
                ))

        return signals

    def _rebalance(self) -> list[Signal]:
        """Open new sector positions based on RS rankings."""
        signals: list[Signal] = []

        if not self._rs_scores:
            return signals

        sorted_rs = sorted(self._rs_scores.items(), key=lambda x: x[1], reverse=True)

        # Check divergence: if spread between best and worst is too small, skip
        if len(sorted_rs) >= 2:
            divergence = sorted_rs[0][1] - sorted_rs[-1][1]
            if divergence < self._min_divergence_pct:
                self._log.debug("insufficient_divergence", divergence=divergence)
                return signals

        # Long top sectors above threshold
        for symbol, rs in sorted_rs:
            if rs < self._rs_long_threshold:
                break
            if symbol in self._trades:
                continue
            if len(self._trades) >= self.config.max_positions:
                break

            signal = self._enter_position(symbol, "long", rs)
            if signal:
                signals.append(signal)

        # Short bottom sectors below threshold
        for symbol, rs in reversed(sorted_rs):
            if rs > self._rs_short_threshold:
                break
            if symbol in self._trades:
                continue
            if len(self._trades) >= self.config.max_positions:
                break

            signal = self._enter_position(symbol, "short", rs)
            if signal:
                signals.append(signal)

        return signals

    def _enter_position(self, symbol: str, direction: str, rs: float) -> Signal | None:
        """Enter a sector rotation position."""
        current_price = self._get_current_price(symbol)
        if current_price is None or current_price <= 0:
            return None

        # Equal-weight sizing across max_positions
        per_position_capital = self._total_capital * (self.config.capital_allocation_pct / 100) / self.config.max_positions
        shares = int(per_position_capital / current_price)

        if shares <= 0:
            return None

        side = OrderSide.BUY if direction == "long" else OrderSide.SELL

        if direction == "long":
            stop_price = current_price * (1 - self._stop_pct / 100)
        else:
            stop_price = current_price * (1 + self._stop_pct / 100)

        capital_needed = shares * current_price
        if not self.reserve_capital(capital_needed):
            return None

        client_id = f"sector_{symbol}_{str(uuid.uuid4())[:8]}"
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
            return None

        sector_name = self._sectors.get(symbol, "Unknown")
        trade = SectorTrade(
            symbol=symbol,
            sector_name=sector_name,
            direction=direction,
            entry_price=current_price,
            entry_rs=rs,
            stop_price=stop_price,
            entry_time=datetime.now(pytz.UTC),
            capital_used=capital_needed,
            last_rebalance=datetime.now(pytz.UTC),
            trade_id=str(uuid.uuid4()),
        )
        self._trades[symbol] = trade

        self._log.info(
            "sector_entry",
            symbol=symbol,
            sector=sector_name,
            direction=direction,
            rs=round(rs, 2),
            price=current_price,
            stop=stop_price,
            shares=shares,
        )

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            direction=SignalDirection.LONG if direction == "long" else SignalDirection.SHORT,
            reason=f"sector_rotation: {sector_name} RS={rs:+.2f}%",
            metadata={"sector": sector_name, "rs": rs},
            timestamp=datetime.now(pytz.UTC),
        )

    def _close_trade(self, symbol: str, trade: SectorTrade, reason: str) -> None:
        """Close a sector trade with position exit safety."""
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
            "sector_exit",
            symbol=symbol,
            sector=trade.sector_name,
            reason=reason,
            pnl=round(pnl, 2),
            entry_rs=round(trade.entry_rs, 2),
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

    def _get_state(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        base = super()._get_state()
        base["trades"] = {}
        for symbol, trade in self._trades.items():
            base["trades"][symbol] = {
                "symbol": trade.symbol,
                "sector_name": trade.sector_name,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "entry_rs": trade.entry_rs,
                "stop_price": trade.stop_price,
                "entry_time": trade.entry_time.isoformat(),
                "capital_used": trade.capital_used,
                "last_rebalance": trade.last_rebalance.isoformat() if trade.last_rebalance else None,
                "trade_id": trade.trade_id,
            }
        base["last_rebalance"] = self._last_rebalance.isoformat() if self._last_rebalance else None
        return base

    def _restore_state(self, state_data: dict[str, Any]) -> None:
        """Restore state from persistence."""
        super()._restore_state(state_data)
        if state_data.get("last_rebalance"):
            self._last_rebalance = datetime.fromisoformat(state_data["last_rebalance"])
        for symbol, saved in state_data.get("trades", {}).items():
            self._trades[symbol] = SectorTrade(
                symbol=saved["symbol"],
                sector_name=saved["sector_name"],
                direction=saved["direction"],
                entry_price=saved["entry_price"],
                entry_rs=saved["entry_rs"],
                stop_price=saved["stop_price"],
                entry_time=datetime.fromisoformat(saved["entry_time"]),
                capital_used=saved.get("capital_used", 0.0),
                last_rebalance=datetime.fromisoformat(saved["last_rebalance"]) if saved.get("last_rebalance") else None,
                trade_id=saved.get("trade_id", ""),
            )
