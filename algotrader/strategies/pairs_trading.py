"""Statistical pairs trading strategy.

Rewritten for the new architecture using DataProvider and Executor abstractions.
Implements: Engle-Granger cointegration tests, OLS hedge ratio, z-score signals,
entry/exit rules with rollback safety.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytz
import structlog
from scipy import stats
from statsmodels.tsa.stattools import coint

from algotrader.core.config import StrategyConfig
from algotrader.core.events import EventBus
from algotrader.core.models import (
    MarketRegime, Order, OrderSide, OrderType, Signal, SignalDirection,
    TimeInForce, TradeRecord,
)
from algotrader.data.provider import DataProvider
from algotrader.execution.executor import Executor
from algotrader.strategies.base import StrategyBase
from algotrader.strategies.registry import register_strategy

logger = structlog.get_logger()


@dataclass
class PairConfig:
    """Configuration for a single trading pair."""
    symbol_a: str
    symbol_b: str
    sector: str = ""

    @property
    def pair_id(self) -> str:
        return f"{self.symbol_a}_{self.symbol_b}"


@dataclass
class PairState:
    """Runtime state for a pair being monitored/traded."""
    config: PairConfig
    hedge_ratio: float = 1.0
    spread_mean: float = 0.0
    spread_std: float = 1.0
    z_score: float = 0.0
    correlation: float = 0.0
    cointegration_pvalue: float = 1.0
    is_cointegrated: bool = False
    bars_in_position: int = 0

    # Position tracking
    is_positioned: bool = False
    position_side: str = ""  # "long_spread" or "short_spread"
    entry_z_score: float = 0.0
    entry_time: datetime | None = None
    qty_a: int = 0
    qty_b: int = 0
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0
    trade_id: str = ""


# Default pairs universe across sectors
DEFAULT_PAIRS = [
    PairConfig("XOM", "CVX", "energy"),
    PairConfig("COP", "EOG", "energy"),
    PairConfig("JPM", "BAC", "financials"),
    PairConfig("GS", "MS", "financials"),
    PairConfig("V", "MA", "financials"),
    PairConfig("AAPL", "MSFT", "tech"),
    PairConfig("GOOG", "META", "tech"),
    PairConfig("AMZN", "SHOP", "tech"),
    PairConfig("JNJ", "PFE", "healthcare"),
    PairConfig("UNH", "CI", "healthcare"),
    PairConfig("HD", "LOW", "consumer"),
    PairConfig("KO", "PEP", "consumer"),
    PairConfig("CAT", "DE", "industrials"),
    PairConfig("UPS", "FDX", "industrials"),
    PairConfig("T", "VZ", "telecom"),
    PairConfig("DUK", "SO", "utilities"),
    PairConfig("PLD", "AMT", "reits"),
    PairConfig("BHP", "RIO", "materials"),
]


@register_strategy("pairs_trading")
class PairsTradingStrategy(StrategyBase):
    """Statistical pairs trading using cointegration and z-score signals.

    Logic:
    1. Warm up: Test cointegration on each pair, calculate hedge ratios
    2. Each cycle: Update z-scores, generate entry/exit signals
    3. Entry: Z-score crosses threshold (>1.2 or <-1.2) with cointegration confirmed
    4. Exit: Z-score reverts (<0.2), hits stop (>2.8), or times out (80 bars)
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

        # Strategy parameters from config
        params = config.params
        self._lookback = params.get("lookback_bars", 252)
        self._z_entry = params.get("z_entry_threshold", 1.2)
        self._z_exit = params.get("z_exit_threshold", 0.2)
        self._z_stop = params.get("z_stop_threshold", 2.8)
        self._max_bars = params.get("max_bars_in_position", 80)
        self._min_correlation = params.get("min_correlation", 0.7)
        self._coint_pvalue = params.get("cointegration_pvalue", 0.05)
        self._per_pair_capital_pct = params.get("per_pair_capital_pct", 2.0)
        self._max_pairs = params.get("max_simultaneous_pairs", 5)
        self._bar_timeframe = params.get("bar_timeframe", "5Min")

        # Load pairs from config or use defaults
        pairs_config = params.get("pairs", None)
        if pairs_config:
            self._pairs = [
                PairConfig(
                    symbol_a=p["symbol_a"],
                    symbol_b=p["symbol_b"],
                    sector=p.get("sector", ""),
                )
                for p in pairs_config
            ]
        else:
            self._pairs = DEFAULT_PAIRS

        # Runtime state
        self._pair_states: dict[str, PairState] = {}
        for pair in self._pairs:
            self._pair_states[pair.pair_id] = PairState(config=pair)

    def warm_up(self) -> None:
        """Run cointegration tests and calculate hedge ratios for all pairs."""
        self._log.info("warming_up", num_pairs=len(self._pairs))

        for pair_id, state in self._pair_states.items():
            try:
                self._update_pair_stats(state)
                if state.is_cointegrated:
                    self._log.info(
                        "pair_cointegrated",
                        pair=pair_id,
                        hedge_ratio=round(state.hedge_ratio, 4),
                        correlation=round(state.correlation, 3),
                        pvalue=round(state.cointegration_pvalue, 4),
                    )
                else:
                    self._log.debug(
                        "pair_not_cointegrated",
                        pair=pair_id,
                        pvalue=round(state.cointegration_pvalue, 4),
                        correlation=round(state.correlation, 3),
                    )
            except Exception:
                self._log.exception("warm_up_pair_failed", pair=pair_id)

        cointegrated = sum(1 for s in self._pair_states.values() if s.is_cointegrated)
        self._log.info("warm_up_complete", cointegrated=cointegrated, total=len(self._pairs))
        self._warmed_up = True

    def run_cycle(self, regime: MarketRegime | None = None) -> list[Signal]:
        """Run one trading cycle: update z-scores, check entry/exit conditions."""
        self._check_day_reset()
        self._last_cycle_time = datetime.now(pytz.UTC)
        signals: list[Signal] = []

        for pair_id, state in self._pair_states.items():
            try:
                # Update spread and z-score
                self._update_z_score(state)

                if state.is_positioned:
                    # Check exit conditions
                    exit_signal = self._check_exit(state)
                    if exit_signal:
                        signals.append(exit_signal)
                        self._execute_exit(state, exit_signal)
                else:
                    # Check entry conditions
                    entry_signal = self._check_entry(state)
                    if entry_signal:
                        signals.append(entry_signal)
                        self._execute_entry(state, entry_signal)

            except Exception:
                self._log.exception("cycle_pair_error", pair=pair_id)

        return signals

    def _update_pair_stats(self, state: PairState) -> None:
        """Run cointegration test and calculate hedge ratio for a pair."""
        pair = state.config

        # Get historical daily bars for cointegration test
        bars_a = self.data_provider.get_bars(pair.symbol_a, "1Day", self._lookback)
        bars_b = self.data_provider.get_bars(pair.symbol_b, "1Day", self._lookback)

        if bars_a.empty or bars_b.empty:
            state.is_cointegrated = False
            return

        # Align on common dates
        closes_a = bars_a["close"]
        closes_b = bars_b["close"]
        common_idx = closes_a.index.intersection(closes_b.index)

        if len(common_idx) < 60:  # Need at least 60 data points
            state.is_cointegrated = False
            return

        prices_a = closes_a.loc[common_idx].values
        prices_b = closes_b.loc[common_idx].values

        # Correlation check
        state.correlation = float(np.corrcoef(prices_a, prices_b)[0, 1])
        if state.correlation < self._min_correlation:
            state.is_cointegrated = False
            return

        # Engle-Granger cointegration test
        try:
            score, pvalue, _ = coint(prices_a, prices_b)
            state.cointegration_pvalue = float(pvalue)
            state.is_cointegrated = pvalue < self._coint_pvalue
        except Exception:
            state.is_cointegrated = False
            return

        if not state.is_cointegrated:
            return

        # OLS hedge ratio: A = beta * B + alpha
        slope, intercept, _, _, _ = stats.linregress(prices_b, prices_a)
        state.hedge_ratio = float(slope)

        # Calculate spread statistics
        spread = prices_a - state.hedge_ratio * prices_b
        state.spread_mean = float(np.mean(spread))
        state.spread_std = float(np.std(spread))
        if state.spread_std == 0:
            state.is_cointegrated = False

    def _update_z_score(self, state: PairState) -> None:
        """Update the current z-score for a pair using intraday bars."""
        pair = state.config

        if not state.is_cointegrated and not state.is_positioned:
            return

        # Get recent bars for current spread value
        bars_a = self.data_provider.get_bars(pair.symbol_a, self._bar_timeframe, 5)
        bars_b = self.data_provider.get_bars(pair.symbol_b, self._bar_timeframe, 5)

        if bars_a.empty or bars_b.empty:
            return

        price_a = float(bars_a["close"].iloc[-1])
        price_b = float(bars_b["close"].iloc[-1])

        current_spread = price_a - state.hedge_ratio * price_b
        if state.spread_std > 0:
            state.z_score = (current_spread - state.spread_mean) / state.spread_std

        if state.is_positioned:
            state.bars_in_position += 1

    def _check_entry(self, state: PairState) -> Signal | None:
        """Check if we should enter a pairs trade."""
        if not state.is_cointegrated:
            return None

        # Check if we've hit max simultaneous pairs
        active_pairs = sum(1 for s in self._pair_states.values() if s.is_positioned)
        if active_pairs >= self._max_pairs:
            return None

        # Check capital
        if self.available_capital <= 0:
            return None

        pair = state.config
        z = state.z_score

        if abs(z) < self._z_entry:
            return None

        # Z > threshold: spread is wide → short the spread (sell A, buy B)
        # Z < -threshold: spread is narrow → long the spread (buy A, sell B)
        if z > self._z_entry:
            direction = SignalDirection.SHORT
            reason = f"z_score={z:.2f} > {self._z_entry} → short spread"
        elif z < -self._z_entry:
            direction = SignalDirection.LONG
            reason = f"z_score={z:.2f} < -{self._z_entry} → long spread"
        else:
            return None

        return Signal(
            strategy_name=self.name,
            symbol=f"{pair.symbol_a}/{pair.symbol_b}",
            direction=direction,
            conviction=min(1.0 + (abs(z) - self._z_entry) * 0.3, 1.5),
            reason=reason,
            metadata={
                "pair_id": pair.pair_id,
                "z_score": z,
                "hedge_ratio": state.hedge_ratio,
                "correlation": state.correlation,
            },
            timestamp=datetime.now(pytz.UTC),
        )

    def _check_exit(self, state: PairState) -> Signal | None:
        """Check if we should exit a pairs position."""
        pair = state.config
        z = state.z_score

        reason = ""

        # Mean reversion profit target
        if state.position_side == "long_spread" and z >= -self._z_exit:
            reason = f"profit_target: z={z:.2f} reverted above -{self._z_exit}"
        elif state.position_side == "short_spread" and z <= self._z_exit:
            reason = f"profit_target: z={z:.2f} reverted below {self._z_exit}"
        # Stop loss
        elif state.position_side == "long_spread" and z < -self._z_stop:
            reason = f"stop_loss: z={z:.2f} < -{self._z_stop}"
        elif state.position_side == "short_spread" and z > self._z_stop:
            reason = f"stop_loss: z={z:.2f} > {self._z_stop}"
        # Time stop
        elif state.bars_in_position >= self._max_bars:
            reason = f"time_stop: {state.bars_in_position} bars >= {self._max_bars}"

        if not reason:
            return None

        return Signal(
            strategy_name=self.name,
            symbol=f"{pair.symbol_a}/{pair.symbol_b}",
            direction=SignalDirection.CLOSE,
            reason=reason,
            metadata={
                "pair_id": pair.pair_id,
                "z_score": z,
                "bars_in_position": state.bars_in_position,
                "position_side": state.position_side,
            },
            timestamp=datetime.now(pytz.UTC),
        )

    def _execute_entry(self, state: PairState, signal: Signal) -> None:
        """Execute a pairs entry with rollback safety."""
        pair = state.config
        log = self._log.bind(pair=pair.pair_id, direction=signal.direction.value)

        # Calculate position sizes
        bars_a = self.data_provider.get_bars(pair.symbol_a, self._bar_timeframe, 1)
        bars_b = self.data_provider.get_bars(pair.symbol_b, self._bar_timeframe, 1)

        if bars_a.empty or bars_b.empty:
            log.warning("entry_skipped_no_prices")
            return

        price_a = float(bars_a["close"].iloc[-1])
        price_b = float(bars_b["close"].iloc[-1])

        # Calculate shares: allocate per_pair_capital_pct of total capital
        pair_capital = self._total_capital * (self._per_pair_capital_pct / 100)

        # For leg A
        qty_a = int(pair_capital / (2 * price_a))
        # For leg B: adjust by hedge ratio
        qty_b = int(qty_a * abs(state.hedge_ratio) * price_a / price_b)

        if qty_a <= 0 or qty_b <= 0:
            log.warning("entry_skipped_zero_qty", price_a=price_a, price_b=price_b)
            return

        # Reserve capital
        capital_needed = qty_a * price_a + qty_b * price_b
        if not self.reserve_capital(capital_needed):
            return

        # Determine sides based on signal direction
        if signal.direction == SignalDirection.LONG:
            # Long spread: buy A, sell B
            side_a, side_b = OrderSide.BUY, OrderSide.SELL
            position_side = "long_spread"
        else:
            # Short spread: sell A, buy B
            side_a, side_b = OrderSide.SELL, OrderSide.BUY
            position_side = "short_spread"

        client_id = str(uuid.uuid4())[:8]

        # Submit leg A
        order_a = self.executor.submit_order(
            symbol=pair.symbol_a,
            qty=qty_a,
            side=side_a,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"pairs_{pair.pair_id}_A_{client_id}",
        )

        # Submit leg B
        order_b = self.executor.submit_order(
            symbol=pair.symbol_b,
            qty=qty_b,
            side=side_b,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"pairs_{pair.pair_id}_B_{client_id}",
        )

        # Rollback safety: if one leg fails, close the filled leg
        if order_a and not order_b:
            log.warning("entry_rollback_leg_b_failed")
            self.executor.close_position(pair.symbol_a)
            self.release_capital(capital_needed)
            return
        elif order_b and not order_a:
            log.warning("entry_rollback_leg_a_failed")
            self.executor.close_position(pair.symbol_b)
            self.release_capital(capital_needed)
            return
        elif not order_a and not order_b:
            log.warning("entry_both_legs_failed")
            self.release_capital(capital_needed)
            return

        # Both legs submitted successfully
        state.is_positioned = True
        state.position_side = position_side
        state.entry_z_score = state.z_score
        state.entry_time = datetime.now(pytz.UTC)
        state.qty_a = qty_a
        state.qty_b = qty_b
        state.entry_price_a = price_a
        state.entry_price_b = price_b
        state.bars_in_position = 0
        state.trade_id = str(uuid.uuid4())

        log.info(
            "pair_entry",
            position_side=position_side,
            qty_a=qty_a,
            qty_b=qty_b,
            price_a=price_a,
            price_b=price_b,
            z_score=state.z_score,
        )

    def _execute_exit(self, state: PairState, signal: Signal) -> None:
        """Execute a pairs exit with P&L tracking."""
        pair = state.config
        log = self._log.bind(pair=pair.pair_id)

        # Get current P&L from broker before closing
        broker_pos_a = self.executor.get_position(pair.symbol_a)
        broker_pos_b = self.executor.get_position(pair.symbol_b)

        pnl_a = float(broker_pos_a.unrealized_pnl) if broker_pos_a else 0.0
        pnl_b = float(broker_pos_b.unrealized_pnl) if broker_pos_b else 0.0
        total_pnl = pnl_a + pnl_b

        # Close both legs
        close_a = self.executor.close_position(pair.symbol_a)
        close_b = self.executor.close_position(pair.symbol_b)

        if not close_a:
            log.error("exit_leg_a_failed", symbol=pair.symbol_a)
            return  # Keep in tracking, retry next cycle
        if not close_b:
            log.error("exit_leg_b_failed", symbol=pair.symbol_b)
            return  # Keep in tracking, retry next cycle

        # Release capital
        capital_used = state.qty_a * state.entry_price_a + state.qty_b * state.entry_price_b
        self.release_capital(capital_used)

        # Record trade
        self.record_trade(total_pnl)

        log.info(
            "pair_exit",
            reason=signal.reason,
            pnl=round(total_pnl, 2),
            bars_held=state.bars_in_position,
            z_entry=round(state.entry_z_score, 2),
            z_exit=round(state.z_score, 2),
        )

        # Reset state
        state.is_positioned = False
        state.position_side = ""
        state.entry_z_score = 0.0
        state.entry_time = None
        state.qty_a = 0
        state.qty_b = 0
        state.bars_in_position = 0
        state.trade_id = ""

    def on_fill(self, order: Order) -> None:
        """Handle fill events for pairs legs."""
        super().on_fill(order)
        # Fills are handled in _execute_entry/_execute_exit directly
        # since we need to track both legs together

    def _get_state(self) -> dict[str, Any]:
        """Serialize strategy state for persistence."""
        base = super()._get_state()
        base["pair_states"] = {}
        for pair_id, state in self._pair_states.items():
            if state.is_positioned:
                base["pair_states"][pair_id] = {
                    "position_side": state.position_side,
                    "entry_z_score": state.entry_z_score,
                    "entry_time": state.entry_time.isoformat() if state.entry_time else None,
                    "qty_a": state.qty_a,
                    "qty_b": state.qty_b,
                    "entry_price_a": state.entry_price_a,
                    "entry_price_b": state.entry_price_b,
                    "bars_in_position": state.bars_in_position,
                    "hedge_ratio": state.hedge_ratio,
                    "spread_mean": state.spread_mean,
                    "spread_std": state.spread_std,
                    "trade_id": state.trade_id,
                }
        return base

    def _restore_state(self, state_data: dict[str, Any]) -> None:
        """Restore strategy state from persistence."""
        super()._restore_state(state_data)
        pair_states = state_data.get("pair_states", {})
        for pair_id, saved in pair_states.items():
            if pair_id in self._pair_states:
                ps = self._pair_states[pair_id]
                ps.is_positioned = True
                ps.position_side = saved["position_side"]
                ps.entry_z_score = saved["entry_z_score"]
                ps.entry_time = (
                    datetime.fromisoformat(saved["entry_time"])
                    if saved.get("entry_time")
                    else None
                )
                ps.qty_a = saved["qty_a"]
                ps.qty_b = saved["qty_b"]
                ps.entry_price_a = saved["entry_price_a"]
                ps.entry_price_b = saved["entry_price_b"]
                ps.bars_in_position = saved["bars_in_position"]
                ps.hedge_ratio = saved["hedge_ratio"]
                ps.spread_mean = saved["spread_mean"]
                ps.spread_std = saved["spread_std"]
                ps.trade_id = saved.get("trade_id", "")
