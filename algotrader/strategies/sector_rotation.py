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
from algotrader.strategies.base import OpportunityAssessment, StrategyBase
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
    bracket_stop_order_id: str = ""
    is_bracket: bool = False


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
        """Run one cycle: manage positions, rebalance if interval elapsed."""
        self._check_day_reset()
        self._last_cycle_time = datetime.now(pytz.UTC)
        signals: list[Signal] = []

        # 1. Calculate RS scores and manage existing positions FIRST (always)
        if self._trades:
            self._calculate_all_rs()
            signals.extend(self._manage_positions())

        # 2. Check regime filter (only gates new entries / rebalance)
        if regime and regime.regime_type.value not in self._allowed_regimes:
            return signals

        # 3. Check if it's time to rebalance
        if not self._should_rebalance():
            return signals

        # 4. Calculate RS scores for rebalance (if not already done above)
        if not self._trades:
            self._calculate_all_rs()

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

        # Equal-weight sizing across max_positions (_total_capital is already allocated)
        per_position_capital = self._total_capital / self.config.max_positions
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
            bracket_stop_price=stop_price,
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
            bracket_stop_order_id=order.stop_order_id,
            is_bracket=order.is_bracket,
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
            entry_reason=f"sector_rotation: {trade.sector_name} RS={trade.entry_rs:+.2f}%",
            exit_reason=reason,
            metadata={"sector": trade.sector_name, "entry_rs": trade.entry_rs},
        )

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

    def assess_opportunities(self, regime: MarketRegime | None = None) -> OpportunityAssessment:
        """Assess sector rotation opportunities via relative strength."""
        try:
            from algotrader.strategy_selector.candidate import CandidateType, TradeCandidate

            # Regime gate
            if regime and regime.regime_type.value not in self._allowed_regimes:
                return OpportunityAssessment()

            # Calculate RS scores
            self._calculate_all_rs()

            if not self._rs_scores:
                return OpportunityAssessment()

            sorted_rs = sorted(self._rs_scores.items(), key=lambda x: x[1], reverse=True)

            # Check divergence threshold
            if len(sorted_rs) >= 2:
                divergence = sorted_rs[0][1] - sorted_rs[-1][1]
                if divergence < self._min_divergence_pct:
                    return OpportunityAssessment(
                        num_candidates=0,
                        confidence=0.0,
                        details=[{"divergence": round(divergence, 2), "min": self._min_divergence_pct}],
                    )

            # Count sectors above long / below short thresholds
            longs = [s for s, rs in sorted_rs if rs >= self._rs_long_threshold and s not in self._trades]
            shorts = [s for s, rs in sorted_rs if rs <= self._rs_short_threshold and s not in self._trades]
            candidates = longs + shorts

            if not candidates:
                return OpportunityAssessment()

            divergence = sorted_rs[0][1] - sorted_rs[-1][1] if len(sorted_rs) >= 2 else 0.0
            if not longs or not shorts:
                return OpportunityAssessment(
                    num_candidates=0,
                    confidence=0.0,
                    details=[{"reason": "need_both_long_and_short_legs", "divergence": round(divergence, 2)}],
                    candidates=[],
                )

            long_symbol = longs[0]
            short_symbol = shorts[0]
            long_rs = self._rs_scores.get(long_symbol, 0.0)
            short_rs = self._rs_scores.get(short_symbol, 0.0)
            long_price = self._get_current_price(long_symbol)
            short_price = self._get_current_price(short_symbol)

            if not long_price or long_price <= 0:
                return OpportunityAssessment()

            stop_price = long_price * (1 - self._stop_pct / 100)
            target_price = long_price + (long_price * self._stop_pct / 100 * 1.5)
            risk_per_share = abs(long_price - stop_price)
            reward_per_share = abs(target_price - long_price)
            rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0

            confidence = 0.10
            if divergence > 3.0:
                confidence += 0.25
            elif divergence >= 2.0:
                confidence += 0.15

            if abs(long_rs) >= self._rs_long_threshold * 1.2 and abs(short_rs) >= abs(self._rs_short_threshold) * 1.2:
                confidence += 0.15

            regime_fit = 0.5
            if regime:
                regime_type = regime.regime_type.value
                if regime_type in ("trending_up", "trending_down"):
                    regime_fit = 0.7
                    confidence += 0.20
                elif regime_type == "ranging":
                    regime_fit = 0.5
                    confidence += 0.10
                elif regime_type == "high_vol":
                    regime_fit = 0.4
                else:
                    regime_fit = 0.5

            try:
                vol_confirmed = False
                if short_price and short_price > 0:
                    long_bars = self.data_provider.get_bars(long_symbol, "1Day", 6)
                    short_bars = self.data_provider.get_bars(short_symbol, "1Day", 6)
                    if not long_bars.empty and not short_bars.empty and len(long_bars) >= 6 and len(short_bars) >= 6:
                        long_ratio = float(long_bars["volume"].iloc[-1]) / max(1.0, float(long_bars["volume"].iloc[-6:-1].mean()))
                        short_ratio = float(short_bars["volume"].iloc[-1]) / max(1.0, float(short_bars["volume"].iloc[-6:-1].mean()))
                        vol_confirmed = long_ratio >= 1.1 and short_ratio >= 1.1
                if vol_confirmed:
                    confidence += 0.10
            except Exception:
                pass

            confidence = min(1.0, max(0.0, confidence))
            risk_pct = (risk_per_share / long_price) * 100.0 if long_price > 0 else 0.0
            edge_pct = max(0.0, (0.50 * rr_ratio - 0.50) * risk_pct)
            risk_budget = self.available_capital * 0.005
            suggested_qty = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0

            trade_candidate = TradeCandidate(
                strategy_name=self.name,
                candidate_type=CandidateType.SECTOR_LONG_SHORT,
                symbol=long_symbol,
                direction="long",
                entry_price=round(long_price, 2),
                stop_price=round(stop_price, 2),
                target_price=round(target_price, 2),
                risk_dollars=round(risk_per_share * max(1, suggested_qty), 2),
                suggested_qty=suggested_qty,
                risk_reward_ratio=round(rr_ratio, 2),
                confidence=round(confidence, 2),
                edge_estimate_pct=round(edge_pct, 2),
                regime_fit=round(regime_fit, 2),
                catalyst=f"rotation_{long_symbol}_vs_{short_symbol}_div_{divergence:.1f}pct",
                symbol_b=short_symbol,
                metadata={
                    "long_symbol": long_symbol,
                    "short_symbol": short_symbol,
                    "long_rs": round(long_rs, 2),
                    "short_rs": round(short_rs, 2),
                    "divergence": round(divergence, 2),
                    "long_price": round(long_price, 2),
                    "short_price": round(short_price, 2) if short_price else 0.0,
                },
            )

            return OpportunityAssessment(
                num_candidates=1,
                avg_risk_reward=round(max(0.0, rr_ratio), 2),
                confidence=round(max(0.0, confidence), 2),
                estimated_daily_trades=min(1, self.config.max_positions - len(self._trades)),
                estimated_edge_pct=round(max(0.0, edge_pct), 2),
                details=[
                    {
                        "symbol": long_symbol,
                        "sector": self._sectors.get(long_symbol, ""),
                        "rs": round(long_rs, 2),
                        "side": "long",
                    },
                    {
                        "symbol": short_symbol,
                        "sector": self._sectors.get(short_symbol, ""),
                        "rs": round(short_rs, 2),
                        "side": "short",
                    }
                ],
                candidates=[trade_candidate],
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
                "sector_name": trade.sector_name,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "entry_rs": trade.entry_rs,
                "stop_price": trade.stop_price,
                "entry_time": trade.entry_time.isoformat(),
                "capital_used": trade.capital_used,
                "last_rebalance": trade.last_rebalance.isoformat() if trade.last_rebalance else None,
                "trade_id": trade.trade_id,
                "bracket_stop_order_id": trade.bracket_stop_order_id,
                "is_bracket": trade.is_bracket,
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
                bracket_stop_order_id=saved.get("bracket_stop_order_id", ""),
                is_bracket=saved.get("is_bracket", False),
            )
