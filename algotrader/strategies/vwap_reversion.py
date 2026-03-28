"""Mean reversion strategy (legacy name: vwap_reversion).

Trades large-cap stocks that deviate significantly from their 20-day
mean and reverts toward that mean over a multi-day horizon.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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


def calculate_mean_reversion_zscore(bars: pd.DataFrame, lookback: int = 20) -> tuple[float, float, float]:
    """Calculate MA20 and z-score from daily closes."""
    if bars.empty or len(bars) < max(lookback + 1, 25):
        return 0.0, 0.0, 0.0

    closes = pd.to_numeric(bars["close"], errors="coerce").dropna()
    if len(closes) < lookback:
        return 0.0, 0.0, 0.0

    window = closes.iloc[-lookback:]
    ma_20 = float(window.mean())
    std_20 = float(window.std())
    if std_20 <= 0:
        return ma_20, 0.0, 0.0

    current_price = float(closes.iloc[-1])
    z_score = (current_price - ma_20) / std_20
    return ma_20, std_20, float(z_score)


def calculate_vwap_zscore(bars: pd.DataFrame) -> tuple[float, float]:
    """Backward-compatible shim kept for existing imports/tests."""
    if bars.empty or len(bars) < 5:
        return 0.0, 0.0

    ma_20, std_20, z_score = calculate_mean_reversion_zscore(bars, lookback=20)
    _ = std_20
    return ma_20, z_score


@dataclass
class VWAPTrade:
    """Internal state for a single mean reversion trade."""

    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    entry_mean: float
    entry_std: float
    entry_z_score: float
    stop_price: float
    target_price: float
    entry_time: datetime
    capital_used: float = 0.0
    trade_id: str = ""
    bracket_stop_order_id: str = ""
    bracket_tp_order_id: str = ""
    is_bracket: bool = False


@register_strategy("vwap_reversion")
class VWAPReversionStrategy(StrategyBase):
    """Daily MA20 mean reversion for large-cap stocks."""

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
        self._mode = str(params.get("mode", "daily_ma")).lower()
        self._universe = params.get("universe", [
            "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "JPM", "V",
        ])

        # Entry
        self._min_z_score = params.get("min_z_score", 2.0)
        self._trending_min_z_score = float(
            params.get("trending_min_z_score", max(2.5, self._min_z_score)),
        )
        self._min_volume_ratio = params.get("min_volume_ratio", 1.0)
        self._max_spread_pct = params.get("max_spread_pct", 0.15)
        self._lookback_days = int(params.get("lookback_days", 20))

        # Exit
        self._exit_z_score = params.get("exit_z_score", 0.5)
        self._stop_pct = params.get("stop_pct", 0.5)
        self._target_pct = params.get("target_pct", 1.0)
        self._max_hold_days = int(params.get("max_hold_days", 3))

        # Data
        self._bar_timeframe = params.get("bar_timeframe", "1Day")
        self._vwap_lookback_bars = params.get("vwap_lookback_bars", 78)

        # Regime filter — ONLY ranging/low_vol
        self._allowed_regimes = params.get(
            "allowed_regimes",
            ["ranging", "low_vol"],
        )
        self._allow_high_vol = bool(params.get("allow_high_vol", False))
        self._high_vol_min_z_score = float(
            params.get("high_vol_min_z_score", max(2.5, self._min_z_score)),
        )

        # Internal state
        self._trades: dict[str, VWAPTrade] = {}

    def warm_up(self) -> None:
        """No heavy warm-up needed; signals derive from daily bars."""
        self._log.info("warming_up", universe_size=len(self._universe))
        self._warmed_up = True

    def _effective_allowed_regimes(self) -> list[str]:
        allowed_regimes = list(self._allowed_regimes)
        if self._allow_high_vol and "high_vol" not in allowed_regimes:
            allowed_regimes.append("high_vol")
        return allowed_regimes

    def _effective_min_z_score(self, regime: MarketRegime | None) -> float:
        if regime and regime.regime_type.value == "high_vol" and self._allow_high_vol:
            return max(self._min_z_score, self._high_vol_min_z_score)
        if regime and regime.regime_type.value in {"trending_up", "trending_down"}:
            return max(self._min_z_score, self._trending_min_z_score)
        return float(self._min_z_score)

    def run_cycle(self, regime: MarketRegime | None = None) -> list[Signal]:
        """Run one cycle: manage positions, scan for mean-reversion setups."""
        self._check_day_reset()
        self._last_cycle_time = datetime.now(pytz.UTC)
        signals: list[Signal] = []

        et_now = datetime.now(ET)

        # 1. Manage existing positions FIRST (always, regardless of regime)
        signals.extend(self._manage_positions(et_now))

        # 2. Check regime filter (only gates new entries)
        allowed_regimes = self._effective_allowed_regimes()
        if regime and regime.regime_type.value not in allowed_regimes:
            return signals

        # 3. Scan for new entries
        if len(self._trades) < self.config.max_positions:
            signals.extend(self._scan_entries(et_now, regime))

        return signals

    def _manage_positions(self, et_now: datetime) -> list[Signal]:
        """Manage open mean-reversion trades."""
        signals: list[Signal] = []

        for symbol, trade in list(self._trades.items()):
            close_reason = ""
            stats = self._daily_reversion_stats(symbol)
            if stats is None:
                continue
            current_price = stats["current_price"]
            current_z = stats["z_score"]
            current_mean = stats["mean_price"]
            days_held = max(0, (et_now.date() - trade.entry_time.astimezone(ET).date()).days)

            if abs(current_z) <= self._exit_z_score:
                close_reason = f"z_reverted: |{current_z:.2f}| <= {self._exit_z_score}"
            elif trade.direction == "long" and current_price <= trade.stop_price:
                close_reason = f"stop_hit: {current_price:.2f} <= {trade.stop_price:.2f}"
            elif trade.direction == "short" and current_price >= trade.stop_price:
                close_reason = f"stop_hit: {current_price:.2f} >= {trade.stop_price:.2f}"
            elif trade.direction == "long" and current_price >= trade.target_price:
                close_reason = f"target_hit: {current_price:.2f} >= {trade.target_price:.2f}"
            elif trade.direction == "short" and current_price <= trade.target_price:
                close_reason = f"target_hit: {current_price:.2f} <= {trade.target_price:.2f}"
            elif days_held >= self._max_hold_days:
                close_reason = f"max_hold_{self._max_hold_days}_days"

            if close_reason:
                self._close_trade(symbol, trade, close_reason)
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    direction=SignalDirection.CLOSE,
                    reason=close_reason,
                    metadata={
                        "entry_z": trade.entry_z_score,
                        "current_z": current_z,
                        "ma_20": current_mean,
                        "days_held": days_held,
                    },
                    timestamp=datetime.now(pytz.UTC),
                ))

        return signals

    def _scan_entries(self, et_now: datetime, regime: MarketRegime | None = None) -> list[Signal]:
        """Scan universe for daily MA mean-reversion entries."""
        signals: list[Signal] = []
        effective_min_z_score = self._effective_min_z_score(regime)

        for symbol in self._universe:
            if symbol in self._trades:
                continue
            if len(self._trades) >= self.config.max_positions:
                break

            stats = self._daily_reversion_stats(symbol)
            if stats is None:
                continue

            current_price = stats["current_price"]
            mean_price = stats["mean_price"]
            std_20 = stats["std_20"]
            z_score = stats["z_score"]

            if abs(z_score) < effective_min_z_score:
                continue

            quote = self.data_provider.get_quote(symbol)
            if quote and quote.is_valid() and quote.spread_pct * 100 > self._max_spread_pct:
                continue

            if z_score > 0:
                direction = "short"
                side = OrderSide.SELL
                stop_price = current_price + std_20
                target_price = mean_price
            else:
                direction = "long"
                side = OrderSide.BUY
                stop_price = current_price - std_20
                target_price = mean_price

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
                bracket_stop_price=stop_price,
                bracket_take_profit_price=target_price,
            )

            if not order:
                self.release_capital(capital_needed)
                continue

            trade = VWAPTrade(
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                entry_mean=mean_price,
                entry_std=std_20,
                entry_z_score=z_score,
                stop_price=stop_price,
                target_price=target_price,
                entry_time=datetime.now(pytz.UTC),
                capital_used=capital_needed,
                trade_id=str(uuid.uuid4()),
                bracket_stop_order_id=order.stop_order_id,
                bracket_tp_order_id=order.tp_order_id,
                is_bracket=order.is_bracket,
            )
            self._trades[symbol] = trade

            self._log.info(
                "vwap_entry",
                symbol=symbol,
                direction=direction,
                z_score=round(z_score, 2),
                ma_20=round(mean_price, 2),
                price=current_price,
                stop=stop_price,
                target=target_price,
                shares=shares,
            )

            signals.append(Signal(
                strategy_name=self.name,
                symbol=symbol,
                direction=SignalDirection.LONG if direction == "long" else SignalDirection.SHORT,
                reason=f"mean_reversion_ma20: z={z_score:.2f}",
                metadata={"z_score": z_score, "ma_20": mean_price, "std_20": std_20},
                timestamp=datetime.now(pytz.UTC),
            ))

        return signals

    def _daily_reversion_stats(self, symbol: str) -> dict[str, float] | None:
        """Return daily MA20/std/z-score stats for a symbol."""
        min_required = max(25, self._lookback_days + 1)
        bars = self.data_provider.get_bars(symbol, "1Day", max(30, self._lookback_days + 10))
        if bars.empty or len(bars) < min_required:
            # Backward-compatible fallback for test stubs and non-daily feeds.
            for timeframe in (self._bar_timeframe, "5Min"):
                bars = self.data_provider.get_bars(
                    symbol,
                    timeframe,
                    max(30, self._lookback_days + 10),
                )
                if not bars.empty and len(bars) >= min_required:
                    break
        if bars.empty or len(bars) < max(25, self._lookback_days + 1):
            return None

        mean_price, std_20, z_score = calculate_mean_reversion_zscore(
            bars,
            lookback=self._lookback_days,
        )
        if mean_price <= 0 or std_20 <= 0:
            return None

        current_price = float(bars["close"].iloc[-1])
        if current_price <= 0:
            return None

        return {
            "current_price": current_price,
            "mean_price": mean_price,
            "std_20": std_20,
            "z_score": z_score,
        }

    def _close_trade(self, symbol: str, trade: VWAPTrade, reason: str) -> None:
        """Close a VWAP trade with position exit safety."""
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
            entry_reason=f"mean_reversion_ma20: z={trade.entry_z_score:.2f}",
            exit_reason=reason,
            metadata={
                "entry_mean": trade.entry_mean,
                "entry_std": trade.entry_std,
                "entry_z": trade.entry_z_score,
            },
        )

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
        """Assess daily mean-reversion opportunities across the configured universe."""
        self._log.info("assess_start", strategy=self.name)
        raw_scanned = len(self._universe)
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

            allowed_regimes = self._effective_allowed_regimes()
            effective_min_z_score = self._effective_min_z_score(regime)
            if regime and regime.regime_type.value not in allowed_regimes:
                self._log.warning(
                    "vwap_regime_rejected",
                    regime=regime_type,
                    allowed_regimes=allowed_regimes,
                )
                return _complete(OpportunityAssessment())

            candidates: list[TradeCandidate] = []
            details: list[dict] = []
            for symbol in self._universe:
                if symbol in self._trades:
                    continue

                stats = self._daily_reversion_stats(symbol)
                if stats is None:
                    continue

                current_price = stats["current_price"]
                mean_price = stats["mean_price"]
                std_20 = stats["std_20"]
                z_score = stats["z_score"]
                abs_z = abs(z_score)

                if abs_z < effective_min_z_score:
                    continue

                direction = "long" if z_score < 0 else "short"
                candidate_type = CandidateType.LONG_EQUITY if direction == "long" else CandidateType.SHORT_EQUITY
                stop_price = current_price - std_20 if direction == "long" else current_price + std_20
                target_price = mean_price

                risk_per_share = abs(current_price - stop_price)
                reward_per_share = abs(target_price - current_price)
                if risk_per_share <= 0:
                    continue
                rr_ratio = reward_per_share / risk_per_share

                confidence = 0.20
                if abs_z >= (effective_min_z_score + 1.0):
                    confidence += 0.30
                elif abs_z >= (effective_min_z_score + 0.5):
                    confidence += 0.20
                else:
                    confidence += 0.10

                regime_fit = 0.55
                if regime:
                    if regime.regime_type.value in {"ranging", "low_vol"}:
                        regime_fit = 0.85
                        confidence += 0.15
                    elif regime.regime_type.value == "high_vol":
                        regime_fit = 0.70
                        confidence += 0.10
                    elif regime.regime_type.value in {"trending_up", "trending_down"}:
                        regime_fit = 0.45

                confidence = min(1.0, max(0.0, confidence))
                risk_pct = (risk_per_share / current_price) * 100.0 if current_price > 0 else 0.0
                edge_pct = max(0.0, (0.54 * rr_ratio - 0.46) * risk_pct)
                risk_budget = self.available_capital * 0.005
                suggested_qty = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0

                candidates.append(TradeCandidate(
                    strategy_name=self.name,
                    candidate_type=candidate_type,
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
                    catalyst=f"mean_reversion_z_{z_score:.1f}_ma20",
                    time_horizon_minutes=0,
                    expiry_time=None,
                    metadata={
                        "z_score": round(z_score, 2),
                        "ma_20": round(mean_price, 2),
                        "vwap": round(mean_price, 2),
                        "std_20": round(std_20, 4),
                        "hold_days": self._max_hold_days,
                        "swing_trade": True,
                    },
                ))
                details.append(
                    {
                        "symbol": symbol,
                        "z_score": round(z_score, 2),
                        "ma_20": round(mean_price, 2),
                        "std_20": round(std_20, 2),
                        "direction": direction,
                    }
                )

            if not candidates:
                return _complete(OpportunityAssessment())

            candidates.sort(key=lambda c: c.expected_value, reverse=True)
            top_candidates = candidates[:3]
            avg_rr = sum(c.risk_reward_ratio for c in candidates) / len(candidates)
            avg_conf = sum(c.confidence for c in candidates) / len(candidates)
            avg_edge = sum(c.edge_estimate_pct for c in candidates) / len(candidates)

            return _complete(OpportunityAssessment(
                num_candidates=len(candidates),
                avg_risk_reward=round(max(0.0, avg_rr), 2),
                confidence=round(max(0.0, min(1.0, avg_conf)), 2),
                estimated_daily_trades=min(len(candidates), self.config.max_positions),
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
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "entry_mean": trade.entry_mean,
                "entry_vwap": trade.entry_mean,
                "entry_std": trade.entry_std,
                "entry_z_score": trade.entry_z_score,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
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
            self._trades[symbol] = VWAPTrade(
                symbol=saved["symbol"],
                direction=saved["direction"],
                entry_price=saved["entry_price"],
                entry_mean=saved.get("entry_mean", saved.get("entry_vwap", saved["entry_price"])),
                entry_std=saved.get("entry_std", 0.0),
                entry_z_score=saved["entry_z_score"],
                stop_price=saved["stop_price"],
                target_price=saved["target_price"],
                entry_time=datetime.fromisoformat(saved["entry_time"]),
                capital_used=saved.get("capital_used", 0.0),
                trade_id=saved.get("trade_id", ""),
                bracket_stop_order_id=saved.get("bracket_stop_order_id", ""),
                bracket_tp_order_id=saved.get("bracket_tp_order_id", ""),
                is_bracket=saved.get("is_bracket", False),
            )

    def close_all_positions(self, reason: str = "") -> int:
        """Force-close all open mean-reversion trades."""
        closed = 0
        for symbol, trade in list(self._trades.items()):
            self._close_trade(symbol, trade, reason or "forced_close")
            if symbol not in self._trades:
                closed += 1
        return closed

    def close_positions_for_eod(self, et_now: datetime) -> int:
        """Run one explicit EOD management pass."""
        before = len(self._trades)
        self._manage_positions(et_now)
        return max(0, before - len(self._trades))



