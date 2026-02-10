"""Options Premium Selling strategy.

Sells credit spreads and iron condors on SPY/QQQ/IWM to capture theta
decay. Uses SMA5 contrarian filter for directional bias.

IEX limitation: Options quotes are 15-min delayed. This strategy uses
the underlying price + modeled delta rather than live options greeks.
Trades may be flagged as "simulated" if options orders aren't supported.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
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
class PremiumTrade:
    """Internal state for a single options premium trade."""

    underlying: str
    structure: str  # "put_spread", "call_spread", "iron_condor"
    short_strike: float
    long_strike: float
    # For iron condor, add call side:
    call_short_strike: float | None = None
    call_long_strike: float | None = None
    contracts: int = 1
    credit_received: float = 0.0  # Estimated
    max_profit: float = 0.0
    max_loss: float = 0.0
    entry_time: datetime | None = None
    expiration: date | None = None
    simulated: bool = False  # True if couldn't submit real order
    capital_used: float = 0.0
    trade_id: str = ""


@register_strategy("options_premium")
class OptionsPremiumStrategy(StrategyBase):
    """Options premium selling via credit spreads / iron condors.

    Sells premium on SPY/QQQ/IWM during elevated vol using SMA5 contrarian
    filter. Falls back to simulated tracking if options orders aren't
    supported by the current broker.
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
        self._underlyings = params.get("underlyings", ["SPY", "QQQ", "IWM"])
        self._structure = params.get("structure", "credit_spread")
        self._short_delta_target = params.get("short_delta_target", 14)
        self._wing_width = params.get("wing_width", 5)

        # Entry window
        self._entry_start_hour = params.get("entry_start_hour", 10)
        self._entry_start_minute = params.get("entry_start_minute", 15)
        self._entry_end_hour = params.get("entry_end_hour", 10)
        self._entry_end_minute = params.get("entry_end_minute", 45)

        # DTE
        self._target_dte = params.get("target_dte", 0)
        self._max_dte = params.get("max_dte", 2)

        # Filters
        self._min_vix_proxy = params.get("min_vix_proxy", 15.0)
        self._use_sma5_filter = params.get("use_sma5_filter", True)

        # Exit
        self._profit_target_pct = params.get("profit_target_pct", 50.0)
        self._stop_loss_pct = params.get("stop_loss_pct", 200.0)
        self._close_before_minutes = params.get("close_before_minutes", 15)

        # Sizing
        self._max_risk_per_trade = params.get("max_risk_per_trade", 500)
        self._max_contracts = params.get("max_contracts", 5)

        # Regime filter
        self._allowed_regimes = params.get(
            "allowed_regimes",
            ["high_vol", "ranging", "trending_up", "trending_down"],
        )
        self._blocked_regimes = params.get("blocked_regimes", ["event_day"])

        # Internal state
        self._trades: dict[str, PremiumTrade] = {}
        self._entry_attempted_today: bool = False

    def warm_up(self) -> None:
        """No heavy warm-up needed."""
        self._log.info("warming_up", underlyings=self._underlyings)
        self._warmed_up = True

    def run_cycle(self, regime: MarketRegime | None = None) -> list[Signal]:
        """Run one cycle: manage positions, check entry window."""
        self._check_day_reset()
        self._last_cycle_time = datetime.now(pytz.UTC)
        signals: list[Signal] = []

        # Reset entry flag on new day
        et_now = datetime.now(ET)
        if et_now.hour < self._entry_start_hour:
            self._entry_attempted_today = False

        # 1. Manage existing positions FIRST (always, regardless of regime)
        signals.extend(self._manage_positions(et_now, regime))

        # 2. Check regime filter (only gates new entries)
        if regime:
            if regime.regime_type.value in self._blocked_regimes:
                return signals
            if regime.regime_type.value not in self._allowed_regimes:
                return signals

        # 3. Check entry window
        if not self._entry_attempted_today and len(self._trades) < self.config.max_positions:
            if self._in_entry_window(et_now):
                signals.extend(self._scan_entries(et_now, regime))
                self._entry_attempted_today = True

        return signals

    def _in_entry_window(self, et_now: datetime) -> bool:
        """Check if current time is within the entry window."""
        start = et_now.replace(hour=self._entry_start_hour, minute=self._entry_start_minute)
        end = et_now.replace(hour=self._entry_end_hour, minute=self._entry_end_minute)
        return start <= et_now <= end

    def _manage_positions(self, et_now: datetime, regime: MarketRegime | None) -> list[Signal]:
        """Manage open premium trades."""
        signals: list[Signal] = []

        for trade_key, trade in list(self._trades.items()):
            close_reason = ""

            # Get current underlying price
            current_price = self._get_current_price(trade.underlying)
            if current_price is None:
                continue

            # Estimate current P&L from underlying price movement
            estimated_pnl_pct = self._estimate_pnl_pct(trade, current_price)

            # 1. Profit target (close at 50% of max profit)
            if estimated_pnl_pct >= self._profit_target_pct:
                close_reason = f"profit_target: est {estimated_pnl_pct:.0f}% of max"

            # 2. Stop loss (close at 200% of credit = net 100% loss)
            elif estimated_pnl_pct <= -self._stop_loss_pct:
                close_reason = f"stop_loss: est {estimated_pnl_pct:.0f}% of max"

            # 3. Close before market close
            market_close = et_now.replace(hour=16, minute=0)
            minutes_to_close = (market_close - et_now).total_seconds() / 60
            if minutes_to_close <= self._close_before_minutes:
                close_reason = "close_before_eod"

            if close_reason:
                self._close_trade(trade_key, trade, close_reason)
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=trade.underlying,
                    direction=SignalDirection.CLOSE,
                    reason=close_reason,
                    metadata={"structure": trade.structure, "simulated": trade.simulated},
                    timestamp=datetime.now(pytz.UTC),
                ))

        return signals

    def _scan_entries(self, et_now: datetime, regime: MarketRegime | None) -> list[Signal]:
        """Scan for premium selling opportunities."""
        signals: list[Signal] = []

        # Check VIX proxy level
        vix_level = regime.vix_level if regime and regime.vix_level else None
        if vix_level is not None and vix_level < self._min_vix_proxy:
            self._log.debug("vix_too_low", vix=vix_level, min_required=self._min_vix_proxy)
            return signals

        for underlying in self._underlyings:
            if underlying in self._trades:
                continue
            if len(self._trades) >= self.config.max_positions:
                break

            current_price = self._get_current_price(underlying)
            if current_price is None or current_price <= 0:
                continue

            # Determine bias using SMA5 filter
            bias = self._get_sma5_bias(underlying, current_price)

            # Calculate strikes
            atr = self._get_atr(underlying)
            if atr is None or atr <= 0:
                continue

            # Estimate 1-sigma intraday move
            sigma = atr * 0.6
            # For target delta ~14: ~1.1 sigma from current price
            strike_distance = sigma * 1.1

            # Determine structure based on bias
            if bias == "bullish" or (bias == "neutral" and self._structure == "credit_spread"):
                structure = "put_spread"
                short_strike = round(current_price - strike_distance)
                long_strike = short_strike - self._wing_width
            elif bias == "bearish":
                structure = "call_spread"
                short_strike = round(current_price + strike_distance)
                long_strike = short_strike + self._wing_width
            elif bias == "neutral":
                structure = "iron_condor"
                short_strike = round(current_price - strike_distance)  # Put side
                long_strike = short_strike - self._wing_width
            else:
                continue

            # Calculate contracts and risk
            spread_width = self._wing_width * 100  # Per contract in dollars
            estimated_credit = spread_width * 0.25  # ~25% of width as premium
            max_loss_per_contract = spread_width - estimated_credit
            contracts = min(
                self._max_contracts,
                int(self._max_risk_per_trade / max_loss_per_contract) if max_loss_per_contract > 0 else 1,
            )
            contracts = max(contracts, 1)

            total_risk = contracts * max_loss_per_contract
            total_credit = contracts * estimated_credit

            # For iron condor, double the credit (two sides)
            call_short = None
            call_long = None
            if structure == "iron_condor":
                call_short = round(current_price + strike_distance)
                call_long = call_short + self._wing_width
                total_credit *= 2
                total_risk = contracts * spread_width  # Max loss is one side

            # Reserve capital for margin
            capital_needed = total_risk
            if not self.reserve_capital(capital_needed):
                continue

            # Attempt to submit the trade
            submitted = self._submit_credit_spread(
                underlying, short_strike, long_strike,
                structure, contracts, et_now.date(),
            )

            trade = PremiumTrade(
                underlying=underlying,
                structure=structure,
                short_strike=short_strike,
                long_strike=long_strike,
                call_short_strike=call_short,
                call_long_strike=call_long,
                contracts=contracts,
                credit_received=total_credit,
                max_profit=total_credit,
                max_loss=total_risk,
                entry_time=datetime.now(pytz.UTC),
                expiration=et_now.date() + timedelta(days=self._target_dte),
                simulated=not submitted,
                capital_used=capital_needed,
                trade_id=str(uuid.uuid4()),
            )
            self._trades[underlying] = trade

            self._log.info(
                "premium_entry",
                underlying=underlying,
                structure=structure,
                short_strike=short_strike,
                long_strike=long_strike,
                call_short=call_short,
                call_long=call_long,
                contracts=contracts,
                credit=round(total_credit, 2),
                max_loss=round(total_risk, 2),
                simulated=trade.simulated,
                bias=bias,
            )

            signals.append(Signal(
                strategy_name=self.name,
                symbol=underlying,
                direction=SignalDirection.SHORT,
                reason=f"premium_sell: {structure} bias={bias}",
                metadata={
                    "structure": structure,
                    "short_strike": short_strike,
                    "long_strike": long_strike,
                    "contracts": contracts,
                    "simulated": trade.simulated,
                },
                timestamp=datetime.now(pytz.UTC),
            ))

        return signals

    def _get_sma5_bias(self, underlying: str, current_price: float) -> str:
        """Determine directional bias using SMA5 contrarian filter."""
        if not self._use_sma5_filter:
            return "neutral"

        try:
            bars = self.data_provider.get_bars(underlying, "1Day", 10)
            if bars.empty or len(bars) < 5:
                return "neutral"

            sma5 = float(bars["close"].iloc[-5:].mean())

            if current_price > sma5 * 1.002:
                return "bullish"  # Sell put spreads
            elif current_price < sma5 * 0.998:
                return "bearish"  # Sell call spreads
            else:
                return "neutral"  # Iron condor
        except Exception:
            return "neutral"

    def _get_atr(self, symbol: str) -> float | None:
        """Get ATR(14) for a symbol."""
        try:
            bars = self.data_provider.get_bars(symbol, "1Day", 20)
            if bars.empty or len(bars) < 15:
                return None

            highs = bars["high"].values
            lows = bars["low"].values
            closes = bars["close"].values

            tr_values = []
            for i in range(1, len(bars)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
                tr_values.append(tr)

            if len(tr_values) < 14:
                return None
            return float(np.mean(tr_values[-14:]))
        except Exception:
            return None

    def _estimate_pnl_pct(self, trade: PremiumTrade, current_price: float) -> float:
        """Estimate current P&L as percentage of max profit.

        Positive = winning, negative = losing.
        Uses a simplified model based on short strike proximity.
        """
        if trade.max_profit <= 0:
            return 0.0

        if trade.structure == "put_spread":
            if current_price >= trade.short_strike:
                # Above short strike — likely profitable
                return min(100.0, 50.0 + (current_price - trade.short_strike) / trade.short_strike * 1000)
            else:
                # Below short strike — losing
                loss_ratio = (trade.short_strike - current_price) / self._wing_width
                return -loss_ratio * 100

        elif trade.structure == "call_spread":
            if current_price <= trade.short_strike:
                return min(100.0, 50.0 + (trade.short_strike - current_price) / trade.short_strike * 1000)
            else:
                loss_ratio = (current_price - trade.short_strike) / self._wing_width
                return -loss_ratio * 100

        elif trade.structure == "iron_condor":
            # Iron condor: profitable if price stays between short strikes
            if (trade.call_short_strike and
                    trade.short_strike <= current_price <= trade.call_short_strike):
                return 50.0  # In the safe zone
            elif current_price < trade.short_strike:
                loss_ratio = (trade.short_strike - current_price) / self._wing_width
                return -loss_ratio * 100
            elif trade.call_short_strike and current_price > trade.call_short_strike:
                loss_ratio = (current_price - trade.call_short_strike) / self._wing_width
                return -loss_ratio * 100

        return 0.0

    def _submit_credit_spread(
        self, underlying: str, short_strike: float, long_strike: float,
        structure: str, contracts: int, expiration: date,
    ) -> bool:
        """Submit a credit spread order. Returns False if not supported."""
        try:
            # Alpaca may not support multi-leg options orders via alpaca-py
            # Log the intended trade and flag as simulated
            self._log.warning(
                "options_order_simulated",
                underlying=underlying,
                structure=structure,
                short_strike=short_strike,
                long_strike=long_strike,
                contracts=contracts,
                note="Options orders require IBKR integration for live execution",
            )
            return False
        except Exception as e:
            self._log.warning("options_order_failed", error=str(e))
            return False

    def _close_trade(self, trade_key: str, trade: PremiumTrade, reason: str) -> None:
        """Close a premium trade."""
        # For simulated trades, just remove from tracking
        if trade.simulated:
            self._trades.pop(trade_key, None)
            self.release_capital(trade.capital_used)
            # Estimate P&L for the journal
            current_price = self._get_current_price(trade.underlying) or 0.0
            pnl = self._estimate_simulated_pnl(trade, current_price)
            self.record_trade(pnl)
            self._log.info(
                "premium_exit_simulated",
                underlying=trade.underlying,
                structure=trade.structure,
                reason=reason,
                est_pnl=round(pnl, 2),
            )
            return

        # For real trades, close via executor
        close_success = self.executor.close_position(trade.underlying)
        if not close_success:
            self._log.error("premium_close_failed", underlying=trade.underlying)
            return

        self._trades.pop(trade_key, None)
        self.release_capital(trade.capital_used)
        self.record_trade(0.0)  # Real P&L comes from broker

        self._log.info(
            "premium_exit",
            underlying=trade.underlying,
            structure=trade.structure,
            reason=reason,
        )

    def _estimate_simulated_pnl(self, trade: PremiumTrade, current_price: float) -> float:
        """Estimate P&L for a simulated trade based on underlying movement."""
        if current_price <= 0 or trade.max_profit <= 0:
            return 0.0
        pnl_pct = self._estimate_pnl_pct(trade, current_price)
        return trade.max_profit * (pnl_pct / 100)

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
        """Assess options premium selling conditions."""
        try:
            # Regime gate
            if regime:
                if regime.regime_type.value in self._blocked_regimes:
                    return OpportunityAssessment()
                if regime.regime_type.value not in self._allowed_regimes:
                    return OpportunityAssessment()

            # VIX proxy check
            vix_level = regime.vix_level if regime and regime.vix_level else None
            if vix_level is not None and vix_level < self._min_vix_proxy:
                return OpportunityAssessment(
                    num_candidates=0,
                    confidence=0.0,
                    details=[{"reason": f"VIX {vix_level:.1f} < min {self._min_vix_proxy}"}],
                )

            # Count available underlyings not already traded
            available = [u for u in self._underlyings if u not in self._trades]
            if not available:
                return OpportunityAssessment()

            # Estimate credit quality from ATR
            valid = []
            for underlying in available:
                try:
                    atr = self._get_atr(underlying)
                    if atr and atr > 0:
                        valid.append({"symbol": underlying, "atr": round(atr, 2)})
                except Exception:
                    continue

            if not valid:
                return OpportunityAssessment()

            # Higher VIX = higher confidence for premium selling
            vix_boost = min(0.3, (vix_level - self._min_vix_proxy) * 0.02) if vix_level else 0.0
            confidence = min(1.0, 0.4 + len(valid) * 0.1 + vix_boost)

            # R:R for credit spreads: ~0.33 (25% credit / 75% risk)
            rr = 0.33

            return OpportunityAssessment(
                num_candidates=len(valid),
                avg_risk_reward=rr,
                confidence=round(max(0.0, confidence), 2),
                estimated_daily_trades=min(len(valid), self.config.max_positions),
                estimated_edge_pct=round(rr * 0.5, 2),
                details=valid,
            )
        except Exception:
            self._log.debug("assess_opportunities_failed")
            return OpportunityAssessment()

    def _get_state(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        base = super()._get_state()
        base["trades"] = {}
        for key, trade in self._trades.items():
            base["trades"][key] = {
                "underlying": trade.underlying,
                "structure": trade.structure,
                "short_strike": trade.short_strike,
                "long_strike": trade.long_strike,
                "call_short_strike": trade.call_short_strike,
                "call_long_strike": trade.call_long_strike,
                "contracts": trade.contracts,
                "credit_received": trade.credit_received,
                "max_profit": trade.max_profit,
                "max_loss": trade.max_loss,
                "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
                "expiration": trade.expiration.isoformat() if trade.expiration else None,
                "simulated": trade.simulated,
                "capital_used": trade.capital_used,
                "trade_id": trade.trade_id,
            }
        return base

    def _restore_state(self, state_data: dict[str, Any]) -> None:
        """Restore state from persistence."""
        super()._restore_state(state_data)
        for key, saved in state_data.get("trades", {}).items():
            self._trades[key] = PremiumTrade(
                underlying=saved["underlying"],
                structure=saved["structure"],
                short_strike=saved["short_strike"],
                long_strike=saved["long_strike"],
                call_short_strike=saved.get("call_short_strike"),
                call_long_strike=saved.get("call_long_strike"),
                contracts=saved.get("contracts", 1),
                credit_received=saved.get("credit_received", 0.0),
                max_profit=saved.get("max_profit", 0.0),
                max_loss=saved.get("max_loss", 0.0),
                entry_time=datetime.fromisoformat(saved["entry_time"]) if saved.get("entry_time") else None,
                expiration=date.fromisoformat(saved["expiration"]) if saved.get("expiration") else None,
                simulated=saved.get("simulated", True),
                capital_used=saved.get("capital_used", 0.0),
                trade_id=saved.get("trade_id", ""),
            )
