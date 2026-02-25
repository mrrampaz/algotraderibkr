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
    simulated: bool = True   # False only when a real MLEG order was placed
    capital_used: float = 0.0
    trade_id: str = ""
    # Real execution fields (empty when simulated=True)
    open_order_id: str = ""       # Broker order ID for the opening MLEG order
    short_occ_symbol: str = ""    # OCC symbol for the short leg (put or call)
    long_occ_symbol: str = ""     # OCC symbol for the long (protective) leg
    call_short_occ_symbol: str = ""  # Iron condor call-side short OCC symbol
    call_long_occ_symbol: str = ""   # Iron condor call-side long OCC symbol


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
        self._credit_estimate_pct = params.get("credit_estimate_pct", 0.30)
        self._win_rate_credit_spread = params.get("win_rate_credit_spread", 0.70)
        self._win_rate_iron_condor = params.get("win_rate_iron_condor", 0.72)
        self._event_day_confidence_penalty = params.get("event_day_confidence_penalty", 0.25)

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
            estimated_credit = spread_width * self._credit_estimate_pct
            max_loss_per_contract = max(1.0, spread_width - estimated_credit)
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

            # Attempt to submit the trade as a real MLEG order
            order_result = self._submit_credit_spread(
                underlying, short_strike, long_strike,
                structure, contracts, et_now.date(),
                call_short_strike=call_short,
                call_long_strike=call_long,
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
                simulated=order_result is None,
                capital_used=capital_needed,
                trade_id=str(uuid.uuid4()),
                open_order_id=order_result["order_id"] if order_result else "",
                short_occ_symbol=order_result["short_occ"] if order_result else "",
                long_occ_symbol=order_result["long_occ"] if order_result else "",
                call_short_occ_symbol=order_result.get("call_short_occ", "") if order_result else "",
                call_long_occ_symbol=order_result.get("call_long_occ", "") if order_result else "",
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
        self,
        underlying: str,
        short_strike: float,
        long_strike: float,
        structure: str,
        contracts: int,
        expiration: date,
        call_short_strike: float | None = None,
        call_long_strike: float | None = None,
    ) -> dict | None:
        """Submit a credit spread or iron condor via Alpaca MLEG order.

        Looks up the OCC contract symbols for each leg, then submits a single
        multi-leg market order (order_class="mleg"). Requires Level 3 options
        approval on the Alpaca account (self-serve toggle in paper accounts).

        Returns a dict with order details if a real order was placed:
            {"order_id": str, "short_occ": str, "long_occ": str,
             "call_short_occ": str, "call_long_occ": str}
        Returns None to indicate simulated tracking should be used instead,
        either because the executor doesn't support MLEG or the lookup failed.
        """
        if not hasattr(self.executor, "submit_mleg_order"):
            # Executor (e.g. IBKR stub) doesn't implement MLEG yet — simulate
            self._log.info("options_executor_no_mleg", underlying=underlying)
            return None

        # Resolve option type for each spread direction
        if structure == "put_spread" or structure == "iron_condor":
            put_or_call = "put"
        else:  # call_spread
            put_or_call = "call"

        # Look up OCC symbols for the primary (put or call) spread legs
        short_occ = self.executor.lookup_option_contract(
            underlying, put_or_call, short_strike, expiration,
        )
        long_occ = self.executor.lookup_option_contract(
            underlying, put_or_call, long_strike, expiration,
        )

        if not short_occ or not long_occ:
            self._log.warning(
                "options_leg_lookup_failed",
                underlying=underlying,
                structure=structure,
                short_strike=short_strike,
                long_strike=long_strike,
                expiration=str(expiration),
            )
            return None

        # Build the leg list: short = sell-to-open, long (hedge) = buy-to-open
        legs: list[dict] = [
            {"symbol": short_occ, "ratio_qty": 1, "side": OrderSide.SELL, "position_intent": "sell_to_open"},
            {"symbol": long_occ,  "ratio_qty": 1, "side": OrderSide.BUY,  "position_intent": "buy_to_open"},
        ]

        # Iron condor: add the call-side spread as two additional legs
        call_short_occ = ""
        call_long_occ = ""
        if structure == "iron_condor" and call_short_strike and call_long_strike:
            call_short_occ = self.executor.lookup_option_contract(
                underlying, "call", call_short_strike, expiration,
            ) or ""
            call_long_occ = self.executor.lookup_option_contract(
                underlying, "call", call_long_strike, expiration,
            ) or ""
            if call_short_occ and call_long_occ:
                legs.extend([
                    {"symbol": call_short_occ, "ratio_qty": 1, "side": OrderSide.SELL, "position_intent": "sell_to_open"},
                    {"symbol": call_long_occ,  "ratio_qty": 1, "side": OrderSide.BUY,  "position_intent": "buy_to_open"},
                ])
            else:
                # Can't complete iron condor — fall back to the put spread only
                self._log.warning(
                    "iron_condor_call_lookup_failed",
                    underlying=underlying,
                    call_short_strike=call_short_strike,
                    call_long_strike=call_long_strike,
                )

        order_id = self.executor.submit_mleg_order(legs, qty=contracts)
        if order_id is None:
            return None

        self._log.info(
            "options_mleg_submitted",
            underlying=underlying,
            structure=structure,
            contracts=contracts,
            order_id=order_id,
            legs=[leg["symbol"] for leg in legs],
        )
        return {
            "order_id": order_id,
            "short_occ": short_occ,
            "long_occ": long_occ,
            "call_short_occ": call_short_occ,
            "call_long_occ": call_long_occ,
        }

    def _close_real_spread(self, trade: PremiumTrade) -> bool:
        """Submit a closing MLEG order to buy back an open credit spread.

        Reverses each opening leg:
          sell-to-open short → buy-to-close
          buy-to-open long   → sell-to-close

        Returns True if the closing order was submitted, False otherwise.
        """
        if not hasattr(self.executor, "submit_mleg_order"):
            return False
        if not trade.short_occ_symbol or not trade.long_occ_symbol:
            self._log.error(
                "close_spread_missing_occ_symbols",
                underlying=trade.underlying,
            )
            return False

        legs: list[dict] = [
            {"symbol": trade.short_occ_symbol, "ratio_qty": 1, "side": OrderSide.BUY,  "position_intent": "buy_to_close"},
            {"symbol": trade.long_occ_symbol,  "ratio_qty": 1, "side": OrderSide.SELL, "position_intent": "sell_to_close"},
        ]
        # Iron condor: also close the call side
        if trade.call_short_occ_symbol and trade.call_long_occ_symbol:
            legs.extend([
                {"symbol": trade.call_short_occ_symbol, "ratio_qty": 1, "side": OrderSide.BUY,  "position_intent": "buy_to_close"},
                {"symbol": trade.call_long_occ_symbol,  "ratio_qty": 1, "side": OrderSide.SELL, "position_intent": "sell_to_close"},
            ])

        order_id = self.executor.submit_mleg_order(legs, qty=trade.contracts)
        if order_id:
            self._log.info(
                "options_close_mleg_submitted",
                underlying=trade.underlying,
                structure=trade.structure,
                order_id=order_id,
            )
        return order_id is not None

    def _close_trade(self, trade_key: str, trade: PremiumTrade, reason: str) -> None:
        """Close a premium trade."""
        # For simulated trades, just remove from tracking
        if trade.simulated:
            self._trades.pop(trade_key, None)
            self.release_capital(trade.capital_used)
            # Estimate P&L for the journal
            current_price = self._get_current_price(trade.underlying) or 0.0
            pnl = self._estimate_simulated_pnl(trade, current_price)
            self.record_trade(
                pnl,
                symbol=trade.underlying,
                side=OrderSide.SELL,
                qty=trade.contracts,
                entry_price=trade.credit_received,
                exit_price=trade.credit_received - pnl if trade.contracts else 0,
                entry_time=trade.entry_time,
                entry_reason=f"premium_sell: {trade.structure}",
                exit_reason=reason,
                metadata={
                    "structure": trade.structure,
                    "short_strike": trade.short_strike,
                    "long_strike": trade.long_strike,
                    "simulated": True,
                },
            )
            self._log.info(
                "premium_exit_simulated",
                underlying=trade.underlying,
                structure=trade.structure,
                reason=reason,
                est_pnl=round(pnl, 2),
            )
            return

        # For real trades, close by reversing the MLEG legs
        # P&L is estimated from underlying price (options legs aren't equity positions)
        current_price = self._get_current_price(trade.underlying) or 0.0
        pnl = self._estimate_simulated_pnl(trade, current_price)

        close_success = self._close_real_spread(trade)
        if not close_success:
            self._log.error("premium_close_failed", underlying=trade.underlying)
            return

        self._trades.pop(trade_key, None)
        self.release_capital(trade.capital_used)
        self.record_trade(
            pnl,
            symbol=trade.underlying,
            side=OrderSide.SELL,
            qty=trade.contracts,
            entry_price=trade.short_strike,
            exit_price=current_price,
            entry_time=trade.entry_time,
            entry_reason=f"premium_sell: {trade.structure}",
            exit_reason=reason,
            metadata={
                "structure": trade.structure,
                "short_strike": trade.short_strike,
                "long_strike": trade.long_strike,
                "short_occ": trade.short_occ_symbol,
                "long_occ": trade.long_occ_symbol,
                "simulated": False,
            },
        )

        self._log.info(
            "premium_exit",
            underlying=trade.underlying,
            structure=trade.structure,
            reason=reason,
            est_pnl=round(pnl, 2),
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

    def _structure_win_rate(self, structure: str) -> float:
        if structure == "iron_condor":
            return float(self._win_rate_iron_condor)
        return float(self._win_rate_credit_spread)

    def _adjusted_rr(self, win_rate: float, credit: float, max_loss: float) -> float:
        if max_loss <= 0 or credit <= 0:
            return 0.0
        loss_rate = max(1e-6, 1.0 - win_rate)
        return (win_rate * credit) / (loss_rate * max_loss)

    def _edge_pct(self, win_rate: float, credit: float, max_loss: float) -> float:
        if max_loss <= 0:
            return 0.0
        expected_pnl = win_rate * credit - (1.0 - win_rate) * max_loss
        return (expected_pnl / max_loss) * 100.0

    def assess_opportunities(self, regime: MarketRegime | None = None) -> OpportunityAssessment:
        """Assess options premium selling conditions."""
        self._log.info("assess_start", strategy=self.name)
        raw_scanned = 0
        try:
            from algotrader.intelligence.calendar.events import EventCalendar
            from algotrader.strategy_selector.candidate import CandidateType, TradeCandidate

            # Regime gate
            if regime:
                if regime.regime_type.value in self._blocked_regimes:
                    self._log.info(
                        "assess_complete",
                        strategy=self.name,
                        num_candidates=0,
                        num_raw_scanned=raw_scanned,
                    )
                    return OpportunityAssessment()
                if regime.regime_type.value not in self._allowed_regimes:
                    self._log.info(
                        "assess_complete",
                        strategy=self.name,
                        num_candidates=0,
                        num_raw_scanned=raw_scanned,
                    )
                    return OpportunityAssessment()

            et_now = datetime.now(ET)
            window_end = et_now.replace(
                hour=self._entry_end_hour,
                minute=self._entry_end_minute,
                second=0,
                microsecond=0,
            )
            if et_now > window_end:
                self._log.info(
                    "assess_complete",
                    strategy=self.name,
                    num_candidates=0,
                    num_raw_scanned=raw_scanned,
                )
                return OpportunityAssessment(
                    num_candidates=0,
                    confidence=0.0,
                    details=[{"reason": "entry_window_closed"}],
                    candidates=[],
                )
            expiry_time = window_end.astimezone(pytz.UTC)

            # VIX proxy check
            vix_level = regime.vix_level if regime and regime.vix_level else None
            if vix_level is not None and vix_level < self._min_vix_proxy:
                self._log.info(
                    "assess_complete",
                    strategy=self.name,
                    num_candidates=0,
                    num_raw_scanned=raw_scanned,
                )
                return OpportunityAssessment(
                    num_candidates=0,
                    confidence=0.0,
                    details=[{"reason": f"VIX {vix_level:.1f} < min {self._min_vix_proxy}"}],
                    candidates=[],
                )

            # Count available underlyings not already traded
            available = [u for u in self._underlyings if u not in self._trades]
            raw_scanned = len(available)
            if not available:
                self._log.info(
                    "assess_complete",
                    strategy=self.name,
                    num_candidates=0,
                    num_raw_scanned=raw_scanned,
                )
                return OpportunityAssessment()

            event_calendar = EventCalendar()
            is_event_day = event_calendar.is_event_day()
            details: list[dict] = []
            candidate_rows: list[TradeCandidate] = []

            for underlying in available:
                try:
                    atr = self._get_atr(underlying)
                    current_price = self._get_current_price(underlying)
                    if not atr or atr <= 0 or not current_price or current_price <= 0:
                        continue

                    bias = self._get_sma5_bias(underlying, current_price)
                    sigma = atr * 0.6
                    strike_distance = max(0.5, sigma * 1.1)

                    # Reuse the same structure logic as _scan_entries.
                    if bias == "bullish" or (bias == "neutral" and self._structure == "credit_spread"):
                        structure = "put_spread"
                        short_strike = round(current_price - strike_distance)
                        long_strike = short_strike - self._wing_width
                        call_short = 0.0
                        call_long = 0.0
                        direction = "short"
                        candidate_type = CandidateType.CREDIT_SPREAD
                    elif bias == "bearish":
                        structure = "call_spread"
                        short_strike = round(current_price + strike_distance)
                        long_strike = short_strike + self._wing_width
                        call_short = 0.0
                        call_long = 0.0
                        direction = "short"
                        candidate_type = CandidateType.CREDIT_SPREAD
                    else:
                        structure = "iron_condor"
                        short_strike = round(current_price - strike_distance)  # put short
                        long_strike = short_strike - self._wing_width          # put long
                        call_short = round(current_price + strike_distance)     # call short
                        call_long = call_short + self._wing_width               # call long
                        direction = "neutral"
                        candidate_type = CandidateType.IRON_CONDOR

                    spread_width = max(1.0, self._wing_width * 100.0)
                    credit_single = spread_width * self._credit_estimate_pct
                    if structure == "iron_condor":
                        credit_per_contract = credit_single * 2.0
                        max_loss_per_contract = spread_width
                    else:
                        credit_per_contract = credit_single
                        max_loss_per_contract = max(1.0, spread_width - credit_single)

                    contracts = min(
                        self._max_contracts,
                        int(self._max_risk_per_trade / max_loss_per_contract) if max_loss_per_contract > 0 else 1,
                    )
                    contracts = max(1, contracts)

                    total_credit = credit_per_contract * contracts
                    total_max_loss = max_loss_per_contract * contracts
                    rr_ratio = (total_credit / total_max_loss) if total_max_loss > 0 else 0.0

                    confidence = 0.0
                    if vix_level is not None:
                        if vix_level > 20:
                            confidence += 0.25
                        elif vix_level >= 15:
                            confidence += 0.15
                        else:
                            confidence += 0.05
                    else:
                        confidence += 0.10

                    if self._in_entry_window(et_now):
                        confidence += 0.15
                    if self._use_sma5_filter and bias != "neutral":
                        confidence += 0.10

                    regime_fit = 0.5
                    if regime:
                        regime_type = regime.regime_type.value
                        if regime_type == "high_vol":
                            regime_fit = 0.9
                            confidence += 0.15
                        elif regime_type == "ranging":
                            regime_fit = 0.85
                            confidence += 0.15
                        elif regime_type in ("trending_up", "trending_down"):
                            regime_fit = 0.55
                            confidence += 0.05
                        elif regime_type == "event_day":
                            regime_fit = 0.1
                            confidence -= 0.30
                        else:
                            regime_fit = 0.4

                    if is_event_day:
                        confidence -= self._event_day_confidence_penalty
                    elif not event_calendar.has_earnings(underlying):
                        confidence += 0.10

                    confidence = min(1.0, max(0.0, confidence))

                    win_rate = self._structure_win_rate(structure)
                    edge_pct = self._edge_pct(
                        win_rate=win_rate,
                        credit=total_credit,
                        max_loss=total_max_loss,
                    )
                    adjusted_rr = self._adjusted_rr(
                        win_rate=win_rate,
                        credit=total_credit,
                        max_loss=total_max_loss,
                    )

                    candidate_rows.append(TradeCandidate(
                        strategy_name=self.name,
                        candidate_type=candidate_type,
                        symbol=underlying,
                        direction=direction,
                        entry_price=round(current_price, 2),
                        stop_price=round(short_strike, 2),
                        target_price=round(current_price, 2),
                        risk_dollars=round(total_max_loss, 2),
                        suggested_qty=contracts,
                        risk_reward_ratio=round(adjusted_rr, 2),
                        confidence=round(confidence, 2),
                        edge_estimate_pct=round(edge_pct, 2),
                        regime_fit=round(regime_fit, 2),
                        catalyst=f"vix_{int(round(vix_level)) if vix_level is not None else 'na'}_{bias}_premium",
                        time_horizon_minutes=45,
                        expiry_time=expiry_time,
                        options_structure=structure,
                        short_strike=round(short_strike, 2),
                        long_strike=round(long_strike, 2),
                        contracts=contracts,
                        credit_received=round(total_credit, 2),
                        max_loss=round(total_max_loss, 2),
                        metadata={
                            "bias": bias,
                            "atr": round(atr, 2),
                            "call_short_strike": call_short,
                            "call_long_strike": call_long,
                            "win_rate": round(win_rate, 4),
                            "raw_rr": round(rr_ratio, 4),
                            "adjusted_rr": round(adjusted_rr, 4),
                            "event_day": is_event_day,
                        },
                    ))
                    details.append({
                        "symbol": underlying,
                        "atr": round(atr, 2),
                        "bias": bias,
                        "structure": structure,
                        "contracts": contracts,
                        "credit": round(total_credit, 2),
                        "max_loss": round(total_max_loss, 2),
                        "edge_pct": round(edge_pct, 2),
                        "adjusted_rr": round(adjusted_rr, 2),
                    })
                except Exception as exc:
                    self._log.debug(
                        "assess_candidate_build_failed",
                        strategy=self.name,
                        symbol=underlying,
                        error=str(exc),
                    )
                    continue

            if not candidate_rows:
                self._log.info(
                    "assess_complete",
                    strategy=self.name,
                    num_candidates=0,
                    num_raw_scanned=raw_scanned,
                )
                return OpportunityAssessment()

            candidate_rows.sort(key=lambda c: c.expected_value, reverse=True)
            top_candidates = candidate_rows[:3]
            avg_rr = sum(c.risk_reward_ratio for c in candidate_rows) / len(candidate_rows)
            avg_conf = sum(c.confidence for c in candidate_rows) / len(candidate_rows)
            avg_edge = sum(c.edge_estimate_pct for c in candidate_rows) / len(candidate_rows)

            self._log.info(
                "assess_complete",
                strategy=self.name,
                num_candidates=len(candidate_rows),
                num_raw_scanned=raw_scanned,
            )
            return OpportunityAssessment(
                num_candidates=len(candidate_rows),
                avg_risk_reward=round(avg_rr, 2),
                confidence=round(max(0.0, min(1.0, avg_conf)), 2),
                estimated_daily_trades=min(len(candidate_rows), self.config.max_positions),
                estimated_edge_pct=round(avg_edge, 2),
                details=details[:5],
                candidates=top_candidates,
            )
        except Exception as exc:
            self._log.error("assess_failed", strategy=self.name, error=str(exc), exc_info=True)
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
                "open_order_id": trade.open_order_id,
                "short_occ_symbol": trade.short_occ_symbol,
                "long_occ_symbol": trade.long_occ_symbol,
                "call_short_occ_symbol": trade.call_short_occ_symbol,
                "call_long_occ_symbol": trade.call_long_occ_symbol,
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
                open_order_id=saved.get("open_order_id", ""),
                short_occ_symbol=saved.get("short_occ_symbol", ""),
                long_occ_symbol=saved.get("long_occ_symbol", ""),
                call_short_occ_symbol=saved.get("call_short_occ_symbol", ""),
                call_long_occ_symbol=saved.get("call_long_occ_symbol", ""),
            )
