"""Options Premium Selling strategy.

Sells credit spreads and iron condors on SPY/QQQ to capture theta
decay. Uses SMA5 contrarian filter for directional bias.

IEX limitation: Options quotes are 15-min delayed. This strategy uses
the underlying price + modeled delta rather than live options greeks.
Trades may be flagged as "simulated" if options orders aren't supported.
"""

from __future__ import annotations

import math
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
STRIKE_INCREMENTS: dict[str, float] = {
    "SPY": 1.0,
    "QQQ": 1.0,
    "IWM": 1.0,
}


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

    Sells premium on SPY/QQQ during elevated vol using SMA5 contrarian
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
        self._underlyings = params.get("underlyings", ["SPY", "QQQ"])
        self._structure = params.get("structure", "credit_spread")
        self._short_delta_target = params.get("short_delta_target", 14)
        self._wing_width = params.get("wing_width", 5)

        # Entry window
        self._entry_start_hour = params.get("entry_start_hour", 10)
        self._entry_start_minute = params.get("entry_start_minute", 0)
        self._entry_end_hour = params.get("entry_end_hour", 11)
        self._entry_end_minute = params.get("entry_end_minute", 30)

        # DTE
        self._min_dte = int(params.get("min_dte", params.get("target_dte", 2)))
        self._max_dte = int(params.get("max_dte", 5))
        self._max_hold_days = int(params.get("max_hold_days", 5))

        # Filters
        self._min_vix_proxy = params.get("min_vix_proxy", 15.0)
        self._use_sma5_filter = params.get("use_sma5_filter", True)
        self._credit_estimate_pct = params.get("credit_estimate_pct", 0.30)
        self._strike_atr_distance = float(params.get("strike_atr_distance", 1.0))
        self._win_rate_credit_spread = params.get("win_rate_credit_spread", 0.70)
        self._win_rate_iron_condor = params.get("win_rate_iron_condor", 0.72)
        self._event_day_confidence_penalty = params.get("event_day_confidence_penalty", 0.25)
        self._min_credit_per_spread = float(params.get("min_credit_per_spread", 50.0))
        self._min_credit_per_contract = float(params.get("min_credit_per_contract", 10.0))
        self._min_credit_per_contract = max(0.0, self._min_credit_per_contract)
        self._min_credit_to_max_loss_ratio = float(params.get("min_credit_to_max_loss_ratio", 0.05))
        self._min_credit_to_max_loss_ratio = max(0.0, min(1.0, self._min_credit_to_max_loss_ratio))
        self._require_live_credit_estimate = bool(params.get("require_live_credit_estimate", True))
        self._max_commission_per_spread = float(params.get("max_commission_per_spread", 4.0))

        # Exit
        self._profit_target_pct = params.get("profit_target_pct", 50.0)
        self._stop_loss_pct = params.get("stop_loss_pct", 200.0)
        self._stop_loss_multiple = float(params.get("stop_loss_multiple", 3.0))
        self._close_before_expiry_days = int(params.get("close_before_expiry_days", 1))
        self._min_hold_minutes_for_profit_target = int(params.get("min_hold_minutes_for_profit_target", 0))
        self._close_before_minutes = params.get("close_before_minutes", 15)
        self._expiry_risk_close_minutes = int(
            params.get("expiry_risk_close_minutes", max(15, self._close_before_minutes)),
        )

        # Sizing
        self._max_risk_per_trade = params.get("max_risk_per_trade", 500)
        self._max_contracts = int(params.get("max_contracts", 5))
        self._max_contracts = max(1, min(self._max_contracts, 5))
        self._max_contracts_per_spread = int(
            params.get("max_contracts_per_spread", self._max_contracts),
        )
        self._max_contracts_per_spread = max(
            1,
            min(self._max_contracts_per_spread, self._max_contracts),
        )
        self._sizing_method = str(params.get("sizing_method", "exercise_exposure")).lower()
        self._max_loss_risk_budget_pct = float(params.get("max_loss_risk_budget_pct", 2.0))
        self._exercise_exposure_cap_pct = float(params.get("exercise_exposure_cap_pct", 20.0))
        self._brain_max_contracts_override: int | None = None

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

    def set_brain_contract_cap(self, max_contracts: int | None) -> None:
        """Allow Brain decisions to override per-spread contract caps at runtime."""
        if max_contracts is None:
            self._brain_max_contracts_override = None
            self._log.info(
                "options_brain_contract_cap_cleared",
                configured_max_contracts=self._max_contracts_per_spread,
            )
            return

        clamped = max(1, min(int(max_contracts), self._max_contracts))
        self._brain_max_contracts_override = clamped
        self._log.info(
            "options_brain_contract_cap_set",
            requested=max_contracts,
            effective=clamped,
            configured_max_contracts=self._max_contracts_per_spread,
            hard_max_contracts=self._max_contracts,
        )

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
        utc_now = datetime.now(pytz.UTC)

        for trade_key, trade in list(self._trades.items()):
            close_reason = ""
            days_held = 0
            if trade.entry_time is not None:
                days_held = max(0, (et_now.date() - trade.entry_time.astimezone(ET).date()).days)
            days_to_expiry = None
            if trade.expiration is not None:
                days_to_expiry = (trade.expiration - et_now.date()).days

            # Force-close same-day expiry options near the close to prevent exercise.
            if (
                trade.expiration is not None
                and trade.expiration <= et_now.date()
                and (et_now.hour > 15 or (et_now.hour == 15 and et_now.minute >= 45))
            ):
                close_reason = f"prevent_exercise_{self._expiry_risk_close_minutes}m"

            # Get current underlying price
            current_price = self._get_current_price(trade.underlying)
            if current_price is None:
                continue

            # Prefer live options marks when available; fallback to an underlying proxy.
            live_pnl = self._estimate_live_option_pnl(trade)
            if live_pnl is not None:
                estimated_pnl_pct = live_pnl["pnl_pct"]
                pnl_source = "options_mark"
                current_spread_value = max(0.0, float(live_pnl.get("close_debit", 0.0)))
            else:
                estimated_pnl_pct = self._estimate_pnl_pct(trade, current_price)
                pnl_source = "underlying_proxy"
                modeled_pnl = trade.max_profit * (estimated_pnl_pct / 100.0)
                current_spread_value = max(0.0, trade.credit_received - modeled_pnl)

            held_minutes = 0.0
            if trade.entry_time is not None:
                held_minutes = max(
                    0.0,
                    (utc_now - trade.entry_time).total_seconds() / 60.0,
                )

            # 1. Profit target (close at configured % of max profit).
            if not close_reason and estimated_pnl_pct >= self._profit_target_pct:
                if held_minutes >= self._min_hold_minutes_for_profit_target:
                    close_reason = f"profit_target: est {estimated_pnl_pct:.0f}% of max"
                else:
                    self._log.info(
                        "profit_target_deferred_min_hold",
                        underlying=trade.underlying,
                        structure=trade.structure,
                        held_minutes=round(held_minutes, 1),
                        min_hold_minutes=self._min_hold_minutes_for_profit_target,
                        est_pnl_pct=round(estimated_pnl_pct, 2),
                        pnl_source=pnl_source,
                    )

            # 2. Stop loss based on spread repricing vs entry credit.
            elif not close_reason and current_spread_value >= (trade.credit_received * self._stop_loss_multiple):
                close_reason = (
                    f"stop_loss: spread_value={current_spread_value:.2f} "
                    f">= {self._stop_loss_multiple:.1f}x_credit"
                )

            # 3. Exit before expiry week/day gamma acceleration.
            if (
                not close_reason
                and days_to_expiry is not None
                and days_to_expiry > 0
                and days_to_expiry <= self._close_before_expiry_days
            ):
                close_reason = f"pre_expiry_{self._close_before_expiry_days}d"

            # 4. Max hold as swing position.
            if not close_reason and days_held >= self._max_hold_days:
                close_reason = f"max_hold_{self._max_hold_days}_days"

            if close_reason:
                self._close_trade(trade_key, trade, close_reason)
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=trade.underlying,
                    direction=SignalDirection.CLOSE,
                    reason=close_reason,
                    metadata={
                        "structure": trade.structure,
                        "simulated": trade.simulated,
                        "pnl_source": pnl_source,
                        "held_minutes": round(held_minutes, 1),
                        "days_held": days_held,
                        "days_to_expiry": days_to_expiry,
                    },
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

            # Swing strikes: target roughly 15-25 delta using ATR distance.
            strike_distance = max(0.5, atr * self._strike_atr_distance)
            strike_increment = self._get_strike_increment(underlying)
            wing_width = self._normalize_wing_width(self._wing_width, strike_increment)

            # Determine structure based on bias
            if bias == "bullish" or (bias == "neutral" and self._structure == "credit_spread"):
                structure = "put_spread"
                raw_short_strike = current_price - strike_distance
                short_strike = self._round_strike(raw_short_strike, strike_increment, mode="down")
                long_strike = short_strike - wing_width
            elif bias == "bearish":
                structure = "call_spread"
                raw_short_strike = current_price + strike_distance
                short_strike = self._round_strike(raw_short_strike, strike_increment, mode="up")
                long_strike = short_strike + wing_width
            elif bias == "neutral":
                structure = "iron_condor"
                raw_put_short_strike = current_price - strike_distance
                short_strike = self._round_strike(raw_put_short_strike, strike_increment, mode="down")
                long_strike = short_strike - wing_width
            else:
                continue

            # Calculate contracts and risk
            spread_width = wing_width * 100  # Per contract in dollars
            estimated_credit_per_contract = spread_width * self._credit_estimate_pct
            # For iron condor, double the credit (two sides)
            call_short = None
            call_long = None
            if structure == "iron_condor":
                raw_call_short_strike = current_price + strike_distance
                call_short = self._round_strike(raw_call_short_strike, strike_increment, mode="up")
                call_long = call_short + wing_width
                estimated_credit_per_contract *= 2
            max_loss_per_contract = max(1.0, spread_width - estimated_credit_per_contract)

            modeled_ok, modeled_ratio, modeled_reason = self._passes_credit_quality_gate(
                credit_per_contract=estimated_credit_per_contract,
                max_loss_per_contract=max_loss_per_contract,
            )
            if not modeled_ok:
                self._log.info(
                    "options_credit_quality_filtered",
                    stage="entry_model",
                    reason=modeled_reason,
                    underlying=underlying,
                    structure=structure,
                    credit_per_contract=round(estimated_credit_per_contract, 2),
                    credit_per_contract_quote=round(estimated_credit_per_contract / 100.0, 4),
                    max_loss_per_contract=round(max_loss_per_contract, 2),
                    credit_to_max_loss_ratio=round(modeled_ratio, 4),
                    min_credit_per_contract=round(self._min_credit_per_contract, 2),
                    min_credit_per_contract_quote=round(self._min_credit_per_contract / 100.0, 4),
                    min_credit_to_max_loss_ratio=round(self._min_credit_to_max_loss_ratio, 4),
                )
                continue

            contracts = self._size_contracts_for_structure(
                structure=structure,
                underlying=underlying,
                short_strike=short_strike,
                max_loss_per_contract=max_loss_per_contract,
                call_short_strike=call_short,
                context="entry",
            )
            if contracts <= 0:
                continue

            entry_expiration = self._find_expiry(
                underlying,
                min_dte=self._min_dte,
                max_dte=self._max_dte,
            )
            if entry_expiration is None:
                self._log.info(
                    "options_no_valid_expiry",
                    underlying=underlying,
                    min_dte=self._min_dte,
                    max_dte=self._max_dte,
                )
                continue
            live_credit_per_contract = self._estimate_live_credit_per_contract(
                underlying=underlying,
                short_strike=short_strike,
                long_strike=long_strike,
                structure=structure,
                expiration=entry_expiration,
                call_short_strike=call_short,
                call_long_strike=call_long,
            )
            selected_credit_per_contract = estimated_credit_per_contract
            if live_credit_per_contract is None:
                log_fn = self._log.warning if self._require_live_credit_estimate else self._log.info
                log_fn(
                    "options_live_credit_unavailable",
                    underlying=underlying,
                    structure=structure,
                    contracts=contracts,
                    reason="falling_back_to_modeled_credit_estimate",
                )
            else:
                live_max_loss_per_contract = max(1.0, spread_width - live_credit_per_contract)
                live_ok, live_ratio, live_reason = self._passes_credit_quality_gate(
                    credit_per_contract=live_credit_per_contract,
                    max_loss_per_contract=live_max_loss_per_contract,
                )
                self._log.info(
                    "options_live_credit_estimate",
                    underlying=underlying,
                    structure=structure,
                    contracts=contracts,
                    credit_per_contract=round(live_credit_per_contract, 2),
                    credit_per_contract_quote=round(live_credit_per_contract / 100.0, 4),
                    max_loss_per_contract=round(live_max_loss_per_contract, 2),
                    credit_to_max_loss_ratio=round(live_ratio, 4),
                    passed=live_ok,
                    reason=live_reason if live_reason else "",
                )
                if not live_ok:
                    self._log.info(
                        "options_credit_quality_filtered",
                        stage="entry_live",
                        reason=live_reason,
                        underlying=underlying,
                        structure=structure,
                        contracts=contracts,
                        credit_per_contract=round(live_credit_per_contract, 2),
                        credit_per_contract_quote=round(live_credit_per_contract / 100.0, 4),
                        max_loss_per_contract=round(live_max_loss_per_contract, 2),
                        credit_to_max_loss_ratio=round(live_ratio, 4),
                        min_credit_per_contract=round(self._min_credit_per_contract, 2),
                        min_credit_per_contract_quote=round(self._min_credit_per_contract / 100.0, 4),
                        min_credit_to_max_loss_ratio=round(self._min_credit_to_max_loss_ratio, 4),
                    )
                    continue
                selected_credit_per_contract = live_credit_per_contract
                max_loss_per_contract = live_max_loss_per_contract

            total_risk = contracts * max_loss_per_contract
            total_credit = contracts * selected_credit_per_contract
            estimated_round_trip_cost = contracts * self._max_commission_per_spread
            if structure == "iron_condor":
                estimated_round_trip_cost *= 2

            # Reserve capital for margin
            net_credit_after_cost = total_credit - estimated_round_trip_cost
            if total_credit < self._min_credit_per_spread or net_credit_after_cost <= 0:
                self._log.info(
                    "options_credit_filtered",
                    underlying=underlying,
                    structure=structure,
                    contracts=contracts,
                    gross_credit=round(total_credit, 2),
                    est_round_trip_cost=round(estimated_round_trip_cost, 2),
                    net_credit_after_cost=round(net_credit_after_cost, 2),
                    min_credit_per_spread=self._min_credit_per_spread,
                )
                continue

            capital_needed = total_risk
            if not self.reserve_capital(capital_needed):
                continue

            # Attempt to submit the trade as a real MLEG order
            order_result = self._submit_credit_spread(
                underlying, short_strike, long_strike,
                structure, contracts, entry_expiration,
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
                expiration=entry_expiration,
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
                expiration=entry_expiration.isoformat(),
                short_strike=short_strike,
                long_strike=long_strike,
                call_short=call_short,
                call_long=call_long,
                contracts=contracts,
                credit=round(total_credit, 2),
                max_loss=round(total_risk, 2),
                credit_per_contract=round(selected_credit_per_contract, 2),
                credit_per_contract_quote=round(selected_credit_per_contract / 100.0, 4),
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

    def _get_strike_increment(self, underlying: str) -> float:
        increment = STRIKE_INCREMENTS.get(underlying.upper(), 1.0)
        if increment <= 0:
            return 1.0
        return float(increment)

    @staticmethod
    def _normalize_wing_width(wing_width: float, increment: float) -> float:
        if increment <= 0:
            increment = 1.0
        units = int(math.floor((float(wing_width) / increment) + 0.5))
        units = max(1, units)
        return round(units * increment, 6)

    @staticmethod
    def _round_strike(strike: float, increment: float, *, mode: str) -> float:
        if increment <= 0:
            increment = 1.0
        scaled = strike / increment
        if mode == "down":
            rounded = math.floor(scaled + 1e-9)
        elif mode == "up":
            rounded = math.ceil(scaled - 1e-9)
        else:
            rounded = round(scaled)
        return round(rounded * increment, 6)

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(parsed):
            return None
        return parsed

    def _passes_credit_quality_gate(
        self,
        *,
        credit_per_contract: float,
        max_loss_per_contract: float,
    ) -> tuple[bool, float, str]:
        ratio = (credit_per_contract / max_loss_per_contract) if max_loss_per_contract > 0 else 0.0
        if credit_per_contract < self._min_credit_per_contract:
            return False, ratio, "min_credit_per_contract"
        if ratio < self._min_credit_to_max_loss_ratio:
            return False, ratio, "min_credit_to_max_loss_ratio"
        return True, ratio, ""

    def _find_expiry(self, underlying: str, *, min_dte: int, max_dte: int) -> date | None:
        """Find the best available expiry between min_dte and max_dte days out."""
        today = date.today()
        min_dte = max(1, int(min_dte))
        max_dte = max(min_dte, int(max_dte))

        def _parse_expiries(chain_df: Any) -> list[date]:
            if chain_df is None or chain_df.empty or "expiration" not in chain_df.columns:
                return []

            expiries: list[date] = []
            for raw in sorted(chain_df["expiration"].dropna().unique()):
                parsed: date | None = None
                if isinstance(raw, datetime):
                    parsed = raw.date()
                elif isinstance(raw, date):
                    parsed = raw
                else:
                    raw_str = str(raw).strip()
                    for fmt in ("%Y-%m-%d", "%Y%m%d"):
                        try:
                            parsed = datetime.strptime(raw_str, fmt).date()
                            break
                        except ValueError:
                            continue
                if parsed is not None and parsed >= today:
                    expiries.append(parsed)
            return sorted(set(expiries))

        try:
            chain = self.data_provider.get_option_chain(underlying)
        except Exception:
            chain = None

        expiries = _parse_expiries(chain)

        for exp in expiries:
            dte = (exp - today).days
            if min_dte <= dte <= max_dte:
                return exp

        for exp in expiries:
            if (exp - today).days >= min_dte:
                return exp

        for dte in range(min_dte, max_dte + 1):
            candidate = today + timedelta(days=dte)
            if candidate.weekday() >= 5:
                continue
            try:
                probe = self.data_provider.get_option_chain(underlying, expiration=candidate)
            except Exception:
                continue
            if probe is not None and not probe.empty:
                return candidate

        if expiries:
            return expiries[0]
        return today + timedelta(days=min_dte)

    def _estimate_live_credit_per_contract(
        self,
        *,
        underlying: str,
        short_strike: float,
        long_strike: float,
        structure: str,
        expiration: date,
        call_short_strike: float | None = None,
        call_long_strike: float | None = None,
    ) -> float | None:
        try:
            chain = self.data_provider.get_option_chain(underlying, expiration=expiration)
        except Exception:
            self._log.debug(
                "options_live_credit_estimate_failed",
                underlying=underlying,
                structure=structure,
                reason="option_chain_fetch_failed",
            )
            return None
        if chain is None or chain.empty:
            return None

        primary_option_type = "put" if structure in {"put_spread", "iron_condor"} else "call"
        base_credit_quote = self._estimate_vertical_credit_quote(
            chain=chain,
            option_type=primary_option_type,
            short_strike=short_strike,
            long_strike=long_strike,
        )
        if base_credit_quote is None:
            return None

        total_quote_credit = base_credit_quote
        if structure == "iron_condor":
            if call_short_strike is None or call_long_strike is None:
                return None
            call_credit_quote = self._estimate_vertical_credit_quote(
                chain=chain,
                option_type="call",
                short_strike=call_short_strike,
                long_strike=call_long_strike,
            )
            if call_credit_quote is None:
                return None
            total_quote_credit += call_credit_quote

        return max(0.0, total_quote_credit * 100.0)

    def _estimate_vertical_credit_quote(
        self,
        *,
        chain: Any,
        option_type: str,
        short_strike: float,
        long_strike: float,
    ) -> float | None:
        short_row = self._find_chain_row(chain, option_type, short_strike)
        long_row = self._find_chain_row(chain, option_type, long_strike)
        if short_row is None or long_row is None:
            return None

        short_sell = self._estimate_leg_fill_quote(short_row, opening_side="sell")
        long_buy = self._estimate_leg_fill_quote(long_row, opening_side="buy")
        if short_sell is None or long_buy is None:
            return None

        return max(0.0, short_sell - long_buy)

    def _find_chain_row(self, chain: Any, option_type: str, target_strike: float) -> Any | None:
        normalized_type = option_type.lower()
        best_row: Any | None = None
        best_diff = float("inf")

        for _, row in chain.iterrows():
            row_type = str(row.get("type", "")).lower()
            if row_type != normalized_type:
                continue
            strike = self._as_float(row.get("strike"))
            if strike is None:
                continue
            diff = abs(strike - target_strike)
            if diff < best_diff:
                best_diff = diff
                best_row = row

        if best_row is None:
            return None
        # ETF strikes are typically in 0.50 increments. Reject mismatched rows.
        if best_diff > 0.51:
            return None
        return best_row

    def _estimate_leg_fill_quote(self, row: Any, *, opening_side: str) -> float | None:
        bid = self._as_float(row.get("bid"))
        ask = self._as_float(row.get("ask"))
        last = self._as_float(row.get("last"))

        midpoint = None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            midpoint = (bid + ask) / 2.0

        if opening_side == "sell":
            if bid is not None and bid > 0:
                return bid
            if midpoint is not None:
                return midpoint
            if last is not None and last > 0:
                return last
            return None

        if ask is not None and ask > 0:
            return ask
        if midpoint is not None:
            return midpoint
        if last is not None and last > 0:
            return last
        return None

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
        Uses a conservative proxy when live option marks are unavailable.
        """
        if trade.max_profit <= 0:
            return 0.0

        width = max(0.5, float(self._wing_width))

        if trade.structure == "put_spread":
            distance = current_price - trade.short_strike
            if distance >= 0:
                return min(100.0, (distance / width) * 100.0)
            return (distance / width) * 100.0

        if trade.structure == "call_spread":
            distance = trade.short_strike - current_price
            if distance >= 0:
                return min(100.0, (distance / width) * 100.0)
            return (distance / width) * 100.0

        if trade.structure == "iron_condor":
            if trade.call_short_strike is None:
                return 0.0
            if trade.short_strike <= current_price <= trade.call_short_strike:
                safe_buffer = min(
                    current_price - trade.short_strike,
                    trade.call_short_strike - current_price,
                )
                return min(100.0, (safe_buffer / width) * 100.0)
            if current_price < trade.short_strike:
                return ((current_price - trade.short_strike) / width) * 100.0
            return ((trade.call_short_strike - current_price) / width) * 100.0

        return 0.0

    @staticmethod
    def _normalize_option_local_symbol(symbol: str | None) -> str:
        return str(symbol or "").strip()

    def _estimate_live_option_pnl(self, trade: PremiumTrade) -> dict[str, float] | None:
        """Estimate spread P&L using broker-reported option leg marks."""
        if not hasattr(self.executor, "get_option_positions"):
            return None

        try:
            positions = self.executor.get_option_positions()
        except Exception:
            return None

        if not positions:
            return None

        leg_symbols = [
            self._normalize_option_local_symbol(trade.short_occ_symbol),
            self._normalize_option_local_symbol(trade.long_occ_symbol),
        ]
        if trade.call_short_occ_symbol and trade.call_long_occ_symbol:
            leg_symbols.extend([
                self._normalize_option_local_symbol(trade.call_short_occ_symbol),
                self._normalize_option_local_symbol(trade.call_long_occ_symbol),
            ])
        if any(not s for s in leg_symbols):
            return None

        by_symbol = {
            self._normalize_option_local_symbol(pos.get("local_symbol")): pos
            for pos in positions
        }

        entry_credit = 0.0
        close_debit = 0.0
        for symbol in leg_symbols:
            pos = by_symbol.get(symbol)
            if pos is None:
                return None

            qty = float(pos.get("qty", 0.0))
            avg_cost = float(pos.get("average_cost", 0.0))
            mark_price = float(pos.get("market_price", 0.0))
            if qty == 0 or avg_cost <= 0 or mark_price <= 0:
                return None

            mark_cost = mark_price * 100.0
            entry_credit += (-qty) * avg_cost
            close_debit += (-qty) * mark_cost

        if entry_credit <= 0:
            return None

        pnl_dollars = entry_credit - close_debit
        pnl_pct = (pnl_dollars / entry_credit) * 100.0
        return {
            "pnl_pct": pnl_pct,
            "pnl_dollars": pnl_dollars,
            "entry_credit": entry_credit,
            "close_debit": close_debit,
        }

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
            # Executor (e.g. IBKR stub) doesn't implement MLEG yet - simulate
            self._log.info("options_executor_no_mleg", underlying=underlying)
            return None

        if structure == "put_spread" and short_strike <= long_strike:
            self._log.error(
                "options_invalid_strike_geometry",
                structure=structure,
                short_strike=short_strike,
                long_strike=long_strike,
            )
            return None
        if structure == "call_spread" and short_strike >= long_strike:
            self._log.error(
                "options_invalid_strike_geometry",
                structure=structure,
                short_strike=short_strike,
                long_strike=long_strike,
            )
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
        primary_right = "P" if put_or_call == "put" else "C"

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
                # Can't complete iron condor - fall back to the put spread only
                self._log.warning(
                    "iron_condor_call_lookup_failed",
                    underlying=underlying,
                    call_short_strike=call_short_strike,
                    call_long_strike=call_long_strike,
                )

        self._log.info(
            "options_order_construction",
            strategy=self.name,
            intended_structure=structure,
            direction="short",
            symbol=underlying,
            short_strike=short_strike,
            long_strike=long_strike,
            leg1_action="SELL",
            leg1_symbol=short_occ,
            leg1_strike=short_strike,
            leg1_right=primary_right,
            leg2_action="BUY",
            leg2_symbol=long_occ,
            leg2_strike=long_strike,
            leg2_right=primary_right,
            leg3_action="SELL" if call_short_occ else "",
            leg3_symbol=call_short_occ,
            leg3_strike=call_short_strike if call_short_occ else None,
            leg3_right="C" if call_short_occ else "",
            leg4_action="BUY" if call_long_occ else "",
            leg4_symbol=call_long_occ,
            leg4_strike=call_long_strike if call_long_occ else None,
            leg4_right="C" if call_long_occ else "",
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
          sell-to-open short -> buy-to-close
          buy-to-open long   -> sell-to-close

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

        # For real trades, close by reversing the MLEG legs.
        # Prefer live mark-based option P&L when available.
        live_pnl = self._estimate_live_option_pnl(trade)
        if live_pnl is not None:
            pnl = live_pnl["pnl_dollars"]
        else:
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
            exit_price=self._get_current_price(trade.underlying) or 0.0,
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

    def _get_account_equity(self) -> float | None:
        """Best-effort account equity fetch for exercise-exposure caps."""
        try:
            account = self.executor.get_account()
            equity = float(getattr(account, "equity", 0.0) or 0.0)
            if equity > 0:
                return equity
        except Exception:
            self._log.debug("options_account_equity_fetch_failed")
        return None

    def _cap_contracts_by_exercise_exposure(
        self,
        *,
        contracts: int,
        strike_for_exposure: float,
        underlying: str,
        structure: str,
        context: str,
    ) -> int:
        """Cap contracts so notional exercise exposure stays within policy."""
        if contracts <= 0:
            return 0
        if strike_for_exposure <= 0:
            return contracts

        max_capital_pct = max(0.0, self._exercise_exposure_cap_pct) / 100.0
        if max_capital_pct <= 0:
            return contracts

        account_equity = self._get_account_equity()
        if account_equity is None:
            return contracts

        exposure_per_contract = strike_for_exposure * 100.0
        if exposure_per_contract <= 0:
            return contracts

        max_contracts_by_exposure = int(account_equity * max_capital_pct / exposure_per_contract)
        max_contracts_by_exposure = max(1, max_contracts_by_exposure)
        capped_contracts = min(contracts, max_contracts_by_exposure)
        if capped_contracts < contracts:
            self._log.info(
                "contracts_capped_exercise_exposure",
                context=context,
                underlying=underlying,
                structure=structure,
                original=contracts,
                capped=capped_contracts,
                strike_for_exposure=round(strike_for_exposure, 2),
                exposure_per_contract=round(exposure_per_contract, 2),
                equity=round(account_equity, 2),
                max_allowed=round(account_equity * max_capital_pct, 2),
                max_capital_pct=round(max_capital_pct * 100.0, 2),
            )
        return capped_contracts

    def _spread_risk_budget(self, *, context: str) -> float:
        account_equity = self._get_account_equity()
        if account_equity is None or account_equity <= 0:
            account_equity = self._total_capital if self._total_capital > 0 else 0.0

        equity_budget = 0.0
        if account_equity > 0 and self._max_loss_risk_budget_pct > 0:
            equity_budget = account_equity * (self._max_loss_risk_budget_pct / 100.0)

        if context == "entry":
            if self.available_capital > 0 and self._brain_max_contracts_override is not None:
                return self.available_capital
            if self.available_capital > 0 and equity_budget > 0:
                return min(self.available_capital, equity_budget)
            if self.available_capital > 0:
                return self.available_capital
            return 0.0

        if equity_budget > 0:
            return equity_budget
        if self._max_risk_per_trade > 0:
            return float(self._max_risk_per_trade)
        if self._total_capital > 0:
            return self._total_capital
        return 0.0

    def _effective_max_contracts_per_spread(self) -> int:
        configured_cap = max(1, min(self._max_contracts_per_spread, self._max_contracts))
        if self._brain_max_contracts_override is None:
            return configured_cap
        return max(1, min(self._brain_max_contracts_override, self._max_contracts))

    def _size_contracts_for_structure(
        self,
        *,
        structure: str,
        underlying: str,
        short_strike: float,
        max_loss_per_contract: float,
        call_short_strike: float | None,
        context: str,
    ) -> int:
        if max_loss_per_contract <= 0:
            return 0 if context == "entry" else 1

        effective_contract_cap = self._effective_max_contracts_per_spread()
        if structure in {"put_spread", "call_spread"} and self._sizing_method == "max_loss":
            risk_budget = self._spread_risk_budget(context=context)
            raw_contracts = int(risk_budget / max_loss_per_contract) if risk_budget > 0 else 0
            contracts = min(effective_contract_cap, raw_contracts)
            if context == "assess":
                contracts = max(contracts, 1)
            self._log.info(
                "options_contract_sizing",
                context=context,
                underlying=underlying,
                structure=structure,
                sizing_method="max_loss",
                contracts=contracts,
                max_contracts_per_spread=effective_contract_cap,
                configured_max_contracts_per_spread=self._max_contracts_per_spread,
                brain_max_contracts_override=self._brain_max_contracts_override,
                max_loss_per_contract=round(max_loss_per_contract, 2),
                risk_budget=round(risk_budget, 2),
                available_capital=round(self.available_capital, 2),
                risk_budget_pct=round(self._max_loss_risk_budget_pct, 2),
            )
            self._log.info(
                "options_sizing_decision",
                context=context,
                underlying=underlying,
                structure=structure,
                sizing_method="max_loss",
                brain_cap=self._brain_max_contracts_override,
                config_max=self._max_contracts_per_spread,
                budget_contracts=raw_contracts,
                final_contracts=contracts,
            )
            return contracts

        budget_contracts = (
            int(self._max_risk_per_trade / max_loss_per_contract) if max_loss_per_contract > 0 else 0
        )
        contracts = min(
            effective_contract_cap,
            budget_contracts,
        )
        contracts = max(contracts, 1)
        pre_exposure_contracts = contracts
        exposure_strike = short_strike
        if structure == "iron_condor" and call_short_strike is not None and call_short_strike > 0:
            exposure_strike = max(short_strike, call_short_strike)
        contracts = self._cap_contracts_by_exercise_exposure(
            contracts=contracts,
            strike_for_exposure=exposure_strike,
            underlying=underlying,
            structure=structure,
            context=context,
        )
        self._log.info(
            "options_contract_sizing",
            context=context,
            underlying=underlying,
            structure=structure,
            sizing_method="exercise_exposure",
            contracts=contracts,
            max_contracts_per_spread=effective_contract_cap,
            configured_max_contracts_per_spread=self._max_contracts_per_spread,
            brain_max_contracts_override=self._brain_max_contracts_override,
            max_loss_per_contract=round(max_loss_per_contract, 2),
            available_capital=round(self.available_capital, 2),
            exercise_exposure_cap_pct=round(self._exercise_exposure_cap_pct, 2),
        )
        self._log.info(
            "options_sizing_decision",
            context=context,
            underlying=underlying,
            structure=structure,
            sizing_method="exercise_exposure",
            brain_cap=self._brain_max_contracts_override,
            config_max=self._max_contracts_per_spread,
            budget_contracts=budget_contracts,
            pre_exposure_contracts=pre_exposure_contracts,
            final_contracts=contracts,
        )
        return contracts

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
            from algotrader.intelligence.calendar.events import EventCalendar
            from algotrader.strategy_selector.candidate import CandidateType, TradeCandidate

            regime_blocked = regime is not None and regime.regime_type.value in self._blocked_regimes
            regime_allowed = regime is None or regime.regime_type.value in self._allowed_regimes
            self._log.info(
                "options_regime_check",
                regime=regime_type,
                blocked=regime_blocked,
                allowed=regime_allowed,
                allowed_regimes=self._allowed_regimes,
                blocked_regimes=self._blocked_regimes,
            )
            if regime_blocked or not regime_allowed:
                self._log.warning(
                    "options_regime_rejected",
                    regime=regime_type,
                    blocked=regime_blocked,
                )
                return _complete(OpportunityAssessment())

            et_now = datetime.now(ET)
            window_end = et_now.replace(
                hour=self._entry_end_hour,
                minute=self._entry_end_minute,
                second=0,
                microsecond=0,
            )
            filter_counts = {
                "entry_window_closed": 0,
                "vix_below_min": 0,
                "missing_atr_or_price": 0,
                "credit_quality_rejected": 0,
                "candidate_built": 0,
            }
            if et_now > window_end:
                filter_counts["entry_window_closed"] = 1
                self._log.info(
                    "options_scan_result",
                    regime=regime_type,
                    final_candidates=0,
                    filter_counts=filter_counts,
                )
                return _complete(OpportunityAssessment(
                    num_candidates=0,
                    confidence=0.0,
                    details=[{"reason": "entry_window_closed"}],
                    candidates=[],
                ))

            # VIX proxy check
            vix_level = regime.vix_level if regime and regime.vix_level else None
            if vix_level is not None and vix_level < self._min_vix_proxy:
                filter_counts["vix_below_min"] = 1
                self._log.info(
                    "options_scan_result",
                    regime=regime_type,
                    final_candidates=0,
                    filter_counts=filter_counts,
                )
                return _complete(OpportunityAssessment(
                    num_candidates=0,
                    confidence=0.0,
                    details=[{"reason": f"VIX {vix_level:.1f} < min {self._min_vix_proxy}"}],
                    candidates=[],
                ))

            # Count available underlyings not already traded
            available = [u for u in self._underlyings if u not in self._trades]
            raw_scanned = len(available)
            self._log.info(
                "options_scan_universe",
                regime=regime_type,
                count=len(available),
                symbols=available,
            )
            if not available:
                self._log.info(
                    "options_scan_result",
                    regime=regime_type,
                    final_candidates=0,
                    filter_counts=filter_counts,
                )
                return _complete(OpportunityAssessment())

            event_calendar = EventCalendar()
            is_event_day = event_calendar.is_event_day()
            details: list[dict] = []
            candidate_rows: list[TradeCandidate] = []

            for underlying in available:
                try:
                    atr = self._get_atr(underlying)
                    current_price = self._get_current_price(underlying)
                    if not atr or atr <= 0 or not current_price or current_price <= 0:
                        filter_counts["missing_atr_or_price"] += 1
                        continue

                    bias = self._get_sma5_bias(underlying, current_price)
                    strike_distance = max(0.5, atr * self._strike_atr_distance)
                    strike_increment = self._get_strike_increment(underlying)
                    wing_width = self._normalize_wing_width(self._wing_width, strike_increment)
                    expiry = self._find_expiry(
                        underlying,
                        min_dte=self._min_dte,
                        max_dte=self._max_dte,
                    )
                    if expiry is None:
                        filter_counts["missing_atr_or_price"] += 1
                        continue

                    # Reuse the same structure logic as _scan_entries.
                    if bias == "bullish" or (bias == "neutral" and self._structure == "credit_spread"):
                        structure = "put_spread"
                        raw_short_strike = current_price - strike_distance
                        short_strike = self._round_strike(raw_short_strike, strike_increment, mode="down")
                        long_strike = short_strike - wing_width
                        call_short = 0.0
                        call_long = 0.0
                        direction = "short"
                        candidate_type = CandidateType.CREDIT_SPREAD
                    elif bias == "bearish":
                        structure = "call_spread"
                        raw_short_strike = current_price + strike_distance
                        short_strike = self._round_strike(raw_short_strike, strike_increment, mode="up")
                        long_strike = short_strike + wing_width
                        call_short = 0.0
                        call_long = 0.0
                        direction = "short"
                        candidate_type = CandidateType.CREDIT_SPREAD
                    else:
                        structure = "iron_condor"
                        raw_put_short_strike = current_price - strike_distance
                        short_strike = self._round_strike(raw_put_short_strike, strike_increment, mode="down")
                        long_strike = short_strike - wing_width
                        raw_call_short_strike = current_price + strike_distance
                        call_short = self._round_strike(raw_call_short_strike, strike_increment, mode="up")
                        call_long = call_short + wing_width
                        direction = "neutral"
                        candidate_type = CandidateType.IRON_CONDOR

                    spread_width = max(1.0, wing_width * 100.0)
                    credit_single = spread_width * self._credit_estimate_pct
                    if structure == "iron_condor":
                        credit_per_contract = credit_single * 2.0
                    else:
                        credit_per_contract = credit_single
                    max_loss_per_contract = max(1.0, spread_width - credit_per_contract)

                    quality_ok, ratio, quality_reason = self._passes_credit_quality_gate(
                        credit_per_contract=credit_per_contract,
                        max_loss_per_contract=max_loss_per_contract,
                    )
                    if not quality_ok:
                        filter_counts["credit_quality_rejected"] += 1
                        self._log.info(
                            "options_credit_quality_filtered",
                            stage="assessment_model",
                            reason=quality_reason,
                            underlying=underlying,
                            structure=structure,
                            credit_per_contract=round(credit_per_contract, 2),
                            credit_per_contract_quote=round(credit_per_contract / 100.0, 4),
                            max_loss_per_contract=round(max_loss_per_contract, 2),
                            credit_to_max_loss_ratio=round(ratio, 4),
                            min_credit_per_contract=round(self._min_credit_per_contract, 2),
                            min_credit_per_contract_quote=round(self._min_credit_per_contract / 100.0, 4),
                            min_credit_to_max_loss_ratio=round(self._min_credit_to_max_loss_ratio, 4),
                        )
                        continue

                    contracts = self._size_contracts_for_structure(
                        structure=structure,
                        underlying=underlying,
                        short_strike=short_strike,
                        max_loss_per_contract=max_loss_per_contract,
                        call_short_strike=call_short,
                        context="assess",
                    )
                    if contracts <= 0:
                        filter_counts["missing_atr_or_price"] += 1
                        continue

                    total_credit = credit_per_contract * contracts
                    total_max_loss = max_loss_per_contract * contracts
                    rr_ratio = (total_credit / total_max_loss) if total_max_loss > 0 else 0.0

                    confidence = 0.05
                    if vix_level is not None:
                        if 18.0 <= vix_level <= 28.0:
                            confidence += 0.25
                        elif 15.0 <= vix_level < 18.0:
                            confidence += 0.18
                        elif vix_level > 28.0:
                            confidence += 0.20
                        elif vix_level >= 15:
                            confidence += 0.12
                        else:
                            confidence += 0.08
                    else:
                        confidence += 0.10

                    if self._in_entry_window(et_now):
                        confidence += 0.15
                    if self._use_sma5_filter and bias != "neutral":
                        confidence += 0.10
                    elif structure == "iron_condor":
                        confidence += 0.07

                    regime_fit = 0.5
                    if regime:
                        regime_type = regime.regime_type.value
                        if regime_type == "ranging":
                            regime_fit = 0.90
                            confidence += 0.18
                        elif regime_type == "high_vol":
                            regime_fit = 0.85
                            confidence += 0.16
                        elif regime_type in ("trending_up", "trending_down"):
                            regime_fit = 0.60
                            confidence += 0.10
                        elif regime_type == "low_vol":
                            regime_fit = 0.55
                            confidence += 0.08
                        elif regime_type == "event_day":
                            regime_fit = 0.1
                            confidence -= 0.35
                        else:
                            regime_fit = 0.45
                            confidence += 0.05

                    has_earnings_today = event_calendar.has_earnings(underlying)
                    if is_event_day or has_earnings_today:
                        confidence -= self._event_day_confidence_penalty
                    else:
                        confidence += 0.10

                    confidence = min(1.0, max(0.0, confidence))

                    win_rate = self._structure_win_rate(structure)
                    if vix_level is not None:
                        if 17.0 <= vix_level <= 25.0:
                            win_rate += 0.06
                        elif 15.0 <= vix_level < 17.0:
                            win_rate += 0.04
                        elif vix_level > 30.0:
                            win_rate -= 0.05
                    if regime and regime.regime_type.value == "ranging":
                        win_rate += 0.05
                    elif regime and regime.regime_type.value == "high_vol":
                        win_rate += 0.03
                    elif regime and regime.regime_type.value in ("trending_up", "trending_down"):
                        win_rate -= 0.02
                    if has_earnings_today or is_event_day:
                        win_rate -= 0.10
                    if bias != "neutral":
                        win_rate += 0.02
                    win_rate = min(0.90, max(0.45, win_rate))

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
                        risk_reward_ratio=round(rr_ratio, 2),
                        confidence=round(confidence, 2),
                        edge_estimate_pct=round(edge_pct, 2),
                        regime_fit=round(regime_fit, 2),
                        catalyst=f"vix_{int(round(vix_level)) if vix_level is not None else 'na'}_{bias}_premium",
                        time_horizon_minutes=0,
                        expiry_time=None,
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
                            "has_earnings_today": has_earnings_today,
                            "expiry": expiry.isoformat(),
                            "hold_days": self._max_hold_days,
                            "swing_trade": True,
                        },
                    ))
                    filter_counts["candidate_built"] += 1
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
                    "options_scan_result",
                    regime=regime_type,
                    final_candidates=0,
                    filter_counts=filter_counts,
                )
                return _complete(OpportunityAssessment())

            candidate_rows.sort(key=lambda c: c.expected_value, reverse=True)
            top_candidates = candidate_rows[:3]
            avg_rr = sum(c.risk_reward_ratio for c in candidate_rows) / len(candidate_rows)
            avg_conf = sum(c.confidence for c in candidate_rows) / len(candidate_rows)
            avg_edge = sum(c.edge_estimate_pct for c in candidate_rows) / len(candidate_rows)

            self._log.info(
                "options_scan_result",
                regime=regime_type,
                final_candidates=len(top_candidates),
                filter_counts=filter_counts,
            )
            return _complete(OpportunityAssessment(
                num_candidates=len(candidate_rows),
                avg_risk_reward=round(avg_rr, 2),
                confidence=round(max(0.0, min(1.0, avg_conf)), 2),
                estimated_daily_trades=min(len(candidate_rows), self.config.max_positions),
                estimated_edge_pct=round(avg_edge, 2),
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

    def close_all_positions(self, reason: str = "") -> int:
        """Force-close all open premium trades."""
        closed = 0
        for trade_key, trade in list(self._trades.items()):
            self._close_trade(trade_key, trade, reason or "forced_close")
            if trade_key not in self._trades:
                closed += 1
        return closed

    def close_positions_for_eod(self, et_now: datetime) -> int:
        """Run one explicit EOD management pass."""
        before = len(self._trades)
        self._manage_positions(et_now, regime=None)
        return max(0, before - len(self._trades))
