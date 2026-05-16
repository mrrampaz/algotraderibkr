"""Position lifecycle for the single-stock tool.

Manages exactly ONE open option position at a time:
- Opens it via OptionExecutor at NBBO mid limit.
- Intraday: tracks premium P&L vs configured loss/target thresholds,
  underlying-stop, days-held cap, news-delta re-checks.
- EOD: decides carry vs close based on trend continuation.
- Expiry day: mandatory close before market close.

Drives no scheduling itself — the orchestrator calls
check_intraday() / eod_review() / expiry_check() on a schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Callable

import pytz
import structlog

from algotrader.data.provider import DataProvider
from algotrader.singlestock.option_executor import OptionExecutor
from algotrader.singlestock.options_picker import PickedOption
from algotrader.singlestock.pdt_guard import PDTGuard
from algotrader.singlestock.state import OpenPosition, SingleStockState
from algotrader.singlestock.thesis import Direction, TradeThesis

logger = structlog.get_logger()
ET = pytz.timezone("America/New_York")


class CloseReason(str, Enum):
    PREMIUM_TARGET = "premium_target_hit"
    PREMIUM_STOP = "premium_stop_hit"
    UNDERLYING_STOP = "underlying_stop_hit"
    MAX_HOLD_DAYS = "max_hold_days_reached"
    EOD_TREND_BROKEN = "eod_trend_broken"
    EXPIRY_DAY = "expiry_day_close"
    KILL_SWITCH = "daily_kill_switch"
    NEWS_REVERSAL = "news_reversal"
    MANUAL = "manual"


@dataclass
class PositionCheckResult:
    closed: bool
    reason: CloseReason | None
    realized_pnl: float | None
    summary: str


class PositionManager:
    def __init__(
        self,
        symbol: str,
        data_provider: DataProvider,
        option_executor: OptionExecutor,
        state: SingleStockState,
        pdt_guard: PDTGuard,
        max_hold_days: int = 3,
        premium_loss_close_pct: float = 35.0,
        premium_gain_target_pct: float = 50.0,
        enable_trailing_stop: bool = True,
        expiry_day_close_time_et: str = "15:30",
    ) -> None:
        self._symbol = symbol.upper()
        self._data = data_provider
        self._opt_exec = option_executor
        self._state = state
        self._pdt = pdt_guard
        self._max_hold_days = max_hold_days
        self._loss_pct = premium_loss_close_pct
        self._gain_pct = premium_gain_target_pct
        self._trailing = enable_trailing_stop
        self._expiry_close_hh, self._expiry_close_mm = (
            int(x) for x in expiry_day_close_time_et.split(":", 1)
        )
        self._log = logger.bind(component="singlestock_position_manager", symbol=self._symbol)

    # ── Entry ──────────────────────────────────────────────────────────────

    def open_position(
        self,
        thesis: TradeThesis,
        picked: PickedOption,
    ) -> OpenPosition | None:
        if self._state.open_position is not None:
            self._log.warning("singlestock_open_blocked_existing_position")
            return None

        contract = self._opt_exec.qualify_option(
            underlying=self._symbol,
            right=picked.right,
            strike=picked.strike,
            expiry=picked.expiry,
        )
        if contract is None:
            return None

        # Refresh NBBO mid right before submit to avoid stale price.
        live_mid = self._opt_exec.get_nbbo_mid(contract) or picked.mid
        # Pay up slightly to improve fill probability without crossing the ask.
        limit_price = round(min(live_mid * 1.01, picked.ask if picked.ask > 0 else live_mid), 2)

        filled = self._opt_exec.submit_long_option(
            contract=contract,
            qty=picked.contracts,
            limit_price=limit_price,
        )
        if filled is None:
            self._log.warning("singlestock_open_no_fill")
            return None

        underlying_at_entry = thesis.technicals.current_price if thesis.technicals else 0.0
        stop_underlying = float(thesis.stop_price or 0.0)
        target_premium = filled.avg_fill_price * (1.0 + self._gain_pct / 100.0)

        position = OpenPosition(
            con_id=filled.con_id,
            local_symbol=filled.local_symbol,
            right=filled.right,
            strike=filled.strike,
            expiry=filled.expiry,
            qty=filled.qty,
            entry_premium=filled.avg_fill_price,
            entry_time=datetime.now(ET),
            direction=thesis.direction.value,
            underlying_at_entry=underlying_at_entry,
            stop_underlying=stop_underlying,
            target_premium=target_premium,
            client_order_id=filled.client_order_id,
        )
        self._state.record_entry(position)
        self._log.info(
            "singlestock_position_opened",
            con_id=position.con_id,
            local_symbol=position.local_symbol,
            qty=position.qty,
            entry_premium=position.entry_premium,
            direction=position.direction,
            stop_underlying=stop_underlying,
            target_premium=target_premium,
        )
        return position

    # ── Intraday ───────────────────────────────────────────────────────────

    def check_intraday(self) -> PositionCheckResult:
        pos = self._state.open_position
        if pos is None:
            return PositionCheckResult(False, None, None, "no_open_position")

        days_held = (datetime.now(ET).date() - pos.entry_time.astimezone(ET).date()).days
        if days_held >= self._max_hold_days:
            return self._close(pos, CloseReason.MAX_HOLD_DAYS)

        # Underlying stop
        current_underlying = self._get_underlying_price()
        if current_underlying is not None and pos.stop_underlying > 0:
            if pos.direction == "long" and current_underlying <= pos.stop_underlying:
                return self._close(pos, CloseReason.UNDERLYING_STOP)
            if pos.direction == "short" and current_underlying >= pos.stop_underlying:
                return self._close(pos, CloseReason.UNDERLYING_STOP)

        # Premium stop / target
        current_mid = self._current_premium(pos.con_id)
        if current_mid is not None and pos.entry_premium > 0:
            pct_change = (current_mid - pos.entry_premium) / pos.entry_premium * 100.0
            if pct_change <= -self._loss_pct:
                return self._close(pos, CloseReason.PREMIUM_STOP)
            if pct_change >= self._gain_pct:
                if self._trailing:
                    # Move stop up: tighten by raising effective premium-loss
                    # threshold so subsequent moves down close us out
                    # closer to current. Track via target_premium bump.
                    new_target = pos.entry_premium * (
                        1.0 + (self._gain_pct + (pct_change - self._gain_pct) * 0.5) / 100.0
                    )
                    pos.target_premium = round(new_target, 2)
                    self._state.save()
                    self._log.info(
                        "singlestock_trailing_stop_advanced",
                        new_target_premium=pos.target_premium,
                        current_mid=current_mid,
                    )
                else:
                    return self._close(pos, CloseReason.PREMIUM_TARGET)

        # Days-held counter persisted for visibility.
        if pos.days_held != days_held:
            pos.days_held = days_held
            self._state.save()

        return PositionCheckResult(False, None, None, "ok_holding")

    # ── End of day ─────────────────────────────────────────────────────────

    def eod_review(self, trend_intact: bool) -> PositionCheckResult:
        pos = self._state.open_position
        if pos is None:
            return PositionCheckResult(False, None, None, "no_open_position")

        days_held = (datetime.now(ET).date() - pos.entry_time.astimezone(ET).date()).days
        if days_held >= self._max_hold_days:
            return self._close(pos, CloseReason.MAX_HOLD_DAYS)

        if not trend_intact:
            return self._close(pos, CloseReason.EOD_TREND_BROKEN)

        # Otherwise: carry overnight. Persist days_held bump.
        pos.days_held = days_held
        self._state.save()
        self._log.info(
            "singlestock_eod_carrying",
            days_held=days_held,
            max_hold_days=self._max_hold_days,
        )
        return PositionCheckResult(False, None, None, "carry_overnight")

    # ── Expiry day ─────────────────────────────────────────────────────────

    def expiry_day_check(self) -> PositionCheckResult:
        pos = self._state.open_position
        if pos is None:
            return PositionCheckResult(False, None, None, "no_open_position")
        try:
            expiry_date = datetime.strptime(pos.expiry, "%Y%m%d").date()
        except (ValueError, TypeError):
            self._log.warning("singlestock_expiry_unparseable", expiry=pos.expiry)
            return PositionCheckResult(False, None, None, "expiry_unparseable")

        today = datetime.now(ET).date()
        if today < expiry_date:
            return PositionCheckResult(False, None, None, "before_expiry")

        now = datetime.now(ET)
        if (now.hour, now.minute) < (self._expiry_close_hh, self._expiry_close_mm):
            return PositionCheckResult(False, None, None, "expiry_day_before_close_time")

        return self._close(pos, CloseReason.EXPIRY_DAY)

    # ── Forced close (kill switch / manual) ────────────────────────────────

    def force_close(self, reason: CloseReason) -> PositionCheckResult:
        pos = self._state.open_position
        if pos is None:
            return PositionCheckResult(False, None, None, "no_open_position")
        return self._close(pos, reason)

    # ── Internals ──────────────────────────────────────────────────────────

    def _close(self, pos: OpenPosition, reason: CloseReason) -> PositionCheckResult:
        if not self._pdt.can_close_today(pos.entry_time):
            self._log.info("singlestock_close_deferred_pdt", reason=reason.value)
            return PositionCheckResult(False, None, None, "deferred_pdt")

        contract = self._opt_exec.qualify_by_con_id(pos.con_id)
        if contract is None:
            self._log.error("singlestock_close_unqualified_contract", con_id=pos.con_id)
            return PositionCheckResult(False, None, None, "close_unqualified")

        mid = self._opt_exec.get_nbbo_mid(contract)
        if mid is None or mid <= 0:
            # Fall back to last entry premium minus a haircut for marketable limit
            mid = max(0.01, pos.entry_premium * 0.5)
        # Lower the offer slightly to improve hit rate on close.
        limit_price = max(0.01, round(mid * 0.99, 2))

        fill_price = self._opt_exec.close_long_option(
            con_id=pos.con_id,
            qty=pos.qty,
            limit_price=limit_price,
        )
        if fill_price is None:
            self._log.warning("singlestock_close_not_filled", reason=reason.value)
            return PositionCheckResult(False, None, None, "close_not_filled")

        realized = round((fill_price - pos.entry_premium) * 100.0 * pos.qty, 2)
        self._state.record_close(realized)
        self._log.info(
            "singlestock_position_closed",
            reason=reason.value,
            entry_premium=pos.entry_premium,
            exit_premium=fill_price,
            qty=pos.qty,
            realized_pnl_dollars=realized,
            days_held=pos.days_held,
        )
        return PositionCheckResult(True, reason, realized, f"closed_{reason.value}")

    def _get_underlying_price(self) -> float | None:
        try:
            snap = self._data.get_snapshot(self._symbol)
        except Exception:
            self._log.exception("singlestock_underlying_price_failed")
            return None
        if snap is None:
            return None
        last = getattr(snap, "latest_trade_price", None)
        if last is None:
            quote = getattr(snap, "latest_quote", None)
            if quote is not None:
                mid = getattr(quote, "mid_price", None)
                if mid:
                    return float(mid)
            bar = getattr(snap, "minute_bar", None) or getattr(snap, "daily_bar", None)
            if bar is not None:
                close = getattr(bar, "close", None)
                if close:
                    return float(close)
            return None
        try:
            return float(last)
        except (TypeError, ValueError):
            return None

    def _current_premium(self, con_id: int) -> float | None:
        contract = self._opt_exec.qualify_by_con_id(con_id)
        if contract is None:
            return None
        return self._opt_exec.get_nbbo_mid(contract)
