"""Single-leg long option execution path for the single-stock tool.

The main IBKRExecutor only supports stock orders (submit_order) and
multi-leg BAG spreads (submit_mleg_order). Closing helpers use market
orders. For directional swing trading on slightly-ITM weeklies we need:

- Single-leg BUY-to-open at a LIMIT price (NBBO mid).
- Single-leg SELL-to-close at a LIMIT price.
- Cancel-and-replace on stale limits.

We build the contract via ib_async directly off the shared IBKRConnection
singleton rather than extending the main executor — smaller blast radius,
and the main executor's scope-guardrails forbid edits there.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import structlog
from ib_async import Contract, LimitOrder, Option

from algotrader.execution.ibkr_connection import IBKRConnection

logger = structlog.get_logger()


@dataclass
class FilledOption:
    con_id: int
    local_symbol: str
    right: str
    strike: float
    expiry: str
    qty: int
    avg_fill_price: float
    client_order_id: str
    ib_order_id: int


def _round_price(price: float) -> float:
    return float(Decimal(str(price)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class OptionExecutor:
    def __init__(self, connection: IBKRConnection) -> None:
        self._connection = connection
        self._log = logger.bind(component="singlestock_option_executor")

    # ── Contract resolution ────────────────────────────────────────────────

    def qualify_option(
        self,
        underlying: str,
        right: str,
        strike: float,
        expiry: date | str,
    ) -> Contract | None:
        if isinstance(expiry, date):
            expiry_str = expiry.strftime("%Y%m%d")
        else:
            expiry_str = str(expiry)

        right = right.upper()[0]
        if right not in ("C", "P"):
            self._log.error("singlestock_option_bad_right", right=right)
            return None

        opt = Option(
            symbol=underlying.upper(),
            lastTradeDateOrContractMonth=expiry_str,
            strike=float(strike),
            right=right,
            exchange="SMART",
            currency="USD",
        )
        try:
            qualified = self._connection.execute(lambda ib: ib.qualifyContracts(opt))
        except Exception:
            self._log.exception(
                "singlestock_option_qualify_failed",
                underlying=underlying,
                right=right,
                strike=strike,
                expiry=expiry_str,
            )
            return None
        if not qualified or qualified[0] is None:
            self._log.warning(
                "singlestock_option_unqualified",
                underlying=underlying,
                right=right,
                strike=strike,
                expiry=expiry_str,
            )
            return None
        return qualified[0]

    def qualify_by_con_id(self, con_id: int) -> Contract | None:
        base = Contract(conId=int(con_id), exchange="SMART")
        try:
            qualified = self._connection.execute(lambda ib: ib.qualifyContracts(base))
        except Exception:
            self._log.exception("singlestock_option_qualify_conid_failed", con_id=con_id)
            return None
        if not qualified or qualified[0] is None:
            return None
        return qualified[0]

    # ── NBBO mid ───────────────────────────────────────────────────────────

    def get_nbbo_mid(self, contract: Contract) -> float | None:
        try:
            tickers = self._connection.execute(lambda ib: ib.reqTickers(contract))
        except Exception:
            self._log.exception("singlestock_option_nbbo_failed", con_id=getattr(contract, "conId", 0))
            return None
        if not tickers:
            return None
        t = tickers[0]
        bid = float(getattr(t, "bid", 0.0) or 0.0)
        ask = float(getattr(t, "ask", 0.0) or 0.0)
        if bid <= 0 or ask <= 0 or ask < bid:
            last = float(getattr(t, "last", 0.0) or 0.0)
            close = float(getattr(t, "close", 0.0) or 0.0)
            fallback = last if last > 0 else close
            return _round_price(fallback) if fallback > 0 else None
        return _round_price((bid + ask) / 2.0)

    # ── Open / close ───────────────────────────────────────────────────────

    def submit_long_option(
        self,
        contract: Contract,
        qty: int,
        limit_price: float,
        client_order_id: str | None = None,
        tif: str = "DAY",
    ) -> FilledOption | None:
        """Submit a BUY-to-open LIMIT order and wait for fill.

        Returns FilledOption on success, None on failure or unfilled.
        """
        if qty <= 0 or limit_price <= 0:
            self._log.error("singlestock_option_open_bad_args", qty=qty, limit_price=limit_price)
            return None

        client_order_id = client_order_id or f"ss_open_{uuid.uuid4().hex[:12]}"
        order = LimitOrder(
            action="BUY",
            totalQuantity=float(qty),
            lmtPrice=_round_price(limit_price),
            tif=tif,
            orderRef=client_order_id,
        )

        self._log.info(
            "singlestock_option_open_submitting",
            con_id=int(getattr(contract, "conId", 0) or 0),
            local_symbol=str(getattr(contract, "localSymbol", "") or ""),
            qty=qty,
            limit_price=order.lmtPrice,
            client_order_id=client_order_id,
        )

        try:
            trade = self._connection.execute(lambda ib: ib.placeOrder(contract, order))
        except Exception:
            self._log.exception("singlestock_option_open_submit_failed")
            return None

        filled = self._await_fill(trade, timeout_seconds=60)
        if filled is None:
            self._log.warning(
                "singlestock_option_open_not_filled",
                client_order_id=client_order_id,
                limit_price=order.lmtPrice,
            )
            return None

        return FilledOption(
            con_id=int(getattr(contract, "conId", 0) or 0),
            local_symbol=str(getattr(contract, "localSymbol", "") or ""),
            right=str(getattr(contract, "right", "") or "").upper(),
            strike=float(getattr(contract, "strike", 0.0) or 0.0),
            expiry=str(getattr(contract, "lastTradeDateOrContractMonth", "") or ""),
            qty=qty,
            avg_fill_price=filled,
            client_order_id=client_order_id,
            ib_order_id=int(trade.order.orderId),
        )

    def close_long_option(
        self,
        con_id: int,
        qty: int,
        limit_price: float,
        client_order_id: str | None = None,
        tif: str = "DAY",
    ) -> float | None:
        """Submit a SELL-to-close LIMIT order and wait for fill.

        Returns the average fill price on success, None on failure.
        """
        if con_id <= 0 or qty <= 0 or limit_price <= 0:
            self._log.error(
                "singlestock_option_close_bad_args",
                con_id=con_id,
                qty=qty,
                limit_price=limit_price,
            )
            return None

        contract = self.qualify_by_con_id(con_id)
        if contract is None:
            return None

        client_order_id = client_order_id or f"ss_close_{uuid.uuid4().hex[:12]}"
        order = LimitOrder(
            action="SELL",
            totalQuantity=float(qty),
            lmtPrice=_round_price(limit_price),
            tif=tif,
            orderRef=client_order_id,
        )

        self._log.info(
            "singlestock_option_close_submitting",
            con_id=con_id,
            qty=qty,
            limit_price=order.lmtPrice,
            client_order_id=client_order_id,
        )

        try:
            trade = self._connection.execute(lambda ib: ib.placeOrder(contract, order))
        except Exception:
            self._log.exception("singlestock_option_close_submit_failed")
            return None

        filled = self._await_fill(trade, timeout_seconds=120)
        if filled is None:
            self._log.warning(
                "singlestock_option_close_not_filled",
                client_order_id=client_order_id,
                limit_price=order.lmtPrice,
            )
            return None
        return filled

    # ── Fill waiter ────────────────────────────────────────────────────────

    def _await_fill(self, trade: Any, timeout_seconds: int) -> float | None:
        """Poll trade status until terminal or timeout."""
        deadline = time.time() + timeout_seconds
        ib = self._connection.ib
        while time.time() < deadline:
            ib.sleep(0.5)
            status = str(getattr(trade.orderStatus, "status", "") or "").lower()
            if status == "filled":
                avg = float(getattr(trade.orderStatus, "avgFillPrice", 0.0) or 0.0)
                if avg <= 0:
                    fills = trade.fills or []
                    if fills:
                        total_qty = sum(float(f.execution.shares) for f in fills)
                        total_dollars = sum(
                            float(f.execution.shares) * float(f.execution.price)
                            for f in fills
                        )
                        if total_qty > 0:
                            avg = total_dollars / total_qty
                return avg if avg > 0 else None
            if status in ("cancelled", "apicancelled", "inactive", "rejected"):
                return None
        return None
