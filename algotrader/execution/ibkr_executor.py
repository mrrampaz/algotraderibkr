"""IBKR Executor implementation."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import pytz
import structlog
from ib_async import (
    ComboLeg,
    Contract,
    LimitOrder,
    MarketOrder,
    Option,
    StopLimitOrder,
    StopOrder,
    Stock,
)

from algotrader.core.models import (
    AccountInfo,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from algotrader.execution.ibkr_connection import IBKRConnection

logger = structlog.get_logger()

_TIF_TO_IB = {
    TimeInForce.DAY: "DAY",
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
}

_TIF_FROM_IB = {
    "DAY": TimeInForce.DAY,
    "GTC": TimeInForce.GTC,
    "IOC": TimeInForce.IOC,
    "FOK": TimeInForce.FOK,
    "OPG": TimeInForce.OPG,
    "CLS": TimeInForce.CLS,
}

_STATUS_MAP = {
    "pendingsubmit": OrderStatus.PENDING,
    "presubmitted": OrderStatus.SUBMITTED,
    "submitted": OrderStatus.SUBMITTED,
    "pendingcancel": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "apicancelled": OrderStatus.CANCELLED,
    "filled": OrderStatus.FILLED,
    "inactive": OrderStatus.REJECTED,
}


class IBKRExecutor:
    """Executor implementation for Interactive Brokers."""

    def __init__(self, connection: IBKRConnection) -> None:
        self._connection = connection
        self._config = connection.config
        self._order_id_map: dict[str, int] = {}
        self._trade_map: dict[int, Any] = {}
        self._stock_contract_cache: dict[str, Any] = {}
        self._option_contract_map: dict[str, Any] = {}
        self._log = logger.bind(
            component="ibkr_executor",
            host=self._config.host,
            port=self._config.port,
            client_id=self._config.client_id,
            readonly=self._config.readonly,
        )
        self._log.info("ibkr_executor_initialized")

    @staticmethod
    def _round_price(price: float) -> float:
        """Round price to valid tick using Decimal to avoid float precision issues."""
        return float(Decimal(str(price)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_optional_float(value) -> float | None:
        try:
            if value is None:
                return None
            out = float(value)
            if out == -1:
                return None
            return out
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _local_order_id(ib_order_id: int) -> str:
        return f"ibkr_{ib_order_id}"

    def _ib_order_id(self, order_id: str) -> int | None:
        if order_id in self._order_id_map:
            return self._order_id_map[order_id]

        if order_id.startswith("ibkr_"):
            raw = order_id.replace("ibkr_", "", 1)
            if raw.isdigit():
                return int(raw)

        if order_id.isdigit():
            return int(order_id)

        return None

    def _register_trade(self, trade) -> None:
        order = getattr(trade, "order", None)
        if order is None:
            return
        ib_order_id = int(getattr(order, "orderId", 0) or 0)
        if ib_order_id <= 0:
            return
        local_id = self._local_order_id(ib_order_id)
        self._order_id_map[local_id] = ib_order_id
        self._trade_map[ib_order_id] = trade

    def _qualify_stock_contract(self, symbol: str):
        symbol = symbol.upper()
        cached = self._stock_contract_cache.get(symbol)
        if cached is not None:
            return cached

        def _qualify(ib):
            contract = Stock(symbol, "SMART", "USD")
            contracts = ib.qualifyContracts(contract)
            return contracts[0] if contracts else None

        contract = self._connection.execute(_qualify)
        if contract is None:
            self._log.warning("ibkr_contract_qualification_failed", symbol=symbol)
            return None
        self._stock_contract_cache[symbol] = contract
        return contract

    def _find_trade(self, ib_order_id: int):
        trade = self._trade_map.get(ib_order_id)
        if trade is not None:
            return trade

        def _find(ib):
            for candidate in ib.trades():
                order = getattr(candidate, "order", None)
                if order and int(getattr(order, "orderId", 0) or 0) == ib_order_id:
                    return candidate
            for candidate in ib.openTrades():
                order = getattr(candidate, "order", None)
                if order and int(getattr(order, "orderId", 0) or 0) == ib_order_id:
                    return candidate
            return None

        trade = self._connection.execute(_find)
        if trade is not None:
            self._register_trade(trade)
        return trade

    def _convert_order(
        self,
        trade,
        *,
        stop_order_id: str = "",
        tp_order_id: str = "",
        is_bracket: bool = False,
    ) -> Order:
        order_obj = getattr(trade, "order", None)
        order_status = getattr(trade, "orderStatus", None)
        contract = getattr(trade, "contract", None)

        ib_order_id = int(getattr(order_obj, "orderId", 0) or 0)
        local_id = self._local_order_id(ib_order_id) if ib_order_id > 0 else ""
        if ib_order_id > 0:
            self._order_id_map[local_id] = ib_order_id

        status_raw = str(getattr(order_status, "status", "")).lower()
        ib_order_type = str(getattr(order_obj, "orderType", "MKT")).upper()
        tif = str(getattr(order_obj, "tif", "DAY")).upper()

        if ib_order_type == "LMT":
            order_type = OrderType.LIMIT
        elif ib_order_type == "STP":
            order_type = OrderType.STOP
        elif ib_order_type == "STP LMT":
            order_type = OrderType.STOP_LIMIT
        elif ib_order_type == "TRAIL":
            order_type = OrderType.TRAILING_STOP
        else:
            order_type = OrderType.MARKET

        side = OrderSide.BUY if str(getattr(order_obj, "action", "BUY")).upper() == "BUY" else OrderSide.SELL
        symbol = str(getattr(contract, "symbol", "") or "")

        submitted_at = datetime.now(pytz.UTC)
        filled_at = submitted_at if _STATUS_MAP.get(status_raw) == OrderStatus.FILLED else None

        return Order(
            id=local_id,
            client_order_id=str(getattr(order_obj, "orderRef", "") or ""),
            symbol=symbol,
            side=side,
            order_type=order_type,
            time_in_force=_TIF_FROM_IB.get(tif, TimeInForce.DAY),
            qty=self._safe_float(getattr(order_obj, "totalQuantity", 0.0)),
            limit_price=self._safe_optional_float(getattr(order_obj, "lmtPrice", None)),
            stop_price=self._safe_optional_float(getattr(order_obj, "auxPrice", None)),
            status=_STATUS_MAP.get(status_raw, OrderStatus.PENDING),
            filled_qty=self._safe_float(getattr(order_status, "filled", 0.0)),
            filled_avg_price=self._safe_optional_float(getattr(order_status, "avgFillPrice", None)),
            submitted_at=submitted_at,
            filled_at=filled_at,
            stop_order_id=stop_order_id,
            tp_order_id=tp_order_id,
            is_bracket=is_bracket,
        )

    def _require_writable(self, action: str) -> bool:
        if self._config.readonly:
            self._log.error("ibkr_readonly_mode_blocked", action=action)
            return False
        return True

    def _build_order(
        self,
        *,
        action: str,
        qty: float,
        order_type: OrderType,
        time_in_force: TimeInForce,
        limit_price: float | None,
        stop_price: float | None,
        client_order_id: str,
    ):
        tif = _TIF_TO_IB.get(time_in_force, "DAY")

        if order_type == OrderType.MARKET:
            return MarketOrder(
                action=action,
                totalQuantity=qty,
                tif=tif,
                orderRef=client_order_id,
            )
        if order_type == OrderType.LIMIT:
            if limit_price is None:
                return None
            return LimitOrder(
                action=action,
                totalQuantity=qty,
                lmtPrice=self._round_price(limit_price),
                tif=tif,
                orderRef=client_order_id,
            )
        if order_type == OrderType.STOP:
            if stop_price is None:
                return None
            return StopOrder(
                action=action,
                totalQuantity=qty,
                stopPrice=self._round_price(stop_price),
                tif=tif,
                orderRef=client_order_id,
            )
        if order_type == OrderType.STOP_LIMIT:
            if stop_price is None or limit_price is None:
                return None
            return StopLimitOrder(
                action=action,
                totalQuantity=qty,
                lmtPrice=self._round_price(limit_price),
                stopPrice=self._round_price(stop_price),
                tif=tif,
                orderRef=client_order_id,
            )
        # TRAILING_STOP is not currently used in strategies; keep explicit fallback.
        return None

    def _reference_price(self, contract, side: OrderSide) -> float | None:
        ticker = self._connection.execute(lambda ib: (ib.reqTickers(contract) or [None])[0])
        if ticker is None:
            return None

        if side == OrderSide.BUY:
            candidates = [
                getattr(ticker, "ask", None),
                getattr(ticker, "last", None),
                getattr(ticker, "close", None),
                getattr(ticker, "bid", None),
            ]
        else:
            candidates = [
                getattr(ticker, "bid", None),
                getattr(ticker, "last", None),
                getattr(ticker, "close", None),
                getattr(ticker, "ask", None),
            ]

        for c in candidates:
            px = self._safe_optional_float(c)
            if px is not None and px > 0:
                return px
        return None

    def _submit_bracket_order(
        self,
        *,
        contract,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType,
        time_in_force: TimeInForce,
        limit_price: float | None,
        client_order_id: str,
        bracket_stop_price: float,
        bracket_take_profit_price: float | None,
    ) -> Order | None:
        action = "BUY" if side == OrderSide.BUY else "SELL"
        opposite_action = "SELL" if side == OrderSide.BUY else "BUY"
        tif = _TIF_TO_IB.get(time_in_force, "DAY")

        if order_type == OrderType.LIMIT:
            if limit_price is None:
                self._log.error("ibkr_limit_order_missing_price", symbol=symbol)
                return None
            entry_limit = self._round_price(limit_price)
        else:
            ref_price = self._reference_price(contract, side)
            if ref_price is None:
                self._log.error("ibkr_bracket_reference_price_missing", symbol=symbol)
                return None
            # Marketable limit as parent entry for bracket support.
            marketable = ref_price * (1.002 if side == OrderSide.BUY else 0.998)
            entry_limit = self._round_price(marketable)

        stop_px = self._round_price(bracket_stop_price)
        if bracket_take_profit_price is not None:
            tp_px = self._round_price(bracket_take_profit_price)

            def _submit_native(ib):
                parent, take_profit, stop_loss = ib.bracketOrder(
                    action=action,
                    quantity=qty,
                    limitPrice=entry_limit,
                    takeProfitPrice=tp_px,
                    stopLossPrice=stop_px,
                )

                parent.tif = tif
                parent.orderRef = client_order_id
                take_profit.tif = tif
                take_profit.orderRef = f"{client_order_id}:tp"
                stop_loss.tif = tif
                stop_loss.orderRef = f"{client_order_id}:sl"

                t_parent = ib.placeOrder(contract, parent)
                t_tp = ib.placeOrder(contract, take_profit)
                t_sl = ib.placeOrder(contract, stop_loss)
                return t_parent, t_tp, t_sl

            try:
                t_parent, t_tp, t_sl = self._connection.execute(_submit_native)
                self._register_trade(t_parent)
                self._register_trade(t_tp)
                self._register_trade(t_sl)

                parent_id = self._local_order_id(int(t_parent.order.orderId))
                tp_id = self._local_order_id(int(t_tp.order.orderId))
                sl_id = self._local_order_id(int(t_sl.order.orderId))

                converted = self._convert_order(
                    t_parent,
                    stop_order_id=sl_id,
                    tp_order_id=tp_id,
                    is_bracket=True,
                )
                self._log.info(
                    "ibkr_bracket_order_submitted",
                    symbol=symbol,
                    order_id=parent_id,
                    stop_order_id=sl_id,
                    tp_order_id=tp_id,
                )
                return converted
            except Exception:
                self._log.exception("ibkr_bracket_order_failed", symbol=symbol)
                return None

        # Stop-only bracket: place parent then child stop order.
        parent = self._build_order(
            action=action,
            qty=qty,
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=entry_limit if order_type == OrderType.LIMIT else None,
            stop_price=None,
            client_order_id=client_order_id,
        )
        if parent is None:
            # Market parent fallback when using marketable entry.
            parent = LimitOrder(
                action=action,
                totalQuantity=qty,
                lmtPrice=entry_limit,
                tif=tif,
                orderRef=client_order_id,
            )

        try:
            parent_trade = self._connection.execute(lambda ib: ib.placeOrder(contract, parent))
            self._register_trade(parent_trade)
            parent_ib_id = int(parent_trade.order.orderId)

            stop_child = StopOrder(
                action=opposite_action,
                totalQuantity=qty,
                stopPrice=stop_px,
                tif=tif,
                parentId=parent_ib_id,
                transmit=True,
                orderRef=f"{client_order_id}:sl",
            )
            stop_trade = self._connection.execute(lambda ib: ib.placeOrder(contract, stop_child))
            self._register_trade(stop_trade)

            stop_id = self._local_order_id(int(stop_trade.order.orderId))
            converted = self._convert_order(
                parent_trade,
                stop_order_id=stop_id,
                is_bracket=True,
            )
            self._log.info(
                "ibkr_stop_bracket_submitted",
                symbol=symbol,
                order_id=converted.id,
                stop_order_id=stop_id,
            )
            return converted
        except Exception:
            self._log.exception("ibkr_stop_bracket_failed", symbol=symbol)
            return None

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: float | None = None,
        stop_price: float | None = None,
        client_order_id: str | None = None,
        bracket_stop_price: float | None = None,
        bracket_take_profit_price: float | None = None,
    ) -> Order | None:
        if not self._require_writable("submit_order"):
            return None

        symbol = symbol.upper()
        client_order_id = client_order_id or str(uuid.uuid4())
        action = "BUY" if side == OrderSide.BUY else "SELL"

        contract = self._qualify_stock_contract(symbol)
        if contract is None:
            return None

        if bracket_stop_price is not None and order_type in (OrderType.MARKET, OrderType.LIMIT):
            return self._submit_bracket_order(
                contract=contract,
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                client_order_id=client_order_id,
                bracket_stop_price=bracket_stop_price,
                bracket_take_profit_price=bracket_take_profit_price,
            )

        order = self._build_order(
            action=action,
            qty=qty,
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            client_order_id=client_order_id,
        )
        if order is None:
            self._log.error(
                "ibkr_order_build_failed",
                symbol=symbol,
                order_type=order_type.value,
            )
            return None

        try:
            trade = self._connection.execute(lambda ib: ib.placeOrder(contract, order))
            self._register_trade(trade)
            converted = self._convert_order(trade)
            self._log.info(
                "ibkr_order_submitted",
                symbol=symbol,
                order_id=converted.id,
                side=side.value,
                order_type=order_type.value,
                qty=qty,
            )
            return converted
        except Exception:
            self._log.exception(
                "ibkr_order_submit_failed",
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
            )
            return None

    def cancel_order(self, order_id: str) -> bool:
        if not self._require_writable("cancel_order"):
            return False

        ib_order_id = self._ib_order_id(order_id)
        if ib_order_id is None:
            self._log.warning("ibkr_cancel_order_not_found", order_id=order_id)
            return True

        trade = self._find_trade(ib_order_id)
        if trade is None:
            self._log.info("ibkr_cancel_order_already_closed", order_id=order_id)
            return True

        try:
            self._connection.execute(lambda ib: ib.cancelOrder(trade.order))
            self._log.info("ibkr_order_cancelled", order_id=order_id)
            return True
        except Exception:
            self._log.exception("ibkr_cancel_order_failed", order_id=order_id)
            return False

    def close_position(self, symbol: str) -> bool:
        if not self._require_writable("close_position"):
            return False

        symbol = symbol.upper()
        pos = self.get_position(symbol)
        if pos is None:
            self._log.info("ibkr_ghost_position_closed", symbol=symbol)
            return True

        close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
        order = self.submit_order(
            symbol=symbol,
            qty=abs(pos.qty),
            side=close_side,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"close_{symbol}_{uuid.uuid4()}",
        )
        return order is not None

    def get_position(self, symbol: str) -> Position | None:
        symbol = symbol.upper()
        for pos in self.get_positions():
            if pos.symbol.upper() == symbol:
                return pos
        return None

    def get_positions(self) -> list[Position]:
        try:
            portfolio = self._connection.execute(lambda ib: ib.portfolio())
        except Exception:
            self._log.exception("ibkr_get_positions_failed")
            return []

        out: list[Position] = []
        for item in portfolio or []:
            contract = getattr(item, "contract", None)
            if contract is None:
                continue
            if str(getattr(contract, "secType", "")).upper() != "STK":
                continue

            raw_qty = self._safe_float(getattr(item, "position", 0.0))
            if raw_qty == 0:
                continue

            qty = abs(raw_qty)
            side = OrderSide.BUY if raw_qty > 0 else OrderSide.SELL
            avg_entry = abs(self._safe_float(getattr(item, "averageCost", 0.0)))
            current_price = self._safe_float(getattr(item, "marketPrice", 0.0))
            market_value = self._safe_float(getattr(item, "marketValue", 0.0))
            unrealized = self._safe_float(getattr(item, "unrealizedPNL", 0.0))
            basis = avg_entry * qty
            unrealized_pct = (unrealized / basis) if basis > 0 else 0.0

            out.append(
                Position(
                    symbol=str(getattr(contract, "symbol", "")),
                    qty=qty,
                    side=side,
                    avg_entry_price=avg_entry,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized,
                    unrealized_pnl_pct=unrealized_pct,
                )
            )
        return out

    def get_account(self) -> AccountInfo:
        summary = self._connection.execute(lambda ib: ib.accountSummary(self._config.account or ""))

        def get_value(*tags: str) -> float:
            for tag in tags:
                for row in summary or []:
                    if str(getattr(row, "tag", "")).lower() == tag.lower():
                        try:
                            return float(getattr(row, "value", 0.0))
                        except (TypeError, ValueError):
                            continue
            return 0.0

        equity = get_value("NetLiquidation")
        cash = get_value("TotalCashValue")
        buying_power = get_value("BuyingPower", "AvailableFunds")
        gross_position = get_value("GrossPositionValue")
        portfolio_value = gross_position + cash if gross_position > 0 else equity

        return AccountInfo(
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            portfolio_value=portfolio_value,
            day_trade_count=0,
            pattern_day_trader=False,
        )

    def get_order(self, order_id: str) -> Order | None:
        ib_order_id = self._ib_order_id(order_id)
        if ib_order_id is None:
            return None
        trade = self._find_trade(ib_order_id)
        if trade is None:
            return None
        return self._convert_order(trade)

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        symbol = symbol.upper() if symbol else None
        try:
            open_trades = self._connection.execute(lambda ib: ib.openTrades())
        except Exception:
            self._log.exception("ibkr_get_open_orders_failed")
            return []

        orders: list[Order] = []
        for trade in open_trades or []:
            self._register_trade(trade)
            converted = self._convert_order(trade)
            if symbol and converted.symbol.upper() != symbol:
                continue
            orders.append(converted)
        return orders

    def replace_stop_order(
        self,
        order_id: str,
        new_stop_price: float,
    ) -> Order | None:
        if not self._require_writable("replace_stop_order"):
            return None

        ib_order_id = self._ib_order_id(order_id)
        if ib_order_id is None:
            return None

        trade = self._find_trade(ib_order_id)
        if trade is None:
            self._log.warning("ibkr_replace_stop_order_missing_trade", order_id=order_id)
            return None

        order = trade.order
        order_type = str(getattr(order, "orderType", "")).upper()
        if order_type not in ("STP", "STP LMT"):
            self._log.warning(
                "ibkr_replace_stop_order_wrong_type",
                order_id=order_id,
                order_type=order_type,
            )
            return None

        order.auxPrice = self._round_price(new_stop_price)

        try:
            updated_trade = self._connection.execute(lambda ib: ib.placeOrder(trade.contract, order))
            self._register_trade(updated_trade)
            updated = self._convert_order(updated_trade)
            self._log.info("ibkr_stop_order_replaced", order_id=order_id, new_stop_price=order.auxPrice)
            return updated
        except Exception:
            self._log.exception("ibkr_replace_stop_order_failed", order_id=order_id)
            return None

    def lookup_option_contract(
        self,
        underlying: str,
        contract_type: str,
        strike: float,
        expiration: date,
    ) -> str | None:
        right = "C" if contract_type.lower().startswith("c") else "P"
        expiration_str = expiration.strftime("%Y%m%d")
        underlying = underlying.upper()

        option = Option(
            symbol=underlying,
            lastTradeDateOrContractMonth=expiration_str,
            strike=float(strike),
            right=right,
            exchange="SMART",
            currency="USD",
        )

        try:
            qualified = self._connection.execute(lambda ib: ib.qualifyContracts(option))
        except Exception:
            self._log.exception(
                "ibkr_lookup_option_contract_failed",
                underlying=underlying,
                contract_type=contract_type,
                strike=strike,
                expiration=str(expiration),
            )
            return None

        if not qualified:
            self._log.warning(
                "ibkr_option_contract_not_found",
                underlying=underlying,
                contract_type=contract_type,
                strike=strike,
                expiration=str(expiration),
            )
            return None

        contract = qualified[0]
        identifier = str(getattr(contract, "localSymbol", "") or f"conId:{contract.conId}")
        self._option_contract_map[identifier] = contract
        self._option_contract_map[f"conId:{contract.conId}"] = contract
        return identifier

    def _resolve_option_contract(self, symbol: str):
        contract = self._option_contract_map.get(symbol)
        if contract is not None:
            return contract

        if symbol.startswith("conId:"):
            raw = symbol.replace("conId:", "", 1)
            if raw.isdigit():
                con_id = int(raw)
                contract = Contract(conId=con_id, exchange="SMART")
                qualified = self._connection.execute(lambda ib: ib.qualifyContracts(contract))
                if qualified:
                    resolved = qualified[0]
                    key = str(getattr(resolved, "localSymbol", "") or f"conId:{resolved.conId}")
                    self._option_contract_map[key] = resolved
                    self._option_contract_map[f"conId:{resolved.conId}"] = resolved
                    return resolved
        return None

    def submit_mleg_order(
        self,
        legs: list[dict],
        qty: int = 1,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
    ) -> str | None:
        if not self._require_writable("submit_mleg_order"):
            return None
        if not legs:
            return None

        client_order_id = client_order_id or str(uuid.uuid4())
        tif = _TIF_TO_IB.get(time_in_force, "DAY")

        resolved_contracts = []
        for leg in legs:
            leg_symbol = str(leg.get("symbol", ""))
            contract = self._resolve_option_contract(leg_symbol)
            if contract is None:
                self._log.error("ibkr_mleg_leg_unresolved", symbol=leg_symbol)
                return None
            resolved_contracts.append((leg, contract))

        combo_legs = []
        for leg, contract in resolved_contracts:
            side = leg.get("side", OrderSide.BUY)
            action = "BUY" if side == OrderSide.BUY else "SELL"
            ratio = int(leg.get("ratio_qty", 1))
            combo_legs.append(
                ComboLeg(
                    conId=int(contract.conId),
                    ratio=ratio,
                    action=action,
                    exchange="SMART",
                )
            )

        first_contract = resolved_contracts[0][1]
        bag = Contract(
            symbol=str(getattr(first_contract, "symbol", "")),
            secType="BAG",
            currency="USD",
            exchange="SMART",
            comboLegs=combo_legs,
        )

        combo_action = "BUY" if legs[0].get("side") == OrderSide.BUY else "SELL"
        order = MarketOrder(
            action=combo_action,
            totalQuantity=float(qty),
            tif=tif,
            orderRef=client_order_id,
        )

        try:
            trade = self._connection.execute(lambda ib: ib.placeOrder(bag, order))
            self._register_trade(trade)
            local_id = self._local_order_id(int(trade.order.orderId))
            self._log.info(
                "ibkr_mleg_order_submitted",
                order_id=local_id,
                qty=qty,
                num_legs=len(legs),
            )
            return local_id
        except Exception:
            self._log.exception("ibkr_mleg_submit_failed")
            return None
