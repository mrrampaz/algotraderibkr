"""Alpaca order execution implementation."""

from __future__ import annotations

import uuid

import structlog
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    TrailingStopOrderRequest,
    GetOrdersRequest,
    ReplaceOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import (
    OrderSide as AlpacaOrderSide,
    OrderType as AlpacaOrderType,
    TimeInForce as AlpacaTimeInForce,
    QueryOrderStatus,
    OrderClass,
)
from alpaca.common.exceptions import APIError

from algotrader.core.config import AlpacaConfig
from algotrader.core.models import (
    Order, OrderSide, OrderType, OrderStatus, TimeInForce,
    Position, AccountInfo,
)

logger = structlog.get_logger()

# Map our enums to Alpaca enums
_SIDE_MAP = {
    OrderSide.BUY: AlpacaOrderSide.BUY,
    OrderSide.SELL: AlpacaOrderSide.SELL,
}

_TIF_MAP = {
    TimeInForce.DAY: AlpacaTimeInForce.DAY,
    TimeInForce.GTC: AlpacaTimeInForce.GTC,
    TimeInForce.IOC: AlpacaTimeInForce.IOC,
    TimeInForce.FOK: AlpacaTimeInForce.FOK,
    TimeInForce.OPG: AlpacaTimeInForce.OPG,
    TimeInForce.CLS: AlpacaTimeInForce.CLS,
}

_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "pending_cancel": OrderStatus.CANCELLED,
    "pending_replace": OrderStatus.SUBMITTED,
    "replaced": OrderStatus.SUBMITTED,
    "done_for_day": OrderStatus.CANCELLED,
}


class AlpacaExecutor:
    """Executor implementation for Alpaca broker.

    Includes:
    - Idempotent orders via client_order_id
    - Wash trade prevention (cancel conflicting orders before entry)
    - Marketable limit orders
    - Ghost position handling (close returns True if position not found)
    - Bracket orders with broker-side stop loss protection
    """

    def __init__(self, config: AlpacaConfig) -> None:
        self._config = config
        self._log = logger.bind(component="alpaca_executor", paper=config.paper)
        self._client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )
        self._log.info("alpaca_executor_initialized")

    def _convert_order(self, alpaca_order) -> Order:
        """Convert an Alpaca order object to our Order model."""
        status_str = str(alpaca_order.status.value).lower() if hasattr(alpaca_order.status, 'value') else str(alpaca_order.status).lower()
        return Order(
            id=str(alpaca_order.id),
            client_order_id=str(alpaca_order.client_order_id or ""),
            symbol=alpaca_order.symbol,
            side=OrderSide.BUY if str(alpaca_order.side.value).lower() == "buy" else OrderSide.SELL,
            order_type=OrderType(str(alpaca_order.order_type.value).lower()) if hasattr(alpaca_order.order_type, 'value') else OrderType.MARKET,
            status=_STATUS_MAP.get(status_str, OrderStatus.PENDING),
            qty=float(alpaca_order.qty or 0),
            limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
            stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
            filled_qty=float(alpaca_order.filled_qty or 0),
            filled_avg_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
            submitted_at=alpaca_order.submitted_at,
            filled_at=alpaca_order.filled_at,
        )

    def _cancel_conflicting_orders(self, symbol: str, side: OrderSide) -> None:
        """Cancel open orders on the opposite side to prevent wash trade rejections."""
        opposite_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        open_orders = self.get_open_orders(symbol)

        for order in open_orders:
            if order.side == opposite_side:
                self._log.info(
                    "cancelling_conflicting_order",
                    symbol=symbol,
                    order_id=order.id,
                    conflicting_side=opposite_side.value,
                )
                self.cancel_order(order.id)

    def _build_bracket_kwargs(
        self,
        bracket_stop_price: float | None,
        bracket_take_profit_price: float | None,
    ) -> dict:
        """Build bracket order kwargs if stop/target prices are provided."""
        if bracket_stop_price is None:
            return {}

        kwargs: dict = {
            "order_class": OrderClass.BRACKET,
            "stop_loss": StopLossRequest(stop_price=bracket_stop_price),
        }
        if bracket_take_profit_price is not None:
            kwargs["take_profit"] = TakeProfitRequest(
                limit_price=bracket_take_profit_price,
            )
        return kwargs

    def _extract_bracket_legs(self, alpaca_order, order: Order) -> None:
        """Extract child order IDs from bracket order legs."""
        legs = getattr(alpaca_order, "legs", None)
        if not legs:
            return

        order.is_bracket = True
        for leg in legs:
            leg_type = str(getattr(leg, "order_type", "")).lower()
            if hasattr(leg_type, "value"):
                leg_type = leg_type.value.lower()
            leg_id = str(leg.id)

            if "stop" in leg_type:
                order.stop_order_id = leg_id
            elif "limit" in leg_type:
                order.tp_order_id = leg_id

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
        """Submit an order to Alpaca.

        When bracket_stop_price is provided for MARKET or LIMIT orders,
        a bracket order is created with broker-side stop loss protection.
        """
        # Generate idempotent client_order_id if not provided
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())

        # Cancel conflicting orders to prevent wash trade rejections
        self._cancel_conflicting_orders(symbol, side)

        use_bracket = bracket_stop_price is not None
        log = self._log.bind(
            symbol=symbol, qty=qty, side=side.value,
            order_type=order_type.value, client_order_id=client_order_id,
            bracket=use_bracket,
        )

        try:
            alpaca_side = _SIDE_MAP[side]
            alpaca_tif = _TIF_MAP[time_in_force]
            bracket_kwargs = self._build_bracket_kwargs(
                bracket_stop_price, bracket_take_profit_price,
            )

            if order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    client_order_id=client_order_id,
                    **bracket_kwargs,
                )
            elif order_type == OrderType.LIMIT:
                if limit_price is None:
                    log.error("limit_order_missing_price")
                    return None
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    limit_price=limit_price,
                    client_order_id=client_order_id,
                    **bracket_kwargs,
                )
            elif order_type == OrderType.STOP:
                if stop_price is None:
                    log.error("stop_order_missing_price")
                    return None
                request = StopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    stop_price=stop_price,
                    client_order_id=client_order_id,
                )
            elif order_type == OrderType.STOP_LIMIT:
                if stop_price is None or limit_price is None:
                    log.error("stop_limit_order_missing_prices")
                    return None
                request = StopLimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    stop_price=stop_price,
                    limit_price=limit_price,
                    client_order_id=client_order_id,
                )
            elif order_type == OrderType.TRAILING_STOP:
                request = TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    trail_percent=stop_price,  # Use stop_price field for trail %
                    client_order_id=client_order_id,
                )
            else:
                log.error("unsupported_order_type", order_type=order_type.value)
                return None

            result = self._client.submit_order(request)
            order = self._convert_order(result)

            # Extract bracket child order IDs
            if use_bracket:
                self._extract_bracket_legs(result, order)

            log.info(
                "order_submitted",
                order_id=order.id,
                status=order.status.value,
                is_bracket=order.is_bracket,
                stop_order_id=order.stop_order_id or None,
                tp_order_id=order.tp_order_id or None,
            )
            return order

        except APIError as e:
            # If bracket order fails, fall back to simple order
            if use_bracket:
                log.warning(
                    "bracket_order_failed_falling_back",
                    error=str(e),
                    bracket_stop=bracket_stop_price,
                )
                return self.submit_order(
                    symbol=symbol, qty=qty, side=side,
                    order_type=order_type, time_in_force=time_in_force,
                    limit_price=limit_price, stop_price=stop_price,
                    client_order_id=f"{client_order_id}_fb" if client_order_id else None,
                )
            log.error("order_submit_failed", error=str(e))
            return None
        except Exception:
            log.exception("order_submit_error")
            return None

    def replace_stop_order(
        self,
        order_id: str,
        new_stop_price: float,
    ) -> Order | None:
        """Replace a bracket child stop order with a new stop price.

        Used for trailing stop updates. Returns updated Order with
        the new order ID, or None if the replace failed.
        """
        log = self._log.bind(order_id=order_id, new_stop_price=new_stop_price)
        try:
            result = self._client.replace_order_by_id(
                order_id=order_id,
                order_data=ReplaceOrderRequest(stop_price=new_stop_price),
            )
            order = self._convert_order(result)
            log.info("stop_order_replaced", new_order_id=order.id)
            return order
        except APIError as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                log.info("stop_order_already_triggered")
            else:
                log.error("stop_replace_failed", error=str(e))
            return None
        except Exception:
            log.exception("stop_replace_error")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            self._client.cancel_order_by_id(order_id)
            self._log.info("order_cancelled", order_id=order_id)
            return True
        except APIError as e:
            if "not found" in str(e).lower() or "not cancelable" in str(e).lower():
                self._log.info("order_already_gone", order_id=order_id)
                return True
            self._log.error("cancel_order_failed", order_id=order_id, error=str(e))
            return False
        except Exception:
            self._log.exception("cancel_order_error", order_id=order_id)
            return False

    def close_position(self, symbol: str) -> bool:
        """Close entire position. Returns True even if position not found (ghost handling)."""
        try:
            self._client.close_position(symbol)
            self._log.info("position_closed", symbol=symbol)
            return True
        except APIError as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "no position" in error_msg:
                # Ghost position — already closed at broker
                self._log.info("ghost_position_closed", symbol=symbol)
                return True
            self._log.error("close_position_failed", symbol=symbol, error=str(e))
            return False
        except Exception:
            self._log.exception("close_position_error", symbol=symbol)
            return False

    def get_position(self, symbol: str) -> Position | None:
        """Get a single position by symbol."""
        try:
            pos = self._client.get_open_position(symbol)
            return Position(
                symbol=pos.symbol,
                qty=abs(float(pos.qty)),
                side=OrderSide.BUY if float(pos.qty) > 0 else OrderSide.SELL,
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_pct=float(pos.unrealized_plpc),
            )
        except APIError as e:
            if "not found" in str(e).lower():
                return None
            self._log.error("get_position_failed", symbol=symbol, error=str(e))
            return None
        except Exception:
            self._log.exception("get_position_error", symbol=symbol)
            return None

    def get_positions(self) -> list[Position]:
        """Get all open positions from broker."""
        try:
            positions = self._client.get_all_positions()
            return [
                Position(
                    symbol=pos.symbol,
                    qty=abs(float(pos.qty)),
                    side=OrderSide.BUY if float(pos.qty) > 0 else OrderSide.SELL,
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc),
                )
                for pos in positions
            ]
        except Exception:
            self._log.exception("get_positions_error")
            return []

    def get_account(self) -> AccountInfo:
        """Get account info."""
        account = self._client.get_account()
        return AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value or account.equity),
            day_trade_count=int(account.daytrade_count or 0),
            pattern_day_trader=bool(account.pattern_day_trader),
        )

    def get_order(self, order_id: str) -> Order | None:
        """Get an order by ID (any status)."""
        try:
            result = self._client.get_order_by_id(order_id)
            return self._convert_order(result)
        except APIError:
            return None
        except Exception:
            self._log.exception("get_order_error", order_id=order_id)
            return None

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get open orders, optionally filtered by symbol."""
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            if symbol:
                request.symbols = [symbol]
            orders = self._client.get_orders(request)
            return [self._convert_order(o) for o in orders]
        except Exception:
            self._log.exception("get_open_orders_error")
            return []
