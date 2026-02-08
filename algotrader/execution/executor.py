"""Executor protocol — abstract interface for order execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from algotrader.core.models import Order, OrderSide, OrderType, TimeInForce, Position, AccountInfo


@runtime_checkable
class Executor(Protocol):
    """Abstract execution interface.

    Swap implementations (Alpaca, IBKR, paper) without changing strategy code.
    """

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
    ) -> Order | None:
        """Submit an order to the broker.

        Args:
            symbol: Ticker symbol
            qty: Number of shares
            side: Buy or sell
            order_type: Market, limit, stop, etc.
            time_in_force: Day, GTC, IOC, etc.
            limit_price: Limit price (for limit/stop-limit orders)
            stop_price: Stop price (for stop/stop-limit orders)
            client_order_id: Idempotent order ID to prevent duplicates

        Returns:
            Order object if submitted, None if failed
        """
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled."""
        ...

    def close_position(self, symbol: str) -> bool:
        """Close an entire position for a symbol.

        Returns True if closed (or position didn't exist — ghost position handling).
        Returns False only if close actually failed.
        """
        ...

    def get_position(self, symbol: str) -> Position | None:
        """Get a single position by symbol. None if no position."""
        ...

    def get_positions(self) -> list[Position]:
        """Get all open positions from the broker."""
        ...

    def get_account(self) -> AccountInfo:
        """Get account info (equity, cash, buying power)."""
        ...

    def get_order(self, order_id: str) -> Order | None:
        """Get an order by ID (any status)."""
        ...

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get open orders, optionally filtered by symbol."""
        ...
