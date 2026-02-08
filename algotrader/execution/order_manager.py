"""Order tracking, fill callbacks, and retry logic."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import pytz
import structlog

from algotrader.core.events import EventBus, ORDER_FILLED, ORDER_CANCELLED, ORDER_REJECTED
from algotrader.core.models import Order, OrderStatus
from algotrader.execution.executor import Executor

logger = structlog.get_logger()


class OrderManager:
    """Tracks pending/filled/cancelled orders with fill callbacks and retry logic.

    Polls the executor for order status updates and fires events on the EventBus
    when orders transition to terminal states (filled, cancelled, rejected).
    """

    def __init__(self, executor: Executor, event_bus: EventBus, max_retries: int = 2) -> None:
        self._executor = executor
        self._event_bus = event_bus
        self._max_retries = max_retries
        self._log = logger.bind(component="order_manager")

        # Tracked orders by order ID
        self._pending_orders: dict[str, Order] = {}
        self._filled_orders: dict[str, Order] = {}
        self._failed_orders: dict[str, Order] = {}

        # Per-order callbacks: order_id -> list of callbacks
        self._fill_callbacks: dict[str, list[Callable[[Order], Any]]] = {}
        self._retry_counts: dict[str, int] = {}

    def track_order(
        self,
        order: Order,
        on_fill: Callable[[Order], Any] | None = None,
    ) -> None:
        """Start tracking an order. Optionally register a fill callback."""
        self._pending_orders[order.id] = order
        self._retry_counts[order.id] = 0
        if on_fill:
            self._fill_callbacks.setdefault(order.id, []).append(on_fill)
        self._log.info("order_tracked", order_id=order.id, symbol=order.symbol, side=order.side.value)

    def check_orders(self) -> None:
        """Poll broker for status updates on all pending orders."""
        if not self._pending_orders:
            return

        broker_orders = self._executor.get_open_orders()
        broker_order_map = {o.id: o for o in broker_orders}

        completed_ids: list[str] = []

        for order_id, tracked_order in list(self._pending_orders.items()):
            broker_order = broker_order_map.get(order_id)

            if broker_order:
                # Order still open — update status
                tracked_order.status = broker_order.status
                tracked_order.filled_qty = broker_order.filled_qty
                tracked_order.filled_avg_price = broker_order.filled_avg_price
                continue

            # Order not in open orders — check if it was filled or cancelled
            try:
                updated = self._get_order_status(order_id)
                if updated:
                    tracked_order.status = updated.status
                    tracked_order.filled_qty = updated.filled_qty
                    tracked_order.filled_avg_price = updated.filled_avg_price
                    tracked_order.filled_at = updated.filled_at
            except Exception:
                # If we can't fetch it, assume filled (most common case)
                tracked_order.status = OrderStatus.FILLED
                self._log.warning("order_status_assumed_filled", order_id=order_id)

            if tracked_order.status == OrderStatus.FILLED:
                self._on_order_filled(tracked_order)
                completed_ids.append(order_id)
            elif tracked_order.status in (OrderStatus.CANCELLED, OrderStatus.EXPIRED):
                self._on_order_cancelled(tracked_order)
                completed_ids.append(order_id)
            elif tracked_order.status == OrderStatus.REJECTED:
                self._on_order_rejected(tracked_order)
                completed_ids.append(order_id)

        # Move completed orders out of pending
        for order_id in completed_ids:
            order = self._pending_orders.pop(order_id, None)
            if order and order.status == OrderStatus.FILLED:
                self._filled_orders[order_id] = order
            elif order:
                self._failed_orders[order_id] = order

    def _get_order_status(self, order_id: str) -> Order | None:
        """Try to get final order status from broker via Executor protocol."""
        try:
            return self._executor.get_order(order_id)
        except Exception:
            self._log.debug("order_status_fetch_failed", order_id=order_id)
        return None

    def _on_order_filled(self, order: Order) -> None:
        """Handle a filled order."""
        self._log.info(
            "order_filled",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            qty=order.filled_qty,
            avg_price=order.filled_avg_price,
        )

        # Fire registered callbacks
        for callback in self._fill_callbacks.pop(order.id, []):
            try:
                callback(order)
            except Exception:
                self._log.exception("fill_callback_error", order_id=order.id)

        # Publish event
        self._event_bus.publish(
            ORDER_FILLED,
            order=order,
        )

    def _on_order_cancelled(self, order: Order) -> None:
        """Handle a cancelled/expired order. May retry."""
        retry_count = self._retry_counts.get(order.id, 0)

        if retry_count < self._max_retries:
            self._retry_counts[order.id] = retry_count + 1
            self._log.info(
                "order_retry",
                order_id=order.id,
                symbol=order.symbol,
                attempt=retry_count + 1,
            )
            # Resubmit the order
            new_order = self._executor.submit_order(
                symbol=order.symbol,
                qty=order.qty,
                side=order.side,
                order_type=order.order_type,
                time_in_force=order.time_in_force,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
            )
            if new_order:
                # Transfer callbacks to new order
                callbacks = self._fill_callbacks.pop(order.id, [])
                if callbacks:
                    self._fill_callbacks[new_order.id] = callbacks
                self._retry_counts[new_order.id] = retry_count + 1
                self._pending_orders[new_order.id] = new_order
                return

        self._log.warning("order_cancelled", order_id=order.id, symbol=order.symbol)
        self._fill_callbacks.pop(order.id, None)
        self._event_bus.publish(ORDER_CANCELLED, order=order)

    def _on_order_rejected(self, order: Order) -> None:
        """Handle a rejected order."""
        self._log.error("order_rejected", order_id=order.id, symbol=order.symbol)
        self._fill_callbacks.pop(order.id, None)
        self._event_bus.publish(ORDER_REJECTED, order=order)

    @property
    def pending_count(self) -> int:
        return len(self._pending_orders)

    @property
    def pending_orders(self) -> list[Order]:
        return list(self._pending_orders.values())

    @property
    def filled_today(self) -> list[Order]:
        today = datetime.now(pytz.UTC).date()
        return [
            o for o in self._filled_orders.values()
            if o.filled_at and o.filled_at.date() == today
        ]

    def clear_history(self) -> None:
        """Clear filled/failed order history (call at end of day)."""
        self._filled_orders.clear()
        self._failed_orders.clear()
        self._fill_callbacks.clear()
        self._retry_counts.clear()
