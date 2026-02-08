"""Simple pub/sub event bus for inter-component communication."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


# Event type constants
ORDER_FILLED = "order_filled"
ORDER_CANCELLED = "order_cancelled"
ORDER_REJECTED = "order_rejected"
REGIME_CHANGED = "regime_changed"
RISK_ALERT = "risk_alert"
KILL_SWITCH = "kill_switch"
POSITION_OPENED = "position_opened"
POSITION_CLOSED = "position_closed"
STRATEGY_DISABLED = "strategy_disabled"


class EventBus:
    """Simple synchronous pub/sub event bus.

    Components subscribe to event types and get called when those events fire.
    All callbacks are called synchronously in subscription order.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[..., Any]]] = defaultdict(list)
        self._log = logger.bind(component="event_bus")

    def subscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Register a callback for an event type."""
        self._subscribers[event_type].append(callback)
        self._log.debug("subscriber_added", event_type=event_type, callback=callback.__qualname__)

    def unsubscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Remove a callback for an event type."""
        try:
            self._subscribers[event_type].remove(callback)
        except ValueError:
            pass

    def publish(self, event_type: str, **data: Any) -> None:
        """Publish an event to all subscribers."""
        subscribers = self._subscribers.get(event_type, [])
        if not subscribers:
            return

        self._log.debug("event_published", event_type=event_type, subscriber_count=len(subscribers))

        for callback in subscribers:
            try:
                callback(**data)
            except Exception:
                self._log.exception(
                    "subscriber_error",
                    event_type=event_type,
                    callback=callback.__qualname__,
                )

    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()
