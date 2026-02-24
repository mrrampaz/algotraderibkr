"""Shared IBKR connection manager."""

from __future__ import annotations

import threading
import time
from typing import Callable, TypeVar

import nest_asyncio
import structlog
from ib_async import IB

from algotrader.core.config import IBKRConfig
from algotrader.core.events import EventBus

logger = structlog.get_logger()

T = TypeVar("T")

BROKER_CONNECTED = "broker_connected"
BROKER_DISCONNECTED = "broker_disconnected"
BROKER_RECONNECTED = "broker_reconnected"
BROKER_RECONNECT_FAILED = "broker_reconnect_failed"


class IBKRConnection:
    """Singleton IBKR connection manager shared by data provider and executor."""

    _instance: IBKRConnection | None = None
    _instance_lock = threading.Lock()

    def __init__(self, config: IBKRConfig, event_bus: EventBus | None = None) -> None:
        self._ib = IB()
        self._config = config
        self._event_bus = event_bus
        self._connected = False
        self._disconnecting = False
        self._reconnecting = False
        self._reconnect_count = 0
        self._lock = threading.RLock()
        self._log = logger.bind(
            component="ibkr_connection",
            host=config.host,
            port=config.port,
            client_id=config.client_id,
            readonly=config.readonly,
        )

        # Allow sync calls in the orchestrator's synchronous run loop.
        nest_asyncio.apply()

        # Register connection lifecycle handlers.
        self._ib.connectedEvent += self._on_connected
        self._ib.disconnectedEvent += self._on_disconnected

    @classmethod
    def get_instance(
        cls,
        config: IBKRConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> IBKRConnection:
        """Get or create the singleton connection."""
        with cls._instance_lock:
            if cls._instance is None:
                if config is None:
                    raise ValueError("IBKRConfig is required for first IBKRConnection initialization")
                cls._instance = cls(config=config, event_bus=event_bus)
            elif event_bus is not None and cls._instance._event_bus is None:
                cls._instance._event_bus = event_bus
            return cls._instance

    def _publish(self, event_type: str, **data) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(event_type, **data)
        except Exception:
            self._log.exception("ibkr_connection_event_publish_failed", event_type=event_type)

    def _on_connected(self, *_args) -> None:
        with self._lock:
            self._connected = True
            reconnecting = self._reconnecting
            self._reconnect_count = 0

        self._log.info("ibkr_connected_event")
        if reconnecting:
            self._publish(BROKER_RECONNECTED, provider="ibkr")
        else:
            self._publish(BROKER_CONNECTED, provider="ibkr")

    def _on_disconnected(self, *_args) -> None:
        with self._lock:
            self._connected = False
            manual_disconnect = self._disconnecting
            already_reconnecting = self._reconnecting

        if manual_disconnect:
            self._log.info("ibkr_disconnected")
            return

        self._log.warning("ibkr_disconnected_unexpected")
        self._publish(BROKER_DISCONNECTED, provider="ibkr", expected=False)

        if not already_reconnecting:
            self._attempt_reconnect()

    def _attempt_reconnect(self) -> None:
        with self._lock:
            if self._reconnecting:
                return
            self._reconnecting = True

        max_attempts = max(1, int(self._config.max_reconnect_attempts))
        base_delay = max(1, int(self._config.reconnect_delay_seconds))

        try:
            for attempt in range(1, max_attempts + 1):
                delay = 0 if attempt == 1 else base_delay * (2 ** (attempt - 2))
                if delay > 0:
                    self._log.warning(
                        "ibkr_reconnect_waiting",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay_seconds=delay,
                    )
                    time.sleep(delay)

                with self._lock:
                    self._reconnect_count = attempt

                self._log.warning(
                    "ibkr_reconnect_attempt",
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                if self.connect():
                    self._log.info("ibkr_reconnect_succeeded", attempt=attempt)
                    self._publish(BROKER_RECONNECTED, provider="ibkr", attempt=attempt)
                    return

            self._log.error("ibkr_reconnect_exhausted", attempts=max_attempts)
            self._publish(BROKER_RECONNECT_FAILED, provider="ibkr", attempts=max_attempts)
        finally:
            with self._lock:
                self._reconnecting = False

    def connect(self) -> bool:
        """Connect to IBKR and validate by requesting account summary."""
        with self._lock:
            if self._ib.isConnected():
                self._connected = True
                return True

            self._disconnecting = False
            self._log.info("ibkr_connecting")

            try:
                self._ib.connect(
                    host=self._config.host,
                    port=self._config.port,
                    clientId=self._config.client_id,
                    timeout=self._config.timeout,
                    readonly=self._config.readonly,
                    account=self._config.account or "",
                )

                if not self._ib.isConnected():
                    self._connected = False
                    self._log.error("ibkr_connect_failed_not_connected")
                    return False

                # Validation probe to confirm the session is usable.
                summary = self._ib.accountSummary(self._config.account or "")
                if summary is None:
                    self._log.warning("ibkr_account_summary_empty")

                self._connected = True
                self._reconnect_count = 0
                self._log.info("ibkr_connected")
                self._publish(BROKER_CONNECTED, provider="ibkr")
                return True
            except Exception:
                self._connected = False
                self._log.exception("ibkr_connect_failed")
                return False

    def disconnect(self) -> None:
        """Disconnect cleanly from IBKR and cancel active market data subscriptions."""
        with self._lock:
            self._disconnecting = True
            try:
                if self._ib.isConnected():
                    # Cancel only active request IDs to avoid noisy
                    # "No reqId found for contract" messages on snapshot tickers.
                    req_map = getattr(getattr(self._ib, "wrapper", None), "reqId2Ticker", {}) or {}
                    for req_id in list(req_map.keys()):
                        try:
                            self._ib.client.cancelMktData(req_id)
                        except Exception:
                            self._log.debug("ibkr_cancel_mkt_data_failed", req_id=req_id)

                    self._ib.disconnect()
            finally:
                self._connected = False
                self._disconnecting = False
                self._log.info("ibkr_disconnected_manual")
                self._publish(BROKER_DISCONNECTED, provider="ibkr", expected=True)

    def ensure_connected(self) -> bool:
        """Ensure the IB session is connected."""
        with self._lock:
            if self.connected:
                return True
        return self.connect()

    def execute(self, operation: Callable[[IB], T]) -> T:
        """Run an IB operation under a connection lock on an active connection."""
        with self._lock:
            if not self.ensure_connected():
                raise RuntimeError("IBKR is not connected")
            return operation(self._ib)

    @property
    def ib(self) -> IB:
        """Access the underlying IB instance. Always call ensure_connected() first."""
        with self._lock:
            if not self.connected:
                raise RuntimeError("IBKR is not connected")
            return self._ib

    @property
    def connected(self) -> bool:
        return self._connected and self._ib.isConnected()

    @property
    def config(self) -> IBKRConfig:
        return self._config
