"""Alerting system — sends notifications on critical trading events.

Supports multiple backends (log-only, file, webhook). Telegram/email
can be added later as separate alert backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

import pytz
import structlog

from algotrader.core.events import EventBus, KILL_SWITCH, REGIME_CHANGED, STRATEGY_DISABLED

logger = structlog.get_logger()


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    KILL_SWITCH = "kill_switch"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    STRATEGY_DISABLED = "strategy_disabled"
    BIG_WIN = "big_win"
    BIG_LOSS = "big_loss"
    REGIME_CHANGE = "regime_change"
    SYSTEM_ERROR = "system_error"
    DAILY_SUMMARY = "daily_summary"
    WEIGHT_ADJUSTMENT = "weight_adjustment"


@dataclass
class Alert:
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    metadata: dict = field(default_factory=dict)


class AlertBackend(Protocol):
    """Protocol for alert delivery backends."""

    def send(self, alert: Alert) -> bool: ...


class LogAlertBackend:
    """Writes alerts to structlog (always active)."""

    def __init__(self) -> None:
        self._log = structlog.get_logger().bind(component="alerts")

    def send(self, alert: Alert) -> bool:
        log_method = {
            AlertLevel.INFO: self._log.info,
            AlertLevel.WARNING: self._log.warning,
            AlertLevel.CRITICAL: self._log.error,
        }.get(alert.level, self._log.info)

        log_method(
            "alert",
            alert_type=alert.alert_type.value,
            title=alert.title,
            message=alert.message,
        )
        return True


class FileAlertBackend:
    """Writes alerts to a dedicated alerts file for easy monitoring."""

    def __init__(self, filepath: str = "data/logs/alerts.log") -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self._filepath = filepath

    def send(self, alert: Alert) -> bool:
        line = (
            f"[{alert.timestamp.isoformat()}] "
            f"[{alert.level.value.upper()}] "
            f"[{alert.alert_type.value}] "
            f"{alert.title}: {alert.message}\n"
        )
        with open(self._filepath, "a") as f:
            f.write(line)
        return True


class WebhookAlertBackend:
    """Sends alerts to a webhook URL (for Telegram bots, Slack, Discord, etc).

    Expects the webhook to accept POST with JSON body:
    {"text": "...", "level": "...", "type": "..."}

    Configure webhook_url in settings.yaml under alerts.webhook_url
    """

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url
        self._log = structlog.get_logger().bind(component="webhook_alert")

    def send(self, alert: Alert) -> bool:
        if not self._url:
            return False
        try:
            import httpx

            payload = {
                "text": f"*{alert.title}*\n{alert.message}",
                "level": alert.level.value,
                "type": alert.alert_type.value,
            }
            response = httpx.post(self._url, json=payload, timeout=10)
            return response.status_code < 300
        except Exception:
            self._log.debug("webhook_send_failed", url=self._url)
            return False


class AlertManager:
    """Central alert manager. Routes alerts to all configured backends.

    Subscribes to EventBus events and generates alerts automatically.
    Also provides manual alert methods for strategies and orchestrator.
    """

    def __init__(
        self,
        event_bus: EventBus,
        backends: list | None = None,
        big_win_threshold: float = 500.0,
        big_loss_threshold: float = -300.0,
    ) -> None:
        self._backends = backends or [LogAlertBackend(), FileAlertBackend()]
        self._event_bus = event_bus
        self._big_win = big_win_threshold
        self._big_loss = big_loss_threshold
        self._log = structlog.get_logger().bind(component="alert_manager")

        # Subscribe to events
        event_bus.subscribe(KILL_SWITCH, self._on_kill_switch)
        event_bus.subscribe(REGIME_CHANGED, self._on_regime_change)
        event_bus.subscribe(STRATEGY_DISABLED, self._on_strategy_disabled)

    def send_alert(self, alert: Alert) -> None:
        """Send alert to all backends."""
        for backend in self._backends:
            try:
                backend.send(alert)
            except Exception:
                self._log.debug("alert_backend_failed", backend=type(backend).__name__)

    def check_trade_alert(self, pnl: float, strategy: str, symbol: str) -> None:
        """Check if a closed trade warrants an alert."""
        if pnl >= self._big_win:
            self.send_alert(Alert(
                alert_type=AlertType.BIG_WIN,
                level=AlertLevel.INFO,
                title=f"Big Win: {strategy}",
                message=f"+${pnl:.2f} on {symbol}",
                timestamp=datetime.now(pytz.UTC),
                metadata={"strategy": strategy, "symbol": symbol, "pnl": pnl},
            ))
        elif pnl <= self._big_loss:
            self.send_alert(Alert(
                alert_type=AlertType.BIG_LOSS,
                level=AlertLevel.WARNING,
                title=f"Big Loss: {strategy}",
                message=f"-${abs(pnl):.2f} on {symbol}",
                timestamp=datetime.now(pytz.UTC),
                metadata={"strategy": strategy, "symbol": symbol, "pnl": pnl},
            ))

    def send_daily_summary(self, summary: dict) -> None:
        """Send end-of-day summary alert."""
        pnl = summary.get("total_pnl", 0)
        trades = summary.get("total_trades", 0)
        wins = summary.get("wins", 0)
        wr = summary.get("win_rate", 0)

        level = AlertLevel.INFO
        self.send_alert(Alert(
            alert_type=AlertType.DAILY_SUMMARY,
            level=level,
            title="Daily Summary",
            message=(
                f"P&L: ${pnl:+.2f} | "
                f"Trades: {trades} | "
                f"Wins: {wins} ({wr:.0%}) | "
                f"Best: ${summary.get('best_trade', 0):.2f} | "
                f"Worst: ${summary.get('worst_trade', 0):.2f}"
            ),
            timestamp=datetime.now(pytz.UTC),
            metadata=summary,
        ))

    def _on_kill_switch(self, reason: str = "", **kwargs) -> None:
        self.send_alert(Alert(
            alert_type=AlertType.KILL_SWITCH,
            level=AlertLevel.CRITICAL,
            title="KILL SWITCH ACTIVATED",
            message=f"Reason: {reason}. All positions closed, trading halted.",
            timestamp=datetime.now(pytz.UTC),
        ))

    def _on_regime_change(self, regime=None, old_type=None, **kwargs) -> None:
        old = old_type.value if old_type else "none"
        new = regime.regime_type.value if regime else "unknown"
        self.send_alert(Alert(
            alert_type=AlertType.REGIME_CHANGE,
            level=AlertLevel.INFO,
            title="Regime Change",
            message=f"{old} -> {new}",
            timestamp=datetime.now(pytz.UTC),
        ))

    def _on_strategy_disabled(self, strategy_name: str = "", reason: str = "", **kwargs) -> None:
        self.send_alert(Alert(
            alert_type=AlertType.STRATEGY_DISABLED,
            level=AlertLevel.WARNING,
            title=f"Strategy Disabled: {strategy_name}",
            message=f"Reason: {reason}",
            timestamp=datetime.now(pytz.UTC),
        ))
