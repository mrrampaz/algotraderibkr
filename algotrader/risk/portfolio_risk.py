"""Portfolio-level risk management with kill switch."""

from __future__ import annotations

from datetime import datetime

import pytz
import structlog

from algotrader.core.config import RiskConfig
from algotrader.core.events import EventBus, RISK_ALERT, KILL_SWITCH, STRATEGY_DISABLED
from algotrader.core.models import AccountInfo, StrategyStatus
from algotrader.execution.executor import Executor

logger = structlog.get_logger()


class PortfolioRiskManager:
    """Portfolio-level risk controls.

    Enforces:
    - Max daily loss (2%) → kill switch, close all
    - Max drawdown (8%) → reduce sizes
    - Max gross exposure (80%) → block new entries
    - Per-strategy loss limit (-1%) → disable strategy
    - Kill switch for emergency shutdown
    """

    def __init__(
        self,
        config: RiskConfig,
        executor: Executor,
        event_bus: EventBus,
        starting_equity: float,
    ) -> None:
        self._config = config
        self._executor = executor
        self._event_bus = event_bus
        self._log = logger.bind(component="portfolio_risk")

        self._starting_equity = starting_equity
        self._peak_equity = starting_equity
        self._day_start_equity = starting_equity
        self._killed = False

        # Thresholds (absolute $ amounts)
        self._max_daily_loss = starting_equity * (config.max_daily_loss_pct / 100)
        self._max_drawdown = starting_equity * (config.max_drawdown_pct / 100)
        self._max_exposure = starting_equity * (config.max_gross_exposure_pct / 100)
        self._strategy_loss_limit = starting_equity * (config.strategy_daily_loss_limit_pct / 100)

        self._log.info(
            "risk_manager_initialized",
            max_daily_loss=self._max_daily_loss,
            max_drawdown=self._max_drawdown,
            max_exposure=self._max_exposure,
        )

    @property
    def is_killed(self) -> bool:
        return self._killed

    def reset_day(self, current_equity: float) -> None:
        """Reset for a new trading day."""
        self._day_start_equity = current_equity
        self._max_daily_loss = current_equity * (self._config.max_daily_loss_pct / 100)
        self._strategy_loss_limit = current_equity * (self._config.strategy_daily_loss_limit_pct / 100)
        self._killed = False
        self._log.info("risk_day_reset", equity=current_equity)

    def check_risk(self, account: AccountInfo, strategy_statuses: list[StrategyStatus]) -> dict:
        """Run all risk checks. Returns a status dict.

        Should be called each cycle.
        """
        if self._killed:
            return {"status": "killed", "can_trade": False}

        current_equity = account.equity
        daily_pnl = current_equity - self._day_start_equity

        # Update peak equity for drawdown tracking
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        drawdown = self._peak_equity - current_equity
        drawdown_pct = (drawdown / self._peak_equity * 100) if self._peak_equity > 0 else 0

        # Calculate gross exposure
        positions = self._executor.get_positions()
        gross_exposure = sum(abs(p.market_value) for p in positions)

        result = {
            "status": "ok",
            "can_trade": True,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": (daily_pnl / self._day_start_equity * 100) if self._day_start_equity > 0 else 0,
            "drawdown": drawdown,
            "drawdown_pct": drawdown_pct,
            "gross_exposure": gross_exposure,
            "exposure_pct": (gross_exposure / current_equity * 100) if current_equity > 0 else 0,
        }

        # Check daily loss limit
        if daily_pnl <= -self._max_daily_loss:
            self._log.error(
                "daily_loss_limit_hit",
                daily_pnl=daily_pnl,
                limit=-self._max_daily_loss,
            )
            self._trigger_kill_switch("daily_loss_limit")
            result["status"] = "killed"
            result["can_trade"] = False
            return result

        # Check drawdown limit
        if drawdown >= self._max_drawdown:
            self._log.warning(
                "drawdown_limit_hit",
                drawdown=drawdown,
                drawdown_pct=drawdown_pct,
            )
            self._event_bus.publish(
                RISK_ALERT,
                alert_type="drawdown_limit",
                drawdown=drawdown,
                drawdown_pct=drawdown_pct,
            )
            result["status"] = "drawdown_warning"
            result["reduce_size"] = True

        # Check exposure limit
        if gross_exposure >= self._max_exposure:
            self._log.warning(
                "exposure_limit_hit",
                gross_exposure=gross_exposure,
                limit=self._max_exposure,
            )
            result["can_open_new"] = False

        # Check per-strategy loss limits
        for status in strategy_statuses:
            if status.daily_pnl <= -self._strategy_loss_limit:
                self._log.warning(
                    "strategy_loss_limit_hit",
                    strategy=status.name,
                    daily_pnl=status.daily_pnl,
                    limit=-self._strategy_loss_limit,
                )
                self._event_bus.publish(
                    STRATEGY_DISABLED,
                    strategy_name=status.name,
                    reason="daily_loss_limit",
                )

        return result

    def can_open_position(self, position_value: float, account: AccountInfo) -> bool:
        """Check if a new position can be opened within risk limits."""
        if self._killed:
            return False

        # Check single position size
        max_single = account.equity * (self._config.max_single_position_pct / 100)
        if position_value > max_single:
            self._log.warning(
                "position_too_large",
                value=position_value,
                max=max_single,
            )
            return False

        # Check total exposure
        positions = self._executor.get_positions()
        gross_exposure = sum(abs(p.market_value) for p in positions) + position_value
        if gross_exposure > self._max_exposure:
            self._log.warning(
                "exposure_limit_exceeded",
                new_exposure=gross_exposure,
                limit=self._max_exposure,
            )
            return False

        return True

    def _trigger_kill_switch(self, reason: str) -> None:
        """Emergency shutdown — close all positions, cancel all orders."""
        self._killed = True
        self._log.error("kill_switch_triggered", reason=reason)

        # Cancel all open orders
        try:
            open_orders = self._executor.get_open_orders()
            for order in open_orders:
                self._executor.cancel_order(order.id)
                self._log.info("kill_switch_cancelled_order", order_id=order.id)
        except Exception:
            self._log.exception("kill_switch_cancel_orders_failed")

        # Close all positions
        try:
            positions = self._executor.get_positions()
            for pos in positions:
                self._executor.close_position(pos.symbol)
                self._log.info("kill_switch_closed_position", symbol=pos.symbol)
        except Exception:
            self._log.exception("kill_switch_close_positions_failed")

        self._event_bus.publish(KILL_SWITCH, reason=reason)
