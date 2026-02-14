"""Live portfolio state tracking."""

from __future__ import annotations

from datetime import datetime, date

import pytz
import structlog

from algotrader.core.models import Position, AccountInfo
from algotrader.execution.executor import Executor

logger = structlog.get_logger()


class PortfolioTracker:
    """Track live portfolio state: positions, P&L, daily metrics.

    Always reads from the broker for ground truth.
    """

    def __init__(self, executor: Executor, starting_equity: float) -> None:
        self._executor = executor
        self._log = logger.bind(component="portfolio_tracker")

        self._starting_equity = starting_equity
        self._day_start_equity = starting_equity
        self._peak_equity = starting_equity
        self._metrics_date: date | None = None

        # Daily tracking
        self._realized_pnl: float = 0.0
        self._trades_today: int = 0

        # Strategy attribution: symbol -> [strategy_names]
        self._strategy_symbol_map: dict[str, list[str]] = {}

    def set_strategy_symbol_map(self, mapping: dict[str, list[str]]) -> None:
        """Update the symbol -> strategy_name mapping for position attribution."""
        self._strategy_symbol_map = mapping

    def reset_day(self, current_equity: float) -> None:
        """Reset daily tracking for a new trading day."""
        today = datetime.now(pytz.UTC).date()
        if self._metrics_date == today:
            return  # Already reset

        self._day_start_equity = current_equity
        self._realized_pnl = 0.0
        self._trades_today = 0
        self._metrics_date = today
        self._log.info("portfolio_day_reset", equity=current_equity)

    def record_realized_pnl(self, pnl: float) -> None:
        """Record a realized P&L from a closed trade."""
        self._realized_pnl += pnl
        self._trades_today += 1

    def get_snapshot(self) -> dict:
        """Get current portfolio snapshot from broker."""
        account = self._executor.get_account()
        positions = self._executor.get_positions()

        # Enrich positions with strategy attribution
        for pos in positions:
            strategies = self._strategy_symbol_map.get(pos.symbol, [])
            pos.strategy_name = "+".join(strategies) if strategies else "untracked"

        # Update peak equity
        if account.equity > self._peak_equity:
            self._peak_equity = account.equity

        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        gross_exposure = sum(abs(p.market_value) for p in positions)
        daily_pnl = account.equity - self._day_start_equity

        long_exposure = sum(p.market_value for p in positions if p.side.value == "buy")
        short_exposure = sum(abs(p.market_value) for p in positions if p.side.value == "sell")

        return {
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": (daily_pnl / self._day_start_equity * 100) if self._day_start_equity > 0 else 0,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": self._realized_pnl,
            "gross_exposure": gross_exposure,
            "exposure_pct": (gross_exposure / account.equity * 100) if account.equity > 0 else 0,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "num_positions": len(positions),
            "positions": [p.model_dump() for p in positions],
            "drawdown": self._peak_equity - account.equity,
            "drawdown_pct": ((self._peak_equity - account.equity) / self._peak_equity * 100) if self._peak_equity > 0 else 0,
            "trades_today": self._trades_today,
        }

    def get_positions(self) -> list[Position]:
        """Get all positions from broker."""
        return self._executor.get_positions()

    def get_account(self) -> AccountInfo:
        """Get account info from broker."""
        return self._executor.get_account()
