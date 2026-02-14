"""StrategyBase ABC — base class for all trading strategies."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any

import pytz
import structlog

from algotrader.core.config import StrategyConfig
from algotrader.core.events import EventBus
from algotrader.core.models import (
    Order, OrderSide, RegimeType, Signal, StrategyStatus, TradeRecord,
    MarketRegime,
)
from algotrader.data.provider import DataProvider
from algotrader.execution.executor import Executor

from algotrader.tracking.journal import TradeJournal

logger = structlog.get_logger()


@dataclass
class OpportunityAssessment:
    """What a strategy sees right now."""

    num_candidates: int = 0
    avg_risk_reward: float = 0.0
    confidence: float = 0.0          # 0.0-1.0
    estimated_daily_trades: int = 0
    estimated_edge_pct: float = 0.0  # Expected return on deployed capital
    details: list[dict] = field(default_factory=list)

    @property
    def has_opportunities(self) -> bool:
        return self.num_candidates > 0 and self.confidence > 0.0


class StrategyBase(ABC):
    """Abstract base class for all trading strategies.

    Provides:
    - Capital tracking (reserve/release)
    - Daily metrics with auto-reset
    - State save/restore
    - Standard lifecycle: warm_up → run_cycle → on_fill
    """

    def __init__(
        self,
        name: str,
        config: StrategyConfig,
        data_provider: DataProvider,
        executor: Executor,
        event_bus: EventBus,
    ) -> None:
        self.name = name
        self.config = config
        self.data_provider = data_provider
        self.executor = executor
        self.event_bus = event_bus
        self._log = logger.bind(strategy=name)

        # Capital management
        self._total_capital: float = 0.0
        self._capital_reserved: float = 0.0
        self._capital_used: float = 0.0

        # Daily metrics (auto-reset each day)
        self._metrics_date: date | None = None
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._daily_wins: int = 0
        self._daily_losses: int = 0

        # Trade journal (set by orchestrator via set_journal)
        self._journal: TradeJournal | None = None

        # State
        self._enabled: bool = config.enabled
        self._warmed_up: bool = False
        self._last_cycle_time: datetime | None = None

    # ── Capital Management ────────────────────────────────────────────

    def set_capital(self, total_capital: float) -> None:
        """Set total capital allocated to this strategy."""
        self._total_capital = total_capital
        self._log.info("capital_set", total=total_capital)

    @property
    def available_capital(self) -> float:
        """Capital available for new positions."""
        return self._total_capital - self._capital_reserved

    def reserve_capital(self, amount: float) -> bool:
        """Reserve capital for a new position. Returns False if insufficient."""
        if amount > self.available_capital:
            self._log.warning(
                "insufficient_capital",
                requested=amount,
                available=self.available_capital,
            )
            return False
        self._capital_reserved += amount
        return True

    def set_journal(self, journal: TradeJournal) -> None:
        """Inject trade journal for recording completed trades."""
        self._journal = journal

    def release_capital(self, amount: float) -> None:
        """Release reserved capital when a position is closed."""
        self._capital_reserved = max(0, self._capital_reserved - amount)

    def get_held_symbols(self) -> list[str]:
        """Return symbols currently held by this strategy.

        Default: inspects self._trades if it exists.
        Override in strategies with non-standard tracking (e.g. pairs).
        """
        trades = getattr(self, "_trades", {})
        symbols = []
        for key, trade in trades.items():
            sym = getattr(trade, "symbol", key)
            symbols.append(sym)
        return symbols

    # ── Daily Metrics ─────────────────────────────────────────────────

    def _check_day_reset(self) -> None:
        """Reset daily metrics if we've crossed into a new trading day."""
        today = datetime.now(pytz.UTC).date()
        if self._metrics_date != today:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._daily_wins = 0
            self._daily_losses = 0
            self._capital_used = 0.0
            self._metrics_date = today
            self._log.info("daily_metrics_reset")

    def record_trade(
        self,
        pnl: float,
        *,
        symbol: str = "",
        side: OrderSide | None = None,
        qty: float = 0,
        entry_price: float = 0,
        exit_price: float = 0,
        entry_time: datetime | None = None,
        entry_reason: str = "",
        exit_reason: str = "",
        conviction: float = 1.0,
        regime: RegimeType | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a completed trade's P&L and write to journal if available."""
        self._check_day_reset()
        self._daily_pnl += pnl
        self._daily_trades += 1
        if pnl >= 0:
            self._daily_wins += 1
        else:
            self._daily_losses += 1

        # Write to trade journal DB
        if self._journal and symbol:
            try:
                record = TradeRecord(
                    strategy_name=self.name,
                    symbol=symbol,
                    side=side or OrderSide.BUY,
                    qty=qty,
                    entry_price=entry_price,
                    exit_price=exit_price or entry_price,
                    entry_time=entry_time,
                    exit_time=datetime.now(pytz.UTC),
                    realized_pnl=pnl,
                    conviction=conviction,
                    regime=regime,
                    entry_reason=entry_reason,
                    exit_reason=exit_reason,
                    metadata=metadata or {},
                )
                self._journal.record_trade(record)
            except Exception:
                self._log.exception("journal_record_failed", symbol=symbol)

    # ── Lifecycle ─────────────────────────────────────────────────────

    @abstractmethod
    def warm_up(self) -> None:
        """Initialize strategy state, load historical data, run calcs.

        Called once at startup before trading begins.
        """
        ...

    @abstractmethod
    def run_cycle(self, regime: MarketRegime | None = None) -> list[Signal]:
        """Execute one trading cycle.

        Scan for signals, manage existing positions, generate new signals.
        Called on each cycle (e.g. every 5 minutes for IEX).

        Returns list of signals generated this cycle.
        """
        ...

    def on_fill(self, order: Order) -> None:
        """Called when an order from this strategy is filled.

        Override to handle fill logic (e.g. update internal position tracking).
        """
        self._log.info(
            "order_filled",
            symbol=order.symbol,
            side=order.side.value,
            qty=order.filled_qty,
            price=order.filled_avg_price,
        )

    def on_signal(self, signal: Signal) -> None:
        """Called when a signal is generated. Override for custom handling."""
        pass

    def assess_opportunities(self, regime: MarketRegime | None = None) -> OpportunityAssessment:
        """Assess tradeable opportunities right now. Override in each strategy."""
        return OpportunityAssessment()

    # ── Status ────────────────────────────────────────────────────────

    def get_status(self) -> StrategyStatus:
        """Return current strategy status."""
        self._check_day_reset()
        return StrategyStatus(
            name=self.name,
            enabled=self._enabled,
            capital_reserved=self._capital_reserved,
            capital_used=self._capital_used,
            daily_pnl=self._daily_pnl,
            daily_trades=self._daily_trades,
            daily_wins=self._daily_wins,
            daily_losses=self._daily_losses,
            last_cycle_time=self._last_cycle_time,
        )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True
        self._log.info("strategy_enabled")

    def disable(self, reason: str = "") -> None:
        self._enabled = False
        self._log.info("strategy_disabled", reason=reason)

    # ── State Persistence ─────────────────────────────────────────────

    def save_state(self, state_dir: str = "data/state") -> None:
        """Save strategy state to disk for recovery."""
        path = Path(state_dir) / f"{self.name}_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        state = self._get_state()
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        self._log.debug("state_saved", path=str(path))

    def restore_state(self, state_dir: str = "data/state") -> bool:
        """Restore strategy state from disk. Returns True if restored."""
        path = Path(state_dir) / f"{self.name}_state.json"
        if not path.exists():
            return False

        try:
            with open(path, "r") as f:
                state = json.load(f)
            self._restore_state(state)
            self._log.info("state_restored", path=str(path))
            return True
        except Exception:
            self._log.exception("state_restore_failed")
            return False

    def _get_state(self) -> dict[str, Any]:
        """Get serializable state. Override to add strategy-specific state."""
        return {
            "name": self.name,
            "enabled": self._enabled,
            "capital_reserved": self._capital_reserved,
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "metrics_date": str(self._metrics_date) if self._metrics_date else None,
        }

    def _restore_state(self, state: dict[str, Any]) -> None:
        """Restore from state dict. Override to restore strategy-specific state."""
        self._enabled = state.get("enabled", True)
        self._capital_reserved = state.get("capital_reserved", 0.0)
        self._daily_pnl = state.get("daily_pnl", 0.0)
        self._daily_trades = state.get("daily_trades", 0)
