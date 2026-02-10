"""Mid-day performance review and rebalance.

Triggers:
1. Scheduled review at review_time (default noon ET)
2. On regime change (from event bus)
3. Manual trigger (from orchestrator via force=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pytz
import structlog

from algotrader.core.config import StrategyConfig
from algotrader.core.events import EventBus, REGIME_CHANGED
from algotrader.core.models import MarketRegime
from algotrader.strategies.base import StrategyBase
from algotrader.strategy_selector.allocator import CapitalAllocator
from algotrader.strategy_selector.scorer import StrategyScorer

logger = structlog.get_logger()


@dataclass
class ReviewAction:
    """An action taken by the reviewer."""

    strategy_name: str
    action: str                  # "scale_up", "scale_down", "disable", "no_change", "reallocated"
    old_capital: float
    new_capital: float
    reason: str


class MidDayReviewer:
    """Review strategy performance and adjust allocations mid-day.

    Checks each strategy's daily P&L and scales capital up/down:
    - Losing > 0.5% of allocation → scale down 50%
    - Losing > 0.8% of allocation → disable (set capital to 0)
    - Winning > 0.3% with 50%+ WR → scale up 25% (capped at config max)
    """

    def __init__(
        self,
        scorer: StrategyScorer,
        allocator: CapitalAllocator,
        event_bus: EventBus,
        strategy_configs: dict[str, StrategyConfig],
        total_capital: float = 60000.0,
        review_hour: int = 12,
        review_minute: int = 0,
        scale_down_threshold_pct: float = -0.5,
        disable_threshold_pct: float = -0.8,
        scale_up_threshold_pct: float = 0.3,
        scale_up_factor: float = 1.25,
        scale_down_factor: float = 0.5,
        min_trades_for_review: int = 2,
    ) -> None:
        self._scorer = scorer
        self._allocator = allocator
        self._strategy_configs = strategy_configs
        self._total_capital = total_capital
        self._log = logger.bind(component="midday_reviewer")

        self._review_hour = review_hour
        self._review_minute = review_minute
        self._scale_down_threshold = scale_down_threshold_pct
        self._disable_threshold = disable_threshold_pct
        self._scale_up_threshold = scale_up_threshold_pct
        self._scale_up_factor = scale_up_factor
        self._scale_down_factor = scale_down_factor
        self._min_trades = min_trades_for_review

        self._last_review: datetime | None = None
        self._review_count: int = 0
        self._pending_regime_review: bool = False
        self._pending_regime: MarketRegime | None = None

        # Subscribe to regime changes
        event_bus.subscribe(REGIME_CHANGED, self._on_regime_change)

        self._log.info(
            "reviewer_initialized",
            review_time=f"{review_hour:02d}:{review_minute:02d} ET",
            thresholds={
                "scale_down": scale_down_threshold_pct,
                "disable": disable_threshold_pct,
                "scale_up": scale_up_threshold_pct,
            },
        )

    @property
    def needs_review(self) -> bool:
        """Check if scheduled review is due."""
        et_now = datetime.now(pytz.timezone("America/New_York"))

        # Only review during market hours
        if et_now.hour < 10 or et_now.hour >= 16:
            return False

        # Check for pending regime change review
        if self._pending_regime_review:
            return True

        # Check if we've already reviewed today at the scheduled time
        if self._last_review:
            last_et = self._last_review.astimezone(pytz.timezone("America/New_York"))
            if last_et.date() == et_now.date() and last_et.hour >= self._review_hour:
                return False  # Already reviewed today

        # Is it time?
        if et_now.hour > self._review_hour:
            return True
        if et_now.hour == self._review_hour and et_now.minute >= self._review_minute:
            return True

        return False

    def review(
        self,
        strategies: dict[str, StrategyBase],
        regime: MarketRegime,
        vix_level: float | None = None,
        force: bool = False,
    ) -> list[ReviewAction]:
        """Run a performance review and adjust allocations.

        Returns list of actions taken.
        """
        if not force and not self.needs_review:
            return []

        # Check if this is a regime-change forced review
        is_regime_change = self._pending_regime_review and self._pending_regime is not None
        if is_regime_change:
            regime = self._pending_regime
            self._pending_regime_review = False
            self._pending_regime = None

        actions = []

        if is_regime_change or force:
            # Full re-score and re-allocate on regime change
            actions = self._full_reallocation(strategies, regime, vix_level)
        else:
            # Normal scheduled review — adjust based on performance
            actions = self._performance_review(strategies)

        # Mark review as done
        self._last_review = datetime.now(pytz.UTC)
        self._review_count += 1

        # Log summary
        scaled_up = [a for a in actions if a.action == "scale_up"]
        scaled_down = [a for a in actions if a.action in ("scale_down", "disable")]
        self._log.info(
            "midday_review_complete",
            review_number=self._review_count,
            scaled_up=len(scaled_up),
            scaled_down=len(scaled_down),
            total_actions=len(actions),
            trigger="regime_change" if is_regime_change else "scheduled",
        )

        return actions

    def _performance_review(self, strategies: dict[str, StrategyBase]) -> list[ReviewAction]:
        """Review each strategy's morning performance and adjust capital."""
        actions = []

        for name, strategy in strategies.items():
            status = strategy.get_status()

            # Calculate daily P&L as % of allocation
            if strategy._total_capital > 0:
                daily_pnl_pct = status.daily_pnl / strategy._total_capital * 100
            else:
                daily_pnl_pct = 0.0

            old_capital = strategy._total_capital

            # Case A: Strategy is losing badly (> disable threshold)
            if daily_pnl_pct < self._disable_threshold:
                strategy.set_capital(0.0)
                actions.append(ReviewAction(
                    strategy_name=name,
                    action="disable",
                    old_capital=old_capital,
                    new_capital=0.0,
                    reason=f"Severe loss: {daily_pnl_pct:.2f}%",
                ))

            # Case B: Strategy is losing significantly
            elif daily_pnl_pct < self._scale_down_threshold and status.daily_trades >= self._min_trades:
                new_capital = old_capital * self._scale_down_factor
                strategy.set_capital(new_capital)
                actions.append(ReviewAction(
                    strategy_name=name,
                    action="scale_down",
                    old_capital=old_capital,
                    new_capital=round(new_capital, 2),
                    reason=f"Morning loss: {daily_pnl_pct:.2f}% on {status.daily_trades} trades",
                ))

            # Case C: Strategy is winning well
            elif (
                daily_pnl_pct > self._scale_up_threshold
                and status.daily_trades >= self._min_trades
                and status.win_rate >= 0.5
            ):
                config = self._strategy_configs.get(name, StrategyConfig())
                max_from_config = self._total_capital * (config.capital_allocation_pct / 100)
                new_capital = min(old_capital * self._scale_up_factor, max_from_config)
                if new_capital > old_capital:
                    strategy.set_capital(new_capital)
                    actions.append(ReviewAction(
                        strategy_name=name,
                        action="scale_up",
                        old_capital=old_capital,
                        new_capital=round(new_capital, 2),
                        reason=f"Morning win: {daily_pnl_pct:+.2f}%, WR={status.win_rate:.0%}",
                    ))
                else:
                    actions.append(ReviewAction(
                        strategy_name=name,
                        action="no_change",
                        old_capital=old_capital,
                        new_capital=old_capital,
                        reason=f"P&L: {daily_pnl_pct:+.2f}%, already at max config cap",
                    ))

            # Case D: No significant change
            else:
                actions.append(ReviewAction(
                    strategy_name=name,
                    action="no_change",
                    old_capital=old_capital,
                    new_capital=old_capital,
                    reason=f"P&L: {daily_pnl_pct:+.2f}%, {status.daily_trades} trades",
                ))

        return actions

    def _full_reallocation(
        self,
        strategies: dict[str, StrategyBase],
        regime: MarketRegime,
        vix_level: float | None,
    ) -> list[ReviewAction]:
        """Full re-score and re-allocate (triggered by regime change)."""
        strategy_names = list(strategies.keys())

        # Gather opportunity assessments from all strategies
        assessments = {}
        for name, strategy in strategies.items():
            try:
                assessments[name] = strategy.assess_opportunities(regime)
            except Exception:
                pass

        new_scores = self._scorer.score_strategies(
            strategy_names, regime, vix_level, assessments=assessments,
        )
        new_allocations = self._allocator.allocate(new_scores)
        self._allocator.apply_allocations(new_allocations, strategies)

        actions = []
        for alloc in new_allocations:
            if alloc.strategy_name in strategies:
                actions.append(ReviewAction(
                    strategy_name=alloc.strategy_name,
                    action="reallocated",
                    old_capital=strategies[alloc.strategy_name]._total_capital,
                    new_capital=alloc.allocated_capital,
                    reason=f"Regime change → rescore: {alloc.score:.2f}",
                ))

        return actions

    def _on_regime_change(self, regime: MarketRegime | None = None, old_type=None, **kwargs) -> None:
        """Trigger a forced reallocation when regime changes mid-day."""
        if regime is None:
            return
        self._log.info(
            "regime_change_detected",
            old=old_type.value if old_type else "none",
            new=regime.regime_type.value,
        )
        self._pending_regime_review = True
        self._pending_regime = regime
