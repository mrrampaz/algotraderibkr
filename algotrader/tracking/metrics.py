"""Performance metrics calculation from trade history.

Used by attribution, learner, and dashboard to compute standard
quant performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import structlog

from algotrader.core.models import TradeRecord
from algotrader.tracking.journal import TradeJournal

logger = structlog.get_logger()


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a time period."""

    period_start: date
    period_end: date
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # 0.0–1.0
    total_pnl: float
    avg_pnl_per_trade: float
    avg_winner: float
    avg_loser: float
    profit_factor: float  # gross_profit / gross_loss
    largest_win: float
    largest_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_hold_time_minutes: float
    sharpe_ratio: float | None  # Annualized, if enough data
    max_drawdown_pct: float
    expectancy: float  # (win_rate * avg_win) - (loss_rate * avg_loss)
    daily_returns: list[float] = field(default_factory=list)


class MetricsCalculator:
    """Calculate performance metrics from trade records."""

    def __init__(self, trade_journal: TradeJournal) -> None:
        self._journal = trade_journal
        self._log = logger.bind(component="metrics_calculator")

    def calculate(
        self,
        strategy_name: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> PerformanceMetrics:
        """Calculate metrics for a strategy (or all) over a time period."""
        if end_date is None:
            end_date = datetime.now(pytz.UTC).date()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        trades = self._journal.get_trades(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        return self._compute_metrics(trades, start_date, end_date)

    def calculate_daily_returns(
        self,
        strategy_name: str | None = None,
        days: int = 30,
    ) -> pd.DataFrame:
        """Get daily P&L returns as a DataFrame.

        Columns: date, strategy, pnl, num_trades, win_rate
        """
        end_date = datetime.now(pytz.UTC).date()
        start_date = end_date - timedelta(days=days)

        trades = self._journal.get_trades(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        if not trades:
            return pd.DataFrame(columns=["date", "strategy", "pnl", "num_trades", "win_rate"])

        rows = []
        # Group trades by date and strategy
        by_day: dict[tuple[date, str], list[TradeRecord]] = {}
        for t in trades:
            if t.entry_time is None:
                continue
            d = t.entry_time.date()
            key = (d, t.strategy_name)
            by_day.setdefault(key, []).append(t)

        for (d, strat), day_trades in sorted(by_day.items()):
            pnl = sum(t.realized_pnl for t in day_trades)
            wins = sum(1 for t in day_trades if t.realized_pnl > 0)
            total = len(day_trades)
            rows.append({
                "date": d,
                "strategy": strat,
                "pnl": pnl,
                "num_trades": total,
                "win_rate": wins / total if total > 0 else 0.0,
            })

        return pd.DataFrame(rows)

    def calculate_by_regime(
        self,
        strategy_name: str,
        days: int = 30,
    ) -> dict[str, PerformanceMetrics]:
        """Calculate metrics grouped by regime type.

        Returns: {regime_type_value: PerformanceMetrics}
        """
        end_date = datetime.now(pytz.UTC).date()
        start_date = end_date - timedelta(days=days)

        trades = self._journal.get_trades(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        # Group by regime
        by_regime: dict[str, list[TradeRecord]] = {}
        for t in trades:
            regime_val = t.regime.value if t.regime else "unknown"
            by_regime.setdefault(regime_val, []).append(t)

        results = {}
        for regime_val, regime_trades in by_regime.items():
            results[regime_val] = self._compute_metrics(regime_trades, start_date, end_date)

        return results

    def _compute_metrics(
        self,
        trades: list[TradeRecord],
        start_date: date,
        end_date: date,
    ) -> PerformanceMetrics:
        """Core metrics computation from a list of trades."""
        if not trades:
            return PerformanceMetrics(
                period_start=start_date,
                period_end=end_date,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl_per_trade=0.0,
                avg_winner=0.0,
                avg_loser=0.0,
                profit_factor=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                avg_hold_time_minutes=0.0,
                sharpe_ratio=None,
                max_drawdown_pct=0.0,
                expectancy=0.0,
            )

        winners = [t for t in trades if t.realized_pnl > 0]
        losers = [t for t in trades if t.realized_pnl < 0]
        pnls = [t.realized_pnl for t in trades]

        total = len(trades)
        win_rate = len(winners) / total if total > 0 else 0.0

        avg_winner = float(np.mean([t.realized_pnl for t in winners])) if winners else 0.0
        avg_loser = float(np.mean([t.realized_pnl for t in losers])) if losers else 0.0

        # Daily P&L for Sharpe and drawdown
        daily_pnls = self._aggregate_daily_pnls(trades)

        return PerformanceMetrics(
            period_start=start_date,
            period_end=end_date,
            total_trades=total,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,
            total_pnl=sum(pnls),
            avg_pnl_per_trade=float(np.mean(pnls)),
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            profit_factor=self._calculate_profit_factor(trades),
            largest_win=max(pnls) if pnls else 0.0,
            largest_loss=min(pnls) if pnls else 0.0,
            max_consecutive_wins=self._max_consecutive(trades, winning=True),
            max_consecutive_losses=self._max_consecutive(trades, winning=False),
            avg_hold_time_minutes=self._avg_hold_time(trades),
            sharpe_ratio=self._calculate_sharpe(daily_pnls),
            max_drawdown_pct=self._calculate_max_drawdown(daily_pnls),
            expectancy=self._calculate_expectancy(trades),
            daily_returns=daily_pnls,
        )

    def _calculate_profit_factor(self, trades: list[TradeRecord]) -> float:
        gross_profit = sum(t.realized_pnl for t in trades if t.realized_pnl > 0)
        gross_loss = abs(sum(t.realized_pnl for t in trades if t.realized_pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _calculate_sharpe(self, daily_returns: list[float]) -> float | None:
        if len(daily_returns) < 5:
            return None
        returns = np.array(daily_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        if std_return == 0:
            return None
        # Annualize: sqrt(252) for daily returns
        return float((mean_return / std_return) * np.sqrt(252))

    def _calculate_expectancy(self, trades: list[TradeRecord]) -> float:
        """Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)"""
        winners = [t for t in trades if t.realized_pnl > 0]
        losers = [t for t in trades if t.realized_pnl < 0]
        if not trades:
            return 0.0
        win_rate = len(winners) / len(trades)
        loss_rate = len(losers) / len(trades)
        avg_win = float(np.mean([t.realized_pnl for t in winners])) if winners else 0.0
        avg_loss = abs(float(np.mean([t.realized_pnl for t in losers]))) if losers else 0.0
        return (win_rate * avg_win) - (loss_rate * avg_loss)

    def _calculate_max_drawdown(self, daily_pnls: list[float]) -> float:
        """Max drawdown as a percentage of peak cumulative P&L."""
        if not daily_pnls:
            return 0.0
        cumulative = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        if peak.max() == 0:
            return 0.0
        return float(drawdown.max() / peak.max() * 100) if peak.max() > 0 else 0.0

    def _aggregate_daily_pnls(self, trades: list[TradeRecord]) -> list[float]:
        """Aggregate trade P&Ls into daily totals."""
        by_day: dict[date, float] = {}
        for t in trades:
            if t.entry_time is None:
                continue
            d = t.entry_time.date()
            by_day[d] = by_day.get(d, 0.0) + t.realized_pnl

        if not by_day:
            return []

        return [pnl for _, pnl in sorted(by_day.items())]

    def _max_consecutive(self, trades: list[TradeRecord], winning: bool) -> int:
        """Count max consecutive wins or losses."""
        max_streak = 0
        current = 0
        for t in trades:
            if (winning and t.realized_pnl > 0) or (not winning and t.realized_pnl < 0):
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def _avg_hold_time(self, trades: list[TradeRecord]) -> float:
        """Average hold time in minutes."""
        durations = []
        for t in trades:
            if t.entry_time and t.exit_time:
                delta = t.exit_time - t.entry_time
                durations.append(delta.total_seconds() / 60.0)
        if not durations:
            return 0.0
        return float(np.mean(durations))
