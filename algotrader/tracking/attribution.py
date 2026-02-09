"""Performance attribution — breaks down P&L by strategy, regime, session.

Runs post-market to answer:
- Which strategy made/lost money today?
- Did the regime classification help or hurt?
- Which time of day was most profitable?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pytz
import structlog

from algotrader.core.models import TradeRecord
from algotrader.tracking.journal import TradeJournal
from algotrader.tracking.metrics import MetricsCalculator

logger = structlog.get_logger()

ET = pytz.timezone("America/New_York")


@dataclass
class AttributionReport:
    """Daily attribution breakdown."""

    date: date
    total_pnl: float

    # By strategy
    by_strategy: dict[str, float] = field(default_factory=dict)  # {strategy_name: pnl}
    by_strategy_trades: dict[str, int] = field(default_factory=dict)  # {strategy_name: trade_count}

    # By regime
    by_regime: dict[str, float] = field(default_factory=dict)  # {regime_type: pnl}
    regime_accuracy: float = 0.0  # Was regime classification helpful?

    # By time of day
    by_session: dict[str, float] = field(default_factory=dict)  # {"morning": pnl, ...}

    # Best/worst
    best_trade: TradeRecord | None = None
    worst_trade: TradeRecord | None = None

    # Capital efficiency
    capital_deployed_avg: float = 0.0
    return_on_deployed: float = 0.0


class PerformanceAttribution:
    """Attribute P&L to strategies, regimes, and time periods."""

    def __init__(
        self,
        trade_journal: TradeJournal,
        metrics_calculator: MetricsCalculator,
    ) -> None:
        self._journal = trade_journal
        self._metrics = metrics_calculator
        self._log = logger.bind(component="attribution")

    def daily_report(self, target_date: date | None = None) -> AttributionReport:
        """Generate attribution report for a single day."""
        if target_date is None:
            target_date = datetime.now(pytz.UTC).date()

        trades = self._journal.get_trades(
            start_date=target_date,
            end_date=target_date,
            limit=10000,
        )

        total_pnl = sum(t.realized_pnl for t in trades)

        report = AttributionReport(
            date=target_date,
            total_pnl=total_pnl,
        )

        # By strategy
        for t in trades:
            report.by_strategy[t.strategy_name] = (
                report.by_strategy.get(t.strategy_name, 0.0) + t.realized_pnl
            )
            report.by_strategy_trades[t.strategy_name] = (
                report.by_strategy_trades.get(t.strategy_name, 0) + 1
            )

        # By regime
        for t in trades:
            regime_val = t.regime.value if t.regime else "unknown"
            report.by_regime[regime_val] = (
                report.by_regime.get(regime_val, 0.0) + t.realized_pnl
            )

        # By session
        for t in trades:
            session = self._classify_session(t.entry_time)
            report.by_session[session] = (
                report.by_session.get(session, 0.0) + t.realized_pnl
            )

        # Best/worst
        if trades:
            report.best_trade = max(trades, key=lambda t: t.realized_pnl)
            report.worst_trade = min(trades, key=lambda t: t.realized_pnl)

        # Capital efficiency
        if trades:
            deployed = [t.entry_price * t.qty for t in trades]
            report.capital_deployed_avg = sum(deployed) / len(deployed) if deployed else 0.0
            if report.capital_deployed_avg > 0:
                report.return_on_deployed = (total_pnl / report.capital_deployed_avg) * 100

        # Regime accuracy
        try:
            report.regime_accuracy = self.regime_accuracy_score(days=20)
        except Exception:
            self._log.debug("regime_accuracy_calculation_failed")

        self._log.info(
            "daily_attribution_generated",
            date=str(target_date),
            total_pnl=round(total_pnl, 2),
            strategies=len(report.by_strategy),
            trades=len(trades),
        )

        return report

    def weekly_report(self) -> dict[str, Any]:
        """Generate weekly summary with trends."""
        today = datetime.now(pytz.UTC).date()
        # Go back to last Monday
        days_since_monday = today.weekday()
        start = today - timedelta(days=days_since_monday)

        daily_reports = []
        for i in range(min(days_since_monday + 1, 5)):  # Up to 5 trading days
            d = start + timedelta(days=i)
            try:
                report = self.daily_report(d)
                if report.total_pnl != 0 or report.by_strategy:
                    daily_reports.append(report)
            except Exception:
                continue

        total_pnl = sum(r.total_pnl for r in daily_reports)
        trading_days = len(daily_reports)

        # Aggregate by strategy across the week
        weekly_by_strategy: dict[str, float] = {}
        weekly_by_strategy_trades: dict[str, int] = {}
        for r in daily_reports:
            for strat, pnl in r.by_strategy.items():
                weekly_by_strategy[strat] = weekly_by_strategy.get(strat, 0.0) + pnl
            for strat, count in r.by_strategy_trades.items():
                weekly_by_strategy_trades[strat] = weekly_by_strategy_trades.get(strat, 0) + count

        # Best/worst day
        best_day = max(daily_reports, key=lambda r: r.total_pnl) if daily_reports else None
        worst_day = min(daily_reports, key=lambda r: r.total_pnl) if daily_reports else None

        return {
            "week_start": str(start),
            "week_end": str(today),
            "trading_days": trading_days,
            "total_pnl": round(total_pnl, 2),
            "avg_daily_pnl": round(total_pnl / trading_days, 2) if trading_days > 0 else 0.0,
            "by_strategy": {k: round(v, 2) for k, v in weekly_by_strategy.items()},
            "by_strategy_trades": weekly_by_strategy_trades,
            "best_day": str(best_day.date) if best_day else None,
            "best_day_pnl": round(best_day.total_pnl, 2) if best_day else 0.0,
            "worst_day": str(worst_day.date) if worst_day else None,
            "worst_day_pnl": round(worst_day.total_pnl, 2) if worst_day else 0.0,
        }

    def regime_accuracy_score(self, days: int = 20) -> float:
        """How well did regime classification predict strategy performance?

        For each day, check: did the highest-weighted strategies for that
        day's regime actually outperform? Score 0-1.
        """
        today = datetime.now(pytz.UTC).date()
        day_scores = []

        for i in range(days):
            d = today - timedelta(days=i)
            trades = self._journal.get_trades(
                start_date=d,
                end_date=d,
                limit=10000,
            )
            if len(trades) < 3:
                continue

            # Get regimes present this day
            regimes = [t.regime.value for t in trades if t.regime]
            if not regimes:
                continue

            # Most common regime this day
            from collections import Counter
            regime_counts = Counter(regimes)
            dominant_regime = regime_counts.most_common(1)[0][0]

            # Get strategy P&L for this day
            strat_pnl: dict[str, float] = {}
            for t in trades:
                strat_pnl[t.strategy_name] = strat_pnl.get(t.strategy_name, 0.0) + t.realized_pnl

            if len(strat_pnl) < 2:
                continue

            # Top 3 by actual P&L
            actual_top = sorted(strat_pnl.keys(), key=lambda s: strat_pnl[s], reverse=True)[:3]

            # Top 3 by regime weight (load from regimes.yaml weights)
            try:
                from algotrader.core.config import load_yaml
                regime_config = load_yaml("config/regimes.yaml")
                weights = regime_config.get("regime_weights", {}).get(dominant_regime, {})
                if not weights:
                    continue
                # Only consider strategies that actually traded
                active_weights = {s: w for s, w in weights.items() if s in strat_pnl}
                predicted_top = sorted(active_weights.keys(), key=lambda s: active_weights[s], reverse=True)[:3]
            except Exception:
                continue

            # Score: overlap between predicted and actual top 3
            overlap = len(set(actual_top) & set(predicted_top))
            max_possible = min(3, len(strat_pnl))
            score = overlap / max_possible if max_possible > 0 else 0.0
            day_scores.append(score)

        if not day_scores:
            return 0.0

        return sum(day_scores) / len(day_scores)

    def _classify_session(self, trade_time: datetime | None) -> str:
        """Classify trade into morning/midday/afternoon session."""
        if trade_time is None:
            return "unknown"
        et = trade_time.astimezone(ET)
        hour = et.hour + et.minute / 60
        if hour < 11.0:
            return "morning"  # 9:30-11:00
        elif hour < 14.0:
            return "midday"  # 11:00-2:00
        else:
            return "afternoon"  # 2:00-4:00
