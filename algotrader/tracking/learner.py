"""Strategy weight learner — adjusts regime-strategy weights from actual results.

Conservative adjustments. The initial regime weights are based on sound trading
theory. The learner nudges weights, not overhauls them. Overfitting to recent
performance is the main risk.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytz
import yaml
import structlog

from algotrader.core.models import TradeRecord
from algotrader.tracking.journal import TradeJournal
from algotrader.tracking.metrics import MetricsCalculator

logger = structlog.get_logger()


@dataclass
class WeightAdjustment:
    """A proposed weight change."""

    regime: str
    strategy: str
    old_weight: float
    new_weight: float
    delta: float
    evidence_trades: int  # How many trades support this adjustment
    confidence: float  # 0-1, based on sample size


class StrategyWeightLearner:
    """Adjust regime-strategy weights from historical performance.

    Runs post-market (or weekly). Analyzes trade journal grouped by regime
    and strategy, then proposes weight adjustments.

    Safety constraints:
    - Minimum 20 trades per regime-strategy combo before adjusting
    - Max adjustment per cycle: +/-0.10
    - Weights always stay in [0.0, 1.0]
    - Changes are written to regimes.yaml (with backup)
    - Human review flag: if total weight change > 0.5, flag for review
    """

    def __init__(
        self,
        trade_journal: TradeJournal,
        metrics_calculator: MetricsCalculator,
        regime_config_path: str = "config/regimes.yaml",
        min_trades: int = 20,
        max_adjustment: float = 0.10,
        learning_rate: float = 0.3,
    ) -> None:
        self._journal = trade_journal
        self._metrics = metrics_calculator
        self._regime_config_path = regime_config_path
        self._min_trades = min_trades
        self._max_adjustment = max_adjustment
        self._learning_rate = learning_rate
        self._log = logger.bind(component="weight_learner")

    def analyze(self, days: int = 30) -> list[WeightAdjustment]:
        """Analyze recent trades and propose weight adjustments.

        Does NOT apply changes — returns proposals for review.
        """
        # Load current weights from regimes.yaml
        config_path = Path(self._regime_config_path)
        if not config_path.exists():
            self._log.warning("regime_config_not_found", path=str(config_path))
            return []

        with open(config_path) as f:
            config = yaml.safe_load(f)

        current_weights = config.get("regime_weights", {})
        if not current_weights:
            self._log.warning("no_regime_weights_in_config")
            return []

        # Get all trades from last N days
        end_date = datetime.now(pytz.UTC).date()
        start_date = end_date - timedelta(days=days)
        trades = self._journal.get_trades(
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        if not trades:
            self._log.info("no_trades_for_learning", days=days)
            return []

        # Group by (regime, strategy)
        groups: dict[tuple[str, str], list[TradeRecord]] = {}
        for trade in trades:
            if trade.regime is None:
                continue
            key = (trade.regime.value, trade.strategy_name)
            groups.setdefault(key, []).append(trade)

        adjustments = []
        for (regime, strategy), group_trades in groups.items():
            if len(group_trades) < self._min_trades:
                continue  # Not enough data

            # Calculate performance score for this regime-strategy combo
            win_rate = len([t for t in group_trades if t.realized_pnl > 0]) / len(group_trades)
            avg_pnl = float(np.mean([t.realized_pnl for t in group_trades]))
            profit_factor = self._calculate_profit_factor(group_trades)

            # Composite performance score (0-1)
            perf_score = (
                0.4 * win_rate
                + 0.3 * min(profit_factor / 3.0, 1.0)  # PF of 3+ -> max score
                + 0.3 * (1.0 if avg_pnl > 0 else 0.0)  # Binary: profitable or not
            )

            # Current weight
            current = current_weights.get(regime, {}).get(strategy, 0.5)

            # Target weight = blend of current weight and performance score
            target = current * (1 - self._learning_rate) + perf_score * self._learning_rate

            # Clamp adjustment
            delta = target - current
            delta = max(-self._max_adjustment, min(self._max_adjustment, delta))
            new_weight = max(0.0, min(1.0, current + delta))

            # Confidence based on sample size
            confidence = min(1.0, len(group_trades) / (self._min_trades * 3))

            if abs(delta) > 0.01:  # Only report meaningful changes
                adjustments.append(WeightAdjustment(
                    regime=regime,
                    strategy=strategy,
                    old_weight=round(current, 3),
                    new_weight=round(new_weight, 3),
                    delta=round(delta, 3),
                    evidence_trades=len(group_trades),
                    confidence=round(confidence, 2),
                ))

        self._log.info(
            "learning_analysis_complete",
            days=days,
            total_trades=len(trades),
            groups_analyzed=len(groups),
            adjustments_proposed=len(adjustments),
        )

        return adjustments

    def apply_adjustments(
        self,
        adjustments: list[WeightAdjustment],
        backup: bool = True,
    ) -> None:
        """Apply approved adjustments to regimes.yaml.

        Creates a timestamped backup before modifying.
        """
        if not adjustments:
            return

        config_path = Path(self._regime_config_path)

        # Backup
        if backup:
            backup_path = config_path.with_suffix(
                f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml.bak"
            )
            shutil.copy2(config_path, backup_path)
            self._log.info("regime_config_backed_up", path=str(backup_path))

        # Load current YAML
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Apply adjustments
        for adj in adjustments:
            if adj.regime in config.get("regime_weights", {}):
                config["regime_weights"][adj.regime][adj.strategy] = adj.new_weight

        # Write back
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self._log.info(
            "regime_weights_updated",
            adjustments=len(adjustments),
            total_delta=round(sum(abs(a.delta) for a in adjustments), 3),
        )

    def auto_learn(self, days: int = 30) -> list[WeightAdjustment]:
        """Analyze and auto-apply safe adjustments.

        Only applies adjustments where:
        - confidence >= 0.6
        - evidence_trades >= min_trades
        - abs(delta) <= max_adjustment

        Flags large total changes for human review.
        """
        adjustments = self.analyze(days=days)
        if not adjustments:
            return []

        # Filter to safe adjustments
        safe = [
            a for a in adjustments
            if a.confidence >= 0.6
            and a.evidence_trades >= self._min_trades
            and abs(a.delta) <= self._max_adjustment
        ]

        if not safe:
            self._log.info("no_safe_adjustments", total_proposed=len(adjustments))
            return []

        # Check if human review needed
        if self._needs_human_review(safe):
            self._log.warning(
                "auto_learn_skipped_large_change",
                adjustments=len(safe),
                message="Adjustments saved but not applied — review recommended",
            )
            # Still return them so orchestrator can log/alert
            return safe

        # Apply safe adjustments
        self.apply_adjustments(safe)

        self._log.info(
            "auto_learn_applied",
            adjustments=len(safe),
            details=[
                {"regime": a.regime, "strategy": a.strategy,
                 "delta": a.delta, "trades": a.evidence_trades}
                for a in safe
            ],
        )

        return safe

    def _needs_human_review(self, adjustments: list[WeightAdjustment]) -> bool:
        """Flag if total weight change is too large."""
        total_change = sum(abs(a.delta) for a in adjustments)
        if total_change > 0.5:
            self._log.warning(
                "large_weight_change_flagged",
                total_delta=round(total_change, 3),
                num_adjustments=len(adjustments),
                message="Review proposed changes before next trading day",
            )
            return True
        return False

    def _calculate_profit_factor(self, trades: list[TradeRecord]) -> float:
        gross_profit = sum(t.realized_pnl for t in trades if t.realized_pnl > 0)
        gross_loss = abs(sum(t.realized_pnl for t in trades if t.realized_pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
