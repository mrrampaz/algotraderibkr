"""Distribute capital across strategies based on scores.

Allocation algorithm:
1. Filter to active strategies (score >= threshold)
2. Cash threshold: sit in cash if best score too low
3. Power-law weighting: score ** concentration_power
4. Cap single strategy at max_single_strategy_pct
5. Apply floor (min 3% if active)
6. Scale down if total exceeds deployable capital (80%)
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from algotrader.core.config import RiskConfig, StrategyConfig
from algotrader.strategies.base import StrategyBase
from algotrader.strategy_selector.scorer import StrategyScore

logger = structlog.get_logger()


@dataclass
class StrategyAllocation:
    """Capital allocation for a single strategy."""

    strategy_name: str
    score: float
    allocated_capital: float
    allocation_pct: float       # % of total capital
    is_active: bool
    reason: str                 # Why this allocation (for logging)


class CapitalAllocator:
    """Distribute capital across strategies using power-law concentration.

    Takes strategy scores and total available capital. Concentrates capital
    toward the highest-scoring strategies using power-law weighting.
    """

    def __init__(
        self,
        total_capital: float,
        risk_config: RiskConfig,
        strategy_configs: dict[str, StrategyConfig],
        min_allocation_pct: float = 3.0,
        cash_threshold: float = 0.25,
        max_single_strategy_pct: float = 70.0,
        concentration_power: float = 2.0,
    ) -> None:
        self._total_capital = total_capital
        self._risk_config = risk_config
        self._strategy_configs = strategy_configs
        self._min_allocation_pct = min_allocation_pct
        self._cash_threshold = cash_threshold
        self._max_single_strategy_pct = max_single_strategy_pct
        self._concentration_power = concentration_power
        self._log = logger.bind(component="capital_allocator")

        self._log.info(
            "allocator_initialized",
            total_capital=total_capital,
            max_exposure_pct=risk_config.max_gross_exposure_pct,
            cash_threshold=cash_threshold,
            concentration_power=concentration_power,
            max_single_pct=max_single_strategy_pct,
        )

    def allocate(
        self,
        scores: list[StrategyScore],
        current_equity: float | None = None,
    ) -> list[StrategyAllocation]:
        """Calculate capital allocations using power-law concentration."""
        total = current_equity or self._total_capital

        # 1. Deployable capital = total * max_gross_exposure_pct
        deployable = total * (self._risk_config.max_gross_exposure_pct / 100)

        # 2. Filter to active strategies
        active = [s for s in scores if s.is_active]

        # 3. Cash threshold: sit in cash if best score too low
        best_score = max((s.total_score for s in scores), default=0.0)
        if not active or best_score < self._cash_threshold:
            self._log.info(
                "sitting_in_cash",
                best_score=round(best_score, 3),
                threshold=self._cash_threshold,
                reason="no active strategies" if not active else "best score below cash threshold",
            )
            return [
                StrategyAllocation(
                    strategy_name=s.strategy_name,
                    score=s.total_score,
                    allocated_capital=0.0,
                    allocation_pct=0.0,
                    is_active=False,
                    reason=f"Cash: score {s.total_score:.2f}, threshold {self._cash_threshold}",
                )
                for s in scores
            ]

        # 4. Power-law weighting: score ** concentration_power
        powered: dict[str, float] = {}
        for s in active:
            powered[s.strategy_name] = s.total_score ** self._concentration_power
        total_powered = sum(powered.values())
        if total_powered == 0:
            return []

        raw_allocations: dict[str, float] = {}
        for name, pw in powered.items():
            raw_allocations[name] = deployable * (pw / total_powered)

        # 5. Apply floor: minimum allocation if active
        min_capital = total * (self._min_allocation_pct / 100)
        for name in raw_allocations:
            if raw_allocations[name] < min_capital:
                raw_allocations[name] = min_capital

        # 6. Cap single strategy at max_single_strategy_pct
        max_single = total * (self._max_single_strategy_pct / 100)
        excess = 0.0
        capped_names: set[str] = set()
        for name, capital in raw_allocations.items():
            if capital > max_single:
                excess += capital - max_single
                raw_allocations[name] = max_single
                capped_names.add(name)

        # 7. Redistribute excess to uncapped strategies proportionally
        if excess > 0:
            uncapped = {n: c for n, c in raw_allocations.items() if n not in capped_names}
            uncapped_total = sum(uncapped.values())
            if uncapped_total > 0:
                for name in uncapped:
                    share = uncapped[name] / uncapped_total
                    raw_allocations[name] += excess * share

        # 8. Final check: total must not exceed deployable capital
        total_allocated = sum(raw_allocations.values())
        if total_allocated > deployable:
            scale = deployable / total_allocated
            raw_allocations = {n: c * scale for n, c in raw_allocations.items()}

        # 9. Build result
        results = []
        for s in scores:
            capital = raw_allocations.get(s.strategy_name, 0.0)
            results.append(StrategyAllocation(
                strategy_name=s.strategy_name,
                score=s.total_score,
                allocated_capital=round(capital, 2),
                allocation_pct=round(capital / total * 100, 2) if total > 0 else 0,
                is_active=s.is_active and capital > 0,
                reason=_build_reason(s, capital, total),
            ))

        active_allocs = [a for a in results if a.is_active]
        self._log.info(
            "allocation_complete",
            active=len(active_allocs),
            total_deployed=round(sum(a.allocated_capital for a in active_allocs), 2),
            deployable=round(deployable, 2),
        )

        return results

    def apply_allocations(
        self,
        allocations: list[StrategyAllocation],
        strategies: dict[str, StrategyBase],
    ) -> None:
        """Push allocations to strategy instances via set_capital()."""
        for alloc in allocations:
            strategy = strategies.get(alloc.strategy_name)
            if not strategy:
                continue

            if alloc.is_active:
                strategy.set_capital(alloc.allocated_capital)
                if not strategy.is_enabled:
                    strategy.enable()
            else:
                # Don't immediately disable — let existing positions wind down
                # Just set capital to 0 so no new positions open
                strategy.set_capital(0.0)

        self._log.info(
            "allocations_applied",
            active=[a.strategy_name for a in allocations if a.is_active],
            inactive=[a.strategy_name for a in allocations if not a.is_active],
        )


def _build_reason(score: StrategyScore, capital: float, total: float) -> str:
    """Build a human-readable reason string for this allocation."""
    if not score.is_active:
        return f"Inactive (score={score.total_score:.2f}, opp={score.opportunity_score:.2f})"

    parts = [f"base={score.base_weight:.2f}", f"opp={score.opportunity_score:.2f}"]
    if score.vix_modifier != 0:
        parts.append(f"vix={score.vix_modifier:+.2f}")
    if score.performance_modifier != 0:
        parts.append(f"perf={score.performance_modifier:+.2f}")
    if score.time_modifier != 0:
        parts.append(f"time={score.time_modifier:+.2f}")
    if score.event_modifier != 0:
        parts.append(f"event={score.event_modifier:+.2f}")

    pct = capital / total * 100 if total > 0 else 0
    return f"Score {score.total_score:.2f} ({', '.join(parts)}) → {pct:.1f}%"
