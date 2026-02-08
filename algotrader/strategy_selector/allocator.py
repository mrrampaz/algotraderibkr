"""Distribute capital across strategies based on scores.

Allocation algorithm:
1. Filter to active strategies (score >= threshold)
2. Normalize scores to sum to 1.0
3. Multiply by deployable capital
4. Apply floor (min 3% if active) and ceiling (max from YAML config)
5. Redistribute any excess from ceiling caps
6. Scale down if total exceeds deployable capital
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
    """Distribute capital across strategies based on scores.

    Takes strategy scores and total available capital. Distributes capital
    proportionally to scores, respecting min/max bounds per strategy.
    """

    def __init__(
        self,
        total_capital: float,
        risk_config: RiskConfig,
        strategy_configs: dict[str, StrategyConfig],
        min_allocation_pct: float = 3.0,
    ) -> None:
        self._total_capital = total_capital
        self._risk_config = risk_config
        self._strategy_configs = strategy_configs
        self._min_allocation_pct = min_allocation_pct
        self._log = logger.bind(component="capital_allocator")

        self._log.info(
            "allocator_initialized",
            total_capital=total_capital,
            max_exposure_pct=risk_config.max_gross_exposure_pct,
            strategies=len(strategy_configs),
        )

    def allocate(
        self,
        scores: list[StrategyScore],
        current_equity: float | None = None,
    ) -> list[StrategyAllocation]:
        """Calculate capital allocations from scores."""
        total = current_equity or self._total_capital

        # 1. Deployable capital = total * max_gross_exposure_pct
        deployable = total * (self._risk_config.max_gross_exposure_pct / 100)

        # 2. Filter to active strategies only
        active = [s for s in scores if s.is_active]
        if not active:
            self._log.warning("no_active_strategies")
            return [
                StrategyAllocation(
                    strategy_name=s.strategy_name,
                    score=s.total_score,
                    allocated_capital=0.0,
                    allocation_pct=0.0,
                    is_active=False,
                    reason=f"Score {s.total_score:.2f} below threshold",
                )
                for s in scores
            ]

        # 3. Score-weighted allocation
        total_score = sum(s.total_score for s in active)
        if total_score == 0:
            self._log.warning("zero_total_score")
            return []

        raw_allocations: dict[str, float] = {}
        for s in active:
            weight = s.total_score / total_score
            raw_allocations[s.strategy_name] = deployable * weight

        # 4. Apply floor: minimum allocation if active
        min_capital = total * (self._min_allocation_pct / 100)
        for name in raw_allocations:
            if raw_allocations[name] < min_capital:
                raw_allocations[name] = min_capital

        # 5. Apply ceiling: max from each strategy's config
        excess = 0.0
        capped_names: set[str] = set()
        for name, capital in raw_allocations.items():
            config = self._strategy_configs.get(name, StrategyConfig())
            max_capital = total * (config.capital_allocation_pct / 100)
            if capital > max_capital:
                excess += capital - max_capital
                raw_allocations[name] = max_capital
                capped_names.add(name)

        # 6. Redistribute excess to uncapped strategies (proportionally)
        if excess > 0:
            uncapped = {n: c for n, c in raw_allocations.items() if n not in capped_names}
            uncapped_total = sum(uncapped.values())
            if uncapped_total > 0:
                for name in uncapped:
                    share = uncapped[name] / uncapped_total
                    raw_allocations[name] += excess * share

        # 7. Final check: total allocated should not exceed deployable capital
        total_allocated = sum(raw_allocations.values())
        if total_allocated > deployable:
            scale = deployable / total_allocated
            raw_allocations = {n: c * scale for n, c in raw_allocations.items()}

        # 8. Build result
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
        return f"Inactive (score={score.total_score:.2f})"

    parts = [f"base={score.base_weight:.2f}"]
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
