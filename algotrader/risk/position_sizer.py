"""Risk-based position sizing with conviction multiplier."""

from __future__ import annotations

import structlog

from algotrader.core.config import RiskConfig

logger = structlog.get_logger()


class PositionSizer:
    """Calculate position sizes based on risk parameters.

    Rules:
    - Risk 0.25-0.5% of capital per trade
    - Max single position: 5% of capital
    - Conviction multiplier: 0.5x to 1.5x base size
    """

    def __init__(self, config: RiskConfig, total_capital: float) -> None:
        self._config = config
        self._total_capital = total_capital
        self._log = logger.bind(component="position_sizer")

        # Base risk per trade: 0.25% to 0.5%
        self._base_risk_pct = 0.0035  # 0.35% default (middle of range)
        self._min_risk_pct = 0.0025   # 0.25%
        self._max_risk_pct = 0.005    # 0.5%

    def update_capital(self, total_capital: float) -> None:
        """Update the total capital for sizing calculations."""
        self._total_capital = total_capital

    def calculate_shares(
        self,
        price: float,
        stop_distance: float,
        conviction: float = 0.5,
        reduce_for_drawdown: bool = False,
    ) -> int:
        """Calculate number of shares based on risk.

        Args:
            price: Entry price per share
            stop_distance: Absolute distance from entry to stop loss
            conviction: Signal conviction (0.0 to 1.0). Mapped to size
                multiplier via 0.5 + conviction (range 0.5x to 1.5x).
            reduce_for_drawdown: If True, reduce size by 50% (drawdown mode)

        Returns:
            Number of whole shares to trade (0 if can't meet minimums)
        """
        if price <= 0 or stop_distance <= 0:
            return 0

        # Map conviction (0.0-1.0) to size multiplier (0.5x-1.5x)
        conviction = max(0.0, min(1.0, conviction))
        conviction_mult = 0.5 + conviction

        # Dollar risk per trade
        risk_amount = self._total_capital * self._base_risk_pct * conviction_mult

        if reduce_for_drawdown:
            risk_amount *= 0.5

        # Shares from risk-based sizing
        shares = int(risk_amount / stop_distance)

        # Apply max single position limit
        max_position_value = self._total_capital * (self._config.max_single_position_pct / 100)
        max_shares_by_value = int(max_position_value / price)

        shares = min(shares, max_shares_by_value)

        if shares <= 0:
            self._log.debug(
                "position_size_zero",
                price=price,
                stop_distance=stop_distance,
                conviction=conviction,
                conviction_mult=conviction_mult,
            )
            return 0

        self._log.debug(
            "position_sized",
            shares=shares,
            price=price,
            stop_distance=stop_distance,
            conviction=conviction,
            conviction_mult=conviction_mult,
            risk_amount=risk_amount,
            position_value=shares * price,
        )

        return shares

    def calculate_shares_by_pct(
        self,
        price: float,
        allocation_pct: float,
        conviction: float = 0.5,
    ) -> int:
        """Calculate shares by percentage of capital allocation.

        Simpler sizing for strategies that don't use stop-based sizing
        (e.g., pairs trading where the hedge provides protection).

        Args:
            price: Entry price per share
            allocation_pct: Percentage of capital to allocate (e.g., 2.0 = 2%)
            conviction: Signal conviction (0.0 to 1.0). Mapped to size
                multiplier via 0.5 + conviction (range 0.5x to 1.5x).
        """
        if price <= 0:
            return 0

        conviction = max(0.0, min(1.0, conviction))
        conviction_mult = 0.5 + conviction
        target_value = self._total_capital * (allocation_pct / 100) * conviction_mult

        # Cap at max single position
        max_position_value = self._total_capital * (self._config.max_single_position_pct / 100)
        target_value = min(target_value, max_position_value)

        shares = int(target_value / price)
        return max(shares, 0)

    def max_position_value(self) -> float:
        """Get the maximum value for a single position."""
        return self._total_capital * (self._config.max_single_position_pct / 100)
