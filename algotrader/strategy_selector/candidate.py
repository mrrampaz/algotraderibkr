"""Trade candidate model used by the Daily Brain decision engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CandidateType(str, Enum):
    """Type of trade candidate."""

    LONG_EQUITY = "long_equity"
    SHORT_EQUITY = "short_equity"
    PAIRS = "pairs"
    CREDIT_SPREAD = "credit_spread"
    IRON_CONDOR = "iron_condor"
    SECTOR_LONG_SHORT = "sector_long_short"
    EVENT_DIRECTIONAL = "event_directional"


@dataclass
class TradeCandidate:
    """A specific, actionable trade idea produced by a strategy."""

    # Identity
    strategy_name: str
    candidate_type: CandidateType
    symbol: str

    # Trade parameters
    direction: str
    entry_price: float
    stop_price: float
    target_price: float

    # Sizing guidance
    risk_dollars: float
    suggested_qty: int = 0

    # Quality metrics
    risk_reward_ratio: float = 0.0
    confidence: float = 0.0
    edge_estimate_pct: float = 0.0

    # Context
    regime_fit: float = 0.0
    catalyst: str = ""
    time_horizon_minutes: int = 0
    expiry_time: datetime | None = None

    # Options fields
    options_structure: str = ""
    short_strike: float = 0.0
    long_strike: float = 0.0
    contracts: int = 0
    credit_received: float = 0.0
    max_loss: float = 0.0

    # Pair-specific fields
    symbol_b: str = ""
    hedge_ratio: float = 0.0
    z_score: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expected_value(self) -> float:
        """Simple expected value proxy."""
        return self.confidence * self.edge_estimate_pct

    @property
    def is_options(self) -> bool:
        return self.candidate_type in (
            CandidateType.CREDIT_SPREAD,
            CandidateType.IRON_CONDOR,
        )

    @property
    def is_expired(self) -> bool:
        if self.expiry_time is None:
            return False
        return datetime.utcnow() > self.expiry_time
