"""TradeThesis: the output of the investigator pipeline.

Consumed by the orchestrator to decide whether to open a position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass
class NewsThesis:
    direction: Direction
    confidence: float
    key_catalysts: list[str] = field(default_factory=list)
    avg_sentiment: float = 0.0
    article_count: int = 0
    summary: str = ""


@dataclass
class AnnouncementsThesis:
    direction: Direction
    material_event_score: float
    recent_filings: list[str] = field(default_factory=list)
    earnings_within_days: int | None = None
    summary: str = ""


@dataclass
class MarketContext:
    beta_vs_spy: float
    spy_trend: str
    vix_level: float
    regime: str
    market_aligned: bool


@dataclass
class TechnicalContext:
    direction: Direction
    vwap: float
    rsi_14: float
    atr_14: float
    breakout_level_up: float
    breakdown_level_down: float
    current_price: float
    gap_pct: float


@dataclass
class TradeThesis:
    """Final consolidated thesis emitted by DecisionAgent."""

    symbol: str
    direction: Direction
    conviction: float
    timestamp: datetime
    entry_zone: tuple[float, float] | None = None
    stop_price: float | None = None
    target_price: float | None = None
    rationale: str = ""
    blackout_reason: str | None = None
    news: NewsThesis | None = None
    announcements: AnnouncementsThesis | None = None
    market: MarketContext | None = None
    technicals: TechnicalContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return (
            self.direction != Direction.NONE
            and self.blackout_reason is None
            and self.conviction > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        def _enum(v):
            return v.value if isinstance(v, Direction) else v

        def _dc(obj):
            if obj is None:
                return None
            from dataclasses import asdict
            d = asdict(obj)
            for k, v in list(d.items()):
                if isinstance(v, Direction):
                    d[k] = v.value
            return d

        return {
            "symbol": self.symbol,
            "direction": _enum(self.direction),
            "conviction": self.conviction,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "entry_zone": list(self.entry_zone) if self.entry_zone else None,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "rationale": self.rationale,
            "blackout_reason": self.blackout_reason,
            "news": _dc(self.news),
            "announcements": _dc(self.announcements),
            "market": _dc(self.market),
            "technicals": _dc(self.technicals),
            "metadata": self.metadata,
        }
