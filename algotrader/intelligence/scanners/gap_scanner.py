"""Pre-market gap detection scanner.

Scans for stocks gapping >2% with volume >500K using Alpaca snapshots API.
Classifies gaps as "gap_up" or "gap_down" and ranks by gap percentage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pytz
import structlog

from algotrader.data.provider import DataProvider

logger = structlog.get_logger()


@dataclass
class GapResult:
    """A detected gap for a single symbol."""
    symbol: str
    gap_pct: float          # Positive = gap up, negative = gap down
    prev_close: float
    current_price: float
    pre_market_volume: float
    direction: str          # "gap_up" or "gap_down"
    timestamp: datetime | None = None


class GapScanner:
    """Pre-market gap scanner.

    Scans a universe of symbols for stocks gapping >threshold% from
    prior close with sufficient pre-market volume.

    Intended to run during pre-market (6:00-9:25 AM ET) to identify
    gap-and-go and gap-fade opportunities.
    """

    def __init__(
        self,
        data_provider: DataProvider,
        min_gap_pct: float = 2.0,
        min_volume: float = 500_000,
    ) -> None:
        self._data = data_provider
        self._min_gap_pct = min_gap_pct
        self._min_volume = min_volume
        self._log = logger.bind(component="gap_scanner")

        # Default scan universe: large-cap + liquid ETFs
        self._universe: list[str] = [
            # Major ETFs
            "SPY", "QQQ", "IWM", "DIA",
            # Mega caps
            "AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "TSLA",
            "JPM", "BAC", "GS", "V", "MA",
            "JNJ", "PFE", "UNH", "MRK", "ABBV",
            "XOM", "CVX", "COP",
            "HD", "LOW", "WMT", "COST", "TGT",
            "CAT", "DE", "BA", "GE",
            "T", "VZ", "NFLX", "DIS",
            # Sector ETFs
            "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "XLY",
        ]

    @property
    def universe(self) -> list[str]:
        return self._universe

    @universe.setter
    def universe(self, symbols: list[str]) -> None:
        self._universe = symbols

    def scan(self, symbols: list[str] | None = None) -> list[GapResult]:
        """Scan for gaps in the given symbols (or default universe).

        Returns list of GapResult sorted by absolute gap percentage (largest first).
        """
        scan_symbols = symbols or self._universe
        self._log.info("gap_scan_starting", num_symbols=len(scan_symbols))

        # Batch fetch snapshots
        snapshots = self._data.get_snapshots(scan_symbols)
        if not snapshots:
            self._log.warning("no_snapshots_returned")
            return []

        gaps: list[GapResult] = []
        now = datetime.now(pytz.UTC)

        for symbol, snap in snapshots.items():
            try:
                gap = self._check_gap(symbol, snap, now)
                if gap:
                    gaps.append(gap)
            except Exception:
                self._log.debug("gap_check_failed", symbol=symbol)

        # Sort by absolute gap percentage (largest first)
        gaps.sort(key=lambda g: abs(g.gap_pct), reverse=True)

        self._log.info(
            "gap_scan_complete",
            gappers=len(gaps),
            scanned=len(scan_symbols),
        )

        for gap in gaps[:10]:  # Log top 10
            self._log.info(
                "gap_detected",
                symbol=gap.symbol,
                gap_pct=round(gap.gap_pct, 2),
                direction=gap.direction,
                volume=gap.pre_market_volume,
            )

        return gaps

    def _check_gap(self, symbol: str, snap, now: datetime) -> GapResult | None:
        """Check if a single snapshot qualifies as a gap."""
        from algotrader.core.models import Snapshot

        if not isinstance(snap, Snapshot):
            return None

        # Need previous daily bar for prior close
        if not snap.prev_daily_bar:
            return None

        prev_close = snap.prev_daily_bar.close
        if prev_close <= 0:
            return None

        # Get current price — prefer latest trade, fall back to daily bar open
        current_price = None
        if snap.latest_trade_price and snap.latest_trade_price > 0:
            current_price = snap.latest_trade_price
        elif snap.daily_bar and snap.daily_bar.open > 0:
            current_price = snap.daily_bar.open

        if current_price is None:
            return None

        # Calculate gap
        gap_pct = ((current_price - prev_close) / prev_close) * 100

        if abs(gap_pct) < self._min_gap_pct:
            return None

        # Check volume — use daily bar volume if available
        volume = 0.0
        if snap.daily_bar and snap.daily_bar.volume:
            volume = snap.daily_bar.volume

        if volume < self._min_volume:
            return None

        direction = "gap_up" if gap_pct > 0 else "gap_down"

        return GapResult(
            symbol=symbol,
            gap_pct=round(gap_pct, 2),
            prev_close=prev_close,
            current_price=current_price,
            pre_market_volume=volume,
            direction=direction,
            timestamp=now,
        )
