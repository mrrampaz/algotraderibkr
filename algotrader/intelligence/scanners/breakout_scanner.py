"""Breakout detection scanner.

Detects stocks breaking out of consolidation ranges on above-average volume.
Used by the Momentum/Breakout strategy to find entry candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pytz
import structlog

from algotrader.data.provider import DataProvider

logger = structlog.get_logger()


@dataclass
class BreakoutResult:
    """A detected breakout for a single symbol."""

    symbol: str
    breakout_type: str  # "resistance_break", "support_break", "range_expansion"
    breakout_price: float  # The level that was broken
    current_price: float
    volume_ratio: float  # Current vs 20d average
    consolidation_days: int  # How long the range held
    range_high: float
    range_low: float
    atr: float  # For stop placement
    timestamp: datetime | None = None


class BreakoutScanner:
    """Breakout scanner for consolidation range breaks.

    Identifies stocks that have been trading in a tight range and are
    now breaking out with above-average volume. Volume confirmation
    filters false breakouts.
    """

    def __init__(
        self,
        data_provider: DataProvider,
        min_volume_ratio: float = 1.5,
        max_range_pct: float = 5.0,
        min_consolidation_days: int = 3,
    ) -> None:
        self._data = data_provider
        self._min_volume_ratio = min_volume_ratio
        self._max_range_pct = max_range_pct
        self._min_consolidation_days = min_consolidation_days
        self._log = logger.bind(component="breakout_scanner")

        # Default scan universe
        self._universe: list[str] = [
            "SPY", "QQQ", "IWM", "DIA",
            "AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "TSLA",
            "JPM", "BAC", "GS", "V", "MA",
            "JNJ", "PFE", "UNH", "MRK", "ABBV",
            "XOM", "CVX", "COP",
            "HD", "LOW", "WMT", "COST", "TGT",
            "CAT", "DE", "BA", "GE",
            "T", "VZ", "NFLX", "DIS",
            "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "XLY",
        ]

    @property
    def universe(self) -> list[str]:
        return self._universe

    @universe.setter
    def universe(self, symbols: list[str]) -> None:
        self._universe = symbols

    def scan(self, symbols: list[str] | None = None) -> list[BreakoutResult]:
        """Scan for breakouts in the given symbols (or default universe).

        Returns list of BreakoutResult sorted by volume ratio (highest first).
        """
        scan_symbols = symbols or self._universe
        self._log.info("breakout_scan_starting", num_symbols=len(scan_symbols))

        results: list[BreakoutResult] = []
        now = datetime.now(pytz.UTC)

        for symbol in scan_symbols:
            try:
                result = self._check_breakout(symbol, now)
                if result:
                    results.append(result)
            except Exception:
                self._log.debug("breakout_check_failed", symbol=symbol)

        results.sort(key=lambda r: r.volume_ratio, reverse=True)

        self._log.info(
            "breakout_scan_complete",
            breakouts=len(results),
            scanned=len(scan_symbols),
        )

        for r in results[:10]:
            self._log.info(
                "breakout_detected",
                symbol=r.symbol,
                type=r.breakout_type,
                volume_ratio=round(r.volume_ratio, 1),
                consolidation_days=r.consolidation_days,
            )

        return results

    def _check_breakout(self, symbol: str, now: datetime) -> BreakoutResult | None:
        """Check if a symbol is breaking out of a consolidation range."""
        # Get 20 daily bars for consolidation detection + ATR
        bars = self._data.get_bars(symbol, "1Day", 25)
        if bars.empty or len(bars) < 15:
            return None

        highs = bars["high"].values
        lows = bars["low"].values
        closes = bars["close"].values
        volumes = bars["volume"].values

        # Calculate ATR(14)
        atr = self._calculate_atr(highs, lows, closes, period=14)
        if atr is None or atr <= 0:
            return None

        # Identify consolidation range from last 10 bars (excluding today)
        lookback = min(10, len(bars) - 1)
        range_highs = highs[-(lookback + 1):-1]
        range_lows = lows[-(lookback + 1):-1]
        range_closes = closes[-(lookback + 1):-1]

        if len(range_highs) < self._min_consolidation_days:
            return None

        range_high = float(np.max(range_highs))
        range_low = float(np.min(range_lows))
        avg_close = float(np.mean(range_closes))

        if avg_close <= 0:
            return None

        # Check if this qualifies as consolidation
        range_pct = ((range_high - range_low) / avg_close) * 100
        if range_pct > self._max_range_pct:
            return None  # Range too wide — not consolidating

        # Count consolidation days (how many bars stayed within range)
        consolidation_days = 0
        for i in range(len(range_highs) - 1, -1, -1):
            if range_highs[i] <= range_high and range_lows[i] >= range_low:
                consolidation_days += 1
            else:
                break

        if consolidation_days < self._min_consolidation_days:
            return None

        # Get current price from latest bar
        current_price = float(closes[-1])

        # Check volume confirmation
        avg_volume_20d = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
        current_volume = float(volumes[-1])

        if avg_volume_20d <= 0:
            return None

        volume_ratio = current_volume / avg_volume_20d
        if volume_ratio < self._min_volume_ratio:
            return None  # No volume confirmation

        # Check for breakout
        breakout_type = None
        breakout_price = 0.0

        if current_price > range_high:
            breakout_type = "resistance_break"
            breakout_price = range_high
        elif current_price < range_low:
            breakout_type = "support_break"
            breakout_price = range_low

        if breakout_type is None:
            return None

        return BreakoutResult(
            symbol=symbol,
            breakout_type=breakout_type,
            breakout_price=breakout_price,
            current_price=current_price,
            volume_ratio=round(volume_ratio, 2),
            consolidation_days=consolidation_days,
            range_high=range_high,
            range_low=range_low,
            atr=round(atr, 4),
            timestamp=now,
        )

    @staticmethod
    def _calculate_atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float | None:
        """Calculate Average True Range over the given period."""
        if len(highs) < period + 1:
            return None

        tr_values = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            tr_values.append(tr)

        if len(tr_values) < period:
            return None

        return float(np.mean(tr_values[-period:]))
