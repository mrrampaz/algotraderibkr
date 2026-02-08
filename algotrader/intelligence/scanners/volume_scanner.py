"""Unusual volume detection scanner.

Detects stocks trading at >2x their 20-day average volume during market hours.
High unusual volume often signals institutional activity or news-driven moves.
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
class VolumeResult:
    """A detected unusual volume event."""
    symbol: str
    current_volume: float
    avg_volume_20d: float
    volume_ratio: float     # current / average (e.g. 3.2 = 3.2x normal)
    current_price: float
    price_change_pct: float  # Intraday % change
    timestamp: datetime | None = None


class VolumeScanner:
    """Unusual volume scanner.

    Detects stocks with volume significantly above their 20-day average.
    Runs during market hours to find intraday opportunities driven by
    institutional activity, news, or sector rotation.
    """

    def __init__(
        self,
        data_provider: DataProvider,
        min_volume_ratio: float = 2.0,
        min_avg_volume: float = 500_000,
    ) -> None:
        self._data = data_provider
        self._min_volume_ratio = min_volume_ratio
        self._min_avg_volume = min_avg_volume
        self._log = logger.bind(component="volume_scanner")

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

        # Cache average volumes to avoid repeated lookups
        self._avg_volumes: dict[str, float] = {}

    @property
    def universe(self) -> list[str]:
        return self._universe

    @universe.setter
    def universe(self, symbols: list[str]) -> None:
        self._universe = symbols
        self._avg_volumes.clear()

    def warm_up(self) -> None:
        """Pre-calculate 20-day average volumes for the universe.

        Call once during pre-market to avoid repeated API calls.
        """
        self._log.info("volume_scanner_warming_up", symbols=len(self._universe))

        for symbol in self._universe:
            try:
                bars = self._data.get_bars(symbol, "1Day", 25)
                if bars.empty or len(bars) < 10:
                    continue
                avg_vol = float(np.mean(bars["volume"].values[-20:]))
                self._avg_volumes[symbol] = avg_vol
            except Exception:
                self._log.debug("avg_volume_fetch_failed", symbol=symbol)

        self._log.info(
            "volume_scanner_warmed_up",
            cached=len(self._avg_volumes),
        )

    def scan(self, symbols: list[str] | None = None) -> list[VolumeResult]:
        """Scan for unusual volume.

        Returns list of VolumeResult sorted by volume ratio (highest first).
        """
        scan_symbols = symbols or self._universe
        self._log.info("volume_scan_starting", num_symbols=len(scan_symbols))

        # Get current snapshots
        snapshots = self._data.get_snapshots(scan_symbols)
        if not snapshots:
            self._log.warning("no_snapshots_returned")
            return []

        results: list[VolumeResult] = []
        now = datetime.now(pytz.UTC)

        for symbol, snap in snapshots.items():
            try:
                result = self._check_volume(symbol, snap, now)
                if result:
                    results.append(result)
            except Exception:
                self._log.debug("volume_check_failed", symbol=symbol)

        # Sort by volume ratio (highest first)
        results.sort(key=lambda r: r.volume_ratio, reverse=True)

        self._log.info(
            "volume_scan_complete",
            unusual=len(results),
            scanned=len(scan_symbols),
        )

        for r in results[:10]:
            self._log.info(
                "unusual_volume",
                symbol=r.symbol,
                ratio=round(r.volume_ratio, 1),
                volume=r.current_volume,
                price_change=round(r.price_change_pct, 2),
            )

        return results

    def _check_volume(self, symbol: str, snap, now: datetime) -> VolumeResult | None:
        """Check if a snapshot shows unusual volume."""
        from algotrader.core.models import Snapshot

        if not isinstance(snap, Snapshot):
            return None

        if not snap.daily_bar:
            return None

        current_volume = snap.daily_bar.volume
        if current_volume <= 0:
            return None

        # Get average volume (from cache or calculate)
        avg_volume = self._avg_volumes.get(symbol)
        if avg_volume is None:
            avg_volume = self._calculate_avg_volume(symbol)
            if avg_volume is not None:
                self._avg_volumes[symbol] = avg_volume

        if avg_volume is None or avg_volume < self._min_avg_volume:
            return None

        # Account for time of day — scale expected volume proportionally
        # Market is 6.5 hours; if it's only been 2 hours, expect ~30% of daily vol
        et_now = now.astimezone(pytz.timezone("America/New_York"))
        market_open_hour = 9.5  # 9:30 AM
        current_hour = et_now.hour + et_now.minute / 60.0
        hours_elapsed = max(0.0, current_hour - market_open_hour)
        hours_elapsed = min(hours_elapsed, 6.5)

        if hours_elapsed <= 0:
            # Pre-market — compare raw volumes
            expected_fraction = 1.0
        else:
            expected_fraction = hours_elapsed / 6.5

        expected_volume = avg_volume * expected_fraction
        if expected_volume <= 0:
            return None

        volume_ratio = current_volume / expected_volume

        if volume_ratio < self._min_volume_ratio:
            return None

        # Calculate intraday price change
        price_change_pct = 0.0
        current_price = snap.latest_trade_price or snap.daily_bar.close
        if snap.prev_daily_bar and snap.prev_daily_bar.close > 0:
            price_change_pct = ((current_price - snap.prev_daily_bar.close) / snap.prev_daily_bar.close) * 100

        return VolumeResult(
            symbol=symbol,
            current_volume=current_volume,
            avg_volume_20d=avg_volume,
            volume_ratio=round(volume_ratio, 2),
            current_price=current_price,
            price_change_pct=round(price_change_pct, 2),
            timestamp=now,
        )

    def _calculate_avg_volume(self, symbol: str) -> float | None:
        """Calculate 20-day average volume from historical bars."""
        try:
            bars = self._data.get_bars(symbol, "1Day", 25)
            if bars.empty or len(bars) < 10:
                return None
            return float(np.mean(bars["volume"].values[-20:]))
        except Exception:
            return None
