"""TTL-based in-memory cache for quotes and bars to reduce API calls."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import structlog

from algotrader.core.models import Quote, Snapshot

logger = structlog.get_logger()


class CacheEntry:
    """A cached value with expiration time."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl_seconds: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl_seconds

    @property
    def is_expired(self) -> bool:
        return time.monotonic() >= self.expires_at


class DataCache:
    """TTL-based cache for market data.

    Reduces redundant API calls when multiple strategies request the
    same data within a short window.
    """

    def __init__(
        self,
        quote_ttl: float = 30.0,
        bar_ttl: float = 60.0,
        snapshot_ttl: float = 30.0,
    ) -> None:
        self._quote_ttl = quote_ttl
        self._bar_ttl = bar_ttl
        self._snapshot_ttl = snapshot_ttl

        self._quotes: dict[str, CacheEntry] = {}
        self._bars: dict[str, CacheEntry] = {}
        self._snapshots: dict[str, CacheEntry] = {}

        self._log = logger.bind(component="data_cache")
        self._hits = 0
        self._misses = 0

    def _bar_key(self, symbol: str, timeframe: str, limit: int) -> str:
        return f"{symbol}:{timeframe}:{limit}"

    # ── Quotes ────────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Quote | None:
        entry = self._quotes.get(symbol)
        if entry and not entry.is_expired:
            self._hits += 1
            return entry.value
        self._misses += 1
        return None

    def set_quote(self, symbol: str, quote: Quote) -> None:
        self._quotes[symbol] = CacheEntry(quote, self._quote_ttl)

    # ── Bars ──────────────────────────────────────────────────────────

    def get_bars(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
        key = self._bar_key(symbol, timeframe, limit)
        entry = self._bars.get(key)
        if entry and not entry.is_expired:
            self._hits += 1
            return entry.value
        self._misses += 1
        return None

    def set_bars(self, symbol: str, timeframe: str, limit: int, df: pd.DataFrame) -> None:
        key = self._bar_key(symbol, timeframe, limit)
        self._bars[key] = CacheEntry(df, self._bar_ttl)

    # ── Snapshots ─────────────────────────────────────────────────────

    def get_snapshot(self, symbol: str) -> Snapshot | None:
        entry = self._snapshots.get(symbol)
        if entry and not entry.is_expired:
            self._hits += 1
            return entry.value
        self._misses += 1
        return None

    def set_snapshot(self, symbol: str, snapshot: Snapshot) -> None:
        self._snapshots[symbol] = CacheEntry(snapshot, self._snapshot_ttl)

    # ── Maintenance ───────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all cached data."""
        self._quotes.clear()
        self._bars.clear()
        self._snapshots.clear()

    def evict_expired(self) -> int:
        """Remove expired entries. Returns number of entries evicted."""
        evicted = 0
        for store in (self._quotes, self._bars, self._snapshots):
            expired_keys = [k for k, v in store.items() if v.is_expired]
            for key in expired_keys:
                del store[key]
                evicted += 1
        return evicted

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "quotes_cached": len(self._quotes),
            "bars_cached": len(self._bars),
            "snapshots_cached": len(self._snapshots),
        }
