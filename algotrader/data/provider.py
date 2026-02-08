"""DataProvider protocol — abstract interface for market data."""

from __future__ import annotations

from datetime import datetime, date
from typing import Protocol, runtime_checkable

import pandas as pd

from algotrader.core.models import Bar, Quote, Snapshot, MarketClock, NewsItem


@runtime_checkable
class DataProvider(Protocol):
    """Abstract data provider interface.

    Swap implementations (Alpaca IEX, Alpaca SIP, IBKR) without changing
    strategy code. All methods should handle retries internally.
    """

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV bars for a symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL")
            timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")
            limit: Max number of bars to return
            start: Start time (UTC). If None, returns most recent bars.
            end: End time (UTC).

        Returns:
            DataFrame with columns: open, high, low, close, volume, vwap, trade_count
            Index is datetime (UTC). Sorted ascending (oldest first).

        Note:
            Alpaca returns the FIRST N bars, not the last. Implementations
            must handle this (e.g. use .tail(limit) or pass start datetime).
        """
        ...

    def get_quote(self, symbol: str) -> Quote | None:
        """Get the latest bid/ask quote for a symbol."""
        ...

    def get_snapshot(self, symbol: str) -> Snapshot | None:
        """Get a full snapshot (latest trade, quote, minute bar, daily bar)."""
        ...

    def get_snapshots(self, symbols: list[str]) -> dict[str, Snapshot]:
        """Get snapshots for multiple symbols in one call."""
        ...

    def get_option_chain(
        self,
        underlying: str,
        expiration: date | None = None,
    ) -> pd.DataFrame:
        """Get option chain for an underlying symbol.

        Returns DataFrame with columns: symbol, expiration, strike, type (call/put),
        bid, ask, last, volume, open_interest, implied_vol, delta, gamma, theta, vega
        """
        ...

    def get_news(
        self,
        symbols: list[str] | None = None,
        limit: int = 50,
    ) -> list[NewsItem]:
        """Get news articles, optionally filtered by symbols."""
        ...

    def is_market_open(self) -> bool:
        """Check if the stock market is currently open."""
        ...

    def get_clock(self) -> MarketClock:
        """Get market clock with open/close times."""
        ...
