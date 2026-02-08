"""Alpaca IEX/SIP DataProvider implementation."""

from __future__ import annotations

import time
from datetime import datetime, date, timedelta

import pandas as pd
import pytz
import structlog
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from alpaca.common.exceptions import APIError

from algotrader.core.config import AlpacaConfig
from algotrader.core.models import Bar, Quote, Snapshot, MarketClock, NewsItem

logger = structlog.get_logger()

# Map string timeframes to Alpaca TimeFrame objects
TIMEFRAME_MAP = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "30Min": TimeFrame(30, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1Day": TimeFrame(1, TimeFrameUnit.Day),
}


class AlpacaDataProvider:
    """DataProvider implementation using Alpaca market data API.

    Handles the IEX (free) and SIP (paid) data feeds. Includes retry/backoff
    logic for API calls.
    """

    def __init__(self, config: AlpacaConfig, feed: str = "iex") -> None:
        self._config = config
        self._feed = feed
        self._log = logger.bind(component="alpaca_data", feed=feed)

        self._data_client = StockHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
        )
        self._trading_client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )

        self._log.info("alpaca_data_provider_initialized")

    def _retry(self, func, *args, max_retries: int = 3, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except APIError as e:
                if attempt == max_retries - 1:
                    self._log.error("api_call_failed", error=str(e), attempts=max_retries)
                    raise
                wait = 2 ** attempt
                self._log.warning("api_retry", attempt=attempt + 1, wait_seconds=wait, error=str(e))
                time.sleep(wait)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                self._log.warning("api_retry", attempt=attempt + 1, wait_seconds=wait, error=str(e))
                time.sleep(wait)

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "5Min",
        limit: int = 100,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV bars. Handles Alpaca's bar limit quirk (returns first N)."""
        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}. Use one of {list(TIMEFRAME_MAP.keys())}")

        # If no start specified, calculate one to get the LAST N bars
        # Alpaca returns FIRST N bars, so we need to set start appropriately
        if start is None:
            now = datetime.now(pytz.UTC)
            if timeframe == "1Day":
                start = now - timedelta(days=limit * 2)  # extra buffer for weekends
            elif timeframe == "1Hour":
                start = now - timedelta(hours=limit * 2)
            else:
                # For minute bars, account for market hours only (~6.5h/day)
                minutes_per_bar = int(timeframe.replace("Min", ""))
                trading_minutes_needed = limit * minutes_per_bar
                trading_days_needed = (trading_minutes_needed / 390) + 2  # 390 min/day
                start = now - timedelta(days=max(trading_days_needed * 1.5, 3))

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=None,  # Don't limit at API level — we'll tail ourselves
            feed=self._feed,
        )

        def _fetch():
            bars = self._data_client.get_stock_bars(request)
            return bars

        result = self._retry(_fetch)

        # Convert to DataFrame
        if not result or symbol not in result:
            self._log.warning("no_bars_returned", symbol=symbol, timeframe=timeframe)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap", "trade_count"])

        df = result[symbol].df if hasattr(result[symbol], 'df') else result.df

        # Handle multi-index if present
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        # Return last N bars (Alpaca quirk: returns first N, not last)
        df = df.tail(limit)
        return df

    def get_quote(self, symbol: str) -> Quote | None:
        """Get the latest bid/ask quote."""
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._feed)

        def _fetch():
            return self._data_client.get_stock_latest_quote(request)

        try:
            result = self._retry(_fetch)
            if not result or symbol not in result:
                return None

            q = result[symbol]
            return Quote(
                symbol=symbol,
                timestamp=q.timestamp,
                bid_price=float(q.bid_price),
                bid_size=float(q.bid_size),
                ask_price=float(q.ask_price),
                ask_size=float(q.ask_size),
            )
        except Exception:
            self._log.exception("get_quote_failed", symbol=symbol)
            return None

    def get_snapshot(self, symbol: str) -> Snapshot | None:
        """Get a full snapshot for a symbol."""
        request = StockSnapshotRequest(symbol_or_symbols=symbol, feed=self._feed)

        def _fetch():
            return self._data_client.get_stock_snapshot(request)

        try:
            result = self._retry(_fetch)
            if not result:
                return None

            snap = result if not isinstance(result, dict) else result.get(symbol)
            if snap is None:
                return None

            snapshot = Snapshot(symbol=symbol)

            if snap.latest_trade:
                snapshot.latest_trade_price = float(snap.latest_trade.price)
                snapshot.latest_trade_timestamp = snap.latest_trade.timestamp

            if snap.latest_quote:
                snapshot.latest_quote = Quote(
                    symbol=symbol,
                    timestamp=snap.latest_quote.timestamp,
                    bid_price=float(snap.latest_quote.bid_price),
                    bid_size=float(snap.latest_quote.bid_size),
                    ask_price=float(snap.latest_quote.ask_price),
                    ask_size=float(snap.latest_quote.ask_size),
                )

            if snap.minute_bar:
                snapshot.minute_bar = Bar(
                    symbol=symbol,
                    timestamp=snap.minute_bar.timestamp,
                    open=float(snap.minute_bar.open),
                    high=float(snap.minute_bar.high),
                    low=float(snap.minute_bar.low),
                    close=float(snap.minute_bar.close),
                    volume=float(snap.minute_bar.volume),
                    vwap=float(snap.minute_bar.vwap) if snap.minute_bar.vwap else None,
                    trade_count=snap.minute_bar.trade_count,
                )

            if snap.daily_bar:
                snapshot.daily_bar = Bar(
                    symbol=symbol,
                    timestamp=snap.daily_bar.timestamp,
                    open=float(snap.daily_bar.open),
                    high=float(snap.daily_bar.high),
                    low=float(snap.daily_bar.low),
                    close=float(snap.daily_bar.close),
                    volume=float(snap.daily_bar.volume),
                    vwap=float(snap.daily_bar.vwap) if snap.daily_bar.vwap else None,
                    trade_count=snap.daily_bar.trade_count,
                )

            if snap.previous_daily_bar:
                snapshot.prev_daily_bar = Bar(
                    symbol=symbol,
                    timestamp=snap.previous_daily_bar.timestamp,
                    open=float(snap.previous_daily_bar.open),
                    high=float(snap.previous_daily_bar.high),
                    low=float(snap.previous_daily_bar.low),
                    close=float(snap.previous_daily_bar.close),
                    volume=float(snap.previous_daily_bar.volume),
                    vwap=float(snap.previous_daily_bar.vwap) if snap.previous_daily_bar.vwap else None,
                    trade_count=snap.previous_daily_bar.trade_count,
                )

            return snapshot
        except Exception:
            self._log.exception("get_snapshot_failed", symbol=symbol)
            return None

    def get_snapshots(self, symbols: list[str]) -> dict[str, Snapshot]:
        """Get snapshots for multiple symbols in one call."""
        request = StockSnapshotRequest(symbol_or_symbols=symbols, feed=self._feed)

        def _fetch():
            return self._data_client.get_stock_snapshot(request)

        try:
            result = self._retry(_fetch)
            if not result:
                return {}

            snapshots: dict[str, Snapshot] = {}
            items = result.items() if isinstance(result, dict) else [(symbols[0], result)]

            for sym, snap in items:
                snapshot = Snapshot(symbol=sym)

                if snap.latest_trade:
                    snapshot.latest_trade_price = float(snap.latest_trade.price)
                    snapshot.latest_trade_timestamp = snap.latest_trade.timestamp

                if snap.latest_quote:
                    snapshot.latest_quote = Quote(
                        symbol=sym,
                        timestamp=snap.latest_quote.timestamp,
                        bid_price=float(snap.latest_quote.bid_price),
                        bid_size=float(snap.latest_quote.bid_size),
                        ask_price=float(snap.latest_quote.ask_price),
                        ask_size=float(snap.latest_quote.ask_size),
                    )

                if snap.daily_bar:
                    snapshot.daily_bar = Bar(
                        symbol=sym,
                        timestamp=snap.daily_bar.timestamp,
                        open=float(snap.daily_bar.open),
                        high=float(snap.daily_bar.high),
                        low=float(snap.daily_bar.low),
                        close=float(snap.daily_bar.close),
                        volume=float(snap.daily_bar.volume),
                        vwap=float(snap.daily_bar.vwap) if snap.daily_bar.vwap else None,
                    )

                if snap.previous_daily_bar:
                    snapshot.prev_daily_bar = Bar(
                        symbol=sym,
                        timestamp=snap.previous_daily_bar.timestamp,
                        open=float(snap.previous_daily_bar.open),
                        high=float(snap.previous_daily_bar.high),
                        low=float(snap.previous_daily_bar.low),
                        close=float(snap.previous_daily_bar.close),
                        volume=float(snap.previous_daily_bar.volume),
                        vwap=float(snap.previous_daily_bar.vwap) if snap.previous_daily_bar.vwap else None,
                    )

                snapshots[sym] = snapshot

            return snapshots
        except Exception:
            self._log.exception("get_snapshots_failed", symbols=symbols)
            return {}

    def get_option_chain(
        self,
        underlying: str,
        expiration: date | None = None,
    ) -> pd.DataFrame:
        """Get option chain. Note: IEX options data is 15-min delayed."""
        self._log.warning("option_chain_iex_delay", underlying=underlying)
        # Alpaca options data via alpaca-py — implementation depends on API version
        # For now, return empty DataFrame as placeholder
        return pd.DataFrame()

    def get_news(
        self,
        symbols: list[str] | None = None,
        limit: int = 50,
    ) -> list[NewsItem]:
        """Get news from Alpaca news API."""
        try:
            from alpaca.data.requests import NewsRequest

            request_params = {"limit": limit}
            if symbols:
                request_params["symbols"] = symbols

            request = NewsRequest(**request_params)
            news = self._data_client.get_news(request)

            items = []
            for article in news.news:
                items.append(NewsItem(
                    id=str(article.id),
                    headline=article.headline,
                    summary=article.summary or "",
                    source=article.source or "",
                    url=article.url or "",
                    symbols=article.symbols or [],
                    timestamp=article.created_at,
                ))
            return items
        except Exception:
            self._log.exception("get_news_failed")
            return []

    def is_market_open(self) -> bool:
        """Check if the stock market is currently open."""
        try:
            clock = self._trading_client.get_clock()
            return clock.is_open
        except Exception:
            self._log.exception("is_market_open_failed")
            return False

    def get_clock(self) -> MarketClock:
        """Get market clock with open/close times."""
        clock = self._trading_client.get_clock()
        return MarketClock(
            timestamp=clock.timestamp,
            is_open=clock.is_open,
            next_open=clock.next_open,
            next_close=clock.next_close,
        )
