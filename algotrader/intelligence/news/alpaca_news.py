"""Alpaca news API client.

Pulls news feed from Alpaca, categorizes by symbol and sector,
and provides simple sentiment scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytz
import structlog

from algotrader.core.models import NewsItem
from algotrader.data.provider import DataProvider

logger = structlog.get_logger()

# Sector mapping for common symbols
SECTOR_MAP: dict[str, str] = {
    "AAPL": "tech", "MSFT": "tech", "GOOG": "tech", "GOOGL": "tech",
    "META": "tech", "AMZN": "tech", "NVDA": "tech", "TSLA": "tech",
    "NFLX": "tech", "CRM": "tech", "ADBE": "tech", "ORCL": "tech",
    "JPM": "financials", "BAC": "financials", "GS": "financials",
    "MS": "financials", "V": "financials", "MA": "financials",
    "WFC": "financials", "C": "financials",
    "XOM": "energy", "CVX": "energy", "COP": "energy", "EOG": "energy",
    "SLB": "energy",
    "JNJ": "healthcare", "PFE": "healthcare", "UNH": "healthcare",
    "MRK": "healthcare", "ABBV": "healthcare", "LLY": "healthcare",
    "HD": "consumer", "LOW": "consumer", "WMT": "consumer",
    "COST": "consumer", "TGT": "consumer", "KO": "consumer",
    "PEP": "consumer", "PG": "consumer", "MCD": "consumer",
    "CAT": "industrials", "DE": "industrials", "BA": "industrials",
    "GE": "industrials", "UPS": "industrials", "FDX": "industrials",
    "T": "telecom", "VZ": "telecom", "TMUS": "telecom",
    "DUK": "utilities", "SO": "utilities", "NEE": "utilities",
    "PLD": "reits", "AMT": "reits",
    "BHP": "materials", "RIO": "materials", "FCX": "materials",
    # Sector ETFs
    "XLK": "tech", "XLF": "financials", "XLE": "energy",
    "XLV": "healthcare", "XLI": "industrials", "XLP": "consumer_staples",
    "XLU": "utilities", "XLB": "materials", "XLRE": "reits",
    "XLC": "communication", "XLY": "consumer_discretionary",
    "SPY": "index", "QQQ": "index", "IWM": "index", "DIA": "index",
}

# Sentiment keywords
BULLISH_KEYWORDS = [
    "beats", "exceeds", "surges", "rally", "upgrade", "strong",
    "record", "growth", "bullish", "outperform", "raised",
    "positive", "jumps", "soars", "buy", "breakout", "accelerat",
]
BEARISH_KEYWORDS = [
    "misses", "below", "plunges", "sell-off", "downgrade", "weak",
    "decline", "loss", "bearish", "underperform", "cut", "lowered",
    "negative", "drops", "falls", "crash", "warning", "layoff",
    "recall", "lawsuit", "investigation", "fraud",
]


@dataclass
class CategorizedNews:
    """News categorized by symbol and sector."""
    items: list[NewsItem] = field(default_factory=list)
    by_symbol: dict[str, list[NewsItem]] = field(default_factory=dict)
    by_sector: dict[str, list[NewsItem]] = field(default_factory=dict)
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0


class AlpacaNewsClient:
    """Client for fetching and categorizing Alpaca news.

    Fetches news from the Alpaca news API, scores sentiment using
    keyword matching, and groups by symbol/sector.
    """

    def __init__(self, data_provider: DataProvider) -> None:
        self._data = data_provider
        self._log = logger.bind(component="alpaca_news")

    def fetch_news(
        self,
        symbols: list[str] | None = None,
        limit: int = 50,
    ) -> CategorizedNews:
        """Fetch and categorize recent news.

        Args:
            symbols: Filter to specific symbols (None = all)
            limit: Max number of articles to fetch

        Returns:
            CategorizedNews with items grouped by symbol and sector
        """
        self._log.info("fetching_news", symbols=symbols, limit=limit)

        raw_items = self._data.get_news(symbols=symbols, limit=limit)
        if not raw_items:
            self._log.info("no_news_returned")
            return CategorizedNews()

        result = CategorizedNews()

        for item in raw_items:
            # Score sentiment
            item.sentiment_score = self._score_sentiment(item.headline, item.summary)

            result.items.append(item)

            if item.sentiment_score and item.sentiment_score > 0.2:
                result.bullish_count += 1
            elif item.sentiment_score and item.sentiment_score < -0.2:
                result.bearish_count += 1
            else:
                result.neutral_count += 1

            # Group by symbol
            for sym in item.symbols:
                result.by_symbol.setdefault(sym, []).append(item)

                # Group by sector
                sector = SECTOR_MAP.get(sym)
                if sector:
                    result.by_sector.setdefault(sector, []).append(item)

        self._log.info(
            "news_fetched",
            total=len(result.items),
            bullish=result.bullish_count,
            bearish=result.bearish_count,
            neutral=result.neutral_count,
            symbols_mentioned=len(result.by_symbol),
        )

        return result

    def get_symbol_sentiment(self, symbol: str, hours: int = 24) -> float:
        """Get aggregated sentiment score for a symbol over recent hours.

        Returns float in [-1, 1]: negative = bearish, positive = bullish.
        """
        news = self._data.get_news(symbols=[symbol], limit=20)
        if not news:
            return 0.0

        cutoff = datetime.now(pytz.UTC) - timedelta(hours=hours)
        scores: list[float] = []

        for item in news:
            if item.timestamp and item.timestamp < cutoff:
                continue
            score = self._score_sentiment(item.headline, item.summary)
            if score is not None:
                scores.append(score)

        if not scores:
            return 0.0

        return round(sum(scores) / len(scores), 3)

    def _score_sentiment(self, headline: str, summary: str = "") -> float:
        """Score sentiment using keyword matching.

        Returns float in [-1, 1]. Simple but fast — no LLM needed.
        """
        text = (headline + " " + summary).lower()

        bullish = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
        bearish = sum(1 for kw in BEARISH_KEYWORDS if kw in text)

        total = bullish + bearish
        if total == 0:
            return 0.0

        # Net score: +1 = all bullish, -1 = all bearish
        return round((bullish - bearish) / total, 3)
