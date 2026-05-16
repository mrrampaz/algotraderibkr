"""Per-symbol RSS news fetcher (Yahoo Finance + MarketWatch).

Free, no auth, no rate limits in practice. Returns deduplicated
NewsArticle objects sorted by publication date descending. The
news_agent consumes these as the primary source when IBKR news
subscriptions are unavailable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import structlog
from bs4 import BeautifulSoup

logger = structlog.get_logger()


YAHOO_FEED = "https://finance.yahoo.com/rss/headline?s={symbol}"
MARKETWATCH_FEED = "https://feeds.content.dj-us.com/public/rss/mw_topstories"
SEEKING_ALPHA_FEED = "https://seekingalpha.com/api/sa/combined/{symbol}.xml"


@dataclass
class NewsArticle:
    article_id: str
    title: str
    summary: str
    url: str
    source: str
    published_utc: datetime
    symbol: str = ""

    @property
    def age_hours(self) -> float:
        now = datetime.now(timezone.utc)
        if self.published_utc.tzinfo is None:
            pub = self.published_utc.replace(tzinfo=timezone.utc)
        else:
            pub = self.published_utc
        return (now - pub).total_seconds() / 3600.0


class RSSNewsClient:
    def __init__(self, timeout_seconds: float = 8.0) -> None:
        self._timeout = timeout_seconds
        self._log = logger.bind(component="rss_news")

    @staticmethod
    def _make_id(title: str, url: str) -> str:
        h = hashlib.sha1()
        h.update(title.encode("utf-8", errors="ignore"))
        h.update(b"|")
        h.update(url.encode("utf-8", errors="ignore"))
        return h.hexdigest()[:16]

    @staticmethod
    def _parse_pubdate(raw: str | None) -> datetime:
        if not raw:
            return datetime.now(timezone.utc)
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (TypeError, ValueError):
            return datetime.now(timezone.utc)

    def fetch(
        self,
        symbol: str,
        max_age_hours: float = 48.0,
        max_articles: int = 40,
    ) -> list[NewsArticle]:
        symbol = symbol.upper()
        urls = [
            ("yahoo", YAHOO_FEED.format(symbol=symbol)),
            ("seekingalpha", SEEKING_ALPHA_FEED.format(symbol=symbol.lower())),
        ]
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        seen_ids: set[str] = set()
        out: list[NewsArticle] = []

        for source, url in urls:
            try:
                with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                    resp = client.get(url, headers={"User-Agent": "Mozilla/5.0 algotrader-singlestock"})
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "xml")
            except Exception:
                self._log.warning("rss_news_fetch_failed", source=source, url=url)
                continue

            for item in soup.find_all("item"):
                title = (item.find("title").get_text(strip=True) if item.find("title") else "").strip()
                link = (item.find("link").get_text(strip=True) if item.find("link") else "").strip()
                description = (
                    item.find("description").get_text(strip=True)
                    if item.find("description")
                    else ""
                )
                pubdate_raw = item.find("pubDate").get_text(strip=True) if item.find("pubDate") else None
                pub = self._parse_pubdate(pubdate_raw)
                if pub < cutoff:
                    continue
                if not title:
                    continue
                aid = self._make_id(title, link)
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)
                out.append(
                    NewsArticle(
                        article_id=aid,
                        title=title,
                        summary=description[:800],
                        url=link,
                        source=source,
                        published_utc=pub,
                        symbol=symbol,
                    )
                )
                if len(out) >= max_articles:
                    break

        out.sort(key=lambda a: a.published_utc, reverse=True)
        self._log.info("rss_news_fetched", symbol=symbol, count=len(out))
        return out
