"""LLM-backed news agent.

Pulls articles from RSS feeds (Yahoo, SeekingAlpha) and optionally IBKR
news, runs a deterministic keyword sentiment baseline, then asks Claude
to synthesize a directional thesis.

If the LLM is unavailable or returns bad JSON, falls back to the
keyword baseline alone — typically producing direction="none" with low
confidence, which the decision agent will treat as no-signal.
"""

from __future__ import annotations

from typing import Any

import structlog

from algotrader.intelligence.news.alpaca_news import (
    BEARISH_KEYWORDS,
    BULLISH_KEYWORDS,
    NewsClient,
)
from algotrader.singlestock.feeds.rss_news import NewsArticle, RSSNewsClient
from algotrader.singlestock.llm_client import LLMClient
from algotrader.singlestock.thesis import Direction, NewsThesis

logger = structlog.get_logger()


def _keyword_sentiment(text: str) -> float:
    text_lower = text.lower()
    bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    if bull + bear == 0:
        return 0.0
    return (bull - bear) / (bull + bear)


class NewsAgent:
    def __init__(
        self,
        rss_client: RSSNewsClient,
        llm_client: LLMClient,
        ibkr_news_client: NewsClient | None = None,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self._rss = rss_client
        self._ibkr_news = ibkr_news_client
        self._llm = llm_client
        self._model = model
        self._log = logger.bind(component="news_agent")
        self._system_prompt: str | None = None

    def _load_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = self._llm.load_prompt("news_synthesis.md")
        return self._system_prompt

    def fetch_articles(
        self,
        symbol: str,
        max_age_hours: float = 48.0,
        max_articles: int = 40,
    ) -> list[NewsArticle]:
        out = self._rss.fetch(symbol, max_age_hours=max_age_hours, max_articles=max_articles)

        # Best-effort augment with IBKR news (paper accounts often lack
        # the subscription so this typically returns nothing).
        if self._ibkr_news is not None:
            try:
                ibkr = self._ibkr_news.fetch_news(symbols=[symbol], limit=20)
                items = ibkr.symbols.get(symbol.upper(), []) if hasattr(ibkr, "symbols") else []
                for it in items:
                    out.append(
                        NewsArticle(
                            article_id=f"ibkr_{hash(it.headline) & 0xFFFFFFFF:08x}",
                            title=str(it.headline or ""),
                            summary=str(getattr(it, "summary", "") or "")[:800],
                            url=str(getattr(it, "url", "") or ""),
                            source="ibkr",
                            published_utc=getattr(it, "timestamp", None),
                            symbol=symbol.upper(),
                        )
                    )
            except Exception:
                self._log.debug("news_agent_ibkr_fetch_skipped")

        return out

    def synthesize(
        self,
        symbol: str,
        current_price: float,
        articles: list[NewsArticle],
    ) -> NewsThesis:
        if not articles:
            return NewsThesis(
                direction=Direction.NONE,
                confidence=0.0,
                summary="no articles in lookback window",
            )

        combined_text = " ".join(
            (a.title + " " + (a.summary or "")) for a in articles
        )
        keyword_sentiment = _keyword_sentiment(combined_text)

        payload = {
            "symbol": symbol,
            "current_price": current_price,
            "existing_keyword_sentiment": round(keyword_sentiment, 3),
            "articles": [
                {
                    "title": a.title,
                    "summary": a.summary,
                    "source": a.source,
                    "published_utc": a.published_utc.isoformat() if a.published_utc else None,
                    "age_hours": round(a.age_hours, 2),
                }
                for a in articles[:30]
            ],
        }

        if not self._llm.available:
            # Deterministic fallback — keyword baseline only, low confidence.
            direction = (
                Direction.LONG if keyword_sentiment > 0.25
                else Direction.SHORT if keyword_sentiment < -0.25
                else Direction.NONE
            )
            return NewsThesis(
                direction=direction,
                confidence=min(0.4, abs(keyword_sentiment)),
                key_catalysts=[],
                avg_sentiment=keyword_sentiment,
                article_count=len(articles),
                summary=f"LLM unavailable; keyword baseline sentiment {keyword_sentiment:+.2f}",
            )

        result = self._llm.call_json(
            model=self._model,
            system_prompt=self._load_system_prompt(),
            user_payload=payload,
            max_tokens=900,
            agent_name="news",
        )
        if result is None:
            return NewsThesis(
                direction=Direction.NONE,
                confidence=0.0,
                avg_sentiment=keyword_sentiment,
                article_count=len(articles),
                summary="LLM call failed; cash-default",
            )

        return NewsThesis(
            direction=_parse_direction(result.get("direction")),
            confidence=_clip01(result.get("confidence", 0.0)),
            key_catalysts=list(result.get("key_catalysts", []) or [])[:8],
            avg_sentiment=float(result.get("avg_sentiment", keyword_sentiment) or 0.0),
            article_count=len(articles),
            summary=str(result.get("summary", "") or "")[:600],
        )


def _parse_direction(raw: Any) -> Direction:
    s = str(raw or "none").lower().strip()
    if s == "long":
        return Direction.LONG
    if s == "short":
        return Direction.SHORT
    return Direction.NONE


def _clip01(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))
