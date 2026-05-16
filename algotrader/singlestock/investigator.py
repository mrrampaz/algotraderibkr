"""Coordinates the five investigation agents.

Pre-market each day, the investigator:
1. Computes deterministic market context + technicals.
2. Fetches news articles (RSS + IBKR).
3. Runs the news LLM agent for narrative synthesis.
4. Fetches SEC filings + earnings proximity.
5. Runs the announcements LLM agent.
6. Calls the decision LLM agent to produce a final TradeThesis.

Returns the TradeThesis. The orchestrator decides what to do with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from algotrader.singlestock.agents.announcements_agent import AnnouncementsAgent
from algotrader.singlestock.agents.decision_agent import DecisionAgent
from algotrader.singlestock.agents.market_context_agent import MarketContextAgent
from algotrader.singlestock.agents.news_agent import NewsAgent
from algotrader.singlestock.agents.technical_agent import TechnicalAgent
from algotrader.singlestock.feeds.rss_news import NewsArticle
from algotrader.singlestock.thesis import TradeThesis

logger = structlog.get_logger()


@dataclass
class InvestigationResult:
    thesis: TradeThesis
    news_article_ids: list[str]
    article_count: int


class Investigator:
    def __init__(
        self,
        symbol: str,
        market_context_agent: MarketContextAgent,
        technical_agent: TechnicalAgent,
        news_agent: NewsAgent,
        announcements_agent: AnnouncementsAgent,
        decision_agent: DecisionAgent,
    ) -> None:
        self._symbol = symbol.upper()
        self._market_ctx = market_context_agent
        self._tech = technical_agent
        self._news = news_agent
        self._announcements = announcements_agent
        self._decision = decision_agent
        self._log = logger.bind(component="singlestock_investigator", symbol=self._symbol)

    def investigate(self) -> InvestigationResult:
        self._log.info("singlestock_investigation_starting")

        market = self._market_ctx.compute(self._symbol)
        self._log.info(
            "singlestock_market_context",
            beta=market.beta_vs_spy,
            spy_trend=market.spy_trend,
            vix=market.vix_level,
            regime=market.regime,
            aligned=market.market_aligned,
        )

        technicals = self._tech.compute(self._symbol)
        if technicals is None:
            self._log.warning("singlestock_technicals_unavailable")
        else:
            self._log.info(
                "singlestock_technicals",
                direction=technicals.direction.value,
                vwap=technicals.vwap,
                rsi=technicals.rsi_14,
                atr=technicals.atr_14,
                current_price=technicals.current_price,
                gap_pct=technicals.gap_pct,
            )

        articles: list[NewsArticle] = self._news.fetch_articles(self._symbol)
        current_price = technicals.current_price if technicals else 0.0
        news = self._news.synthesize(self._symbol, current_price, articles)
        self._log.info(
            "singlestock_news_thesis",
            direction=news.direction.value,
            confidence=news.confidence,
            article_count=news.article_count,
        )

        announcements = self._announcements.assess(self._symbol)
        self._log.info(
            "singlestock_announcements_thesis",
            direction=announcements.direction.value,
            material_score=announcements.material_event_score,
            earnings_within_days=announcements.earnings_within_days,
        )

        thesis = self._decision.decide(
            symbol=self._symbol,
            news=news,
            announcements=announcements,
            market=market,
            technicals=technicals,
        )
        self._log.info(
            "singlestock_decision",
            direction=thesis.direction.value,
            conviction=thesis.conviction,
            blackout=thesis.blackout_reason,
            rationale=thesis.rationale[:200],
        )

        return InvestigationResult(
            thesis=thesis,
            news_article_ids=[a.article_id for a in articles],
            article_count=len(articles),
        )
