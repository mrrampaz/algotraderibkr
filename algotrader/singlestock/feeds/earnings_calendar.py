"""Earnings dates for mega-caps (seeded + Yahoo fallback).

The main EventCalendar's ``has_earnings()`` only checks a single day and
only seeds NVDA. For the single-stock tool we need:

- A window check ("any earnings in the next N days?").
- Per-symbol seeded data for at least the next quarter.
- A fallback fetch for symbols not seeded.

Update SEEDED_EARNINGS each quarter from investor-relations pages.
Dates are the announcement date (after-market or before-market — we
treat the whole day as blackout).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import httpx
import structlog
from bs4 import BeautifulSoup

logger = structlog.get_logger()


# Confirmed AAPL FY26 earnings dates plus mega-cap peers. Refresh each
# quarter from each company's IR page.
SEEDED_EARNINGS: dict[str, list[date]] = {
    "AAPL": [
        date(2026, 1, 28),
        date(2026, 4, 30),
        date(2026, 7, 30),
        date(2026, 10, 29),
    ],
    "MSFT": [
        date(2026, 1, 27),
        date(2026, 4, 28),
        date(2026, 7, 28),
        date(2026, 10, 27),
    ],
    "NVDA": [
        date(2026, 2, 25),
        date(2026, 5, 27),
        date(2026, 8, 26),
        date(2026, 11, 18),
    ],
    "GOOGL": [
        date(2026, 1, 27),
        date(2026, 4, 28),
        date(2026, 7, 28),
        date(2026, 10, 27),
    ],
    "META": [
        date(2026, 1, 28),
        date(2026, 4, 29),
        date(2026, 7, 29),
        date(2026, 10, 28),
    ],
    "TSLA": [
        date(2026, 1, 21),
        date(2026, 4, 22),
        date(2026, 7, 22),
        date(2026, 10, 21),
    ],
    "AMZN": [
        date(2026, 1, 29),
        date(2026, 4, 30),
        date(2026, 7, 30),
        date(2026, 10, 29),
    ],
}


@dataclass
class NextEarnings:
    symbol: str
    earnings_date: date
    days_away: int
    source: str  # "seeded" or "yahoo"


def _next_earnings_from_seed(symbol: str, today: date) -> NextEarnings | None:
    dates = SEEDED_EARNINGS.get(symbol.upper(), [])
    future = sorted(d for d in dates if d >= today)
    if not future:
        return None
    earn = future[0]
    return NextEarnings(
        symbol=symbol.upper(),
        earnings_date=earn,
        days_away=(earn - today).days,
        source="seeded",
    )


def _next_earnings_from_yahoo(
    symbol: str,
    today: date,
    timeout_seconds: float = 6.0,
) -> NextEarnings | None:
    """Best-effort scrape of Yahoo's earnings calendar HTML.

    Failures are non-fatal — returns None. The caller treats "unknown
    earnings date" as "no blackout" but the agent should consider the
    fallback unreliable and prefer cash when ambiguous.
    """
    url = f"https://finance.yahoo.com/calendar/earnings?symbol={symbol.upper()}"
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0 algotrader-singlestock"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        logger.warning("earnings_yahoo_fetch_failed", symbol=symbol)
        return None

    # Yahoo's HTML changes frequently. Look for a date in any table cell
    # that parses to a future date matching common formats.
    from datetime import datetime
    for cell in soup.find_all("td"):
        txt = cell.get_text(strip=True)
        for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
            try:
                d = datetime.strptime(txt[:20], fmt).date()
            except ValueError:
                continue
            if d >= today:
                return NextEarnings(
                    symbol=symbol.upper(),
                    earnings_date=d,
                    days_away=(d - today).days,
                    source="yahoo",
                )
    return None


def next_earnings(symbol: str, today: date | None = None) -> NextEarnings | None:
    today = today or date.today()
    seeded = _next_earnings_from_seed(symbol, today)
    if seeded is not None:
        return seeded
    return _next_earnings_from_yahoo(symbol, today)


def earnings_within_days(symbol: str, days: int, today: date | None = None) -> bool:
    today = today or date.today()
    nxt = next_earnings(symbol, today)
    if nxt is None:
        return False
    return nxt.days_away <= days
