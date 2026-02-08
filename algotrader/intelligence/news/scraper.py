"""Web scraper for Finviz screener and Yahoo Finance.

Scrapes:
- Finviz: Pre-market gaps, unusual volume, sector performance
- Yahoo Finance: Earnings calendar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date

import httpx
import pytz
import structlog
from bs4 import BeautifulSoup

logger = structlog.get_logger()

FINVIZ_SCREENER_URL = "https://finviz.com/screener.ashx"
YAHOO_EARNINGS_URL = "https://finance.yahoo.com/calendar/earnings"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


@dataclass
class ScreenerResult:
    """A result row from Finviz screener."""
    symbol: str
    company: str = ""
    sector: str = ""
    price: float = 0.0
    change_pct: float = 0.0
    volume: float = 0.0
    relative_volume: float = 0.0


@dataclass
class EarningsEvent:
    """An earnings event from Yahoo Finance."""
    symbol: str
    company: str = ""
    earnings_date: date | None = None
    eps_estimate: float | None = None
    reported_eps: float | None = None
    surprise_pct: float | None = None
    time: str = ""  # "BMO" (before market open) or "AMC" (after market close)


class WebScraper:
    """Scraper for Finviz and Yahoo Finance.

    Provides supplementary market intelligence beyond what Alpaca API offers.
    All requests use httpx with timeouts and error handling.
    """

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout
        self._log = logger.bind(component="web_scraper")

    def scan_finviz_gaps(self, min_gap_pct: float = 3.0) -> list[ScreenerResult]:
        """Scrape Finviz for stocks with significant pre-market/intraday gaps.

        Uses the Finviz screener with performance filter for top gainers/losers.
        """
        results: list[ScreenerResult] = []

        for direction in ["ta_change_u3", "ta_change_d3"]:  # >3% up, >3% down
            try:
                page_results = self._scrape_finviz_screener(
                    filters=f"sh_avgvol_o500&ta_perf_{direction}",
                )
                results.extend(page_results)
            except Exception:
                self._log.exception("finviz_gap_scan_failed", direction=direction)

        # Filter by min gap and sort
        results = [r for r in results if abs(r.change_pct) >= min_gap_pct]
        results.sort(key=lambda r: abs(r.change_pct), reverse=True)

        self._log.info("finviz_gaps_scanned", results=len(results))
        return results

    def scan_finviz_unusual_volume(self, min_rel_volume: float = 2.0) -> list[ScreenerResult]:
        """Scrape Finviz for stocks with unusual relative volume."""
        try:
            results = self._scrape_finviz_screener(
                filters="sh_avgvol_o500&sh_relvol_o2",  # Avg vol > 500K, rel vol > 2x
            )
            results = [r for r in results if r.relative_volume >= min_rel_volume]
            results.sort(key=lambda r: r.relative_volume, reverse=True)

            self._log.info("finviz_volume_scanned", results=len(results))
            return results
        except Exception:
            self._log.exception("finviz_volume_scan_failed")
            return []

    def get_earnings_calendar(self, target_date: date | None = None) -> list[EarningsEvent]:
        """Scrape Yahoo Finance earnings calendar for a given date."""
        if target_date is None:
            target_date = datetime.now(pytz.timezone("America/New_York")).date()

        date_str = target_date.strftime("%Y-%m-%d")
        url = f"{YAHOO_EARNINGS_URL}?day={date_str}"

        self._log.info("scraping_earnings_calendar", date=date_str)

        try:
            response = httpx.get(url, headers=HEADERS, timeout=self._timeout, follow_redirects=True)
            response.raise_for_status()

            earnings = self._parse_yahoo_earnings(response.text, target_date)
            self._log.info("earnings_calendar_scraped", date=date_str, events=len(earnings))
            return earnings
        except httpx.HTTPStatusError as e:
            self._log.warning("yahoo_earnings_http_error", status=e.response.status_code)
            return []
        except Exception:
            self._log.exception("yahoo_earnings_scrape_failed")
            return []

    def _scrape_finviz_screener(self, filters: str) -> list[ScreenerResult]:
        """Scrape a Finviz screener page with given filters."""
        url = f"{FINVIZ_SCREENER_URL}?v=171&f={filters}"

        response = httpx.get(url, headers=HEADERS, timeout=self._timeout, follow_redirects=True)
        response.raise_for_status()

        return self._parse_finviz_table(response.text)

    def _parse_finviz_table(self, html: str) -> list[ScreenerResult]:
        """Parse the Finviz screener results table."""
        soup = BeautifulSoup(html, "html.parser")
        results: list[ScreenerResult] = []

        # Finviz uses table with class "screener_table" or specific id
        table = soup.find("table", {"id": "screener-views-table"})
        if not table:
            # Try alternate selector
            tables = soup.find_all("table", class_="table-light")
            table = tables[-1] if tables else None

        if not table:
            self._log.debug("finviz_table_not_found")
            return results

        rows = table.find_all("tr")[1:]  # Skip header
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 10:
                continue

            try:
                result = ScreenerResult(
                    symbol=self._clean_text(cols[1]),
                    company=self._clean_text(cols[2]),
                    sector=self._clean_text(cols[3]),
                    price=self._parse_float(cols[8]),
                    change_pct=self._parse_pct(cols[9]),
                    volume=self._parse_volume(cols[6]),
                    relative_volume=self._parse_float(cols[7]),
                )
                if result.symbol:
                    results.append(result)
            except (IndexError, ValueError):
                continue

        return results

    def _parse_yahoo_earnings(self, html: str, target_date: date) -> list[EarningsEvent]:
        """Parse Yahoo Finance earnings calendar HTML."""
        soup = BeautifulSoup(html, "html.parser")
        events: list[EarningsEvent] = []

        # Find the earnings table
        table = soup.find("table")
        if not table:
            return events

        rows = table.find_all("tr")[1:]  # Skip header
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            try:
                symbol = self._clean_text(cols[0])
                company = self._clean_text(cols[1])

                event = EarningsEvent(
                    symbol=symbol,
                    company=company,
                    earnings_date=target_date,
                )

                # Try to parse EPS estimate and reported
                if len(cols) > 2:
                    event.eps_estimate = self._parse_float_safe(cols[2])
                if len(cols) > 3:
                    event.reported_eps = self._parse_float_safe(cols[3])
                if len(cols) > 4:
                    event.surprise_pct = self._parse_pct_safe(cols[4])

                # Try to determine timing (BMO/AMC)
                row_text = row.get_text().lower()
                if "before" in row_text or "bmo" in row_text:
                    event.time = "BMO"
                elif "after" in row_text or "amc" in row_text:
                    event.time = "AMC"

                if symbol:
                    events.append(event)
            except Exception:
                continue

        return events

    @staticmethod
    def _clean_text(element) -> str:
        """Extract clean text from a BS4 element."""
        return element.get_text(strip=True) if element else ""

    @staticmethod
    def _parse_float(element) -> float:
        """Parse a float from a BS4 element."""
        text = element.get_text(strip=True).replace(",", "")
        return float(text)

    @staticmethod
    def _parse_float_safe(element) -> float | None:
        """Parse a float, returning None on failure."""
        try:
            text = element.get_text(strip=True).replace(",", "").replace("$", "")
            if text == "-" or text == "N/A" or not text:
                return None
            return float(text)
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _parse_pct(element) -> float:
        """Parse a percentage like '3.45%' to 3.45."""
        text = element.get_text(strip=True).replace("%", "").replace(",", "")
        return float(text)

    @staticmethod
    def _parse_pct_safe(element) -> float | None:
        """Parse a percentage, returning None on failure."""
        try:
            text = element.get_text(strip=True).replace("%", "").replace(",", "")
            if text == "-" or text == "N/A" or not text:
                return None
            return float(text)
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _parse_volume(element) -> float:
        """Parse volume strings like '1.2M' or '500K'."""
        text = element.get_text(strip=True).replace(",", "").upper()
        multiplier = 1.0
        if text.endswith("M"):
            text = text[:-1]
            multiplier = 1_000_000
        elif text.endswith("K"):
            text = text[:-1]
            multiplier = 1_000
        elif text.endswith("B"):
            text = text[:-1]
            multiplier = 1_000_000_000
        try:
            return float(text) * multiplier
        except ValueError:
            return 0.0
