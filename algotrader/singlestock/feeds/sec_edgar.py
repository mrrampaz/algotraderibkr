"""SEC EDGAR client for 8-K, 10-Q, 10-K, and 4 (insider) filings.

EDGAR is free, no auth, but requires a polite User-Agent header.
We fetch the company's recent filings index as JSON and extract material
event filings within a configurable lookback window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


# Mega-cap CIK numbers. EDGAR API takes the 10-digit zero-padded CIK.
SYMBOL_CIK: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "META": "0001326801",
    "TSLA": "0001318605",
    "AMZN": "0001018724",
}

# Filings we treat as potentially market-moving.
MATERIAL_FORMS = {"8-K", "10-Q", "10-K", "DEF 14A", "S-1", "424B"}
# Insider activity — Form 4 — useful for sentiment but lower-weight.
INSIDER_FORMS = {"4", "4/A"}


@dataclass
class Filing:
    form: str
    filed_date: date
    accession: str
    primary_doc: str
    items: str  # 8-K item codes (e.g. "1.01,2.02") or empty
    is_material: bool
    is_insider: bool

    @property
    def url(self) -> str:
        acc_no_dash = self.accession.replace("-", "")
        return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={acc_no_dash}"


class SECEdgarClient:
    """Polite SEC EDGAR fetcher.

    SEC's fair-use policy requires a descriptive User-Agent including an
    email address. Override via the ``user_agent`` parameter.
    """

    BASE = "https://data.sec.gov"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(
        self,
        user_agent: str = "algotrader-singlestock contact@algotraderibkr.local",
        timeout_seconds: float = 10.0,
    ) -> None:
        self._user_agent = user_agent
        self._timeout = timeout_seconds
        self._log = logger.bind(component="sec_edgar")

    def get_cik(self, symbol: str) -> str | None:
        return SYMBOL_CIK.get(symbol.upper())

    def fetch_recent_filings(
        self,
        symbol: str,
        lookback_days: int = 30,
        max_filings: int = 25,
    ) -> list[Filing]:
        cik = self.get_cik(symbol)
        if cik is None:
            self._log.warning("sec_edgar_unknown_cik", symbol=symbol)
            return []

        url = self.SUBMISSIONS_URL.format(cik=cik)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, headers={"User-Agent": self._user_agent})
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            self._log.exception("sec_edgar_fetch_failed", symbol=symbol, url=url)
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        items = recent.get("items", [])

        cutoff = date.today() - timedelta(days=lookback_days)
        out: list[Filing] = []
        for i in range(min(len(forms), len(dates), len(accessions))):
            form = str(forms[i] or "").strip()
            try:
                filed = datetime.strptime(dates[i], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            if filed < cutoff:
                continue
            primary = str(primary_docs[i] or "") if i < len(primary_docs) else ""
            item_codes = str(items[i] or "") if i < len(items) else ""
            f = Filing(
                form=form,
                filed_date=filed,
                accession=str(accessions[i] or ""),
                primary_doc=primary,
                items=item_codes,
                is_material=form in MATERIAL_FORMS,
                is_insider=form in INSIDER_FORMS,
            )
            out.append(f)
            if len(out) >= max_filings:
                break

        out.sort(key=lambda f: f.filed_date, reverse=True)
        self._log.info(
            "sec_edgar_filings_fetched",
            symbol=symbol,
            count=len(out),
            lookback_days=lookback_days,
        )
        return out

    def fetch_filing_summary(self, filing: Filing, symbol: str) -> str:
        """Return a brief plaintext extract of the filing's primary doc.

        Used by the announcements_agent to feed the LLM something more
        than just a form code. Failures are non-fatal — return empty.
        """
        cik = self.get_cik(symbol)
        if cik is None or not filing.primary_doc:
            return ""

        acc_no_dash = filing.accession.replace("-", "")
        url = f"{self.BASE}/Archives/edgar/data/{int(cik)}/{acc_no_dash}/{filing.primary_doc}"
        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": self._user_agent})
                resp.raise_for_status()
                text = resp.text
        except Exception:
            self._log.warning("sec_edgar_filing_doc_fetch_failed", url=url)
            return ""

        # Strip HTML tags crudely — bs4 is already a project dep but
        # keeping this module dependency-light. The LLM tolerates noise.
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, "html.parser")
            content = soup.get_text(separator=" ", strip=True)
        except Exception:
            content = text
        # Trim to ~4000 chars so LLM payload stays small.
        return content[:4000]
