"""IBKR DataProvider implementation."""

from __future__ import annotations

import math
import time
from datetime import date, datetime, timedelta

import pandas as pd
import pytz
import structlog
from ib_async import Option, Stock

from algotrader.core.models import Bar, MarketClock, NewsItem, Quote, Snapshot
from algotrader.execution.ibkr_connection import IBKRConnection

logger = structlog.get_logger()

ET = pytz.timezone("America/New_York")

TIMEFRAME_MAP = {
    "1Min": "1 min",
    "5Min": "5 mins",
    "15Min": "15 mins",
    "30Min": "30 mins",
    "1Hour": "1 hour",
    "1Day": "1 day",
}

OPTION_CHAIN_COLUMNS = [
    "symbol",
    "expiration",
    "strike",
    "type",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "implied_vol",
    "delta",
    "gamma",
    "theta",
    "vega",
]


class IBKRDataProvider:
    """Data provider backed by IBKR via ib_async."""

    def __init__(self, connection: IBKRConnection) -> None:
        self._connection = connection
        self._stock_contract_cache: dict[str, object] = {}
        self._log = logger.bind(component="ibkr_data")
        self._log.info("ibkr_data_provider_initialized")

    @staticmethod
    def _empty_bars_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "vwap", "trade_count"]
        )

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return default
        if math.isnan(as_float) or math.isinf(as_float) or as_float == -1:
            return default
        return as_float

    @staticmethod
    def _safe_optional_float(value) -> float | None:
        if value is None:
            return None
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(as_float) or math.isinf(as_float) or as_float == -1:
            return None
        return as_float

    @staticmethod
    def _to_utc_timestamp(value) -> datetime:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()

    @staticmethod
    def _duration_str(
        timeframe: str,
        limit: int,
        start: datetime | None,
        end: datetime | None,
    ) -> str:
        def _format_days(days: int) -> str:
            if days > 365:
                years = max(1, math.ceil(days / 365))
                return f"{years} Y"
            return f"{days} D"

        if start is not None:
            end_ts = end or datetime.now(pytz.UTC)
            if start.tzinfo is None:
                start = start.replace(tzinfo=pytz.UTC)
            else:
                start = start.astimezone(pytz.UTC)
            if end_ts.tzinfo is None:
                end_ts = end_ts.replace(tzinfo=pytz.UTC)
            else:
                end_ts = end_ts.astimezone(pytz.UTC)
            days = max(1, math.ceil((end_ts - start).total_seconds() / 86400))
            return _format_days(days)

        if timeframe == "1Day":
            days = max(5, int(limit * 3))
            return _format_days(days)
        if timeframe == "1Hour":
            days = max(5, int(math.ceil((limit / 6.5) * 5)))
            return _format_days(days)
        if timeframe == "30Min":
            days = max(5, int(math.ceil((limit * 30 / 390) * 6)))
            return _format_days(days)
        if timeframe == "15Min":
            days = max(5, int(math.ceil((limit * 15 / 390) * 6)))
            return _format_days(days)
        if timeframe == "5Min":
            days = max(3, int(math.ceil((limit * 5 / 390) * 6)))
            return _format_days(days)
        # 1Min default
        days = max(2, int(math.ceil((limit / 390) * 6)))
        return _format_days(days)

    @staticmethod
    def _row_to_bar(symbol: str, timestamp: datetime, row: pd.Series) -> Bar:
        trade_count_raw = row.get("trade_count", 0)
        try:
            trade_count = int(trade_count_raw) if trade_count_raw is not None else 0
        except (TypeError, ValueError):
            trade_count = 0

        return Bar(
            symbol=symbol,
            timestamp=timestamp,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            vwap=float(row["vwap"]) if pd.notna(row["vwap"]) else None,
            trade_count=trade_count,
        )

    def _qualify_stock_contract(self, symbol: str):
        symbol = symbol.upper()
        cached = self._stock_contract_cache.get(symbol)
        if cached is not None:
            return cached

        def _qualify(ib):
            contract = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(contract)
            return qualified[0] if qualified else None

        contract = self._connection.execute(_qualify)
        if contract is None:
            self._log.warning("ibkr_contract_qualification_failed", symbol=symbol)
            return None

        self._stock_contract_cache[symbol] = contract
        return contract

    def _build_quote_from_ticker(self, symbol: str, ticker, now_utc: datetime) -> Quote | None:
        bid = self._safe_float(getattr(ticker, "bid", None))
        ask = self._safe_float(getattr(ticker, "ask", None))
        if bid <= 0 and ask <= 0:
            return None

        ts = getattr(ticker, "time", None) or now_utc
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=pytz.UTC)
            else:
                ts = ts.astimezone(pytz.UTC)

        return Quote(
            symbol=symbol,
            timestamp=ts,
            bid_price=bid,
            bid_size=self._safe_float(getattr(ticker, "bidSize", None)),
            ask_price=ask,
            ask_size=self._safe_float(getattr(ticker, "askSize", None)),
        )

    def _snapshot_from_ticker(self, symbol: str, ticker, now_utc: datetime) -> Snapshot:
        snapshot = Snapshot(symbol=symbol)

        last = self._safe_optional_float(getattr(ticker, "last", None))
        close = self._safe_optional_float(getattr(ticker, "close", None))
        market_price = self._safe_optional_float(getattr(ticker, "marketPrice", None))
        latest_price = last or market_price or close

        if latest_price is not None and latest_price > 0:
            snapshot.latest_trade_price = latest_price
            ts = getattr(ticker, "time", None) or now_utc
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=pytz.UTC)
                else:
                    ts = ts.astimezone(pytz.UTC)
            snapshot.latest_trade_timestamp = ts

        snapshot.latest_quote = self._build_quote_from_ticker(symbol, ticker, now_utc)

        open_px = self._safe_float(getattr(ticker, "open", None))
        high_px = self._safe_float(getattr(ticker, "high", None))
        low_px = self._safe_float(getattr(ticker, "low", None))
        close_px = latest_price if latest_price is not None else close or 0.0
        volume = self._safe_float(getattr(ticker, "volume", None))

        if close_px > 0 or volume > 0:
            snapshot.daily_bar = Bar(
                symbol=symbol,
                timestamp=now_utc,
                open=open_px if open_px > 0 else close_px,
                high=high_px if high_px > 0 else close_px,
                low=low_px if low_px > 0 else close_px,
                close=close_px,
                volume=volume,
                vwap=None,
                trade_count=0,
            )

        if close is not None and close > 0:
            snapshot.prev_daily_bar = Bar(
                symbol=symbol,
                timestamp=now_utc - timedelta(days=1),
                open=close,
                high=close,
                low=close,
                close=close,
                volume=0.0,
                vwap=close,
                trade_count=0,
            )

        return snapshot

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        bar_size = TIMEFRAME_MAP.get(timeframe)
        if bar_size is None:
            raise ValueError(f"Unknown timeframe: {timeframe}. Use one of {list(TIMEFRAME_MAP.keys())}")

        contract = self._qualify_stock_contract(symbol)
        if contract is None:
            return self._empty_bars_df()

        duration = self._duration_str(timeframe=timeframe, limit=limit, start=start, end=end)
        end_dt = ""
        if end is not None:
            end_dt = end if end.tzinfo else end.replace(tzinfo=pytz.UTC)

        try:
            request_timeout = max(5, min(30, int(self._connection.config.timeout)))
            bars = self._connection.execute(
                lambda ib: ib.reqHistoricalData(
                    contract=contract,
                    endDateTime=end_dt,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=2,
                    keepUpToDate=False,
                    timeout=request_timeout,
                )
            )
        except Exception:
            self._log.exception(
                "ibkr_get_bars_failed",
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
            return self._empty_bars_df()

        if not bars:
            return self._empty_bars_df()

        rows: list[dict] = []
        cumulative_notional = 0.0
        cumulative_volume = 0.0
        for bar in bars:
            volume = self._safe_float(getattr(bar, "volume", 0.0))
            close = self._safe_float(getattr(bar, "close", 0.0))
            cumulative_notional += close * volume
            cumulative_volume += volume
            fallback_vwap = (cumulative_notional / cumulative_volume) if cumulative_volume > 0 else math.nan
            rows.append(
                {
                    "timestamp": self._to_utc_timestamp(getattr(bar, "date")),
                    "open": self._safe_float(getattr(bar, "open", 0.0)),
                    "high": self._safe_float(getattr(bar, "high", 0.0)),
                    "low": self._safe_float(getattr(bar, "low", 0.0)),
                    "close": close,
                    "volume": volume,
                    "vwap": self._safe_optional_float(getattr(bar, "average", None))
                    or self._safe_optional_float(getattr(bar, "wap", None))
                    or fallback_vwap,
                    "trade_count": int(self._safe_float(getattr(bar, "barCount", 0), default=0.0)),
                }
            )

        df = pd.DataFrame(rows).set_index("timestamp")
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        df = df.sort_index().tail(limit)

        return df[["open", "high", "low", "close", "volume", "vwap", "trade_count"]]

    def _request_snapshot_ticker(self, contract, wait_seconds: float = 2.0):
        def _request(ib):
            ticker = ib.reqMktData(
                contract=contract,
                genericTickList="",
                snapshot=True,
                regulatorySnapshot=False,
            )

            deadline = time.time() + wait_seconds
            while time.time() < deadline:
                ib.sleep(0.2)
                if (
                    self._safe_float(getattr(ticker, "bid", None)) > 0
                    or self._safe_float(getattr(ticker, "ask", None)) > 0
                    or self._safe_float(getattr(ticker, "last", None)) > 0
                ):
                    break
            return ticker

        return self._connection.execute(_request)

    def get_quote(self, symbol: str) -> Quote | None:
        contract = self._qualify_stock_contract(symbol)
        if contract is None:
            return None

        try:
            ticker = self._request_snapshot_ticker(contract)
            now_utc = datetime.now(pytz.UTC)
            return self._build_quote_from_ticker(symbol.upper(), ticker, now_utc)
        except Exception:
            self._log.exception("ibkr_get_quote_failed", symbol=symbol)
            return None

    def get_snapshot(self, symbol: str) -> Snapshot | None:
        symbol = symbol.upper()
        snapshots = self.get_snapshots([symbol])
        snapshot = snapshots.get(symbol)
        if snapshot is None:
            return None

        # Enrich single-symbol snapshot with recent minute and daily bars.
        try:
            minute = self.get_bars(symbol=symbol, timeframe="1Min", limit=1)
            if not minute.empty:
                ts = minute.index[-1].to_pydatetime()
                snapshot.minute_bar = self._row_to_bar(symbol, ts, minute.iloc[-1])
        except Exception:
            self._log.debug("ibkr_snapshot_minute_bar_failed", symbol=symbol)

        try:
            daily = self.get_bars(symbol=symbol, timeframe="1Day", limit=2)
            if not daily.empty:
                latest_ts = daily.index[-1].to_pydatetime()
                snapshot.daily_bar = self._row_to_bar(symbol, latest_ts, daily.iloc[-1])
                if len(daily) >= 2:
                    prev_ts = daily.index[-2].to_pydatetime()
                    snapshot.prev_daily_bar = self._row_to_bar(symbol, prev_ts, daily.iloc[-2])
        except Exception:
            self._log.debug("ibkr_snapshot_daily_bar_failed", symbol=symbol)

        return snapshot

    def get_snapshots(self, symbols: list[str]) -> dict[str, Snapshot]:
        if not symbols:
            return {}

        clean_symbols = [s.upper() for s in symbols]
        contracts = [Stock(sym, "SMART", "USD") for sym in clean_symbols]

        try:
            def _fetch(ib):
                qualified = ib.qualifyContracts(*contracts)
                if not qualified:
                    return []
                return ib.reqTickers(*qualified)

            tickers = self._connection.execute(_fetch)
        except Exception:
            self._log.exception("ibkr_get_snapshots_failed", symbols=clean_symbols)
            return {}

        now_utc = datetime.now(pytz.UTC)
        snapshots: dict[str, Snapshot] = {}
        for ticker in tickers or []:
            contract = getattr(ticker, "contract", None)
            symbol = getattr(contract, "symbol", None)
            if not symbol:
                continue
            snapshots[symbol.upper()] = self._snapshot_from_ticker(
                symbol=symbol.upper(),
                ticker=ticker,
                now_utc=now_utc,
            )

        return snapshots

    def get_option_chain(
        self,
        underlying: str,
        expiration: date | None = None,
    ) -> pd.DataFrame:
        underlying = underlying.upper()
        empty = pd.DataFrame(columns=OPTION_CHAIN_COLUMNS)

        stock_contract = self._qualify_stock_contract(underlying)
        if stock_contract is None:
            return empty

        try:
            chains, spot = self._connection.execute(
                lambda ib: (
                    ib.reqSecDefOptParams(
                        underlyingSymbol=stock_contract.symbol,
                        futFopExchange="",
                        underlyingSecType=stock_contract.secType,
                        underlyingConId=stock_contract.conId,
                    ),
                    self._safe_optional_float(
                        getattr(
                            (ib.reqTickers(stock_contract) or [None])[0],
                            "last",
                            None,
                        )
                    ),
                )
            )
        except Exception:
            self._log.exception("ibkr_option_chain_fetch_failed", underlying=underlying)
            return empty

        if not chains:
            return empty

        def _parse_strikes(chain_obj) -> list[float]:
            out: list[float] = []
            for s in getattr(chain_obj, "strikes", []) or []:
                try:
                    strike = float(s)
                except (TypeError, ValueError):
                    continue
                if strike > 0:
                    out.append(strike)
            return sorted(set(out))

        preferred = [c for c in chains if str(getattr(c, "exchange", "")).upper() == "SMART"]
        chain_pool = preferred if preferred else list(chains)

        # Pick the chain that best matches the underlying and spot price.
        def _chain_score(chain_obj) -> float:
            strikes_local = _parse_strikes(chain_obj)
            if not strikes_local:
                return float("inf")
            class_name = str(getattr(chain_obj, "tradingClass", "")).upper()
            class_penalty = 0.0 if class_name == underlying else 10_000.0
            if spot is not None and spot > 0:
                nearest = min(abs(s - spot) for s in strikes_local)
            else:
                nearest = strikes_local[0]
            return class_penalty + nearest

        chain = min(chain_pool, key=_chain_score)

        expirations = sorted(getattr(chain, "expirations", []))
        if not expirations:
            return empty

        if expiration is not None:
            exp_str = expiration.strftime("%Y%m%d")
            if exp_str not in expirations:
                self._log.warning(
                    "ibkr_option_expiration_not_available",
                    underlying=underlying,
                    requested=exp_str,
                )
                return empty
        else:
            today_str = datetime.now(pytz.UTC).strftime("%Y%m%d")
            future = [e for e in expirations if e >= today_str]
            exp_str = future[0] if future else expirations[0]

        strikes = _parse_strikes(chain)
        if not strikes:
            return empty

        # Guard against malformed chain metadata (e.g. 10.01 / 10010.0 strikes for SPY).
        if spot is not None and spot > 0:
            bounded = [s for s in strikes if (spot * 0.2) <= s <= (spot * 5.0)]
            if bounded:
                strikes = bounded

        if spot is not None and spot > 0:
            nearest_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
            low = max(0, nearest_idx - 10)
            high = min(len(strikes), nearest_idx + 11)
            selected_strikes = strikes[low:high]
        else:
            selected_strikes = strikes[:20]

        option_contracts = []
        trading_class = getattr(chain, "tradingClass", "")
        for strike in selected_strikes:
            for right in ("C", "P"):
                option_contracts.append(
                    Option(
                        symbol=underlying,
                        lastTradeDateOrContractMonth=exp_str,
                        strike=strike,
                        right=right,
                        exchange="SMART",
                        currency="USD",
                        tradingClass=trading_class,
                    )
                )

        if not option_contracts:
            return empty

        try:
            def _fetch(ib):
                qualified_raw = ib.qualifyContracts(*option_contracts)
                qualified = [c for c in qualified_raw if c is not None]
                if not qualified:
                    return []
                tickers = []
                for i in range(0, len(qualified), 40):
                    chunk = qualified[i:i + 40]
                    if not chunk:
                        continue
                    tickers.extend(ib.reqTickers(*chunk))
                    if i + 40 < len(qualified):
                        ib.sleep(0.2)
                return tickers

            option_tickers = self._connection.execute(_fetch)
        except Exception:
            self._log.exception("ibkr_option_chain_quotes_failed", underlying=underlying)
            return empty

        exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
        rows: list[dict] = []

        for ticker in option_tickers or []:
            contract = getattr(ticker, "contract", None)
            if contract is None:
                continue

            right = str(getattr(contract, "right", "")).upper()
            greeks = getattr(ticker, "modelGreeks", None)

            call_oi = self._safe_optional_float(getattr(ticker, "callOpenInterest", None))
            put_oi = self._safe_optional_float(getattr(ticker, "putOpenInterest", None))
            open_interest = call_oi if right == "C" else put_oi
            if open_interest is None:
                open_interest = self._safe_optional_float(getattr(ticker, "openInterest", None))

            rows.append(
                {
                    "symbol": getattr(contract, "localSymbol", "") or f"{underlying}-{exp_str}-{contract.strike}-{right}",
                    "expiration": exp_date,
                    "strike": self._safe_float(getattr(contract, "strike", 0.0)),
                    "type": "call" if right == "C" else "put",
                    "bid": self._safe_optional_float(getattr(ticker, "bid", None)),
                    "ask": self._safe_optional_float(getattr(ticker, "ask", None)),
                    "last": self._safe_optional_float(getattr(ticker, "last", None))
                    or self._safe_optional_float(getattr(ticker, "close", None)),
                    "volume": self._safe_optional_float(getattr(ticker, "volume", None)),
                    "open_interest": open_interest,
                    "implied_vol": self._safe_optional_float(getattr(ticker, "impliedVol", None)),
                    "delta": self._safe_optional_float(getattr(greeks, "delta", None)),
                    "gamma": self._safe_optional_float(getattr(greeks, "gamma", None)),
                    "theta": self._safe_optional_float(getattr(greeks, "theta", None)),
                    "vega": self._safe_optional_float(getattr(greeks, "vega", None)),
                }
            )

        if not rows:
            return empty

        df = pd.DataFrame(rows)
        return df[OPTION_CHAIN_COLUMNS].sort_values(by=["expiration", "strike", "type"]).reset_index(drop=True)

    def get_news(
        self,
        symbols: list[str] | None = None,
        limit: int = 50,
    ) -> list[NewsItem]:
        if not symbols:
            return []

        symbols = [s.upper() for s in symbols]
        per_symbol = max(1, limit // max(1, len(symbols)))
        items: list[NewsItem] = []

        try:
            providers = self._connection.execute(lambda ib: ib.reqNewsProviders())
        except Exception:
            self._log.exception("ibkr_news_provider_lookup_failed")
            return []

        provider_codes = "+".join(
            p.code for p in providers or [] if getattr(p, "code", "")
        )
        if not provider_codes:
            return []

        for symbol in symbols:
            contract = self._qualify_stock_contract(symbol)
            if contract is None:
                continue

            try:
                headlines = self._connection.execute(
                    lambda ib: ib.reqHistoricalNews(
                        conId=contract.conId,
                        providerCodes=provider_codes,
                        startDateTime="",
                        endDateTime="",
                        totalResults=per_symbol,
                    )
                )
            except Exception:
                self._log.debug("ibkr_symbol_news_failed", symbol=symbol)
                continue

            for article in headlines or []:
                timestamp = getattr(article, "time", None)
                if isinstance(timestamp, datetime):
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=pytz.UTC)
                    else:
                        timestamp = timestamp.astimezone(pytz.UTC)
                items.append(
                    NewsItem(
                        id=f"{getattr(article, 'providerCode', '')}:{getattr(article, 'articleId', '')}",
                        headline=getattr(article, "headline", ""),
                        summary="",
                        source=getattr(article, "providerCode", ""),
                        url="",
                        symbols=[symbol],
                        timestamp=timestamp,
                    )
                )

        items.sort(
            key=lambda i: i.timestamp if i.timestamp is not None else datetime.min.replace(tzinfo=pytz.UTC),
            reverse=True,
        )
        return items[:limit]

    @staticmethod
    def _next_weekday(d: date) -> date:
        next_day = d
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day

    def get_clock(self) -> MarketClock:
        now_utc = datetime.now(pytz.UTC)
        now_et = now_utc.astimezone(ET)

        open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        is_weekday = now_et.weekday() < 5
        is_open = is_weekday and open_et <= now_et < close_et

        if is_open:
            next_close_et = close_et
            next_open_date = self._next_weekday(now_et.date() + timedelta(days=1))
            next_open_et = ET.localize(datetime.combine(next_open_date, datetime.min.time())).replace(
                hour=9, minute=30
            )
        else:
            if is_weekday and now_et < open_et:
                next_open_et = open_et
                next_close_et = close_et
            else:
                next_open_date = self._next_weekday(now_et.date() + timedelta(days=1))
                next_open_et = ET.localize(datetime.combine(next_open_date, datetime.min.time())).replace(
                    hour=9, minute=30
                )
                next_close_et = next_open_et.replace(hour=16, minute=0)

        return MarketClock(
            timestamp=now_utc,
            is_open=is_open,
            next_open=next_open_et.astimezone(pytz.UTC),
            next_close=next_close_et.astimezone(pytz.UTC),
        )

    def is_market_open(self) -> bool:
        return self.get_clock().is_open
