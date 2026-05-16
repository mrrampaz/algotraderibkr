"""Deterministic market-context agent.

Computes:
- The symbol's beta vs SPY over a rolling 60-day window.
- SPY's short-term trend (5-day vs 20-day SMA cross direction).
- Current VIX level (cached by RegimeDetector when available).
- Whether the symbol's near-term price direction is aligned with SPY.

No LLM calls. Pure Python from historical bars.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import structlog

from algotrader.data.provider import DataProvider
from algotrader.intelligence.regime import RegimeDetector
from algotrader.singlestock.thesis import MarketContext

logger = structlog.get_logger()


class MarketContextAgent:
    def __init__(
        self,
        data_provider: DataProvider,
        regime_detector: RegimeDetector,
        lookback_days: int = 60,
    ) -> None:
        self._data = data_provider
        self._regime = regime_detector
        self._lookback_days = lookback_days
        self._log = logger.bind(component="market_context_agent")

    def compute(self, symbol: str) -> MarketContext:
        beta = self._compute_beta(symbol, "SPY")
        spy_trend = self._compute_spy_trend()
        try:
            regime = self._regime.detect()
            vix = float(regime.vix_level or 0.0)
            regime_name = str(regime.regime_type.value)
        except Exception:
            self._log.exception("market_context_regime_failed")
            vix = 0.0
            regime_name = "unknown"

        # Symbol's recent direction (last 5 days return) vs SPY's
        sym_5d = self._n_day_return(symbol, days=5)
        spy_5d = self._n_day_return("SPY", days=5)
        if sym_5d is None or spy_5d is None:
            aligned = False
        else:
            # Aligned if both same sign AND symbol move has at least
            # 50% of SPY's magnitude scaled by beta (so a low-beta
            # name doesn't get judged for being slow).
            same_sign = (sym_5d * spy_5d) > 0
            expected = spy_5d * (beta or 1.0)
            magnitude_ok = abs(sym_5d) >= abs(expected) * 0.5 if expected != 0 else True
            aligned = bool(same_sign and magnitude_ok)

        return MarketContext(
            beta_vs_spy=beta,
            spy_trend=spy_trend,
            vix_level=vix,
            regime=regime_name,
            market_aligned=aligned,
        )

    def _compute_beta(self, symbol: str, benchmark: str) -> float:
        try:
            sym_bars = self._data.get_bars(symbol, "1Day", limit=self._lookback_days + 5)
            bench_bars = self._data.get_bars(benchmark, "1Day", limit=self._lookback_days + 5)
        except Exception:
            self._log.exception("market_context_bars_failed", symbol=symbol)
            return 1.0
        if sym_bars is None or sym_bars.empty or bench_bars is None or bench_bars.empty:
            return 1.0

        sym_ret = sym_bars["close"].pct_change().dropna().tail(self._lookback_days)
        bench_ret = bench_bars["close"].pct_change().dropna().tail(self._lookback_days)
        joined = pd.concat([sym_ret, bench_ret], axis=1, join="inner").dropna()
        if len(joined) < 20:
            return 1.0
        cov = float(np.cov(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1])
        var = float(np.var(joined.iloc[:, 1]))
        if var <= 0 or math.isnan(cov):
            return 1.0
        return round(cov / var, 3)

    def _compute_spy_trend(self) -> str:
        try:
            bars = self._data.get_bars("SPY", "1Day", limit=30)
        except Exception:
            return "unknown"
        if bars is None or bars.empty or len(bars) < 21:
            return "unknown"
        sma5 = bars["close"].tail(5).mean()
        sma20 = bars["close"].tail(20).mean()
        last = float(bars["close"].iloc[-1])
        if sma5 > sma20 and last >= sma20 * 0.998:
            return "up"
        if sma5 < sma20 and last <= sma20 * 1.002:
            return "down"
        return "sideways"

    def _n_day_return(self, symbol: str, days: int) -> float | None:
        try:
            bars = self._data.get_bars(symbol, "1Day", limit=days + 2)
        except Exception:
            return None
        if bars is None or bars.empty or len(bars) < days + 1:
            return None
        close = bars["close"]
        start = float(close.iloc[-(days + 1)])
        end = float(close.iloc[-1])
        if start <= 0:
            return None
        return (end - start) / start
