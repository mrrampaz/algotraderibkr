"""Deterministic technical agent.

Computes day-trade-relevant levels and indicators on the symbol:
- VWAP (today's session, intraday bars)
- RSI(14) on daily closes
- ATR(14) on daily bars
- 5/20-day breakout (high) and breakdown (low) levels
- Gap vs prior close
- Naive direction call: above VWAP + RSI > 55 + breaking 20-day high → long.

Pure Python. No LLM. Feeds the decision agent.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import structlog

from algotrader.data.provider import DataProvider
from algotrader.singlestock.thesis import Direction, TechnicalContext

logger = structlog.get_logger()


class TechnicalAgent:
    def __init__(self, data_provider: DataProvider) -> None:
        self._data = data_provider
        self._log = logger.bind(component="technical_agent")

    def compute(self, symbol: str) -> TechnicalContext | None:
        try:
            daily = self._data.get_bars(symbol, "1Day", limit=60)
        except Exception:
            self._log.exception("technical_daily_bars_failed", symbol=symbol)
            return None
        if daily is None or daily.empty or len(daily) < 21:
            self._log.warning("technical_insufficient_daily_bars", symbol=symbol)
            return None

        try:
            intraday = self._data.get_bars(symbol, "5Min", limit=120)
        except Exception:
            self._log.warning("technical_intraday_bars_failed", symbol=symbol)
            intraday = None

        current_price = float(daily["close"].iloc[-1])
        prior_close = float(daily["close"].iloc[-2])
        gap_pct = ((current_price - prior_close) / prior_close) if prior_close > 0 else 0.0

        # VWAP from session bars if available, else last daily close.
        vwap = self._compute_vwap(intraday) if intraday is not None else current_price
        if vwap is None or vwap <= 0:
            vwap = current_price

        rsi = self._rsi(daily["close"], period=14)
        atr = self._atr(daily, period=14)

        recent_high_20 = float(daily["high"].tail(20).max())
        recent_low_20 = float(daily["low"].tail(20).min())

        direction = self._infer_direction(
            current_price=current_price,
            vwap=vwap,
            rsi=rsi,
            recent_high_20=recent_high_20,
            recent_low_20=recent_low_20,
            gap_pct=gap_pct,
        )

        return TechnicalContext(
            direction=direction,
            vwap=round(vwap, 2),
            rsi_14=round(rsi, 2),
            atr_14=round(atr, 4),
            breakout_level_up=round(recent_high_20, 2),
            breakdown_level_down=round(recent_low_20, 2),
            current_price=round(current_price, 2),
            gap_pct=round(gap_pct * 100, 3),
        )

    @staticmethod
    def _compute_vwap(intraday: pd.DataFrame) -> float | None:
        if intraday is None or intraday.empty:
            return None
        try:
            session = intraday.tail(78)  # ~6.5h of 5Min bars
            typical = (session["high"] + session["low"] + session["close"]) / 3.0
            vol = session["volume"]
            denom = float(vol.sum())
            if denom <= 0:
                return None
            return float((typical * vol).sum() / denom)
        except Exception:
            return None

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        delta = close.diff().dropna()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.tail(period).mean()
        avg_loss = loss.tail(period).mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @staticmethod
    def _atr(daily: pd.DataFrame, period: int = 14) -> float:
        if len(daily) < period + 1:
            return 0.0
        high = daily["high"]
        low = daily["low"]
        close_prev = daily["close"].shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - close_prev).abs(),
                (low - close_prev).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.tail(period).mean())

    @staticmethod
    def _infer_direction(
        current_price: float,
        vwap: float,
        rsi: float,
        recent_high_20: float,
        recent_low_20: float,
        gap_pct: float,
    ) -> Direction:
        above_vwap = current_price >= vwap
        below_vwap = current_price <= vwap
        near_breakout = current_price >= recent_high_20 * 0.995
        near_breakdown = current_price <= recent_low_20 * 1.005

        if above_vwap and rsi >= 55 and (near_breakout or gap_pct > 0.01):
            return Direction.LONG
        if below_vwap and rsi <= 45 and (near_breakdown or gap_pct < -0.01):
            return Direction.SHORT
        return Direction.NONE
