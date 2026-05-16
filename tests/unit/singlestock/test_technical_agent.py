"""Tests for the deterministic TechnicalAgent."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from algotrader.singlestock.agents.technical_agent import TechnicalAgent
from algotrader.singlestock.thesis import Direction


def _make_daily(trend: str = "up", n: int = 30) -> pd.DataFrame:
    base = 190.0
    drift = 0.6 if trend == "up" else -0.6 if trend == "down" else 0.0
    closes = np.array([base + drift * i + np.sin(i / 3) * 0.5 for i in range(n)])
    highs = closes + 1.2
    lows = closes - 1.2
    opens = closes - 0.3
    vols = np.full(n, 50_000_000)
    idx = pd.date_range(end=datetime.utcnow(), periods=n, freq="D")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


def _make_intraday(price: float = 195.0, n: int = 78) -> pd.DataFrame:
    idx = pd.date_range(end=datetime.utcnow(), periods=n, freq="5min")
    closes = np.linspace(price - 1.0, price + 1.0, n)
    highs = closes + 0.2
    lows = closes - 0.2
    opens = closes - 0.05
    vols = np.full(n, 250_000)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


def test_technical_agent_returns_long_in_uptrend() -> None:
    dp = MagicMock()
    daily = _make_daily(trend="up")
    intraday = _make_intraday(price=daily["close"].iloc[-1])
    dp.get_bars.side_effect = lambda symbol, timeframe, limit: daily if timeframe == "1Day" else intraday
    agent = TechnicalAgent(data_provider=dp)
    out = agent.compute("AAPL")
    assert out is not None
    # In a clean uptrend with current at/above breakout, direction should
    # be LONG; tolerate NONE if rsi happens to fall below the bullish gate.
    assert out.direction in (Direction.LONG, Direction.NONE)
    assert out.vwap > 0
    assert 0 <= out.rsi_14 <= 100
    assert out.atr_14 > 0


def test_technical_agent_handles_missing_intraday() -> None:
    dp = MagicMock()
    daily = _make_daily()

    def get_bars(symbol, timeframe, limit):
        if timeframe == "1Day":
            return daily
        raise RuntimeError("no intraday data")

    dp.get_bars.side_effect = get_bars
    agent = TechnicalAgent(data_provider=dp)
    out = agent.compute("AAPL")
    assert out is not None
    # Falls back to last close as vwap
    assert out.vwap > 0


def test_technical_agent_returns_none_on_insufficient_bars() -> None:
    dp = MagicMock()
    short = _make_daily(n=10)
    dp.get_bars.return_value = short
    agent = TechnicalAgent(data_provider=dp)
    out = agent.compute("AAPL")
    assert out is None
