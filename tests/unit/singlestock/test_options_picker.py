"""Tests for OptionsPicker — sizing and contract selection."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest
import pytz

from algotrader.singlestock.options_picker import OptionsPicker
from algotrader.singlestock.thesis import Direction


def _make_chain(target_strike: float = 25.0) -> pd.DataFrame:
    # Cheap synthetic chain (think a $25 stock) so the exercise-exposure
    # cap doesn't reduce contract size to zero in tests at sane slice
    # capitals. Real AAPL strikes work the same way given enough capital.
    strikes = [23.0, 24.0, 25.0, 26.0, 27.0]
    rows = []
    for s in strikes:
        moneyness = target_strike / s
        delta_call = max(0.05, min(0.95, 0.50 + (target_strike - s) * 0.08))
        delta_put = -max(0.05, min(0.95, 0.50 - (target_strike - s) * 0.08))
        rows.append({
            "symbol": f"AAPL  XXX{int(s*1000):08d}C",
            "expiration": date.today() + timedelta(days=12),
            "strike": s,
            "type": "call",
            "bid": 4.50 + (target_strike - s) * 0.4,
            "ask": 4.65 + (target_strike - s) * 0.4,
            "last": 4.55 + (target_strike - s) * 0.4,
            "volume": 1200,
            "open_interest": 5000,
            "implied_vol": 0.25,
            "delta": delta_call,
            "gamma": 0.04,
            "theta": -0.08,
            "vega": 0.12,
        })
        rows.append({
            "symbol": f"AAPL  XXX{int(s*1000):08d}P",
            "expiration": date.today() + timedelta(days=12),
            "strike": s,
            "type": "put",
            "bid": 3.40 + (s - target_strike) * 0.4,
            "ask": 3.55 + (s - target_strike) * 0.4,
            "last": 3.45 + (s - target_strike) * 0.4,
            "volume": 900,
            "open_interest": 4000,
            "implied_vol": 0.26,
            "delta": delta_put,
            "gamma": 0.04,
            "theta": -0.08,
            "vega": 0.12,
        })
    df = pd.DataFrame(rows)
    # Ensure bid <= ask
    df["ask"] = df[["bid", "ask"]].max(axis=1) + 0.05
    return df


def test_picker_returns_none_for_direction_none() -> None:
    picker = OptionsPicker(data_provider=MagicMock())
    out = picker.pick("AAPL", Direction.NONE, slice_capital=20000)
    assert out is None


def test_picker_selects_closest_delta_call() -> None:
    mock_dp = MagicMock()
    mock_dp.get_option_chain.return_value = _make_chain()
    picker = OptionsPicker(
        data_provider=mock_dp,
        target_delta=0.70,
        min_dte=10,
        max_dte=14,
        max_contracts_per_trade=5,
        max_position_premium_pct=5.0,
        exercise_exposure_cap_pct=20.0,
    )
    out = picker.pick("AAPL", Direction.LONG, slice_capital=20000)
    assert out is not None
    assert out.right == "C"
    # Selected strike should be one of the fixture's strikes
    assert out.strike in {23.0, 24.0, 25.0, 26.0, 27.0}
    # mid is the average of bid/ask
    assert out.mid == pytest.approx((out.bid + out.ask) / 2.0, rel=0.01)
    assert out.contracts >= 1


def test_picker_size_caps_at_premium_budget() -> None:
    mock_dp = MagicMock()
    mock_dp.get_option_chain.return_value = _make_chain()
    # Slice 50k, premium budget 5% = $2500. Per contract mid*100 ≈ $450,
    # so 5 fit by premium. Cap at max_contracts_per_trade=2 should win.
    picker = OptionsPicker(
        data_provider=mock_dp,
        target_delta=0.70,
        max_contracts_per_trade=2,
        max_position_premium_pct=5.0,
        exercise_exposure_cap_pct=100.0,
    )
    out = picker.pick("AAPL", Direction.LONG, slice_capital=50_000)
    assert out is not None
    assert out.contracts <= 2


def test_picker_returns_none_when_size_zero() -> None:
    mock_dp = MagicMock()
    mock_dp.get_option_chain.return_value = _make_chain()
    picker = OptionsPicker(
        data_provider=mock_dp,
        target_delta=0.70,
        max_contracts_per_trade=5,
        max_position_premium_pct=0.001,  # too tight to afford a single contract
        exercise_exposure_cap_pct=20.0,
    )
    out = picker.pick("AAPL", Direction.LONG, slice_capital=20000)
    assert out is None


def test_picker_selects_put_for_short() -> None:
    mock_dp = MagicMock()
    mock_dp.get_option_chain.return_value = _make_chain()
    picker = OptionsPicker(data_provider=mock_dp, target_delta=0.70)
    out = picker.pick("AAPL", Direction.SHORT, slice_capital=20000)
    assert out is not None
    assert out.right == "P"
