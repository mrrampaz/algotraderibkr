"""Tests for SingleStockState persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from algotrader.singlestock.state import OpenPosition, SingleStockState


def test_state_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ss.json"
    s = SingleStockState(path)
    s.update_thesis({"direction": "long", "conviction": 0.72})
    s.update_news_baseline(["a1", "a2"])
    pos = OpenPosition(
        con_id=12345,
        local_symbol="AAPL  260123C00195000",
        right="C",
        strike=195.0,
        expiry="20260123",
        qty=2,
        entry_premium=4.55,
        entry_time=datetime(2026, 1, 15, 14, 35, tzinfo=timezone.utc),
        direction="long",
        underlying_at_entry=196.10,
        stop_underlying=193.00,
        target_premium=6.83,
        client_order_id="ss_open_test",
    )
    s.record_entry(pos)

    loaded = SingleStockState(path)
    assert loaded.open_position is not None
    assert loaded.open_position.con_id == 12345
    assert loaded.open_position.qty == 2
    assert loaded.open_position.entry_premium == 4.55
    assert loaded.thesis_json == {"direction": "long", "conviction": 0.72}
    assert loaded.news_baseline_ids == ["a1", "a2"]


def test_state_close_clears_position(tmp_path: Path) -> None:
    path = tmp_path / "ss.json"
    s = SingleStockState(path)
    pos = OpenPosition(
        con_id=1,
        local_symbol="X",
        right="C",
        strike=100.0,
        expiry="20260123",
        qty=1,
        entry_premium=2.0,
        entry_time=datetime.now(timezone.utc),
        direction="long",
        underlying_at_entry=100.0,
        stop_underlying=98.0,
        target_premium=3.0,
        client_order_id="x",
    )
    s.record_entry(pos)
    s.record_close(realized_pnl_dollars=150.0)
    loaded = SingleStockState(path)
    assert loaded.open_position is None
    assert loaded.counters.closes_today == 1
    assert loaded.counters.realized_pnl_dollars == pytest.approx(150.0)


def test_state_counters_reset_on_new_day(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "ss.json"
    s = SingleStockState(path)
    s.counters.llm_calls = 5
    s.counters.realized_pnl_dollars = 99.0
    s.save()

    from algotrader.singlestock import state as state_mod

    monkeypatch.setattr(
        state_mod.SingleStockState,
        "_today_et",
        staticmethod(lambda: "2099-12-31"),
    )
    loaded = SingleStockState(path)
    assert loaded.counters.llm_calls == 0
    assert loaded.counters.realized_pnl_dollars == 0.0
