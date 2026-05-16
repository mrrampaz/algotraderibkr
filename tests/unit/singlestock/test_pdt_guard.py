"""Tests for the PDT guard."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytz

from algotrader.singlestock.pdt_guard import PDTGuard

ET = pytz.timezone("America/New_York")


def test_pdt_disabled_always_allows_close() -> None:
    g = PDTGuard(enabled=False)
    entry = datetime.now(ET) - timedelta(hours=1)
    assert g.can_close_today(entry)


def test_pdt_enabled_blocks_same_day_close() -> None:
    g = PDTGuard(enabled=True)
    entry = datetime.now(ET) - timedelta(hours=1)
    assert not g.can_close_today(entry)


def test_pdt_enabled_allows_next_day_close() -> None:
    g = PDTGuard(enabled=True)
    entry = datetime.now(ET) - timedelta(days=1)
    assert g.can_close_today(entry)
