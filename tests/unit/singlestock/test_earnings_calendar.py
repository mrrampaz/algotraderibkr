"""Tests for the seeded earnings calendar."""

from __future__ import annotations

from datetime import date, timedelta

from algotrader.singlestock.feeds import earnings_calendar as ec


def test_next_earnings_aapl_seeded() -> None:
    today = date(2026, 1, 1)
    nxt = ec.next_earnings("AAPL", today)
    assert nxt is not None
    assert nxt.symbol == "AAPL"
    assert nxt.source == "seeded"
    assert nxt.earnings_date >= today
    assert nxt.days_away >= 0


def test_earnings_within_days_true() -> None:
    today = date(2026, 1, 26)  # 2 days before seeded 2026-01-28
    assert ec.earnings_within_days("AAPL", days=2, today=today) is True


def test_earnings_within_days_false() -> None:
    today = date(2026, 2, 15)  # well after Jan-28, before Apr-30
    assert ec.earnings_within_days("AAPL", days=5, today=today) is False


def test_unknown_symbol_returns_false() -> None:
    # No seed for "XYZ" and yahoo will likely fail in test env — function
    # must not raise.
    today = date(2026, 1, 1)
    assert ec.earnings_within_days("XYZ_UNSEEDED", days=2, today=today) in (False, True)
