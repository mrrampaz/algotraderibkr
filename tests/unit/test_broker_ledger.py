"""Unit tests for BrokerLedger — broker fill ledger source of truth."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch

import pytz
import pytest

from algotrader.tracking.broker_ledger import BrokerLedger

ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ledger(tmp_path) -> BrokerLedger:
    """Create a BrokerLedger backed by a temp SQLite file, no event wiring."""
    db = str(tmp_path / "trades.db")
    return BrokerLedger(db_path=db, ib=None)


def _fill(
    ledger: BrokerLedger,
    exec_id: str,
    perm_id: int,
    symbol: str = "SPY",
    sec_type: str = "OPT",
    right: str = "P",
    strike: float = 560.0,
    expiry: str = "20260502",
    side: str = "SLD",
    qty: float = 1.0,
    price: float = 1.50,
    commission: float | None = None,
    realized_pnl: float | None = None,
    ts: str | None = None,
) -> None:
    if ts is None:
        ts = datetime.now(pytz.UTC).isoformat()
    ledger.record_fill(
        exec_id=exec_id,
        perm_id=perm_id,
        order_ref=None,
        account="DU12345",
        timestamp_utc=ts,
        symbol=symbol,
        sec_type=sec_type,
        local_symbol=None,
        right=right,
        strike=strike,
        expiry=expiry,
        side=side,
        quantity=qty,
        price=price,
        commission=commission,
        realized_pnl=realized_pnl,
    )


def _today_ts() -> str:
    """ISO UTC timestamp for right now."""
    return datetime.now(pytz.UTC).isoformat()


def _today_et_str() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_same_exec_id_fired_twice_gives_one_row(self, tmp_path):
        """Same execDetails fired twice → INSERT OR IGNORE → one row."""
        ledger = _make_ledger(tmp_path)
        _fill(ledger, exec_id="EXEC001", perm_id=1001)
        _fill(ledger, exec_id="EXEC001", perm_id=1001)  # duplicate

        fills = ledger.get_fills_since(datetime(2000, 1, 1, tzinfo=pytz.UTC))
        assert len(fills) == 1

    def test_different_exec_ids_give_two_rows(self, tmp_path):
        ledger = _make_ledger(tmp_path)
        _fill(ledger, exec_id="EXEC001", perm_id=1001)
        _fill(ledger, exec_id="EXEC002", perm_id=1001)

        fills = ledger.get_fills_since(datetime(2000, 1, 1, tzinfo=pytz.UTC))
        assert len(fills) == 2


# ---------------------------------------------------------------------------
# Commission update
# ---------------------------------------------------------------------------

class TestCommissionUpdate:
    def test_commission_updated_after_exec(self, tmp_path):
        """execDetails first, commissionReport second → row has both."""
        ledger = _make_ledger(tmp_path)
        _fill(ledger, exec_id="EXEC001", perm_id=1001, commission=None, realized_pnl=None)

        # Simulate commissionReport arriving afterward
        ledger.update_fill_commission("EXEC001", commission=0.65, realized_pnl=None)

        fills = ledger.get_fills_since(datetime(2000, 1, 1, tzinfo=pytz.UTC))
        assert len(fills) == 1
        assert fills[0]["commission"] == pytest.approx(0.65)
        assert fills[0]["realized_pnl"] is None

    def test_realized_pnl_set_on_closing_fill(self, tmp_path):
        ledger = _make_ledger(tmp_path)
        _fill(ledger, exec_id="EXEC002", perm_id=1002)
        ledger.update_fill_commission("EXEC002", commission=0.65, realized_pnl=120.50)

        fills = ledger.get_fills_since(datetime(2000, 1, 1, tzinfo=pytz.UTC))
        assert fills[0]["realized_pnl"] == pytest.approx(120.50)


# ---------------------------------------------------------------------------
# get_open_option_legs — net position math
# ---------------------------------------------------------------------------

class TestOpenOptionLegs:
    def test_fully_closed_position_not_returned(self, tmp_path):
        """Open then close fully → net_qty = 0 → not in open legs."""
        ledger = _make_ledger(tmp_path)
        ts = _today_ts()
        # Open: SLD 1 contract (short put)
        _fill(ledger, exec_id="O1", perm_id=1001, side="SLD", qty=1.0, ts=ts)
        # Close: BOT 1 contract
        _fill(ledger, exec_id="C1", perm_id=2001, side="BOT", qty=1.0, ts=ts)

        legs = ledger.get_open_option_legs()
        assert legs == []

    def test_partially_closed_position_shows_remainder(self, tmp_path):
        """Open 2 contracts, close 1 → net_qty = -1 still open."""
        ledger = _make_ledger(tmp_path)
        ts = _today_ts()
        _fill(ledger, exec_id="O1", perm_id=1001, side="SLD", qty=2.0, ts=ts)
        _fill(ledger, exec_id="C1", perm_id=2001, side="BOT", qty=1.0, ts=ts)

        legs = ledger.get_open_option_legs()
        assert len(legs) == 1
        assert legs[0]["net_qty"] == pytest.approx(-1.0)

    def test_open_position_returned(self, tmp_path):
        """Single open short put → net_qty = -1."""
        ledger = _make_ledger(tmp_path)
        _fill(ledger, exec_id="O1", perm_id=1001, side="SLD", qty=1.0, ts=_today_ts())

        legs = ledger.get_open_option_legs()
        assert len(legs) == 1
        assert legs[0]["symbol"] == "SPY"
        assert legs[0]["right"] == "P"
        assert legs[0]["net_qty"] == pytest.approx(-1.0)

    def test_two_different_strikes_both_open(self, tmp_path):
        """Short 560P + long 555P both open → two legs returned."""
        ledger = _make_ledger(tmp_path)
        ts = _today_ts()
        _fill(ledger, exec_id="S1", perm_id=1001, side="SLD", strike=560.0, ts=ts)
        _fill(ledger, exec_id="L1", perm_id=1001, side="BOT", strike=555.0, ts=ts)

        legs = ledger.get_open_option_legs()
        assert len(legs) == 2
        net_qtys = {(r["strike"], r["side"] if "side" in r else None): r["net_qty"] for r in legs}
        strikes = {r["strike"] for r in legs}
        assert 560.0 in strikes
        assert 555.0 in strikes

    def test_non_option_fills_excluded(self, tmp_path):
        """STK fills should not appear in open option legs."""
        ledger = _make_ledger(tmp_path)
        _fill(ledger, exec_id="E1", perm_id=9001, sec_type="STK", right=None,
              strike=None, expiry=None, ts=_today_ts())
        assert ledger.get_open_option_legs() == []


# ---------------------------------------------------------------------------
# get_recent_closed_round_trips
# ---------------------------------------------------------------------------

class TestClosedRoundTrips:
    def _insert_round_trip(
        self,
        ledger: BrokerLedger,
        perm_id: int,
        symbol: str = "SPY",
        realized_pnl: float = 100.0,
        commission: float = 1.30,
        ts: str | None = None,
    ) -> None:
        ts = ts or _today_ts()
        # Opening fill (no realized_pnl)
        _fill(ledger, exec_id=f"O_{perm_id}", perm_id=perm_id, side="SLD",
              commission=commission / 2, realized_pnl=None, ts=ts)
        # Closing fill (realized_pnl set)
        _fill(ledger, exec_id=f"C_{perm_id}", perm_id=perm_id + 1000,
              side="BOT", commission=commission / 2,
              realized_pnl=realized_pnl, ts=ts, symbol=symbol)

    def test_groups_by_perm_id(self, tmp_path):
        """Multiple fills with same perm_id → one round-trip."""
        ledger = _make_ledger(tmp_path)
        ts = _today_ts()
        # Three fills in same BAG close order share perm_id
        for i, exec_id in enumerate(["C1", "C2", "C3"]):
            _fill(ledger, exec_id=exec_id, perm_id=5001,
                  realized_pnl=50.0 if i == 0 else None,
                  commission=0.65, ts=ts)

        trips = ledger.get_recent_closed_round_trips(n=10)
        assert len(trips) == 1
        assert trips[0]["perm_id"] == 5001

    def test_pnl_calculation_short_premium(self, tmp_path):
        """
        Round-trip P&L for short premium:
          gross_pnl = sum(realized_pnl from closing fills)
          net_pnl   = gross_pnl - total_commission
        """
        ledger = _make_ledger(tmp_path)
        ts = _today_ts()
        # Open spread: SLD 560P, BOT 555P — no realized_pnl on opening
        _fill(ledger, exec_id="O_short", perm_id=7001, side="SLD",
              strike=560.0, commission=0.65, realized_pnl=None, ts=ts)
        _fill(ledger, exec_id="O_long", perm_id=7001, side="BOT",
              strike=555.0, commission=0.65, realized_pnl=None, ts=ts)
        # Close spread: BOT 560P, SLD 555P — IBKR sets realized_pnl on each
        _fill(ledger, exec_id="C_short", perm_id=8001, side="BOT",
              strike=560.0, commission=0.65, realized_pnl=130.0, ts=ts)
        _fill(ledger, exec_id="C_long", perm_id=8001, side="SLD",
              strike=555.0, commission=0.65, realized_pnl=-5.0, ts=ts)

        trips = ledger.get_recent_closed_round_trips(n=10)
        # Only the perm_id=8001 group has realized_pnl (closing order)
        close_trip = next(t for t in trips if t["perm_id"] == 8001)
        assert close_trip["gross_pnl"] == pytest.approx(125.0)
        assert close_trip["commission"] == pytest.approx(1.30)
        assert close_trip["net_pnl"] == pytest.approx(123.70)

    def test_opening_fills_not_counted_as_round_trips(self, tmp_path):
        """Opening fills (NULL realized_pnl) don't produce round-trip entries."""
        ledger = _make_ledger(tmp_path)
        ts = _today_ts()
        # Opening fills only
        _fill(ledger, exec_id="O1", perm_id=3001, realized_pnl=None, ts=ts)
        _fill(ledger, exec_id="O2", perm_id=3001, realized_pnl=None, ts=ts)

        trips = ledger.get_recent_closed_round_trips(n=10)
        assert trips == []

    def test_limit_n_is_respected(self, tmp_path):
        """get_recent_closed_round_trips(n=2) returns at most 2."""
        ledger = _make_ledger(tmp_path)
        ts = _today_ts()
        for i in range(5):
            _fill(ledger, exec_id=f"C{i}", perm_id=9000 + i,
                  realized_pnl=50.0, commission=0.65, ts=ts)

        trips = ledger.get_recent_closed_round_trips(n=2)
        assert len(trips) <= 2

    def test_win_rate_calculation(self, tmp_path):
        """3 winners and 1 loser → win_rate = 0.75."""
        ledger = _make_ledger(tmp_path)
        ts = _today_ts()
        for perm_id, pnl in [(100, 80.0), (101, 120.0), (102, 60.0), (103, -40.0)]:
            _fill(ledger, exec_id=f"C_{perm_id}", perm_id=perm_id,
                  realized_pnl=pnl, commission=0.65, ts=ts)

        trips = ledger.get_recent_closed_round_trips(n=10)
        wins = sum(1 for t in trips if t["net_pnl"] > 0)
        assert len(trips) == 4
        assert wins == 3


# ---------------------------------------------------------------------------
# get_daily_summary
# ---------------------------------------------------------------------------

class TestDailySummary:
    def test_summary_matches_broker_fills(self, tmp_path):
        """Daily summary P&L matches sum of today's closed round-trip realized_pnl."""
        ledger = _make_ledger(tmp_path)
        ts = datetime.now(pytz.UTC).isoformat()

        # Three round-trips today: +$300, +$200, -$50
        for perm_id, pnl in [(10, 300.0), (11, 200.0), (12, -50.0)]:
            _fill(ledger, exec_id=f"C_{perm_id}", perm_id=perm_id,
                  realized_pnl=pnl, commission=1.30, ts=ts)

        today = datetime.now(ET).date()
        summary = ledger.get_daily_summary(trade_date=today)

        assert summary["total_trades"] == 3
        assert summary["wins"] == 2
        assert summary["losses"] == 1
        # Each net_pnl = realized_pnl - commission (1.30)
        expected_total = (300 - 1.30) + (200 - 1.30) + (-50 - 1.30)
        assert summary["total_pnl"] == pytest.approx(expected_total)
        assert summary["source"] == "broker_ledger"

    def test_empty_day_returns_zeros(self, tmp_path):
        ledger = _make_ledger(tmp_path)
        summary = ledger.get_daily_summary()
        assert summary["total_trades"] == 0
        assert summary["total_pnl"] == pytest.approx(0.0)
        assert summary["win_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Brain integration — _get_recent_win_rate reads from ledger
# ---------------------------------------------------------------------------

class TestBrainUsesLedger:
    def test_brain_reads_ledger_not_trades_db(self, tmp_path):
        """Brain uses broker_ledger for win rate when it has >= 5 round-trips."""
        from algotrader.strategy_selector.brain import DailyBrain

        ledger = _make_ledger(tmp_path)
        ts = datetime.now(pytz.UTC).isoformat()

        # Insert 8 round-trips: 6 wins (+pnl), 2 losses (-pnl)
        for i in range(6):
            _fill(ledger, exec_id=f"W_{i}", perm_id=200 + i,
                  realized_pnl=100.0, commission=1.30, ts=ts)
        for i in range(2):
            _fill(ledger, exec_id=f"L_{i}", perm_id=290 + i,
                  realized_pnl=-50.0, commission=1.30, ts=ts)

        brain = DailyBrain(
            total_capital=100_000,
            broker_ledger=ledger,
            recent_win_rate_lookback_trades=15,
            recent_win_rate_fallback=0.80,
        )

        rate = brain._get_recent_win_rate(lookback_trades=15)
        assert rate is not None
        # 6 wins / 8 trips = 0.75
        assert rate == pytest.approx(0.75)

    def test_brain_falls_back_when_ledger_insufficient(self, tmp_path):
        """Brain returns None (→ fallback) when ledger has < 5 round-trips."""
        from algotrader.strategy_selector.brain import DailyBrain
        from algotrader.tracking.journal import TradeJournal

        ledger = _make_ledger(tmp_path)
        empty_journal = TradeJournal(db_path=str(tmp_path / "empty_journal.db"))
        ts = datetime.now(pytz.UTC).isoformat()
        # Only 3 round-trips — below the minimum of 5
        for i in range(3):
            _fill(ledger, exec_id=f"W_{i}", perm_id=300 + i,
                  realized_pnl=100.0, commission=0.65, ts=ts)

        brain = DailyBrain(
            total_capital=100_000,
            trade_journal=empty_journal,
            broker_ledger=ledger,
            recent_win_rate_fallback=0.80,
        )
        rate = brain._get_recent_win_rate(lookback_trades=15)
        assert rate is None  # triggers fallback path in caller
        empty_journal.close()

    def test_brain_uses_journal_when_ledger_insufficient_but_journal_has_history(self, tmp_path):
        """Insufficient broker-ledger history should not hide strategy-journal history."""
        from algotrader.core.models import OrderSide, TradeRecord
        from algotrader.strategy_selector.brain import DailyBrain
        from algotrader.tracking.journal import TradeJournal

        ledger = _make_ledger(tmp_path)
        journal = TradeJournal(db_path=ledger._db_path)
        ts = datetime.now(pytz.UTC).isoformat()
        for i in range(3):
            _fill(ledger, exec_id=f"L_{i}", perm_id=400 + i,
                  realized_pnl=100.0, commission=0.65, ts=ts)

        base_time = datetime.now(pytz.UTC) - timedelta(days=1)
        for i, pnl in enumerate([100.0, 120.0, -50.0, 80.0, -25.0]):
            journal.record_trade(
                TradeRecord(
                    strategy_name="options_premium",
                    symbol="SPY",
                    side=OrderSide.SELL,
                    qty=1,
                    entry_price=1.0,
                    exit_price=0.5,
                    entry_time=base_time + timedelta(minutes=i),
                    exit_time=base_time + timedelta(minutes=i + 1),
                    realized_pnl=pnl,
                )
            )

        brain = DailyBrain(
            total_capital=100_000,
            trade_journal=journal,
            broker_ledger=ledger,
            recent_win_rate_fallback=0.80,
        )
        rate = brain._get_recent_win_rate(lookback_trades=15)
        assert rate == pytest.approx(0.6)
        journal.close()

    def test_brain_falls_back_to_journal_when_no_ledger(self, tmp_path):
        """Brain uses strategy-journal path when broker_ledger=None, returns None for empty DB."""
        from algotrader.strategy_selector.brain import DailyBrain
        from algotrader.tracking.journal import TradeJournal

        # Point journal at a fresh empty DB in tmp_path
        empty_journal = TradeJournal(db_path=str(tmp_path / "empty.db"))
        brain = DailyBrain(
            total_capital=100_000,
            trade_journal=empty_journal,
            broker_ledger=None,
        )
        # Empty DB → < 5 rows → returns None
        rate = brain._get_recent_win_rate(lookback_trades=15)
        assert rate is None


# ---------------------------------------------------------------------------
# Reconciliation helper — detect orphaned / phantom legs
# ---------------------------------------------------------------------------

class TestReconciliation:
    """
    Tests for _collect_strategy_tracked_legs and the reconciliation logic.

    We test the BrokerLedger's get_open_option_legs() vs a fake strategy._trades
    dict, mirroring what _reconcile_broker_vs_strategy does.
    """

    def _make_premium_trade(
        self,
        underlying: str = "SPY",
        structure: str = "put_spread",
        short_strike: float = 560.0,
        long_strike: float = 555.0,
        expiration_str: str = "20260502",
        call_short_strike: float | None = None,
        call_long_strike: float | None = None,
    ):
        from datetime import datetime
        from algotrader.strategies.options_premium import PremiumTrade

        expiration = datetime.strptime(expiration_str, "%Y%m%d").date()
        return PremiumTrade(
            underlying=underlying,
            structure=structure,
            short_strike=short_strike,
            long_strike=long_strike,
            call_short_strike=call_short_strike,
            call_long_strike=call_long_strike,
            expiration=expiration,
        )

    def test_orphaned_broker_leg_detected(self, tmp_path):
        """
        Broker has SPY 560P short, strategy._trades is empty
        → broker_legs - strategy_legs = {("SPY","20260502",560.0,"P")}
        """
        ledger = _make_ledger(tmp_path)
        _fill(ledger, exec_id="O1", perm_id=1001, side="SLD",
              symbol="SPY", strike=560.0, right="P", expiry="20260502",
              ts=_today_ts())

        broker_legs = {
            (r["symbol"], r["expiry"], r["strike"], r["right"])
            for r in ledger.get_open_option_legs()
        }
        strategy_legs: set = set()

        orphans = broker_legs - strategy_legs
        assert len(orphans) == 1
        assert ("SPY", "20260502", 560.0, "P") in orphans

    def test_phantom_strategy_leg_detected(self, tmp_path):
        """
        Strategy._trades has QQQ 444P, broker has nothing
        → strategy_legs - broker_legs = {("QQQ","20260502",444.0,"P")}
        """
        from datetime import date as dt_date

        class _FakeStrategy:
            _trades = {}

        fake = _FakeStrategy()
        trade = self._make_premium_trade(
            underlying="QQQ", short_strike=444.0, long_strike=440.0,
            expiration_str="20260502",
        )
        fake._trades["QQQ"] = trade

        # Build legs from the fake strategy (mirrors _collect_strategy_tracked_legs)
        strategy_legs: set = set()
        for t in fake._trades.values():
            exp = t.expiration.strftime("%Y%m%d")
            strategy_legs.add((t.underlying, exp, float(t.short_strike), "P"))
            strategy_legs.add((t.underlying, exp, float(t.long_strike), "P"))

        ledger = _make_ledger(tmp_path)
        broker_legs: set = {
            (r["symbol"], r["expiry"], r["strike"], r["right"])
            for r in ledger.get_open_option_legs()
        }

        phantoms = strategy_legs - broker_legs
        assert ("QQQ", "20260502", 444.0, "P") in phantoms
        assert ("QQQ", "20260502", 440.0, "P") in phantoms

    def test_matched_legs_no_discrepancy(self, tmp_path):
        """
        Broker has SPY 560P short + 555P long, strategy tracks the same spread
        → symmetric difference is empty.
        """
        ledger = _make_ledger(tmp_path)
        ts = _today_ts()
        _fill(ledger, exec_id="S1", perm_id=1001, side="SLD",
              strike=560.0, right="P", expiry="20260502", ts=ts)
        _fill(ledger, exec_id="L1", perm_id=1001, side="BOT",
              strike=555.0, right="P", expiry="20260502", ts=ts)

        broker_legs = {
            (r["symbol"], r["expiry"], r["strike"], r["right"])
            for r in ledger.get_open_option_legs()
        }

        strategy_legs = {
            ("SPY", "20260502", 560.0, "P"),
            ("SPY", "20260502", 555.0, "P"),
        }

        assert broker_legs == strategy_legs
