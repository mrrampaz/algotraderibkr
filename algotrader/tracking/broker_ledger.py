"""Broker fill ledger — source of truth for what IBKR actually executed.

Consumes ib_async execDetailsEvent and commissionReportEvent, persists every
fill to a broker_fills table in the same SQLite DB as trades.db. This table
is the authoritative record for daily P&L summaries, Brain win-rate, and
position reconciliation against strategy state.

trades.db (existing trades table) stays as the strategy-internal accounting
record and is NOT modified by this module.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, timedelta
from typing import Any

import pytz
import structlog

logger = structlog.get_logger()

ET = pytz.timezone("America/New_York")

CREATE_FILLS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS broker_fills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    exec_id     TEXT UNIQUE NOT NULL,
    perm_id     INTEGER,
    order_ref   TEXT,
    account     TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    sec_type    TEXT NOT NULL,
    local_symbol TEXT,
    right       TEXT,
    strike      REAL,
    expiry      TEXT,
    side        TEXT NOT NULL,
    quantity    REAL NOT NULL,
    price       REAL NOT NULL,
    commission  REAL,
    realized_pnl REAL,
    raw_json    TEXT NOT NULL
)
"""

CREATE_FILLS_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_broker_fills_timestamp  ON broker_fills(timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_broker_fills_perm_id    ON broker_fills(perm_id)",
    "CREATE INDEX IF NOT EXISTS idx_broker_fills_order_ref  ON broker_fills(order_ref)",
    "CREATE INDEX IF NOT EXISTS idx_broker_fills_sym_expiry ON broker_fills(symbol, expiry)",
]

# IBKR uses Double.MAX_VALUE as sentinel for "no realized P&L" in opening fills.
_IBKR_PNL_SENTINEL = 1e15


class BrokerLedger:
    """
    Persists every IBKR fill to broker_fills table, provides query methods.

    Pass ib=None to run in test/offline mode (no event wiring).
    """

    def __init__(self, db_path: str, ib: Any = None) -> None:
        from pathlib import Path

        self._db_path = db_path
        self._ib = ib
        self._log = logger.bind(component="broker_ledger")
        self._lock = threading.Lock()

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        if ib is not None:
            self._wire_events()

    # ------------------------------------------------------------------
    # DB setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        conn = self._open_conn()
        try:
            conn.execute(CREATE_FILLS_TABLE_SQL)
            for sql in CREATE_FILLS_INDEXES_SQL:
                conn.execute(sql)
            conn.commit()
        finally:
            conn.close()
        self._log.info("broker_ledger_initialized", db_path=self._db_path)

    def _open_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------

    def _wire_events(self) -> None:
        self._ib.execDetailsEvent += self._on_exec_details
        self._ib.commissionReportEvent += self._on_commission_report
        self._log.info("broker_ledger_events_wired")

    # ------------------------------------------------------------------
    # IBKR event handlers
    # ------------------------------------------------------------------

    def _on_exec_details(self, trade: Any, fill: Any) -> None:
        """INSERT OR IGNORE fill row when execDetails fires."""
        try:
            execution = fill.execution
            contract = fill.contract

            # fill.time is a datetime object from ib_async
            fill_time: datetime = fill.time
            if fill_time is None:
                fill_time = datetime.now(pytz.UTC)
            if fill_time.tzinfo is None:
                fill_time = ET.localize(fill_time)
            ts_utc = fill_time.astimezone(pytz.UTC).isoformat()

            expiry = contract.lastTradeDateOrContractMonth or None
            if expiry == "":
                expiry = None

            right = contract.right or None
            if right in ("", "0", "?"):
                right = None

            strike = float(contract.strike) if contract.strike else None
            if strike == 0.0:
                strike = None

            raw = {
                "execId": execution.execId,
                "time": str(fill_time),
                "acctNumber": execution.acctNumber,
                "side": execution.side,
                "shares": execution.shares,
                "price": execution.price,
                "permId": execution.permId,
                "orderId": execution.orderId,
                "orderRef": execution.orderRef,
                "symbol": contract.symbol,
                "secType": contract.secType,
                "localSymbol": contract.localSymbol,
                "right": contract.right,
                "strike": strike,
                "expiry": expiry,
            }

            with self._lock:
                conn = self._open_conn()
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO broker_fills
                            (exec_id, perm_id, order_ref, account, timestamp_utc,
                             symbol, sec_type, local_symbol, right, strike, expiry,
                             side, quantity, price, raw_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            execution.execId,
                            execution.permId or None,
                            execution.orderRef or None,
                            execution.acctNumber,
                            ts_utc,
                            contract.symbol,
                            contract.secType,
                            contract.localSymbol or None,
                            right,
                            strike,
                            expiry,
                            execution.side,
                            float(execution.shares),
                            float(execution.price),
                            json.dumps(raw),
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()

            self._log.info(
                "broker_fill_recorded",
                exec_id=execution.execId,
                symbol=contract.symbol,
                sec_type=contract.secType,
                side=execution.side,
                qty=execution.shares,
                price=execution.price,
            )
        except Exception:
            self._log.exception("broker_fill_record_failed")

    def _on_commission_report(self, trade: Any, fill: Any, commission_report: Any) -> None:
        """UPDATE fill row with commission and realized_pnl when commissionReport fires."""
        try:
            exec_id = commission_report.execId
            commission = (
                float(commission_report.commission)
                if commission_report.commission is not None
                else None
            )
            realized_pnl_raw = commission_report.realizedPNL
            if realized_pnl_raw is None:
                realized_pnl = None
            else:
                realized_pnl = float(realized_pnl_raw)
                # IBKR uses Double.MAX_VALUE as sentinel for "no realized P&L yet"
                # (opening fills). Treat these as NULL.
                if abs(realized_pnl) > _IBKR_PNL_SENTINEL:
                    realized_pnl = None

            with self._lock:
                conn = self._open_conn()
                try:
                    conn.execute(
                        """
                        UPDATE broker_fills
                           SET commission   = ?,
                               realized_pnl = ?
                         WHERE exec_id = ?
                        """,
                        (commission, realized_pnl, exec_id),
                    )
                    conn.commit()
                finally:
                    conn.close()

            self._log.debug(
                "broker_fill_commission_updated",
                exec_id=exec_id,
                commission=commission,
                realized_pnl=realized_pnl,
            )
        except Exception:
            self._log.exception("broker_commission_update_failed")

    # ------------------------------------------------------------------
    # Direct write methods (used in tests and manual backfill)
    # ------------------------------------------------------------------

    def record_fill(
        self,
        exec_id: str,
        perm_id: int | None,
        order_ref: str | None,
        account: str,
        timestamp_utc: str,
        symbol: str,
        sec_type: str,
        local_symbol: str | None,
        right: str | None,
        strike: float | None,
        expiry: str | None,
        side: str,
        quantity: float,
        price: float,
        commission: float | None = None,
        realized_pnl: float | None = None,
        raw_json: str = "{}",
    ) -> None:
        """Insert a fill directly. Idempotent (INSERT OR IGNORE on exec_id)."""
        with self._lock:
            conn = self._open_conn()
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO broker_fills
                        (exec_id, perm_id, order_ref, account, timestamp_utc,
                         symbol, sec_type, local_symbol, right, strike, expiry,
                         side, quantity, price, commission, realized_pnl, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        exec_id, perm_id, order_ref, account, timestamp_utc,
                        symbol, sec_type, local_symbol, right, strike, expiry,
                        side, quantity, price, commission, realized_pnl, raw_json,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def update_fill_commission(
        self,
        exec_id: str,
        commission: float | None,
        realized_pnl: float | None,
    ) -> None:
        """Update commission/realized_pnl on an existing fill."""
        with self._lock:
            conn = self._open_conn()
            try:
                conn.execute(
                    "UPDATE broker_fills SET commission=?, realized_pnl=? WHERE exec_id=?",
                    (commission, realized_pnl, exec_id),
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_fills_since(self, since_utc: datetime) -> list[dict]:
        """Return all fills at or after since_utc, ordered by timestamp."""
        since_str = since_utc.astimezone(pytz.UTC).isoformat()
        conn = self._open_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM broker_fills WHERE timestamp_utc >= ? ORDER BY timestamp_utc",
                (since_str,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_open_option_legs(self) -> list[dict]:
        """
        Compute net open option legs from all broker_fills.

        For each (symbol, expiry, strike, right) tuple, sums signed quantities:
          BOT = +quantity, SLD = -quantity.

        Returns only legs where |net_qty| > epsilon — i.e., positions that
        have not been fully closed. Used by reconciliation to detect orphans.

        Return shape: [{"symbol", "expiry", "strike", "right", "net_qty"}, ...]
        """
        conn = self._open_conn()
        try:
            cursor = conn.execute(
                """
                SELECT
                    symbol,
                    expiry,
                    strike,
                    right,
                    SUM(CASE WHEN side = 'BOT' THEN quantity ELSE -quantity END) AS net_qty
                FROM broker_fills
                WHERE sec_type   = 'OPT'
                  AND expiry     IS NOT NULL
                  AND right      IS NOT NULL
                  AND strike     IS NOT NULL
                GROUP BY symbol, expiry, strike, right
                HAVING ABS(SUM(CASE WHEN side='BOT' THEN quantity ELSE -quantity END)) > 0.0001
                """
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_realized_pnl_today(self) -> float:
        """Sum realized_pnl for fills recorded today (ET date)."""
        today_et = datetime.now(ET).date()
        return self._realized_pnl_for_date(today_et)

    def _realized_pnl_for_date(self, trade_date: date) -> float:
        et_start = ET.localize(datetime(trade_date.year, trade_date.month, trade_date.day))
        et_end = et_start + timedelta(days=1)
        utc_start = et_start.astimezone(pytz.UTC).isoformat()
        utc_end = et_end.astimezone(pytz.UTC).isoformat()

        conn = self._open_conn()
        try:
            cursor = conn.execute(
                """
                SELECT COALESCE(SUM(realized_pnl), 0.0)
                FROM broker_fills
                WHERE realized_pnl IS NOT NULL
                  AND timestamp_utc >= ?
                  AND timestamp_utc <  ?
                """,
                (utc_start, utc_end),
            )
            row = cursor.fetchone()
            return float(row[0]) if row else 0.0
        finally:
            conn.close()

    def get_recent_closed_round_trips(self, n: int = 30) -> list[dict]:
        """
        Return the last N closed round-trips, grouped by perm_id.

        A "closed round-trip" is a perm_id group that contains at least one
        fill with a non-NULL realized_pnl. IBKR sets realized_pnl on closing
        fills (opening fills arrive with NULL after sentinel filtering).

        net_pnl = sum(realized_pnl) - sum(commission)

        Return shape: [{"perm_id", "symbol", "net_pnl", "gross_pnl",
                         "commission", "close_time", "fill_count"}, ...]
        ordered by close_time DESC.
        """
        conn = self._open_conn()
        try:
            cursor = conn.execute(
                """
                SELECT
                    perm_id,
                    symbol,
                    SUM(CASE WHEN realized_pnl IS NOT NULL THEN realized_pnl ELSE 0.0 END) AS gross_pnl,
                    SUM(CASE WHEN commission   IS NOT NULL THEN commission   ELSE 0.0 END) AS total_commission,
                    MAX(timestamp_utc) AS close_time,
                    COUNT(*)           AS fill_count
                FROM broker_fills
                WHERE perm_id IS NOT NULL
                GROUP BY perm_id, symbol
                HAVING SUM(CASE WHEN realized_pnl IS NOT NULL THEN 1 ELSE 0 END) > 0
                ORDER BY close_time DESC
                LIMIT ?
                """,
                (max(1, int(n)),),
            )
            rows = []
            for row in cursor.fetchall():
                gross = float(row["gross_pnl"])
                comm = float(row["total_commission"])
                rows.append({
                    "perm_id": row["perm_id"],
                    "symbol": row["symbol"],
                    "net_pnl": gross - comm,
                    "gross_pnl": gross,
                    "commission": comm,
                    "close_time": row["close_time"],
                    "fill_count": row["fill_count"],
                })
            return rows
        finally:
            conn.close()

    def get_daily_summary(self, trade_date: date | None = None) -> dict[str, Any]:
        """
        Return a summary dict compatible with TradeJournal.get_daily_summary(),
        but sourced entirely from broker_fills.

        Counts closed round-trips (perm_id groups with non-NULL realized_pnl)
        whose latest fill falls within the ET trading day.
        """
        if trade_date is None:
            trade_date = datetime.now(ET).date()

        et_start = ET.localize(datetime(trade_date.year, trade_date.month, trade_date.day))
        et_end = et_start + timedelta(days=1)
        utc_start = et_start.astimezone(pytz.UTC).isoformat()
        utc_end = et_end.astimezone(pytz.UTC).isoformat()

        conn = self._open_conn()
        try:
            # Each perm_id group that has at least one non-NULL realized_pnl
            # and whose latest fill is within today ET = one closed round-trip.
            cursor = conn.execute(
                """
                SELECT
                    perm_id,
                    SUM(CASE WHEN realized_pnl IS NOT NULL THEN realized_pnl ELSE 0.0 END) AS gross_pnl,
                    SUM(CASE WHEN commission   IS NOT NULL THEN commission   ELSE 0.0 END) AS total_commission,
                    MAX(timestamp_utc) AS close_time
                FROM broker_fills
                WHERE perm_id IS NOT NULL
                GROUP BY perm_id
                HAVING SUM(CASE WHEN realized_pnl IS NOT NULL THEN 1 ELSE 0 END) > 0
                   AND MAX(timestamp_utc) >= ?
                   AND MAX(timestamp_utc) <  ?
                """,
                (utc_start, utc_end),
            )
            round_trips = []
            for row in cursor.fetchall():
                net = float(row["gross_pnl"]) - float(row["total_commission"])
                round_trips.append(net)
        finally:
            conn.close()

        total_trades = len(round_trips)
        wins = sum(1 for pnl in round_trips if pnl > 0)
        losses = total_trades - wins
        total_pnl = sum(round_trips)

        return {
            "date": trade_date.isoformat(),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_trades if total_trades > 0 else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / total_trades if total_trades > 0 else 0.0,
            "best_trade": max(round_trips) if round_trips else 0.0,
            "worst_trade": min(round_trips) if round_trips else 0.0,
            "source": "broker_ledger",
        }
