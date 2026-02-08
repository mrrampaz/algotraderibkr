"""SQLite-backed trade journal for recording every trade with full context."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any

import pytz
import structlog

from algotrader.core.models import TradeRecord, OrderSide, RegimeType

logger = structlog.get_logger()

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    entry_time TEXT,
    exit_time TEXT,
    realized_pnl REAL DEFAULT 0.0,
    unrealized_pnl REAL DEFAULT 0.0,
    conviction REAL DEFAULT 1.0,
    regime TEXT,
    entry_reason TEXT DEFAULT '',
    exit_reason TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
)
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name)",
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)",
]


class TradeJournal:
    """SQLite-backed trade journal.

    Records every trade with full context: strategy, regime, conviction,
    entry/exit reasons, and arbitrary metadata.
    """

    def __init__(self, db_path: str = "data/journal/trades.db") -> None:
        self._db_path = db_path
        self._log = logger.bind(component="trade_journal")

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._log.info("trade_journal_initialized", db_path=db_path)

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        cursor = self._conn.cursor()
        cursor.execute(CREATE_TABLE_SQL)
        for sql in CREATE_INDEX_SQL:
            cursor.execute(sql)
        self._conn.commit()

    def record_trade(self, trade: TradeRecord) -> str:
        """Record a trade. Returns the trade ID."""
        if not trade.id:
            trade.id = str(uuid.uuid4())

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO trades (id, strategy_name, symbol, side, qty,
                entry_price, exit_price, entry_time, exit_time,
                realized_pnl, unrealized_pnl, conviction, regime,
                entry_reason, exit_reason, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.id,
                trade.strategy_name,
                trade.symbol,
                trade.side.value,
                trade.qty,
                trade.entry_price,
                trade.exit_price,
                trade.entry_time.isoformat() if trade.entry_time else None,
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.realized_pnl,
                trade.unrealized_pnl,
                trade.conviction,
                trade.regime.value if trade.regime else None,
                trade.entry_reason,
                trade.exit_reason,
                json.dumps(trade.metadata, default=str),
                datetime.now(pytz.UTC).isoformat(),
            ),
        )
        self._conn.commit()
        self._log.info(
            "trade_recorded",
            trade_id=trade.id,
            strategy=trade.strategy_name,
            symbol=trade.symbol,
            side=trade.side.value,
            pnl=trade.realized_pnl,
        )
        return trade.id

    def update_trade(self, trade_id: str, **updates: Any) -> None:
        """Update fields on an existing trade record."""
        if not updates:
            return

        # Handle special serialization
        if "exit_time" in updates and isinstance(updates["exit_time"], datetime):
            updates["exit_time"] = updates["exit_time"].isoformat()
        if "entry_time" in updates and isinstance(updates["entry_time"], datetime):
            updates["entry_time"] = updates["entry_time"].isoformat()
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"], default=str)
        if "regime" in updates and isinstance(updates["regime"], RegimeType):
            updates["regime"] = updates["regime"].value
        if "side" in updates and isinstance(updates["side"], OrderSide):
            updates["side"] = updates["side"].value

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [trade_id]

        cursor = self._conn.cursor()
        cursor.execute(f"UPDATE trades SET {set_clause} WHERE id = ?", values)
        self._conn.commit()

    def get_trades(
        self,
        strategy_name: str | None = None,
        symbol: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 100,
    ) -> list[TradeRecord]:
        """Query trades with optional filters."""
        conditions = []
        params: list[Any] = []

        if strategy_name:
            conditions.append("strategy_name = ?")
            params.append(strategy_name)
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_date:
            conditions.append("entry_time >= ?")
            params.append(start_date.isoformat())
        if end_date:
            conditions.append("entry_time <= ?")
            params.append(end_date.isoformat() + "T23:59:59")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        cursor = self._conn.cursor()
        cursor.execute(
            f"SELECT * FROM trades {where} ORDER BY entry_time DESC LIMIT ?",
            params,
        )

        return [self._row_to_trade(row) for row in cursor.fetchall()]

    def get_daily_summary(self, trade_date: date | None = None) -> dict[str, Any]:
        """Get summary statistics for a trading day."""
        if trade_date is None:
            trade_date = datetime.now(pytz.UTC).date()

        date_str = trade_date.isoformat()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl >= 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
                SUM(realized_pnl) as total_pnl,
                AVG(realized_pnl) as avg_pnl,
                MAX(realized_pnl) as best_trade,
                MIN(realized_pnl) as worst_trade
            FROM trades
            WHERE entry_time >= ? AND entry_time < date(?, '+1 day')
            """,
            (date_str, date_str),
        )
        row = cursor.fetchone()

        total_trades = row["total_trades"] or 0
        wins = row["wins"] or 0
        losses = row["losses"] or 0

        return {
            "date": date_str,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_trades if total_trades > 0 else 0,
            "total_pnl": row["total_pnl"] or 0,
            "avg_pnl": row["avg_pnl"] or 0,
            "best_trade": row["best_trade"] or 0,
            "worst_trade": row["worst_trade"] or 0,
        }

    def get_strategy_summary(self, strategy_name: str, days: int = 30) -> dict[str, Any]:
        """Get summary statistics for a strategy over recent days."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl >= 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
                SUM(realized_pnl) as total_pnl,
                AVG(realized_pnl) as avg_pnl
            FROM trades
            WHERE strategy_name = ?
              AND entry_time >= date('now', ?)
            """,
            (strategy_name, f"-{days} days"),
        )
        row = cursor.fetchone()

        total_trades = row["total_trades"] or 0
        wins = row["wins"] or 0

        return {
            "strategy": strategy_name,
            "days": days,
            "total_trades": total_trades,
            "wins": wins,
            "losses": row["losses"] or 0,
            "win_rate": wins / total_trades if total_trades > 0 else 0,
            "total_pnl": row["total_pnl"] or 0,
            "avg_pnl": row["avg_pnl"] or 0,
        }

    def _row_to_trade(self, row: sqlite3.Row) -> TradeRecord:
        """Convert a database row to a TradeRecord."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        return TradeRecord(
            id=row["id"],
            strategy_name=row["strategy_name"],
            symbol=row["symbol"],
            side=OrderSide(row["side"]),
            qty=row["qty"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            entry_time=datetime.fromisoformat(row["entry_time"]) if row["entry_time"] else None,
            exit_time=datetime.fromisoformat(row["exit_time"]) if row["exit_time"] else None,
            realized_pnl=row["realized_pnl"],
            unrealized_pnl=row["unrealized_pnl"],
            conviction=row["conviction"],
            regime=RegimeType(row["regime"]) if row["regime"] else None,
            entry_reason=row["entry_reason"] or "",
            exit_reason=row["exit_reason"] or "",
            metadata=metadata,
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
