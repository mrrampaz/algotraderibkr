"""JSON-persisted state for the single-stock tool.

Stores the current open position (if any), today's thesis, news baseline
for mid-day delta checks, daily P&L counter, and a daily-call counter
for the LLM. Survives process restarts.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz
import structlog

ET = pytz.timezone("America/New_York")
logger = structlog.get_logger()


@dataclass
class OpenPosition:
    con_id: int
    local_symbol: str
    right: str
    strike: float
    expiry: str
    qty: int
    entry_premium: float
    entry_time: datetime
    direction: str  # "long" (call) or "short" (put)
    underlying_at_entry: float
    stop_underlying: float
    target_premium: float
    client_order_id: str
    days_held: int = 0


@dataclass
class DailyCounters:
    et_date: str
    llm_calls: int = 0
    realized_pnl_dollars: float = 0.0
    entries_today: int = 0
    closes_today: int = 0


class SingleStockState:
    """Persistent state — load on startup, save after every mutation."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._log = logger.bind(component="singlestock_state", path=str(self._path))

        # In-memory state
        self.open_position: OpenPosition | None = None
        self.thesis_json: dict[str, Any] | None = None
        self.news_baseline_ids: list[str] = []
        self.counters: DailyCounters = DailyCounters(et_date=self._today_et())

        self._load()

    @staticmethod
    def _today_et() -> str:
        return datetime.now(ET).strftime("%Y-%m-%d")

    def _load(self) -> None:
        if not self._path.exists():
            self._log.info("singlestock_state_no_file_starting_fresh")
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._log.exception("singlestock_state_load_failed")
            return

        # Open position
        op = raw.get("open_position")
        if op:
            try:
                op["entry_time"] = datetime.fromisoformat(op["entry_time"])
                self.open_position = OpenPosition(**op)
            except Exception:
                self._log.exception("singlestock_state_position_decode_failed")
                self.open_position = None

        self.thesis_json = raw.get("thesis")
        self.news_baseline_ids = list(raw.get("news_baseline_ids", []))

        counters = raw.get("counters", {})
        et_today = self._today_et()
        if counters.get("et_date") == et_today:
            self.counters = DailyCounters(
                et_date=et_today,
                llm_calls=int(counters.get("llm_calls", 0)),
                realized_pnl_dollars=float(counters.get("realized_pnl_dollars", 0.0)),
                entries_today=int(counters.get("entries_today", 0)),
                closes_today=int(counters.get("closes_today", 0)),
            )
        else:
            # New day — reset counters; persist for posterity below
            self.counters = DailyCounters(et_date=et_today)

        self._log.info(
            "singlestock_state_loaded",
            has_position=self.open_position is not None,
            llm_calls_today=self.counters.llm_calls,
        )

    def save(self) -> None:
        et_today = self._today_et()
        if self.counters.et_date != et_today:
            # Lazy day-reset on save too
            self.counters = DailyCounters(et_date=et_today)

        op_dict = None
        if self.open_position:
            d = asdict(self.open_position)
            d["entry_time"] = self.open_position.entry_time.isoformat()
            op_dict = d

        payload = {
            "open_position": op_dict,
            "thesis": self.thesis_json,
            "news_baseline_ids": self.news_baseline_ids,
            "counters": asdict(self.counters),
            "saved_at": datetime.now(ET).isoformat(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self._path)

    # ── Mutators ────────────────────────────────────────────────────────────

    def record_entry(self, position: OpenPosition) -> None:
        self.open_position = position
        self.counters.entries_today += 1
        self.save()

    def record_close(self, realized_pnl_dollars: float) -> None:
        self.counters.realized_pnl_dollars += realized_pnl_dollars
        self.counters.closes_today += 1
        self.open_position = None
        self.save()

    def update_thesis(self, thesis_json: dict[str, Any]) -> None:
        self.thesis_json = thesis_json
        self.save()

    def update_news_baseline(self, ids: list[str]) -> None:
        self.news_baseline_ids = list(ids)
        self.save()

    def increment_llm_calls(self, n: int = 1) -> None:
        self.counters.llm_calls += n
        self.save()
