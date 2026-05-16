"""Pattern-Day-Trader guard for accounts under $25k.

When pdt_safe_mode is true, refuse to CLOSE a position opened in the
same trading session. The position is allowed to roll to next morning
and close then. This is conservative; small accounts can easily
trip 4 day-trades in 5 business days otherwise.

For accounts >= $25k (default for the paper account currently in use),
this guard is a no-op.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytz
import structlog

ET = pytz.timezone("America/New_York")
logger = structlog.get_logger()


class PDTGuard:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._log = logger.bind(component="pdt_guard", enabled=enabled)

    def can_close_today(self, entry_time: datetime) -> bool:
        if not self._enabled:
            return True
        if entry_time is None:
            return True

        now_et = datetime.now(ET)
        entry_et = entry_time.astimezone(ET) if entry_time.tzinfo else ET.localize(entry_time)
        same_day = entry_et.date() == now_et.date()
        if same_day:
            self._log.warning(
                "pdt_guard_blocking_same_day_close",
                entry_time=entry_et.isoformat(),
                now=now_et.isoformat(),
            )
            return False
        return True
