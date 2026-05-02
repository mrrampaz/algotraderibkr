from __future__ import annotations

import logging
import re
from datetime import datetime

import pytest

from algotrader.core.logging import ET, ETMidnightHandler, setup_logging


@pytest.fixture(autouse=True)
def close_root_handlers():
    yield
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()


def _configured_file_handler() -> ETMidnightHandler:
    for handler in logging.getLogger().handlers:
        if isinstance(handler, ETMidnightHandler):
            return handler
    raise AssertionError("ETMidnightHandler was not configured")


def test_log_handler_rotates_daily_at_et_midnight(tmp_path) -> None:
    """Verify daily rotation handler is configured for US/Eastern."""

    setup_logging(log_file=str(tmp_path / "algotrader.log"))

    handler = _configured_file_handler()
    assert handler.rollover_timezone is ET
    assert handler.when == "MIDNIGHT"

    current_time = datetime(2026, 5, 2, 15, 30, tzinfo=ET).timestamp()
    rollover = datetime.fromtimestamp(handler.computeRollover(current_time), ET)
    assert rollover == datetime(2026, 5, 3, 0, 0, tzinfo=ET)


def test_log_handler_keeps_30_days(tmp_path) -> None:
    """Backup count is 30 to preserve roughly 6 weeks of trading sessions."""

    setup_logging(log_file=str(tmp_path / "algotrader.log"))

    handler = _configured_file_handler()
    assert handler.backupCount == 30


def test_log_handler_suffix_is_iso_date(tmp_path) -> None:
    setup_logging(log_file=str(tmp_path / "algotrader.log"))

    handler = _configured_file_handler()
    assert handler.suffix == "%Y-%m-%d"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", datetime.now(ET).strftime(handler.suffix))
