"""Structured logging setup using structlog."""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

import structlog

ET = ZoneInfo("America/New_York")


class ETMidnightHandler(TimedRotatingFileHandler):
    """TimedRotatingFileHandler that rotates at midnight US/Eastern."""

    rollover_timezone = ET

    def computeRollover(self, currentTime: float) -> int:
        """Return the next midnight timestamp in US/Eastern."""

        current_et = datetime.fromtimestamp(currentTime, self.rollover_timezone)
        next_midnight_et = (current_et + timedelta(days=1)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        return int(next_midnight_et.timestamp())

    def doRollover(self) -> None:
        """Rotate using an ET date suffix, independent of system timezone."""

        current_time = int(time.time())
        rollover_et = datetime.fromtimestamp(self.rolloverAt, self.rollover_timezone)
        suffix_date = rollover_et - timedelta(days=1)
        dfn = self.rotation_filename(
            self.baseFilename + "." + suffix_date.strftime(self.suffix)
        )
        if os.path.exists(dfn):
            return

        if self.stream:
            self.stream.close()
            self.stream = None
        self.rotate(self.baseFilename, dfn)
        if self.backupCount > 0:
            for filename in self.getFilesToDelete():
                os.remove(filename)
        if not self.delay:
            self.stream = self._open()
        self.rolloverAt = self.computeRollover(current_time)


class _IBKRFilledOrderNoiseFilter(logging.Filter):
    """Downgrade harmless IBKR "already filled" wrapper errors to debug noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "ib_async.wrapper":
            return True

        try:
            message = record.getMessage()
        except Exception:
            return True

        if "Error 201" in message and "already filled" in message.lower():
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
        return True


def setup_logging(level: str = "INFO", log_file: str | None = None, json_format: bool = True) -> None:
    """Configure structlog with JSON output to file and human-readable to console."""

    # Ensure log directory exists
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Shared processors for all outputs
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Configure stdlib logging
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler — human-readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
        foreign_pre_chain=shared_processors,
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler — JSON format
    if log_file:
        file_handler = ETMidnightHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
            utc=False,
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(numeric_level)

        if json_format:
            file_formatter = structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
                foreign_pre_chain=shared_processors,
            )
        else:
            file_formatter = structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(colors=False),
                ],
                foreign_pre_chain=shared_processors,
            )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Reduce noisy IBKR wrapper error logs for already-filled orders.
    ib_wrapper_logger = logging.getLogger("ib_async.wrapper")
    if not any(isinstance(f, _IBKRFilledOrderNoiseFilter) for f in ib_wrapper_logger.filters):
        ib_wrapper_logger.addFilter(_IBKRFilledOrderNoiseFilter())
