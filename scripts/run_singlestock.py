"""Entry point for the standalone single-stock day-trading tool.

Runs as its own process alongside the main bot. Uses its own IBKR
client_id, its own lockfile, and its own log file. Reads the same
config/settings.yaml as the main bot but only consumes the
``singlestock:`` block.

Usage:
    uv run python scripts/run_singlestock.py [--dry-run]
"""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    from algotrader.core.config import load_settings
    from algotrader.singlestock.lockfile import acquire_lock, release_lock

    dry_run = "--dry-run" in sys.argv

    config_path = os.getenv("ALGOTRADER_CONFIG", "config/settings.yaml")
    settings = load_settings(config_path)

    if not settings.singlestock.enabled:
        print("singlestock.enabled is false in config. Nothing to do.")
        sys.exit(0)

    # Safety: never run on live ports during development.
    if settings.broker.ibkr.port in (7496, 4001):
        print("ERROR: IBKR live port detected. Single-stock tool refuses live mode.")
        sys.exit(1)

    if not settings.trading.paper_mode:
        print("ERROR: trading.paper_mode is false. Single-stock tool requires paper.")
        sys.exit(1)

    # Capital coordination contract
    if settings.trading.reserved_for_singlestock_pct < settings.singlestock.capital_pct:
        print(
            "ERROR: trading.reserved_for_singlestock_pct "
            f"({settings.trading.reserved_for_singlestock_pct}) is less than "
            f"singlestock.capital_pct ({settings.singlestock.capital_pct}). "
            "Main bot will double-count buying power. Fix settings.yaml."
        )
        sys.exit(1)

    try:
        import ib_async  # noqa: F401
    except ModuleNotFoundError:
        print("ERROR: ib_async is not installed in the current Python environment.")
        sys.exit(1)

    acquire_lock(settings.singlestock.lock_file)
    atexit.register(release_lock, settings.singlestock.lock_file)

    print(
        f"Single-stock starting (symbol={settings.singlestock.symbol}, "
        f"client_id={settings.singlestock.ibkr_client_id}, dry_run={dry_run})"
    )

    # Import after lock acquisition so a duplicate launch fails fast
    # without triggering connection logic.
    from algotrader.singlestock.orchestrator import SingleStockOrchestrator

    orch = SingleStockOrchestrator(settings, dry_run=dry_run)
    try:
        orch.run()
    finally:
        release_lock(settings.singlestock.lock_file)


if __name__ == "__main__":
    main()
