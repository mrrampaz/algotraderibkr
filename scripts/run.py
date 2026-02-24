"""Main entry point for the algotrader system."""

from __future__ import annotations

import os
import sys

# Ensure working directory is project root (needed for relative config paths)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    from algotrader.core.config import load_settings
    from algotrader.orchestrator import Orchestrator

    config_path = os.getenv("ALGOTRADER_CONFIG", "config/settings.yaml")
    settings = load_settings(config_path)
    broker_provider = settings.broker.provider.lower()

    if broker_provider == "alpaca":
        # Verify Alpaca paper mode
        paper = os.getenv("ALPACA_PAPER_TRADE", "True").lower() in ("true", "1", "yes")
        if not paper:
            print("ERROR: ALPACA_PAPER_TRADE is not set to True.")
            print("Set ALPACA_PAPER_TRADE=True in your .env file for safety.")
            sys.exit(1)

        api_key = os.getenv("ALPACA_API_KEY", "")
        if not api_key:
            print("ERROR: ALPACA_API_KEY is not set.")
            print("Copy .env.example to .env and add your Alpaca credentials.")
            sys.exit(1)
    elif broker_provider == "ibkr":
        try:
            import ib_async  # noqa: F401
        except ModuleNotFoundError:
            print("ERROR: ib_async is not installed in the current Python environment.")
            print(f"Current interpreter: {sys.executable}")
            print("Fix:")
            print("  1) uv sync")
            print("  2) Use the project venv interpreter: .venv\\Scripts\\python.exe")
            sys.exit(1)

        if settings.broker.ibkr.port in (7496, 4001):
            print("ERROR: IBKR live port detected in config.")
            print("Use paper ports 7497 (TWS) or 4002 (Gateway) for development.")
            sys.exit(1)
    else:
        print(f"ERROR: Unsupported broker provider '{settings.broker.provider}'.")
        print("Use broker.provider: alpaca or broker.provider: ibkr")
        sys.exit(1)

    # Safety check: ensure paper mode in config matches env
    if not settings.trading.paper_mode:
        print("ERROR: trading.paper_mode is false in config.")
        print("Set paper_mode: true in config/settings.yaml for safety.")
        sys.exit(1)

    print(f"AlgoTrader starting (broker={broker_provider}, config={config_path})")

    orchestrator = Orchestrator(settings)
    orchestrator.run()


if __name__ == "__main__":
    main()
