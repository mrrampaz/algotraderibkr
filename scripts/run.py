"""Main entry point for the algotrader system."""

from __future__ import annotations

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    # Verify paper trading mode
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

    from algotrader.core.config import load_settings
    from algotrader.orchestrator import Orchestrator

    config_path = os.getenv("ALGOTRADER_CONFIG", "config/settings.yaml")
    settings = load_settings(config_path)

    # Safety check: ensure paper mode in config matches env
    if not settings.trading.paper_mode:
        print("ERROR: trading.paper_mode is false in config.")
        print("Set paper_mode: true in config/settings.yaml for safety.")
        sys.exit(1)

    print(f"AlgoTrader starting (paper={paper}, config={config_path})")

    orchestrator = Orchestrator(settings)
    orchestrator.run()


if __name__ == "__main__":
    main()
