"""Emergency cleanup: cancel all orders, close all positions via IBKR.

Usage: python scripts/cleanup.py --confirm
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    if "--confirm" not in sys.argv:
        print("WARNING: This will cancel ALL open orders and close ALL positions.")
        print("Run with --confirm to proceed.")
        print()
        print("  python scripts/cleanup.py --confirm")
        sys.exit(1)

    from algotrader.core.config import load_settings
    from algotrader.core.events import EventBus
    from algotrader.execution.ibkr_connection import IBKRConnection
    from algotrader.execution.ibkr_executor import IBKRExecutor

    settings = load_settings()

    if not settings.trading.paper_mode:
        print("ERROR: paper_mode is false. This script only runs in paper mode.")
        sys.exit(1)

    event_bus = EventBus()
    conn = IBKRConnection.get_instance(config=settings.broker.ibkr, event_bus=event_bus)
    if not conn.connect():
        print("ERROR: Could not connect to IBKR TWS/Gateway.")
        sys.exit(1)

    executor = IBKRExecutor(connection=conn)

    # Cancel all open orders
    orders = executor.get_open_orders()
    print(f"Cancelling {len(orders)} open orders...")
    cancelled = 0
    for order in orders:
        try:
            if executor.cancel_order(order.id):
                cancelled += 1
                print(f"  Cancelled: {order.symbol} {order.side.value} {order.qty}")
        except Exception as e:
            print(f"  Failed to cancel {order.id}: {e}")
    print(f"Cancelled {cancelled}/{len(orders)} orders.")
    print()

    # Close all positions
    positions = executor.get_positions()
    print(f"Closing {len(positions)} positions...")
    closed = 0
    for pos in positions:
        try:
            if executor.close_position(pos.symbol):
                closed += 1
                print(f"  Closed: {pos.symbol} ({pos.qty} shares, P&L: ${pos.unrealized_pnl:+.2f})")
        except Exception as e:
            print(f"  Failed to close {pos.symbol}: {e}")
    print(f"Closed {closed}/{len(positions)} positions.")
    print()

    conn.disconnect()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
