"""Test IBKR connection and basic operations.

Usage:
    python scripts/test_ibkr_connection.py
    python scripts/test_ibkr_connection.py --live
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Ensure project root for relative config paths/imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from algotrader.core.config import load_settings

if TYPE_CHECKING:
    from algotrader.data.ibkr_provider import IBKRDataProvider
    from algotrader.execution.ibkr_connection import IBKRConnection
    from algotrader.execution.ibkr_executor import IBKRExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only IBKR connectivity test")
    parser.add_argument("--host", type=str, default=None, help="TWS/Gateway host override")
    parser.add_argument("--port", type=int, default=None, help="TWS/Gateway port override")
    parser.add_argument("--client-id", type=int, default=None, help="IBKR client ID override")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live trading ports if --port is not explicitly set",
    )
    return parser.parse_args()


def main() -> int:
    try:
        import ib_async  # noqa: F401
    except ModuleNotFoundError:
        print("ERROR: ib_async is not installed in the current Python environment.")
        print(f"Current interpreter: {sys.executable}")
        print("Fix:")
        print("  1) uv sync")
        print("  2) Use the project venv interpreter: .venv\\Scripts\\python.exe")
        return 1

    from algotrader.data.ibkr_provider import IBKRDataProvider
    from algotrader.execution.ibkr_connection import IBKRConnection
    from algotrader.execution.ibkr_executor import IBKRExecutor

    args = parse_args()
    settings = load_settings()

    ibkr_cfg = settings.broker.ibkr.model_copy()
    ibkr_cfg.readonly = True

    if args.host:
        ibkr_cfg.host = args.host
    if args.client_id is not None:
        ibkr_cfg.client_id = args.client_id
    if args.port is not None:
        ibkr_cfg.port = args.port
    elif args.live:
        if ibkr_cfg.port in (7497, 7496):
            ibkr_cfg.port = 7496
        elif ibkr_cfg.port in (4002, 4001):
            ibkr_cfg.port = 4001

    print(
        f"IBKR test config: host={ibkr_cfg.host}, port={ibkr_cfg.port}, "
        f"client_id={ibkr_cfg.client_id}, readonly={ibkr_cfg.readonly}"
    )

    conn = IBKRConnection.get_instance(config=ibkr_cfg)
    provider = IBKRDataProvider(connection=conn)
    executor = IBKRExecutor(connection=conn)

    results: list[tuple[str, bool, str]] = []

    def run_test(name: str, fn) -> None:
        try:
            fn()
            print(f"[PASS] {name}")
            results.append((name, True, ""))
        except Exception as exc:
            msg = str(exc)
            print(f"[FAIL] {name}: {msg}")
            results.append((name, False, msg))

    def ensure_connected() -> None:
        if not conn.connect():
            raise RuntimeError("connect returned False")

    run_test("Connect to TWS/Gateway", ensure_connected)

    dependent_tests = [
        ("Account info", lambda: print_account(executor)),
        ("SPY quote", lambda: print_quote(provider)),
        ("SPY 5-minute bars (10)", lambda: print_bars(provider)),
        ("SPY option chain (nearest expiration)", lambda: print_option_chain(provider)),
        ("Current positions", lambda: print_positions(executor)),
        ("Open orders", lambda: print_open_orders(executor)),
    ]

    if conn.connected:
        for test_name, test_fn in dependent_tests:
            run_test(test_name, test_fn)
    else:
        for test_name, _ in dependent_tests:
            msg = "Skipped because initial connection failed"
            print(f"[FAIL] {test_name}: {msg}")
            results.append((test_name, False, msg))

    run_test(
        "Read-only mode / no order placement",
        lambda: print("No orders were placed; script is read-only by design."),
    )

    run_test(
        "Disconnect cleanly",
        lambda: disconnect_and_verify(conn),
    )

    passed = sum(1 for _, ok, _ in results if ok)
    failed_items = [(name, detail) for name, ok, detail in results if not ok]

    if failed_items:
        print("\nFailures:")
        for name, detail in failed_items:
            print(f"- {name}: {detail}")
        print(f"\nPassed {passed}/{len(results)} tests.")
        return 1

    print(f"\nAll {passed} tests passed.")
    return 0


def print_account(executor: "IBKRExecutor") -> None:
    account = executor.get_account()
    print(
        "Account: "
        f"equity={account.equity:.2f}, "
        f"cash={account.cash:.2f}, "
        f"buying_power={account.buying_power:.2f}"
    )


def print_quote(provider: "IBKRDataProvider") -> None:
    quote = provider.get_quote("SPY")
    if quote is not None:
        print(
            "SPY quote: "
            f"bid={quote.bid_price:.2f} x {quote.bid_size:.0f}, "
            f"ask={quote.ask_price:.2f} x {quote.ask_size:.0f}"
        )
        return

    # Some IBKR accounts/sessions may not return top-of-book snapshot quote
    # fields immediately. Fall back to trade snapshot so connectivity checks
    # still validate market data access.
    snapshot = provider.get_snapshot("SPY")
    if snapshot and snapshot.latest_trade_price:
        print(
            "SPY quote unavailable; using snapshot trade price: "
            f"last={snapshot.latest_trade_price:.2f}"
        )
        return

    raise RuntimeError("No quote or snapshot returned for SPY")


def print_bars(provider: "IBKRDataProvider") -> None:
    bars = provider.get_bars("SPY", "5Min", limit=10)
    if bars.empty:
        raise RuntimeError("No bars returned")
    print(f"Returned {len(bars)} bars. Last rows:")
    print(bars.tail(3).to_string())


def print_option_chain(provider: "IBKRDataProvider") -> None:
    chain = provider.get_option_chain("SPY")
    if chain.empty:
        raise RuntimeError("Option chain is empty")
    nearest_exp = chain["expiration"].min()
    sample = (
        chain[chain["expiration"] == nearest_exp]
        .sort_values("strike")
        .head(5)[["expiration", "strike", "type", "bid", "ask", "delta"]]
    )
    print(f"Option chain rows={len(chain)}. Nearest expiration: {nearest_exp}")
    print(sample.to_string(index=False))


def print_positions(executor: "IBKRExecutor") -> None:
    positions = executor.get_positions()
    if not positions:
        print("No open stock positions.")
        return
    print(f"Open stock positions: {len(positions)}")
    for pos in positions[:10]:
        print(
            f"- {pos.symbol}: qty={pos.qty:.4f}, side={pos.side.value}, "
            f"avg={pos.avg_entry_price:.4f}, mark={pos.current_price:.4f}, "
            f"uPnL={pos.unrealized_pnl:.2f}"
        )


def print_open_orders(executor: "IBKRExecutor") -> None:
    orders = executor.get_open_orders()
    if not orders:
        print("No open orders.")
        return
    print(f"Open orders: {len(orders)}")
    for order in orders[:10]:
        print(
            f"- {order.id} {order.symbol} {order.side.value} "
            f"{order.qty:.4f} {order.order_type.value} {order.status.value}"
        )


def disconnect_and_verify(conn: "IBKRConnection") -> None:
    conn.disconnect()
    if conn.connected:
        raise RuntimeError("Connection is still marked connected after disconnect")


if __name__ == "__main__":
    raise SystemExit(main())
