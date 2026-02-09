"""Query live positions, orders, account from Alpaca.

Usage: python scripts/check_live.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    from algotrader.core.config import load_settings
    from algotrader.execution.alpaca_executor import AlpacaExecutor
    from algotrader.tracking.journal import TradeJournal

    settings = load_settings()
    executor = AlpacaExecutor(config=settings.alpaca)

    # Account summary
    account = executor.get_account()
    print("=" * 60)
    print("ACCOUNT SUMMARY")
    print("=" * 60)
    print(f"  Equity:        ${account.equity:,.2f}")
    print(f"  Cash:          ${account.cash:,.2f}")
    print(f"  Buying Power:  ${account.buying_power:,.2f}")
    print(f"  Day Trades:    {account.day_trade_count}")
    print(f"  PDT Flag:      {account.pattern_day_trader}")
    print()

    # Open positions
    positions = executor.get_positions()
    print("=" * 60)
    print(f"OPEN POSITIONS ({len(positions)})")
    print("=" * 60)
    if positions:
        print(f"  {'Symbol':<10} {'Qty':>8} {'Entry':>10} {'Current':>10} {'P&L':>12} {'P&L%':>8}")
        print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
        total_unrealized = 0.0
        for p in positions:
            total_unrealized += p.unrealized_pnl
            print(
                f"  {p.symbol:<10} {p.qty:>8.1f} "
                f"${p.avg_entry_price:>9.2f} ${p.current_price:>9.2f} "
                f"${p.unrealized_pnl:>+11.2f} {p.unrealized_pnl_pct:>+7.2f}%"
            )
        print(f"\n  Total Unrealized P&L: ${total_unrealized:+,.2f}")
    else:
        print("  No open positions.")
    print()

    # Open orders
    orders = executor.get_open_orders()
    print("=" * 60)
    print(f"OPEN ORDERS ({len(orders)})")
    print("=" * 60)
    if orders:
        print(f"  {'Symbol':<10} {'Side':<6} {'Qty':>8} {'Type':<8} {'Status':<10}")
        print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
        for o in orders:
            print(
                f"  {o.symbol:<10} {o.side.value:<6} {o.qty:>8.1f} "
                f"{o.order_type.value:<8} {o.status.value:<10}"
            )
    else:
        print("  No open orders.")
    print()

    # Today's trade count from journal
    try:
        journal = TradeJournal()
        summary = journal.get_daily_summary()
        print("=" * 60)
        print("TODAY'S JOURNAL")
        print("=" * 60)
        print(f"  Trades:    {summary['total_trades']}")
        print(f"  Wins:      {summary['wins']}")
        print(f"  Losses:    {summary['losses']}")
        print(f"  Total P&L: ${summary['total_pnl']:+,.2f}")
        journal.close()
    except Exception as e:
        print(f"  Journal unavailable: {e}")

    print()


if __name__ == "__main__":
    main()
