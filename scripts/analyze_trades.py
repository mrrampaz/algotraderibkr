"""Historical trade analysis from journal database.

Usage:
  python scripts/analyze_trades.py
  python scripts/analyze_trades.py --strategy pairs_trading
  python scripts/analyze_trades.py --days 60
  python scripts/analyze_trades.py --regime trending_up
  python scripts/analyze_trades.py --export trades.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trade history")
    parser.add_argument("--strategy", type=str, default=None, help="Filter by strategy name")
    parser.add_argument("--days", type=int, default=30, help="Lookback days (default: 30)")
    parser.add_argument("--regime", type=str, default=None, help="Filter analysis by regime type")
    parser.add_argument("--export", type=str, default=None, help="Export trades to CSV file")
    args = parser.parse_args()

    from algotrader.tracking.journal import TradeJournal
    from algotrader.tracking.metrics import MetricsCalculator

    journal = TradeJournal()
    metrics = MetricsCalculator(trade_journal=journal)

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    print("=" * 60)
    print(f"TRADE ANALYSIS — Last {args.days} days")
    if args.strategy:
        print(f"Strategy: {args.strategy}")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 60)
    print()

    # Overall metrics
    m = metrics.calculate(
        strategy_name=args.strategy,
        start_date=start_date,
        end_date=end_date,
    )

    print("PERFORMANCE METRICS")
    print("-" * 40)
    print(f"  Total Trades:          {m.total_trades}")
    print(f"  Win / Loss:            {m.winning_trades} / {m.losing_trades}")
    print(f"  Win Rate:              {m.win_rate:.1%}")
    print(f"  Total P&L:             ${m.total_pnl:+,.2f}")
    print(f"  Avg P&L/Trade:         ${m.avg_pnl_per_trade:+,.2f}")
    print(f"  Avg Winner:            ${m.avg_winner:+,.2f}")
    print(f"  Avg Loser:             ${m.avg_loser:+,.2f}")
    print(f"  Profit Factor:         {m.profit_factor:.2f}")
    print(f"  Expectancy:            ${m.expectancy:+,.2f}")
    print(f"  Largest Win:           ${m.largest_win:+,.2f}")
    print(f"  Largest Loss:          ${m.largest_loss:+,.2f}")
    print(f"  Max Consec. Wins:      {m.max_consecutive_wins}")
    print(f"  Max Consec. Losses:    {m.max_consecutive_losses}")
    print(f"  Avg Hold Time (min):   {m.avg_hold_time_minutes:.1f}")
    sharpe_str = f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio is not None else "N/A"
    print(f"  Sharpe Ratio:          {sharpe_str}")
    print(f"  Max Drawdown:          {m.max_drawdown_pct:.2f}%")
    print()

    # By regime
    if args.strategy:
        print("PERFORMANCE BY REGIME")
        print("-" * 40)
        regime_metrics = metrics.calculate_by_regime(args.strategy, days=args.days)
        if regime_metrics:
            for regime, rm in sorted(regime_metrics.items()):
                if rm.total_trades == 0:
                    continue
                print(f"  {regime:<16} {rm.total_trades:>4} trades  "
                      f"WR: {rm.win_rate:.0%}  P&L: ${rm.total_pnl:+,.2f}  "
                      f"PF: {rm.profit_factor:.2f}")
        else:
            print("  No regime data available.")
        print()

    # Daily returns
    daily_df = metrics.calculate_daily_returns(strategy_name=args.strategy, days=args.days)
    if not daily_df.empty:
        print("DAILY P&L (last 10 days)")
        print("-" * 40)
        recent = daily_df.groupby("date").agg(
            pnl=("pnl", "sum"),
            trades=("num_trades", "sum"),
        ).tail(10)
        for idx, row in recent.iterrows():
            print(f"  {idx}  ${row['pnl']:>+10,.2f}  ({int(row['trades'])} trades)")
        print()

    # Best/worst trades
    trades = journal.get_trades(
        strategy_name=args.strategy,
        start_date=start_date,
        end_date=end_date,
        limit=10000,
    )

    if trades:
        best = max(trades, key=lambda t: t.realized_pnl)
        worst = min(trades, key=lambda t: t.realized_pnl)

        print("BEST TRADE")
        print("-" * 40)
        print(f"  {best.strategy_name} | {best.symbol} {best.side.value} "
              f"| P&L: ${best.realized_pnl:+,.2f} | {best.entry_reason}")
        print()

        print("WORST TRADE")
        print("-" * 40)
        print(f"  {worst.strategy_name} | {worst.symbol} {worst.side.value} "
              f"| P&L: ${worst.realized_pnl:+,.2f} | {worst.exit_reason}")
        print()

    # Export
    if args.export and trades:
        import pandas as pd

        rows = []
        for t in trades:
            rows.append({
                "id": t.id,
                "strategy": t.strategy_name,
                "symbol": t.symbol,
                "side": t.side.value,
                "qty": t.qty,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "realized_pnl": t.realized_pnl,
                "regime": t.regime.value if t.regime else "",
                "entry_reason": t.entry_reason,
                "exit_reason": t.exit_reason,
            })
        df = pd.DataFrame(rows)
        df.to_csv(args.export, index=False)
        print(f"Exported {len(rows)} trades to {args.export}")

    journal.close()


if __name__ == "__main__":
    main()
