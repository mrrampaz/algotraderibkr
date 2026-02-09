"""Streamlit dashboard for AlgoTrader.

Reads from SQLite trade journal, state files, and alert logs.
Runs as a separate process — does NOT import the trading loop.

Launch: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Page config
st.set_page_config(
    page_title="AlgoTrader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data Loading ─────────────────────────────────────────────────────────────


@st.cache_resource
def get_db_connection():
    """Connect to trade journal SQLite database."""
    db_path = "data/journal/trades.db"
    if not Path(db_path).exists():
        return None
    return sqlite3.connect(db_path, check_same_thread=False)


def load_trades(conn, days=30, strategy=None) -> pd.DataFrame:
    """Load trades from journal database."""
    query = """
        SELECT * FROM trades
        WHERE entry_time >= date('now', ?)
    """
    params = [f"-{days} days"]
    if strategy:
        query += " AND strategy_name = ?"
        params.append(strategy)
    query += " ORDER BY entry_time DESC"
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()


def load_daily_pnl(conn, days=30) -> pd.DataFrame:
    """Aggregate P&L by day."""
    query = """
        SELECT
            date(entry_time) as trade_date,
            SUM(realized_pnl) as daily_pnl,
            COUNT(*) as trades,
            SUM(CASE WHEN realized_pnl >= 0 THEN 1 ELSE 0 END) as wins
        FROM trades
        WHERE entry_time >= date('now', ?)
        GROUP BY date(entry_time)
        ORDER BY trade_date
    """
    try:
        return pd.read_sql_query(query, conn, params=[f"-{days} days"])
    except Exception:
        return pd.DataFrame()


def load_strategy_summary(conn, days=30) -> pd.DataFrame:
    """Aggregate by strategy."""
    query = """
        SELECT
            strategy_name,
            COUNT(*) as trades,
            SUM(CASE WHEN realized_pnl >= 0 THEN 1 ELSE 0 END) as wins,
            SUM(realized_pnl) as total_pnl,
            AVG(realized_pnl) as avg_pnl,
            MAX(realized_pnl) as best,
            MIN(realized_pnl) as worst
        FROM trades
        WHERE entry_time >= date('now', ?)
        GROUP BY strategy_name
        ORDER BY total_pnl DESC
    """
    try:
        return pd.read_sql_query(query, conn, params=[f"-{days} days"])
    except Exception:
        return pd.DataFrame()


def load_broker_state() -> dict | None:
    """Try to load live broker state from a shared state file."""
    state_path = Path("data/state/broker_snapshot.json")
    if state_path.exists():
        try:
            with open(state_path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_recent_alerts(n=20) -> list[str]:
    """Load recent alerts from alert log."""
    alert_path = Path("data/logs/alerts.log")
    if not alert_path.exists():
        return []
    try:
        lines = alert_path.read_text().strip().split("\n")
        return lines[-n:]
    except Exception:
        return []


def load_regime_state() -> dict | None:
    """Load current regime from state file."""
    path = Path("data/state/regime.json")
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_json_state(filename: str) -> list | None:
    """Load a JSON state file from data/state/."""
    path = Path(f"data/state/{filename}")
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_strategy_names(conn) -> list[str]:
    """Get distinct strategy names from database."""
    try:
        df = pd.read_sql_query("SELECT DISTINCT strategy_name FROM trades", conn)
        return df["strategy_name"].tolist()
    except Exception:
        return []


# ── Sidebar ──────────────────────────────────────────────────────────────────


def render_sidebar() -> tuple[int, bool]:
    st.sidebar.title("AlgoTrader")

    # Broker state
    broker = load_broker_state()
    if broker:
        st.sidebar.metric("Equity", f"${broker.get('equity', 0):,.2f}")
        daily_pnl = broker.get("daily_pnl", 0)
        st.sidebar.metric(
            "Daily P&L",
            f"${daily_pnl:+,.2f}",
            delta=f"{broker.get('daily_pnl_pct', 0):+.2f}%",
        )
        st.sidebar.metric("Positions", broker.get("num_positions", 0))
        st.sidebar.metric("Drawdown", f"{broker.get('drawdown_pct', 0):.2f}%")
    else:
        st.sidebar.warning("No live broker data")

    st.sidebar.divider()

    # Regime
    regime = load_regime_state()
    if regime:
        regime_type = regime.get("regime_type", "unknown")
        confidence = regime.get("confidence", 0)
        regime_colors = {
            "trending_up": "UP",
            "trending_down": "DOWN",
            "ranging": "RANGE",
            "high_vol": "HIGH VOL",
            "low_vol": "LOW VOL",
            "event_day": "EVENT",
        }
        label = regime_colors.get(regime_type, regime_type.upper())
        st.sidebar.subheader(f"Regime: {label}")
        st.sidebar.progress(confidence, text=f"Confidence: {confidence:.0%}")
    else:
        st.sidebar.info("No regime data")

    st.sidebar.divider()

    # Recent alerts
    st.sidebar.subheader("Recent Alerts")
    alerts = load_recent_alerts(5)
    if alerts:
        for alert in reversed(alerts):
            if not alert.strip():
                continue
            if "CRITICAL" in alert:
                st.sidebar.error(alert[:100])
            elif "WARNING" in alert:
                st.sidebar.warning(alert[:100])
            else:
                st.sidebar.info(alert[:100])
    else:
        st.sidebar.caption("No alerts yet")

    # Settings
    st.sidebar.divider()
    days = st.sidebar.slider("Lookback (days)", 1, 90, 30)
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

    return days, auto_refresh


# ── Tab 1: Overview ──────────────────────────────────────────────────────────


def render_overview(conn, days: int) -> None:
    st.header("Overview")

    # Daily P&L bar chart
    daily = load_daily_pnl(conn, days)
    if not daily.empty:
        col1, col2, col3, col4 = st.columns(4)
        total_pnl = daily["daily_pnl"].sum()
        total_trades = int(daily["trades"].sum())
        total_wins = int(daily["wins"].sum())
        win_rate = total_wins / total_trades if total_trades > 0 else 0

        col1.metric("Total P&L", f"${total_pnl:+,.2f}")
        col2.metric("Total Trades", total_trades)
        col3.metric("Win Rate", f"{win_rate:.0%}")
        col4.metric("Trading Days", len(daily))

        # Bar chart
        st.subheader("Daily P&L")
        chart_data = daily.set_index("trade_date")["daily_pnl"]
        st.bar_chart(chart_data)
    else:
        st.info("No daily P&L data yet.")

    # Strategy allocation
    strat_summary = load_strategy_summary(conn, days)
    if not strat_summary.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("P&L by Strategy")
            display_df = strat_summary[["strategy_name", "trades", "wins", "total_pnl", "avg_pnl"]].copy()
            display_df["total_pnl"] = display_df["total_pnl"].round(2)
            display_df["avg_pnl"] = display_df["avg_pnl"].round(2)
            st.dataframe(display_df, hide_index=True)
        with col2:
            st.subheader("Trade Distribution")
            st.bar_chart(strat_summary.set_index("strategy_name")["total_pnl"])

    # Active positions
    broker = load_broker_state()
    if broker and broker.get("positions"):
        st.subheader("Active Positions")
        pos_df = pd.DataFrame(broker["positions"])
        st.dataframe(pos_df, hide_index=True)


# ── Tab 2: Strategies ────────────────────────────────────────────────────────


def render_strategies(conn, days: int) -> None:
    st.header("Strategy Performance")

    strat_summary = load_strategy_summary(conn, days)
    if strat_summary.empty:
        st.info("No trades recorded yet.")
        return

    for _, row in strat_summary.iterrows():
        name = row["strategy_name"]
        with st.expander(
            f"**{name}** - ${row['total_pnl']:+,.2f} ({int(row['trades'])} trades)",
            expanded=True,
        ):
            col1, col2, col3, col4 = st.columns(4)
            wr = row["wins"] / row["trades"] if row["trades"] > 0 else 0
            col1.metric("Win Rate", f"{wr:.0%}")
            col2.metric("Avg P&L", f"${row['avg_pnl']:+.2f}")
            col3.metric("Best Trade", f"${row['best']:+.2f}")
            col4.metric("Worst Trade", f"${row['worst']:+.2f}")

            # Strategy daily P&L
            try:
                strat_daily = pd.read_sql_query(
                    """SELECT date(entry_time) as d, SUM(realized_pnl) as pnl
                       FROM trades WHERE strategy_name = ? AND entry_time >= date('now', ?)
                       GROUP BY d ORDER BY d""",
                    conn,
                    params=[name, f"-{days} days"],
                )
                if not strat_daily.empty:
                    st.bar_chart(strat_daily.set_index("d")["pnl"])
            except Exception:
                pass


# ── Tab 3: Trades ────────────────────────────────────────────────────────────


def render_trades(conn, days: int) -> None:
    st.header("Trade Log")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        names = get_strategy_names(conn)
        strategy_filter = st.selectbox("Strategy", ["All"] + names)
    with col2:
        sort_by = st.selectbox("Sort by", ["entry_time", "realized_pnl", "symbol"])

    strat = None if strategy_filter == "All" else strategy_filter
    trades = load_trades(conn, days, strat)

    if trades.empty:
        st.info("No trades in this period.")
        return

    # Summary row
    st.metric("Showing", f"{len(trades)} trades")

    # Table
    display_cols = [
        "entry_time",
        "strategy_name",
        "symbol",
        "side",
        "qty",
        "entry_price",
        "exit_price",
        "realized_pnl",
        "entry_reason",
        "exit_reason",
    ]
    available = [c for c in display_cols if c in trades.columns]
    st.dataframe(
        trades[available].sort_values(sort_by, ascending=False),
        hide_index=True,
    )

    # P&L distribution
    if "realized_pnl" in trades.columns and len(trades) > 1:
        st.subheader("P&L Distribution")
        st.bar_chart(trades["realized_pnl"].value_counts(bins=20).sort_index())


# ── Tab 4: Performance ───────────────────────────────────────────────────────


def render_performance(conn, days: int) -> None:
    st.header("Performance Analytics")

    # Cumulative P&L by strategy
    trades = load_trades(conn, days)
    if trades.empty:
        st.info("No trade data yet.")
        return

    st.subheader("Cumulative P&L")
    trades_sorted = trades.sort_values("entry_time")

    # Build cumulative P&L per strategy for combined chart
    cum_data = {}
    for strat in trades_sorted["strategy_name"].unique():
        strat_trades = trades_sorted[trades_sorted["strategy_name"] == strat].copy()
        strat_trades["cum_pnl"] = strat_trades["realized_pnl"].cumsum()
        for _, row in strat_trades.iterrows():
            idx = row["entry_time"]
            if idx not in cum_data:
                cum_data[idx] = {}
            cum_data[idx][strat] = row["cum_pnl"]

    if cum_data:
        cum_df = pd.DataFrame.from_dict(cum_data, orient="index").sort_index()
        cum_df = cum_df.ffill()
        st.line_chart(cum_df)

    # Weekly metrics table
    st.subheader("Weekly Summary")
    try:
        trades_sorted = trades_sorted.copy()
        trades_sorted["week"] = pd.to_datetime(trades_sorted["entry_time"]).dt.isocalendar().week
        weekly = (
            trades_sorted.groupby("week")
            .agg(
                trades=("realized_pnl", "count"),
                pnl=("realized_pnl", "sum"),
                avg=("realized_pnl", "mean"),
            )
            .reset_index()
        )
        weekly["pnl"] = weekly["pnl"].round(2)
        weekly["avg"] = weekly["avg"].round(2)
        st.dataframe(weekly, hide_index=True)
    except Exception:
        st.caption("Could not compute weekly summary.")


# ── Tab 5: Intelligence ──────────────────────────────────────────────────────


def render_intelligence() -> None:
    st.header("Market Intelligence")

    # Gap scanner results
    st.subheader("Today's Gaps")
    gaps = load_json_state("gaps.json")
    if gaps:
        st.dataframe(pd.DataFrame(gaps), hide_index=True)
    else:
        st.caption("No gap data. Run the trading system to populate.")

    # Volume scanner results
    st.subheader("Unusual Volume")
    vol = load_json_state("unusual_volume.json")
    if vol:
        st.dataframe(pd.DataFrame(vol), hide_index=True)
    else:
        st.caption("No unusual volume data.")

    # Event calendar
    st.subheader("Upcoming Events")
    events = load_json_state("upcoming_events.json")
    if events:
        st.table(pd.DataFrame(events))
    else:
        st.caption("No event data. Run the trading system to populate.")


# ── Main App ─────────────────────────────────────────────────────────────────


def main() -> None:
    days, auto_refresh = render_sidebar()

    conn = get_db_connection()
    if conn is None:
        st.error("Trade journal database not found. Start the trading system first.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Strategies", "Trades", "Performance", "Intelligence"]
    )

    with tab1:
        render_overview(conn, days)
    with tab2:
        render_strategies(conn, days)
    with tab3:
        render_trades(conn, days)
    with tab4:
        render_performance(conn, days)
    with tab5:
        render_intelligence()

    # Auto-refresh
    if auto_refresh:
        import time

        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
