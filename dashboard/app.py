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

# Resolve paths from repository root regardless of current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
STATE_DIR = DATA_DIR / "state"
JOURNAL_DB_PATH = DATA_DIR / "journal" / "trades.db"
ALERT_LOG_PATH = DATA_DIR / "logs" / "alerts.log"

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
    if not JOURNAL_DB_PATH.exists():
        return None
    return sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)


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
    state_path = STATE_DIR / "broker_snapshot.json"
    if state_path.exists():
        try:
            with open(state_path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_recent_alerts(n=20) -> list[str]:
    """Load recent alerts from alert log."""
    if not ALERT_LOG_PATH.exists():
        return []
    try:
        lines = ALERT_LOG_PATH.read_text().strip().split("\n")
        return lines[-n:]
    except Exception:
        return []


def load_regime_state() -> dict | None:
    """Load current regime from state file."""
    path = STATE_DIR / "regime.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_json_state(filename: str) -> list | dict | None:
    """Load a JSON state file from data/state/."""
    path = STATE_DIR / filename
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_position_stop_info() -> dict[str, dict]:
    """Build symbol -> {stop, target, is_bracket, strategy} from strategy state files."""
    info: dict[str, dict] = {}
    strategy_states = load_strategy_states()
    for state in strategy_states:
        name = state.get("name", "")
        for symbol, trade in state.get("trades", {}).items():
            info[symbol] = {
                "strategy": name,
                "stop": trade.get("stop_price", 0.0),
                "target": trade.get("target_price", 0.0),
                "is_bracket": trade.get("is_bracket", False),
            }
            # For trailing stops, use trail_stop if active
            if trade.get("trailing_active") and trade.get("trail_stop"):
                info[symbol]["stop"] = trade["trail_stop"]
    return info


def get_strategy_names(conn) -> list[str]:
    """Get distinct strategy names from database."""
    try:
        df = pd.read_sql_query("SELECT DISTINCT strategy_name FROM trades", conn)
        return df["strategy_name"].tolist()
    except Exception:
        return []


def load_strategy_states() -> list[dict]:
    """Load live strategy state from JSON files in data/state/."""
    state_dir = STATE_DIR
    states = []
    strategy_names = [
        "pairs_trading", "gap_reversal", "momentum", "vwap_reversion",
        "options_premium", "sector_rotation", "event_driven",
    ]
    for name in strategy_names:
        path = state_dir / f"{name}_state.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                    data.setdefault("name", name)
                    states.append(data)
            except Exception:
                pass
    return states


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


# ── Tab 1: Today's Plan ──────────────────────────────────────────────────────


def render_todays_plan() -> None:
    st.header("Today's Plan")

    plan = load_json_state("todays_plan.json")
    regime = load_regime_state()
    assessments = load_json_state("assessments.json")
    broker = load_broker_state()
    gaps = load_json_state("gaps.json")

    if not regime:
        st.info("No regime data yet. Start the trading system to populate.")
        return

    # ── Step 1: Regime Detection ──
    st.subheader("1. Regime Detection")
    regime_labels = {
        "trending_up": "Trending Up", "trending_down": "Trending Down",
        "ranging": "Range-Bound", "high_vol": "High Volatility",
        "low_vol": "Low Volatility", "event_day": "Event Day",
    }
    regime_type = regime.get("regime_type", "unknown")
    regime_label = regime_labels.get(regime_type, regime_type)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Regime", regime_label)
    col2.metric("VIX Proxy", f"{regime.get('vix_level', 0):.1f}")
    col3.metric("SPY Trend", f"{regime.get('spy_trend', 0):+.4f}")
    col4.metric("Vol Percentile", f"{regime.get('volatility_percentile', 0):.0%}")

    regime_descriptions = {
        "ranging": "Range-bound market favors mean reversion strategies (VWAP, options premium). Momentum and breakout strategies are deprioritized.",
        "trending_up": "Uptrend favors momentum longs and sector rotation. Mean reversion is deprioritized.",
        "trending_down": "Downtrend favors momentum shorts and defensive plays. Long-biased strategies are scaled back.",
        "high_vol": "High volatility favors options premium selling (wider spreads) and tighter risk limits.",
        "low_vol": "Low volatility favors VWAP reversion and pairs trading. Options premiums are thin.",
        "event_day": "Event day: reduced position sizing, event-driven strategy prioritized.",
    }
    st.caption(regime_descriptions.get(regime_type, ""))

    # ── Step 2: Strategy Scoring ──
    st.subheader("2. Strategy Scoring")
    if plan and plan.get("scores"):
        scores = sorted(plan["scores"], key=lambda s: s["total_score"], reverse=True)
        rows = []
        for s in scores:
            rows.append({
                "Strategy": s["strategy"],
                "Score": f"{s['total_score']:.3f}",
                "Base Weight": f"{s['base_weight']:.3f}",
                "Opp Quality": f"{s['opportunity_score']:.3f}",
                "VIX Mod": f"{s['vix_modifier']:+.3f}",
                "Perf Mod": f"{s['performance_modifier']:+.3f}",
                "Time Mod": f"{s['time_modifier']:+.3f}",
                "Event Mod": f"{s['event_modifier']:+.3f}",
                "Active": "YES" if s["is_active"] else "no",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.caption("No scoring data yet. Populated after pre-market intelligence runs.")

    # ── Step 3: Capital Allocation ──
    st.subheader("3. Capital Allocation")
    if plan and plan.get("allocations"):
        allocs = plan["allocations"]
        active = [a for a in allocs if a["is_active"]]
        inactive = [a for a in allocs if not a["is_active"]]

        if active:
            total_deployed = sum(a["allocated_capital"] for a in active)
            equity = broker.get("equity", 0) if broker else 0
            cash_pct = ((equity - total_deployed) / equity * 100) if equity else 0

            cols = st.columns(len(active) + 1)
            for i, a in enumerate(active):
                cols[i].metric(
                    a["strategy"],
                    f"${a['allocated_capital']:,.0f}",
                    delta=f"{a['allocation_pct']:.1f}%",
                )
            cols[-1].metric("Cash Reserve", f"{cash_pct:.0f}%")

        if inactive:
            st.caption(
                "Inactive (below threshold): "
                + ", ".join(a["strategy"] for a in inactive)
            )
    else:
        st.caption("No allocation data yet.")

    # ── Step 4: Planned Trades ──
    st.subheader("4. Opportunity Targets")
    if assessments:
        active_names = set()
        if plan and plan.get("allocations"):
            active_names = {a["strategy"] for a in plan["allocations"] if a["is_active"]}

        for a in assessments:
            name = a.get("strategy", "")
            is_active = name in active_names
            details = a.get("details", [])
            if not details:
                continue

            status = "ACTIVE" if is_active else "inactive"
            with st.expander(
                f"**{name}** [{status}] - {a.get('num_candidates', 0)} candidates, "
                f"{a.get('confidence', 0):.0%} confidence",
                expanded=is_active,
            ):
                if not is_active:
                    st.caption("Strategy did not score high enough for capital allocation today.")

                detail_df = pd.DataFrame(details)
                st.dataframe(detail_df, hide_index=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Avg R:R", f"{a.get('avg_risk_reward', 0):.1f}")
                col2.metric("Est. Edge", f"{a.get('estimated_edge_pct', 0):.2f}%")
                col3.metric("Est. Trades", a.get("estimated_daily_trades", 0))
    else:
        st.caption("No opportunity assessments yet.")

    # ── Step 5: Current Positions ──
    if broker and broker.get("positions"):
        st.subheader("5. Current Positions")
        positions = broker["positions"]
        stop_info = load_position_stop_info()
        pos_rows = []
        for p in positions:
            sym = p["symbol"]
            si = stop_info.get(sym, {})
            strategy = p.get("strategy_name") or si.get("strategy", "")
            stop = si.get("stop", 0.0)
            target = si.get("target", 0.0)
            protected = "Yes" if si.get("is_bracket") else "No"
            pos_rows.append({
                "Symbol": sym,
                "Strategy": strategy,
                "Side": p["side"],
                "Qty": p["qty"],
                "Entry": f"${p['avg_entry_price']:.2f}",
                "Current": f"${p['current_price']:.2f}",
                "Stop": f"${stop:.2f}" if stop else "-",
                "Target": f"${target:.2f}" if target else "-",
                "Protected": protected,
                "P&L": f"${p['unrealized_pnl']:+.2f}",
                "P&L%": f"{p.get('unrealized_pnl_pct', 0):+.2%}",
            })
        st.dataframe(pd.DataFrame(pos_rows), hide_index=True)

    # ── Pre-market Gaps ──
    if gaps:
        st.subheader("Pre-market Gaps Detected")
        st.dataframe(pd.DataFrame(gaps), hide_index=True)

    # Timestamp
    if plan and plan.get("timestamp"):
        st.caption(f"Plan last updated: {plan['timestamp']}")


# ── Tab 2: Overview ──────────────────────────────────────────────────────────


def render_overview(conn, days: int) -> None:
    st.header("Overview")

    # Live broker metrics (always show if available)
    broker = load_broker_state()
    if broker:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Equity", f"${broker.get('equity', 0):,.2f}")
        daily_pnl = broker.get("daily_pnl", 0)
        col2.metric("Daily P&L", f"${daily_pnl:+,.2f}",
                     delta=f"{broker.get('daily_pnl_pct', 0):+.2f}%")
        col3.metric("Open Positions", broker.get("num_positions", 0))
        col4.metric("Exposure", f"{broker.get('exposure_pct', 0):.1f}%")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Long Exposure", f"${broker.get('long_exposure', 0):,.0f}")
        col6.metric("Short Exposure", f"${broker.get('short_exposure', 0):,.0f}")
        col7.metric("Unrealized P&L", f"${broker.get('unrealized_pnl', 0):+,.2f}")
        col8.metric("Drawdown", f"{broker.get('drawdown_pct', 0):.2f}%")

    # Daily P&L bar chart (from closed trades)
    daily = load_daily_pnl(conn, days)
    if not daily.empty:
        st.subheader("Daily P&L (Closed Trades)")
        col1, col2, col3, col4 = st.columns(4)
        total_pnl = daily["daily_pnl"].sum()
        total_trades = int(daily["trades"].sum())
        total_wins = int(daily["wins"].sum())
        win_rate = total_wins / total_trades if total_trades > 0 else 0

        col1.metric("Total P&L", f"${total_pnl:+,.2f}")
        col2.metric("Total Trades", total_trades)
        col3.metric("Win Rate", f"{win_rate:.0%}")
        col4.metric("Trading Days", len(daily))

        chart_data = daily.set_index("trade_date")["daily_pnl"]
        st.bar_chart(chart_data)

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
    if broker and broker.get("positions"):
        st.subheader("Active Positions")
        stop_info = load_position_stop_info()
        pos_rows = []
        for p in broker["positions"]:
            sym = p["symbol"]
            si = stop_info.get(sym, {})
            strategy = p.get("strategy_name") or si.get("strategy", "")
            stop = si.get("stop", 0.0)
            target = si.get("target", 0.0)
            protected = "Yes" if si.get("is_bracket") else "No"
            pos_rows.append({
                "Symbol": sym,
                "Strategy": strategy,
                "Side": p.get("side", ""),
                "Qty": p.get("qty", 0),
                "Entry": p.get("avg_entry_price", 0),
                "Current": p.get("current_price", 0),
                "Stop": stop if stop else None,
                "Target": target if target else None,
                "Protected": protected,
                "Mkt Value": p.get("market_value", 0),
                "P&L": p.get("unrealized_pnl", 0),
                "P&L%": p.get("unrealized_pnl_pct", 0),
            })
        st.dataframe(pd.DataFrame(pos_rows), hide_index=True)


# ── Tab 2: Strategies ────────────────────────────────────────────────────────


def render_strategies(conn, days: int) -> None:
    st.header("Strategy Performance")

    # Always show live strategy state from JSON files
    strategy_states = load_strategy_states()
    if strategy_states:
        st.subheader("Live Strategy Status")
        rows = []
        for s in strategy_states:
            rows.append({
                "Strategy": s.get("name", "?"),
                "Enabled": "Yes" if s.get("enabled") else "No",
                "Daily P&L": f"${s.get('daily_pnl', 0):+.2f}",
                "Daily Trades": s.get("daily_trades", 0),
                "Capital Reserved": f"${s.get('capital_reserved', 0):,.0f}",
                "Open Positions": len(s.get("trades", {})),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    # Trade-based performance (once trades exist)
    strat_summary = load_strategy_summary(conn, days)
    if strat_summary.empty:
        if not strategy_states:
            st.info("No trades recorded yet.")
        return

    st.subheader("Closed Trade Performance")
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
        broker = load_broker_state()
        n_pos = broker.get("num_positions", 0) if broker else 0
        if n_pos > 0:
            st.info(f"No closed trades yet. {n_pos} position(s) currently open — trades appear here when positions are closed.")
        else:
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
        broker = load_broker_state()
        n_pos = broker.get("num_positions", 0) if broker else 0
        if n_pos > 0:
            st.info(f"No closed trades yet. {n_pos} position(s) currently open — performance charts appear after first trades close.")
        else:
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

    # Current regime
    regime = load_regime_state()
    if regime:
        col1, col2, col3, col4 = st.columns(4)
        regime_labels = {
            "trending_up": "Trending Up", "trending_down": "Trending Down",
            "ranging": "Range-Bound", "high_vol": "High Volatility",
            "low_vol": "Low Volatility", "event_day": "Event Day",
        }
        col1.metric("Regime", regime_labels.get(regime.get("regime_type", ""), regime.get("regime_type", "?")))
        col2.metric("Confidence", f"{regime.get('confidence', 0):.0%}")
        col3.metric("VIX Proxy", f"{regime.get('vix_level', 0):.1f}")
        col4.metric("SPY Trend", f"{regime.get('spy_trend', 0):+.4f}")

    # Gap scanner results
    st.subheader("Today's Gaps")
    gaps = load_json_state("gaps.json")
    if gaps:
        gap_df = pd.DataFrame(gaps)
        # Color-code direction
        st.dataframe(gap_df, hide_index=True)
    else:
        st.caption("No gap data. Run the trading system to populate.")

    # Volume scanner results
    st.subheader("Unusual Volume")
    vol = load_json_state("unusual_volume.json")
    if vol:
        st.dataframe(pd.DataFrame(vol), hide_index=True)
    else:
        st.caption("No unusual volume data yet. Detected during market hours.")

    # Opportunity assessments
    st.subheader("Opportunity Assessments")
    assessments = load_json_state("assessments.json")
    if assessments:
        rows = []
        for a in assessments:
            rows.append({
                "Strategy": a.get("strategy", ""),
                "Opportunities": "Yes" if a.get("has_opportunities") else "No",
                "Candidates": a.get("num_candidates", 0),
                "Confidence": f"{a.get('confidence', 0):.0%}",
                "Avg R:R": f"{a.get('avg_risk_reward', 0):.2f}",
                "Est. Edge": f"{a.get('estimated_edge_pct', 0):.2f}%",
                "Est. Trades": a.get("estimated_daily_trades", 0),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.caption("No assessment data. Run the trading system to populate.")

    # Event calendar
    st.subheader("Upcoming Events")
    events = load_json_state("upcoming_events.json")
    if events:
        st.table(pd.DataFrame(events))
    else:
        st.caption("No upcoming economic events found.")


# ── Main App ─────────────────────────────────────────────────────────────────


def main() -> None:
    days, auto_refresh = render_sidebar()

    conn = get_db_connection()
    if conn is None:
        st.error("Trade journal database not found. Start the trading system first.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Today's Plan", "Overview", "Strategies", "Trades", "Performance", "Intelligence"]
    )

    with tab1:
        render_todays_plan()
    with tab2:
        render_overview(conn, days)
    with tab3:
        render_strategies(conn, days)
    with tab4:
        render_trades(conn, days)
    with tab5:
        render_performance(conn, days)
    with tab6:
        render_intelligence()

    # Auto-refresh
    if auto_refresh:
        import time

        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
