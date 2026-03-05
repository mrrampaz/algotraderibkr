"""Professional Streamlit dashboard for AlgoTrader Brain + IBKR."""

from __future__ import annotations

import json
import math
import platform
import re
import sqlite3
import subprocess
from collections import Counter, defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - graceful fallback if optional dep missing
    go = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
STATE_DIR = DATA_DIR / "state"
LOG_PATH = DATA_DIR / "logs" / "algotrader.log"
JOURNAL_DB_PATH = DATA_DIR / "journal" / "trades.db"
LOCK_FILE = DATA_DIR / ".algotrader.lock"

UTC = timezone.utc
NY_TZ = ZoneInfo("America/New_York")
KNOWN_STRATEGIES = [
    "momentum",
    "options_premium",
    "vwap_reversion",
    "pairs_trading",
    "gap_reversal",
    "sector_rotation",
    "event_driven",
]

st.set_page_config(
    page_title="AlgoTrader - Brain Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    [data-testid="stMetricValue"] { font-size: 1.30rem; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem; }
    .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1.0rem; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
            if value == "":
                return default
        return float(value)
    except Exception:
        return default


def parse_timestamp(value: Any) -> datetime | None:
    """Parse diverse timestamp formats into aware datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    text = str(value).strip()
    if not text:
        return None

    text = text.replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def format_currency(value: float | None, decimals: int = 0, signed: bool = False) -> str:
    """Format numbers as currency for metrics/tables."""
    if value is None:
        return "—"
    if signed:
        return f"${value:+,.{decimals}f}"
    return f"${value:,.{decimals}f}"


def format_percent(value: float | None, decimals: int = 2, signed: bool = False) -> str:
    """Format percentages."""
    if value is None:
        return "—"
    if signed:
        return f"{value:+.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def format_timedelta_short(delta: timedelta | None) -> str:
    """Format a timedelta into a compact string."""
    if delta is None:
        return "—"
    total_seconds = int(max(delta.total_seconds(), 0))
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def age_string(ts: datetime | None) -> str:
    """Humanize time since a timestamp."""
    if ts is None:
        return "—"
    now = datetime.now(UTC)
    return f"{format_timedelta_short(now - ts)} ago"


def pnl_cell_style(val: Any) -> str:
    """Color style for P&L table cells."""
    num = safe_float(val, default=0.0)
    if num > 0:
        return "color: #2E7D32; font-weight: 600;"
    if num < 0:
        return "color: #C62828; font-weight: 600;"
    return "color: #666;"


@st.cache_data(ttl=30, show_spinner=False)
def load_json_state(filename: str) -> dict[str, Any] | list[Any] | None:
    """Load a JSON state file from data/state."""
    path = STATE_DIR / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_brain_decision() -> dict[str, Any] | None:
    data = load_json_state("brain_decision.json")
    return data if isinstance(data, dict) else None


def load_regime_state() -> dict[str, Any] | None:
    data = load_json_state("regime.json")
    return data if isinstance(data, dict) else None


def load_broker_snapshot() -> dict[str, Any] | None:
    data = load_json_state("broker_snapshot.json")
    return data if isinstance(data, dict) else None


@st.cache_data(ttl=30, show_spinner=False)
def read_log_entries(max_lines: int = 60000) -> list[dict[str, Any]]:
    """Read JSON log entries from the main structured log."""
    if not LOG_PATH.exists():
        return []

    entries: list[dict[str, Any]] = []
    try:
        with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in deque(handle, maxlen=max_lines):
                text = line.strip()
                if not text:
                    continue
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    entries.append(data)
    except Exception:
        return []
    return entries


def filter_today_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter entries for current New York trading date."""
    today_ny = datetime.now(NY_TZ).date()
    out: list[dict[str, Any]] = []
    for entry in entries:
        ts = parse_timestamp(entry.get("timestamp"))
        if ts and ts.astimezone(NY_TZ).date() == today_ny:
            out.append(entry)
    return out


def get_last_log_timestamp(entries: list[dict[str, Any]]) -> datetime | None:
    """Get latest parseable log timestamp."""
    timestamps = [parse_timestamp(e.get("timestamp")) for e in entries]
    timestamps = [ts for ts in timestamps if ts is not None]
    return max(timestamps) if timestamps else None


def get_first_today_log_timestamp(entries: list[dict[str, Any]]) -> datetime | None:
    """Get first log timestamp for today in New York time."""
    today_entries = filter_today_entries(entries)
    timestamps = [parse_timestamp(e.get("timestamp")) for e in today_entries]
    timestamps = [ts for ts in timestamps if ts is not None]
    return min(timestamps) if timestamps else None


def latest_vix_delta(today_entries: list[dict[str, Any]]) -> float | None:
    """Compute VIX delta from latest two regime detections."""
    values: list[float] = []
    for entry in today_entries:
        if entry.get("event") != "regime_detected":
            continue
        val = entry.get("vix")
        if val is None:
            continue
        values.append(safe_float(val))
    if len(values) < 2:
        return None
    return values[-1] - values[-2]


def is_pid_running(pid: int) -> bool:
    """Check if a PID is running (Windows + POSIX)."""
    if pid <= 0:
        return False

    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return str(pid) in result.stdout
        except Exception:
            return False

    try:
        subprocess.run(
            ["ps", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return True
    except Exception:
        return False


def get_system_health(log_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer system running status, last cycle, and uptime."""
    running = False

    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text(encoding="utf-8").strip())
            running = is_pid_running(pid)
        except Exception:
            running = False

    last_cycle_ts = get_last_log_timestamp(log_entries)
    if not running and last_cycle_ts:
        if datetime.now(UTC) - last_cycle_ts < timedelta(minutes=7):
            running = True

    start_ts = get_first_today_log_timestamp(log_entries)
    uptime = datetime.now(UTC) - start_ts if running and start_ts else None

    return {
        "running": running,
        "status_label": "🟢 Running" if running else "🔴 Stopped",
        "last_cycle_ts": last_cycle_ts,
        "uptime": format_timedelta_short(uptime),
    }


@st.cache_resource(show_spinner=False)
def get_ibkr_connection() -> Any | None:
    """Dashboard IBKR connection using readonly mode and distinct client ID."""
    try:
        from algotrader.core.config import Settings
        from algotrader.execution.ibkr_connection import IBKRConnection

        settings = Settings()
        ibkr_cfg = settings.broker.ibkr.model_copy(deep=True)
        ibkr_cfg.client_id = int(ibkr_cfg.client_id) + 1
        ibkr_cfg.readonly = True
        ibkr_cfg.timeout = min(int(getattr(ibkr_cfg, "timeout", 5) or 5), 5)

        conn = IBKRConnection(config=ibkr_cfg)
        if conn.connect() and conn.connected:
            return conn
    except Exception:
        return None
    return None


def safe_ibkr_query(func, fallback=None):
    """Run an IBKR query safely."""
    try:
        return func()
    except Exception:
        return fallback


def fetch_ibkr_live_data(conn: Any | None) -> dict[str, Any]:
    """Fetch live account, positions, orders, and trades from IBKR."""
    payload: dict[str, Any] = {
        "connected": False,
        "account_summary": [],
        "portfolio": [],
        "open_trades": [],
        "open_orders": [],
        "trades": [],
    }

    if conn is None or not getattr(conn, "connected", False):
        return payload

    ib = safe_ibkr_query(lambda: conn.ib, fallback=None)
    if ib is None:
        return payload

    payload["connected"] = True
    payload["account_summary"] = safe_ibkr_query(lambda: ib.accountSummary(), fallback=[])
    payload["portfolio"] = safe_ibkr_query(lambda: ib.portfolio(), fallback=[])
    payload["open_trades"] = safe_ibkr_query(lambda: ib.openTrades(), fallback=[])
    payload["open_orders"] = safe_ibkr_query(lambda: ib.openOrders(), fallback=[])
    payload["trades"] = safe_ibkr_query(lambda: ib.trades(), fallback=[])
    return payload


def parse_account_summary(summary_rows: list[Any]) -> dict[str, float]:
    """Map IBKR account summary rows to tag->float."""
    summary: dict[str, float] = {}
    for row in summary_rows or []:
        tag = getattr(row, "tag", None)
        if not tag:
            continue
        summary[tag] = safe_float(getattr(row, "value", None), default=0.0)
    return summary


def build_account_overview(live_data: dict[str, Any], snapshot: dict[str, Any] | None) -> dict[str, float]:
    """Build top-level account metrics from live IBKR with fallback state snapshot."""
    summary = parse_account_summary(live_data.get("account_summary", []))
    portfolio = live_data.get("portfolio", []) or []
    snapshot = snapshot or {}

    nav = summary.get("NetLiquidation", safe_float(snapshot.get("equity")))
    cash = summary.get("TotalCashValue", safe_float(snapshot.get("cash")))
    deployed = summary.get("GrossPositionValue")
    if deployed is None or deployed == 0:
        if portfolio:
            deployed = sum(abs(safe_float(getattr(item, "marketValue", 0.0))) for item in portfolio)
        else:
            deployed = safe_float(snapshot.get("gross_exposure", 0.0))

    daily_pnl = summary.get("DailyPnL")
    if daily_pnl is None or daily_pnl == 0:
        snap_daily = snapshot.get("daily_pnl")
        if snap_daily is not None:
            daily_pnl = safe_float(snap_daily)
        else:
            daily_pnl = summary.get("RealizedPnL", 0.0) + summary.get("UnrealizedPnL", 0.0)

    daily_pnl_pct = snapshot.get("daily_pnl_pct")
    if daily_pnl_pct is None:
        daily_pnl_pct = (daily_pnl / nav * 100.0) if nav else 0.0
    else:
        daily_pnl_pct = safe_float(daily_pnl_pct)

    drawdown_pct = safe_float(snapshot.get("drawdown_pct"))
    cash_pct = (cash / nav * 100.0) if nav else 0.0

    return {
        "nav": nav,
        "cash": cash,
        "cash_pct": cash_pct,
        "daily_pnl": daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "deployed": deployed,
        "drawdown_pct": drawdown_pct,
    }


def build_positions_df(live_data: dict[str, Any], snapshot: dict[str, Any] | None) -> pd.DataFrame:
    """Build active positions DataFrame from live IBKR (or snapshot fallback)."""
    rows: list[dict[str, Any]] = []
    portfolio_items = live_data.get("portfolio", []) or []
    if portfolio_items:
        for item in portfolio_items:
            contract = getattr(item, "contract", None)
            symbol = getattr(contract, "symbol", "—")
            local_symbol = getattr(contract, "localSymbol", "") or ""
            sec_type = getattr(contract, "secType", "—")
            rows.append(
                {
                    "Symbol": local_symbol if sec_type == "OPT" and local_symbol else symbol,
                    "Type": sec_type,
                    "Qty": safe_float(getattr(item, "position", 0.0)),
                    "Avg Cost": safe_float(getattr(item, "averageCost", 0.0)),
                    "Market Price": safe_float(getattr(item, "marketPrice", 0.0)),
                    "Market Value": safe_float(getattr(item, "marketValue", 0.0)),
                    "Unrealized P&L": safe_float(getattr(item, "unrealizedPNL", 0.0)),
                    "Realized P&L": safe_float(getattr(item, "realizedPNL", 0.0)),
                }
            )
        return pd.DataFrame(rows)

    snapshot = snapshot or {}
    for pos in snapshot.get("positions", []) or []:
        rows.append(
            {
                "Symbol": pos.get("symbol", "—"),
                "Type": "STK",
                "Qty": safe_float(pos.get("qty", 0.0)),
                "Avg Cost": safe_float(pos.get("avg_entry_price", 0.0)),
                "Market Price": safe_float(pos.get("current_price", 0.0)),
                "Market Value": safe_float(pos.get("market_value", 0.0)),
                "Unrealized P&L": safe_float(pos.get("unrealized_pnl", 0.0)),
                "Realized P&L": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_open_orders_df(live_data: dict[str, Any]) -> pd.DataFrame:
    """Build open orders table from IBKR live data."""
    rows: list[dict[str, Any]] = []

    for trade in live_data.get("open_trades", []) or []:
        contract = getattr(trade, "contract", None)
        order = getattr(trade, "order", None)
        status = getattr(trade, "orderStatus", None)
        rows.append(
            {
                "Symbol": getattr(contract, "localSymbol", None) or getattr(contract, "symbol", "—"),
                "Type": getattr(contract, "secType", "—"),
                "Action": getattr(order, "action", "—"),
                "Qty": safe_float(getattr(order, "totalQuantity", 0.0)),
                "Order Type": getattr(order, "orderType", "—"),
                "Limit": safe_float(getattr(order, "lmtPrice", 0.0)),
                "Stop": safe_float(getattr(order, "auxPrice", 0.0)),
                "Status": getattr(status, "status", "—"),
            }
        )

    if not rows:
        for order in live_data.get("open_orders", []) or []:
            rows.append(
                {
                    "Symbol": "—",
                    "Type": "—",
                    "Action": getattr(order, "action", "—"),
                    "Qty": safe_float(getattr(order, "totalQuantity", 0.0)),
                    "Order Type": getattr(order, "orderType", "—"),
                    "Limit": safe_float(getattr(order, "lmtPrice", 0.0)),
                    "Stop": safe_float(getattr(order, "auxPrice", 0.0)),
                    "Status": "Submitted",
                }
            )

    return pd.DataFrame(rows)


def trades_from_ibkr_today(live_data: dict[str, Any]) -> pd.DataFrame:
    """Build a fallback trades DataFrame from IBKR execution reports for today."""
    rows: list[dict[str, Any]] = []
    today_ny = datetime.now(NY_TZ).date()

    for trade in live_data.get("trades", []) or []:
        order = getattr(trade, "order", None)
        contract = getattr(trade, "contract", None)
        strategy = (getattr(order, "orderRef", "") or "").strip() or "ibkr_live"
        symbol = getattr(contract, "symbol", "—")

        for fill in getattr(trade, "fills", []) or []:
            execution = getattr(fill, "execution", None)
            report = getattr(fill, "commissionReport", None)
            fill_time = parse_timestamp(getattr(execution, "time", None))
            if fill_time is None or fill_time.astimezone(NY_TZ).date() != today_ny:
                continue

            pnl = safe_float(getattr(report, "realizedPNL", 0.0), default=0.0)
            commission = abs(safe_float(getattr(report, "commission", 0.0), default=0.0))
            rows.append(
                {
                    "strategy": strategy,
                    "symbol": symbol,
                    "pnl": pnl,
                    "commission": commission,
                    "regime": None,
                    "date": fill_time,
                    "entry_dt": fill_time,
                    "exit_dt": fill_time,
                }
            )

    return pd.DataFrame(rows)


def parse_metadata_cell(raw: Any) -> dict[str, Any]:
    """Parse journal metadata JSON safely."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=30, show_spinner=False)
def load_journal_trades() -> pd.DataFrame:
    """Load and normalize trade journal rows."""
    if not JOURNAL_DB_PATH.exists():
        return pd.DataFrame()

    query = """
        SELECT
            id, strategy_name, symbol, side, qty, entry_price, exit_price,
            entry_time, exit_time, realized_pnl, regime, metadata
        FROM trades
        ORDER BY COALESCE(exit_time, entry_time) ASC
    """

    try:
        conn = sqlite3.connect(str(JOURNAL_DB_PATH))
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df["entry_dt"] = df["entry_time"].map(parse_timestamp)
    df["exit_dt"] = df["exit_time"].map(parse_timestamp)
    df["date"] = df["exit_dt"].where(df["exit_dt"].notna(), df["entry_dt"])
    df["pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
    df["strategy"] = df["strategy_name"].fillna("unknown")
    df["metadata_obj"] = df["metadata"].map(parse_metadata_cell)
    df["regime"] = df["regime"].where(df["regime"].notna(), None)
    df["regime"] = df.apply(
        lambda r: r["regime"] if r["regime"] else r["metadata_obj"].get("regime"),
        axis=1,
    )
    df["commission"] = df["metadata_obj"].map(
        lambda m: abs(
            safe_float(
                m.get("commission")
                if "commission" in m
                else m.get("commissions", 0.0),
                default=0.0,
            )
        )
    )

    df = df[(df["exit_dt"].notna()) | (df["pnl"].abs() > 1e-9)].copy()
    return df


def get_trade_dataset(live_data: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Get trades from journal first; fallback to IBKR fills."""
    journal_df = load_journal_trades()
    if not journal_df.empty:
        return journal_df, "journal"

    ib_df = trades_from_ibkr_today(live_data)
    if not ib_df.empty:
        return ib_df, "ibkr_fills"

    return pd.DataFrame(), "none"


def filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Filter trades DataFrame by selected period."""
    if df.empty:
        return df

    now_ny = datetime.now(NY_TZ)
    start_dt: datetime | None = None

    if period == "Today":
        start_dt = datetime(now_ny.year, now_ny.month, now_ny.day, tzinfo=NY_TZ)
    elif period == "This Week":
        start_day = now_ny.date() - timedelta(days=now_ny.weekday())
        start_dt = datetime(start_day.year, start_day.month, start_day.day, tzinfo=NY_TZ)
    elif period == "This Month":
        start_dt = datetime(now_ny.year, now_ny.month, 1, tzinfo=NY_TZ)

    if start_dt is None:
        return df.copy()

    return df[df["date"].map(lambda x: x is not None and x.astimezone(NY_TZ) >= start_dt)].copy()


def sharpe_from_daily_pnl(daily_pnl: pd.Series) -> float | None:
    """Compute annualized Sharpe from daily P&L."""
    if daily_pnl is None or len(daily_pnl) < 2:
        return None
    std = float(daily_pnl.std(ddof=0))
    if std <= 0:
        return None
    return float((daily_pnl.mean() / std) * math.sqrt(252))


def compute_strategy_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute per-strategy scoreboard metrics."""
    if df.empty:
        return []

    metrics: list[dict[str, Any]] = []
    for strategy, group in df.groupby("strategy"):
        pnl_series = group["pnl"].astype(float)
        trade_count = int(len(group))
        wins = int((pnl_series > 0).sum())
        losses = int((pnl_series < 0).sum())
        total_pnl = float(pnl_series.sum())
        avg_win = float(pnl_series[pnl_series > 0].mean()) if wins else 0.0
        avg_loss = float(pnl_series[pnl_series < 0].mean()) if losses else 0.0
        gross_profit = float(pnl_series[pnl_series > 0].sum())
        gross_loss = abs(float(pnl_series[pnl_series < 0].sum()))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
        expectancy = float(total_pnl / trade_count) if trade_count else 0.0

        ordered = group.sort_values("date")
        cum_pnl = ordered["pnl"].cumsum()
        drawdown = cum_pnl - cum_pnl.cummax()
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

        by_day = (
            ordered.assign(day=ordered["date"].map(lambda d: d.date() if d else None))
            .groupby("day")["pnl"]
            .sum()
        )
        sharpe = sharpe_from_daily_pnl(by_day)

        metrics.append(
            {
                "name": strategy,
                "trade_count": trade_count,
                "win_rate": (wins / trade_count) if trade_count else 0.0,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
                "max_dd": max_dd,
                "sharpe": sharpe,
            }
        )

    metrics.sort(key=lambda x: x["total_pnl"], reverse=True)
    return metrics


def build_strategy_activity(
    today_logs: list[dict[str, Any]],
    trades_df: pd.DataFrame,
    brain_decision: dict[str, Any] | None,
) -> dict[str, dict[str, int]]:
    """Build per-strategy activity summary for Command Center."""
    activity: dict[str, dict[str, int]] = defaultdict(
        lambda: {"assess_count": 0, "candidates_today": 0, "selected_today": 0}
    )

    for entry in today_logs:
        if entry.get("event") != "assess_complete":
            continue
        strategy = str(entry.get("strategy", "unknown"))
        activity[strategy]["assess_count"] += 1
        activity[strategy]["candidates_today"] += int(safe_float(entry.get("num_candidates", 0.0)))

    if not trades_df.empty:
        today_ny = datetime.now(NY_TZ).date()
        today_trades = trades_df[
            trades_df["date"].map(lambda d: d is not None and d.astimezone(NY_TZ).date() == today_ny)
        ]
        counts = today_trades.groupby("strategy").size().to_dict()
        for strategy, count in counts.items():
            activity[strategy]["selected_today"] = int(count)

    if brain_decision and all(v["selected_today"] == 0 for v in activity.values()):
        for sel in brain_decision.get("selected_trades", []) or []:
            candidate = sel.get("candidate", {}) if isinstance(sel, dict) else {}
            strategy = str(candidate.get("strategy_name", "unknown"))
            activity[strategy]["selected_today"] += 1

    return activity


def build_funnel_data(
    today_logs: list[dict[str, Any]],
    trades_df: pd.DataFrame,
    brain_decision: dict[str, Any] | None,
) -> pd.DataFrame:
    """Build today's proposed vs selected funnel by strategy."""
    proposed = Counter()
    selected = Counter()

    for entry in today_logs:
        if entry.get("event") != "assess_complete":
            continue
        strategy = str(entry.get("strategy", "unknown"))
        proposed[strategy] += int(safe_float(entry.get("num_candidates", 0.0)))

    today_ny = datetime.now(NY_TZ).date()
    if not trades_df.empty:
        today_trades = trades_df[
            trades_df["date"].map(lambda d: d is not None and d.astimezone(NY_TZ).date() == today_ny)
        ]
        for strategy, count in today_trades.groupby("strategy").size().to_dict().items():
            selected[strategy] += int(count)
    elif brain_decision:
        for sel in brain_decision.get("selected_trades", []) or []:
            candidate = sel.get("candidate", {}) if isinstance(sel, dict) else {}
            strategy = str(candidate.get("strategy_name", "unknown"))
            selected[strategy] += 1

    all_names = sorted(set(KNOWN_STRATEGIES) | set(proposed.keys()) | set(selected.keys()))
    rows = [{"Strategy": s, "Proposed": proposed.get(s, 0), "Selected": selected.get(s, 0)} for s in all_names]
    return pd.DataFrame(rows)


def count_rejections(
    brain_decision: dict[str, Any] | None,
    log_entries: list[dict[str, Any]],
) -> dict[str, int]:
    """Aggregate rejection reasons from decision snapshot and historical logs."""
    counts: Counter[str] = Counter()

    if brain_decision:
        for item in brain_decision.get("rejected_trades", []) or []:
            reason = str(item.get("reason", "")).strip()
            if reason:
                counts[reason] += 1

    for entry in log_entries:
        if entry.get("event") != "brain_cash_day":
            continue
        reason_text = str(entry.get("reason", ""))
        match = re.search(r"Rejected\s+\d+\s*:\s*([^\.]+)", reason_text)
        if not match:
            continue
        for part in match.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                key, raw_val = part.split(":", 1)
                counts[key.strip()] += int(safe_float(raw_val, default=0.0))
            else:
                counts[part] += 1

    return dict(counts)


def compute_regime_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strategy x regime P&L matrix."""
    if df.empty:
        return pd.DataFrame()
    data = df.copy()
    data["regime_label"] = data["regime"].fillna("unknown").astype(str)
    matrix = pd.pivot_table(
        data,
        values="pnl",
        index="strategy",
        columns="regime_label",
        aggfunc="sum",
        fill_value=0.0,
    )
    return matrix.sort_index()


@st.cache_data(ttl=600, show_spinner=False)
def load_config_capital() -> float:
    """Load configured total capital (fallback to 100k)."""
    try:
        from algotrader.core.config import Settings

        return float(Settings().trading.total_capital)
    except Exception:
        return 100000.0


def build_daily_pnl_series(df: pd.DataFrame) -> pd.Series:
    """Build daily realized P&L series from trades."""
    if df.empty:
        return pd.Series(dtype=float)
    daily = (
        df.assign(day=df["date"].map(lambda d: d.date() if d else None))
        .groupby("day")["pnl"]
        .sum()
        .sort_index()
    )
    return daily


def build_equity_curve(df: pd.DataFrame, nav_now: float | None) -> pd.DataFrame:
    """Build equity curve from trade journal daily P&L."""
    daily = build_daily_pnl_series(df)
    initial_capital = load_config_capital()

    if daily.empty:
        if nav_now is not None and nav_now > 0:
            return pd.DataFrame({"date": [datetime.now(NY_TZ).date()], "nav": [nav_now]})
        return pd.DataFrame(columns=["date", "nav"])

    nav_series = initial_capital + daily.cumsum()
    curve = nav_series.reset_index()
    curve.columns = ["date", "nav"]

    if nav_now is not None and nav_now > 0:
        today = datetime.now(NY_TZ).date()
        if curve["date"].iloc[-1] != today:
            curve = pd.concat(
                [curve, pd.DataFrame([{"date": today, "nav": nav_now}])],
                ignore_index=True,
            )
        else:
            curve.loc[curve.index[-1], "nav"] = nav_now

    return curve


def build_summary_stats(
    trades_df: pd.DataFrame,
    daily_pnl: pd.Series,
    equity_curve: pd.DataFrame,
    log_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute summary statistics for performance tab."""
    stats: dict[str, Any] = {
        "total_return_pct": 0.0,
        "overall_win_rate": 0.0,
        "profit_factor": None,
        "max_drawdown_pct": 0.0,
        "total_trades": 0,
        "avg_trade": 0.0,
        "best_day": 0.0,
        "worst_day": 0.0,
        "cash_day_pct": 0.0,
        "trading_days": 0,
        "total_commissions": 0.0,
        "sharpe": None,
    }

    if not trades_df.empty:
        pnl = trades_df["pnl"].astype(float)
        stats["total_trades"] = int(len(trades_df))
        stats["overall_win_rate"] = float((pnl > 0).mean()) if len(pnl) else 0.0
        stats["avg_trade"] = float(pnl.mean()) if len(pnl) else 0.0
        stats["total_commissions"] = float(trades_df.get("commission", pd.Series(dtype=float)).sum())

        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = abs(float(pnl[pnl < 0].sum()))
        stats["profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else None
        stats["trading_days"] = int(trades_df["date"].map(lambda d: d.date() if d else None).nunique())

    if not daily_pnl.empty:
        stats["best_day"] = float(daily_pnl.max())
        stats["worst_day"] = float(daily_pnl.min())
        stats["sharpe"] = sharpe_from_daily_pnl(daily_pnl)

    if not equity_curve.empty and len(equity_curve) >= 2:
        first_nav = safe_float(equity_curve["nav"].iloc[0], default=0.0)
        last_nav = safe_float(equity_curve["nav"].iloc[-1], default=0.0)
        if first_nav > 0:
            stats["total_return_pct"] = (last_nav / first_nav - 1.0) * 100.0

        nav = pd.Series(equity_curve["nav"].astype(float).values)
        dd = (nav / nav.cummax() - 1.0) * 100.0
        stats["max_drawdown_pct"] = abs(float(dd.min())) if not dd.empty else 0.0

    decision_days: set[date] = set()
    cash_days: set[date] = set()
    for entry in log_entries:
        ts = parse_timestamp(entry.get("timestamp"))
        if ts is None:
            continue
        day = ts.astimezone(NY_TZ).date()
        if entry.get("event") == "assess_complete":
            decision_days.add(day)
        if entry.get("event") == "brain_cash_day":
            cash_days.add(day)
    if decision_days:
        stats["cash_day_pct"] = len(cash_days) / len(decision_days) * 100.0
        stats["trading_days"] = max(stats["trading_days"], len(decision_days))

    return stats


def render_sidebar(ibkr_connected: bool, source_label: str) -> None:
    """Render sidebar controls and refresh behavior."""
    st.sidebar.title("AlgoTrader")
    st.sidebar.caption("Brain-native dashboard")
    st.sidebar.caption(f"Trade source: {source_label}")
    st.sidebar.caption(f"IBKR: {'Connected' if ibkr_connected else 'Disconnected'}")

    if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=30000, limit=None, key="brain_dashboard_refresh")
            st.sidebar.caption("Auto-refresh active")
        except Exception:
            components.html(
                "<script>setTimeout(function(){window.parent.location.reload();}, 30000);</script>",
                height=0,
                width=0,
            )
            st.sidebar.caption("Auto-refresh active (fallback mode)")


def render_command_center(
    health: dict[str, Any],
    ibkr_connected: bool,
    regime_state: dict[str, Any] | None,
    vix_delta: float | None,
    account: dict[str, float],
    brain_decision: dict[str, Any] | None,
    positions_df: pd.DataFrame,
    open_orders_df: pd.DataFrame,
    strategy_activity: dict[str, dict[str, int]],
) -> None:
    """Tab 1: Live command center."""
    cols = st.columns(6)
    cols[0].metric("Status", health["status_label"])
    cols[1].metric("IBKR", "Connected" if ibkr_connected else "Disconnected")

    regime_type = "unknown"
    vix_level = None
    if regime_state:
        regime_type = str(regime_state.get("regime_type", "unknown"))
        vix_level = safe_float(regime_state.get("vix_level"), default=None)  # type: ignore[arg-type]
    cols[2].metric("Regime", regime_type)

    vix_delta_text = format_percent(vix_delta, decimals=2, signed=True) if vix_delta is not None else "—"
    cols[3].metric("VIX", f"{vix_level:.2f}" if vix_level is not None else "—", delta=vix_delta_text)
    cols[4].metric("Last Cycle", age_string(health.get("last_cycle_ts")))
    cols[5].metric("Uptime", health.get("uptime", "—"))

    st.divider()

    money = st.columns(5)
    money[0].metric("NAV", format_currency(account["nav"], decimals=0), delta=format_currency(account["daily_pnl"], signed=True))
    money[1].metric(
        "Daily P&L",
        format_currency(account["daily_pnl"], decimals=2, signed=True),
        delta=format_percent(account["daily_pnl_pct"], signed=True),
    )
    money[2].metric("Cash", format_percent(account["cash_pct"], decimals=0))
    money[3].metric("Deployed", format_currency(account["deployed"], decimals=0))
    money[4].metric("Drawdown", format_percent(account["drawdown_pct"], decimals=2))

    st.subheader("🧠 Brain Decision")
    if not brain_decision:
        st.warning("No `brain_decision.json` available yet.")
    else:
        is_cash_day = bool(brain_decision.get("is_cash_day", False))
        if is_cash_day:
            st.info(f"💰 **Cash Day** — {brain_decision.get('reasoning', 'No rationale available.')}")
        else:
            num_trades = int(safe_float(brain_decision.get("num_trades", 0.0)))
            cash_pct = safe_float(brain_decision.get("cash_pct", 100.0))
            st.success(f"**{num_trades} trade(s) selected** — Deploying {max(0.0, 100.0 - cash_pct):.0f}% of capital")

            for trade in brain_decision.get("selected_trades", []) or []:
                candidate = trade.get("candidate", {}) if isinstance(trade, dict) else {}
                strategy = candidate.get("strategy_name", "unknown")
                symbol = candidate.get("symbol", "—")
                direction = str(candidate.get("direction", "")).upper()
                score = safe_float(trade.get("brain_score", 0.0))

                with st.expander(f"✅ {strategy} → {symbol} {direction} (score: {score:.2f})"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Entry", format_currency(safe_float(candidate.get("entry_price", 0.0)), decimals=2))
                    c2.metric("Stop", format_currency(safe_float(candidate.get("stop_price", 0.0)), decimals=2))
                    c3.metric("Target", format_currency(safe_float(candidate.get("target_price", 0.0)), decimals=2))
                    confidence = safe_float(candidate.get("confidence", 0.0)) * 100.0
                    c4.metric("Confidence", f"{confidence:.0f}%")

                    structure = candidate.get("options_structure")
                    if structure:
                        st.caption(
                            " | ".join(
                                [
                                    f"Options: {structure}",
                                    f"Short: {safe_float(candidate.get('short_strike', 0.0)):.2f}",
                                    f"Long: {safe_float(candidate.get('long_strike', 0.0)):.2f}",
                                    f"Credit: {format_currency(safe_float(candidate.get('credit_received', 0.0)), decimals=2)}",
                                ]
                            )
                        )

            rejected = brain_decision.get("rejected_trades", []) or []
            if rejected:
                with st.expander(f"❌ {len(rejected)} rejected candidates"):
                    for item in rejected:
                        candidate = item.get("candidate", {}) if isinstance(item, dict) else {}
                        reason = item.get("reason", "unknown")
                        strategy = candidate.get("strategy_name", "unknown")
                        symbol = candidate.get("symbol", "—")
                        conf = safe_float(candidate.get("confidence", 0.0))
                        rr = safe_float(candidate.get("risk_reward_ratio", 0.0))
                        st.caption(f"{strategy} → {symbol}: **{reason}** (conf: {conf:.2f}, RR: {rr:.2f})")

    st.subheader("📊 Active Positions")
    if positions_df.empty:
        st.caption("No open positions")
    else:
        display = positions_df.copy()
        styled = (
            display.style.format(
                {
                    "Qty": "{:,.2f}",
                    "Avg Cost": "${:,.2f}",
                    "Market Price": "${:,.2f}",
                    "Market Value": "${:,.2f}",
                    "Unrealized P&L": "${:+,.2f}",
                    "Realized P&L": "${:+,.2f}",
                }
            )
            .map(pnl_cell_style, subset=["Unrealized P&L", "Realized P&L"])
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)

    st.subheader("📋 Open Orders")
    if open_orders_df.empty:
        st.caption("No pending orders")
    else:
        display = open_orders_df.copy()
        styled = display.style.format(
            {
                "Qty": "{:,.2f}",
                "Limit": "${:,.2f}",
                "Stop": "${:,.2f}",
            }
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)

    st.subheader("⚡ Strategy Activity")
    for row_start in range(0, len(KNOWN_STRATEGIES), 4):
        cols = st.columns(4)
        for idx, strategy in enumerate(KNOWN_STRATEGIES[row_start : row_start + 4]):
            status = strategy_activity.get(strategy, {})
            candidates = int(status.get("candidates_today", 0))
            selected = int(status.get("selected_today", 0))
            assess_count = int(status.get("assess_count", 0))

            if selected > 0:
                cols[idx].success(f"**{strategy}**\n{candidates} candidates → {selected} traded")
            elif candidates > 0:
                cols[idx].warning(f"**{strategy}**\n{candidates} candidates → 0 traded")
            else:
                cols[idx].info(f"**{strategy}**\n0 candidates ({assess_count} scans)")


def render_scoreboard(
    trades_df: pd.DataFrame,
    today_logs: list[dict[str, Any]],
    all_logs: list[dict[str, Any]],
    brain_decision: dict[str, Any] | None,
) -> None:
    """Tab 2: Strategy effectiveness scoreboard."""
    st.header("Strategy Scoreboard")
    period = st.selectbox("Period", ["Today", "This Week", "This Month", "All Time"], index=2)
    period_df = filter_period(trades_df, period)

    metrics = compute_strategy_metrics(period_df)
    if not metrics:
        st.info("No strategy performance records for this period yet.")
    else:
        scoreboard = pd.DataFrame(
            [
                {
                    "Strategy": m["name"],
                    "Trades": m["trade_count"],
                    "Win Rate": f"{m['win_rate']:.0%}",
                    "Total P&L": format_currency(m["total_pnl"], decimals=2, signed=True),
                    "Avg Win": format_currency(m["avg_win"], decimals=2, signed=True),
                    "Avg Loss": format_currency(m["avg_loss"], decimals=2, signed=True),
                    "Profit Factor": f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "—",
                    "Expectancy": format_currency(m["expectancy"], decimals=2, signed=True),
                    "Max Drawdown": format_currency(m["max_dd"], decimals=2, signed=True),
                    "Sharpe": f"{m['sharpe']:.2f}" if m["sharpe"] is not None else "—",
                    "Status": "🟢" if m["total_pnl"] > 0 else ("🔴" if m["total_pnl"] < 0 else "⚪"),
                }
                for m in metrics
            ]
        )
        st.dataframe(scoreboard, hide_index=True, use_container_width=True)

    st.subheader("Brain Filter Funnel (Today)")
    funnel_df = build_funnel_data(today_logs, trades_df, brain_decision)
    if go is None:
        st.caption("Plotly unavailable; install `plotly` to render charts.")
    elif funnel_df.empty:
        st.caption("No candidate activity recorded today.")
    else:
        fig = go.Figure(
            data=[
                go.Bar(name="Proposed", x=funnel_df["Strategy"], y=funnel_df["Proposed"]),
                go.Bar(name="Selected", x=funnel_df["Strategy"], y=funnel_df["Selected"]),
            ]
        )
        fig.update_layout(
            barmode="group",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why Candidates Get Rejected")
    rejection_counts = count_rejections(brain_decision, all_logs)
    if go is None:
        st.caption("Plotly unavailable; install `plotly` to render charts.")
    elif not rejection_counts:
        st.caption("No rejection-reason history found yet.")
    else:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(rejection_counts.keys()),
                    values=list(rejection_counts.values()),
                    hole=0.45,
                )
            ]
        )
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strategy Performance by Regime")
    matrix = compute_regime_heatmap(period_df)
    if go is None:
        st.caption("Plotly unavailable; install `plotly` to render charts.")
    elif matrix.empty:
        st.caption("Regime-attributed trades are not available yet.")
    else:
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=matrix.columns.tolist(),
                y=matrix.index.tolist(),
                colorscale="RdYlGn",
                text=matrix.round(2).values,
                texttemplate="%{text}",
            )
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


def render_performance(
    trades_df: pd.DataFrame,
    account: dict[str, float],
    all_logs: list[dict[str, Any]],
) -> None:
    """Tab 3: Performance analytics."""
    st.header("Performance Analytics")

    nav_now = account.get("nav")
    equity_curve = build_equity_curve(trades_df, nav_now)
    daily_pnl = build_daily_pnl_series(trades_df)

    if go is None:
        st.caption("Plotly unavailable; install `plotly` to render charts.")
    elif equity_curve.empty:
        st.caption("No equity history available yet.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_curve["date"],
                y=equity_curve["nav"],
                mode="lines",
                name="NAV",
                line=dict(color="#1E88E5", width=2),
                fill="tozeroy",
                fillcolor="rgba(30, 136, 229, 0.12)",
            )
        )
        fig.update_layout(
            height=400,
            yaxis_title="NAV ($)",
            xaxis_title="",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily P&L")
    if go is None:
        st.caption("Plotly unavailable; install `plotly` to render charts.")
    elif daily_pnl.empty:
        st.caption("No closed-trade daily P&L yet.")
    else:
        colors = ["#43A047" if x >= 0 else "#E53935" for x in daily_pnl.values]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=[str(d) for d in daily_pnl.index],
                    y=daily_pnl.values,
                    marker_color=colors,
                )
            ]
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative P&L by Strategy")
    if go is None:
        st.caption("Plotly unavailable; install `plotly` to render charts.")
    elif trades_df.empty:
        st.caption("No closed trades yet.")
    else:
        fig = go.Figure()
        for strategy in sorted(trades_df["strategy"].dropna().unique()):
            strat_df = trades_df[trades_df["strategy"] == strategy].sort_values("date").copy()
            if strat_df.empty:
                continue
            strat_df["cum_pnl"] = strat_df["pnl"].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=strat_df["date"],
                    y=strat_df["cum_pnl"],
                    mode="lines",
                    name=strategy,
                )
            )
        fig.update_layout(height=400, hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    stats = build_summary_stats(trades_df, daily_pnl, equity_curve, all_logs)
    st.subheader("Summary")

    row1 = st.columns(4)
    row1[0].metric("Total Return", format_percent(stats["total_return_pct"], signed=True))
    row1[1].metric("Win Rate", f"{stats['overall_win_rate']:.0%}")
    row1[2].metric(
        "Profit Factor",
        f"{stats['profit_factor']:.2f}" if stats["profit_factor"] is not None else "—",
    )
    row1[3].metric("Max Drawdown", format_percent(stats["max_drawdown_pct"], decimals=2))

    row2 = st.columns(4)
    row2[0].metric("Total Trades", f"{stats['total_trades']}")
    row2[1].metric("Avg Trade", format_currency(stats["avg_trade"], decimals=2, signed=True))
    row2[2].metric("Best Day", format_currency(stats["best_day"], decimals=2, signed=True))
    row2[3].metric("Worst Day", format_currency(stats["worst_day"], decimals=2, signed=True))

    row3 = st.columns(4)
    row3[0].metric("Cash Days", format_percent(stats["cash_day_pct"], decimals=0))
    row3[1].metric("Trading Days", f"{stats['trading_days']}")
    row3[2].metric("Commissions", format_currency(stats["total_commissions"], decimals=2))
    row3[3].metric("Sharpe Ratio", f"{stats['sharpe']:.2f}" if stats["sharpe"] is not None else "—")

    st.subheader("Drawdown")
    if go is None:
        st.caption("Plotly unavailable; install `plotly` to render charts.")
    elif equity_curve.empty:
        st.caption("No drawdown history yet.")
    else:
        nav = pd.Series(equity_curve["nav"].astype(float).values, index=equity_curve["date"])
        drawdown = (nav / nav.cummax() - 1.0) * 100.0
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(drawdown.index),
                y=drawdown.values,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(229, 57, 53, 0.18)",
                line=dict(color="#E53935"),
                name="Drawdown %",
            )
        )
        fig.update_layout(height=260, yaxis_title="Drawdown %", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """App entrypoint."""
    st.title("AlgoTrader Brain Dashboard")

    log_entries = read_log_entries()
    today_logs = filter_today_entries(log_entries)
    health = get_system_health(log_entries)

    brain_decision = load_brain_decision()
    regime_state = load_regime_state()
    broker_snapshot = load_broker_snapshot()

    conn = get_ibkr_connection()
    live_data = fetch_ibkr_live_data(conn)
    ibkr_connected = bool(live_data.get("connected", False))

    if not ibkr_connected:
        st.warning("⚠️ IBKR not connected. Showing cached state/journal data where available.")

    trades_df, source_label = get_trade_dataset(live_data)
    render_sidebar(ibkr_connected=ibkr_connected, source_label=source_label)

    account = build_account_overview(live_data, broker_snapshot)
    positions_df = build_positions_df(live_data, broker_snapshot)
    open_orders_df = build_open_orders_df(live_data)

    strategy_activity = build_strategy_activity(today_logs, trades_df, brain_decision)
    vix_delta = latest_vix_delta(today_logs)

    tab1, tab2, tab3 = st.tabs(["Command Center", "Strategy Scoreboard", "Performance"])

    with tab1:
        render_command_center(
            health=health,
            ibkr_connected=ibkr_connected,
            regime_state=regime_state,
            vix_delta=vix_delta,
            account=account,
            brain_decision=brain_decision,
            positions_df=positions_df,
            open_orders_df=open_orders_df,
            strategy_activity=strategy_activity,
        )

    with tab2:
        render_scoreboard(
            trades_df=trades_df,
            today_logs=today_logs,
            all_logs=log_entries,
            brain_decision=brain_decision,
        )

    with tab3:
        render_performance(
            trades_df=trades_df,
            account=account,
            all_logs=log_entries,
        )


if __name__ == "__main__":
    main()
