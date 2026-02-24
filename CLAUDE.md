# CLAUDE.md - AlgoTrader Reference

## What this repo is
Adaptive multi-strategy trading system with broker-agnostic interfaces.
Strategies operate only through:
- `DataProvider` protocol (`algotrader/data/provider.py`)
- `Executor` protocol (`algotrader/execution/executor.py`)

Broker implementations are swappable:
- Alpaca: `AlpacaDataProvider`, `AlpacaExecutor`
- IBKR: `IBKRDataProvider`, `IBKRExecutor` via shared `IBKRConnection`

## Current architecture (high level)
1. `Orchestrator` loads config, initializes broker, strategies, risk, tracking.
2. Pre-market intelligence runs (regime, gaps, news/events, allocation scoring).
3. Main loop runs on fixed cycles (default 5 minutes).
4. Risk checks gate strategy execution.
5. Dashboard state is written to `data/state/*.json`.
6. Graceful shutdown saves state and disconnects broker.

## Broker configuration
`config/settings.yaml` now supports:
- `broker.provider: alpaca | ibkr`
- `broker.ibkr` connection options (`host`, `port`, `client_id`, timeouts, reconnect)

`scripts/run.py`:
- Validates broker mode before startup
- Blocks IBKR live ports (7496/4001) for safety
- Fails fast with clear message if `ib_async` is missing

## IBKR components
- `algotrader/execution/ibkr_connection.py`
  - Singleton connection manager around `ib_async.IB`
  - `nest_asyncio.apply()` for sync orchestrator compatibility
  - connect/disconnect/ensure_connected lifecycle
  - reconnect attempts and connection state logging
- `algotrader/data/ibkr_provider.py`
  - Implements `DataProvider`
  - Bars, quotes, snapshots, option chains, news, market clock
  - Returns bar DataFrame compatible with Alpaca format
- `algotrader/execution/ibkr_executor.py`
  - Implements `Executor`
  - Order submit/cancel/replace, positions, account, open orders
  - Bracket order support, mleg options support, option lookup support

## Dashboard data path rule
`dashboard/app.py` resolves paths from repo root, not process CWD.
Required runtime files:
- `data/journal/trades.db`
- `data/state/broker_snapshot.json`
- `data/state/*_state.json`, `regime.json`, `todays_plan.json`, etc.

## Alpaca quirks
1. `get_bars()` returns first N bars unless windowing is handled; use tailing logic.
2. Wash trade prevention requires canceling conflicting open orders first.
3. `close_position()` treats missing position as success (ghost position handling).
4. Use Decimal-based rounding for prices to avoid sub-penny rejections.
5. MLEG methods are Alpaca-specific and are checked via `hasattr(...)` in strategies.

## IBKR quirks
1. TWS or IB Gateway must be running; API is a TCP client connection.
2. Each bot/session needs a unique `client_id`.
3. Market data subscriptions are limited; use snapshots when possible.
4. Historical data is paced; batch requests and back off on pacing errors.
5. Always call `qualifyContracts()` before using contracts.
6. IBKR order IDs are integers; wrapper maps to `ibkr_<orderId>`.
7. Bracket orders require placing parent/TP/SL orders correctly.
8. Prefer paper ports for development: 7497 (TWS) or 4002 (Gateway).

## Safety rules
- Never run live trading in development.
- Never hardcode credentials, account IDs, or live ports in source.
- Keep strategy code broker-agnostic (no direct broker API calls in strategies).
- Validate connection health before data/order operations.

## Run commands
```bash
uv sync
.venv/Scripts/python.exe scripts/run.py
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
.venv/Scripts/python.exe scripts/test_ibkr_connection.py
```

## Scope guardrails
Do not modify these unless explicitly requested:
- `algotrader/strategies/*.py`
- Protocol interfaces in `provider.py` and `executor.py`
- Broker-agnostic tracking/risk/journal logic
