# CLAUDE.md — AlgoTrader Build Guide

## What This Project Is

An adaptive, multi-strategy algorithmic trading system for Alpaca Markets API. It autonomously prepares each morning, detects the market regime, selects the highest-probability strategies, trades stocks/ETFs/options long and short, captures intraday opportunities, and learns from every trade.

**Target**: 0.5–1% daily return on deployed capital through statistically favorable, risk-managed trades.

## Current Status

**Phase 1 — Foundation**: Building core abstractions, data layer, execution layer, orchestrator, and migrating pairs trading as the first strategy.

## Tech Stack

- **Python 3.12+** with **uv** for package management
- **alpaca-py** for broker API
- **pandas, numpy, scipy** for quantitative analysis
- **httpx + beautifulsoup4** for async web scraping
- **pydantic** for config and data models
- **APScheduler** for lifecycle scheduling
- **structlog** for structured JSON logging
- **SQLite** for trade journal persistence
- **Streamlit** for dashboard
- **pytest** for testing
- **YAML** for all configuration

## Architecture Overview

```
Orchestrator (lifecycle: pre-market → open → intraday → close → post-market)
    │
    ├── Intelligence Engine
    │   ├── Regime Detector (VIX, trend, volatility classification)
    │   ├── Scanners (gaps, volume, breakouts, options flow)
    │   ├── News (Alpaca API + web scraping)
    │   └── Event Calendar (FOMC, CPI, earnings)
    │
    ├── Strategy Selector (scores strategies vs regime, allocates capital)
    │
    ├── Strategy Arsenal (pluggable strategies, all implement StrategyBase)
    │   ├── Gap & Reversal
    │   ├── Momentum / Breakout
    │   ├── Pairs Trading (migrated from existing bot)
    │   ├── VWAP Mean Reversion
    │   ├── Options Premium Selling
    │   ├── Event-Driven
    │   ├── Sector Rotation
    │   └── Overnight / Swing
    │
    ├── Risk Manager (portfolio-level controls, position sizing, correlation, kill switch)
    │
    └── Tracking (portfolio P&L, trade journal, performance attribution, strategy weight learner)

Abstraction Layers:
    ├── DataProvider protocol (Alpaca IEX now, IBKR later)
    └── Executor protocol (Alpaca now, IBKR later)
```

## Repo Structure

```
algotrader/
├── CLAUDE.md                         # This file
├── README.md
├── pyproject.toml                    # uv/pip dependencies
├── .env.example
│
├── config/
│   ├── settings.yaml                 # Global settings (capital, risk, data feed)
│   ├── strategies/                   # Per-strategy YAML configs
│   │   ├── pairs_trading.yaml
│   │   ├── gap_reversal.yaml
│   │   ├── momentum.yaml
│   │   ├── vwap_reversion.yaml
│   │   ├── options_premium.yaml
│   │   ├── event_driven.yaml
│   │   └── sector_rotation.yaml
│   └── regimes.yaml                  # Regime definitions and strategy weights
│
├── algotrader/                       # Main package
│   ├── __init__.py
│   ├── orchestrator.py               # Main lifecycle controller
│   │
│   ├── core/                         # Core abstractions
│   │   ├── __init__.py
│   │   ├── models.py                 # Pydantic models: Bar, Quote, Order, Position, Signal, etc.
│   │   ├── events.py                 # Event bus (pub/sub for components)
│   │   ├── config.py                 # YAML config loader with pydantic validation
│   │   └── logging.py               # structlog setup
│   │
│   ├── data/                         # Data provider layer
│   │   ├── __init__.py
│   │   ├── provider.py               # DataProvider Protocol
│   │   ├── alpaca_provider.py        # Alpaca IEX/SIP implementation
│   │   └── cache.py                  # In-memory quote/bar cache with TTL
│   │
│   ├── execution/                    # Order execution layer
│   │   ├── __init__.py
│   │   ├── executor.py               # Executor Protocol
│   │   ├── alpaca_executor.py        # Alpaca order execution
│   │   └── order_manager.py          # Order tracking, fill callbacks, retry logic
│   │
│   ├── intelligence/                 # Market intelligence engine
│   │   ├── __init__.py
│   │   ├── regime.py                 # Market regime classifier
│   │   ├── scanners/
│   │   │   ├── __init__.py
│   │   │   ├── gap_scanner.py        # Pre-market gap detection
│   │   │   ├── volume_scanner.py     # Unusual volume detection
│   │   │   └── breakout_scanner.py   # Technical breakout scanner
│   │   ├── news/
│   │   │   ├── __init__.py
│   │   │   ├── alpaca_news.py        # Alpaca news API client
│   │   │   ├── scraper.py            # Finviz/Yahoo web scraper
│   │   │   └── sentiment.py          # Keyword-based sentiment scoring
│   │   └── calendar/
│   │       ├── __init__.py
│   │       └── events.py             # Economic + earnings calendar
│   │
│   ├── strategies/                   # Strategy plugins
│   │   ├── __init__.py
│   │   ├── base.py                   # StrategyBase ABC
│   │   ├── registry.py               # Strategy discovery & registration
│   │   ├── pairs_trading.py          # Statistical pairs trading
│   │   ├── gap_reversal.py           # Gap fade / gap-and-go
│   │   ├── momentum.py              # Breakout / momentum
│   │   ├── vwap_reversion.py         # VWAP mean reversion
│   │   ├── options_premium.py        # Credit spreads / iron condors
│   │   ├── event_driven.py           # FOMC/CPI/earnings plays
│   │   ├── sector_rotation.py        # Sector relative strength
│   │   └── overnight.py              # Swing / overnight holds
│   │
│   ├── risk/                         # Risk management
│   │   ├── __init__.py
│   │   ├── portfolio_risk.py         # Portfolio-level: drawdown, exposure, kill switch
│   │   ├── position_sizer.py         # Risk-based position sizing with conviction
│   │   └── correlation.py            # Real-time correlation monitoring
│   │
│   ├── strategy_selector/            # Strategy selection engine
│   │   ├── __init__.py
│   │   ├── scorer.py                 # Score strategies vs regime/conditions
│   │   ├── allocator.py              # Capital allocation across strategies
│   │   └── reviewer.py               # Mid-day performance review
│   │
│   └── tracking/                     # Performance tracking & learning
│       ├── __init__.py
│       ├── portfolio.py              # Live portfolio state
│       ├── journal.py                # Trade journal (SQLite-backed)
│       ├── metrics.py                # Performance metrics calculation
│       └── learner.py                # Strategy weight adjustment from history
│
├── dashboard/
│   └── app.py                        # Streamlit dashboard
│
├── scripts/
│   ├── check_live.py                 # Query live positions/orders/account
│   ├── cleanup.py                    # Cancel all orders, close all positions
│   └── run.py                        # Main entry point
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   └── integration/
│
└── data/                             # Runtime data (gitignored)
    ├── state/
    ├── logs/
    ├── journal/
    └── cache/
```

## Build Phases

### Phase 1 — Foundation (BUILD THIS FIRST)

Build in this exact order:

1. **Project setup**: pyproject.toml with all dependencies, .env.example, directory structure, __init__.py files
2. **Core models** (`algotrader/core/models.py`): Pydantic models for Bar, Quote, Snapshot, Order, OrderSide, OrderStatus, Position, Signal, TradeRecord, MarketRegime, RegimeType
3. **Config loader** (`algotrader/core/config.py`): Load YAML configs with pydantic validation. Global settings + per-strategy configs.
4. **Logging** (`algotrader/core/logging.py`): structlog with JSON output to file + human-readable to console
5. **Event bus** (`algotrader/core/events.py`): Simple pub/sub for order fills, regime changes, risk alerts
6. **DataProvider protocol** (`algotrader/data/provider.py`): Abstract interface with get_bars, get_quote, get_snapshot, get_snapshots, get_option_chain, get_news, is_market_open, get_clock
7. **Alpaca data provider** (`algotrader/data/alpaca_provider.py`): Implement DataProvider for Alpaca IEX. Include retry/backoff logic. Handle bar limit quirk (returns first N, use .tail).
8. **Data cache** (`algotrader/data/cache.py`): TTL-based cache for quotes and bars to reduce API calls
9. **Executor protocol** (`algotrader/execution/executor.py`): Abstract interface with submit_order, cancel_order, close_position, get_positions, get_account, get_open_orders
10. **Alpaca executor** (`algotrader/execution/alpaca_executor.py`): Implement Executor for Alpaca. Include: idempotent orders (client_order_id), wash trade prevention (cancel conflicting orders), marketable limit orders, ghost position handling.
11. **Order manager** (`algotrader/execution/order_manager.py`): Track pending/filled/cancelled orders, fill callbacks, retry failed orders
12. **StrategyBase** (`algotrader/strategies/base.py`): ABC with run_cycle, warm_up, on_signal, on_fill, get_status. Capital tracking (reserve/release). Daily metrics with auto-reset. State save/restore.
13. **Strategy registry** (`algotrader/strategies/registry.py`): Register strategies by name, discover from config
14. **Portfolio risk manager** (`algotrader/risk/portfolio_risk.py`): Max daily loss (2%), max drawdown (8%), max gross exposure (80%), per-strategy loss limit (-1%), kill switch
15. **Position sizer** (`algotrader/risk/position_sizer.py`): Risk-based sizing (risk 0.25-0.5% per trade), max single position (5% capital), conviction multiplier (0.5x-1.5x)
16. **Portfolio tracker** (`algotrader/tracking/portfolio.py`): Live P&L, positions, daily metrics
17. **Trade journal** (`algotrader/tracking/journal.py`): SQLite-backed, records every trade with full context
18. **Pairs trading strategy** (`algotrader/strategies/pairs_trading.py`): Migrate from existing bot. Rewrite to use new DataProvider and Executor interfaces. Keep the proven logic: cointegration tests, z-score signals, hedge ratio calculation, entry/exit rules.
19. **Orchestrator** (`algotrader/orchestrator.py`): Basic lifecycle: initialize → warm up strategies → run loop (check market open → run active strategies → manage risk) → shutdown. 5-minute cycle for IEX data.
20. **Config files**: settings.yaml (global), pairs_trading.yaml (strategy config), .env.example
21. **Entry point** (`scripts/run.py`): Load config, init components, start orchestrator

### Phase 2 — Intelligence Layer

1. **Regime detector** (`algotrader/intelligence/regime.py`): Classify using VIX level, SPY trend (SMA20 slope), intraday range vs ATR. Output: RegimeType (trending_up, trending_down, ranging, high_vol, low_vol, event_day)
2. **Gap scanner** (`algotrader/intelligence/scanners/gap_scanner.py`): Pre-market scan for stocks gapping >2% with volume >500K. Use Alpaca snapshots API.
3. **Volume scanner** (`algotrader/intelligence/scanners/volume_scanner.py`): Detect unusual volume (>2x 20-day average) during market hours
4. **Alpaca news client** (`algotrader/intelligence/news/alpaca_news.py`): Pull news feed, categorize by symbol and sector
5. **Web scraper** (`algotrader/intelligence/news/scraper.py`): Scrape Finviz screener (gaps, unusual volume), Yahoo Finance earnings calendar
6. **Event calendar** (`algotrader/intelligence/calendar/events.py`): Track FOMC, CPI, PPI, earnings dates with impact scores

### Phase 3 — Strategy Arsenal

Build each strategy implementing StrategyBase:
1. Gap & Reversal
2. Momentum / Breakout
3. VWAP Mean Reversion (improved)
4. Options Premium Selling
5. Sector Rotation
6. Event-Driven

### Phase 4 — Strategy Selection

1. Strategy scorer (regime → strategy fitness scores)
2. Capital allocator (distribute across active strategies)
3. Mid-day reviewer (disable losers, scale winners)

### Phase 5 — Learning & Dashboard

1. Performance attribution
2. Strategy weight learner
3. Streamlit dashboard
4. Alerting

## Critical Implementation Rules

### Safety
- **ALWAYS** verify `ALPACA_PAPER_TRADE=True` in .env before any testing
- **NEVER** commit API keys or .env files
- **ALWAYS** check live broker state (not just local state) when reporting positions
- **ALWAYS** use paper trading until explicitly told to switch to live

### Timezone
Always use UTC conversion:
```python
from datetime import datetime
import pytz

now_utc = datetime.now(pytz.UTC)
now_et = now_utc.astimezone(pytz.timezone('America/New_York'))
```
Store all timestamps in UTC. Display in ET.

### Alpaca Quirks
1. **Bar limit**: Returns FIRST N bars, not last. Always use `.tail(limit)` or pass `start` datetime.
2. **Wash trades**: Alpaca rejects orders when opposite-side order exists. Cancel conflicting orders first.
3. **Ghost positions**: `close_position()` should return success when broker says "position not found".
4. **Quote validation**: Always check bid/ask > 0 before placing limit orders.
5. **IEX delays**: Options quotes are 15-min delayed. Plan for this in options strategies.

### Code Patterns

**Position exit safety** — always verify close before removing from tracking:
```python
close_success = executor.close_position(symbol)
if not close_success:
    return  # Keep in tracking, retry next cycle
positions.pop(symbol)  # Only after confirmed close
```

**Pairs entry rollback** — if one leg fails, close the filled leg:
```python
order_a, order_b = place_pair_orders(...)
if order_a and not order_b:
    executor.close_position(symbol_a)  # Rollback
```

**P&L tracking** — get unrealized P&L from broker before closing:
```python
broker_pos = executor.get_position(symbol)
realized_pnl = float(broker_pos.unrealized_pl) if broker_pos else 0.0
executor.close_position(symbol)
```

### Config Pattern
All config in YAML. Load once at startup, validate with pydantic:
```python
class StrategyConfig(BaseModel):
    enabled: bool = True
    capital_allocation_pct: float  # % of total capital
    # ... strategy-specific params
```

### Logging Pattern
Use structlog everywhere. Bind context:
```python
import structlog
logger = structlog.get_logger()

log = logger.bind(strategy="pairs_trading", pair_id="XOM_CVX")
log.info("entry_signal", z_score=2.3, signal="LONG_SPREAD")
```

### Testing Pattern
Every strategy must have unit tests for signal generation (using mock data, not live API).

## Existing Code Reference

The old AlpacaTradingBot repo contains proven logic to migrate:

- **Pairs trading**: Cointegration tests (Engle-Granger), z-score calculation, hedge ratio via OLS regression, entry/exit logic. File: `strategies/pairs_trading.py`
- **Risk management concepts**: Capital reservation/release, daily metric tracking, position sizing. File: `strategies/base.py`
- **Order execution patterns**: Marketable limits, wash trade cancellation, ghost position handling. File: `trading_system/order_execution.py`
- **Options trading**: Multi-leg order submission, Iron Condor structure. File: `strategies/iron_condor.py`
- **Market data**: Bar fetching, snapshot caching, quote validation. File: `trading_system/market_data.py`

Do NOT copy code verbatim. Rewrite for the new architecture using the DataProvider and Executor abstractions.

## Environment Setup

```bash
# Clone repo
git clone https://github.com/USERNAME/algotrader.git
cd algotrader

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install deps
uv sync

# Copy env file and add keys
cp .env.example .env
# Edit .env with ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_TRADE=True

# Run
python scripts/run.py
```

## Global Settings (config/settings.yaml)

```yaml
trading:
  paper_mode: true
  total_capital: 60000
  timezone: "America/New_York"

data:
  provider: "alpaca"
  feed: "iex"  # or "sip"
  cycle_interval_seconds: 300  # 5 min for IEX

risk:
  max_daily_loss_pct: 2.0
  max_drawdown_pct: 8.0
  max_gross_exposure_pct: 80.0
  max_single_position_pct: 5.0
  max_correlated_positions: 3
  strategy_daily_loss_limit_pct: 1.0

execution:
  broker: "alpaca"
  max_spread_pct: 0.3
  use_marketable_limits: true

logging:
  level: "INFO"
  file: "data/logs/algotrader.log"
  json_format: true
```
