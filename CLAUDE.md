# CLAUDE.md — AlgoTrader Reference

## What This Project Is

An adaptive, multi-strategy algorithmic trading system for Alpaca Markets API. It autonomously detects the market regime, assesses real opportunities across 7 strategies, concentrates capital into the highest-EV setups, and learns from every trade.

**Target**: 0.5-1% daily return on deployed capital through statistically favorable, risk-managed trades.

**Core philosophy**: No trade is better than a bad trade. Sit in cash when nothing meets threshold.

## Current Status

All 5 build phases complete. System is live on paper trading.

- **Phase 1 — Foundation**: Core abstractions, data/execution layers, orchestrator, pairs trading
- **Phase 2 — Intelligence**: Regime detector, gap/volume/breakout scanners, news, event calendar
- **Phase 3 — Strategies**: 7 strategies with `@register_strategy` auto-registration
- **Phase 4 — Selection**: EV-based multiplicative scorer, concentration allocator, mid-day reviewer
- **Phase 5 — Learning**: Metrics, attribution, weight learner, alerts, Streamlit dashboard

**Recent redesign**: Replaced additive scoring with multiplicative EV scoring. Strategies now `assess_opportunities()` before allocation. Zero opportunities = zero capital. Concentration allocator uses power-law weighting (up to 70% to single strategy).

## Architecture Overview

```
Orchestrator (pre-market → open → intraday → close → post-market)
    │
    ├── Intelligence Engine
    │   ├── Regime Detector (SPY realized vol as VIX proxy, trend, ATR)
    │   ├── Scanners (gaps, volume, breakouts)
    │   ├── News (Alpaca NewsClient + web scraping)
    │   └── Event Calendar (FOMC, CPI, earnings with impact scores)
    │
    ├── Strategy Arsenal (7 strategies, each implements assess_opportunities + run_cycle)
    │   ├── Pairs Trading — statistical arbitrage via cointegration/z-score
    │   ├── Gap Reversal — gap-and-go (catalyst) + gap fade, time-limited to 11 AM ET
    │   ├── Momentum — consolidation breakouts with ATR trailing stops
    │   ├── VWAP Reversion — mean reversion off VWAP z-score, ranging regimes
    │   ├── Options Premium — credit spreads with SMA5 contrarian filter
    │   ├── Event-Driven — post-FOMC/CPI directional trades, 2:1 R/R
    │   └── Sector Rotation — RS-based long/short sector ETFs, 4-hour rebalance
    │
    ├── Strategy Selector
    │   ├── Scorer — multiplicative EV: base_weight * opp_quality * (1 + modifiers)
    │   ├── Allocator — power-law concentration, cash threshold, max 70% single strategy
    │   └── Reviewer — mid-day re-assessment, regime-change reallocation
    │
    ├── Risk Manager
    │   ├── Portfolio Risk — 2% max daily loss, 8% drawdown, 80% exposure, kill switch
    │   └── Position Sizer — risk-based sizing, conviction multiplier (0.5x-1.5x)
    │
    └── Tracking
        ├── Portfolio — live P&L, positions, daily metrics
        ├── Trade Journal — SQLite-backed, full context per trade
        ├── Metrics — Sharpe, profit factor, expectancy, drawdown
        ├── Attribution — P&L by strategy/regime/session
        ├── Learner — conservative regime weight adjustment from history
        └── Alerts — log/file/webhook backends, EventBus auto-subscribe

Abstraction Layers:
    ├── DataProvider protocol (Alpaca IEX now, IBKR later)
    └── Executor protocol (Alpaca now, IBKR later)
```

## Opportunity Assessment → EV Scoring → Concentration Allocation

The core decision loop each cycle:

1. **Intelligence gathers data**: Regime detection, scanner results, news, calendar events
2. **Strategies assess opportunities**: Each strategy's `assess_opportunities()` returns an `OpportunityAssessment` with candidate count, avg risk/reward, confidence, estimated edge
3. **Scorer computes EV**: `score = clamp(base_weight * opp_quality * (1 + modifiers), 0, 1)` where `opp_quality = 0.3*(candidates/3) + 0.3*(rr/3) + 0.4*confidence`. No opportunities = score 0.
4. **Allocator concentrates capital**: Power-law weighting (`score ** 2.0`), cash threshold 0.25, max 70% single strategy, 80% max deployment, 3% floor if active
5. **Strategies execute**: Only strategies with allocated capital run their `run_cycle()`
6. **Position management runs always**: Existing positions managed regardless of allocation

## Risk Framework

| Parameter | Value |
|-----------|-------|
| Max daily loss | 2% of equity |
| Max drawdown | 8% |
| Max gross exposure | 80% |
| Max single position | 5% of capital |
| Per-strategy daily loss limit | 1% |
| Risk per trade | 0.25-0.5% |
| Conviction multiplier range | 0.5x-1.5x |
| Kill switch | Auto-triggers on max daily loss or max drawdown |

## Daily Lifecycle

1. **Pre-market (7-9:30 AM ET)**: Refresh gaps, regime, news every 15 min. Volume scan at 9:15 AM.
2. **Market open**: Assess opportunities across all strategies. EV score and concentrate allocation.
3. **Intraday (5-min cycles)**: Run allocated strategies, manage all positions, risk checks.
4. **Mid-day (noon ET)**: Re-assess opportunities. Scale down losers, disable severe losers, boost winners. Regime change triggers full reallocation.
5. **Market close**: Close day-only positions.
6. **Post-market**: Attribution analysis. Daily summary alert. Friday: weight learning cycle.

## Tech Stack

- **Python 3.11+** with **uv** for package management
- **alpaca-py** for broker API (paper trading)
- **pandas, numpy, scipy** for quantitative analysis
- **httpx + beautifulsoup4** for web scraping
- **pydantic** for config and data validation
- **APScheduler** for lifecycle scheduling
- **structlog** for structured JSON logging
- **SQLite** for trade journal
- **Streamlit** for dashboard (5 tabs: Overview, Strategies, Trades, Performance, Intelligence)
- **YAML** for all configuration

## Repo Structure

```
algotrader/
├── CLAUDE.md
├── pyproject.toml
├── .env.example
│
├── config/
│   ├── settings.yaml                 # Global: capital, risk, data feed, execution
│   ├── regimes.yaml                  # Regime-strategy weight matrix + modifiers
│   └── strategies/                   # Per-strategy YAML configs
│       ├── pairs_trading.yaml
│       ├── gap_reversal.yaml
│       ├── momentum.yaml
│       ├── vwap_reversion.yaml
│       ├── options_premium.yaml
│       ├── event_driven.yaml
│       └── sector_rotation.yaml
│
├── algotrader/
│   ├── orchestrator.py               # Main lifecycle controller
│   ├── core/
│   │   ├── models.py                 # Bar, Quote, Order, Signal, OpportunityAssessment, etc.
│   │   ├── events.py                 # EventBus pub/sub
│   │   ├── config.py                 # YAML loader + pydantic validation
│   │   └── logging.py               # structlog setup
│   ├── data/
│   │   ├── provider.py               # DataProvider Protocol
│   │   ├── alpaca_provider.py        # Alpaca IEX/SIP implementation
│   │   └── cache.py                  # TTL-based quote/bar cache
│   ├── execution/
│   │   ├── executor.py               # Executor Protocol
│   │   ├── alpaca_executor.py        # Alpaca order execution
│   │   └── order_manager.py          # Order tracking, fills, retries
│   ├── intelligence/
│   │   ├── regime.py                 # Market regime classifier
│   │   ├── scanners/
│   │   │   ├── gap_scanner.py        # Pre-market gap detection
│   │   │   ├── volume_scanner.py     # Unusual volume detection
│   │   │   └── breakout_scanner.py   # Consolidation breakout scanner
│   │   ├── news/
│   │   │   ├── alpaca_news.py        # Alpaca NewsClient
│   │   │   └── scraper.py            # Finviz/Yahoo scraper
│   │   └── calendar/
│   │       └── events.py             # FOMC, CPI, earnings calendar
│   ├── strategies/
│   │   ├── base.py                   # StrategyBase ABC + OpportunityAssessment
│   │   ├── registry.py               # @register_strategy decorator + registry
│   │   ├── pairs_trading.py
│   │   ├── gap_reversal.py
│   │   ├── momentum.py
│   │   ├── vwap_reversion.py
│   │   ├── options_premium.py
│   │   ├── event_driven.py
│   │   └── sector_rotation.py
│   ├── strategy_selector/
│   │   ├── scorer.py                 # Multiplicative EV scoring
│   │   ├── allocator.py              # Power-law concentration allocation
│   │   └── reviewer.py               # Mid-day review + regime-change reallocation
│   ├── risk/
│   │   ├── portfolio_risk.py         # Portfolio-level controls + kill switch
│   │   └── position_sizer.py         # Risk-based sizing with conviction
│   └── tracking/
│       ├── portfolio.py              # Live portfolio state
│       ├── journal.py                # Trade journal (SQLite)
│       ├── metrics.py                # Sharpe, PF, expectancy, drawdown
│       ├── attribution.py            # P&L attribution by strategy/regime/session
│       ├── learner.py                # Strategy weight learning from history
│       └── alerts.py                 # Log/file/webhook alert backends
│
├── dashboard/
│   └── app.py                        # Streamlit dashboard (5 tabs)
│
├── scripts/
│   ├── run.py                        # Main entry point
│   ├── check_live.py                 # Query live account/positions/orders
│   ├── cleanup.py                    # Emergency cancel+close (--confirm)
│   └── analyze_trades.py             # Trade analysis + CSV export
│
├── tests/
│   ├── unit/
│   └── integration/
│
└── data/                             # Runtime data (gitignored)
    ├── state/                        # JSON state files for dashboard
    ├── logs/                         # structlog JSON + alerts
    ├── journal/                      # trades.db (SQLite)
    └── cache/                        # Quote/bar cache
```

## Key Config Files

| File | Controls |
|------|----------|
| `config/settings.yaml` | Capital, risk limits, data feed, execution, logging |
| `config/regimes.yaml` | Regime-strategy weight matrix, score modifiers (VIX, time, events, performance) |
| `config/strategies/*.yaml` | Per-strategy: enabled, allocation cap, entry/exit params, symbols |

## Alpaca Quirks

1. **Bar limit**: Returns FIRST N bars, not last. Always use `.tail(limit)` or pass `start` datetime.
2. **Wash trades**: Alpaca rejects orders when opposite-side order exists. Cancel conflicting orders first.
3. **Ghost positions**: `close_position()` should return success when broker says "position not found".
4. **Quote validation**: Always check bid/ask > 0 before placing limit orders.
5. **IEX delays**: Options quotes are 15-min delayed.
6. **alpaca-py 0.43+ BarSet**: `symbol in result` broken; use `symbol in result.data`. `result[symbol]` returns `list[Bar]` not `.df` — convert manually.
7. **alpaca-py 0.43+ News**: Use `from alpaca.data import NewsClient` (not `alpaca.data.news`). Dedicated `_news_client` in `AlpacaDataProvider.__init__`.

## Critical Code Patterns

**Position exit safety** — only remove from tracking after confirmed close:
```python
close_success = executor.close_position(symbol)
if not close_success:
    return  # Keep in tracking, retry next cycle
positions.pop(symbol)
```

**Pairs entry rollback** — if one leg fails, close the filled leg:
```python
order_a, order_b = place_pair_orders(...)
if order_a and not order_b:
    executor.close_position(symbol_a)
```

**P&L tracking** — get unrealized P&L from broker BEFORE closing:
```python
broker_pos = executor.get_position(symbol)
realized_pnl = float(broker_pos.unrealized_pl) if broker_pos else 0.0
executor.close_position(symbol)
```

## Safety Rules

- **ALWAYS** verify `ALPACA_PAPER_TRADE=True` in .env
- **NEVER** commit API keys or .env files
- **ALWAYS** check live broker state (not just local state) when reporting positions
- Store timestamps in UTC. Display in ET.
- VPS is in Europe; always think in ET for market hours.

## Setup & Run

```bash
# Install
uv sync
cp .env.example .env
# Edit .env: ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_TRADE=True

# Run trading
.venv/Scripts/python.exe scripts/run.py      # Windows
# .venv/bin/python scripts/run.py            # Linux/macOS

# Run dashboard
.venv/Scripts/python.exe -m streamlit run dashboard/app.py

# Utilities
.venv/Scripts/python.exe scripts/check_live.py          # Account status
.venv/Scripts/python.exe scripts/cleanup.py --confirm    # Emergency close all
.venv/Scripts/python.exe scripts/analyze_trades.py       # Trade analysis
```

### VS Code Setup
A `.vscode/launch.json` provides debug configurations for both the trading loop and dashboard. Select the `.venv` interpreter in the status bar.

## Key Design Decisions

1. **No trade > bad trade**: Cash threshold (0.25) means system sits out when no strategy finds opportunities above threshold
2. **Concentration > diversification**: Power-law allocation concentrates into best setups rather than spreading thin
3. **Opportunities are assessed, not assumed**: Every strategy must prove it sees real candidates via `assess_opportunities()` before receiving capital
4. **Position management always runs**: Existing positions get managed regardless of current regime or allocation
5. **Multiplicative scoring**: A strategy's regime weight is multiplied by opportunity quality — zero opportunities always produces zero score
6. **Conservative learning**: Weight adjustments require 20+ trades, max +/-0.10 per cycle, confidence >= 0.6, YAML backup before writes
