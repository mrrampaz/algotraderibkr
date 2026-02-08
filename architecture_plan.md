# AlgoTrader Pro — Architecture & Implementation Plan

## Vision

An adaptive, multi-strategy algorithmic trading system that autonomously prepares each morning, selects the highest-probability strategies for the day's market regime, trades stocks/ETFs/options both long and short, captures intraday opportunities as they emerge, and learns from every trade to improve over time.

**Target**: 0.5–1% daily return on deployed capital through statistically favorable, risk-managed trades.

---

## Design Principles

1. **Regime-first**: Every decision flows from "what kind of market is this today?"
2. **Modular strategies**: Each strategy is a self-contained plugin — easy to add, disable, or replace
3. **Data-source agnostic**: Abstract data layer lets us swap Alpaca IEX → IBKR real-time → any provider
4. **Broker-agnostic execution**: Abstract execution layer (Alpaca now, IBKR later, both simultaneously eventually)
5. **Statistical edge only**: No trade without a quantified edge — every strategy must have backtested expectancy
6. **Capital preservation first**: Position sizing, correlation limits, and portfolio-level risk trump any single trade
7. **Observable**: Every decision is logged with reasoning, every trade attributed to a strategy and regime

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SCHEDULER / ORCHESTRATOR                     │
│                   (Pre-Market → Open → Intraday → Close)            │
└─────────┬───────────────┬──────────────────┬───────────────────┬────┘
          │               │                  │                   │
          ▼               ▼                  ▼                   ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  ┌───────────┐
│  INTELLIGENCE│  │   STRATEGY   │  │   RISK MANAGER   │  │ PORTFOLIO │
│    ENGINE    │  │   SELECTOR   │  │   (Portfolio &    │  │  TRACKER  │
│             │  │              │  │    Per-Trade)     │  │           │
│ • Regime    │  │ • Score each │  │ • Position sizing │  │ • P&L     │
│   Detector  │  │   strategy   │  │ • Correlation     │  │ • Metrics │
│ • Scanners  │  │ • Allocate   │  │ • Drawdown limits │  │ • Journal │
│ • News/     │  │   capital    │  │ • Kill switches   │  │ • Learn   │
│   Events    │  │ • Activate/  │  │ • Regime stops    │  │           │
│ • Sentiment │  │   deactivate │  │                   │  │           │
└──────┬──────┘  └──────┬───────┘  └────────┬──────────┘  └─────┬─────┘
       │                │                   │                   │
       ▼                ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        STRATEGY ARSENAL                              │
│                                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Gap &    │ │ Momentum │ │  Pairs   │ │  VWAP    │ │ Options  │ │
│  │ Reversal │ │ Breakout │ │ Trading  │ │ Mean Rev │ │ Premium  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ Event    │ │ Sector   │ │ Overnight│ │ Custom   │              │
│  │ Driven   │ │ Rotation │ │ Holds    │ │ (future) │              │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ABSTRACTION LAYERS                             │
│                                                                     │
│  ┌─────────────────────────┐    ┌────────────────────────────────┐  │
│  │     DATA PROVIDER       │    │      EXECUTION PROVIDER        │  │
│  │                         │    │                                │  │
│  │  Interface:             │    │  Interface:                    │  │
│  │  • get_bars()           │    │  • submit_order()              │  │
│  │  • get_quote()          │    │  • cancel_order()              │  │
│  │  • get_snapshot()       │    │  • close_position()            │  │
│  │  • get_option_chain()   │    │  • get_positions()             │  │
│  │  • stream_trades()      │    │  • get_account()               │  │
│  │                         │    │                                │  │
│  │  Implementations:       │    │  Implementations:              │  │
│  │  ├─ AlpacaIEXProvider   │    │  ├─ AlpacaExecutor             │  │
│  │  ├─ AlpacaSIPProvider   │    │  ├─ IBKRExecutor (future)      │  │
│  │  └─ IBKRProvider (fut.) │    │  └─ PaperExecutor (sim)        │  │
│  └─────────────────────────┘    └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Daily Lifecycle

### Phase 1: Pre-Market Intelligence (6:00–9:25 AM ET)

The system wakes up and builds a picture of the day ahead.

| Step | Component | What It Does | Output |
|------|-----------|-------------|--------|
| 1 | **Regime Detector** | Analyze VIX level/change, overnight futures (ES, NQ), pre-market SPY/QQQ, put/call ratio, yield spreads | `RegimeSignal`: trending/ranging, risk-on/off, volatility bucket (low/normal/high/extreme) |
| 2 | **Event Calendar** | Check for FOMC, CPI, PPI, earnings (today & this week), ex-dividend dates | `EventCalendar`: list of events with expected impact scores |
| 3 | **Gap Scanner** | Scan pre-market movers: stocks gapping >2% on volume, ETFs with unusual pre-market activity | `GapList`: ranked by gap %, volume, catalyst |
| 4 | **News Scraper** | Pull Alpaca news API + scrape finviz/yahoo for sector news, upgrades/downgrades | `NewsFeed`: categorized headlines with sentiment scores |
| 5 | **Pairs Warmup** | Run cointegration tests on pairs universe, calculate fresh hedge ratios and z-scores | `PairsState`: cointegrated pairs ready to trade |
| 6 | **Strategy Scorer** | Score each strategy against today's regime, events, and scanned opportunities | `StrategyPlan`: activated strategies with capital allocations |

### Phase 2: Market Open (9:30–10:00 AM ET)

High-activity window. Execute planned trades and capture opening volatility.

- **Gap strategies** fire on confirmed gap plays
- **Momentum scanner** watches for breakout confirmations on first 5/15-min candles
- **Options premium** evaluates IV levels for selling opportunities
- **Pairs trading** begins monitoring z-scores

### Phase 3: Intraday (10:00 AM–3:30 PM ET)

Steady-state monitoring and opportunistic trading.

- **Continuous scanning** for new setups (volume spikes, breakouts, mean reversion)
- **Position management** for all active trades (trail stops, profit targets, time stops)
- **Mid-day reassessment** (12:00 PM): Review morning P&L, disable underperforming strategies, scale winners
- **Correlation monitor**: Check if multiple positions are moving together (hidden risk)

### Phase 4: Close (3:30–4:00 PM ET)

- Close all intraday positions (unless flagged for overnight hold)
- Options positions: close or let expire based on rules
- Overnight hold decisions for swing positions (separate capital bucket)

### Phase 5: Post-Market (4:00–5:00 PM ET)

- **Performance attribution**: Which strategy contributed what, was regime classification correct?
- **Trade journal update**: Full record of every trade with context
- **Strategy weight update**: Adjust strategy scorer weights based on recent performance
- **Tomorrow prep**: Flag earnings, events for next day

---

## Strategy Arsenal — Detailed Specifications

### 1. Gap & Reversal Strategy
**Edge**: Stocks that gap up/down on no news tend to fill the gap within the first hour. Stocks gapping on real catalysts tend to continue.

| Parameter | Value |
|-----------|-------|
| Universe | Pre-market gappers > 2%, volume > 500K pre-market |
| Gap-and-go | Enter continuation if gap holds first 5-min candle high/low |
| Gap-fill | Fade gaps with no catalyst after first 15 min if gap starts filling |
| Stop | Below/above first 5-min candle (gap-and-go) or gap high/low (fade) |
| Target | 50% gap fill (fade) or 1:2 R:R (continuation) |
| Time limit | Close by 11:00 AM if target not hit |
| Capital | 10–15% of daily budget |

### 2. Momentum / Breakout Strategy
**Edge**: Stocks breaking out of consolidation on above-average volume have follow-through statistically.

| Parameter | Value |
|-----------|-------|
| Universe | Top 20 liquid stocks/ETFs + any scanner hits |
| Entry | Break above resistance on volume > 1.5x 20-day average |
| Confirmation | Close above level on 5-min candle |
| Stop | Below breakout level or VWAP, whichever is tighter |
| Target | Trailing stop (ATR-based) or fixed R:R |
| Time limit | Intraday, close by 3:30 PM unless swing criteria met |
| Capital | 15–20% of daily budget |

### 3. Pairs Trading (migrated from existing bot)
**Edge**: Cointegrated pairs mean-revert. Proven ~55–60% win rate in your existing system.

| Parameter | Value |
|-----------|-------|
| Universe | 18–21 pairs across 9 sectors (existing config) |
| Entry | Z-score > 1.2, cointegration confirmed, correlation > 0.7 |
| Exit | Z-score < 0.2 (profit), > 2.8 (stop), 80 bars (time) |
| Capital | 20–25% of daily budget |
| Migrated from | Existing `pairs_trading.py` — rewrite for new architecture |

### 4. VWAP Mean Reversion
**Edge**: Large-cap stocks revert to VWAP. Previously underperformed but worth retaining with tighter filters.

| Parameter | Value |
|-----------|-------|
| Universe | Large-cap, high-volume stocks (SPY components) |
| Entry | Z-score > 2.0 from VWAP, volume confirmation |
| Exit | Z-score < 0.5, or 0.5% stop, or 1% target |
| Filter | Only trade in ranging regime (not trending days) |
| Capital | 10% of daily budget, only active in favorable regime |

### 5. Options Premium Selling
**Edge**: 0DTE and short-dated options decay rapidly. Premium selling with contrarian filters has shown >75% win rates.

| Parameter | Value |
|-----------|-------|
| Instruments | SPY, QQQ, IWM credit spreads and iron condors |
| Entry | 10:15–10:45 AM window, VIX > 15, SMA5 contrarian filter |
| Structure | 10–16 delta short strikes, $3–5 wings |
| Stop | 200% of credit received |
| Target | 50% of max profit |
| Capital | 15–20% of daily budget |
| Note | Limited by Alpaca IEX options data delay; will improve with IBKR |

### 6. Event-Driven Strategy
**Edge**: FOMC, CPI, earnings create predictable volatility patterns (IV crush, directional moves).

| Parameter | Value |
|-----------|-------|
| Triggers | FOMC announcements, CPI/PPI releases, high-impact earnings |
| Pre-event | Sell premium (straddles/strangles) to capture IV crush |
| Post-event | Momentum on confirmed direction after announcement |
| Capital | 5–10% of daily budget, only on event days |

### 7. Sector Rotation Strategy
**Edge**: Money flows between sectors in predictable patterns relative to macro conditions.

| Parameter | Value |
|-----------|-------|
| Universe | Sector ETFs (XLK, XLF, XLE, XLV, XLI, XLP, XLU, XLB, XLRE, XLC, XLY) |
| Signal | Relative strength vs SPY on 5-day rolling basis |
| Entry | Long strongest, short weakest on divergence |
| Capital | 10% of daily budget |

### 8. Overnight / Swing Holds
**Edge**: Select high-conviction setups for multi-day holds based on daily chart patterns and catalysts.

| Parameter | Value |
|-----------|-------|
| Criteria | Strong daily trend, upcoming catalyst, low overnight gap risk |
| Sizing | Reduced (50% of intraday sizing) |
| Stop | Daily ATR-based |
| Capital | Separate 10–15% bucket, not competing with intraday |

---

## Risk Management Architecture

### Portfolio-Level Controls

| Control | Value | Action |
|---------|-------|--------|
| Max daily loss | 2% of total capital | Kill switch — close all, stop trading |
| Max drawdown | 8% from peak | Reduce position sizes by 50% |
| Max gross exposure | 80% of capital | Block new entries |
| Max single position | 5% of capital | Hard cap per trade |
| Max correlated positions | 3 positions in same sector | Block new entries in sector |
| Strategy loss limit | -1% of capital per strategy per day | Disable strategy for day |
| Mid-day reassessment | 12:00 PM ET | Reduce/cut underperformers |

### Per-Trade Risk

| Rule | Implementation |
|------|---------------|
| Position sizing | Risk-based: risk 0.25–0.5% of capital per trade |
| Stop loss | Required on every position, set before entry |
| Conviction scoring | 1–5 score affects position size (0.5x to 1.5x base) |
| Correlation check | Before entry, check correlation with existing portfolio |
| Quote validation | Verify bid/ask spread < 0.3% before entry |

### Regime-Based Adjustments

| Regime | Adjustment |
|--------|-----------|
| High volatility (VIX > 25) | Reduce position sizes 50%, widen stops, favor premium selling |
| Low volatility (VIX < 13) | Reduce options premium strategies, favor breakout/momentum |
| Trending day | Disable mean reversion, favor momentum/trend |
| Ranging day | Disable momentum, favor mean reversion/pairs |
| Event day (FOMC/CPI) | Reduce pre-event exposure, activate event strategy |

---

## Data Architecture

### IEX Constraints & Workarounds

| IEX Limitation | Workaround |
|----------------|-----------|
| No real-time options quotes | Use underlying price + modeled IV for strike selection; accept delayed fills |
| Limited snapshot rate | Batch snapshot requests, prioritize active positions |
| No pre-market streaming | Poll pre-market bars every 60s starting at 6 AM |
| Delayed tick data | Use 1-min/5-min bars instead of tick-level strategies |

### Data Provider Interface

```python
class DataProvider(Protocol):
    """Abstract data provider — swap implementations without changing strategies."""

    def get_bars(self, symbol: str, timeframe: str, limit: int, start: datetime = None) -> pd.DataFrame: ...
    def get_quote(self, symbol: str) -> Quote: ...
    def get_snapshot(self, symbol: str) -> Snapshot: ...
    def get_snapshots(self, symbols: list[str]) -> dict[str, Snapshot]: ...
    def get_option_chain(self, underlying: str, expiration: date) -> OptionChain: ...
    def get_news(self, symbols: list[str] = None, limit: int = 50) -> list[NewsItem]: ...
    def is_market_open(self) -> bool: ...
    def get_clock(self) -> MarketClock: ...
```

### Web Scraping Intelligence

| Source | Data | Frequency |
|--------|------|-----------|
| Finviz screener | Pre-market gaps, unusual volume, sector performance | Pre-market, hourly |
| Yahoo Finance | Earnings calendar, economic calendar | Daily pre-market |
| CBOE | VIX term structure, put/call ratios | Pre-market |
| TradingView (public API) | Technical levels, community sentiment | Pre-market |
| SEC EDGAR | Insider transactions (13F, Form 4) | Weekly |

---

## Repo Structure

```
algotrader/
├── README.md
├── CLAUDE.md                         # AI assistant reference
├── pyproject.toml                    # Dependencies (uv/poetry)
├── .env.example
│
├── config/
│   ├── settings.yaml                 # Global settings (capital, risk, data feed)
│   ├── strategies/                   # Per-strategy configs
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
│   │   ├── models.py                 # Shared data models (Quote, Bar, Order, Position, etc.)
│   │   ├── events.py                 # Event bus for inter-component communication
│   │   ├── config.py                 # Config loader
│   │   └── logging.py               # Structured logging setup
│   │
│   ├── data/                         # Data layer
│   │   ├── __init__.py
│   │   ├── provider.py               # DataProvider protocol
│   │   ├── alpaca_provider.py        # Alpaca IEX/SIP implementation
│   │   ├── ibkr_provider.py          # IBKR implementation (future)
│   │   └── cache.py                  # In-memory bar/quote cache
│   │
│   ├── execution/                    # Order execution layer
│   │   ├── __init__.py
│   │   ├── executor.py               # Executor protocol
│   │   ├── alpaca_executor.py        # Alpaca execution
│   │   ├── ibkr_executor.py          # IBKR execution (future)
│   │   └── order_manager.py          # Order tracking, fills, retries
│   │
│   ├── intelligence/                 # Market intelligence
│   │   ├── __init__.py
│   │   ├── regime.py                 # Market regime detection
│   │   ├── scanners/
│   │   │   ├── __init__.py
│   │   │   ├── gap_scanner.py        # Pre-market gap detection
│   │   │   ├── volume_scanner.py     # Unusual volume detection
│   │   │   ├── breakout_scanner.py   # Technical breakout detection
│   │   │   └── options_flow.py       # Unusual options activity
│   │   ├── news/
│   │   │   ├── __init__.py
│   │   │   ├── alpaca_news.py        # Alpaca news API
│   │   │   ├── scraper.py            # Web scraper (finviz, yahoo)
│   │   │   └── sentiment.py          # Simple sentiment scoring
│   │   └── calendar/
│   │       ├── __init__.py
│   │       ├── events.py             # Economic/earnings calendar
│   │       └── earnings.py           # Earnings-specific data
│   │
│   ├── strategies/                   # Strategy plugins
│   │   ├── __init__.py
│   │   ├── base.py                   # StrategyBase ABC
│   │   ├── registry.py               # Strategy registration & discovery
│   │   ├── gap_reversal.py
│   │   ├── momentum.py
│   │   ├── pairs_trading.py
│   │   ├── vwap_reversion.py
│   │   ├── options_premium.py
│   │   ├── event_driven.py
│   │   ├── sector_rotation.py
│   │   └── overnight.py
│   │
│   ├── risk/                         # Risk management
│   │   ├── __init__.py
│   │   ├── portfolio_risk.py         # Portfolio-level controls
│   │   ├── position_sizer.py         # Position sizing engine
│   │   ├── correlation.py            # Real-time correlation monitor
│   │   └── kill_switch.py            # Emergency shutdown
│   │
│   ├── strategy_selector/            # Strategy selection & capital allocation
│   │   ├── __init__.py
│   │   ├── scorer.py                 # Score strategies vs regime
│   │   ├── allocator.py              # Capital allocation optimizer
│   │   └── mid_day_review.py         # Intraday performance review
│   │
│   └── tracking/                     # Performance & learning
│       ├── __init__.py
│       ├── portfolio.py              # Live portfolio tracking
│       ├── journal.py                # Trade journal
│       ├── metrics.py                # Performance metrics
│       ├── attribution.py            # Strategy attribution
│       └── learner.py                # Weight adjustment from historical performance
│
├── dashboard/
│   ├── app.py                        # Streamlit dashboard
│   └── components/                   # Dashboard components
│
├── scripts/
│   ├── check_live.py                 # Query live positions/orders
│   ├── cleanup.py                    # Cancel all, close all
│   ├── backtest.py                   # Backtest framework entry point
│   └── analyze_trades.py             # Historical trade analysis
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── backtest/
│
└── data/                             # Runtime data (gitignored)
    ├── state/                        # Recovery state files
    ├── logs/                         # Log files
    ├── journal/                      # Trade journal data
    └── cache/                        # Data cache
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1–2)
**Goal**: Core abstractions, data layer, execution layer, one working strategy

| Task | Priority | Effort |
|------|----------|--------|
| Project scaffolding (repo, deps, config) | P0 | 2h |
| Core models (Bar, Quote, Order, Position, etc.) | P0 | 4h |
| Event bus for component communication | P0 | 3h |
| DataProvider protocol + Alpaca IEX implementation | P0 | 6h |
| Executor protocol + Alpaca execution | P0 | 6h |
| StrategyBase ABC (new, cleaner version) | P0 | 4h |
| Portfolio risk manager (basic) | P0 | 4h |
| Orchestrator (basic lifecycle: start → run → stop) | P0 | 4h |
| Migrate pairs trading to new architecture | P0 | 6h |
| State persistence & recovery | P1 | 4h |
| **Milestone**: Pairs trading running on new architecture | | **~43h** |

### Phase 2: Intelligence Layer (Week 2–3)
**Goal**: The system understands what kind of day it is

| Task | Priority | Effort |
|------|----------|--------|
| Regime detector (VIX, trend, volatility buckets) | P0 | 6h |
| Gap scanner (pre-market) | P0 | 4h |
| Volume scanner | P0 | 3h |
| Alpaca news integration | P1 | 3h |
| Web scraper (Finviz, Yahoo calendar) | P1 | 6h |
| Sentiment scorer (simple keyword/rule-based) | P2 | 3h |
| Event calendar (earnings, FOMC, CPI) | P1 | 4h |
| **Milestone**: Pre-market intelligence report generated daily | | **~29h** |

### Phase 3: Strategy Arsenal (Week 3–5)
**Goal**: Multiple strategies available for selection

| Task | Priority | Effort |
|------|----------|--------|
| Gap & Reversal strategy | P0 | 8h |
| Momentum / Breakout strategy | P0 | 8h |
| VWAP Mean Reversion (improved from original) | P1 | 6h |
| Options Premium Selling (credit spreads) | P1 | 10h |
| Sector Rotation strategy | P2 | 6h |
| Event-driven strategy | P2 | 8h |
| Strategy registry & plugin system | P0 | 3h |
| **Milestone**: 6+ strategies available, all backtestable | | **~49h** |

### Phase 4: Strategy Selection & Adaptation (Week 5–6)
**Goal**: The system picks the best strategies each day

| Task | Priority | Effort |
|------|----------|--------|
| Strategy scorer (regime → strategy weights) | P0 | 6h |
| Capital allocator (distribute across strategies) | P0 | 4h |
| Mid-day performance review & rebalance | P1 | 4h |
| Correlation monitor (real-time) | P1 | 4h |
| Position sizer with conviction scoring | P1 | 4h |
| **Milestone**: Autonomous daily strategy selection | | **~22h** |

### Phase 5: Learning & Dashboard (Week 6–7)
**Goal**: The system improves over time and is observable

| Task | Priority | Effort |
|------|----------|--------|
| Trade journal (every trade with full context) | P0 | 4h |
| Performance attribution (by strategy, regime, time) | P1 | 4h |
| Strategy weight learner (adjust from historical results) | P1 | 6h |
| Streamlit dashboard (new, regime + multi-strategy) | P1 | 8h |
| Alerting (Telegram/email on kills, big wins/losses) | P2 | 3h |
| **Milestone**: Self-improving system with full observability | | **~25h** |

### Phase 6: IBKR Integration (Week 7–8, future)
**Goal**: Add IBKR for real-time data and futures/index options

| Task | Priority | Effort |
|------|----------|--------|
| IBKRDataProvider implementation | P0 | 8h |
| IBKRExecutor implementation | P0 | 8h |
| SPX/VIX options strategies | P1 | 10h |
| Futures (MES, MNQ) strategies | P2 | 8h |
| Multi-broker orchestration | P1 | 6h |
| **Milestone**: Dual-broker system with full asset coverage | | **~40h** |

---

## Key Decisions & Trade-offs

### Why full rebuild vs incremental?

The existing codebase has good ideas but architectural limitations that would compound:
- **Config is JSON** → Moving to YAML for readability and comments
- **No data abstraction** → Every strategy calls Alpaca directly, making IBKR integration a rewrite anyway
- **No regime awareness** → Would need to be bolted on to every strategy
- **No strategy selection** → The orchestrator hardcodes which strategies run
- **AI Advisor coupling** → The advisor pattern adds complexity without clear value; replacing with simpler conviction scoring

**What we keep**: Statistical logic from pairs trading (cointegration tests, z-score calculations), risk management concepts (position sizing, capital tracking), order execution patterns (marketable limits, wash trade prevention).

### IEX vs upgrading data

For Phase 1–3, IEX is sufficient. The strategies are designed for 1-min/5-min bar timeframes, not tick-level. The main gap is options pricing — we'll model this from underlying price until IBKR data is available. If SIP ($99/mo) becomes justified by paper trading results, it's a config change, not a code change.

### No AI/LLM in the trading loop

The existing AI Advisor adds latency and unpredictability to the trading loop. Instead:
- **Rule-based regime detection** (faster, deterministic, backtestable)
- **Statistical strategy scoring** (quantified edge, not vibes)
- **Post-market LLM analysis** (optional: use Claude API to analyze daily journal and suggest improvements, but never in the hot path)

---

## Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.12+ | Ecosystem, existing knowledge |
| Package manager | uv | Fast, modern, lockfile support |
| Broker SDK | alpaca-py | Official, well-maintained |
| Data analysis | pandas, numpy, scipy | Standard quant stack |
| Web scraping | httpx + beautifulsoup4 | Async-capable, lightweight |
| Config | YAML (pydantic settings) | Type-safe, human-readable |
| Scheduling | APScheduler | Cron-like scheduling in-process |
| Dashboard | Streamlit | Rapid development, existing familiarity |
| Logging | structlog | Structured JSON logs, great for analysis |
| Testing | pytest | Standard, good async support |
| State persistence | SQLite + JSON | Simple, no infra dependency |

---

## What Success Looks Like

### Week 2
- New repo running pairs trading with clean architecture
- Regime detection classifying each day
- Pre-market gap scanner identifying opportunities

### Week 4
- 3+ strategies trading simultaneously
- Strategy selector picking based on regime
- Portfolio risk manager enforcing limits across all strategies

### Week 6
- Full strategy arsenal operational
- Mid-day adaptation working
- Dashboard showing regime, strategy performance, portfolio state
- Paper trading generating daily P&L data

### Week 8
- Performance data sufficient to evaluate live trading readiness
- IBKR integration for real-time data and expanded instruments
- Strategy weights calibrated from weeks of paper results

---

*This document will evolve as we build. Each phase will generate its own detailed design docs.*
