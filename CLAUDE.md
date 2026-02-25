# CLAUDE.md — AlgoTrader IBKR Reference

## What this repo is

Adaptive multi-strategy trading system on Interactive Brokers (or Alpaca fallback) with a Daily Brain decision engine.
The Brain evaluates concrete `TradeCandidate` objects from 7 strategies and deploys capital only into the best setups.
Cash is the default when no candidate clears thresholds.

## Architecture

### Decision Flow
1. Intelligence layer gathers regime + scanners + calendar/news context.
2. All 7 strategies run `assess_opportunities()` and return `OpportunityAssessment` with explicit `TradeCandidate` rows.
3. In `strategy_selector.mode: brain`, `DailyBrain` filters/scores/ranks candidates across strategies.
4. Brain selects a concentrated set of trades under confidence/RR/edge/risk/capital limits.
5. Selected strategies receive capital; non-selected strategies are set to `set_capital(0.0)` for new entries.
6. Strategies still run cycles and manage existing positions.
7. Mid-day review runs once and can tighten entry standards + emit close recommendations.
8. Decisions/state are persisted to `data/state/*.json` (including `brain_decision.json`).

### Brain Thresholds (from `config/settings.yaml`)
- `min_confidence`: 0.60
- `min_risk_reward`: 1.5
- `min_edge_pct`: 0.3
- `max_daily_trades`: 5
- `max_capital_per_trade_pct`: 20.0
- `max_daily_risk_pct`: 2.0
- `cash_is_default`: true

### Strategy Toolbox
All 7 strategies emit concrete `TradeCandidate` objects (top-ranked, max 3 per strategy).

| Strategy | CandidateType | Best Regime | Notes |
|----------|---------------|-------------|-------|
| Momentum/Breakout | `LONG_EQUITY` / `SHORT_EQUITY` | trending | Breakout scanner + volume confirmation |
| 0DTE Options Premium | `CREDIT_SPREAD` / `IRON_CONDOR` | ranging/high_vol | Entry window capped (default close ~10:45 AM ET) |
| VWAP Mean Reversion | `LONG_EQUITY` / `SHORT_EQUITY` | ranging/low_vol | VWAP z-score mean reversion |
| Pairs Trading | `PAIRS` | ranging | Includes anti-churn filter (expected profit >= 3x est. cost) |
| Gap Reversal | `LONG_EQUITY` / `SHORT_EQUITY` | mixed AM regimes | Entry expiry around 11:00 AM ET |
| Sector Rotation | `SECTOR_LONG_SHORT` | trending/ranging | Long strongest sector vs short weakest |
| Event-Driven | `EVENT_DIRECTIONAL` | event_day | Pre/post-event directional setups |

### Key Models
- `TradeCandidate` (`algotrader/strategy_selector/candidate.py`): concrete trade unit (entry/stop/target/confidence/edge/etc.)
- `BrainDecision` (`algotrader/strategy_selector/brain.py`): selected/rejected trades + cash/risk + reasoning
- `OpportunityAssessment` (`algotrader/strategies/base.py`): strategy opportunity summary + candidate list

### Broker Layer
- Broker selection: `settings.broker.provider` (`ibkr` or `alpaca`)
- IBKR connection: `IBKRConnection` singleton (`algotrader/execution/ibkr_connection.py`)
- IBKR data adapter: `IBKRDataProvider` (`algotrader/data/ibkr_provider.py`)
- IBKR executor: `IBKRExecutor` (`algotrader/execution/ibkr_executor.py`)
- Orchestrator switches broker implementation at startup based on config.

## File Map

```text
algotrader/
├── core/                # Models, config, events, logging
├── data/                # DataProvider protocol + Alpaca/IBKR implementations
├── execution/           # Executor protocol + Alpaca/IBKR + order manager + IBKR connection
├── intelligence/        # Regime detector, scanners, calendar, news
├── strategies/          # 7 strategy plugins (emit TradeCandidate)
│   ├── base.py          # StrategyBase + OpportunityAssessment
│   ├── momentum.py
│   ├── options_premium.py
│   ├── vwap_reversion.py
│   ├── pairs_trading.py
│   ├── gap_reversal.py
│   ├── sector_rotation.py
│   └── event_driven.py
├── strategy_selector/   # Brain + classic selector stack
│   ├── brain.py         # DailyBrain decision engine
│   ├── candidate.py     # TradeCandidate + CandidateType
│   ├── scorer.py        # Classic scorer
│   ├── allocator.py     # Classic allocator
│   └── reviewer.py      # Mid-day reviewer
├── risk/                # Portfolio risk, sizing, kill switch
└── tracking/            # Journal, metrics, attribution, learner, portfolio tracker
```

## Run Commands

```bash
# Install dependencies
uv sync

# Run trading system
uv run python scripts/run.py

# Test IBKR connection only
uv run python scripts/test_ibkr_connection.py

# Run dashboard
uv run python -m streamlit run dashboard/app.py

# Run tests
uv run pytest tests/ -q
```

## IBKR Quirks
1. TWS or IB Gateway must already be running; bot is a client connection.
2. `client_id` must be unique per running bot session.
3. IBKR market data/historical requests are paced; back off on pacing violations.
4. Paper ports: `7497` (TWS) or `4002` (Gateway).
5. Live ports (`7496`, `4001`) are blocked by `scripts/run.py` safety checks.
6. Contract qualification is required before many IBKR operations.
7. Wrapper maps native integer order IDs into internal order handling.
8. Connection manager supports reconnect attempts and shared use by provider/executor.

## Safety Rules
- NEVER use IBKR live ports in development (`7496` / `4001`).
- NEVER hardcode credentials in source; use environment variables / `.env`.
- ALWAYS keep strategy code broker-agnostic (through `DataProvider`/`Executor` interfaces).
- ALWAYS validate broker connectivity before data/order operations.
- Cash is the default posture when conviction is insufficient.

## Scope Guardrails
Do not modify unless explicitly requested:
- Protocol interfaces (`algotrader/data/provider.py`, `algotrader/execution/executor.py`)
- Risk management logic (`algotrader/risk/*.py`)
- Tracking/journal/metrics/attribution logic (`algotrader/tracking/*.py`)
