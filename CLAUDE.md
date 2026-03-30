# CLAUDE.md - AlgoTrader IBKR Reference

## What this repo is

Adaptive multi-strategy trading system on Interactive Brokers (or Alpaca fallback) with a Daily Brain decision engine.
The Brain evaluates concrete `TradeCandidate` objects from 7 strategies and deploys capital only into the best setups.
Cash is the default when no candidate clears thresholds.

## Architecture Overview

### Decision Flow
1. Startup initializes broker/data/executor, imports all strategies, and allocates configured capital slices.
2. Warm-up restores saved strategy state (`strategy.restore_state()`) and then runs strategy warm-up logic.
3. Startup morning reconciliation runs `_morning_reconciliation()` and compares broker startup positions vs strategy-restored held symbols, logging `unexpected_position_found` when mismatches exist.
4. Pre-market intelligence refresh runs before regular trading loop.
5. Each cycle detects market regime and asks all 7 strategies for `OpportunityAssessment` with explicit `TradeCandidate` rows.
6. In `strategy_selector.mode: brain`, `DailyBrain` filters/scores/ranks candidates across strategies.
7. Brain selects a concentrated set of trades under confidence/RR/edge/risk/capital limits and applies per-strategy capital (`set_capital`).
8. Strategies still run cycles and manage existing positions even when new-entry capital is set to `0.0`.
9. Midday review runs once and can tighten entry standards and emit close recommendations.
10. At 3:45 PM ET to 4:00 PM ET, orchestrator expiry guard runs `_check_expiry_risk()` and closes same-day option positions to prevent exercise/assignment.
11. Decisions/state are persisted to `data/state/*.json` (including `brain_decision.json`, `assessments.json`, `broker_snapshot.json`).

### Brain Thresholds (from `config/settings.yaml`)
- `min_confidence`: 0.60
- `min_risk_reward`: 1.5
- `min_edge_pct`: 0.3
- `options_min_confidence`: 0.55
- `options_min_risk_reward`: 0.3
- `options_min_edge_pct`: 0.1
- `max_daily_trades`: 5
- `max_capital_per_trade_pct`: 20.0
- `max_daily_risk_pct`: 2.0
- `cash_is_default`: true

### Known Bugs Fixed (Week 1 Changelog)

#### 2026-02-25 (Day 1)
1. VIX feed correctness fix:
   - Prior behavior: SPY-vol proxy went stale (observed frozen around 15.7 while real VIX was materially higher).
   - Fix: live IBKR index path wired in (`Index("VIX", "CBOE")`) via provider/regime stack.
2. Singleton guard in `scripts/run.py`:
   - Added lockfile-based single-process guard.
   - Windows-safe PID liveness check uses `tasklist` subprocess path (instead of Unix-style `os.kill` only).
3. Strategy lifecycle logging:
   - All 7 strategies now emit `assess_start`, `assess_complete`, `assess_failed`.
   - Candidate and funnel diagnostics added across strategy assess paths.
4. Options edge model fix:
   - `edge_estimate_pct` now computes expected value (`win_rate * credit - loss_rate * max_loss`) instead of zero-like placeholder behavior.
5. Event calendar seeding and regime flag:
   - Known-events seed added.
   - Regime detector supports `event_day` via calendar and explicit marks.

#### 2026-02-26 (Day 2)
1. VIX persistent subscription fix:
   - Moved away from request/cancel-only behavior.
   - Provider keeps a persistent index stream and refreshes/resubscribes when stale.
2. Brain options thresholding fix:
   - Options candidates now use `options_min_confidence`, `options_min_risk_reward`, `options_min_edge_pct`.
   - Prevents credit spreads from being rejected by directional-equity RR defaults.
3. Strategy funnel diagnostics expansion:
   - Deeper internal logging for universe/data/filter/candidate stages to isolate why candidates drop to zero.

#### 2026-03-05 (Critical options execution fix)
1. BAG direction inversion fix in `IBKRExecutor.submit_mleg_order()`:
   - Root cause: mixed-leg spread behavior was inverted when combo direction handling was wrong.
   - Fix: force BAG parent `order.action = "BUY"` and keep explicit per-leg `ComboLeg.action` (`SELL` short leg, `BUY` long leg).
2. Spread close-path correction:
   - Same BAG fix prevents close orders from unintentionally adding risk instead of reducing it.
3. Pre-expiry exercise guard:
   - `orchestrator._check_expiry_risk()` executes near close (3:45 PM ET).
   - Closes expiring option positions through executor helper methods.
4. IBKR option position helpers:
   - `get_option_positions()` (uses `ib.portfolio()`).
   - `close_option_position()` (qualified conId close with market order).
5. Strike geometry validation in options strategy:
   - Put spread must satisfy `short_strike > long_strike`.
   - Call spread must satisfy `short_strike < long_strike`.
6. Credit-vs-commission filter:
   - Strategy rejects setups where gross credit is below floor or estimated net after commissions is non-positive.
7. Order-construction trace logging:
   - Strategy: `options_order_construction`.
   - Executor: `ibkr_mleg_order_construction`.
8. Startup broker reconciliation:
   - `orchestrator._morning_reconciliation()` checks broker positions at startup against strategy-restored state.
   - Unexpected startup carries are logged via `unexpected_position_found` and summarized in `morning_reconciliation`.
9. Exercise-exposure position cap:
   - `options_premium` enforces a hard contract cap using `exercise_exposure_cap_pct` (default 20%).
   - Formula: `floor(equity * cap_pct / (strike * 100))`, minimum 1 contract.

#### 2026-03-30 (Options chain + expiry hardening)
1. IBKR option strike normalization:
   - Root cause: malformed secdef strikes for ETF chains (for example `559.78`) generated repeated `Error 200` / `Unknown contract` failures.
   - Fix: `IBKRDataProvider` now snaps SPY/QQQ/IWM strikes to valid ladder increments before contract qualification and logs `ibkr_option_strikes_normalized`.
2. Expiry selection robustness in options strategy:
   - Root cause: some providers return only nearest expiry when `expiration=None`, which could bypass configured `min_dte` / `max_dte` intent.
   - Fix: `_find_expiry()` now probes explicit dates across the configured DTE window, then falls back safely to nearest known expiry (or `today + min_dte` if none known).
3. Regression tests added:
   - `tests/unit/test_ibkr_provider.py` validates ETF strike normalization behavior.
   - `tests/unit/test_swing_trading.py` adds nearest-expiry provider scenarios to verify `_find_expiry()` probing and fallback behavior.

### Strategy Toolbox
All 7 strategies emit concrete `TradeCandidate` objects (top-ranked, max 3 per strategy).

| Strategy | CandidateType | Best Regime | Notes |
|----------|---------------|-------------|-------|
| Momentum/Breakout | `LONG_EQUITY` / `SHORT_EQUITY` | trending | Breakout scanner + volume confirmation |
| 0DTE Options Premium | `CREDIT_SPREAD` | ranging/high_vol | Production mode sells OTM put/call credit spreads with explicit per-leg BAG actions. Pre-expiry auto-close at 3:45 PM ET. Policy min credit: $0.10/contract (current config is stricter). Hard exercise-exposure cap enforced at 20% equity by default. |
| VWAP Mean Reversion | `LONG_EQUITY` / `SHORT_EQUITY` | ranging/low_vol | VWAP z-score mean reversion |
| Pairs Trading | `PAIRS` | ranging | Includes anti-churn filter (expected profit >= 3x est. cost) |
| Gap Reversal | `LONG_EQUITY` / `SHORT_EQUITY` | mixed AM regimes | Entry expiry around 11:00 AM ET |
| Sector Rotation | `SECTOR_LONG_SHORT` | trending/ranging | Long strongest sector vs short weakest |
| Event-Driven | `EVENT_DIRECTIONAL` | event_day | Pre/post-event directional setups |

### Candidate Production Status (Week 1)
Observed by March 2-4 paper sessions: 3 of 7 strategies produced candidates.
- `options_premium`: up to 3 candidates.
- `momentum`: up to 2 candidates.
- `sector_rotation`: up to 1 candidate.
- `vwap_reversion`, `pairs_trading`, `gap_reversal`, `event_driven`: still often 0 (known gap).

### Key Models
- `TradeCandidate` (`algotrader/strategy_selector/candidate.py`): concrete trade unit (entry/stop/target/confidence/edge/etc.).
- `BrainDecision` (`algotrader/strategy_selector/brain.py`): selected/rejected trades + cash/risk + reasoning.
- `OpportunityAssessment` (`algotrader/strategies/base.py`): strategy opportunity summary + candidate list.

### Broker Layer
- Broker selection: `settings.broker.provider` (`ibkr` or `alpaca`).
- IBKR connection: `IBKRConnection` singleton (`algotrader/execution/ibkr_connection.py`).
- IBKR data adapter: `IBKRDataProvider` (`algotrader/data/ibkr_provider.py`).
- IBKR executor: `IBKRExecutor` (`algotrader/execution/ibkr_executor.py`).
- Orchestrator switches broker implementation at startup based on config.

## Week 1 Performance

Paper Trading: Feb 25 - Mar 4, 2026
Starting NAV: $100,049.11
Ending NAV: $101,719.26
Return: +$1,670.15 (+1.67%)

Important context:
- Options strategy P&L over Mar 2-4 was negative (`- $121.45`).
- Reported account gain was driven by accidental stock positions from 0DTE exercise before the March 5 BAG-direction fix.
- Treat pre-fix gains as non-representative of intended options premium strategy performance.

## File Map

```text
algotrader/
|- core/                # Models, config, events, logging
|- data/                # DataProvider protocol + Alpaca/IBKR implementations
|- execution/           # Executor protocol + Alpaca/IBKR + order manager + IBKR connection
|- intelligence/        # Regime detector, scanners, calendar, news
|- strategies/          # 7 strategy plugins (emit TradeCandidate)
|  |- base.py           # StrategyBase + OpportunityAssessment
|  |- momentum.py
|  |- options_premium.py
|  |- vwap_reversion.py
|  |- pairs_trading.py
|  |- gap_reversal.py
|  |- sector_rotation.py
|  `- event_driven.py
|- strategy_selector/   # Brain + classic selector stack
|  |- brain.py          # DailyBrain decision engine
|  |- candidate.py      # TradeCandidate + CandidateType
|  |- scorer.py         # Classic scorer
|  |- allocator.py      # Classic allocator
|  `- reviewer.py       # Mid-day reviewer
|- risk/                # Portfolio risk, sizing, kill switch
`- tracking/            # Journal, metrics, attribution, learner, portfolio tracker
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

# Verify options spread direction after executor/strategy changes
grep "ibkr_mleg_order_construction\|options_order_construction" data/logs/*.log | tail -5

# Check expiry guard activity
grep "expiry_risk\|pre_expiry_close\|check_expiry\|closing_expiring_option\|expiring_option_close" data/logs/*.log | tail -5
```

## Pre-Market Checklist

1. Kill stale bot instances before start.
   - `tasklist | findstr /I "python"`
   - Confirm no stale `scripts/run.py` process/lock remains.
2. Confirm broker mode safety.
   - Paper ports only (`7497`/`4002`), never live (`7496`/`4001`).
3. Check for unexpected overnight positions.
   - `grep "unexpected_position\|morning_reconciliation\|position:" data/logs/*.log`
4. Check broker snapshot state.
   - Inspect `data/state/broker_snapshot.json` (`num_positions` should normally be `0` before new session unless intentional carry).
5. Verify strategy state files and latest `brain_decision.json` timestamp are current.
6. Verify VIX feed is live in logs.
   - `grep "vix_fetch\|ibkr_index_subscription_opened\|vix_quote_stale" data/logs/*.log | tail -20`

## IBKR Quirks (Live Paper Lessons)
1. TWS or IB Gateway must already be running; bot is a client connection.
2. `client_id` must be unique per running bot session.
3. IBKR market data/historical requests are paced; back off on pacing violations.
4. Paper ports: `7497` (TWS) or `4002` (Gateway).
5. Live ports (`7496`, `4001`) are blocked by `scripts/run.py` safety checks.
6. Contract qualification is required before many IBKR operations.
7. Wrapper maps native integer order IDs into internal order handling.
8. Connection manager supports reconnect attempts and shared use by provider/executor.
9. BAG combo behavior for mixed-leg spreads.
   - Set combo parent action to `BUY`.
   - Encode true leg intent on each `ComboLeg.action` (`SELL` short leg, `BUY` long leg).
   - Wrong combo direction can invert credit/debit behavior.
10. 0DTE exercise behavior.
   - ITM options held past close can auto-exercise/assign into stock.
   - Paper accounts can carry large stock positions overnight without proactive warning and typically without an immediate margin-call workflow.
11. Option position visibility.
   - `ib.portfolio()` is the reliable path used by executor helper `get_option_positions()`.
12. Commission reality.
   - Options often cost about `$0.65-$1.05` per contract per leg.
   - Tiny credits are not tradable after fees.

## Safety Rules
- NEVER use IBKR live ports in development (`7496` / `4001`).
- NEVER hardcode credentials in source; use environment variables / `.env`.
- ALWAYS keep strategy code broker-agnostic (through `DataProvider`/`Executor` interfaces).
- ALWAYS validate broker connectivity before data/order operations.
- ALWAYS keep expiry protection active: `_check_expiry_risk()` must run near 3:45 PM ET; do not disable it.
- NEVER submit low-credit options spreads.
  - Policy floor: at least `$0.10` per contract credit.
  - Current config is stricter (`min_credit_per_spread: 50.0`, with commission gating).
- Verify BAG spread fills for direction sanity.
  - If intended credit spread fills as net debit, flatten immediately and investigate executor mapping.
- Position-size options conservatively.
  - Hard cap is enforced in `options_premium` using `exercise_exposure_cap_pct` (default `20.0`).
  - Formula: `max_contracts = floor(equity * max_capital_pct / (strike * 100))`.
  - Example: SPY, $100,000 equity, `max_capital_pct=20%` -> `floor(100000*0.20/(SPY*100))` ~= 1 contract.
  - Align with Brain capital cap (`max_capital_per_trade_pct`) and strategy risk caps (`max_risk_per_trade`, `max_contracts`).
  - Avoid opening contracts that imply unintended stock-equivalent leverage at expiry.
- Cash is the default posture when conviction is insufficient.

## Scope Guardrails
Do not modify unless explicitly requested:
- Protocol interfaces (`algotrader/data/provider.py`, `algotrader/execution/executor.py`)
- Risk management logic (`algotrader/risk/*.py`)
- Tracking/journal/metrics/attribution logic (`algotrader/tracking/*.py`)
