# AlgoTrader IBKR — Architecture (Current State)

*Snapshot as of 2026-05-10. Reflects the system as it actually runs, not as originally designed.*

---

## What this system is

A single-process Python trading system that runs against Interactive Brokers (paper) with an Alpaca fallback. Each cycle, every enabled strategy emits concrete `TradeCandidate` rows; the **Daily Brain** filters, scores, and ranks them across strategies and deploys capital only into the highest-EV setups. Cash is the default posture.

As of 2026-05-10, **all seven strategies are enabled**. The system was previously running options-only because the original Brain only made two decisions per day (open + midday), which couldn't react to intraday regime shifts. The new **dynamic-cadence Brain** re-decides every 60 minutes during the entry window plus on event triggers (regime flip, VIX delta ≥ 1.5), making it safe to bring the parked strategies back online.

Capital: $60,000 paper account. Broker: IBKR via TWS/Gateway on paper port `7497`/`4002`.

---

## Component map

```
                    ┌──────────────────────────┐
                    │     scripts/run.py       │
                    │  (lockfile-guarded entry)│
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │      Orchestrator        │
                    │  lifecycle controller    │
                    └────────────┬─────────────┘
                                 │
   ┌─────────────┬───────────────┼────────────────┬──────────────┐
   ▼             ▼               ▼                ▼              ▼
┌───────┐  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────┐
│ Data  │  │ Executor │  │ Intelligence │  │ Strategies │  │   Risk   │
│Provider│  │          │  │              │  │            │  │          │
│ IBKR/ │  │  IBKR/   │  │ • Regime     │  │ 7 plugins  │  │ Portfolio│
│Alpaca │  │  Alpaca  │  │ • Scanners   │  │ all active │  │ + Sizer  │
└───────┘  └──────────┘  │ • News       │  └─────┬──────┘  └──────────┘
                         │ • Calendar   │        │
                         └──────────────┘        ▼
                                          ┌──────────────┐
                                          │ DailyBrain   │
                                          │ (dynamic     │
                                          │  cadence)    │
                                          └──────┬───────┘
                                                 │
                              ┌──────────────────┼─────────────────┐
                              ▼                  ▼                 ▼
                       ┌────────────┐   ┌──────────────┐   ┌──────────────┐
                       │  Journal   │   │ BrokerLedger │   │  Portfolio   │
                       │ (strategy) │   │  (truth)     │   │   Tracker    │
                       └────────────┘   └──────────────┘   └──────────────┘
```

---

## Daily lifecycle

The orchestrator drives one process per trading day. Cycle interval is 5 minutes (`data.cycle_interval_seconds: 300`). Within that cycle, the Brain re-decides on its own dynamic cadence — not on every cycle.

| Phase | When (ET) | What happens |
|-------|-----------|--------------|
| **Startup** | on launch | Singleton lockfile guard; init broker (IBKR) + data + executor; import + register all strategies; allocate capital slices from config |
| **Warm-up** | on launch | `strategy.restore_state()` on each enabled strategy; `options_premium` purges expired contracts from restored state to prevent stale close loops |
| **Morning reconciliation** | on launch | `_morning_reconciliation()` compares broker positions vs strategy-restored held symbols; emits `unexpected_position_found` on mismatch |
| **Pre-market intel** | before open | Regime detector pulls live VIX (IBKR `Index("VIX","CBOE")` persistent stream); event calendar checked for FOMC/CPI/earnings |
| **Trading loop** | 9:30–15:55 | Each cycle: detect regime → enabled strategies emit `OpportunityAssessment` with `TradeCandidate` rows → Brain may re-decide (cadence + event triggers, see below) → executor places orders → strategies manage existing positions |
| **Morning open review** | ~9:30–11:00 | `_morning_position_review()` can trigger per-strategy `review_positions_at_open()` for gap-stop exits |
| **Expiry guard** | 15:45 | `_check_expiry_risk()` force-closes any options position expiring same day to avoid auto-exercise into stock |
| **Close handling** | ~15:55 | `_handle_market_close()` flattens intraday-only books; per-strategy `close_positions_for_eod()` |
| **Persist** | each cycle / shutdown | `data/state/*.json` (`brain_decision.json`, `assessments.json`, `broker_snapshot.json`), per-strategy state |

---

## Decision engine: Daily Brain (dynamic cadence)

`algotrader/strategy_selector/brain.py` is the active selector. The legacy `scorer.py` + `allocator.py` "classic" path still exists but is gated behind `strategy_selector.mode: "classic"`; current config runs `mode: "brain"`.

### Cadence and triggers (new in 2026-05-10)

Previously the Brain ran exactly twice per day: an open decision and a midday review. That's been replaced by a dynamic cadence with explicit triggers, all configurable under `strategy_selector.brain`:

| Knob | Default | Meaning |
|------|---------|---------|
| `cadence_minutes` | 60 | Re-decide at most once per this interval during entry window |
| `entry_window_start` / `entry_window_end` | `09:30` / `15:30` | ET clock bounds for cadence-driven re-decides |
| `regime_change_triggers_decision` | true | Force a re-decide when `RegimeDetector` flips category (e.g. ranging → trending) |
| `vix_delta_trigger` | 1.5 | Force a re-decide when VIX moves by ≥ this much in one cycle |
| `late_session_confidence_bump` | 0.05 | Tighten the entry confidence floor by this much per hour past 12:00 ET |
| `max_daily_trades` | 5 | Cumulative cap across all cadence runs — cadence cannot bypass this |

Two safeguards keep the new cadence honest:

- **Cumulative trade cap** — `_brain_trades_today` is threaded into `_decide_impl()` as `max_new_trades`, so a 60-min cadence run can never push the day past `max_daily_trades`.
- **Idempotency guard** — if the candidate set hasn't changed since the last decision *and* there are no open positions, the cadence run is skipped (`brain_skip_no_new_info` in debug logs).

### Decision API

`decide_dynamic()` is the unified method called per cycle. It folds the old `decide()` (open) and `review_midday()` (close-early recommendations + P&L stop) into a single method that decides whether to run, what to run, and what tightening to apply based on time of day. The legacy `decide()` and `review_midday()` methods are retained for the classic-mode reviewer code path.

### Filters and gates

- Global gates: `min_confidence: 0.60`, `min_risk_reward: 1.5`, `min_edge_pct: 0.3`
- Options-specific gates (credit spreads need lower RR by nature): `options_min_confidence: 0.55`, `options_min_risk_reward: 0.3`, `options_min_edge_pct: 0.1`
- Per-strategy `strategy_overrides` allow each strategy its own thresholds — e.g. `momentum` requires RR ≥ 2.0, `pairs_trading` only needs RR ≥ 1.0

### Caps

- `max_daily_trades: 5`
- `max_capital_per_trade_pct: 25.0`
- `max_daily_risk_pct: 6.0` (with adaptive sizing tiers for single-strategy / few-strategies / diversified mixes)
- `max_contracts_hard_cap: 10`
- `cash_is_default: true` — when nothing clears thresholds, the Brain holds cash

### Adaptive behaviors

- `drawdown_governor`: tightens sizing at -1.5% and -3.0% intraday drawdown
- `recent_loss_cooldown_hours: 4`: penalizes a strategy after a loss
- `regime_mismatch_penalty: 0.5`, `correlation_penalty: 0.3`: applied during scoring
- `midday_pnl_stop_pct: -1.0`: tightens further if down >1%
- `late_session_confidence_bump`: progressive tightening as the day wears on (see cadence table)

### Output

`BrainDecision` (selected trades, rejected trades with reasons, cash & risk allocation, audit-ready reasoning, plus `context` fields like `regime_change` / `vix_delta` indicating *why* this run fired) → persisted to `data/state/brain_decision.json`.

---

## Strategies

All strategies subclass `StrategyBase` ([algotrader/strategies/base.py](algotrader/strategies/base.py)) and return `OpportunityAssessment` containing up to 3 `TradeCandidate` rows per cycle. The Brain merges across strategies — strategies do not allocate capital themselves.

| Strategy | File | CandidateType | Capital % | Notes |
|----------|------|---------------|-----------|-------|
| Options Premium | [options_premium.py](algotrader/strategies/options_premium.py) | `CREDIT_SPREAD` | 15 | 2-5 DTE SPY/QQQ/IWM credit spreads. BAG combos with parent `BUY` + per-leg `SELL`/`BUY`. ETF strike ladder normalization. Closes one trading day before expiry plus 15:45 ET exercise guard. Hard exercise-exposure cap at 20% equity. Min credit floor: `$0.10`/contract policy, `$50` per-spread current config. |
| Pairs Trading | [pairs_trading.py](algotrader/strategies/pairs_trading.py) | `PAIRS` | 25 | Cointegration + anti-churn (expected profit ≥ 3× est. cost) |
| Momentum / Breakout | [momentum.py](algotrader/strategies/momentum.py) | `LONG_EQUITY` / `SHORT_EQUITY` | 15 | Breakout + 1.5x volume confirmation |
| Gap Reversal | [gap_reversal.py](algotrader/strategies/gap_reversal.py) | `LONG_EQUITY` / `SHORT_EQUITY` | 12 | Entry expiry ~11:00 ET, EOD-flat |
| VWAP Reversion | [vwap_reversion.py](algotrader/strategies/vwap_reversion.py) | `LONG_EQUITY` / `SHORT_EQUITY` | 10 | Z-score mean reversion, ranging regime |
| Sector Rotation | [sector_rotation.py](algotrader/strategies/sector_rotation.py) | `SECTOR_LONG_SHORT` | 10 | Strongest vs weakest sector ETF |
| Event-Driven | [event_driven.py](algotrader/strategies/event_driven.py) | `EVENT_DIRECTIONAL` | 8 | Pre/post-event directional, event-day only |

Total nominal allocation: 95%. All seven won't typically be active simultaneously — regime gating in each strategy + Brain cash-default behavior keeps actual deployed capital in the 50-80% range.

To disable a strategy: flip `enabled: false` in `config/strategies/<name>.yaml` and the matching block in `config/settings.yaml.strategies` (the merge is logical AND).

---

## Broker layer

Broker selection via `settings.broker.provider` ∈ {`ibkr`, `alpaca`}. Currently `ibkr`.

### IBKR stack
- **Connection**: `IBKRConnection` singleton ([algotrader/execution/ibkr_connection.py](algotrader/execution/ibkr_connection.py)) — manages the `ib_async` connection, reconnect attempts, shared by data + executor.
- **Data**: `IBKRDataProvider` ([algotrader/data/ibkr_provider.py](algotrader/data/ibkr_provider.py)) — bars, quotes, snapshots, option chains, news bridge. Persistent VIX index subscription. ETF option strike normalization snaps malformed secdef strikes to ladder increments before contract qualification.
- **Executor**: `IBKRExecutor` ([algotrader/execution/ibkr_executor.py](algotrader/execution/ibkr_executor.py)) — equity orders, BAG multi-leg combos with explicit leg actions, option position helpers (`get_option_positions()` via `ib.portfolio()`, `close_option_position()` qualified by conId).

### Alpaca stack
Maintained as fallback for headless environments (no TWS). Same `DataProvider` / `Executor` interfaces, same strategy code runs against either.

### Broker fill ledger
[algotrader/tracking/broker_ledger.py](algotrader/tracking/broker_ledger.py) consumes `ib_async` `execDetailsEvent` + `commissionReportEvent` and persists every fill into a `broker_fills` table in `trades.db`. **This is the source of truth** for daily P&L, Brain win-rate inputs, and position reconciliation. The strategy-internal `trades` table is preserved as a separate accounting record.

---

## Risk management

Two layers, both active at all times:

### Portfolio-level ([algotrader/risk/portfolio_risk.py](algotrader/risk/portfolio_risk.py))
| Control | Value | Action |
|---------|-------|--------|
| `max_daily_loss_pct` | 2.0% | Kill switch — emit `KILL_SWITCH`, close all, halt |
| `max_drawdown_pct` | 8.0% | Reduce sizing |
| `max_gross_exposure_pct` | 80.0% | Block new entries |
| `max_single_position_pct` | 10.0% | Hard per-trade cap |
| `max_overnight_exposure_pct` | 40.0% | Block overnight carry beyond cap |
| `max_correlated_positions` | 3 | Block new entries in same sector |
| `strategy_daily_loss_limit_pct` | 1.0% | `STRATEGY_DISABLED` event for the day |

### Per-trade
- `PositionSizer` ([algotrader/risk/position_sizer.py](algotrader/risk/position_sizer.py)): risk-based sizing tied to stop distance
- `options_premium` enforces a hard contract cap via `exercise_exposure_cap_pct` (default 20%): `max_contracts = floor(equity * cap_pct / (strike * 100))`, minimum 1
- Quote validation: max spread 0.3% before entry
- Stop required on every position before entry

### Always-on safeguards
- Singleton lockfile guard in `scripts/run.py` (Windows `tasklist`-based liveness check)
- Live IBKR ports `7496`/`4001` rejected at startup
- `_check_expiry_risk()` runs at 15:45 ET — non-disable-able by config
- BAG direction sanity: parent `BUY`, leg-level `SELL`/`BUY`. A credit spread filling as net debit indicates a bug — flatten and investigate.

---

## Intelligence layer

| Module | Purpose |
|--------|---------|
| `intelligence/regime.py` | `RegimeDetector` — VIX level/change, SPY/QQQ trend, volatility bucket, `event_day` flag. **Drives Brain regime-change trigger.** |
| `intelligence/scanners/gap_scanner.py` | Pre-market gap detection |
| `intelligence/scanners/volume_scanner.py` | Unusual-volume detection |
| `intelligence/scanners/breakout_scanner.py` | Technical breakout detection |
| `intelligence/news/alpaca_news.py` | Alpaca news API client |
| `intelligence/news/scraper.py` | Web scrape (finviz/yahoo) |
| `intelligence/calendar/events.py` | Seeded economic + earnings calendar; drives `event_day` regime flag |

Scanners and news are wired into the strategies that need them; the Brain itself reads only regime + candidates + calendar.

---

## Tracking and observability

| Module | Role |
|--------|------|
| `tracking/broker_ledger.py` | **Authoritative** fill ledger from IBKR exec events |
| `tracking/journal.py` | Strategy-side trade journal (intent + outcome) |
| `tracking/portfolio.py` | Live portfolio tracker (NAV, exposure, positions) |
| `tracking/metrics.py` | Performance metrics calculator |
| `tracking/attribution.py` | Per-strategy / per-regime P&L attribution |
| `tracking/learner.py` | `StrategyWeightLearner` adjusts weights from historical performance |
| `tracking/alerts.py` | Alert manager with log/file/webhook backends; big-win/loss thresholds in config |

State persistence (recovered on next launch):
- `data/state/brain_decision.json` — last Brain decision
- `data/state/assessments.json` — last cycle's strategy assessments
- `data/state/broker_snapshot.json` — broker-reported positions snapshot
- Per-strategy state files for restore
- `trades.db` (SQLite) — `broker_fills` + `trades` tables
- `data/logs/algotrader-YYYY-MM-DD.log` — daily-rotated structured JSON logs

### New log signals to watch (post-2026-05-10)

- `regime_change` and `vix_delta` in `brain_decision.context` — the *reason* an out-of-cadence Brain run fired
- `brain_skip_no_new_info` in debug logs — idempotency guard suppressing a cadence run
- Brain decision count per day expected to rise from 2 to ~6 under default cadence

---

## Config

- `config/settings.yaml` — global (capital, broker, risk caps, Brain thresholds + cadence, strategy enable flags)
- `config/regimes.yaml` — regime definitions
- `config/strategies/<name>.yaml` — per-strategy parameters

YAML (not JSON), loaded via pydantic settings.

---

## Repo layout

```
algotrader/
├── orchestrator.py              # Lifecycle controller
├── core/                        # Models, config, events, logging
├── data/                        # DataProvider + Alpaca/IBKR impls + cache
├── execution/                   # Executor + Alpaca/IBKR impls + connection + order_manager
├── intelligence/                # Regime, scanners (gap/volume/breakout), news, calendar
├── strategies/                  # 7 plugins + base + registry
├── strategy_selector/           # brain.py (active), candidate.py, scorer/allocator (legacy), reviewer
├── risk/                        # portfolio_risk, position_sizer
└── tracking/                    # broker_ledger, journal, portfolio, metrics,
                                 # attribution, learner, alerts

config/                          # YAML configs
dashboard/app.py                 # Streamlit dashboard
scripts/                         # run.py, check_live.py, cleanup.py, test_ibkr_connection.py, analyze_trades.py
tests/unit/  tests/integration/  # pytest (93 tests passing)
data/state/  data/logs/          # runtime state and logs (gitignored)
```

---

## Run commands

```bash
uv sync                                            # install deps
uv run python scripts/run.py                       # main bot
uv run python scripts/test_ibkr_connection.py      # check IBKR conn only
uv run python -m streamlit run dashboard/app.py    # dashboard
uv run pytest tests/ -q                            # tests
```

---

## Operational reality

**Last week (Mon 5/4 – Fri 5/8)**: bot ran clean. 51 `brain_cash_day` decisions across the week, all coherent — VIX 16-17 + low-confidence candidates → 100% cash. Wednesday was an `event_day`. Zero trades was correct under the *old* two-decision-per-day Brain. Equity ended at $104,255 with no positions carried over.

**Why this drove the rewrite**: the cash-day pattern wasn't really a candidate-quality problem — it was a cadence problem. The 9:30 open decision saw one regime, and by midday the regime had shifted but the Brain wasn't re-deciding. Re-enabling all seven strategies under the *old* Brain would have produced the same cash days. With dynamic cadence, the Brain now re-evaluates hourly and on regime/VIX shifts, so non-options strategies actually get a chance to fire.

**Expected behavior under the new system**: Brain decisions/day rise from 2 to ~6; all 7 strategies producing candidates; daily trade cap still enforced at 5; new log signals (`regime_change` / `vix_delta` in `brain_decision.context`, `brain_skip_no_new_info`) provide visibility into *why* each run did or didn't fire.

**Known operational quirks** (IBKR-specific):
- TWS/Gateway must be running before bot start; bot is the client
- `client_id` must be unique per concurrent bot session
- Same-day ITM options auto-exercise into stock — paper accounts carry without warning, hence the 15:45 ET expiry guard
- `ib.portfolio()` is the reliable position source, not `positions()`
- Option commissions: `$0.65–$1.05` per contract per leg; tiny credits are unprofitable after fees
- Overnight `Error 1100`/`1102` storms during IBKR Gateway maintenance window (~00:30–00:50 ET) are expected and auto-recover

---

## Scope guardrails

Per [CLAUDE.md](CLAUDE.md), **do not modify without explicit request**:
- Protocol interfaces (`data/provider.py`, `execution/executor.py`)
- Risk management logic (`risk/*.py`)
- Tracking/journal/metrics/attribution logic (`tracking/*.py`)

These are load-bearing for correctness across the system; changes ripple unexpectedly.
