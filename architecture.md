# AlgoTrader IBKR — Architecture (Current State)

*Snapshot as of 2026-05-10. Reflects the system as it actually runs, not as originally designed.*

---

## What this system is

A single-process Python trading system that runs against Interactive Brokers (paper) with an Alpaca fallback. Each cycle, every enabled strategy emits concrete `TradeCandidate` rows; the **Daily Brain** filters, scores, and ranks them across strategies and deploys capital only into the highest-EV setups. Cash is the default posture.

In practice today, **only `options_premium` is enabled**. The other six strategies remain in the repo with code and tests intact, but are config-disabled because eight weeks of paper trading showed only the credit-spread strategy producing real edge in the current regime.

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
│Alpaca │  │  Alpaca  │  │ • Scanners   │  │ (1 active) │  │ + Sizer  │
└───────┘  └──────────┘  │ • News       │  └─────┬──────┘  └──────────┘
                         │ • Calendar   │        │
                         └──────────────┘        ▼
                                          ┌──────────────┐
                                          │ DailyBrain   │
                                          │ (selector)   │
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

The orchestrator drives one process per trading day. Cycle interval is 5 minutes (`data.cycle_interval_seconds: 300`).

| Phase | When (ET) | What happens |
|-------|-----------|--------------|
| **Startup** | on launch | Singleton lockfile guard; init broker (IBKR) + data + executor; import + register all strategies; allocate capital slices from config |
| **Warm-up** | on launch | `strategy.restore_state()` on each enabled strategy; `options_premium` purges expired contracts from restored state to prevent stale close loops |
| **Morning reconciliation** | on launch | `_morning_reconciliation()` compares broker positions vs strategy-restored held symbols; emits `unexpected_position_found` on mismatch |
| **Pre-market intel** | before open | Regime detector pulls live VIX (IBKR `Index("VIX","CBOE")` persistent stream); event calendar checked for FOMC/CPI/earnings |
| **Trading loop** | 9:30–~3:55 | Each cycle: detect regime → enabled strategies emit `OpportunityAssessment` with `TradeCandidate` rows → Brain filters/scores/ranks → executor places orders → strategies manage existing positions |
| **Morning open review** | ~9:30–11:00 | `_morning_position_review()` can trigger per-strategy `review_positions_at_open()` for gap-stop exits |
| **Midday review** | 12:00 | `MidDayReviewer` runs once; can tighten Brain thresholds and emit close recommendations |
| **Expiry guard** | 3:45 | `_check_expiry_risk()` force-closes any options position expiring same day to avoid auto-exercise into stock |
| **Close handling** | ~3:55 | `_handle_market_close()` flattens intraday-only books; per-strategy `close_positions_for_eod()` |
| **Persist** | each cycle / shutdown | `data/state/*.json` (`brain_decision.json`, `assessments.json`, `broker_snapshot.json`, per-strategy state) |

---

## Decision engine: Daily Brain

`algotrader/strategy_selector/brain.py` is the active selector. The legacy `scorer.py` + `allocator.py` "classic" path still exists but is gated behind `strategy_selector.mode: "classic"`; current config runs `mode: "brain"`.

**Inputs**: every `TradeCandidate` from every enabled strategy this cycle, current `MarketRegime`, recent fill history (from `BrokerLedger`), open positions, capital state.

**Filters** (from `config/settings.yaml` → `strategy_selector.brain`):

- Global gates: `min_confidence: 0.60`, `min_risk_reward: 1.5`, `min_edge_pct: 0.3`
- Options-specific gates (credit spreads need lower RR by nature): `options_min_confidence: 0.55`, `options_min_risk_reward: 0.3`, `options_min_edge_pct: 0.1`
- Per-strategy `strategy_overrides` allow each strategy its own thresholds — e.g. `momentum` requires RR ≥ 2.0, `pairs_trading` only needs RR ≥ 1.0

**Caps**:

- `max_daily_trades: 5`
- `max_capital_per_trade_pct: 25.0`
- `max_daily_risk_pct: 6.0` (with adaptive sizing tiers for single-strategy / few-strategies / diversified mixes)
- `max_contracts_hard_cap: 10`
- `cash_is_default: true` — when nothing clears thresholds, the Brain holds cash

**Adaptive behaviors**:

- `drawdown_governor`: tightens sizing at -1.5% and -3.0% intraday drawdown
- `recent_loss_cooldown_hours: 4`: penalizes a strategy after a loss
- `regime_mismatch_penalty: 0.5`, `correlation_penalty: 0.3`: applied during scoring
- `midday_pnl_stop_pct: -1.0`: midday review tightens further if down >1%

**Output**: `BrainDecision` (selected trades, rejected trades with reasons, cash & risk allocation, audit-ready reasoning) → persisted to `data/state/brain_decision.json`.

---

## Strategies

All strategies subclass `StrategyBase` (`algotrader/strategies/base.py`) and return `OpportunityAssessment` containing up to 3 `TradeCandidate` rows per cycle. The Brain merges across strategies — strategies do not allocate capital themselves.

| Strategy | File | CandidateType | Enabled | Notes |
|----------|------|---------------|---------|-------|
| Options Premium | [options_premium.py](algotrader/strategies/options_premium.py) | `CREDIT_SPREAD` | **yes** | 2-5 DTE SPY/QQQ/IWM credit spreads. BAG combos with parent `BUY` + per-leg `SELL`/`BUY`. Strike ladder normalization for ETFs. Closes one trading day before expiry plus 3:45 ET exercise guard. Hard exercise-exposure cap at 20% equity. Min credit floor: `$0.10`/contract policy, `$50` per-spread in current config. |
| Momentum / Breakout | [momentum.py](algotrader/strategies/momentum.py) | `LONG_EQUITY` / `SHORT_EQUITY` | no | Breakout + 1.5x volume confirmation |
| VWAP Reversion | [vwap_reversion.py](algotrader/strategies/vwap_reversion.py) | `LONG_EQUITY` / `SHORT_EQUITY` | no | Z-score mean reversion, ranging regime |
| Pairs Trading | [pairs_trading.py](algotrader/strategies/pairs_trading.py) | `PAIRS` | no | Cointegration + anti-churn (expected profit ≥ 3× est. cost) |
| Gap Reversal | [gap_reversal.py](algotrader/strategies/gap_reversal.py) | `LONG_EQUITY` / `SHORT_EQUITY` | no | Entry expiry ~11:00 ET, EOD-flat |
| Sector Rotation | [sector_rotation.py](algotrader/strategies/sector_rotation.py) | `SECTOR_LONG_SHORT` | no | Strongest vs weakest sector ETF |
| Event-Driven | [event_driven.py](algotrader/strategies/event_driven.py) | `EVENT_DIRECTIONAL` | no | Pre/post-event directional, event-day only |

Disabled strategies still:
- Have their code in the repo and pass tests
- Have Brain threshold overrides retained in `settings.yaml` for future re-activation
- Are skipped at startup (registry honors `enabled: false`)

To re-enable: flip `enabled: true` in `config/strategies/<name>.yaml` (and the matching block in `settings.yaml.strategies` if present).

---

## Broker layer

Broker selection via `settings.broker.provider` ∈ {`ibkr`, `alpaca`}. Currently `ibkr`.

### IBKR stack
- **Connection**: `IBKRConnection` singleton ([algotrader/execution/ibkr_connection.py](algotrader/execution/ibkr_connection.py)) — manages the `ib_async` connection, reconnect attempts, shared by data + executor.
- **Data**: `IBKRDataProvider` ([algotrader/data/ibkr_provider.py](algotrader/data/ibkr_provider.py)) — bars, quotes, snapshots, option chains, news bridge. Persistent VIX index subscription. ETF option strike normalization snaps malformed secdef strikes to ladder increments before contract qualification.
- **Executor**: `IBKRExecutor` ([algotrader/execution/ibkr_executor.py](algotrader/execution/ibkr_executor.py)) — equity orders, BAG multi-leg combos with explicit leg actions, option position helpers (`get_option_positions()` via `ib.portfolio()`, `close_option_position()` qualified by conId).

### Alpaca stack
Maintained as fallback for headless environments (no TWS). Same `DataProvider` / `Executor` interfaces, same strategy code runs against either.

### Broker fill ledger (Phase 1 — commit `e653ea7`)
[algotrader/tracking/broker_ledger.py](algotrader/tracking/broker_ledger.py) consumes `ib_async` `execDetailsEvent` + `commissionReportEvent` and persists every fill into a `broker_fills` table in `trades.db`. **This is the source of truth** for daily P&L, Brain win-rate inputs, and position reconciliation. The strategy-internal `trades` table is preserved as a separate accounting record.

---

## Risk management

Two layers, both active at all times:

### Portfolio-level (`algotrader/risk/portfolio_risk.py`)
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
- `PositionSizer` (`algotrader/risk/position_sizer.py`): risk-based sizing tied to stop distance
- `options_premium` enforces a hard contract cap via `exercise_exposure_cap_pct` (default 20%): `max_contracts = floor(equity * cap_pct / (strike * 100))`, minimum 1
- Quote validation: max spread 0.3% before entry
- Stop required on every position before entry

### Always-on safeguards
- Singleton lockfile guard in `scripts/run.py` (Windows `tasklist`-based liveness check)
- Live IBKR ports `7496`/`4001` rejected at startup
- `_check_expiry_risk()` runs near 3:45 ET — non-disable-able by config
- BAG direction sanity: parent `BUY`, leg-level `SELL`/`BUY`. A credit spread filling as net debit indicates a bug — flatten and investigate.

---

## Intelligence layer

| Module | Purpose |
|--------|---------|
| `intelligence/regime.py` | `RegimeDetector` — VIX level/change, SPY/QQQ trend, volatility bucket, `event_day` flag |
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

---

## Config

- `config/settings.yaml` — global (capital, broker, risk caps, Brain thresholds, strategy enable flags)
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
tests/unit/  tests/integration/  # pytest
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

**Performance to date**: Week 1 paper trading (Feb 25–Mar 4, 2026) showed +1.67% NAV, but options strategy P&L was actually negative (`-$121.45` over Mar 2-4); reported gain came from accidental stock carries via same-day expiry exercise *before* the March 5 BAG-direction fix. Treat pre-fix gains as non-representative.

**Why options-only**: Brain EV-concentration correctly allocated 100% to options because the other strategies either produced zero candidates or candidates below threshold under observed regime conditions. This is a market-regime / candidate-quality issue, not a code issue. The infrastructure to bring strategies back online is one config flag away.

**Known operational quirks** (IBKR-specific):
- TWS/Gateway must be running before bot start; bot is the client
- `client_id` must be unique per concurrent bot session
- Same-day ITM options auto-exercise into stock — paper accounts carry without warning, hence the 3:45 ET expiry guard
- `ib.portfolio()` is the reliable position source, not `positions()`
- Option commissions: `$0.65–$1.05` per contract per leg; tiny credits are unprofitable after fees

---

## Scope guardrails

Per [CLAUDE.md](CLAUDE.md), **do not modify without explicit request**:
- Protocol interfaces (`data/provider.py`, `execution/executor.py`)
- Risk management logic (`risk/*.py`)
- Tracking/journal/metrics/attribution logic (`tracking/*.py`)

These are load-bearing for correctness across the system; changes ripple unexpectedly.
