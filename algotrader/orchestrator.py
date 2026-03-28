"""Main lifecycle controller for the algotrader system.

Lifecycle: initialize → warm up strategies → run loop
(check market open → run active strategies → manage risk) → shutdown.
"""

from __future__ import annotations

import json
import signal as os_signal
import time
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytz
import structlog

from algotrader.core.config import Settings, StrategyConfig, load_strategy_config, load_yaml
from algotrader.core.events import EventBus, KILL_SWITCH, STRATEGY_DISABLED
from algotrader.core.logging import setup_logging
from algotrader.core.models import MarketRegime, OrderSide, RegimeType
from algotrader.data.alpaca_provider import AlpacaDataProvider
from algotrader.data.cache import DataCache
from algotrader.execution.alpaca_executor import AlpacaExecutor
from algotrader.execution.order_manager import OrderManager
from algotrader.intelligence.regime import RegimeDetector
from algotrader.intelligence.scanners.gap_scanner import GapScanner
from algotrader.intelligence.scanners.volume_scanner import VolumeScanner
from algotrader.intelligence.news.alpaca_news import AlpacaNewsClient
from algotrader.intelligence.calendar.events import EventCalendar
from algotrader.risk.portfolio_risk import PortfolioRiskManager
from algotrader.risk.position_sizer import PositionSizer
from algotrader.strategies.base import StrategyBase
from algotrader.strategies.registry import registry
from algotrader.strategy_selector.scorer import StrategyScorer
from algotrader.strategy_selector.allocator import CapitalAllocator
from algotrader.strategy_selector.reviewer import MidDayReviewer
from algotrader.tracking.journal import TradeJournal
from algotrader.tracking.portfolio import PortfolioTracker
from algotrader.tracking.metrics import MetricsCalculator
from algotrader.tracking.attribution import PerformanceAttribution
from algotrader.tracking.learner import StrategyWeightLearner
from algotrader.tracking.alerts import (
    AlertManager, AlertType, AlertLevel, Alert,
    LogAlertBackend, FileAlertBackend, WebhookAlertBackend,
)

logger = structlog.get_logger()

if TYPE_CHECKING:
    from algotrader.strategy_selector.brain import BrainDecision, DailyBrain


class Orchestrator:
    """Main lifecycle controller.

    Manages the full trading day:
    1. Initialize all components
    2. Warm up strategies with historical data
    3. Run trading loop on configured cycle interval
    4. Shutdown gracefully on market close or kill switch
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._running = False
        self._shutdown_requested = False
        self._log = logger.bind(component="orchestrator")

        # Set up logging
        setup_logging(
            level=settings.logging.level,
            log_file=settings.logging.file,
            json_format=settings.logging.json_format,
        )

        # Core components
        self._event_bus = EventBus()
        self._data_cache = DataCache()
        self._ibkr_conn = None

        broker_provider = settings.broker.provider.lower()
        if broker_provider == "ibkr":
            from algotrader.data.ibkr_provider import IBKRDataProvider
            from algotrader.execution.ibkr_connection import IBKRConnection
            from algotrader.execution.ibkr_executor import IBKRExecutor

            self._ibkr_conn = IBKRConnection.get_instance(
                config=settings.broker.ibkr,
                event_bus=self._event_bus,
            )
            if not self._ibkr_conn.connect():
                raise RuntimeError("Failed to connect to IBKR TWS/Gateway")

            self._data_provider = IBKRDataProvider(connection=self._ibkr_conn)
            self._executor = IBKRExecutor(connection=self._ibkr_conn)
            self._log.info("broker_initialized", provider="ibkr")
        else:
            # Data provider
            self._data_provider = AlpacaDataProvider(
                config=settings.alpaca,
                feed=settings.data.feed,
            )

            # Executor
            self._executor = AlpacaExecutor(config=settings.alpaca)
            self._log.info("broker_initialized", provider="alpaca")

        # Order manager
        self._order_manager = OrderManager(
            executor=self._executor,
            event_bus=self._event_bus,
        )

        # Account info for initial capital
        account = self._executor.get_account()
        starting_equity = account.equity
        self._log.info("account_loaded", equity=starting_equity, cash=account.cash)

        # Risk management
        self._risk_manager = PortfolioRiskManager(
            config=settings.risk,
            executor=self._executor,
            event_bus=self._event_bus,
            starting_equity=starting_equity,
        )

        # Position sizer
        self._position_sizer = PositionSizer(
            config=settings.risk,
            total_capital=starting_equity,
        )

        # Tracking
        self._portfolio_tracker = PortfolioTracker(
            executor=self._executor,
            starting_equity=starting_equity,
        )
        self._trade_journal = TradeJournal()

        # Intelligence layer
        self._event_calendar = EventCalendar()
        self._regime_detector = RegimeDetector(
            data_provider=self._data_provider,
            event_bus=self._event_bus,
            event_calendar=self._event_calendar,
        )
        self._gap_scanner = GapScanner(data_provider=self._data_provider)
        self._volume_scanner = VolumeScanner(data_provider=self._data_provider)
        self._news_client = AlpacaNewsClient(data_provider=self._data_provider)
        self._current_regime: MarketRegime | None = None

        # Strategy selector: Daily Brain (new) or classic scorer/allocator path.
        self._scorer: StrategyScorer | None = None
        self._allocator: CapitalAllocator | None = None
        self._reviewer: MidDayReviewer | None = None
        self._brain: DailyBrain | None = None
        self._use_brain: bool = False
        self._last_brain_open_decision: date | None = None
        self._last_brain_midday_review: date | None = None
        self._expiry_guard_date: date | None = None
        self._expiry_guard_attempted_conids: set[int] = set()
        self._last_morning_position_review: date | None = None
        self._eod_handled_date: date | None = None
        self._max_overnight_exposure_pct = float(
            getattr(settings.risk, "max_overnight_exposure_pct", 40.0),
        )

        if settings.strategy_selector.enabled:
            try:
                regime_config = load_yaml(settings.strategy_selector.regime_config)
                selector_mode = settings.strategy_selector.mode.lower().strip()

                if selector_mode == "brain":
                    from algotrader.strategy_selector.brain import DailyBrain

                    brain_cfg = settings.strategy_selector.brain
                    self._brain = DailyBrain(
                        total_capital=settings.trading.total_capital,
                        regime_config=regime_config,
                        trade_journal=self._trade_journal,
                        event_calendar=self._event_calendar,
                        min_confidence=brain_cfg.min_confidence,
                        min_risk_reward=brain_cfg.min_risk_reward,
                        min_edge_pct=brain_cfg.min_edge_pct,
                        options_min_confidence=brain_cfg.options_min_confidence,
                        options_min_risk_reward=brain_cfg.options_min_risk_reward,
                        options_min_edge_pct=brain_cfg.options_min_edge_pct,
                        max_daily_trades=brain_cfg.max_daily_trades,
                        max_capital_per_trade_pct=brain_cfg.max_capital_per_trade_pct,
                        max_daily_risk_pct=brain_cfg.max_daily_risk_pct,
                        cash_is_default=brain_cfg.cash_is_default,
                        regime_mismatch_penalty=brain_cfg.regime_mismatch_penalty,
                        correlation_penalty=brain_cfg.correlation_penalty,
                        recent_loss_cooldown_hours=brain_cfg.recent_loss_cooldown_hours,
                        midday_confidence_multiplier=brain_cfg.midday_confidence_multiplier,
                        midday_pnl_stop_pct=brain_cfg.midday_pnl_stop_pct,
                        adaptive_sizing=brain_cfg.adaptive_sizing,
                        adaptive_risk_tiers=brain_cfg.adaptive_risk_tiers.model_dump(),
                        drawdown_governor=brain_cfg.drawdown_governor.model_dump(),
                        max_contracts_hard_cap=brain_cfg.max_contracts_hard_cap,
                        recent_win_rate_lookback_trades=brain_cfg.recent_win_rate_lookback_trades,
                        recent_win_rate_fallback=brain_cfg.recent_win_rate_fallback,
                    )
                    self._use_brain = True
                    self._log.info("strategy_selector_initialized", mode="brain")
                else:
                    self._scorer = StrategyScorer(
                        regime_config=regime_config,
                        trade_journal=self._trade_journal,
                        event_calendar=self._event_calendar,
                        total_capital=settings.trading.total_capital,
                    )
                    self._allocator = CapitalAllocator(
                        total_capital=settings.trading.total_capital,
                        risk_config=settings.risk,
                        strategy_configs=settings.strategies,
                        min_allocation_pct=settings.strategy_selector.min_allocation_pct,
                        cash_threshold=settings.strategy_selector.cash_threshold,
                        max_single_strategy_pct=settings.strategy_selector.max_single_strategy_pct,
                        concentration_power=settings.strategy_selector.concentration_power,
                    )
                    self._reviewer = MidDayReviewer(
                        scorer=self._scorer,
                        allocator=self._allocator,
                        event_bus=self._event_bus,
                        strategy_configs=settings.strategies,
                        total_capital=settings.trading.total_capital,
                        review_hour=settings.strategy_selector.review_hour,
                        review_minute=settings.strategy_selector.review_minute,
                        scale_down_threshold_pct=settings.strategy_selector.scale_down_threshold_pct,
                        disable_threshold_pct=settings.strategy_selector.disable_threshold_pct,
                        scale_up_threshold_pct=settings.strategy_selector.scale_up_threshold_pct,
                        scale_up_factor=settings.strategy_selector.scale_up_factor,
                        scale_down_factor=settings.strategy_selector.scale_down_factor,
                        min_trades_for_review=settings.strategy_selector.min_trades_for_review,
                    )
                    self._use_brain = False
                    self._log.info("strategy_selector_initialized", mode="classic")
            except Exception:
                self._log.exception("strategy_selector_init_failed")
                self._scorer = None
                self._allocator = None
                self._reviewer = None
                self._brain = None
                self._use_brain = False

        # Phase 5: Metrics, Attribution, Learner
        self._metrics = MetricsCalculator(trade_journal=self._trade_journal)
        self._attribution = PerformanceAttribution(
            trade_journal=self._trade_journal,
            metrics_calculator=self._metrics,
        )
        self._learner = StrategyWeightLearner(
            trade_journal=self._trade_journal,
            metrics_calculator=self._metrics,
        )

        # Phase 5: Alerting
        alert_backends: list = [LogAlertBackend()]
        if settings.alerts.enabled:
            alert_backends.append(FileAlertBackend(filepath=settings.alerts.alert_file))
            if settings.alerts.webhook_url:
                alert_backends.append(WebhookAlertBackend(settings.alerts.webhook_url))
        self._alert_manager = AlertManager(
            event_bus=self._event_bus,
            backends=alert_backends,
            big_win_threshold=settings.alerts.big_win_threshold,
            big_loss_threshold=settings.alerts.big_loss_threshold,
        )

        # Dashboard state cycle counter
        self._cycle_count = 0

        # Pre-market refresh tracking
        self._last_premarket_refresh: float = 0.0
        self._premarket_refresh_interval = 900  # 15 minutes
        self._volume_refreshed_today = False

        # Strategies
        self._strategies: dict[str, StrategyBase] = {}

        # Event subscriptions
        self._event_bus.subscribe(KILL_SWITCH, self._on_kill_switch)
        self._event_bus.subscribe(STRATEGY_DISABLED, self._on_strategy_disabled)

        # Signal handling for graceful shutdown
        os_signal.signal(os_signal.SIGINT, self._signal_handler)
        os_signal.signal(os_signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle OS signals for graceful shutdown."""
        if self._shutdown_requested:
            self._log.info("shutdown_signal_received_again", signal=signum)
            return

        self._log.info("shutdown_signal_received", signal=signum)
        self._shutdown_requested = True
        self._running = False

        # Interrupt active IB requests so warm-up/startup can unwind quickly.
        if self._ibkr_conn is not None:
            try:
                self._ibkr_conn.disconnect()
            except Exception:
                self._log.exception("ibkr_disconnect_on_signal_failed")

    def initialize(self) -> None:
        """Initialize strategies from config and registry."""
        self._log.info("initializing")

        # Import strategy modules to trigger registration
        self._import_strategies()

        # Create strategies from config
        for name in registry.names:
            if self._shutdown_requested:
                self._log.info("initialize_aborted_shutdown_requested")
                break

            config = self._resolve_strategy_config(name)

            if not config.enabled:
                self._log.info("strategy_disabled", name=name)
                continue

            strategy = registry.create(
                name=name,
                config=config,
                data_provider=self._data_provider,
                executor=self._executor,
                event_bus=self._event_bus,
            )

            if strategy:
                # Set capital allocation and trade journal
                capital = self._settings.trading.total_capital * (config.capital_allocation_pct / 100)
                strategy.set_capital(capital)
                strategy.set_journal(self._trade_journal)
                self._strategies[name] = strategy
                self._log.info("strategy_initialized", name=name, capital=capital)

        self._log.info("initialization_complete", strategies=list(self._strategies.keys()))

    def _resolve_strategy_config(self, name: str) -> StrategyConfig:
        """Resolve strategy config by merging file defaults with inline overrides."""
        file_config = load_strategy_config(name)
        inline_config = self._settings.strategies.get(name)
        if inline_config is None:
            return file_config

        merged_params = dict(file_config.params)
        merged_params.update(inline_config.params or {})
        return StrategyConfig(
            enabled=inline_config.enabled,
            capital_allocation_pct=inline_config.capital_allocation_pct,
            max_positions=inline_config.max_positions,
            params=merged_params,
        )

    def _import_strategies(self) -> None:
        """Import strategy modules to trigger @register_strategy decorators."""
        strategy_modules = [
            "algotrader.strategies.pairs_trading",
            "algotrader.strategies.gap_reversal",
            "algotrader.strategies.momentum",
            "algotrader.strategies.vwap_reversion",
            "algotrader.strategies.options_premium",
            "algotrader.strategies.sector_rotation",
            "algotrader.strategies.event_driven",
        ]
        for module in strategy_modules:
            try:
                __import__(module)
            except ImportError:
                self._log.warning("strategy_import_failed", module=module)

    def warm_up(self) -> None:
        """Warm up all strategies with historical data."""
        self._log.info("warming_up_strategies")

        for name, strategy in self._strategies.items():
            if self._shutdown_requested:
                self._log.info("warm_up_aborted_shutdown_requested")
                break

            try:
                # Try to restore saved state first
                if strategy.restore_state():
                    self._log.info("strategy_state_restored", name=name)

                strategy.warm_up()
                self._log.info("strategy_warmed_up", name=name)
            except Exception:
                self._log.exception("strategy_warm_up_failed", name=name)
                strategy.disable("warm_up_failed")

    def run(self) -> None:
        """Main run loop. Blocks until shutdown."""
        self.initialize()
        if self._shutdown_requested:
            self.shutdown()
            return

        self.warm_up()
        if self._shutdown_requested:
            self.shutdown()
            return

        # Startup safety: compare broker positions to strategy-held symbols and
        # flag unexpected carry from prior sessions (e.g., exercise/assignment).
        self._morning_reconciliation()
        if self._shutdown_requested:
            self.shutdown()
            return

        # Pre-market intelligence
        self._run_pre_market_intelligence()
        if self._shutdown_requested:
            self.shutdown()
            return

        self._running = True
        cycle_interval = self._settings.data.cycle_interval_seconds
        self._log.info("trading_loop_starting", cycle_seconds=cycle_interval)

        # Reset portfolio tracker for today
        account = self._executor.get_account()
        self._portfolio_tracker.reset_day(account.equity)
        self._risk_manager.reset_day(account.equity)
        try:
            self._write_dashboard_state()
        except Exception:
            self._log.debug("initial_dashboard_state_write_failed")

        while self._running and not self._shutdown_requested:
            cycle_start = time.time()

            try:
                self._run_cycle()
            except Exception:
                self._log.exception("cycle_error")

            # Sleep until next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, cycle_interval - elapsed)

            if sleep_time > 0 and self._running:
                self._log.debug("cycle_sleeping", seconds=round(sleep_time, 1))
                # Sleep in small increments to allow signal handling
                end_time = time.time() + sleep_time
                while time.time() < end_time and self._running:
                    time.sleep(min(1.0, end_time - time.time()))

        self.shutdown()

    def _run_cycle(self) -> None:
        """Execute one trading cycle."""
        if self._shutdown_requested:
            return

        # Check if market is open
        if not self._data_provider.is_market_open():
            try:
                self._write_dashboard_state()
            except Exception:
                self._log.debug("dashboard_state_write_failed")
            self._maybe_refresh_premarket()
            return

        # Check if killed
        if self._risk_manager.is_killed:
            self._log.warning("risk_killed_skipping_cycle")
            return

        # Update regime detection
        try:
            self._current_regime = self._regime_detector.detect()
        except Exception:
            self._log.exception("regime_detection_failed")

        # Check pending orders
        self._order_manager.check_orders()

        if self._needs_morning_position_review():
            self._morning_position_review()

        # Daily Brain: open decision + mid-day review
        if self._use_brain and self._brain and self._current_regime:
            try:
                if self._needs_brain_open_decision():
                    if self._is_overnight_exposure_limited():
                        self._block_new_entries_for_exposure_limit(context="market_open")
                    else:
                        assessments = self._assess_all_strategies(self._current_regime)
                        decision = self._brain.decide(
                            regime=self._current_regime,
                            assessments=assessments,
                            current_positions=self._executor.get_positions(),
                            daily_pnl=self._get_daily_pnl(),
                            vix_level=self._current_regime.vix_level,
                        )
                        self._apply_brain_decision(decision, context="market_open")
                    self._last_brain_open_decision = datetime.now(
                        pytz.timezone("America/New_York"),
                    ).date()

                if self._needs_brain_midday_review():
                    if self._is_overnight_exposure_limited():
                        self._block_new_entries_for_exposure_limit(context="midday")
                    else:
                        assessments = self._assess_all_strategies(self._current_regime)
                        decision = self._brain.review_midday(
                            regime=self._current_regime,
                            assessments=assessments,
                            current_positions=self._executor.get_positions(),
                            daily_pnl=self._get_daily_pnl(),
                            vix_level=self._current_regime.vix_level,
                        )
                        self._apply_brain_decision(decision, context="midday")

                        # Handle explicit close recommendations from the brain.
                        for rejection in decision.rejected_trades:
                            if not rejection.reason.startswith("close_early:"):
                                continue
                            symbol = rejection.candidate.symbol
                            try:
                                closed = self._executor.close_position(symbol)
                                if closed:
                                    self._log.info(
                                        "brain_close_early_executed",
                                        symbol=symbol,
                                        reason=rejection.reason,
                                    )
                            except Exception:
                                self._log.exception("brain_close_early_failed", symbol=symbol)

                    self._last_brain_midday_review = datetime.now(
                        pytz.timezone("America/New_York"),
                    ).date()
            except Exception:
                self._log.exception("brain_cycle_decision_failed")

        # Classic selector path (legacy scorer/allocator/reviewer)
        elif self._reviewer:
            try:
                if self._reviewer.needs_review:
                    actions = self._reviewer.review(
                        strategies=self._strategies,
                        regime=self._current_regime,
                        vix_level=self._current_regime.vix_level if self._current_regime else None,
                    )
                    if actions:
                        for a in actions:
                            if a.action != "no_change":
                                self._log.info(
                                    "review_action",
                                    strategy=a.strategy_name,
                                    action=a.action,
                                    old_capital=a.old_capital,
                                    new_capital=a.new_capital,
                                    reason=a.reason,
                                )
            except Exception:
                self._log.exception("midday_review_failed")

        # Safety guard: near expiration close any orphan option legs from broker
        # positions to prevent exercise/assignment.
        try:
            self._check_expiry_risk()
        except Exception:
            self._log.exception("expiry_risk_check_failed")

        # Run risk checks
        strategy_statuses = [s.get_status() for s in self._strategies.values()]
        account = self._executor.get_account()
        risk_result = self._risk_manager.check_risk(account, strategy_statuses)

        if not risk_result.get("can_trade", True):
            self._log.warning("risk_blocked", status=risk_result["status"])
            return

        # Run each active strategy with current regime
        for name, strategy in self._strategies.items():
            if not strategy.is_enabled:
                continue

            try:
                signals = strategy.run_cycle(regime=self._current_regime)
                if signals:
                    self._log.info("signals_generated", strategy=name, count=len(signals))
            except Exception:
                self._log.exception("strategy_cycle_error", strategy=name)

        try:
            self._handle_market_close()
        except Exception:
            self._log.exception("market_close_handler_failed")

        # Evict expired cache entries periodically
        self._data_cache.evict_expired()

        # Write dashboard state every 5th cycle
        self._cycle_count += 1
        if self._cycle_count % 5 == 0:
            try:
                self._write_dashboard_state()
            except Exception:
                self._log.debug("dashboard_state_write_failed")

    def _maybe_refresh_premarket(self) -> None:
        """Check if we should refresh pre-market data during the wait loop.

        Only refreshes on trading days between 7:00-9:30 AM ET, throttled
        to every 15 minutes.
        """
        now = time.time()

        # Throttle: skip if refreshed recently
        if now - self._last_premarket_refresh < self._premarket_refresh_interval:
            # Periodic heartbeat so logs don't look dead
            self._cycle_count += 1
            if self._cycle_count % 12 == 0:  # ~1 hour at 5-min cycles
                self._log.info("waiting_for_market_open")
            return

        # Check if it's a trading day in pre-market window
        try:
            clock = self._data_provider.get_clock()
            et_tz = pytz.timezone("America/New_York")
            now_et = datetime.now(pytz.UTC).astimezone(et_tz)

            # next_open is today = it's a trading day and market hasn't opened yet
            next_open_et = clock.next_open.astimezone(et_tz)
            is_trading_day_premarket = (
                not clock.is_open
                and next_open_et.date() == now_et.date()
                and 7 <= now_et.hour < 10  # 7:00 AM - 9:59 AM ET window
            )

            if not is_trading_day_premarket:
                self._cycle_count += 1
                if self._cycle_count % 12 == 0:
                    self._log.info("market_closed_not_trading_day")
                return
        except Exception:
            self._log.exception("premarket_clock_check_failed")
            return

        # Do the refresh
        self._last_premarket_refresh = now
        self._refresh_premarket_intelligence(now_et)

    def _refresh_premarket_intelligence(self, now_et: datetime) -> None:
        """Refresh pre-market data: gaps, regime, news, and optionally volume."""
        if self._shutdown_requested:
            return

        self._log.info("premarket_refresh_starting", time_et=now_et.strftime("%H:%M"))

        # 1. Re-scan gaps (idempotent, cheap)
        gaps = []
        try:
            gaps = self._gap_scanner.scan()
            if gaps:
                self._log.info(
                    "premarket_gaps_refreshed",
                    count=len(gaps),
                    top_gap=f"{gaps[0].symbol} {gaps[0].gap_pct:+.1f}%",
                )
                # Pass gap candidates to gap_reversal strategy
                self._pass_gap_candidates(gaps)
        except Exception:
            self._log.exception("premarket_gap_refresh_failed")

        # 2. Re-detect regime
        try:
            self._current_regime = self._regime_detector.detect()
            self._log.info(
                "premarket_regime_refreshed",
                regime=self._current_regime.regime_type.value,
                confidence=self._current_regime.confidence,
            )
        except Exception:
            self._log.exception("premarket_regime_refresh_failed")

        # 3. Fetch news
        try:
            news = self._news_client.fetch_news(limit=30)
            self._log.info(
                "premarket_news_refreshed",
                total=len(news.items),
                bullish=news.bullish_count,
                bearish=news.bearish_count,
            )
        except Exception:
            self._log.exception("premarket_news_refresh_failed")

        # 4. Volume scanner warm-up once close to open (after 9:15 AM ET)
        if now_et.hour == 9 and now_et.minute >= 15 and not self._volume_refreshed_today:
            try:
                self._volume_scanner.warm_up()
                self._volume_refreshed_today = True
                self._log.info("premarket_volume_refreshed")
            except Exception:
                self._log.exception("premarket_volume_refresh_failed")

        # 5. Refresh selector decisions with fresh data
        if self._use_brain and self._brain and self._current_regime:
            try:
                assessments = self._assess_all_strategies(self._current_regime)
                decision = self._brain.decide(
                    regime=self._current_regime,
                    assessments=assessments,
                    current_positions=self._executor.get_positions(),
                    daily_pnl=self._get_daily_pnl(),
                    vix_level=self._current_regime.vix_level,
                )
                self._apply_brain_decision(decision, context="premarket_refresh")
            except Exception:
                self._log.exception("premarket_brain_refresh_failed")

        elif self._scorer and self._allocator and self._current_regime:
            try:
                strategy_names = list(self._strategies.keys())
                assessments = self._assess_all_strategies(self._current_regime)
                scores = self._scorer.score_strategies(
                    strategy_names,
                    self._current_regime,
                    vix_level=self._current_regime.vix_level,
                    assessments=assessments,
                )
                account = self._executor.get_account()
                allocations = self._allocator.allocate(scores, current_equity=account.equity)
                self._allocator.apply_allocations(allocations, self._strategies)
                self._log.info(
                    "premarket_allocations_refreshed",
                    allocations={
                        a.strategy_name: f"{a.allocation_pct:.1f}%"
                        for a in allocations if a.is_active
                    },
                )
                self._write_todays_plan(scores, allocations)
            except Exception:
                self._log.exception("premarket_allocation_refresh_failed")

        # 6. Write updated state for dashboard
        try:
            upcoming = self._event_calendar.get_upcoming_events(7) if hasattr(self, '_event_calendar') else []
            self._write_intelligence_state(gaps, [], upcoming)
            self._write_dashboard_state()
        except Exception:
            self._log.debug("premarket_state_write_failed")

        self._log.info("premarket_refresh_complete")

    def _run_pre_market_intelligence(self) -> None:
        """Run pre-market intelligence gathering.

        Detects regime, scans for gaps, checks event calendar, fetches news.
        Called once before the trading loop starts.
        """
        if self._shutdown_requested:
            return

        self._log.info("pre_market_intelligence_starting")

        # 1. Check event calendar
        try:
            today_events = self._event_calendar.get_events_for_date()
            if today_events:
                self._log.info(
                    "events_today",
                    count=len(today_events),
                    max_impact=self._event_calendar.max_impact_today(),
                )
                # Mark event day for regime detector
                if self._event_calendar.is_event_day():
                    import pytz as _pytz
                    today_str = datetime.now(_pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
                    self._regime_detector.mark_event_day(today_str)

            upcoming = self._event_calendar.get_upcoming_events(7)
            if upcoming:
                self._log.info("upcoming_events", count=len(upcoming))
        except Exception:
            self._log.exception("event_calendar_check_failed")

        # 2. Detect initial regime
        try:
            self._current_regime = self._regime_detector.detect()
            self._log.info(
                "initial_regime",
                regime=self._current_regime.regime_type.value,
                confidence=self._current_regime.confidence,
            )
        except Exception:
            self._log.exception("initial_regime_detection_failed")

        # 3. Scan for gaps
        gaps = []
        try:
            gaps = self._gap_scanner.scan()
            if gaps:
                self._log.info(
                    "pre_market_gaps",
                    count=len(gaps),
                    top_gap=f"{gaps[0].symbol} {gaps[0].gap_pct:+.1f}%" if gaps else "",
                )
                # Pass gap candidates to gap_reversal strategy
                self._pass_gap_candidates(gaps)
        except Exception:
            self._log.exception("gap_scan_failed")

        # 4. Warm up volume scanner with average volumes
        try:
            self._volume_scanner.warm_up()
        except Exception:
            self._log.exception("volume_scanner_warmup_failed")

        # 5. Fetch recent news
        try:
            news = self._news_client.fetch_news(limit=30)
            self._log.info(
                "pre_market_news",
                total=len(news.items),
                bullish=news.bullish_count,
                bearish=news.bearish_count,
            )
        except Exception:
            self._log.exception("news_fetch_failed")

        # 6. Initial selector decision for the day
        if self._use_brain and self._brain and self._current_regime:
            try:
                assessments = self._assess_all_strategies(self._current_regime)
                decision = self._brain.decide(
                    regime=self._current_regime,
                    assessments=assessments,
                    current_positions=self._executor.get_positions(),
                    daily_pnl=self._get_daily_pnl(),
                    vix_level=self._current_regime.vix_level,
                )
                self._apply_brain_decision(decision, context="pre_market")
            except Exception:
                self._log.exception("pre_market_brain_decision_failed")

        elif self._scorer and self._allocator and self._current_regime:
            try:
                strategy_names = list(self._strategies.keys())
                assessments = self._assess_all_strategies(self._current_regime)
                scores = self._scorer.score_strategies(
                    strategy_names,
                    self._current_regime,
                    vix_level=self._current_regime.vix_level,
                    assessments=assessments,
                )
                account = self._executor.get_account()
                allocations = self._allocator.allocate(scores, current_equity=account.equity)
                self._allocator.apply_allocations(allocations, self._strategies)

                self._log.info(
                    "pre_market_allocations",
                    allocations={
                        a.strategy_name: f"{a.allocation_pct:.1f}%"
                        for a in allocations if a.is_active
                    },
                )
                self._write_todays_plan(scores, allocations)
            except Exception:
                self._log.exception("pre_market_allocation_failed")
                # Fallback: keep static YAML allocations (already set in initialize())

        # 7. Write intelligence state for dashboard
        try:
            upcoming = self._event_calendar.get_upcoming_events(7) if hasattr(self, '_event_calendar') else []
            self._write_intelligence_state(gaps, [], upcoming)
        except Exception:
            self._log.debug("intelligence_state_write_failed")

        self._log.info("pre_market_intelligence_complete")

    def shutdown(self) -> None:
        """Graceful shutdown: save state, log summary."""
        if getattr(self, "_shutdown_completed", False):
            return
        self._shutdown_completed = True
        self._running = False
        self._shutdown_requested = True

        self._log.info("shutting_down")

        # Save strategy states
        for name, strategy in self._strategies.items():
            try:
                strategy.save_state()
                self._log.info("strategy_state_saved", name=name)
            except Exception:
                self._log.exception("strategy_state_save_failed", name=name)

        # Log daily summary
        try:
            summary = self._trade_journal.get_daily_summary()
            portfolio = self._portfolio_tracker.get_snapshot()
            self._log.info(
                "daily_summary",
                trades=summary["total_trades"],
                wins=summary["wins"],
                losses=summary["losses"],
                total_pnl=round(summary["total_pnl"], 2),
                equity=portfolio["equity"],
            )
        except Exception:
            self._log.exception("summary_generation_failed")

        # Log final strategy status
        try:
            for name, strategy in self._strategies.items():
                status = strategy.get_status()
                self._log.info(
                    "strategy_final_status",
                    name=name,
                    enabled=status.enabled,
                    capital=strategy._total_capital,
                    daily_pnl=round(status.daily_pnl, 2),
                    trades=status.daily_trades,
                    win_rate=f"{status.win_rate:.0%}",
                )
        except Exception:
            self._log.exception("strategy_status_log_failed")

        # Post-market analysis: attribution
        try:
            attribution = self._attribution.daily_report()
            self._log.info(
                "daily_attribution",
                total_pnl=attribution.total_pnl,
                by_strategy=attribution.by_strategy,
                by_session=attribution.by_session,
                regime_accuracy=attribution.regime_accuracy,
                return_on_deployed=f"{attribution.return_on_deployed:.2f}%",
            )
        except Exception:
            self._log.exception("attribution_failed")

        # Post-market analysis: weekly weight learning (Fridays)
        try:
            et_now = datetime.now(pytz.timezone("America/New_York"))
            if et_now.weekday() == 4:  # Friday
                adjustments = self._learner.auto_learn(days=30)
                if adjustments:
                    self._log.info(
                        "weight_adjustments_applied",
                        count=len(adjustments),
                        adjustments=[
                            {"regime": a.regime, "strategy": a.strategy,
                             "old": a.old_weight, "new": a.new_weight}
                            for a in adjustments
                        ],
                    )
                    self._alert_manager.send_alert(Alert(
                        alert_type=AlertType.WEIGHT_ADJUSTMENT,
                        level=AlertLevel.INFO,
                        title="Weekly Weight Update",
                        message=f"{len(adjustments)} regime weights adjusted",
                        timestamp=datetime.now(pytz.UTC),
                    ))
        except Exception:
            self._log.exception("weight_learning_failed")

        # Post-market analysis: daily summary alert
        try:
            summary = self._trade_journal.get_daily_summary()
            self._alert_manager.send_daily_summary(summary)
        except Exception:
            self._log.exception("daily_summary_alert_failed")

        # Write final dashboard state
        try:
            self._write_dashboard_state()
        except Exception:
            self._log.debug("final_dashboard_state_write_failed")

        # Clean up
        if self._ibkr_conn is not None:
            try:
                self._ibkr_conn.disconnect()
            except Exception:
                self._log.exception("ibkr_disconnect_failed")

        self._trade_journal.close()
        self._event_bus.clear()
        self._log.info("shutdown_complete")

    def _build_strategy_symbol_map(self) -> dict[str, list[str]]:
        """Build symbol -> [strategy_names] mapping from active strategies."""
        symbol_map: dict[str, list[str]] = {}
        for name, strategy in self._strategies.items():
            try:
                for symbol in strategy.get_held_symbols():
                    symbol_map.setdefault(symbol, []).append(name)
            except Exception:
                self._log.debug("strategy_symbol_map_failed", strategy=name)
        return symbol_map

    def _write_dashboard_state(self) -> None:
        """Write state files for the Streamlit dashboard to read."""
        state_dir = Path("data/state")
        state_dir.mkdir(parents=True, exist_ok=True)

        # Update strategy attribution before snapshot
        self._portfolio_tracker.set_strategy_symbol_map(
            self._build_strategy_symbol_map()
        )

        # Broker snapshot
        try:
            snapshot = self._portfolio_tracker.get_snapshot()
            with open(state_dir / "broker_snapshot.json", "w") as f:
                json.dump(snapshot, f, default=str)
        except Exception:
            pass

        # Regime
        if self._current_regime:
            try:
                with open(state_dir / "regime.json", "w") as f:
                    json.dump(self._current_regime.model_dump(), f, default=str)
            except Exception:
                pass

    def _get_daily_pnl(self) -> float:
        """Get current day P&L from live portfolio snapshot."""
        try:
            snapshot = self._portfolio_tracker.get_snapshot()
            return float(snapshot.get("daily_pnl", 0.0))
        except Exception:
            return 0.0

    def _needs_morning_position_review(self) -> bool:
        """Run one morning review shortly after regular-session open."""
        et_now = datetime.now(pytz.timezone("America/New_York"))
        today = et_now.date()
        if self._last_morning_position_review == today:
            return False
        open_minutes = et_now.hour * 60 + et_now.minute
        return (9 * 60 + 30) <= open_minutes <= (11 * 60)

    def _morning_position_review(self) -> None:
        """Review overnight positions after market open."""
        try:
            positions = self._executor.get_positions()
        except Exception:
            self._log.exception("morning_position_review_failed_fetch_positions")
            return

        reviewed = 0
        for pos in positions:
            qty = float(getattr(pos, "qty", 0.0) or 0.0)
            if qty == 0:
                continue
            symbol = str(getattr(pos, "symbol", "") or "")
            entry_price = float(getattr(pos, "avg_entry_price", 0.0) or 0.0)
            current_price = float(getattr(pos, "current_price", 0.0) or 0.0)
            unrealized_pnl = float(getattr(pos, "unrealized_pnl", 0.0) or 0.0)
            overnight_change_pct = 0.0
            if entry_price > 0:
                overnight_change_pct = (current_price - entry_price) / entry_price * 100.0

            self._log.info(
                "overnight_position_review",
                symbol=symbol,
                qty=qty,
                entry=round(entry_price, 4),
                current=round(current_price, 4),
                overnight_change_pct=round(overnight_change_pct, 2),
                unrealized_pnl=round(unrealized_pnl, 2),
            )
            reviewed += 1

        try:
            overnight = self._risk_manager.compute_overnight_risk(positions)
            self._log.info(
                "overnight_risk_summary",
                positions=reviewed,
                total_exposure=round(float(overnight.get("total_exposure", 0.0)), 2),
                overnight_gap_risk=round(float(overnight.get("overnight_gap_risk", 0.0)), 2),
                exposure_pct=round(float(overnight.get("exposure_pct", 0.0)), 2),
            )
        except Exception:
            self._log.debug("overnight_risk_summary_failed")

        self._last_morning_position_review = datetime.now(
            pytz.timezone("America/New_York"),
        ).date()

    def _is_overnight_exposure_limited(self) -> bool:
        """Return True when existing deployed capital breaches overnight limit."""
        if self._max_overnight_exposure_pct <= 0:
            return False

        try:
            account = self._executor.get_account()
            total_capital = float(getattr(account, "equity", 0.0) or 0.0)
            if total_capital <= 0:
                return False

            positions = self._executor.get_positions()
            deployed_capital = sum(abs(float(getattr(p, "market_value", 0.0) or 0.0)) for p in positions)
            deployed_pct = (deployed_capital / total_capital) * 100.0
            if deployed_pct > self._max_overnight_exposure_pct:
                self._log.info(
                    "overnight_exposure_limit",
                    deployed_pct=round(deployed_pct, 2),
                    max_pct=round(self._max_overnight_exposure_pct, 2),
                    deployed_capital=round(deployed_capital, 2),
                    total_capital=round(total_capital, 2),
                )
                return True
        except Exception:
            self._log.debug("overnight_exposure_check_failed")
        return False

    def _block_new_entries_for_exposure_limit(self, context: str) -> None:
        """Zero strategy entry capital while still allowing position management."""
        for strategy_name, strategy in self._strategies.items():
            strategy.set_capital(0.0)
            if hasattr(strategy, "set_brain_contract_cap"):
                try:
                    strategy.set_brain_contract_cap(None)
                except Exception:
                    self._log.debug(
                        "brain_contract_cap_clear_failed",
                        strategy=strategy_name,
                    )
        self._log.warning(
            "overnight_exposure_entry_blocked",
            context=context,
            max_overnight_exposure_pct=round(self._max_overnight_exposure_pct, 2),
        )

    def _handle_market_close(self) -> None:
        """Close only intraday positions near close; keep swing positions open."""
        et_now = datetime.now(pytz.timezone("America/New_York"))
        if et_now.hour < 15 or (et_now.hour == 15 and et_now.minute < 55) or et_now.hour >= 16:
            return

        today = et_now.date()
        if self._eod_handled_date == today:
            return
        self._eod_handled_date = today

        for strategy_name, strategy in self._strategies.items():
            intraday_only = bool(getattr(strategy.config, "params", {}).get("intraday_only", False))
            if strategy_name == "gap_reversal" or intraday_only:
                close_all = getattr(strategy, "close_all_positions", None)
                if callable(close_all):
                    closed_count = int(close_all(reason="eod_intraday_close"))
                    self._log.info(
                        "eod_intraday_positions_closed",
                        strategy=strategy_name,
                        closed=closed_count,
                    )
                continue

            close_for_eod = getattr(strategy, "close_positions_for_eod", None)
            if callable(close_for_eod):
                closed_count = int(close_for_eod(et_now=et_now))
                self._log.info(
                    "eod_positions_reviewed",
                    strategy=strategy_name,
                    closed=closed_count,
                )

        # Keep explicit expiry guard in place for expiring options.
        self._check_expiry_risk()

    def _morning_reconciliation(self) -> None:
        """Check startup broker positions against strategy-restored state."""
        try:
            expected_symbols: set[str] = set()
            for name, strategy in self._strategies.items():
                try:
                    held = strategy.get_held_symbols()
                    for symbol in held:
                        normalized = str(symbol or "").strip().upper()
                        if normalized:
                            expected_symbols.add(normalized)
                except Exception:
                    self._log.debug("morning_reconciliation_strategy_read_failed", strategy=name)

            broker_positions = self._executor.get_positions()
            if not broker_positions:
                self._log.info("morning_reconciliation", result="clean", positions=0)
                return

            non_zero_positions = [
                pos for pos in broker_positions if abs(float(getattr(pos, "qty", 0.0) or 0.0)) > 0
            ]
            if not non_zero_positions:
                self._log.info("morning_reconciliation", result="clean", positions=0)
                return

            unexpected_count = 0
            for pos in non_zero_positions:
                symbol = str(getattr(pos, "symbol", "") or "").upper()
                qty = float(getattr(pos, "qty", 0.0) or 0.0)
                side = getattr(getattr(pos, "side", None), "value", "")
                if symbol and symbol in expected_symbols:
                    continue

                unexpected_count += 1
                self._log.warning(
                    "unexpected_position_found",
                    symbol=symbol or "unknown",
                    quantity=qty,
                    side=side,
                    reason="Position exists in IBKR at startup; likely prior-session carry or exercise/assignment",
                )

            if unexpected_count > 0:
                self._log.warning(
                    "morning_reconciliation",
                    result="positions_found",
                    count=unexpected_count,
                    broker_positions=len(non_zero_positions),
                    expected_symbols=sorted(expected_symbols),
                    action="manual_review_recommended_close_unexpected_before_trading",
                )
            else:
                self._log.info(
                    "morning_reconciliation",
                    result="clean",
                    positions=len(non_zero_positions),
                    expected_symbols=sorted(expected_symbols),
                )
        except Exception as exc:
            self._log.error("morning_reconciliation_failed", error=str(exc))

    def _check_expiry_risk(self) -> None:
        """Close expiring option positions near the close to prevent exercise."""
        et_now = datetime.now(pytz.timezone("America/New_York"))
        if et_now.hour < 15 or (et_now.hour == 15 and et_now.minute < 45) or et_now.hour >= 16:
            return

        today = et_now.date()
        if self._expiry_guard_date != today:
            self._expiry_guard_date = today
            self._expiry_guard_attempted_conids.clear()

        get_option_positions = getattr(self._executor, "get_option_positions", None)
        close_option_position = getattr(self._executor, "close_option_position", None)
        if not callable(get_option_positions) or not callable(close_option_position):
            return

        option_positions = get_option_positions()
        for pos in option_positions:
            expiry = pos.get("expiry")
            con_id = int(pos.get("con_id", 0) or 0)
            qty = float(pos.get("qty", 0.0) or 0.0)
            if not isinstance(expiry, date) or expiry != today:
                continue
            if con_id <= 0 or qty == 0:
                continue
            if con_id in self._expiry_guard_attempted_conids:
                continue

            close_side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            self._log.warning(
                "closing_expiring_option",
                symbol=pos.get("symbol", ""),
                local_symbol=pos.get("local_symbol", ""),
                con_id=con_id,
                right=pos.get("right", ""),
                strike=pos.get("strike"),
                expiry=str(expiry),
                qty=abs(qty),
                side=close_side.value,
                reason="prevent_exercise",
            )
            close_order_id = close_option_position(
                con_id=con_id,
                qty=abs(qty),
                side=close_side,
            )
            if close_order_id:
                self._expiry_guard_attempted_conids.add(con_id)
                self._log.info(
                    "expiring_option_close_submitted",
                    order_id=close_order_id,
                    con_id=con_id,
                    local_symbol=pos.get("local_symbol", ""),
                )
            else:
                self._log.error(
                    "expiring_option_close_failed",
                    con_id=con_id,
                    local_symbol=pos.get("local_symbol", ""),
                )

    def _needs_brain_open_decision(self) -> bool:
        """Run one brain open decision shortly after market open each day."""
        et_now = datetime.now(pytz.timezone("America/New_York"))
        today = et_now.date()
        if self._last_brain_open_decision == today:
            return False

        open_minutes = et_now.hour * 60 + et_now.minute
        return open_minutes >= (9 * 60 + 30)

    def _needs_brain_midday_review(self) -> bool:
        """Run one brain midday review per day at configured review time."""
        et_now = datetime.now(pytz.timezone("America/New_York"))
        today = et_now.date()
        if self._last_brain_midday_review == today:
            return False

        review_hour = self._settings.strategy_selector.review_hour
        review_minute = self._settings.strategy_selector.review_minute
        current_minutes = et_now.hour * 60 + et_now.minute
        review_minutes = review_hour * 60 + review_minute
        return current_minutes >= review_minutes and et_now.hour < 16

    def _apply_brain_decision(self, decision: BrainDecision, context: str = "") -> None:
        """Apply a BrainDecision to strategy capital controls."""
        self._log.info(
            "brain_decision",
            context=context,
            num_selected=decision.num_trades,
            is_cash_day=decision.is_cash_day,
            cash_pct=decision.cash_pct,
            total_risk_pct=decision.total_risk_pct,
            reasoning=decision.reasoning,
        )

        # Enable capital only for selected strategies. Multiple selected candidates
        # from the same strategy share a summed capital budget.
        strategy_capital: dict[str, float] = {}
        strategy_contract_caps: dict[str, int] = {}
        for selection in decision.selected_trades:
            name = selection.candidate.strategy_name
            strategy_capital[name] = strategy_capital.get(name, 0.0) + selection.allocated_capital
            if selection.candidate.is_options:
                strategy_contract_caps[name] = max(
                    strategy_contract_caps.get(name, 0),
                    max(1, int(selection.position_size)),
                )

        for strategy_name, capital in strategy_capital.items():
            strategy = self._strategies.get(strategy_name)
            if strategy is None:
                continue
            strategy.set_capital(capital)
            if hasattr(strategy, "set_brain_contract_cap"):
                requested_contracts = strategy_contract_caps.get(strategy_name)
                try:
                    strategy.set_brain_contract_cap(requested_contracts)
                    self._log.info(
                        "brain_contract_cap_applied",
                        strategy=strategy_name,
                        brain_contracts=requested_contracts,
                        method="set_brain_contract_cap",
                    )
                except Exception:
                    self._log.exception(
                        "brain_contract_cap_apply_failed",
                        strategy=strategy_name,
                    )
            if not strategy.is_enabled:
                strategy.enable()

        active_strategies = set(strategy_capital.keys())
        for name, strategy in self._strategies.items():
            if name not in active_strategies:
                strategy.set_capital(0.0)  # Allow existing positions to keep being managed
                if hasattr(strategy, "set_brain_contract_cap"):
                    try:
                        strategy.set_brain_contract_cap(None)
                    except Exception:
                        self._log.exception(
                            "brain_contract_cap_clear_failed",
                            strategy=name,
                        )

        self._save_brain_decision(decision)

    def _save_brain_decision(self, decision: BrainDecision) -> None:
        """Save latest brain decision for dashboard and diagnostics."""
        state = decision.to_dict()
        path = Path("data/state/brain_decision.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, default=str))

    def _write_todays_plan(self, scores: list, allocations: list) -> None:
        """Write consolidated today's plan state for the dashboard."""
        state_dir = Path("data/state")
        state_dir.mkdir(parents=True, exist_ok=True)

        regime = self._current_regime
        plan = {
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "regime": regime.model_dump() if regime else None,
            "scores": [],
            "allocations": [],
        }

        # Score breakdown per strategy
        for s in scores:
            plan["scores"].append({
                "strategy": s.strategy_name,
                "total_score": round(s.total_score, 3),
                "base_weight": round(s.base_weight, 3),
                "opportunity_score": round(s.opportunity_score, 3),
                "vix_modifier": round(s.vix_modifier, 3),
                "performance_modifier": round(s.performance_modifier, 3),
                "time_modifier": round(s.time_modifier, 3),
                "event_modifier": round(s.event_modifier, 3),
                "is_active": s.is_active,
            })

        # Allocation breakdown per strategy
        for a in allocations:
            plan["allocations"].append({
                "strategy": a.strategy_name,
                "score": round(a.score, 3),
                "allocated_capital": round(a.allocated_capital, 2),
                "allocation_pct": round(a.allocation_pct, 2),
                "is_active": a.is_active,
                "reason": a.reason,
            })

        try:
            with open(state_dir / "todays_plan.json", "w") as f:
                json.dump(plan, f, default=str)
        except Exception:
            self._log.debug("todays_plan_write_failed")

    def _write_intelligence_state(self, gaps, volume_results, events) -> None:
        """Write scanner/intelligence results for dashboard."""
        state_dir = Path("data/state")
        state_dir.mkdir(parents=True, exist_ok=True)

        if gaps:
            try:
                with open(state_dir / "gaps.json", "w") as f:
                    json.dump(
                        [{"symbol": g.symbol, "gap_pct": g.gap_pct,
                          "direction": g.direction, "volume": g.pre_market_volume}
                         for g in gaps], f, default=str,
                    )
            except Exception:
                pass

        if volume_results:
            try:
                with open(state_dir / "unusual_volume.json", "w") as f:
                    json.dump(
                        [{"symbol": v.symbol, "ratio": v.volume_ratio,
                          "price_change": v.price_change_pct}
                         for v in volume_results], f, default=str,
                    )
            except Exception:
                pass

        if events:
            try:
                with open(state_dir / "upcoming_events.json", "w") as f:
                    json.dump(
                        [{"date": str(e.date), "type": e.event_type.value,
                          "description": e.description, "impact": e.impact}
                         for e in events], f, default=str,
                    )
            except Exception:
                pass

    def _assess_all_strategies(self, regime) -> dict:
        """Assess opportunities for all strategies, log results, and save to state file."""
        from algotrader.strategies.base import OpportunityAssessment
        assessments: dict[str, OpportunityAssessment] = {}
        for name, strategy in self._strategies.items():
            try:
                assessments[name] = strategy.assess_opportunities(regime)
            except Exception:
                self._log.error("assess_opportunities_failed", strategy=name, exc_info=True)
                assessments[name] = OpportunityAssessment()

        for name, assessment in assessments.items():
            self._log.info(
                "strategy_assessment",
                strategy=name,
                num_candidates=len(assessment.candidates),
                has_opportunities=assessment.has_opportunities,
            )

        # Log summary
        with_opps = {n: a for n, a in assessments.items() if a.has_opportunities}
        self._log.info(
            "opportunity_assessments",
            total=len(assessments),
            with_opportunities=len(with_opps),
            details={
                n: {
                    "aggregate_candidates": a.num_candidates,
                    "explicit_candidates": len(a.candidates),
                    "confidence": a.confidence,
                    "rr": a.avg_risk_reward,
                }
                for n, a in with_opps.items()
            },
        )

        # Save to state file for dashboard
        try:
            state_dir = Path("data/state")
            state_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for name, a in assessments.items():
                rows.append({
                    "strategy": name,
                    "has_opportunities": a.has_opportunities,
                    "num_candidates": a.num_candidates,
                    "avg_risk_reward": round(a.avg_risk_reward, 2),
                    "confidence": round(a.confidence, 2),
                    "estimated_daily_trades": a.estimated_daily_trades,
                    "estimated_edge_pct": round(a.estimated_edge_pct, 2),
                    "candidates_count": len(a.candidates),
                    "top_candidate_ev": round(a.top_candidate.expected_value, 4) if a.top_candidate else 0.0,
                    "details": a.details,
                })
            with open(state_dir / "assessments.json", "w") as f:
                json.dump(rows, f, default=str)
        except Exception:
            self._log.debug("assessment_state_write_failed")

        return assessments

    def _pass_gap_candidates(self, gaps: list) -> None:
        """Pass gap scanner results to the gap_reversal strategy."""
        gap_strategy = self._strategies.get("gap_reversal")
        if gap_strategy and hasattr(gap_strategy, "set_gap_candidates"):
            candidates = [
                {
                    "symbol": g.symbol,
                    "gap_pct": g.gap_pct,
                    "prev_close": g.prev_close,
                    "current_price": g.current_price,
                    "direction": g.direction,
                }
                for g in gaps
            ]
            gap_strategy.set_gap_candidates(candidates)

    def _on_kill_switch(self, reason: str = "") -> None:
        """Handle kill switch event."""
        self._log.error("kill_switch_received", reason=reason)
        # Don't stop the loop — risk manager already closed positions
        # Keep running to prevent re-entry but log the situation

    def _on_strategy_disabled(self, strategy_name: str = "", reason: str = "") -> None:
        """Handle strategy disabled event."""
        strategy = self._strategies.get(strategy_name)
        if strategy:
            strategy.disable(reason)
            self._log.warning("strategy_disabled_by_risk", name=strategy_name, reason=reason)
