"""Main lifecycle controller for the algotrader system.

Lifecycle: initialize → warm up strategies → run loop
(check market open → run active strategies → manage risk) → shutdown.
"""

from __future__ import annotations

import json
import signal as os_signal
import time
from datetime import datetime
from pathlib import Path

import pytz
import structlog

from algotrader.core.config import Settings, StrategyConfig, load_strategy_config, load_yaml
from algotrader.core.events import EventBus, KILL_SWITCH, STRATEGY_DISABLED
from algotrader.core.logging import setup_logging
from algotrader.core.models import MarketRegime, RegimeType
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

        # Data provider
        self._data_provider = AlpacaDataProvider(
            config=settings.alpaca,
            feed=settings.data.feed,
        )

        # Executor
        self._executor = AlpacaExecutor(config=settings.alpaca)

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
        self._regime_detector = RegimeDetector(
            data_provider=self._data_provider,
            event_bus=self._event_bus,
        )
        self._gap_scanner = GapScanner(data_provider=self._data_provider)
        self._volume_scanner = VolumeScanner(data_provider=self._data_provider)
        self._news_client = AlpacaNewsClient(data_provider=self._data_provider)
        self._event_calendar = EventCalendar()
        self._current_regime: MarketRegime | None = None

        # Strategy selector (Phase 4)
        self._scorer: StrategyScorer | None = None
        self._allocator: CapitalAllocator | None = None
        self._reviewer: MidDayReviewer | None = None
        if settings.strategy_selector.enabled:
            try:
                regime_config = load_yaml(settings.strategy_selector.regime_config)
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
                self._log.info("strategy_selector_initialized")
            except Exception:
                self._log.exception("strategy_selector_init_failed")
                self._scorer = None
                self._allocator = None
                self._reviewer = None

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
        self._log.info("shutdown_signal_received", signal=signum)
        self._running = False

    def initialize(self) -> None:
        """Initialize strategies from config and registry."""
        self._log.info("initializing")

        # Import strategy modules to trigger registration
        self._import_strategies()

        # Create strategies from config
        for name in registry.names:
            config = self._settings.strategies.get(name)
            if config is None:
                config = load_strategy_config(name)

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
                # Set capital allocation
                capital = self._settings.trading.total_capital * (config.capital_allocation_pct / 100)
                strategy.set_capital(capital)
                self._strategies[name] = strategy
                self._log.info("strategy_initialized", name=name, capital=capital)

        self._log.info("initialization_complete", strategies=list(self._strategies.keys()))

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
        self.warm_up()

        # Pre-market intelligence
        self._run_pre_market_intelligence()

        self._running = True
        cycle_interval = self._settings.data.cycle_interval_seconds
        self._log.info("trading_loop_starting", cycle_seconds=cycle_interval)

        # Reset portfolio tracker for today
        account = self._executor.get_account()
        self._portfolio_tracker.reset_day(account.equity)
        self._risk_manager.reset_day(account.equity)

        while self._running:
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
        # Check if market is open
        if not self._data_provider.is_market_open():
            self._log.debug("market_closed")
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

        # Mid-day review (strategy selector)
        if self._reviewer:
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

        # Evict expired cache entries periodically
        self._data_cache.evict_expired()

        # Write dashboard state every 5th cycle
        self._cycle_count += 1
        if self._cycle_count % 5 == 0:
            try:
                self._write_dashboard_state()
            except Exception:
                self._log.debug("dashboard_state_write_failed")

    def _run_pre_market_intelligence(self) -> None:
        """Run pre-market intelligence gathering.

        Detects regime, scans for gaps, checks event calendar, fetches news.
        Called once before the trading loop starts.
        """
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

        # 6. Score strategies and allocate capital based on initial regime
        if self._scorer and self._allocator and self._current_regime:
            try:
                strategy_names = list(self._strategies.keys())
                scores = self._scorer.score_strategies(
                    strategy_names,
                    self._current_regime,
                    vix_level=self._current_regime.vix_level,
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
        self._trade_journal.close()
        self._event_bus.clear()
        self._log.info("shutdown_complete")

    def _write_dashboard_state(self) -> None:
        """Write state files for the Streamlit dashboard to read."""
        state_dir = Path("data/state")
        state_dir.mkdir(parents=True, exist_ok=True)

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
