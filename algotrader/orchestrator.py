"""Main lifecycle controller for the algotrader system.

Lifecycle: initialize → warm up strategies → run loop
(check market open → run active strategies → manage risk) → shutdown.
"""

from __future__ import annotations

import signal as os_signal
import time
from datetime import datetime

import pytz
import structlog

from algotrader.core.config import Settings, StrategyConfig, load_strategy_config
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
from algotrader.tracking.journal import TradeJournal
from algotrader.tracking.portfolio import PortfolioTracker

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

        # Clean up
        self._trade_journal.close()
        self._event_bus.clear()
        self._log.info("shutdown_complete")

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
