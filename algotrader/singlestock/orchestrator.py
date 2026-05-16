"""SingleStockOrchestrator — top-level loop for the single-stock tool.

Lifecycle (ET):
- 08:00  premarket_investigate(): run the 5 agents → TradeThesis →
         persist. Cash-default if conviction < threshold or blackout.
- 09:35  open_position(): if thesis approved AND no position open,
         pick option, submit limit at NBBO mid, record entry.
- intraday every cfg.intraday_check_minutes: check premium stop/target,
         underlying stop, max-hold-days. News-delta re-check every
         cfg.news_recheck_minutes.
- 15:50  eod_review(): close if trend broken or days-held cap, else
         carry overnight.
- expiry day 15:30: mandatory close.

Runs in its own process with its own IBKR client_id and lockfile. Logs
to data/logs/singlestock.log (separate channel from main bot).
"""

from __future__ import annotations

import signal
import time
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any

import pytz
import structlog

from algotrader.core.config import IBKRConfig, Settings
from algotrader.core.events import EventBus
from algotrader.core.logging import setup_logging
from algotrader.data.ibkr_provider import IBKRDataProvider
from algotrader.execution.ibkr_connection import IBKRConnection
from algotrader.intelligence.calendar.events import EventCalendar
from algotrader.intelligence.news.alpaca_news import NewsClient
from algotrader.intelligence.regime import RegimeDetector
from algotrader.singlestock.agents.announcements_agent import AnnouncementsAgent
from algotrader.singlestock.agents.decision_agent import DecisionAgent
from algotrader.singlestock.agents.market_context_agent import MarketContextAgent
from algotrader.singlestock.agents.news_agent import NewsAgent
from algotrader.singlestock.agents.technical_agent import TechnicalAgent
from algotrader.singlestock.feeds.rss_news import RSSNewsClient
from algotrader.singlestock.feeds.sec_edgar import SECEdgarClient
from algotrader.singlestock.investigator import Investigator
from algotrader.singlestock.llm_client import LLMClient
from algotrader.singlestock.option_executor import OptionExecutor
from algotrader.singlestock.options_picker import OptionsPicker
from algotrader.singlestock.pdt_guard import PDTGuard
from algotrader.singlestock.position_manager import CloseReason, PositionManager
from algotrader.singlestock.state import SingleStockState
from algotrader.singlestock.thesis import Direction, TradeThesis

ET = pytz.timezone("America/New_York")

# Bind base logger once setup_logging has run.
_base_logger = structlog.get_logger()


def _parse_hhmm(s: str) -> dtime:
    hh, mm = s.split(":", 1)
    return dtime(hour=int(hh), minute=int(mm))


class SingleStockOrchestrator:
    """Coordinates premarket investigation, entry, intraday management,
    and EOD review for the single-stock tool.
    """

    def __init__(self, settings: Settings, dry_run: bool = False) -> None:
        self._settings = settings
        self._cfg = settings.singlestock
        self._dry_run = dry_run

        # Logging must be configured BEFORE the first logger.bind() call.
        # See CLAUDE.md 2026-05-13 fix: structlog's cache_logger_on_first_use
        # captures a no-handler config if bind happens before setup.
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        setup_logging(
            level=settings.logging.level,
            log_file=self._cfg.log_file,
            json_format=settings.logging.json_format,
        )
        self._log = _base_logger.bind(
            component="singlestock_orchestrator",
            symbol=self._cfg.symbol,
        )
        self._log.info(
            "singlestock_orchestrator_init",
            dry_run=dry_run,
            client_id=self._cfg.ibkr_client_id,
            capital_pct=self._cfg.capital_pct,
        )

        self._running = False
        self._shutdown_requested = False

        # IBKR connection with our own client_id.
        broker_cfg = settings.broker.ibkr
        ibkr_cfg = IBKRConfig(
            host=broker_cfg.host,
            port=broker_cfg.port,
            client_id=self._cfg.ibkr_client_id,
            timeout=broker_cfg.timeout,
            readonly=broker_cfg.readonly,
            account=broker_cfg.account,
            max_reconnect_attempts=broker_cfg.max_reconnect_attempts,
            reconnect_delay_seconds=broker_cfg.reconnect_delay_seconds,
        )
        self._event_bus = EventBus()
        self._ibkr_conn = IBKRConnection.get_instance(
            config=ibkr_cfg,
            event_bus=self._event_bus,
        )
        if not self._ibkr_conn.connect():
            raise RuntimeError("singlestock: failed to connect to IBKR TWS/Gateway")

        self._data_provider = IBKRDataProvider(connection=self._ibkr_conn)
        self._option_executor = OptionExecutor(connection=self._ibkr_conn)

        # Capital slice from current NetLiquidation.
        account = self._fetch_account_summary()
        net_liq = account.get("NetLiquidation", 0.0)
        self._slice_capital = max(0.0, net_liq * (self._cfg.capital_pct / 100.0))
        self._daily_kill_dollars = self._slice_capital * (self._cfg.daily_loss_kill_pct / 100.0)
        self._log.info(
            "singlestock_capital_slice",
            net_liq=net_liq,
            slice_capital=self._slice_capital,
            daily_kill_dollars=self._daily_kill_dollars,
        )

        # State
        self._state = SingleStockState(path=self._cfg.state_file)

        # Intelligence stack
        self._event_calendar = EventCalendar()
        self._regime_detector = RegimeDetector(
            data_provider=self._data_provider,
            event_bus=self._event_bus,
            event_calendar=self._event_calendar,
        )

        # Agents
        self._llm = LLMClient(timeout_seconds=self._cfg.llm_timeout_seconds)
        if self._cfg.llm_enabled and not self._llm.available:
            self._log.warning("singlestock_llm_unavailable_at_init")

        self._news_agent = NewsAgent(
            rss_client=RSSNewsClient(),
            llm_client=self._llm,
            ibkr_news_client=NewsClient(data_provider=self._data_provider),
            model=self._cfg.llm_model_news,
        )
        self._announcements_agent = AnnouncementsAgent(
            edgar_client=SECEdgarClient(),
            llm_client=self._llm,
            model=self._cfg.llm_model_announcements,
        )
        self._market_agent = MarketContextAgent(
            data_provider=self._data_provider,
            regime_detector=self._regime_detector,
        )
        self._technical_agent = TechnicalAgent(data_provider=self._data_provider)
        self._decision_agent = DecisionAgent(
            llm_client=self._llm,
            model=self._cfg.llm_model_decision,
            min_conviction=self._cfg.min_conviction,
            earnings_blackout_days=self._cfg.earnings_blackout_days,
        )

        self._investigator = Investigator(
            symbol=self._cfg.symbol,
            market_context_agent=self._market_agent,
            technical_agent=self._technical_agent,
            news_agent=self._news_agent,
            announcements_agent=self._announcements_agent,
            decision_agent=self._decision_agent,
        )

        self._options_picker = OptionsPicker(
            data_provider=self._data_provider,
            target_delta=self._cfg.target_delta,
            min_dte=self._cfg.min_dte,
            max_dte=self._cfg.max_dte,
            max_contracts_per_trade=self._cfg.max_contracts_per_trade,
            max_position_premium_pct=self._cfg.max_position_premium_pct,
            exercise_exposure_cap_pct=self._cfg.exercise_exposure_cap_pct,
        )

        self._pdt_guard = PDTGuard(enabled=self._cfg.pdt_safe_mode)
        self._position_manager = PositionManager(
            symbol=self._cfg.symbol,
            data_provider=self._data_provider,
            option_executor=self._option_executor,
            state=self._state,
            pdt_guard=self._pdt_guard,
            max_hold_days=self._cfg.max_hold_days,
            premium_loss_close_pct=self._cfg.premium_loss_close_pct,
            premium_gain_target_pct=self._cfg.premium_gain_target_pct,
            enable_trailing_stop=self._cfg.enable_trailing_stop,
            expiry_day_close_time_et=self._cfg.expiry_day_close_time,
        )

        # Schedule markers (one-fire-per-day flags reset at midnight ET)
        self._fired_today: set[str] = set()
        self._last_schedule_date: str = ""
        self._last_intraday_check: datetime | None = None
        self._last_news_recheck: datetime | None = None

        self._setup_signal_handlers()

    # ── Public ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        self._log.info("singlestock_run_starting", dry_run=self._dry_run)
        try:
            while self._running and not self._shutdown_requested:
                self._tick()
                time.sleep(15)
        except KeyboardInterrupt:
            self._log.info("singlestock_keyboard_interrupt")
        finally:
            self._running = False
            self._log.info("singlestock_run_stopped")

    # ── Scheduling tick ────────────────────────────────────────────────────

    def _tick(self) -> None:
        now = datetime.now(ET)
        today_str = now.strftime("%Y-%m-%d")
        if today_str != self._last_schedule_date:
            self._fired_today.clear()
            self._last_schedule_date = today_str

        # Daily kill switch — halts new entries (does not auto-close)
        if self._state.counters.realized_pnl_dollars <= -self._daily_kill_dollars:
            if "daily_kill_logged" not in self._fired_today:
                self._log.warning(
                    "singlestock_daily_kill_switch_active",
                    realized=self._state.counters.realized_pnl_dollars,
                    threshold=-self._daily_kill_dollars,
                )
                self._fired_today.add("daily_kill_logged")

        # 1. Pre-market investigation
        if (
            "premarket" not in self._fired_today
            and now.time() >= _parse_hhmm(self._cfg.premarket_investigate_time)
            and now.time() < _parse_hhmm(self._cfg.entry_time)
        ):
            self._fire_premarket()
            self._fired_today.add("premarket")

        # 2. Entry at open (if approved and no position)
        if (
            "entry" not in self._fired_today
            and now.time() >= _parse_hhmm(self._cfg.entry_time)
            and now.time() < _parse_hhmm("15:30")
        ):
            self._fire_entry()
            self._fired_today.add("entry")

        # 3. Intraday checks
        if (
            self._state.open_position is not None
            and now.time() >= _parse_hhmm(self._cfg.entry_time)
            and now.time() < _parse_hhmm(self._cfg.eod_review_time)
            and self._should_run_intraday(now)
        ):
            self._fire_intraday(now)

        # 4. EOD review
        if (
            "eod" not in self._fired_today
            and now.time() >= _parse_hhmm(self._cfg.eod_review_time)
            and now.time() < dtime(16, 0)
        ):
            self._fire_eod()
            self._fired_today.add("eod")

        # 5. Expiry-day close (could fire on a non-eod day)
        if self._state.open_position is not None:
            self._position_manager.expiry_day_check()

    # ── Action handlers ────────────────────────────────────────────────────

    def _fire_premarket(self) -> None:
        self._log.info("singlestock_premarket_starting")
        try:
            result = self._investigator.investigate()
        except Exception:
            self._log.exception("singlestock_premarket_failed")
            return

        self._state.update_thesis(result.thesis.to_dict())
        self._state.update_news_baseline(result.news_article_ids)
        self._state.increment_llm_calls(n=self._estimate_llm_calls_for_investigation())

    def _fire_entry(self) -> None:
        thesis_json = self._state.thesis_json
        if thesis_json is None:
            self._log.info("singlestock_entry_no_thesis")
            return
        if thesis_json.get("blackout_reason"):
            self._log.info("singlestock_entry_blackout", reason=thesis_json["blackout_reason"])
            return
        direction = thesis_json.get("direction")
        if direction not in ("long", "short"):
            self._log.info("singlestock_entry_direction_none")
            return
        if float(thesis_json.get("conviction", 0.0)) < self._cfg.min_conviction:
            self._log.info(
                "singlestock_entry_below_threshold",
                conviction=thesis_json.get("conviction"),
                threshold=self._cfg.min_conviction,
            )
            return
        if self._state.open_position is not None:
            self._log.info("singlestock_entry_already_holding")
            return
        if self._state.counters.realized_pnl_dollars <= -self._daily_kill_dollars:
            self._log.warning("singlestock_entry_blocked_kill_switch")
            return

        d = Direction.LONG if direction == "long" else Direction.SHORT
        picked = self._options_picker.pick(
            symbol=self._cfg.symbol,
            direction=d,
            slice_capital=self._slice_capital,
        )
        if picked is None:
            self._log.warning("singlestock_entry_no_contract")
            return

        if self._dry_run:
            self._log.info(
                "singlestock_entry_dry_run",
                picked={
                    "local_symbol": picked.symbol,
                    "right": picked.right,
                    "strike": picked.strike,
                    "expiry": str(picked.expiry),
                    "mid": picked.mid,
                    "contracts": picked.contracts,
                    "estimated_cost": picked.estimated_cost,
                },
            )
            return

        # Build minimal thesis object for position_manager
        thesis = self._rehydrate_thesis(thesis_json, direction=d)
        position = self._position_manager.open_position(thesis=thesis, picked=picked)
        if position is None:
            self._log.warning("singlestock_entry_open_failed")

    def _fire_intraday(self, now: datetime) -> None:
        self._last_intraday_check = now
        try:
            result = self._position_manager.check_intraday()
            self._log.info(
                "singlestock_intraday_check",
                closed=result.closed,
                reason=result.reason.value if result.reason else None,
                realized=result.realized_pnl,
                summary=result.summary,
            )
        except Exception:
            self._log.exception("singlestock_intraday_check_failed")

        # News-delta recheck on its own cadence
        if self._should_recheck_news(now):
            self._last_news_recheck = now
            self._news_delta_check()

    def _fire_eod(self) -> None:
        if self._state.open_position is None:
            self._log.info("singlestock_eod_no_position")
            return
        trend_intact = self._eod_trend_intact()
        try:
            result = self._position_manager.eod_review(trend_intact=trend_intact)
            self._log.info(
                "singlestock_eod_review",
                trend_intact=trend_intact,
                closed=result.closed,
                reason=result.reason.value if result.reason else None,
                summary=result.summary,
            )
        except Exception:
            self._log.exception("singlestock_eod_review_failed")

    # ── News delta ─────────────────────────────────────────────────────────

    def _news_delta_check(self) -> None:
        if self._state.open_position is None:
            return
        baseline = set(self._state.news_baseline_ids)
        try:
            articles = self._news_agent.fetch_articles(self._cfg.symbol, max_age_hours=12.0)
        except Exception:
            self._log.exception("singlestock_news_delta_fetch_failed")
            return

        new_articles = [a for a in articles if a.article_id not in baseline]
        if not new_articles:
            return

        self._log.info(
            "singlestock_news_delta_detected",
            new_article_count=len(new_articles),
        )

        # Re-synthesize news only (cheaper than full investigation).
        tech = self._technical_agent.compute(self._cfg.symbol)
        current_price = tech.current_price if tech else 0.0
        new_thesis = self._news_agent.synthesize(self._cfg.symbol, current_price, articles)
        self._state.increment_llm_calls(n=1)

        # If the news has flipped opposite to our position, close it.
        pos_direction = self._state.open_position.direction
        if pos_direction == "long" and new_thesis.direction == Direction.SHORT and new_thesis.confidence >= 0.5:
            self._log.warning("singlestock_news_reversal_long_to_short")
            self._position_manager.force_close(CloseReason.NEWS_REVERSAL)
        elif pos_direction == "short" and new_thesis.direction == Direction.LONG and new_thesis.confidence >= 0.5:
            self._log.warning("singlestock_news_reversal_short_to_long")
            self._position_manager.force_close(CloseReason.NEWS_REVERSAL)
        else:
            # Update baseline so we don't re-process these articles.
            self._state.update_news_baseline([a.article_id for a in articles])

    # ── Helpers ────────────────────────────────────────────────────────────

    def _should_run_intraday(self, now: datetime) -> bool:
        if self._last_intraday_check is None:
            return True
        elapsed = (now - self._last_intraday_check).total_seconds() / 60.0
        return elapsed >= self._cfg.intraday_check_minutes

    def _should_recheck_news(self, now: datetime) -> bool:
        if self._last_news_recheck is None:
            return True
        elapsed = (now - self._last_news_recheck).total_seconds() / 60.0
        return elapsed >= self._cfg.news_recheck_minutes

    def _eod_trend_intact(self) -> bool:
        pos = self._state.open_position
        if pos is None:
            return False
        tech = self._technical_agent.compute(self._cfg.symbol)
        if tech is None:
            return False
        if pos.direction == "long":
            return tech.current_price >= tech.vwap
        return tech.current_price <= tech.vwap

    def _fetch_account_summary(self) -> dict[str, float]:
        try:
            ib = self._ibkr_conn.ib
            summary = self._ibkr_conn.execute(lambda b: b.accountSummary(""))
        except Exception:
            self._log.exception("singlestock_account_summary_failed")
            return {}
        out: dict[str, float] = {}
        for row in summary or []:
            tag = str(getattr(row, "tag", "") or "")
            try:
                out[tag] = float(getattr(row, "value", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        return out

    def _estimate_llm_calls_for_investigation(self) -> int:
        if not self._cfg.llm_enabled:
            return 0
        # news + announcements + decision
        return 3

    def _rehydrate_thesis(self, thesis_json: dict[str, Any], direction: Direction) -> TradeThesis:
        from algotrader.singlestock.thesis import (
            AnnouncementsThesis,
            MarketContext,
            NewsThesis,
            TechnicalContext,
        )

        def _dir(s: Any) -> Direction:
            v = str(s or "none").lower()
            return Direction.LONG if v == "long" else Direction.SHORT if v == "short" else Direction.NONE

        tech_d = thesis_json.get("technicals") or {}
        tech = (
            TechnicalContext(
                direction=_dir(tech_d.get("direction")),
                vwap=float(tech_d.get("vwap", 0.0) or 0.0),
                rsi_14=float(tech_d.get("rsi_14", 50.0) or 50.0),
                atr_14=float(tech_d.get("atr_14", 0.0) or 0.0),
                breakout_level_up=float(tech_d.get("breakout_level_up", 0.0) or 0.0),
                breakdown_level_down=float(tech_d.get("breakdown_level_down", 0.0) or 0.0),
                current_price=float(tech_d.get("current_price", 0.0) or 0.0),
                gap_pct=float(tech_d.get("gap_pct", 0.0) or 0.0),
            )
            if tech_d
            else None
        )

        return TradeThesis(
            symbol=thesis_json.get("symbol", self._cfg.symbol),
            direction=direction,
            conviction=float(thesis_json.get("conviction", 0.0) or 0.0),
            timestamp=datetime.now(ET),
            entry_zone=tuple(thesis_json["entry_zone"]) if thesis_json.get("entry_zone") else None,
            stop_price=thesis_json.get("stop_price"),
            target_price=thesis_json.get("target_price"),
            rationale=str(thesis_json.get("rationale", "") or ""),
            technicals=tech,
        )

    # ── Signals ────────────────────────────────────────────────────────────

    def _setup_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            self._log.warning("singlestock_signal_received", signal=signum)
            self._shutdown_requested = True

        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except (ValueError, AttributeError):
            # SIGTERM may not be installable on Windows.
            pass
