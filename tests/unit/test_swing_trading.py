from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytz

from algotrader.core.config import Settings, StrategyConfig
from algotrader.core.events import EventBus
from algotrader.core.models import AccountInfo, MarketRegime, RegimeType
from algotrader.orchestrator import Orchestrator
from algotrader.strategies.momentum import MomentumStrategy, MomentumTrade
from algotrader.strategies.options_premium import OptionsPremiumStrategy, PremiumTrade
from algotrader.strategies.vwap_reversion import VWAPReversionStrategy

from .test_strategy_candidates import StubDataProvider, StubExecutor, _bars_from_closes


def _regime(regime_type: RegimeType, vix: float = 18.0) -> MarketRegime:
    return MarketRegime(
        regime_type=regime_type,
        vix_level=vix,
        confidence=0.9,
        timestamp=datetime.now(pytz.UTC),
    )


class _DummyLog:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class _NearestExpiryOnlyProvider(StubDataProvider):
    """Mimic providers that return only the nearest expiry for expiration=None."""

    def get_option_chain(self, underlying, expiration=None):
        chain = self.option_chains.get(underlying)
        if chain is None or chain.empty:
            return pd.DataFrame()

        normalized_dates = pd.to_datetime(chain["expiration"], errors="coerce").dt.date
        if expiration is None:
            nearest = min(d for d in normalized_dates.dropna().unique())
            return chain[normalized_dates == nearest].copy()
        return chain[normalized_dates == expiration].copy()


def test_options_selects_2_5_dte_expiry() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    today = date.today()
    provider.set_option_chain(
        "SPY",
        pd.DataFrame(
            [
                {"expiration": (today + timedelta(days=1)).isoformat(), "strike": 500, "type": "put"},
                {"expiration": (today + timedelta(days=3)).isoformat(), "strike": 500, "type": "put"},
                {"expiration": (today + timedelta(days=7)).isoformat(), "strike": 500, "type": "put"},
            ]
        ),
    )

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={"min_dte": 2, "max_dte": 5}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    expiry = strategy._find_expiry("SPY", min_dte=2, max_dte=5)
    assert expiry == today + timedelta(days=3)


def test_options_find_expiry_probes_window_when_default_chain_is_nearest_only() -> None:
    provider = _NearestExpiryOnlyProvider()
    executor = StubExecutor()
    today = date.today()
    target_expiry = today + timedelta(days=2)
    while target_expiry.weekday() >= 5:
        target_expiry += timedelta(days=1)

    provider.set_option_chain(
        "SPY",
        pd.DataFrame(
            [
                {"expiration": today, "strike": 500, "type": "put"},
                {"expiration": target_expiry, "strike": 500, "type": "put"},
            ]
        ),
    )

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={"min_dte": 2, "max_dte": 5}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )

    expiry = strategy._find_expiry("SPY", min_dte=2, max_dte=5)
    assert expiry == target_expiry


def test_options_find_expiry_falls_back_to_nearest_when_window_unavailable() -> None:
    provider = _NearestExpiryOnlyProvider()
    executor = StubExecutor()
    today = date.today()
    provider.set_option_chain(
        "SPY",
        pd.DataFrame(
            [
                {"expiration": today, "strike": 500, "type": "put"},
            ]
        ),
    )

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={"min_dte": 2, "max_dte": 5}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )

    expiry = strategy._find_expiry("SPY", min_dte=2, max_dte=5)
    assert expiry == today


def test_options_closes_1_day_before_expiry() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("SPY", 490.0)
    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={"close_before_expiry_days": 1, "stop_loss_multiple": 3.0}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    expiry = date.today() + timedelta(days=1)
    while expiry.weekday() >= 5:
        expiry += timedelta(days=1)

    strategy._trades["SPY"] = PremiumTrade(
        underlying="SPY",
        structure="put_spread",
        short_strike=490.0,
        long_strike=485.0,
        contracts=1,
        credit_received=100.0,
        max_profit=100.0,
        max_loss=400.0,
        entry_time=datetime.now(pytz.UTC) - timedelta(days=1),
        expiration=expiry,
        simulated=True,
        capital_used=400.0,
        trade_id="swing_test",
    )

    et_now = datetime.now(pytz.timezone("America/New_York")).replace(
        hour=11,
        minute=0,
        second=0,
        microsecond=0,
    )
    signals = strategy._manage_positions(et_now, None)
    assert signals
    assert "pre_expiry_1d" in signals[0].reason


def test_qqq_strike_rounding_all_paths() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("QQQ", 575.22)
    provider.set_bars("QQQ", "1Day", _bars_from_closes([560 + i for i in range(25)], 2_000_000))

    base_params = {
        "underlyings": ["QQQ"],
        "entry_start_hour": 0,
        "entry_start_minute": 0,
        "entry_end_hour": 23,
        "entry_end_minute": 59,
        "min_vix_proxy": 0.0,
        "strike_atr_distance": 1.0,
    }

    entry_strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params=base_params),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    entry_strategy.set_capital(100000.0)
    entry_strategy._get_atr = lambda _symbol: 14.44
    entry_strategy._get_sma5_bias = lambda _symbol, _price: "bullish"

    entry_signals = entry_strategy._scan_entries(
        datetime.now(pytz.timezone("America/New_York")),
        _regime(RegimeType.HIGH_VOL, vix=24.0),
    )
    assert entry_signals
    entry_trade = entry_strategy._trades["QQQ"]
    assert entry_trade.short_strike == 560.0
    assert float(entry_trade.short_strike).is_integer()

    assess_strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params=base_params),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    assess_strategy.set_capital(100000.0)
    assess_strategy._get_atr = lambda _symbol: 14.44
    assess_strategy._get_sma5_bias = lambda _symbol, _price: "bearish"

    assessment = assess_strategy.assess_opportunities(_regime(RegimeType.HIGH_VOL, vix=24.0))
    assert assessment.candidates
    candidate = assessment.candidates[0]
    assert candidate.short_strike == 590.0
    assert float(candidate.short_strike).is_integer()

    condor_strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={**base_params, "structure": "iron_condor"}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    condor_strategy.set_capital(100000.0)
    condor_strategy._get_atr = lambda _symbol: 14.44
    condor_strategy._get_sma5_bias = lambda _symbol, _price: "neutral"

    condor_signals = condor_strategy._scan_entries(
        datetime.now(pytz.timezone("America/New_York")),
        _regime(RegimeType.HIGH_VOL, vix=24.0),
    )
    assert condor_signals
    condor_trade = condor_strategy._trades["QQQ"]
    assert condor_trade.short_strike == 560.0
    assert condor_trade.call_short_strike == 590.0
    assert float(condor_trade.short_strike).is_integer()
    assert float(condor_trade.call_short_strike).is_integer()


def test_morning_gap_stop_closes_breached_position() -> None:
    provider = StubDataProvider()
    provider.set_price("QQQ", 600.0)
    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={"stop_loss_multiple": 3.0}),
        data_provider=provider,
        executor=StubExecutor(),
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)
    strategy._trades["QQQ"] = PremiumTrade(
        underlying="QQQ",
        structure="call_spread",
        short_strike=579.0,
        long_strike=584.0,
        contracts=1,
        credit_received=66.0,
        max_profit=66.0,
        max_loss=434.0,
        entry_time=datetime.now(pytz.UTC) - timedelta(days=1),
        expiration=date.today() + timedelta(days=1),
        simulated=True,
        capital_used=434.0,
        trade_id="qqq_gap_stop",
    )

    class _ReviewExecutor:
        def get_positions(self):
            return []

    class _RiskManager:
        @staticmethod
        def compute_overnight_risk(_positions):
            return {
                "total_exposure": 0.0,
                "overnight_gap_risk": 0.0,
                "exposure_pct": 0.0,
            }

    orch = Orchestrator.__new__(Orchestrator)
    orch._executor = _ReviewExecutor()
    orch._risk_manager = _RiskManager()
    orch._strategies = {"options_premium": strategy}
    orch._log = _DummyLog()
    orch._last_morning_position_review = None

    orch._morning_position_review()

    assert "QQQ" not in strategy._trades


def test_options_atr_uses_daily_bars() -> None:
    class _AtrTrackingProvider(StubDataProvider):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, str, int]] = []

        def get_bars(self, symbol, timeframe, limit=100, start=None, end=None):
            self.calls.append((symbol, timeframe, limit))
            return super().get_bars(symbol, timeframe, limit=limit, start=start, end=end)

    provider = _AtrTrackingProvider()
    provider.set_bars("QQQ", "1Day", _bars_from_closes([560 + i for i in range(25)], 2_000_000))

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={}),
        data_provider=provider,
        executor=StubExecutor(),
        event_bus=EventBus(),
    )

    atr = strategy._get_atr("QQQ")
    assert atr is not None
    assert provider.calls
    assert provider.calls[0][1] == "1Day"
    assert all(call[1] not in {"1Min", "5Min", "15Min", "30Min", "1Hour"} for call in provider.calls)


def test_momentum_trailing_stop() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("XYZ", 101.0)

    strategy = MomentumStrategy(
        name="momentum",
        config=StrategyConfig(
            params={
                "use_trailing_stop": True,
                "trailing_stop_atr_multiple": 1.5,
                "trail_activation_rr": 0.0,
                "max_hold_days": 3,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)
    strategy._get_daily_atr = lambda _symbol: 2.0

    strategy._trades["XYZ"] = MomentumTrade(
        symbol="XYZ",
        direction="long",
        entry_price=100.0,
        stop_price=95.0,
        initial_stop=95.0,
        target_price=None,
        atr=2.0,
        breakout_level=99.0,
        volume_ratio=2.0,
        entry_time=datetime.now(pytz.UTC) - timedelta(days=1),
        capital_used=5000.0,
        trailing_active=True,
        trail_stop=99.0,
        highest_price=105.0,
        lowest_price=100.0,
    )

    et_now = datetime.now(pytz.timezone("America/New_York"))
    signals = strategy._manage_positions(et_now)
    assert signals
    assert "trail_stop" in signals[0].reason


def test_mean_reversion_daily_ma() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    closes = [100.0] * 30 + [94.0]
    provider.set_bars("XYZ", "1Day", _bars_from_closes(closes, 1_000_000))

    strategy = VWAPReversionStrategy(
        name="vwap_reversion",
        config=StrategyConfig(params={"universe": ["XYZ"], "min_z_score": 2.0}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    assessment = strategy.assess_opportunities(_regime(RegimeType.RANGING))
    assert assessment.candidates
    candidate = assessment.candidates[0]
    assert "ma_20" in candidate.metadata
    assert "std_20" in candidate.metadata
    assert candidate.metadata.get("swing_trade") is True


def test_orchestrator_holds_overnight(monkeypatch) -> None:
    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 3, 10, 15, 56, 0)
            if tz is None:
                return base
            return tz.localize(base)

    monkeypatch.setattr("algotrader.orchestrator.datetime", _FrozenDateTime)

    swing_state = {"review_calls": 0, "close_all_calls": 0}
    swing_strategy = SimpleNamespace(
        config=SimpleNamespace(params={}),
        close_positions_for_eod=lambda et_now: swing_state.__setitem__("review_calls", swing_state["review_calls"] + 1) or 0,
        close_all_positions=lambda reason="": swing_state.__setitem__("close_all_calls", swing_state["close_all_calls"] + 1) or 0,
    )

    orch = Orchestrator.__new__(Orchestrator)
    orch._eod_handled_date = None
    orch._strategies = {"momentum": swing_strategy}
    orch._log = _DummyLog()
    orch._check_expiry_risk = lambda: None

    orch._handle_market_close()
    assert swing_state["review_calls"] == 1
    assert swing_state["close_all_calls"] == 0


def test_disabled_strategy_not_loaded() -> None:
    """Strategy with enabled=false is skipped at orchestrator load."""
    orch = Orchestrator.__new__(Orchestrator)
    orch._settings = Settings()
    orch._shutdown_requested = False
    orch._data_provider = StubDataProvider()
    orch._executor = StubExecutor()
    orch._event_bus = EventBus()
    orch._trade_journal = object()
    orch._log = _DummyLog()
    orch._disabled_strategy_names = []

    orch._import_strategies()
    strategies = orch._load_strategies()

    assert set(strategies) == {"options_premium"}
    assert set(orch._disabled_strategy_names) == {
        "momentum",
        "vwap_reversion",
        "gap_reversal",
        "pairs_trading",
        "sector_rotation",
        "event_driven",
    }


def test_orchestrator_closes_gap_reversal_at_eod(monkeypatch) -> None:
    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 3, 10, 15, 56, 0)
            if tz is None:
                return base
            return tz.localize(base)

    monkeypatch.setattr("algotrader.orchestrator.datetime", _FrozenDateTime)

    state = {"closed": 0}
    gap_strategy = SimpleNamespace(
        config=SimpleNamespace(params={"intraday_only": True}),
        close_all_positions=lambda reason="": state.__setitem__("closed", state["closed"] + 1) or 1,
        close_positions_for_eod=lambda et_now: 0,
    )

    orch = Orchestrator.__new__(Orchestrator)
    orch._eod_handled_date = None
    orch._strategies = {"gap_reversal": gap_strategy}
    orch._log = _DummyLog()
    orch._check_expiry_risk = lambda: None

    orch._handle_market_close()
    assert state["closed"] == 1


def test_overnight_exposure_limit() -> None:
    class _Exec:
        def get_account(self):
            return AccountInfo(
                equity=100000.0,
                cash=50000.0,
                buying_power=100000.0,
                portfolio_value=100000.0,
            )

        def get_positions(self):
            return [SimpleNamespace(market_value=50000.0, symbol="AAPL")]

    class _Strategy:
        def __init__(self):
            self.capital = 1.0

        def set_capital(self, value: float):
            self.capital = value

        def set_brain_contract_cap(self, _value):
            return None

    orch = Orchestrator.__new__(Orchestrator)
    orch._executor = _Exec()
    orch._max_overnight_exposure_pct = 40.0
    orch._log = _DummyLog()
    s = _Strategy()
    orch._strategies = {"options_premium": s}

    assert orch._is_overnight_exposure_limited() is True
    orch._block_new_entries_for_exposure_limit(context="market_open")
    assert s.capital == 0.0


def test_position_state_persistence(tmp_path) -> None:
    provider = StubDataProvider()
    executor = StubExecutor()

    strategy = MomentumStrategy(
        name="momentum",
        config=StrategyConfig(params={"max_hold_days": 3}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy._trades["XYZ"] = MomentumTrade(
        symbol="XYZ",
        direction="long",
        entry_price=100.0,
        stop_price=95.0,
        initial_stop=95.0,
        target_price=110.0,
        atr=2.0,
        breakout_level=99.0,
        volume_ratio=1.8,
        entry_time=datetime.now(pytz.UTC) - timedelta(days=1),
        capital_used=5000.0,
        trade_id="persist_1",
    )
    strategy.save_state(state_dir=str(tmp_path))

    restored = MomentumStrategy(
        name="momentum",
        config=StrategyConfig(params={"max_hold_days": 3}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    assert restored.restore_state(state_dir=str(tmp_path)) is True
    assert "XYZ" in restored._trades
    assert restored._trades["XYZ"].direction == "long"


def test_expired_position_purged_on_startup() -> None:
    provider = StubDataProvider()
    provider.set_price("SPY", 500.0)
    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={}),
        data_provider=provider,
        executor=StubExecutor(),
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)
    strategy.save_state = lambda state_dir="data/state": None  # type: ignore[method-assign]

    expired_date = date.today() - timedelta(days=1)
    strategy._restore_state(
        {
            "enabled": True,
            "capital_reserved": 400.0,
            "daily_pnl": 0.0,
            "daily_trades": 0,
            "trades": {
                "SPY": {
                    "underlying": "SPY",
                    "structure": "put_spread",
                    "short_strike": 500.0,
                    "long_strike": 495.0,
                    "contracts": 1,
                    "credit_received": 100.0,
                    "max_profit": 100.0,
                    "max_loss": 400.0,
                    "entry_time": datetime.now(pytz.UTC).isoformat(),
                    "expiration": expired_date.isoformat(),
                    "simulated": False,
                    "capital_used": 400.0,
                    "trade_id": "expired_startup",
                    "open_order_id": "ibkr_test",
                    "short_occ_symbol": "SPY   260101P00500000",
                    "long_occ_symbol": "SPY   260101P00495000",
                    "call_short_occ_symbol": "",
                    "call_long_occ_symbol": "",
                }
            },
        }
    )

    assert not strategy._trades
    assert strategy._capital_reserved == 0.0


def test_expired_contract_close_failure_purges() -> None:
    class _ExpiredCloseExecutor(StubExecutor):
        def submit_mleg_order(self, legs, qty=1, tif="DAY", client_order_id=None):
            return None

        def get_last_mleg_failure_reason(self):
            return "expired"

        def get_option_positions(self):
            return []

    provider = StubDataProvider()
    provider.set_price("SPY", 500.0)
    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={}),
        data_provider=provider,
        executor=_ExpiredCloseExecutor(),
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)
    strategy.save_state = lambda state_dir="data/state": None  # type: ignore[method-assign]

    strategy._trades["SPY"] = PremiumTrade(
        underlying="SPY",
        structure="put_spread",
        short_strike=500.0,
        long_strike=495.0,
        contracts=1,
        credit_received=100.0,
        max_profit=100.0,
        max_loss=400.0,
        entry_time=datetime.now(pytz.UTC) - timedelta(days=2),
        expiration=date.today() - timedelta(days=1),
        simulated=False,
        capital_used=400.0,
        trade_id="expired_close_failure",
        short_occ_symbol="SPY   260101P00500000",
        long_occ_symbol="SPY   260101P00495000",
    )

    strategy._close_trade("SPY", strategy._trades["SPY"], "pre_expiry_1d")

    assert "SPY" not in strategy._trades


def test_pre_expiry_uses_trading_days() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("SPY", 490.0)
    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={"close_before_expiry_days": 1, "stop_loss_multiple": 3.0}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    today = date.today()
    days_to_friday = (4 - today.weekday()) % 7
    if days_to_friday == 0:
        days_to_friday = 7
    friday = today + timedelta(days=days_to_friday)
    monday = friday + timedelta(days=3)

    strategy._trades["SPY"] = PremiumTrade(
        underlying="SPY",
        structure="put_spread",
        short_strike=490.0,
        long_strike=485.0,
        contracts=1,
        credit_received=100.0,
        max_profit=100.0,
        max_loss=400.0,
        entry_time=datetime.now(pytz.UTC) - timedelta(days=1),
        expiration=monday,
        simulated=True,
        capital_used=400.0,
        trade_id="trading_day_test",
    )

    et_now = datetime.now(pytz.timezone("America/New_York")).replace(
        year=friday.year,
        month=friday.month,
        day=friday.day,
        hour=11,
        minute=0,
        second=0,
        microsecond=0,
    )
    signals = strategy._manage_positions(et_now, None)

    assert signals
    assert "pre_expiry_1d" in signals[0].reason

