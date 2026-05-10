from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytz
import pytest

from algotrader.core.config import StrategyConfig
from algotrader.core.events import EventBus
from algotrader.core.models import (
    AccountInfo,
    MarketClock,
    MarketRegime,
    Order,
    OrderSide,
    Quote,
    RegimeType,
    SignalDirection,
    Snapshot,
)
from algotrader.strategies.event_driven import EventDrivenStrategy
from algotrader.strategies.gap_reversal import GapReversalStrategy
from algotrader.strategies.momentum import MomentumStrategy
from algotrader.strategies.options_premium import OptionsPremiumStrategy, PremiumTrade
from algotrader.strategies.pairs_trading import PairsTradingStrategy
from algotrader.strategies.sector_rotation import SectorRotationStrategy
from algotrader.strategies.vwap_reversion import VWAPReversionStrategy
from algotrader.strategy_selector.candidate import CandidateType


class StubDataProvider:
    def __init__(self) -> None:
        self.bars: dict[tuple[str, str], pd.DataFrame] = {}
        self.prices: dict[str, float] = {}
        self.news: dict[str, list] = {}
        self.option_chains: dict[str, pd.DataFrame] = {}

    def set_bars(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        self.bars[(symbol, timeframe)] = df

    def set_price(self, symbol: str, price: float) -> None:
        self.prices[symbol] = price

    def set_news(self, symbol: str, items: list) -> None:
        self.news[symbol] = items

    def set_option_chain(self, symbol: str, chain: pd.DataFrame) -> None:
        self.option_chains[symbol] = chain.copy()

    def get_bars(self, symbol, timeframe, limit=100, start=None, end=None):
        df = self.bars.get((symbol, timeframe))
        if df is None:
            return pd.DataFrame()
        if limit and len(df) > limit:
            return df.tail(limit).copy()
        return df.copy()

    def get_quote(self, symbol):
        price = self.prices.get(symbol, 100.0)
        return Quote(
            symbol=symbol,
            timestamp=datetime.now(pytz.UTC),
            bid_price=price - 0.01,
            bid_size=100,
            ask_price=price + 0.01,
            ask_size=100,
        )

    def get_snapshot(self, symbol):
        price = self.prices.get(symbol)
        return Snapshot(symbol=symbol, latest_trade_price=price)

    def get_snapshots(self, symbols):
        return {s: self.get_snapshot(s) for s in symbols}

    def get_option_chain(self, underlying, expiration=None):
        chain = self.option_chains.get(underlying)
        if chain is None:
            return pd.DataFrame()
        if expiration is None:
            return chain.copy()
        if "expiration" not in chain.columns:
            return chain.copy()
        filtered = chain[chain["expiration"] == expiration]
        return filtered.copy()

    def get_news(self, symbols=None, limit=50):
        if not symbols:
            out = []
            for rows in self.news.values():
                out.extend(rows)
            return out[:limit]
        out = []
        for sym in symbols:
            out.extend(self.news.get(sym, []))
        return out[:limit]

    def is_market_open(self):
        return True

    def get_clock(self):
        now = datetime.now(pytz.UTC)
        return MarketClock(timestamp=now, is_open=True, next_open=now, next_close=now + timedelta(hours=6))


class StubExecutor:
    def submit_order(self, *args, **kwargs):
        return None

    def cancel_order(self, order_id):
        return False

    def close_position(self, symbol):
        return True

    def get_position(self, symbol):
        return None

    def get_positions(self):
        return []

    def get_account(self):
        return AccountInfo(
            equity=100000.0,
            cash=100000.0,
            buying_power=200000.0,
            portfolio_value=100000.0,
        )

    def get_order(self, order_id):
        return None

    def get_open_orders(self, symbol=None):
        return []

    def replace_stop_order(self, order_id, new_stop_price):
        return None


class OptionsMarkExecutor(StubExecutor):
    def __init__(self, option_positions: list[dict] | None = None) -> None:
        self._option_positions = option_positions or []
        self.mleg_submissions = 0

    def set_option_positions(self, option_positions: list[dict]) -> None:
        self._option_positions = option_positions

    def get_option_positions(self):
        return self._option_positions

    def submit_mleg_order(self, legs, qty=1, tif="DAY", client_order_id=None):
        self.mleg_submissions += 1
        return f"ibkr_test_{self.mleg_submissions}"

    def lookup_option_contract(self, underlying, put_or_call, strike, expiration):
        right = "P" if str(put_or_call).lower() == "put" else "C"
        strike_code = f"{int(round(float(strike) * 1000)):08d}"
        return f"{underlying}   {expiration.strftime('%y%m%d')}{right}{strike_code}"


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def info(self, event: str, **kwargs) -> None:
        self.events.append(("info", event, kwargs))

    def warning(self, event: str, **kwargs) -> None:
        self.events.append(("warning", event, kwargs))

    def debug(self, event: str, **kwargs) -> None:
        self.events.append(("debug", event, kwargs))

    def error(self, event: str, **kwargs) -> None:
        self.events.append(("error", event, kwargs))

    def exception(self, event: str, **kwargs) -> None:
        self.events.append(("exception", event, kwargs))


def _regime(regime_type: RegimeType, vix: float = 18.0) -> MarketRegime:
    return MarketRegime(
        regime_type=regime_type,
        vix_level=vix,
        confidence=0.9,
        timestamp=datetime.now(pytz.UTC),
    )


def _bars_from_closes(closes: list[float], base_volume: float = 1000.0) -> pd.DataFrame:
    rows = []
    for i, close in enumerate(closes):
        rows.append(
            {
                "open": close,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": base_volume * (1.0 if i < len(closes) - 1 else 1.5),
            }
        )
    return pd.DataFrame(rows)


def test_options_premium_returns_candidates() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("SPY", 500.0)
    provider.set_bars("SPY", "1Day", _bars_from_closes([490 + i for i in range(20)], 2_000_000))

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(
            params={
                "underlyings": ["SPY"],
                "entry_start_hour": 0,
                "entry_start_minute": 0,
                "entry_end_hour": 23,
                "entry_end_minute": 59,
                "min_vix_proxy": 10.0,
                "max_contracts": 3,
                "max_risk_per_trade": 800,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    assessment = strategy.assess_opportunities(_regime(RegimeType.RANGING, vix=22.0))
    assert assessment.candidates
    candidate = assessment.candidates[0]
    assert candidate.candidate_type in (CandidateType.CREDIT_SPREAD, CandidateType.IRON_CONDOR)
    assert candidate.options_structure in {"put_spread", "call_spread", "iron_condor"}
    assert candidate.short_strike > 0
    assert candidate.credit_received > 0
    assert candidate.max_loss > 0


def test_options_premium_max_loss_sizing_caps_credit_spread_contracts() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("SPY", 500.0)
    provider.set_bars("SPY", "1Day", _bars_from_closes([490 + i for i in range(20)], 2_000_000))

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(
            params={
                "underlyings": ["SPY"],
                "entry_start_hour": 0,
                "entry_start_minute": 0,
                "entry_end_hour": 23,
                "entry_end_minute": 59,
                "min_vix_proxy": 10.0,
                "sizing_method": "max_loss",
                "max_loss_risk_budget_pct": 2.0,
                "max_contracts_per_spread": 3,
                "max_contracts": 5,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    assessment = strategy.assess_opportunities(_regime(RegimeType.HIGH_VOL, vix=24.0))
    assert assessment.candidates
    candidate = assessment.candidates[0]
    assert candidate.candidate_type == CandidateType.CREDIT_SPREAD
    assert candidate.contracts == 3
    assert candidate.suggested_qty == 3


def test_options_premium_assess_filters_low_modeled_credit_per_contract() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("SPY", 500.0)
    provider.set_bars("SPY", "1Day", _bars_from_closes([490 + i for i in range(20)], 2_000_000))

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(
            params={
                "underlyings": ["SPY"],
                "entry_start_hour": 0,
                "entry_start_minute": 0,
                "entry_end_hour": 23,
                "entry_end_minute": 59,
                "min_vix_proxy": 10.0,
                "credit_estimate_pct": 0.01,  # $5 per contract on a $5-wide spread
                "min_credit_per_contract": 10.0,
                "min_credit_to_max_loss_ratio": 0.0,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    assessment = strategy.assess_opportunities(_regime(RegimeType.HIGH_VOL, vix=24.0))
    assert not assessment.candidates


def test_options_premium_assess_filters_poor_credit_to_max_loss_ratio() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("SPY", 500.0)
    provider.set_bars("SPY", "1Day", _bars_from_closes([490 + i for i in range(20)], 2_000_000))

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(
            params={
                "underlyings": ["SPY"],
                "entry_start_hour": 0,
                "entry_start_minute": 0,
                "entry_end_hour": 23,
                "entry_end_minute": 59,
                "min_vix_proxy": 10.0,
                "credit_estimate_pct": 0.03,  # $15 credit vs ~$485 max loss => ~3.1%
                "min_credit_per_contract": 1.0,
                "min_credit_to_max_loss_ratio": 0.05,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    assessment = strategy.assess_opportunities(_regime(RegimeType.HIGH_VOL, vix=24.0))
    assert not assessment.candidates


def test_options_premium_entry_filters_low_live_credit_before_submit() -> None:
    provider = StubDataProvider()
    executor = OptionsMarkExecutor()
    provider.set_price("SPY", 500.0)
    provider.set_bars("SPY", "1Day", _bars_from_closes([490 + i for i in range(20)], 2_000_000))

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(
            params={
                "underlyings": ["SPY"],
                "entry_start_hour": 0,
                "entry_start_minute": 0,
                "entry_end_hour": 23,
                "entry_end_minute": 59,
                "min_vix_proxy": 10.0,
                "min_credit_per_contract": 10.0,
                "min_credit_to_max_loss_ratio": 0.05,
                "require_live_credit_estimate": True,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    # Keep strike selection deterministic: put spread 499/494.
    strategy._get_current_price = lambda symbol: 500.0
    strategy._get_atr = lambda symbol: 1.0
    strategy._get_sma5_bias = lambda symbol, current_price: "bullish"

    et_now = datetime.now(pytz.timezone("America/New_York")).replace(
        hour=11,
        minute=0,
        second=0,
        microsecond=0,
    )
    provider.set_option_chain(
        "SPY",
        pd.DataFrame(
            [
                {
                    "symbol": "SPY_test_short",
                    "expiration": et_now.date(),
                    "strike": 499.0,
                    "type": "put",
                    "bid": 0.15,
                    "ask": 0.20,
                    "last": 0.17,
                },
                {
                    "symbol": "SPY_test_long",
                    "expiration": et_now.date(),
                    "strike": 494.0,
                    "type": "put",
                    "bid": 0.10,
                    "ask": 0.10,
                    "last": 0.10,
                },
            ]
        ),
    )

    signals = strategy._scan_entries(et_now, _regime(RegimeType.HIGH_VOL, vix=24.0))
    assert not signals
    assert not strategy._trades
    assert executor.mleg_submissions == 0


def test_options_premium_requires_live_credit_when_configured() -> None:
    provider = StubDataProvider()
    executor = OptionsMarkExecutor()
    logger = RecordingLogger()
    provider.set_price("SPY", 500.0)
    provider.set_bars("SPY", "1Day", _bars_from_closes([490 + i for i in range(20)], 2_000_000))

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(
            params={
                "underlyings": ["SPY"],
                "entry_start_hour": 0,
                "entry_start_minute": 0,
                "entry_end_hour": 23,
                "entry_end_minute": 59,
                "min_vix_proxy": 10.0,
                "min_credit_per_contract": 10.0,
                "min_credit_to_max_loss_ratio": 0.05,
                "require_live_credit_estimate": True,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)
    strategy._log = logger
    strategy._get_current_price = lambda symbol: 500.0
    strategy._get_atr = lambda symbol: 1.0
    strategy._get_sma5_bias = lambda symbol, current_price: "bullish"
    strategy._find_expiry = lambda underlying, min_dte=2, max_dte=5: et_now.date()
    strategy._estimate_live_credit_per_contract = lambda **kwargs: None

    et_now = datetime.now(pytz.timezone("America/New_York")).replace(
        hour=11,
        minute=0,
        second=0,
        microsecond=0,
    )
    signals = strategy._scan_entries(et_now, _regime(RegimeType.HIGH_VOL, vix=24.0))

    assert not signals
    assert not strategy._trades
    assert executor.mleg_submissions == 0
    assert (
        "warning",
        "options_live_credit_unavailable",
        {
            "underlying": "SPY",
            "structure": "put_spread",
            "contracts": 1,
            "reason": "live_credit_unavailable",
            "skip_reason": "live_credit_required",
        },
    ) in logger.events


def test_options_premium_can_fall_back_to_modeled_credit_when_live_credit_not_required() -> None:
    provider = StubDataProvider()
    executor = OptionsMarkExecutor()
    logger = RecordingLogger()
    provider.set_price("SPY", 500.0)
    provider.set_bars("SPY", "1Day", _bars_from_closes([490 + i for i in range(20)], 2_000_000))

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(
            params={
                "underlyings": ["SPY"],
                "entry_start_hour": 0,
                "entry_start_minute": 0,
                "entry_end_hour": 23,
                "entry_end_minute": 59,
                "min_vix_proxy": 10.0,
                "min_credit_per_contract": 10.0,
                "min_credit_to_max_loss_ratio": 0.05,
                "require_live_credit_estimate": False,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)
    strategy._log = logger
    strategy._get_current_price = lambda symbol: 500.0
    strategy._get_atr = lambda symbol: 1.0
    strategy._get_sma5_bias = lambda symbol, current_price: "bullish"
    strategy._find_expiry = lambda underlying, min_dte=2, max_dte=5: et_now.date()
    strategy._estimate_live_credit_per_contract = lambda **kwargs: None

    et_now = datetime.now(pytz.timezone("America/New_York")).replace(
        hour=11,
        minute=0,
        second=0,
        microsecond=0,
    )
    signals = strategy._scan_entries(et_now, _regime(RegimeType.HIGH_VOL, vix=24.0))

    assert signals
    assert "SPY" in strategy._trades
    assert executor.mleg_submissions == 1
    assert any(
        level == "info"
        and event == "options_live_credit_unavailable"
        and fields.get("reason") == "falling_back_to_modeled_credit_estimate"
        for level, event, fields in logger.events
    )
    assert any(event == "premium_entry" for _, event, _ in logger.events)


def test_options_premium_brain_contract_cap_override_applies_at_entry() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("SPY", 500.0)
    provider.set_bars("SPY", "1Day", _bars_from_closes([490 + i for i in range(20)], 2_000_000))

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(
            params={
                "underlyings": ["SPY"],
                "entry_start_hour": 0,
                "entry_start_minute": 0,
                "entry_end_hour": 23,
                "entry_end_minute": 59,
                "min_vix_proxy": 10.0,
                "sizing_method": "max_loss",
                "max_loss_risk_budget_pct": 2.0,
                "max_contracts_per_spread": 3,
                "max_contracts": 5,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(5000.0)

    # Baseline: configured spread cap (3) is enforced.
    baseline = strategy._size_contracts_for_structure(
        structure="put_spread",
        underlying="SPY",
        short_strike=495.0,
        max_loss_per_contract=475.0,
        call_short_strike=None,
        context="entry",
    )
    assert baseline == 3

    # Brain override: allow up to 5 contracts using allocated capital budget.
    strategy.set_brain_contract_cap(5)
    overridden = strategy._size_contracts_for_structure(
        structure="put_spread",
        underlying="SPY",
        short_strike=495.0,
        max_loss_per_contract=475.0,
        call_short_strike=None,
        context="entry",
    )
    assert overridden == 5

    strategy.set_brain_contract_cap(None)


def test_options_profit_target_min_hold_blocks_early_close() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("SPY", 500.0)

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={"min_hold_minutes_for_profit_target": 60}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)
    strategy._trades["SPY"] = PremiumTrade(
        underlying="SPY",
        structure="put_spread",
        short_strike=490.0,
        long_strike=485.0,
        contracts=1,
        credit_received=150.0,
        max_profit=150.0,
        max_loss=350.0,
        entry_time=datetime.now(pytz.UTC),
        expiration=datetime.now(pytz.timezone("America/New_York")).date(),
        simulated=True,
        capital_used=350.0,
        trade_id="test_trade",
    )

    et_now = datetime.now(pytz.timezone("America/New_York")).replace(
        hour=11,
        minute=0,
        second=0,
        microsecond=0,
    )
    signals = strategy._manage_positions(et_now, None)

    assert not signals
    assert "SPY" in strategy._trades


def test_options_live_marks_override_proxy_and_prevent_false_profit_close() -> None:
    provider = StubDataProvider()
    provider.set_price("SPY", 500.0)  # Proxy model would look strongly profitable.
    executor = OptionsMarkExecutor(
        option_positions=[
            {
                "local_symbol": "SPY   260306C00679000",
                "qty": -1.0,
                "average_cost": 52.0,
                "market_price": 0.70,
            },
            {
                "local_symbol": "SPY   260306C00684000",
                "qty": 1.0,
                "average_cost": 12.0,
                "market_price": 0.15,
            },
        ],
    )

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={"min_hold_minutes_for_profit_target": 60}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)
    strategy._trades["SPY"] = PremiumTrade(
        underlying="SPY",
        structure="call_spread",
        short_strike=679.0,
        long_strike=684.0,
        contracts=1,
        credit_received=40.0,
        max_profit=40.0,
        max_loss=460.0,
        entry_time=datetime.now(pytz.UTC) - timedelta(hours=2),
        expiration=datetime.now(pytz.timezone("America/New_York")).date(),
        simulated=False,
        capital_used=460.0,
        trade_id="test_trade",
        short_occ_symbol="SPY   260306C00679000",
        long_occ_symbol="SPY   260306C00684000",
    )

    et_now = datetime.now(pytz.timezone("America/New_York")).replace(
        hour=11,
        minute=0,
        second=0,
        microsecond=0,
    )
    signals = strategy._manage_positions(et_now, None)

    assert not signals
    assert "SPY" in strategy._trades


def test_options_close_on_live_mark_profit_target_after_min_hold() -> None:
    provider = StubDataProvider()
    provider.set_price("SPY", 500.0)
    executor = OptionsMarkExecutor(
        option_positions=[
            {
                "local_symbol": "SPY   260306C00679000",
                "qty": -1.0,
                "average_cost": 70.0,
                "market_price": 0.30,
            },
            {
                "local_symbol": "SPY   260306C00684000",
                "qty": 1.0,
                "average_cost": 15.0,
                "market_price": 0.05,
            },
        ],
    )

    strategy = OptionsPremiumStrategy(
        name="options_premium",
        config=StrategyConfig(params={"min_hold_minutes_for_profit_target": 60}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)
    strategy._trades["SPY"] = PremiumTrade(
        underlying="SPY",
        structure="call_spread",
        short_strike=679.0,
        long_strike=684.0,
        contracts=1,
        credit_received=55.0,
        max_profit=55.0,
        max_loss=445.0,
        entry_time=datetime.now(pytz.UTC) - timedelta(hours=2),
        expiration=datetime.now(pytz.timezone("America/New_York")).date(),
        simulated=False,
        capital_used=445.0,
        trade_id="test_trade",
        short_occ_symbol="SPY   260306C00679000",
        long_occ_symbol="SPY   260306C00684000",
    )

    et_now = datetime.now(pytz.timezone("America/New_York")).replace(
        hour=11,
        minute=0,
        second=0,
        microsecond=0,
    )
    signals = strategy._manage_positions(et_now, None)

    assert signals
    assert signals[0].direction == SignalDirection.CLOSE
    assert "profit_target" in signals[0].reason
    assert executor.mleg_submissions == 1
    assert "SPY" not in strategy._trades


def test_vwap_reversion_returns_candidates() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    closes = [100.0] * 35 + [94.0]
    provider.set_bars("XYZ", "5Min", _bars_from_closes(closes, 1_000_000))
    provider.set_price("XYZ", 94.0)

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
    assert candidate.risk_reward_ratio > 0
    assert 0.0 <= candidate.confidence <= 1.0
    if candidate.direction == "long":
        assert candidate.stop_price < candidate.entry_price
    else:
        assert candidate.stop_price > candidate.entry_price
    assert abs(candidate.target_price - candidate.metadata["vwap"]) <= 0.5


def test_vwap_high_vol_requires_config_enable_and_stricter_threshold() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    closes = [100.0] * 35 + [94.0]
    provider.set_bars("XYZ", "5Min", _bars_from_closes(closes, 1_000_000))
    provider.set_price("XYZ", 94.0)

    disabled = VWAPReversionStrategy(
        name="vwap_reversion",
        config=StrategyConfig(params={"universe": ["XYZ"], "min_z_score": 2.0}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    disabled.set_capital(100000.0)
    disabled_assessment = disabled.assess_opportunities(_regime(RegimeType.HIGH_VOL, vix=28.0))
    assert not disabled_assessment.candidates

    strict_enabled = VWAPReversionStrategy(
        name="vwap_reversion",
        config=StrategyConfig(
            params={
                "universe": ["XYZ"],
                "min_z_score": 2.0,
                "allow_high_vol": True,
                "high_vol_min_z_score": 5.0,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strict_enabled.set_capital(100000.0)
    strict_assessment = strict_enabled.assess_opportunities(_regime(RegimeType.HIGH_VOL, vix=28.0))
    assert not strict_assessment.candidates

    enabled = VWAPReversionStrategy(
        name="vwap_reversion",
        config=StrategyConfig(
            params={
                "universe": ["XYZ"],
                "min_z_score": 2.0,
                "allow_high_vol": True,
                "high_vol_min_z_score": 2.5,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    enabled.set_capital(100000.0)
    enabled_assessment = enabled.assess_opportunities(_regime(RegimeType.HIGH_VOL, vix=28.0))
    assert enabled_assessment.candidates


def test_momentum_uses_high_vol_scanner_range(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    captured: dict[str, float] = {}

    class DummyScanner:
        def __init__(self, data_provider, min_volume_ratio, max_range_pct, min_consolidation_days):
            captured["max_range_pct"] = float(max_range_pct)

        def scan(self):
            return []

    import algotrader.intelligence.scanners.breakout_scanner as breakout_module

    monkeypatch.setattr(breakout_module, "BreakoutScanner", DummyScanner)

    strategy = MomentumStrategy(
        name="momentum",
        config=StrategyConfig(
            params={
                "scanner_max_range_pct": 5.0,
                "high_vol_scanner_max_range_pct": 8.0,
                "allowed_regimes": ["high_vol"],
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    strategy.assess_opportunities(_regime(RegimeType.HIGH_VOL, vix=30.0))
    assert captured["max_range_pct"] == 8.0


def test_pairs_trading_antichurn_filter() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_bars("AAA", "5Min", _bars_from_closes([100.0, 100.2], 500_000))
    provider.set_bars("BBB", "5Min", _bars_from_closes([100.0, 99.8], 500_000))

    strategy = PairsTradingStrategy(
        name="pairs_trading",
        config=StrategyConfig(
            params={
                "pairs": [{"symbol_a": "AAA", "symbol_b": "BBB", "sector": "test"}],
                "z_entry_threshold": 0.2,
                "z_exit_threshold": 0.1995,
                "z_stop_threshold": 0.6,
                "per_pair_capital_pct": 2.0,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    state = strategy._pair_states["AAA_BBB"]
    state.is_cointegrated = True
    state.is_positioned = False
    state.correlation = 0.9
    state.cointegration_pvalue = 0.01
    state.hedge_ratio = 1.0
    state.spread_mean = 0.0
    state.spread_std = 1.0

    state.z_score = 0.20  # thin expected edge with custom z_exit
    assessment_thin = strategy.assess_opportunities(_regime(RegimeType.RANGING))
    assert not assessment_thin.candidates

    state.z_score = 2.20
    assessment_strong = strategy.assess_opportunities(_regime(RegimeType.RANGING))
    assert assessment_strong.candidates


def test_pairs_trading_returns_candidates() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_bars("AAA", "5Min", _bars_from_closes([100.0, 101.0], 500_000))
    provider.set_bars("BBB", "5Min", _bars_from_closes([50.0, 49.5], 500_000))

    strategy = PairsTradingStrategy(
        name="pairs_trading",
        config=StrategyConfig(
            params={
                "pairs": [{"symbol_a": "AAA", "symbol_b": "BBB", "sector": "test"}],
                "z_entry_threshold": 1.0,
                "z_exit_threshold": 0.2,
                "z_stop_threshold": 2.8,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    state = strategy._pair_states["AAA_BBB"]
    state.is_cointegrated = True
    state.is_positioned = False
    state.correlation = 0.88
    state.cointegration_pvalue = 0.01
    state.hedge_ratio = 1.1
    state.spread_mean = 1.0
    state.spread_std = 0.5
    state.z_score = 2.0

    assessment = strategy.assess_opportunities(_regime(RegimeType.RANGING))
    assert assessment.candidates
    candidate = assessment.candidates[0]
    assert candidate.candidate_type == CandidateType.PAIRS
    assert candidate.symbol_b == "BBB"
    assert candidate.hedge_ratio > 0
    assert candidate.z_score != 0


def test_gap_reversal_returns_candidates() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("GAP", 102.0)
    provider.set_bars("GAP", "5Min", _bars_from_closes([101.0, 102.0, 102.5], 1_500_000))
    provider.set_news("GAP", [])

    strategy = GapReversalStrategy(
        name="gap_reversal",
        config=StrategyConfig(),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)
    strategy.set_gap_candidates(
        [{"symbol": "GAP", "gap_pct": 3.2, "prev_close": 100.0, "current_price": 103.2, "direction": "up"}]
    )

    assessment = strategy.assess_opportunities(_regime(RegimeType.TRENDING_UP))
    assert assessment.candidates
    candidate = assessment.candidates[0]
    expiry_et = candidate.expiry_time.astimezone(pytz.timezone("America/New_York"))
    assert expiry_et.hour == 11 and expiry_et.minute == 0


def test_sector_rotation_returns_candidates() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()

    provider.set_price("XLK", 200.0)
    provider.set_price("XLU", 80.0)

    provider.set_bars("SPY", "1Day", _bars_from_closes([100 + i * 0.2 for i in range(12)], 1_000_000))
    provider.set_bars("XLK", "1Day", _bars_from_closes([100 + i * 1.0 for i in range(12)], 1_500_000))
    provider.set_bars("XLU", "1Day", _bars_from_closes([100 - i * 0.8 for i in range(12)], 1_500_000))

    strategy = SectorRotationStrategy(
        name="sector_rotation",
        config=StrategyConfig(
            params={
                "sectors": {"XLK": "Technology", "XLU": "Utilities"},
                "rs_long_threshold": 0.5,
                "rs_short_threshold": -0.5,
                "min_divergence_pct": 1.0,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    assessment = strategy.assess_opportunities(_regime(RegimeType.TRENDING_UP))
    assert len(assessment.candidates) <= 1
    if assessment.candidates:
        candidate = assessment.candidates[0]
        assert candidate.candidate_type == CandidateType.SECTOR_LONG_SHORT
        assert "long_symbol" in candidate.metadata and "short_symbol" in candidate.metadata


def test_sector_rotation_skips_candidate_when_capital_below_min_notional() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()

    provider.set_price("XLK", 200.0)
    provider.set_price("XLU", 80.0)
    provider.set_bars("SPY", "1Day", _bars_from_closes([100 + i * 0.2 for i in range(12)], 1_000_000))
    provider.set_bars("XLK", "1Day", _bars_from_closes([100 + i * 1.0 for i in range(12)], 1_500_000))
    provider.set_bars("XLU", "1Day", _bars_from_closes([100 - i * 0.8 for i in range(12)], 1_500_000))

    strategy = SectorRotationStrategy(
        name="sector_rotation",
        config=StrategyConfig(
            params={
                "sectors": {"XLK": "Technology", "XLU": "Utilities"},
                "rs_long_threshold": 0.5,
                "rs_short_threshold": -0.5,
                "min_divergence_pct": 1.0,
                "min_notional_per_leg": 2000,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(3000.0)

    assessment = strategy.assess_opportunities(_regime(RegimeType.TRENDING_UP))
    assert not assessment.candidates


def test_sector_rotation_candidate_respects_min_notional_qty() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()

    provider.set_price("XLK", 200.0)
    provider.set_price("XLU", 80.0)
    provider.set_bars("SPY", "1Day", _bars_from_closes([100 + i * 0.2 for i in range(12)], 1_000_000))
    provider.set_bars("XLK", "1Day", _bars_from_closes([100 + i * 1.0 for i in range(12)], 1_500_000))
    provider.set_bars("XLU", "1Day", _bars_from_closes([100 - i * 0.8 for i in range(12)], 1_500_000))

    strategy = SectorRotationStrategy(
        name="sector_rotation",
        config=StrategyConfig(
            params={
                "sectors": {"XLK": "Technology", "XLU": "Utilities"},
                "rs_long_threshold": 0.5,
                "rs_short_threshold": -0.5,
                "min_divergence_pct": 1.0,
                "min_notional_per_leg": 2000,
            }
        ),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(20000.0)

    assessment = strategy.assess_opportunities(_regime(RegimeType.TRENDING_UP))
    assert assessment.candidates
    candidate = assessment.candidates[0]
    # At $200/$80 with $2,000 minimum notional per leg, qty floor is max(10, 25)=25.
    assert candidate.suggested_qty >= 25


def test_event_driven_returns_candidates_on_event_day() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("SPY", 500.0)
    provider.set_bars("SPY", "5Min", _bars_from_closes([498.0 + i * 0.2 for i in range(30)], 1_200_000))

    strategy = EventDrivenStrategy(
        name="event_driven",
        config=StrategyConfig(params={"instruments": ["SPY"]}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    now_et = datetime.now(pytz.timezone("America/New_York"))
    strategy._events_checked_today = True
    strategy._today_events = [
        {"type": "fomc", "time": (now_et - timedelta(minutes=10)).strftime("%H:%M"), "impact": 3}
    ]
    strategy._pre_event_prices["SPY"] = 498.0

    assessment = strategy.assess_opportunities(_regime(RegimeType.EVENT_DAY))
    assert assessment.candidates

    strategy._today_events = []
    empty = strategy.assess_opportunities(_regime(RegimeType.EVENT_DAY))
    assert not empty.candidates


def test_event_driven_low_confidence_pre_event() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    provider.set_price("SPY", 500.0)
    provider.set_bars("SPY", "5Min", _bars_from_closes([499.0 + i * 0.1 for i in range(30)], 1_100_000))

    strategy = EventDrivenStrategy(
        name="event_driven",
        config=StrategyConfig(params={"instruments": ["SPY"], "pre_event_entry_hours": 3}),
        data_provider=provider,
        executor=executor,
        event_bus=EventBus(),
    )
    strategy.set_capital(100000.0)

    now_et = datetime.now(pytz.timezone("America/New_York"))
    strategy._events_checked_today = True
    strategy._today_events = [
        {"type": "cpi", "time": (now_et + timedelta(hours=1)).strftime("%H:%M"), "impact": 3}
    ]
    strategy._pre_event_prices["SPY"] = 499.5

    assessment = strategy.assess_opportunities(_regime(RegimeType.EVENT_DAY))
    assert assessment.candidates
    assert all(c.confidence <= 0.40 for c in assessment.candidates)


def test_all_strategies_backward_compatible() -> None:
    provider = StubDataProvider()
    executor = StubExecutor()
    event_bus = EventBus()

    provider.set_price("SPY", 500.0)
    provider.set_price("XYZ", 95.0)
    provider.set_price("AAA", 100.0)
    provider.set_price("BBB", 100.0)
    provider.set_price("GAP", 102.0)
    provider.set_price("XLK", 200.0)
    provider.set_price("XLU", 80.0)

    provider.set_bars("SPY", "1Day", _bars_from_closes([490 + i for i in range(20)], 2_000_000))
    provider.set_bars("SPY", "5Min", _bars_from_closes([498.0 + i * 0.2 for i in range(30)], 1_000_000))
    provider.set_bars("XYZ", "5Min", _bars_from_closes([100.0] * 35 + [94.0], 1_000_000))
    provider.set_bars("AAA", "5Min", _bars_from_closes([100.0, 101.0], 500_000))
    provider.set_bars("BBB", "5Min", _bars_from_closes([100.0, 99.0], 500_000))
    provider.set_bars("GAP", "5Min", _bars_from_closes([101.0, 102.0, 102.5], 1_500_000))
    provider.set_bars("XLK", "1Day", _bars_from_closes([100 + i * 1.0 for i in range(12)], 1_500_000))
    provider.set_bars("XLU", "1Day", _bars_from_closes([100 - i * 0.8 for i in range(12)], 1_500_000))

    options = OptionsPremiumStrategy(
        "options_premium",
        StrategyConfig(params={"underlyings": ["SPY"], "entry_start_hour": 0, "entry_end_hour": 23, "min_vix_proxy": 10.0}),
        provider,
        executor,
        event_bus,
    )
    options.set_capital(100000.0)

    vwap = VWAPReversionStrategy(
        "vwap_reversion",
        StrategyConfig(params={"universe": ["XYZ"], "min_z_score": 2.0}),
        provider,
        executor,
        event_bus,
    )
    vwap.set_capital(100000.0)

    pairs = PairsTradingStrategy(
        "pairs_trading",
        StrategyConfig(params={"pairs": [{"symbol_a": "AAA", "symbol_b": "BBB"}], "z_entry_threshold": 1.0}),
        provider,
        executor,
        event_bus,
    )
    pairs.set_capital(100000.0)
    ps = pairs._pair_states["AAA_BBB"]
    ps.is_cointegrated = True
    ps.z_score = 2.0
    ps.correlation = 0.9
    ps.cointegration_pvalue = 0.01
    ps.hedge_ratio = 1.0
    ps.spread_mean = 0.0
    ps.spread_std = 1.0

    gap = GapReversalStrategy("gap_reversal", StrategyConfig(), provider, executor, event_bus)
    gap.set_capital(100000.0)
    gap.set_gap_candidates([{"symbol": "GAP", "gap_pct": 3.2, "prev_close": 100.0, "current_price": 103.2}])

    sector = SectorRotationStrategy(
        "sector_rotation",
        StrategyConfig(
            params={"sectors": {"XLK": "Technology", "XLU": "Utilities"}, "rs_long_threshold": 0.5, "rs_short_threshold": -0.5, "min_divergence_pct": 1.0}
        ),
        provider,
        executor,
        event_bus,
    )
    sector.set_capital(100000.0)

    event = EventDrivenStrategy(
        "event_driven",
        StrategyConfig(params={"instruments": ["SPY"]}),
        provider,
        executor,
        event_bus,
    )
    event.set_capital(100000.0)
    now_et = datetime.now(pytz.timezone("America/New_York"))
    event._events_checked_today = True
    event._today_events = [{"type": "fomc", "time": (now_et - timedelta(minutes=10)).strftime("%H:%M"), "impact": 3}]
    event._pre_event_prices["SPY"] = 499.0

    assessments = [
        options.assess_opportunities(_regime(RegimeType.RANGING, vix=22.0)),
        vwap.assess_opportunities(_regime(RegimeType.RANGING)),
        pairs.assess_opportunities(_regime(RegimeType.RANGING)),
        gap.assess_opportunities(_regime(RegimeType.TRENDING_UP)),
        sector.assess_opportunities(_regime(RegimeType.TRENDING_UP)),
        event.assess_opportunities(_regime(RegimeType.EVENT_DAY)),
    ]

    for assessment in assessments:
        assert assessment.num_candidates >= 0
        assert assessment.avg_risk_reward >= 0
        assert isinstance(assessment.candidates, list)
