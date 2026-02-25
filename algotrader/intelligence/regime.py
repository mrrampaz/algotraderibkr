"""Market regime classifier.

Classifies the current market regime using:
- VIX level and change
- SPY trend (SMA20 slope)
- Intraday range vs ATR (volatility)

Output: RegimeType (trending_up, trending_down, ranging, high_vol, low_vol, event_day)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import structlog

from algotrader.core.events import EventBus, REGIME_CHANGED
from algotrader.core.models import MarketRegime, RegimeType
from algotrader.data.provider import DataProvider
from algotrader.intelligence.calendar.events import EventCalendar

logger = structlog.get_logger()

# VIX thresholds
VIX_LOW = 13.0
VIX_NORMAL_HIGH = 20.0
VIX_HIGH = 25.0
VIX_EXTREME = 35.0

# Trend thresholds (SMA20 daily slope, annualized %)
TREND_UP_THRESHOLD = 0.05      # > 5% annualized slope = uptrend
TREND_DOWN_THRESHOLD = -0.05   # < -5% = downtrend

# ATR ratio thresholds
ATR_EXPANSION_RATIO = 1.5      # Today's range > 1.5x ATR = expanding vol
ATR_CONTRACTION_RATIO = 0.5    # Today's range < 0.5x ATR = contracting vol

# VIX data quality safeguards
VIX_SANITY_MIN = 10.0
VIX_SANITY_MAX = 80.0
VIX_STALE_MINUTES = 30
VIX_STALE_EPSILON = 1e-4


class RegimeDetector:
    """Classify the current market regime.

    Uses VIX level, SPY trend direction, and volatility characteristics
    to determine which regime the market is in. Strategies use this to
    decide whether to activate and how to size positions.
    """

    def __init__(
        self,
        data_provider: DataProvider,
        event_bus: EventBus,
        spy_symbol: str = "SPY",
        vix_symbol: str = "VIX",  # live VIX index when supported by provider
        event_calendar: EventCalendar | None = None,
    ) -> None:
        self._data = data_provider
        self._event_bus = event_bus
        self._spy_symbol = spy_symbol
        self._vix_symbol = vix_symbol
        self._event_calendar = event_calendar or EventCalendar()
        self._log = logger.bind(component="regime_detector")

        self._current_regime: MarketRegime | None = None
        self._event_days: set[str] = set()  # Dates marked as event days (YYYY-MM-DD)
        self._last_vix_level: float | None = None
        self._last_vix_fetch_at: datetime | None = None
        self._last_vix_change_at: datetime | None = None

    @property
    def current_regime(self) -> MarketRegime | None:
        return self._current_regime

    def mark_event_day(self, date_str: str) -> None:
        """Mark a date as an event day (FOMC, CPI, etc). Format: YYYY-MM-DD."""
        self._event_days.add(date_str)

    def detect(self) -> MarketRegime:
        """Run regime detection and return the classified regime.

        Called pre-market and periodically during the day.
        """
        now = datetime.now(pytz.UTC)
        today_et = now.astimezone(pytz.timezone("America/New_York")).date()
        today_str = today_et.strftime("%Y-%m-%d")

        # Gather data
        vix_level = self._get_vix_level()
        spy_trend = self._get_spy_trend()
        atr_ratio = self._get_atr_ratio()

        is_event_day = today_str in self._event_days or self._event_calendar.is_event_day(today_et)
        if is_event_day:
            regime = MarketRegime(
                regime_type=RegimeType.EVENT_DAY,
                vix_level=vix_level,
                spy_trend=spy_trend,
                volatility_percentile=atr_ratio,
                confidence=self._calculate_confidence(vix_level, spy_trend, atr_ratio, RegimeType.EVENT_DAY),
                timestamp=now,
            )
            self._update_regime(regime)
            return regime

        # Classify
        regime_type = self._classify(vix_level, spy_trend, atr_ratio)
        confidence = self._calculate_confidence(vix_level, spy_trend, atr_ratio, regime_type)

        regime = MarketRegime(
            regime_type=regime_type,
            vix_level=vix_level,
            spy_trend=spy_trend,
            volatility_percentile=atr_ratio,
            confidence=confidence,
            timestamp=now,
        )

        self._update_regime(regime)
        return regime

    def _classify(
        self,
        vix_level: float | None,
        spy_trend: float | None,
        atr_ratio: float | None,
    ) -> RegimeType:
        """Determine regime from indicators."""

        # High vol override: VIX > 25 always means high vol regime
        if vix_level is not None and vix_level >= VIX_HIGH:
            return RegimeType.HIGH_VOL

        # Low vol: VIX < 13 and low ATR
        if vix_level is not None and vix_level < VIX_LOW:
            if atr_ratio is not None and atr_ratio < ATR_CONTRACTION_RATIO:
                return RegimeType.LOW_VOL
            # Low VIX but normal range — could still be trending
            if spy_trend is not None:
                if spy_trend > TREND_UP_THRESHOLD:
                    return RegimeType.TRENDING_UP
                elif spy_trend < TREND_DOWN_THRESHOLD:
                    return RegimeType.TRENDING_DOWN
            return RegimeType.LOW_VOL

        # Normal VIX range: classify by trend
        if spy_trend is not None:
            if spy_trend > TREND_UP_THRESHOLD:
                return RegimeType.TRENDING_UP
            elif spy_trend < TREND_DOWN_THRESHOLD:
                return RegimeType.TRENDING_DOWN

        # Elevated vol with no clear trend
        if atr_ratio is not None and atr_ratio > ATR_EXPANSION_RATIO:
            return RegimeType.HIGH_VOL

        return RegimeType.RANGING

    def _calculate_confidence(
        self,
        vix_level: float | None,
        spy_trend: float | None,
        atr_ratio: float | None,
        regime_type: RegimeType,
    ) -> float:
        """Calculate confidence in the regime classification (0-1)."""
        scores: list[float] = []

        if regime_type == RegimeType.HIGH_VOL and vix_level is not None:
            # More confident the higher VIX is above threshold
            scores.append(min(1.0, (vix_level - VIX_NORMAL_HIGH) / (VIX_EXTREME - VIX_NORMAL_HIGH)))

        if regime_type in (RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN) and spy_trend is not None:
            # More confident the steeper the slope
            scores.append(min(1.0, abs(spy_trend) / 0.15))

        if regime_type == RegimeType.RANGING:
            # Ranging is the default/uncertain classification
            scores.append(0.5)

        if regime_type == RegimeType.LOW_VOL and vix_level is not None:
            scores.append(min(1.0, (VIX_LOW - vix_level) / 5.0 + 0.5))

        if regime_type == RegimeType.EVENT_DAY:
            scores.append(0.9)
            if vix_level is not None and vix_level >= VIX_NORMAL_HIGH:
                scores.append(1.0)

        return round(np.mean(scores) if scores else 0.5, 2)

    def _get_vix_level(self) -> float | None:
        """Fetch VIX from live data source with staleness and sanity checks."""
        now = datetime.now(pytz.UTC)

        # Primary path: IBKR index quote via optional provider method.
        provider_get_index = getattr(self._data, "get_index_price", None)
        if callable(provider_get_index):
            try:
                vix_live = provider_get_index("VIX", exchange="CBOE", wait_seconds=2.0)
                if vix_live is not None:
                    if self._is_vix_stale(vix_live, now):
                        self._log.warning(
                            "vix_quote_stale",
                            vix=round(vix_live, 4),
                            stale_minutes=self._stale_vix_minutes(now),
                            last_change_at=(
                                self._last_vix_change_at.isoformat()
                                if self._last_vix_change_at else None
                            ),
                        )
                        # Re-fetch once with a longer wait before accepting stale print.
                        refreshed = provider_get_index("VIX", exchange="CBOE", wait_seconds=3.5)
                        if refreshed is not None:
                            vix_live = refreshed

                    self._validate_vix(vix_live)
                    self._record_vix(vix_live, now)
                    self._log.debug("vix_live_quote", source="ibkr_index", vix=round(vix_live, 4))
                    return vix_live
            except Exception:
                self._log.exception("vix_live_fetch_failed")

        # Fallback path: provider snapshot for configured proxy symbol.
        try:
            snap = self._data.get_snapshot(self._vix_symbol)
            if snap and snap.latest_trade_price and snap.latest_trade_price > 0:
                proxy_vix = float(snap.latest_trade_price)
                self._validate_vix(proxy_vix)
                self._record_vix(proxy_vix, now)
                self._log.warning(
                    "vix_proxy_used",
                    symbol=self._vix_symbol,
                    vix=round(proxy_vix, 4),
                )
                return proxy_vix
        except Exception:
            self._log.exception("vix_proxy_snapshot_failed")

        # Last-resort fallback: realized-vol estimate from SPY.
        try:
            bars = self._data.get_bars(self._spy_symbol, "1Day", 25)
            if bars.empty or len(bars) < 10:
                return None
            returns = bars["close"].pct_change().dropna()
            realized_vol = float(returns.std() * np.sqrt(252) * 100)
            vix_estimate = realized_vol / 0.75
            self._validate_vix(vix_estimate)
            self._record_vix(vix_estimate, now)
            self._log.debug(
                "vix_proxy_estimated_from_spy",
                realized_vol=round(realized_vol, 2),
                vix_estimate=round(vix_estimate, 2),
            )
            return vix_estimate
        except Exception:
            self._log.exception("vix_proxy_failed")
        return None

    def _record_vix(self, vix_level: float, now: datetime) -> None:
        if self._last_vix_level is None or abs(vix_level - self._last_vix_level) > VIX_STALE_EPSILON:
            self._last_vix_change_at = now
        self._last_vix_level = vix_level
        self._last_vix_fetch_at = now

    def _is_vix_stale(self, vix_level: float, now: datetime) -> bool:
        if self._last_vix_level is None or self._last_vix_change_at is None:
            return False
        unchanged = abs(vix_level - self._last_vix_level) <= VIX_STALE_EPSILON
        if not unchanged:
            return False
        return (now - self._last_vix_change_at) >= timedelta(minutes=VIX_STALE_MINUTES)

    def _stale_vix_minutes(self, now: datetime) -> float:
        if self._last_vix_change_at is None:
            return 0.0
        return round((now - self._last_vix_change_at).total_seconds() / 60.0, 1)

    def _validate_vix(self, vix_level: float) -> None:
        if vix_level < VIX_SANITY_MIN or vix_level > VIX_SANITY_MAX:
            self._log.warning(
                "vix_sanity_warning",
                vix=round(vix_level, 4),
                expected_range=f"{VIX_SANITY_MIN}-{VIX_SANITY_MAX}",
            )

    def _get_spy_trend(self) -> float | None:
        """Calculate SPY SMA20 slope as annualized rate of change.

        Positive = uptrend, negative = downtrend, near zero = ranging.
        """
        try:
            bars = self._data.get_bars(self._spy_symbol, "1Day", 30)
            if bars.empty or len(bars) < 20:
                return None

            closes = bars["close"].values
            sma20 = pd.Series(closes).rolling(20).mean().dropna().values

            if len(sma20) < 5:
                return None

            # Linear regression slope on the SMA values
            x = np.arange(len(sma20))
            slope, _ = np.polyfit(x, sma20, 1)

            # Normalize: slope per day -> annualized percentage
            current_price = sma20[-1]
            if current_price > 0:
                daily_pct = slope / current_price
                annualized = daily_pct * 252
                return round(float(annualized), 4)
        except Exception:
            self._log.exception("spy_trend_failed")
        return None

    def _get_atr_ratio(self) -> float | None:
        """Calculate today's range as a ratio of the 14-day ATR.

        > 1.5 = expanding volatility
        < 0.5 = contracting volatility
        ~1.0 = normal
        """
        try:
            bars = self._data.get_bars(self._spy_symbol, "1Day", 20)
            if bars.empty or len(bars) < 15:
                return None

            highs = bars["high"].values
            lows = bars["low"].values
            closes = bars["close"].values

            # True Range
            tr_values = []
            for i in range(1, len(bars)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
                tr_values.append(tr)

            if len(tr_values) < 14:
                return None

            atr_14 = float(np.mean(tr_values[-14:]))
            if atr_14 == 0:
                return None

            # Today's range
            today_range = highs[-1] - lows[-1]
            return round(float(today_range / atr_14), 3)
        except Exception:
            self._log.exception("atr_ratio_failed")
        return None

    def _update_regime(self, regime: MarketRegime) -> None:
        """Update current regime and publish event if changed."""
        old_type = self._current_regime.regime_type if self._current_regime else None
        self._current_regime = regime

        self._log.info(
            "regime_detected",
            regime=regime.regime_type.value,
            vix=regime.vix_level,
            spy_trend=regime.spy_trend,
            atr_ratio=regime.volatility_percentile,
            confidence=regime.confidence,
        )

        if old_type != regime.regime_type:
            self._log.info(
                "regime_changed",
                old=old_type.value if old_type else "none",
                new=regime.regime_type.value,
            )
            self._event_bus.publish(
                REGIME_CHANGED,
                regime=regime,
                old_type=old_type,
            )
