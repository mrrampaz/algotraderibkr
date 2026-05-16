"""Pick a slightly-ITM weekly option for a directional thesis.

Given a symbol, direction, and the DTE window, query the option chain
and return the contract whose delta is closest to the configured target
(default 0.70 for ITM weekly). Also computes the recommended contract
count under the exercise-exposure cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import pytz
import structlog

from algotrader.data.provider import DataProvider
from algotrader.singlestock.thesis import Direction

logger = structlog.get_logger()


@dataclass
class PickedOption:
    symbol: str  # local option symbol
    right: str  # "C" or "P"
    strike: float
    expiry: date
    bid: float
    ask: float
    mid: float
    delta: float
    iv: float | None
    contracts: int
    estimated_cost: float  # per single contract, mid * 100


class OptionsPicker:
    def __init__(
        self,
        data_provider: DataProvider,
        target_delta: float = 0.70,
        min_dte: int = 10,
        max_dte: int = 14,
        max_contracts_per_trade: int = 5,
        max_position_premium_pct: float = 5.0,
        exercise_exposure_cap_pct: float = 20.0,
    ) -> None:
        self._data = data_provider
        self._target_delta = target_delta
        self._min_dte = min_dte
        self._max_dte = max_dte
        self._max_contracts = max_contracts_per_trade
        self._max_premium_pct = max_position_premium_pct
        self._exposure_cap_pct = exercise_exposure_cap_pct
        self._log = logger.bind(component="singlestock_options_picker")

    def pick(
        self,
        symbol: str,
        direction: Direction,
        slice_capital: float,
    ) -> PickedOption | None:
        if direction == Direction.NONE:
            return None
        right_letter = "C" if direction == Direction.LONG else "P"

        expiry = self._pick_expiry(symbol)
        if expiry is None:
            self._log.warning("singlestock_options_no_expiry_in_window", symbol=symbol)
            return None

        try:
            chain = self._data.get_option_chain(symbol, expiration=expiry)
        except Exception:
            self._log.exception("singlestock_options_chain_fetch_failed", symbol=symbol)
            return None
        if chain is None or chain.empty:
            self._log.warning("singlestock_options_empty_chain", symbol=symbol, expiry=str(expiry))
            return None

        kind = "call" if right_letter == "C" else "put"
        filtered = chain[chain["type"] == kind].copy()
        if filtered.empty:
            return None

        # delta sign: puts have negative delta in chain — compare on |delta|.
        filtered["abs_delta"] = filtered["delta"].astype(float).abs()
        filtered = filtered[filtered["abs_delta"].between(0.40, 0.85)]
        filtered = filtered.dropna(subset=["bid", "ask"])
        filtered = filtered[(filtered["bid"] > 0) & (filtered["ask"] > 0) & (filtered["ask"] >= filtered["bid"])]
        if filtered.empty:
            self._log.warning(
                "singlestock_options_no_valid_strikes",
                symbol=symbol,
                kind=kind,
                expiry=str(expiry),
            )
            return None

        filtered["distance"] = (filtered["abs_delta"] - self._target_delta).abs()
        best = filtered.sort_values("distance").iloc[0]

        bid = float(best["bid"])
        ask = float(best["ask"])
        mid = round((bid + ask) / 2.0, 2)
        strike = float(best["strike"])
        delta = float(best["delta"])
        iv = float(best["implied_vol"]) if pd.notna(best.get("implied_vol")) else None
        local_symbol = str(best["symbol"])

        contracts = self._size_contracts(
            slice_capital=slice_capital,
            mid_premium=mid,
            strike=strike,
        )
        if contracts <= 0:
            self._log.warning(
                "singlestock_options_size_zero",
                symbol=symbol,
                mid=mid,
                strike=strike,
                slice_capital=slice_capital,
            )
            return None

        estimated_cost = mid * 100 * contracts
        self._log.info(
            "singlestock_options_picked",
            symbol=symbol,
            local_symbol=local_symbol,
            right=right_letter,
            strike=strike,
            expiry=str(expiry),
            delta=delta,
            bid=bid,
            ask=ask,
            mid=mid,
            iv=iv,
            contracts=contracts,
            estimated_cost=estimated_cost,
        )
        return PickedOption(
            symbol=local_symbol,
            right=right_letter,
            strike=strike,
            expiry=expiry,
            bid=bid,
            ask=ask,
            mid=mid,
            delta=delta,
            iv=iv,
            contracts=contracts,
            estimated_cost=estimated_cost,
        )

    def _pick_expiry(self, symbol: str) -> date | None:
        today = datetime.now(pytz.timezone("America/New_York")).date()
        # Probe candidate dates in the DTE window — try each Friday and
        # the explicit nearest dates; the provider rejects ones not in
        # the chain.
        candidates: list[date] = []
        for d in range(self._min_dte, self._max_dte + 1):
            candidates.append(today + timedelta(days=d))
        # Try Fridays specifically since weeklies expire on Fridays.
        seen: set[date] = set()
        ordered: list[date] = []
        for c in candidates:
            # Snap forward to next Friday if not Friday.
            offset = (4 - c.weekday()) % 7
            friday = c + timedelta(days=offset)
            dte = (friday - today).days
            if self._min_dte <= dte <= self._max_dte and friday not in seen:
                seen.add(friday)
                ordered.append(friday)

        for cand in ordered:
            try:
                chain = self._data.get_option_chain(symbol, expiration=cand)
            except Exception:
                continue
            if chain is not None and not chain.empty:
                return cand
        return None

    def _size_contracts(
        self,
        slice_capital: float,
        mid_premium: float,
        strike: float,
    ) -> int:
        if mid_premium <= 0 or strike <= 0 or slice_capital <= 0:
            return 0

        # Cap on premium spent.
        premium_budget = slice_capital * (self._max_premium_pct / 100.0)
        per_contract_cost = mid_premium * 100.0
        by_premium = math.floor(premium_budget / per_contract_cost)

        # Cap on exercise exposure (stock-equivalent notional at strike).
        exposure_budget = slice_capital * (self._exposure_cap_pct / 100.0)
        per_contract_notional = strike * 100.0
        by_exposure = math.floor(exposure_budget / per_contract_notional) if per_contract_notional > 0 else 0

        size = min(by_premium, by_exposure, self._max_contracts)
        return max(0, int(size))
