"""Tests for SingleStockConfig defaults and YAML loading."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from algotrader.core.config import Settings, load_settings


def test_singlestock_config_defaults(tmp_path: Path, monkeypatch) -> None:
    # Bypass the auto-load-from-config behavior so we test the pydantic
    # defaults rather than whatever the real settings.yaml has.
    monkeypatch.setenv("ALGOTRADER_CONFIG", str(tmp_path / "nonexistent.yaml"))
    s = Settings(**{})
    assert s.singlestock.enabled is True
    assert s.singlestock.symbol == "AAPL"
    assert s.singlestock.capital_pct == 20.0
    assert s.singlestock.min_conviction == 0.65
    assert s.singlestock.max_hold_days == 3
    assert s.singlestock.target_delta == 0.70
    assert s.singlestock.min_dte == 10
    assert s.singlestock.max_dte == 14
    assert s.singlestock.ibkr_client_id == 118
    # Default is 0; settings.yaml overrides to 20 in this repo.
    assert s.trading.reserved_for_singlestock_pct == 0.0


def test_singlestock_config_yaml_override(tmp_path: Path) -> None:
    yml = tmp_path / "settings.yaml"
    yml.write_text(dedent("""
        trading:
          reserved_for_singlestock_pct: 25.0
        singlestock:
          symbol: NVDA
          capital_pct: 25.0
          min_conviction: 0.75
          target_delta: 0.65
          pdt_safe_mode: true
    """), encoding="utf-8")
    s = load_settings(yml)
    assert s.singlestock.symbol == "NVDA"
    assert s.singlestock.capital_pct == 25.0
    assert s.singlestock.min_conviction == 0.75
    assert s.singlestock.target_delta == 0.65
    assert s.singlestock.pdt_safe_mode is True
    assert s.trading.reserved_for_singlestock_pct == 25.0
