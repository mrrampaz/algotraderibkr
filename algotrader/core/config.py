"""YAML config loader with pydantic validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# Load .env file
load_dotenv()


# ── Config Models ────────────────────────────────────────────────────────────


class TradingConfig(BaseModel):
    paper_mode: bool = True
    total_capital: float = 60000.0
    timezone: str = "America/New_York"


class DataConfig(BaseModel):
    provider: str = "alpaca"
    feed: str = "iex"
    cycle_interval_seconds: int = 300


class RiskConfig(BaseModel):
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 8.0
    max_gross_exposure_pct: float = 80.0
    max_single_position_pct: float = 5.0
    max_correlated_positions: int = 3
    strategy_daily_loss_limit_pct: float = 1.0


class ExecutionConfig(BaseModel):
    broker: str = "alpaca"
    max_spread_pct: float = 0.3
    use_marketable_limits: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "data/logs/algotrader.log"
    json_format: bool = True


class AlpacaConfig(BaseModel):
    api_key: str = ""
    secret_key: str = ""
    paper: bool = True
    base_url: str = ""

    @classmethod
    def from_env(cls) -> AlpacaConfig:
        paper = os.getenv("ALPACA_PAPER_TRADE", "True").lower() in ("true", "1", "yes")
        return cls(
            api_key=os.getenv("ALPACA_API_KEY", ""),
            secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            paper=paper,
            base_url=(
                "https://paper-api.alpaca.markets"
                if paper
                else "https://api.alpaca.markets"
            ),
        )


class StrategyConfig(BaseModel):
    """Base config for any strategy. Strategies extend this with their own fields."""

    enabled: bool = True
    capital_allocation_pct: float = 10.0
    max_positions: int = 5
    params: dict[str, Any] = Field(default_factory=dict)


class Settings(BaseModel):
    """Top-level application settings."""

    trading: TradingConfig = Field(default_factory=TradingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    strategies: dict[str, StrategyConfig] = Field(default_factory=dict)


# ── Loader ───────────────────────────────────────────────────────────────────


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file and return the parsed dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load global settings from YAML, overlay Alpaca creds from env."""
    if config_path is None:
        config_path = os.getenv("ALGOTRADER_CONFIG", "config/settings.yaml")

    path = Path(config_path)
    if path.exists():
        raw = load_yaml(path)
    else:
        raw = {}

    settings = Settings(**raw)

    # Always overlay Alpaca creds from environment
    settings.alpaca = AlpacaConfig.from_env()

    return settings


def load_strategy_config(strategy_name: str, config_dir: Path | str = "config/strategies") -> StrategyConfig:
    """Load a strategy-specific YAML config."""
    path = Path(config_dir) / f"{strategy_name}.yaml"
    if not path.exists():
        return StrategyConfig()
    raw = load_yaml(path)
    return StrategyConfig(**raw)
