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
    max_capital: float = 0.0  # 0 = use full broker account equity
    timezone: str = "America/New_York"
    # Capital carved out for the single-stock tool running in a separate
    # process. The main orchestrator subtracts this slice from
    # _effective_capital so the two processes don't double-count NetLiq.
    reserved_for_singlestock_pct: float = 0.0


class DataConfig(BaseModel):
    provider: str = "ibkr"
    feed: str = "iex"
    cycle_interval_seconds: int = 300


class RiskConfig(BaseModel):
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 8.0
    max_gross_exposure_pct: float = 80.0
    max_single_position_pct: float = 5.0
    max_overnight_exposure_pct: float = 40.0
    overnight_gap_buffer_pct: float = 5.0
    max_correlated_positions: int = 3
    strategy_daily_loss_limit_pct: float = 1.0


class ExecutionConfig(BaseModel):
    broker: str = "ibkr"
    max_spread_pct: float = 0.3
    use_marketable_limits: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "data/logs/algotrader.log"
    json_format: bool = True


class IBKRConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    timeout: int = 30
    readonly: bool = False
    account: str = ""
    max_reconnect_attempts: int = 5
    reconnect_delay_seconds: int = 10


class BrokerConfig(BaseModel):
    provider: str = "ibkr"
    ibkr: IBKRConfig = Field(default_factory=IBKRConfig)


class StrategyConfig(BaseModel):
    """Base config for any strategy. Strategies extend this with their own fields."""

    enabled: bool = True
    capital_allocation_pct: float = 10.0
    max_positions: int = 5
    params: dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access for small config inspection scripts."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.params.get(key, default)


class AlertsConfig(BaseModel):
    """Config for the alerting system."""

    enabled: bool = True
    big_win_threshold: float = 500.0
    big_loss_threshold: float = -300.0
    webhook_url: str = ""
    alert_file: str = "data/logs/alerts.log"


class BrainConfig(BaseModel):
    """Config for the Daily Brain selector mode."""

    class AdaptiveRiskTiersConfig(BaseModel):
        single_strategy_risk_pct: float = 5.0
        few_strategies_risk_pct: float = 4.0
        diversified_risk_pct: float = 3.0

    class DrawdownGovernorConfig(BaseModel):
        moderate_threshold_pct: float = 1.5
        severe_threshold_pct: float = 3.0

    class StrategyThresholdOverrideConfig(BaseModel):
        min_confidence: float | None = None
        min_rr: float | None = None
        min_edge: float | None = None

    min_confidence: float = 0.60
    min_risk_reward: float = 1.5
    min_edge_pct: float = 0.3
    options_min_confidence: float = 0.55
    options_min_risk_reward: float = 0.3
    options_min_edge_pct: float = 0.1
    strategy_overrides: dict[str, StrategyThresholdOverrideConfig] = Field(default_factory=dict)
    max_daily_trades: int = 5
    max_capital_per_trade_pct: float = 20.0
    max_daily_risk_pct: float = 2.0
    cash_is_default: bool = True
    regime_mismatch_penalty: float = 0.5
    correlation_penalty: float = 0.3
    recent_loss_cooldown_hours: int = 4
    midday_confidence_multiplier: float = 1.2
    midday_pnl_stop_pct: float = -1.0
    # Dynamic-cadence Brain: when set, the Brain re-decides multiple times
    # per day instead of just at open + midday. Cadence and triggers below.
    cadence_minutes: int = 60
    entry_window_start: str = "09:30"
    entry_window_end: str = "15:30"
    regime_change_triggers_decision: bool = True
    vix_delta_trigger: float = 1.5
    late_session_confidence_bump: float = 0.05
    adaptive_sizing: bool = True
    adaptive_risk_tiers: AdaptiveRiskTiersConfig = Field(default_factory=AdaptiveRiskTiersConfig)
    drawdown_governor: DrawdownGovernorConfig = Field(default_factory=DrawdownGovernorConfig)
    max_contracts_hard_cap: int = 10
    recent_win_rate_lookback_trades: int = 15
    recent_win_rate_fallback: float = 0.80


class StrategySelectorConfig(BaseModel):
    """Config for the Phase 4 strategy selection engine."""

    enabled: bool = True
    mode: str = "brain"  # "brain" or "classic"
    regime_config: str = "config/regimes.yaml"
    brain: BrainConfig = Field(default_factory=BrainConfig)

    # Classic mode parameters
    min_activation_score: float = 0.35
    review_hour: int = 12
    review_minute: int = 0
    min_allocation_pct: float = 3.0
    max_total_deployment_pct: float = 80.0
    scale_down_threshold_pct: float = -0.5
    disable_threshold_pct: float = -0.8
    scale_up_threshold_pct: float = 0.3
    scale_up_factor: float = 1.25
    scale_down_factor: float = 0.5
    min_trades_for_review: int = 2
    cash_threshold: float = 0.25
    max_single_strategy_pct: float = 70.0
    concentration_power: float = 2.0


class SingleStockConfig(BaseModel):
    """Config for the standalone single-stock day-trading tool.

    The tool runs in its own process (scripts/run_singlestock.py) with its
    own IBKR client_id and lockfile. It investigates a configurable symbol
    each morning via 5 agents, opens at most one slightly-ITM weekly option
    position, and manages it for up to max_hold_days.
    """

    enabled: bool = True
    symbol: str = "AAPL"
    # Slice of NetLiquidation this tool may use. Main bot must set
    # trading.reserved_for_singlestock_pct >= this value to prevent
    # buying-power double-count.
    capital_pct: float = 20.0
    min_conviction: float = 0.65
    max_hold_days: int = 3

    # Hard risk caps
    max_position_premium_pct: float = 5.0
    daily_loss_kill_pct: float = 3.0
    max_contracts_per_trade: int = 5
    exercise_exposure_cap_pct: float = 20.0

    # Position management
    premium_loss_close_pct: float = 35.0
    premium_gain_target_pct: float = 50.0
    enable_trailing_stop: bool = True
    intraday_check_minutes: int = 20
    news_recheck_minutes: int = 30

    # Options selection
    target_delta: float = 0.70
    min_dte: int = 10
    max_dte: int = 14

    # IBKR session (own client_id, separate from main bot)
    ibkr_client_id: int = 118

    # PDT
    pdt_safe_mode: bool = False

    # Blackouts
    earnings_blackout_days: int = 2
    iv_rank_max: float = 0.75

    # LLM
    llm_enabled: bool = True
    llm_model_news: str = "claude-sonnet-4-6"
    llm_model_announcements: str = "claude-sonnet-4-6"
    llm_model_decision: str = "claude-opus-4-7"
    llm_max_calls_per_day: int = 10
    llm_timeout_seconds: int = 60

    # Schedule (ET)
    premarket_investigate_time: str = "08:00"
    entry_time: str = "09:35"
    eod_review_time: str = "15:50"
    expiry_day_close_time: str = "15:30"

    # Paths
    log_file: str = "data/logs/singlestock.log"
    state_file: str = "data/state/singlestock.json"
    lock_file: str = "data/.singlestock.lock"


class Settings(BaseModel):
    """Top-level application settings."""

    trading: TradingConfig = Field(default_factory=TradingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    strategies: dict[str, StrategyConfig] = Field(default_factory=dict)
    strategy_selector: StrategySelectorConfig = Field(default_factory=StrategySelectorConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    singlestock: SingleStockConfig = Field(default_factory=SingleStockConfig)

    def __init__(self, **data: Any) -> None:
        """Load config/settings.yaml by default when called with no overrides."""
        if not data:
            config_path = os.getenv("ALGOTRADER_CONFIG", "config/settings.yaml")
            path = Path(config_path)
            if path.exists():
                data = load_yaml(path)
        super().__init__(**data)


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
    """Load global settings from YAML."""
    if config_path is None:
        config_path = os.getenv("ALGOTRADER_CONFIG", "config/settings.yaml")

    path = Path(config_path)
    raw = load_yaml(path) if path.exists() else {}
    return Settings(**raw)


def load_strategy_config(strategy_name: str, config_dir: Path | str = "config/strategies") -> StrategyConfig:
    """Load a strategy-specific YAML config."""
    path = Path(config_dir) / f"{strategy_name}.yaml"
    if not path.exists():
        return StrategyConfig()
    raw = load_yaml(path)
    return StrategyConfig(**raw)
