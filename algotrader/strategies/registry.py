"""Strategy discovery and registration."""

from __future__ import annotations

from typing import Type

import structlog

from algotrader.core.config import StrategyConfig
from algotrader.core.events import EventBus
from algotrader.data.provider import DataProvider
from algotrader.execution.executor import Executor
from algotrader.strategies.base import StrategyBase

logger = structlog.get_logger()


class StrategyRegistry:
    """Register strategies by name and instantiate them from config.

    Strategies register themselves with a decorator or explicit call.
    The orchestrator uses the registry to discover and create strategies.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Type[StrategyBase]] = {}
        self._log = logger.bind(component="strategy_registry")

    def register(self, name: str, strategy_class: Type[StrategyBase]) -> None:
        """Register a strategy class under a name."""
        if name in self._registry:
            self._log.warning("strategy_already_registered", name=name)
        self._registry[name] = strategy_class
        self._log.info("strategy_registered", name=name, cls=strategy_class.__name__)

    def get(self, name: str) -> Type[StrategyBase] | None:
        """Get a registered strategy class by name."""
        return self._registry.get(name)

    @property
    def names(self) -> list[str]:
        """List all registered strategy names."""
        return list(self._registry.keys())

    def create(
        self,
        name: str,
        config: StrategyConfig,
        data_provider: DataProvider,
        executor: Executor,
        event_bus: EventBus,
    ) -> StrategyBase | None:
        """Create a strategy instance from the registry."""
        cls = self._registry.get(name)
        if cls is None:
            self._log.error("strategy_not_found", name=name)
            return None

        try:
            strategy = cls(
                name=name,
                config=config,
                data_provider=data_provider,
                executor=executor,
                event_bus=event_bus,
            )
            self._log.info("strategy_created", name=name, cls=cls.__name__)
            return strategy
        except Exception:
            self._log.exception("strategy_creation_failed", name=name)
            return None

    def create_all(
        self,
        strategy_configs: dict[str, StrategyConfig],
        data_provider: DataProvider,
        executor: Executor,
        event_bus: EventBus,
    ) -> dict[str, StrategyBase]:
        """Create all enabled strategies from configs."""
        strategies: dict[str, StrategyBase] = {}

        for name, config in strategy_configs.items():
            if not config.enabled:
                self._log.info("strategy_disabled_in_config", name=name)
                continue

            strategy = self.create(name, config, data_provider, executor, event_bus)
            if strategy:
                strategies[name] = strategy

        return strategies


# Global registry instance
registry = StrategyRegistry()


def register_strategy(name: str):
    """Decorator to register a strategy class."""
    def decorator(cls: Type[StrategyBase]) -> Type[StrategyBase]:
        registry.register(name, cls)
        return cls
    return decorator
