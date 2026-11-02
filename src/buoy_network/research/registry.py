"""Explicit dependency injection. Factories are loaded only when requested."""

from __future__ import annotations

from typing import Any, Callable

from .config import ExperimentConfig

Factory = Callable[[ExperimentConfig, str], Any]


class ComponentUnavailable(ValueError):
    """A planned component has no registered implementation yet."""


class ComponentRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Factory]] = {}

    def register(self, kind: str, name: str, factory: Factory) -> None:
        group = self._factories.setdefault(kind, {})
        if name in group:
            raise ValueError("Component already registered: {}:{}".format(kind, name))
        if not callable(factory):
            raise TypeError("Component factory must be callable")
        group[name] = factory

    def require(self, kind: str, name: str) -> Factory:
        try:
            return self._factories[kind][name]
        except KeyError:
            choices = ", ".join(sorted(self._factories.get(kind, {}))) or "none"
            raise ComponentUnavailable(
                "{}:{} is not implemented/registered. Available: {}. "
                "Register a factory via --plugin MODULE (register_components).".format(
                    kind, name, choices
                )
            ) from None

    def build(self, kind: str, name: str, config: ExperimentConfig, device: str) -> Any:
        return self.require(kind, name)(config, device)

    def describe(self) -> dict[str, list[str]]:
        return {kind: sorted(group) for kind, group in sorted(self._factories.items())}
