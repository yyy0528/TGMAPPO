"""Composition root: the only place that selects concrete component classes."""

import importlib
from pathlib import Path

from .config import ExperimentConfig
from .envs.reference import ReferenceGridEnvironment
from .models.disabled import DisabledProgressModel, DisabledWorldModel
from .policies.rules import RulePolicy
from .registry import ComponentRegistry, ComponentUnavailable
from .training.runner import EpisodeRunner


def default_registry() -> ComponentRegistry:
    registry = ComponentRegistry()

    def reference_environment(config, device):
        if config.policy.use_earp:
            raise ComponentUnavailable("reference_grid does not implement EARP; register a research environment")
        return ReferenceGridEnvironment(config.environment)

    def graph_policy(config, device):
        # Rule-only commands continue to work without importing torch/PyG.
        from .policies.graph import GraphMAPPOPolicy
        return GraphMAPPOPolicy(config, device)

    def graph_trainer(config, device):
        from .training.ppo import GraphMAPPOTrainer
        return GraphMAPPOTrainer(config, device)

    registry.register("environment", "reference_grid", reference_environment)
    registry.register("policy", "geographic", lambda config, device: RulePolicy("geographic"))
    registry.register("policy", "random", lambda config, device: RulePolicy("random"))
    registry.register("policy", "graph_mappo", graph_policy)
    registry.register("trainer", "graph_mappo", graph_trainer)
    registry.register("world_model", "disabled", lambda config, device: DisabledWorldModel())
    registry.register("progress_model", "disabled", lambda config, device: DisabledProgressModel())
    return registry


def load_extensions(registry: ComponentRegistry, modules: list[str]) -> None:
    for name in modules:
        module = importlib.import_module(name)
        register = getattr(module, "register_components", None)
        if not callable(register):
            raise ValueError(name + " must export register_components(registry)")
        register(registry)


def build_runner(config: ExperimentConfig, device: str,
                 registry: ComponentRegistry) -> EpisodeRunner:
    selections = (
        ("environment", config.environment.backend), ("policy", config.policy.backend),
        ("world_model", config.world_model.backend if config.world_model.enabled else "disabled"),
        ("progress_model", config.progress.backend if config.progress.enabled else "disabled"),
    )
    for kind, name in selections:
        registry.require(kind, name)
    for section in (config.world_model, config.progress):
        if section.enabled and not Path(section.checkpoint).is_file():
            raise ValueError("Checkpoint does not exist: " + section.checkpoint)
    components = []
    try:
        for kind, name in selections:
            components.append(registry.build(kind, name, config, device))
        return EpisodeRunner(config, *components)
    except Exception:
        if components:
            components[0].close()
        raise
