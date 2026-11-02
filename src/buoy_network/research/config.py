"""Single configuration definition shared by the CLI and component factories."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class EnvironmentConfig:
    version: str = "research_v1"
    backend: str = "reference_grid"
    grid_size: int = 5
    comm_range: float = 1.5
    task_horizon: int = 80
    task_ttl: int = 80
    num_moving_buoys: int = 5
    mobility_std: float = 0.0
    loss_probability: float = 0.0
    reward_mode: str = "sparse"
    delivery_reward: float = 1.0
    drop_reward: float = -1.0
    wait_cost: float = -0.01

    @property
    def num_nodes(self) -> int:
        return self.grid_size ** 2


@dataclass(frozen=True)
class PolicyConfig:
    backend: str = "geographic"
    trainer: str = "graph_mappo"
    rollout_length: int = 80
    gamma: float = 0.99
    gae_lambda: float = 0.95
    use_earp: bool = False
    hidden_size: int = 64
    graph_layers: int = 2
    checkpoint: str = ""
    deterministic: bool | None = None
    learning_rate: float = 3e-4
    ppo_epochs: int = 4
    minibatch_steps: int = 8
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class WorldModelConfig:
    enabled: bool = False
    backend: str = "local_graph_gru"
    trainer: str = "supervised_wm"
    checkpoint: str = ""
    history_len: int = 4
    hidden_size: int = 128
    graph_hidden_size: int = 64
    graph_layers: int = 2
    batch_size: int = 16
    learning_rate: float = 3e-4
    horizon: int = 1


@dataclass(frozen=True)
class ProgressConfig:
    enabled: bool = False
    backend: str = "prefix_gru"
    trainer: str = "supervised_pm"
    checkpoint: str = ""
    hidden_size: int = 64
    batch_size: int = 16
    learning_rate: float = 1e-3
    heads: int = 1
    alpha: float = 0.1
    update_mode: str = "frozen"


@dataclass(frozen=True)
class DataConfig:
    replay_capacity_steps: int = 20000
    train_fraction: float = 0.7
    validation_fraction: float = 0.15


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 12
    episodes: int = 3
    device: str = "auto"
    output_root: str = "results/research"
    run_name: str = ""
    dataset: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> ExperimentConfig:
        e, p, w, m, d, r = (
            self.environment, self.policy, self.world_model,
            self.progress, self.data, self.runtime,
        )
        for group in fields(self):
            section = getattr(self, group.name)
            defaults = type(section)()
            for item in fields(section):
                value, default = getattr(section, item.name), getattr(defaults, item.name)
                valid = type(value) is type(default)
                if group.name == "policy" and item.name == "deterministic":
                    valid = value is None or type(value) is bool
                if type(default) is float:
                    valid = type(value) in (float, int) and math.isfinite(value)
                if not valid:
                    raise ValueError("Invalid type/value for {}.{}".format(group.name, item.name))
        for name, value in (
            ("grid_size", e.grid_size), ("task_horizon", e.task_horizon),
            ("task_ttl", e.task_ttl), ("rollout_length", p.rollout_length),
            ("wm_history_len", w.history_len), ("wm_hidden_size", w.hidden_size),
            ("wm_batch_size", w.batch_size), ("pm_hidden_size", m.hidden_size),
            ("wm_graph_hidden_size", w.graph_hidden_size), ("wm_graph_layers", w.graph_layers),
            ("pm_batch_size", m.batch_size), ("episodes", r.episodes),
            ("replay_capacity_steps", d.replay_capacity_steps),
            ("policy_hidden_size", p.hidden_size), ("policy_graph_layers", p.graph_layers),
            ("ppo_epochs", p.ppo_epochs), ("minibatch_steps", p.minibatch_steps),
        ):
            if value < 1:
                raise ValueError(name + " must be positive")
        if e.grid_size < 2 or e.comm_range <= 0:
            raise ValueError("grid_size must be >= 2 and comm_range must be positive")
        if not 0 <= e.num_moving_buoys <= e.num_nodes:
            raise ValueError("num_moving_buoys must be within [0, num_nodes]")
        if e.mobility_std < 0 or not 0 <= e.loss_probability <= 1:
            raise ValueError("Invalid mobility_std or loss_probability")
        if e.version != "research_v1":
            raise ValueError("This entry supports research_v1; use run_algorithm.py for legacy")
        if e.reward_mode not in ("sparse", "task_dense"):
            raise ValueError("reward_mode must be sparse or task_dense")
        if not 0 < p.gamma <= 1 or not 0 <= p.gae_lambda <= 1:
            raise ValueError("Invalid gamma or gae_lambda")
        if (p.learning_rate <= 0 or not 0 < p.clip_param <= 1 or p.entropy_coef < 0
                or p.value_loss_coef <= 0 or p.max_grad_norm <= 0):
            raise ValueError("Invalid PPO learning rate, clipping or loss coefficients")
        if w.horizon != 1:
            raise ValueError("The first research schema supports wm_horizon=1 only")
        if w.learning_rate <= 0 or m.learning_rate <= 0:
            raise ValueError("Auxiliary learning rates must be positive")
        if m.heads not in (1, 3) or m.alpha < 0:
            raise ValueError("pm_heads must be 1 or 3; pm_alpha must be nonnegative")
        if m.update_mode not in ("frozen", "episode_boundary"):
            raise ValueError("Invalid pm_update_mode")
        if not 0 < d.train_fraction < 1 or not 0 < d.validation_fraction < 1:
            raise ValueError("Dataset split fractions must be in (0, 1)")
        if d.train_fraction + d.validation_fraction >= 1:
            raise ValueError("Leave a nonempty test fraction")
        if r.device not in ("auto", "cpu", "cuda") or r.seed < 0:
            raise ValueError("Invalid device or seed")
        if not r.output_root.strip():
            raise ValueError("output_root cannot be empty")
        for section, enabled in ((w, w.enabled), (m, m.enabled)):
            if enabled and not section.checkpoint:
                raise ValueError("Enabled auxiliary models require a checkpoint")
        return self

    def with_overrides(self, overrides: Mapping[str, Any]) -> ExperimentConfig:
        result = self
        for dotted_key, value in overrides.items():
            if not isinstance(dotted_key, str) or dotted_key.count(".") != 1:
                raise ValueError("Expected a config override named section.field")
            group, name = dotted_key.split(".", 1)
            if (group not in {item.name for item in fields(self)}
                    or name not in {item.name for item in fields(getattr(self, group))}):
                raise ValueError("Unknown config override: " + dotted_key)
            result = replace(result, **{group: replace(getattr(result, group), **{name: value})})
        return result.validate()

    @classmethod
    def load(cls, path: str | Path | None = None, *, validate: bool = True) -> ExperimentConfig:
        if path is None:
            return cls().validate()
        with Path(path).open(encoding="utf-8-sig") as stream:
            raw = json.load(stream)
        if not isinstance(raw, dict):
            raise ValueError("Configuration root must be an object")
        defaults = cls()
        unknown = set(raw) - {f.name for f in fields(defaults)}
        if unknown:
            raise ValueError("Unknown config sections: " + ", ".join(sorted(unknown)))
        sections = {}
        for name, values in raw.items():
            section_type = type(getattr(defaults, name))
            if not isinstance(values, dict):
                raise ValueError(name + " must be an object")
            unknown = set(values) - {f.name for f in fields(section_type)}
            if unknown:
                raise ValueError("Unknown {} options: {}".format(name, sorted(unknown)))
            sections[name] = section_type(**values)
        result = cls(**sections)
        return result.validate() if validate else result
