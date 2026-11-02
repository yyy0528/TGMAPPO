"""Checkpoint compatibility boundary; tensor/optimizer payloads belong to trainers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import ExperimentConfig
from ..contracts import CANDIDATE_FEATURES, NODE_FEATURES, SCHEMA_VERSION


@dataclass(frozen=True)
class CheckpointMetadata:
    component: str
    model_version: str
    schema_version: int
    env_version: str
    environment_backend: str
    num_nodes: int
    num_actions: int
    node_features: tuple[str, ...]
    candidate_features: tuple[str, ...]
    observation_schema: str = "local_graph_v1"
    action_schema: str = "global_node_wait_v1"

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (
                self.component, self.model_version, self.env_version, self.environment_backend,
                self.observation_schema, self.action_schema)):
            raise ValueError("Checkpoint names and versions must be nonempty strings")
        if any(type(value) is not int or value < 1 for value in (
                self.schema_version, self.num_nodes, self.num_actions)):
            raise ValueError("Checkpoint schema and dimensions must be positive integers")
        if any(not isinstance(names, tuple) or not names or any(
                not isinstance(name, str) or not name for name in names)
               for names in (self.node_features, self.candidate_features)):
            raise ValueError("Checkpoint feature definitions must be tuples of names")

    @classmethod
    def for_config(cls, config: ExperimentConfig, component: str,
                   model_version: str) -> CheckpointMetadata:
        return cls(component, model_version, SCHEMA_VERSION, config.environment.version,
                   config.environment.backend, config.environment.num_nodes,
                   config.environment.num_nodes + 1, NODE_FEATURES, CANDIDATE_FEATURES)

    def validate(self, config: ExperimentConfig, component: str) -> None:
        expected = self.for_config(config, component, self.model_version)
        differences = [name for name, value in asdict(expected).items()
                       if getattr(self, name) != value]
        if differences:
            raise ValueError("Incompatible checkpoint: " + ", ".join(differences))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(asdict(self), stream, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path: str | Path) -> CheckpointMetadata:
        with Path(path).open(encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, dict):
            raise ValueError("Checkpoint metadata must be an object")
        try:
            for name in ("node_features", "candidate_features"):
                if not isinstance(raw[name], list):
                    raise ValueError("Checkpoint feature definitions must be lists of names")
                raw[name] = tuple(raw[name])
            return cls(**raw)
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid checkpoint metadata: " + str(exc)) from None
