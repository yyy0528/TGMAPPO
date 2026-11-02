"""Immutable CPU-side contracts. No tensor or simulator object crosses this boundary.

Each batch represents one environment; vectorization adds an outer batch axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

SCHEMA_VERSION = 1
NODE_FEATURES = ("relative_x", "relative_y", "queue_fraction", "is_sink")
CANDIDATE_FEATURES = (
    "send_probability", "sender_queue_delta", "receiver_queue_delta",
    "delivery_probability", "drop_probability", "prediction_disagreement",
)


@dataclass(frozen=True)
class PacketKey:
    episode_id: str
    packet_id: int


@dataclass(frozen=True)
class NodeState:
    node_id: int
    position: Tuple[float, float]
    queue: Tuple[int, ...]


@dataclass(frozen=True)
class PacketState:
    key: PacketKey
    source: int
    destination: int
    node_id: int
    ttl: int
    hops: int


@dataclass(frozen=True)
class StateSnapshot:
    env_version: str
    episode_id: str
    env_step: int
    seed: int
    nodes: Tuple[NodeState, ...]
    packets: Tuple[PacketState, ...]
    generated: int
    delivered: int
    dropped: int


@dataclass(frozen=True)
class LocalObservation:
    agent_id: int
    node_features: Tuple[Tuple[float, ...], ...]
    adjacency: Tuple[Tuple[float, ...], ...]
    visible_mask: Tuple[bool, ...]
    available_actions: Tuple[bool, ...]
    active: bool
    goal_offset: Tuple[float, float]
    head_packet_features: Tuple[float, ...]


@dataclass(frozen=True)
class ObservationBatch:
    episode_id: str
    env_step: int
    agents: Tuple[LocalObservation, ...]


@dataclass(frozen=True)
class CandidateFeatures:
    # [N, N+1, 6], computed before action sampling and cached verbatim.
    values: Tuple[Tuple[Tuple[float, ...], ...], ...]
    model_version: str


@dataclass(frozen=True)
class ActionBatch:
    actions: Tuple[int, ...]
    log_probs: Tuple[float, ...]
    values: Tuple[float, ...]
    policy_version: str
    # Flattened per-agent PRE-sampling recurrent states; shape belongs to schema.
    actor_states: Tuple[Tuple[float, ...], ...] = ()
    critic_states: Tuple[Tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class AgentHistory:
    episode_id: str
    agent_id: int
    env_steps: Tuple[int, ...]
    observations: Tuple[LocalObservation, ...]
    # Own past actions only; never condition a local WM on other agents' actions.
    past_actions: Tuple[int, ...]


@dataclass(frozen=True)
class LocalHistoryBatch:
    episode_id: str
    env_step: int
    agents: Tuple[AgentHistory, ...]


@dataclass(frozen=True)
class ExecutedEvent:
    key: PacketKey
    sender: int
    receiver: int
    attempted: bool
    success: bool


@dataclass(frozen=True)
class PacketEvent:
    key: PacketKey
    env_step: int
    kind: str
    reason: str
    node_id: int


@dataclass(frozen=True)
class PacketStep:
    env_step: int
    node_id: int
    action: int
    event: str
    # TTL fraction, age, hops, visible exits, queue fraction, goal distance.
    features: Tuple[float, ...]


@dataclass(frozen=True)
class PacketTrace:
    key: PacketKey
    source: int
    destination: int
    generated_step: int
    history: Tuple[PacketStep, ...]
    end_step: int | None = None
    outcome: bool | None = None
    end_reason: str = ""


@dataclass(frozen=True)
class StepResult:
    observation: ObservationBatch
    state: StateSnapshot
    executed_actions: Tuple[int, ...]
    executed_events: Tuple[ExecutedEvent, ...]
    packet_events: Tuple[PacketEvent, ...]
    task_reward: float
    env_terminated: bool
    legacy_dense_reward: float = 0.0
    earp_reward: float = 0.0


@dataclass(frozen=True)
class TransitionRecord:
    schema_version: int
    env_version: str
    episode_id: str
    env_step: int
    seed: int
    state_t: StateSnapshot
    state_tp1: StateSnapshot
    obs_t: ObservationBatch
    obs_tp1: ObservationBatch
    action_batch: ActionBatch
    wm_features: CandidateFeatures | None
    wm_version: str
    pm_version: str
    executed_events: Tuple[ExecutedEvent, ...]
    packet_events: Tuple[PacketEvent, ...]
    task_reward: float
    legacy_dense_reward: float
    earp_reward: float
    shaping_reward: float
    train_reward: float
    env_terminated: bool
    rollout_truncated: bool


@dataclass(frozen=True)
class TrainingReport:
    component: str
    model_version: str
    metrics: dict
    checkpoint: str
