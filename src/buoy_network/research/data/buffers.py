from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

from ..contracts import (
    ActionBatch, CANDIDATE_FEATURES, CandidateFeatures, NODE_FEATURES,
    ObservationBatch, SCHEMA_VERSION, TransitionRecord,
)


def validate_observation(observation: ObservationBatch, num_nodes: int) -> None:
    if num_nodes < 1 or len(observation.agents) != num_nodes:
        raise ValueError("Incorrect observation batch length")
    for index, local in enumerate(observation.agents):
        if local.agent_id != index:
            raise ValueError("Observation agent order does not match global node indices")
        if (len(local.node_features) != num_nodes or len(local.visible_mask) != num_nodes
                or len(local.adjacency) != num_nodes
                or len(local.available_actions) != num_nodes + 1
                or not any(local.available_actions)):
            raise ValueError("Invalid local observation/mask shape")
        if any(len(row) != num_nodes for row in local.adjacency):
            raise ValueError("Invalid adjacency shape")
        if any(len(row) != len(NODE_FEATURES) for row in local.node_features):
            raise ValueError("Invalid node feature shape")
        numbers = (*local.goal_offset, *local.head_packet_features,
                   *(value for row in local.node_features for value in row),
                   *(value for row in local.adjacency for value in row))
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Nonfinite observation")


def validate_action_batch(observation: ObservationBatch, batch: ActionBatch) -> None:
    n = len(observation.agents)
    if any(len(values) != n for values in (batch.actions, batch.log_probs, batch.values)):
        raise ValueError("Incorrect action/log-probability/value batch length")
    if not isinstance(batch.policy_version, str) or not batch.policy_version:
        raise ValueError("A policy version is required")
    for local, action in zip(observation.agents, batch.actions):
        if (type(action) is not int or not 0 <= action <= n
                or not local.available_actions[action]):
            raise ValueError("Action violates its decision-time mask")
    if not all(math.isfinite(value) for value in (*batch.log_probs, *batch.values)):
        raise ValueError("Nonfinite action log probability or value")
    for states in (batch.actor_states, batch.critic_states):
        if states and (len(states) != n or not all(
                math.isfinite(value) for row in states for value in row)):
            raise ValueError("Invalid pre-sampling recurrent states")


def validate_candidate_features(features: CandidateFeatures | None, num_nodes: int,
                                model_version: str) -> None:
    if features is None:
        if model_version != "disabled":
            raise ValueError("An enabled world model must return candidate features")
        return
    if features.model_version != model_version or len(features.values) != num_nodes:
        raise ValueError("World model feature version/shape mismatch")
    for candidates in features.values:
        if len(candidates) != num_nodes + 1:
            raise ValueError("Expected N+1 candidate predictions")
        for vector in candidates:
            if (len(vector) != len(CANDIDATE_FEATURES)
                    or not all(math.isfinite(value) for value in vector)):
                raise ValueError("Invalid candidate feature vector")


def validate_transition(record: TransitionRecord) -> None:
    if record.schema_version != SCHEMA_VERSION:
        raise ValueError("Incompatible transition schema")
    if record.env_terminated and record.rollout_truncated:
        raise ValueError("A real terminal transition is not a sampling truncation")
    for state, obs, step in (
        (record.state_t, record.obs_t, record.env_step),
        (record.state_tp1, record.obs_tp1, record.env_step + 1),
    ):
        if (state.episode_id != record.episode_id or obs.episode_id != record.episode_id
                or state.env_step != step or obs.env_step != step
                or state.env_version != record.env_version or state.seed != record.seed):
            raise ValueError("Snapshot/observation time or episode mismatch")
        if state.generated != len(state.packets) + state.delivered + state.dropped:
            raise ValueError("Packet conservation failed")
    n = len(record.obs_t.agents)
    for observation, state in ((record.obs_t, record.state_t), (record.obs_tp1, record.state_tp1)):
        validate_observation(observation, n)
        if len(state.nodes) != n or any(type(value) is not int or value < 0 for value in (
                state.generated, state.delivered, state.dropped)):
            raise ValueError("Invalid state node count or packet counters")
    validate_action_batch(record.obs_t, record.action_batch)
    validate_candidate_features(record.wm_features, n, record.wm_version)
    scalars = (record.task_reward, record.train_reward,
               record.legacy_dense_reward, record.earp_reward, record.shaping_reward)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("Nonfinite rollout data")
    if not math.isclose(record.train_reward,
                        record.task_reward + record.earp_reward + record.shaping_reward,
                        abs_tol=1e-7):
        raise ValueError("Training reward components do not add up")


@dataclass(frozen=True)
class RolloutBatch:
    transitions: tuple[TransitionRecord, ...]
    bootstrap_values: tuple[float, ...]

    def compute_gae(self, gamma: float, gae_lambda: float
                    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
        """Use shaped team rewards; bootstrap a nonterminal sampling boundary."""
        if not self.transitions:
            raise ValueError("Cannot compute GAE on an empty rollout")
        if (not math.isfinite(gamma) or not 0 < gamma <= 1
                or not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1):
            raise ValueError("Invalid GAE gamma or lambda")
        n = len(self.transitions[0].action_batch.values)
        if (len(self.bootstrap_values) != n
                or not all(math.isfinite(value) for value in self.bootstrap_values)):
            raise ValueError("Invalid bootstrap values")
        if self.transitions[-1].env_terminated and any(self.bootstrap_values):
            raise ValueError("Terminal rollouts require zero bootstrap")
        for transition in self.transitions:
            validate_transition(transition)
        advantages = [(0.0,) * n for _ in self.transitions]
        next_values, next_advantages = self.bootstrap_values, (0.0,) * n
        for index in range(len(self.transitions) - 1, -1, -1):
            transition = self.transitions[index]
            continuation = 0.0 if transition.env_terminated else 1.0
            advantages[index] = tuple(
                transition.train_reward + gamma * continuation * next_values[agent]
                - transition.action_batch.values[agent]
                + gamma * gae_lambda * continuation * next_advantages[agent]
                for agent in range(n)
            )
            next_values, next_advantages = transition.action_batch.values, advantages[index]
        returns = tuple(tuple(advantage + value for advantage, value in zip(row, t.action_batch.values))
                        for row, t in zip(advantages, self.transitions))
        return tuple(advantages), returns


class RolloutBuffer:
    """Exactly T transitions; initial and next observations live in each record."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("Rollout capacity must be positive")
        self.capacity = capacity
        self._records: list[TransitionRecord] = []

    def append(self, record: TransitionRecord) -> None:
        validate_transition(record)
        if len(self._records) >= self.capacity:
            raise ValueError("Rollout overflow; no modulo overwrite is allowed")
        if self._records:
            previous = self._records[-1]
            if previous.env_terminated or previous.rollout_truncated:
                raise ValueError("A rollout cannot continue beyond its boundary")
            if previous.state_tp1 != record.state_t or previous.obs_tp1 != record.obs_t:
                raise ValueError("Discontinuous rollout")
            if (previous.action_batch.policy_version != record.action_batch.policy_version
                    or previous.wm_version != record.wm_version
                    or previous.pm_version != record.pm_version):
                raise ValueError("Do not change model versions inside a rollout")
        self._records.append(record)

    def finish(self, bootstrap_values: tuple[float, ...]) -> RolloutBatch:
        if not self._records:
            raise ValueError("Cannot finish an empty rollout")
        if not (self._records[-1].env_terminated or self._records[-1].rollout_truncated):
            raise ValueError("Rollout must end with an explicit boundary")
        n = len(self._records[-1].obs_tp1.agents)
        if len(bootstrap_values) != n or not all(math.isfinite(v) for v in bootstrap_values):
            raise ValueError("Invalid bootstrap values")
        if self._records[-1].env_terminated and any(bootstrap_values):
            raise ValueError("Terminal rollouts require zero bootstrap")
        return RolloutBatch(tuple(self._records), tuple(bootstrap_values))


class EpisodeReplay:
    """Auxiliary-model replay evicts whole episodes and never mixes PPO history."""

    def __init__(self, capacity_steps: int):
        if capacity_steps < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity_steps = capacity_steps
        self._episodes: deque[tuple[TransitionRecord, ...]] = deque()
        self.steps = 0

    def add_episode(self, records: tuple[TransitionRecord, ...]) -> None:
        if not records or not records[-1].env_terminated:
            raise ValueError("Replay expects a complete finite task episode")
        if len(records) > self.capacity_steps:
            raise ValueError("Episode exceeds replay capacity")
        episode_id = records[0].episode_id
        if any(episode[0].episode_id == episode_id for episode in self._episodes):
            raise ValueError("Duplicate episode ID")
        for index, record in enumerate(records):
            validate_transition(record)
            if record.episode_id != episode_id or record.env_step != index:
                raise ValueError("Replay episodes must start at t=0 and be continuous")
            if index < len(records) - 1 and record.env_terminated:
                raise ValueError("Terminal transition before the end of an episode")
            if index and (records[index - 1].state_tp1 != record.state_t
                          or records[index - 1].obs_tp1 != record.obs_t):
                raise ValueError("Discontinuous episode states")
        while self._episodes and self.steps + len(records) > self.capacity_steps:
            self.steps -= len(self._episodes.popleft())
        self._episodes.append(tuple(records))
        self.steps += len(records)

    def sample_window(self, length: int, rng: random.Random) -> tuple[TransitionRecord, ...]:
        if length < 1:
            raise ValueError("Window length must be positive")
        eligible = [episode for episode in self._episodes if len(episode) >= length]
        if not eligible:
            raise ValueError("No complete episode is long enough")
        episode = rng.choice(eligible)
        start = rng.randrange(len(episode) - length + 1)
        return episode[start:start + length]
