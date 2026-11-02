"""Dependency-injected data flow. No model optimizer is hidden in the simulator."""

from __future__ import annotations

import math
from collections import deque
from typing import Callable

from ..config import ExperimentConfig
from ..contracts import AgentHistory, LocalHistoryBatch, SCHEMA_VERSION, TransitionRecord
from ..data.buffers import (
    RolloutBatch, RolloutBuffer, validate_action_batch, validate_candidate_features,
    validate_observation,
)
from ..envs.base import ResearchEnvironment
from ..models.base import ProgressModel, WorldModel
from ..models.rewards import PotentialRewardShaper
from ..policies.base import RoutingPolicy


class EpisodeRunner:
    def __init__(self, config: ExperimentConfig, environment: ResearchEnvironment,
                 policy: RoutingPolicy, world_model: WorldModel, progress_model: ProgressModel):
        self.config, self.environment, self.policy = config, environment, policy
        self.world_model, self.progress_model = world_model, progress_model
        self.shaper = PotentialRewardShaper(
            config.policy.gamma, config.progress.alpha if config.progress.enabled else 0.0
        )
        self._started = False

    def reset(self, episode_id: str, seed: int) -> None:
        if self._started and not self.terminated:
            raise ValueError("Do not reset an unfinished task at a rollout boundary")
        self.observation = self.environment.reset(seed, episode_id)
        self.policy.reset(seed + 1)
        self.history = deque([self.observation], maxlen=self.config.world_model.history_len)
        self.action_history = deque(maxlen=self.config.world_model.history_len - 1)
        self.initial_count = self.environment.export_state().generated
        self.phi = self.progress_model.team_potential(self.environment.packet_traces(), self.initial_count)
        self._task_pm_version = self.progress_model.version
        self.terminated = False
        self._started = True

    def local_history(self) -> LocalHistoryBatch:
        observations = tuple(self.history)
        actions = tuple(self.action_history)
        return LocalHistoryBatch(self.observation.episode_id, self.observation.env_step, tuple(
            AgentHistory(
                self.observation.episode_id, agent,
                tuple(obs.env_step for obs in observations),
                tuple(obs.agents[agent] for obs in observations),
                tuple(batch[agent] for batch in actions),
            ) for agent in range(len(self.observation.agents))
        ))

    def collect_rollout(self) -> RolloutBatch:
        if not self._started or self.terminated:
            raise RuntimeError("Reset a new task before collecting")
        buffer = RolloutBuffer(self.config.policy.rollout_length)
        wm_version, policy_version = self.world_model.version, self.policy.version
        for index in range(self.config.policy.rollout_length):
            if self.progress_model.version != self._task_pm_version:
                raise ValueError("Progress model version changed inside a task")
            if self.world_model.version != wm_version or self.policy.version != policy_version:
                raise ValueError("Policy/WM version changed inside a rollout")
            state_t = self.environment.export_state()
            obs_t = self.observation
            validate_observation(obs_t, self.config.environment.num_nodes)
            features = self.world_model.predict_candidates(self.local_history())
            validate_candidate_features(features, len(obs_t.agents), wm_version)
            actions = self.policy.sample(obs_t, features)
            validate_action_batch(obs_t, actions)
            if actions.policy_version != policy_version:
                raise ValueError("Sampled actions have an unexpected policy version")
            evaluated = self.policy.evaluate_actions(obs_t, actions, features)
            if (len(evaluated) != len(actions.log_probs)
                    or not all(math.isfinite(new) and math.isclose(old, new, abs_tol=1e-5)
                               for old, new in zip(actions.log_probs, evaluated))):
                raise ValueError("Sampling/evaluate_actions log probabilities disagree")
            result = self.environment.step_joint(actions.actions)
            if result.executed_actions != actions.actions:
                raise ValueError("Environment replaced the sampled actions")
            self.history.append(result.observation)
            self.action_history.append(actions.actions)
            phi_next = self.progress_model.team_potential(
                self.environment.packet_traces(), self.initial_count
            )
            base_reward = result.task_reward + (result.earp_reward if self.config.policy.use_earp else 0.0)
            train_reward, shaping = self.shaper.apply(
                base_reward, self.phi, phi_next, result.env_terminated
            )
            truncated = index == self.config.policy.rollout_length - 1 and not result.env_terminated
            buffer.append(TransitionRecord(
                SCHEMA_VERSION, state_t.env_version, state_t.episode_id, state_t.env_step,
                state_t.seed, state_t, result.state, obs_t, result.observation, actions,
                features, wm_version, self._task_pm_version, result.executed_events,
                result.packet_events, result.task_reward, result.legacy_dense_reward,
                result.earp_reward if self.config.policy.use_earp else 0.0,
                shaping, train_reward, result.env_terminated, truncated,
            ))
            self.phi = 0.0 if result.env_terminated else phi_next
            self.observation, self.terminated = result.observation, result.env_terminated
            if self.terminated:
                break
        if (self.progress_model.version != self._task_pm_version
                or self.world_model.version != wm_version or self.policy.version != policy_version):
            raise ValueError("Model version changed during collection")
        if self.terminated:
            bootstrap = (0.0,) * len(self.observation.agents)
        else:
            features = self.world_model.predict_candidates(self.local_history())
            validate_candidate_features(features, len(self.observation.agents), wm_version)
            bootstrap = self.policy.values(self.observation, features)
        if (self.progress_model.version != self._task_pm_version
                or self.world_model.version != wm_version or self.policy.version != policy_version):
            raise ValueError("Model version changed during bootstrap")
        return buffer.finish(bootstrap)

    def run_episode(self, episode_id: str, seed: int,
                    on_rollout: Callable[[RolloutBatch], None] | None = None
                    ) -> tuple[TransitionRecord, ...]:
        """A trainer may update PPO at each boundary; PM stays fixed for the task."""
        self.reset(episode_id, seed)
        records = []
        while not self.terminated:
            rollout = self.collect_rollout()
            if on_rollout is not None:
                on_rollout(rollout)
            records.extend(rollout.transitions)
        return tuple(records)

    def metrics(self, records: tuple[TransitionRecord, ...], episode: int) -> dict:
        traces = self.environment.packet_traces()
        successful = [trace for trace in traces if trace.outcome is True]
        delays = sorted(trace.end_step - trace.generated_step for trace in successful)
        hops = [sum(step.event in ("forwarded", "delivered") for step in trace.history)
                for trace in successful]
        count = max(len(traces), 1)
        return {
            "episode": episode, "seed": records[0].seed, "environment_steps": len(records),
            "delivery_rate": len(successful) / count,
            "drop_rate": sum(trace.outcome is False for trace in traces) / count,
            "successful_mean_hops": sum(hops) / len(hops) if hops else None,
            "successful_mean_delay_slots": sum(delays) / len(delays) if delays else None,
            "successful_p95_delay_slots": delays[math.ceil(0.95 * len(delays)) - 1] if delays else None,
            "task_return": sum(record.task_reward for record in records),
            "earp_return": sum(record.earp_reward for record in records),
            "shaping_return": sum(record.shaping_reward for record in records),
            "train_return": sum(record.train_reward for record in records),
        }

    def close(self) -> None:
        self.environment.close()
