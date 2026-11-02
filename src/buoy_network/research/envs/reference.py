"""Small executable contract reference, not the final research simulator.

One packet starts at each non-Sink node; links use a distance radius and IID
loss. Queues are unbounded, transmissions do not collide, and mobility is an
optional Gaussian drift. The Sink coordinates are a known destination beacon.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from ..config import EnvironmentConfig
from ..contracts import (
    ExecutedEvent, LocalObservation, NodeState, ObservationBatch, PacketEvent,
    PacketKey, PacketState, PacketStep, PacketTrace, StateSnapshot, StepResult,
)
from .base import ResearchEnvironment


@dataclass
class _Packet:
    key: PacketKey
    source: int
    destination: int
    node_id: int
    ttl: int
    hops: int = 0
    history: list[PacketStep] = field(default_factory=list)


class ReferenceGridEnvironment(ResearchEnvironment):
    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        self.n = config.num_nodes
        self.sink = self.n - 1
        self._observation: ObservationBatch | None = None

    def reset(self, seed: int, episode_id: str) -> ObservationBatch:
        self.rng = random.Random(seed)
        self.seed, self.episode_id, self.step = seed, episode_id, 0
        self.positions = [
            (float(i), float(j))
            for i in range(self.config.grid_size) for j in range(self.config.grid_size)
        ]
        self.queues: list[list[int]] = [[] for _ in range(self.n)]
        self.packets: dict[int, _Packet] = {}
        self.finished: dict[int, PacketTrace] = {}
        self.delivered = self.dropped = 0
        for node in range(self.n - 1):
            self.packets[node] = _Packet(
                PacketKey(episode_id, node), node, self.sink, node, self.config.task_ttl
            )
            self.queues[node].append(node)
        self.generated = len(self.packets)
        self.terminated = False
        self._observation = self._build_observation()
        return self._observation

    def _distance(self, first: int, second: int) -> float:
        return math.dist(self.positions[first], self.positions[second])

    def _build_observation(self) -> ObservationBatch:
        observations = []
        for agent in range(self.n):
            visible = tuple(
                self._distance(agent, other) <= self.config.comm_range
                for other in range(self.n)
            )
            active = bool(self.queues[agent]) and agent != self.sink
            mask = tuple(
                active and visible[other] and agent != other for other in range(self.n)
            ) + (True,)
            x, y = self.positions[agent]
            node_features = tuple(
                (self.positions[j][0] - x, self.positions[j][1] - y,
                 len(self.queues[j]) / self.generated, float(j == self.sink))
                if visible[j] else (0.0, 0.0, 0.0, 0.0)
                for j in range(self.n)
            )
            adjacency = tuple(tuple(
                float(i != j and visible[i] and visible[j]
                      and self._distance(i, j) <= self.config.comm_range)
                for j in range(self.n)) for i in range(self.n))
            head = self.packets[self.queues[agent][0]] if active else None
            observations.append(LocalObservation(
                agent, node_features, adjacency, visible, mask, active,
                (self.positions[self.sink][0] - x, self.positions[self.sink][1] - y),
                (head.ttl / self.config.task_ttl, head.hops / self.n) if head else (0.0, 0.0),
            ))
        return ObservationBatch(self.episode_id, self.step, tuple(observations))

    def observe(self) -> ObservationBatch:
        if self._observation is None:
            raise RuntimeError("Call reset before observing the environment")
        return self._observation

    def export_state(self) -> StateSnapshot:
        self.observe()
        return StateSnapshot(
            self.config.version, self.episode_id, self.step, self.seed,
            tuple(NodeState(i, tuple(pos), tuple(self.queues[i]))
                  for i, pos in enumerate(self.positions)),
            tuple(PacketState(p.key, p.source, p.destination, p.node_id, p.ttl, p.hops)
                  for _, p in sorted(self.packets.items())),
            self.generated, self.delivered, self.dropped,
        )

    def _finish(self, packet_id: int, outcome: bool, reason: str) -> PacketEvent:
        packet = self.packets.pop(packet_id)
        if packet_id in self.queues[packet.node_id]:
            self.queues[packet.node_id].remove(packet_id)
        self.finished[packet_id] = PacketTrace(
            packet.key, packet.source, packet.destination, 0, tuple(packet.history),
            self.step, outcome, reason,
        )
        if outcome:
            self.delivered += 1
        else:
            self.dropped += 1
        return PacketEvent(packet.key, self.step, "delivered" if outcome else "dropped",
                           reason, packet.node_id)

    def step_joint(self, actions: Sequence[int]) -> StepResult:
        observation = self.observe()
        if self.terminated:
            raise RuntimeError("Task has terminated; call reset")
        if len(actions) != self.n:
            raise ValueError("Expected one action per node")
        actions = tuple(actions)
        for local, action in zip(observation.agents, actions):
            if type(action) is not int or not 0 <= action <= self.n:
                raise ValueError("Actions must be integer global node indices or WAIT")
            if not local.available_actions[action]:
                raise ValueError("Illegal action {} for node {}".format(action, local.agent_id))

        # Fix the service set before any queue is changed.
        start = {key: (p.node_id, p.ttl, p.hops) for key, p in self.packets.items()}
        heads = {i: queue[0] for i, queue in enumerate(self.queues) if queue}
        outcomes: dict[int, str] = {}
        arrivals = []
        delivered = []
        executed = []
        for sender, packet_id in heads.items():
            packet, receiver = self.packets[packet_id], actions[sender]
            attempted = receiver != self.n
            success = attempted and self.rng.random() >= self.config.loss_probability
            executed.append(ExecutedEvent(packet.key, sender, receiver, attempted, success))
            outcomes[packet_id] = "waited" if not attempted else "failed"
            if success:
                self.queues[sender].pop(0)
                packet.node_id = receiver
                packet.hops += 1
                outcomes[packet_id] = "forwarded"
                if receiver == self.sink:
                    delivered.append(packet_id)
                    outcomes[packet_id] = "delivered"
                else:
                    arrivals.append((receiver, packet_id))
        for receiver, packet_id in arrivals:
            self.queues[receiver].append(packet_id)

        for packet_id, (node, ttl, hops) in start.items():
            local = observation.agents[node]
            packet = self.packets[packet_id]
            packet.history.append(PacketStep(
                self.step, node, actions[node] if heads.get(node) == packet_id else self.n,
                outcomes.get(packet_id, "queued"),
                (ttl / self.config.task_ttl, self.step / self.config.task_horizon,
                 hops / self.n, sum(local.available_actions[:-1]) / self.n,
                 local.node_features[node][2], math.hypot(*local.goal_offset)),
            ))

        self.step += 1
        events = [self._finish(packet_id, True, "sink") for packet_id in delivered]
        for packet_id in list(self.packets):
            packet = self.packets[packet_id]
            packet.ttl -= 1
            if packet.ttl <= 0:
                events.append(self._finish(packet_id, False, "ttl"))
        if self.step >= self.config.task_horizon:
            for packet_id in list(self.packets):
                events.append(self._finish(packet_id, False, "task_deadline"))

        if self.config.mobility_std:
            count = self.config.num_moving_buoys
            moving = [(i * self.n) // count for i in range(count)] if count else []
            bound = float(self.config.grid_size - 1)
            for node in moving:
                self.positions[node] = tuple(
                    min(bound, max(0.0, coordinate + self.rng.gauss(0, self.config.mobility_std)))
                    for coordinate in self.positions[node]
                )
        self.terminated = not self.packets
        reward = sum(self.config.delivery_reward if event.kind == "delivered"
                     else self.config.drop_reward for event in events)
        if self.config.reward_mode == "task_dense":
            reward += self.config.wait_cost * len(start)
        self._observation = self._build_observation()
        return StepResult(
            self._observation, self.export_state(), actions, tuple(executed),
            tuple(events), reward, self.terminated,
        )

    def packet_traces(self) -> tuple[PacketTrace, ...]:
        traces = dict(self.finished)
        traces.update({key: PacketTrace(
            packet.key, packet.source, packet.destination, 0, tuple(packet.history)
        ) for key, packet in self.packets.items()})
        return tuple(trace for _, trace in sorted(traces.items()))
