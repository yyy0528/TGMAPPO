"""Shared local graph Actor and joint-observation centralized Critic.

This feedforward baseline needs PyTorch only. No PyG or legacy simulator import
is required. The Critic aggregates observable information from all agents; its
global context never enters the Actor's logits.
"""

from __future__ import annotations

import pickle

import torch
from torch import nn
from torch.distributions import Categorical

from ..config import ExperimentConfig
from ..contracts import ActionBatch, CandidateFeatures, ObservationBatch
from ..data.buffers import (
    validate_action_batch, validate_candidate_features, validate_observation,
)
from ..runtime.checkpoints import CheckpointMetadata
from .base import RoutingPolicy


class LocalGraphEncoder(nn.Module):
    def __init__(self, hidden_size: int, layers: int):
        super().__init__()
        self.input = nn.Linear(4, hidden_size)
        self.layers = nn.ModuleList(nn.Linear(2 * hidden_size, hidden_size) for _ in range(layers))

    def forward(self, nodes, adjacency, visible):
        mask = visible.unsqueeze(-1).to(nodes.dtype)
        links = (adjacency > 0) & visible.unsqueeze(-1) & visible.unsqueeze(-2)
        identity = torch.eye(nodes.shape[-2], dtype=torch.bool, device=nodes.device)
        links = links | (identity & visible.unsqueeze(-1))
        weights = links.to(nodes.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1.0)
        hidden = torch.relu(self.input(nodes * mask)) * mask
        for layer in self.layers:
            messages = torch.bmm(weights, hidden)
            hidden = torch.relu(layer(torch.cat((hidden, messages), -1))) * mask
        return hidden


class GraphActorCritic(nn.Module):
    def __init__(self, hidden_size: int, layers: int):
        super().__init__()
        self.actor_encoder = LocalGraphEncoder(hidden_size, layers)
        self.critic_encoder = LocalGraphEncoder(hidden_size, layers)
        self.wait_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.actor_head = nn.Sequential(nn.Linear(2 * hidden_size + 10, hidden_size),
                                        nn.Tanh(), nn.Linear(hidden_size, 1))
        self.value_head = nn.Sequential(nn.Linear(3 * hidden_size + 4, hidden_size),
                                        nn.Tanh(), nn.Linear(hidden_size, 1))

    def forward(self, nodes, adjacency, visible, context, candidates, available):
        batch, agents, entities, _ = nodes.shape
        flat_nodes = nodes.reshape(batch * agents, entities, 4)
        flat_adj = adjacency.reshape(batch * agents, entities, entities)
        flat_visible = visible.reshape(batch * agents, entities)
        encoded = self.actor_encoder(flat_nodes, flat_adj, flat_visible)
        indices = torch.arange(agents, device=nodes.device).repeat(batch)
        rows = torch.arange(batch * agents, device=nodes.device)
        ego = encoded[rows, indices]
        targets = torch.cat((encoded, self.wait_embedding.expand(batch * agents, 1, -1)), 1)
        actor_context = context.reshape(batch * agents, 4).unsqueeze(1).expand(-1, entities + 1, -1)
        actor_input = torch.cat((ego.unsqueeze(1).expand_as(targets), targets, actor_context,
                                 candidates.reshape(batch * agents, entities + 1, 6)), -1)
        logits = self.actor_head(actor_input).squeeze(-1).reshape(batch, agents, entities + 1)
        logits = logits.masked_fill(~available, float("-inf"))

        critic = self.critic_encoder(flat_nodes, flat_adj, flat_visible)
        pooled = critic.sum(1) / flat_visible.sum(1, keepdim=True).clamp(min=1)
        local = pooled.reshape(batch, agents, -1)
        joint = local.mean(1, keepdim=True).expand_as(local)
        critic_input = torch.cat((critic[rows, indices].reshape_as(local), local, joint, context), -1)
        values = self.value_head(critic_input).squeeze(-1)
        return logits, values


class GraphMAPPOPolicy(RoutingPolicy):
    architecture_version = "local_actor_joint_observation_critic_v1"

    def __init__(self, config: ExperimentConfig, device: str):
        self.config, self.device = config, torch.device(device)
        self.num_nodes = config.environment.num_nodes
        self.updates = 0
        self.version = "graph_mappo_v1:0" + (":greedy" if config.policy.deterministic else "")
        # Model initialization is reproducible without consuming the caller's CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.runtime.seed)
            self.network = GraphActorCritic(config.policy.hidden_size, config.policy.graph_layers).to(self.device)
        self.network.eval()
        self.generator = torch.Generator(device=self.device)
        self.reset(config.runtime.seed)
        if config.policy.checkpoint:
            self.load_checkpoint(config.policy.checkpoint)

    def load_checkpoint(self, path):
        """Validate the complete payload before changing any network parameter."""
        try:
            try:
                payload = torch.load(path, map_location=self.device, weights_only=True)
            except Exception as exc:
                # Limit the broad exception boundary to decoding untrusted file bytes.
                raise ValueError("Cannot decode checkpoint: {}".format(exc)) from None
            if not isinstance(payload, dict):
                raise ValueError("Expected a checkpoint object")
            metadata = CheckpointMetadata(**payload["metadata"])
            metadata.validate(self.config, "policy")
            if (payload["architecture_version"] != self.architecture_version
                    or payload["hidden_size"] != self.config.policy.hidden_size
                    or payload["graph_layers"] != self.config.policy.graph_layers):
                raise ValueError("Incompatible policy architecture/hidden_size/graph_layers")
            updates = payload["updates"]
            if (type(updates) is not int or updates < 0
                    or payload["model_version"] != "graph_mappo_v1:{}".format(updates)
                    or metadata.model_version != payload["model_version"]):
                raise ValueError("Inconsistent policy version/update count")
            state = payload["model_state"]
            expected = self.network.state_dict()
            if not isinstance(state, dict) or set(state) != set(expected):
                raise ValueError("Incompatible policy parameter names")
            for name, value in state.items():
                if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                        or value.dtype != expected[name].dtype or value.layout != expected[name].layout
                        or not torch.isfinite(value).all()):
                    raise ValueError("Invalid policy tensor: " + name)
            self.network.load_state_dict(state, strict=True)
            self.version = payload["model_version"] + (":greedy" if self.config.policy.deterministic else "")
            self.updates = updates
            self.network.eval()
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
            raise ValueError("Invalid graph policy checkpoint {}: {}".format(path, exc)) from None

    def reset(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def _inputs(self, observations, features):
        if not observations or len(observations) != len(features):
            raise ValueError("Expected complete observation/feature batches")
        for observation, predictions in zip(observations, features):
            validate_observation(observation, self.num_nodes)
            validate_candidate_features(predictions, self.num_nodes,
                                        "disabled" if predictions is None else predictions.model_version)
            if any(len(local.goal_offset) != 2 or len(local.head_packet_features) != 2
                   for local in observation.agents):
                raise ValueError("Graph policy requires two goal and two head-packet features")
        nodes = torch.tensor([[a.node_features for a in o.agents] for o in observations],
                             dtype=torch.float32, device=self.device)
        adjacency = torch.tensor([[a.adjacency for a in o.agents] for o in observations],
                                 dtype=torch.float32, device=self.device)
        visible = torch.tensor([[a.visible_mask for a in o.agents] for o in observations],
                               dtype=torch.bool, device=self.device)
        context = torch.tensor([[(*a.goal_offset, *a.head_packet_features) for a in o.agents]
                                for o in observations], dtype=torch.float32, device=self.device)
        available = torch.tensor([[a.available_actions for a in o.agents] for o in observations],
                                 dtype=torch.bool, device=self.device)
        candidates = torch.zeros((len(observations), self.num_nodes, self.num_nodes + 1, 6),
                                 dtype=torch.float32, device=self.device)
        for index, predictions in enumerate(features):
            if predictions is not None:
                candidates[index] = torch.tensor(predictions.values, dtype=torch.float32, device=self.device)
        return nodes, adjacency, visible, context, candidates, available

    def distribution(self, observations, features):
        logits, values = self.network(*self._inputs(observations, features))
        # Remove a large common offset before Categorical normalizes in float32.
        # Otherwise sampled probabilities and reported log probabilities can differ.
        logits = logits - logits.amax(dim=-1, keepdim=True)
        if self.config.policy.deterministic:
            chosen = logits.argmax(-1, keepdim=True)
            logits = torch.full_like(logits, float("-inf")).scatter(-1, chosen, 0.0)
        return Categorical(logits=logits), values

    @torch.no_grad()
    def sample(self, observation: ObservationBatch, features: CandidateFeatures | None) -> ActionBatch:
        distribution, values = self.distribution((observation,), (features,))
        actions = (distribution.probs[0].argmax(-1) if self.config.policy.deterministic
                   else torch.multinomial(distribution.probs[0], 1, generator=self.generator).squeeze(-1))
        log_probs = distribution.log_prob(actions.unsqueeze(0))[0]
        return ActionBatch(tuple(actions.cpu().tolist()), tuple(log_probs.cpu().tolist()),
                           tuple(values[0].cpu().tolist()), self.version)

    @torch.no_grad()
    def evaluate_actions(self, observation: ObservationBatch, action_batch: ActionBatch,
                         features: CandidateFeatures | None) -> tuple[float, ...]:
        validate_action_batch(observation, action_batch)
        if action_batch.policy_version != self.version:
            raise ValueError("Cannot replay public action probabilities with a different policy version")
        distribution, _ = self.distribution((observation,), (features,))
        actions = torch.tensor([action_batch.actions], dtype=torch.long, device=self.device)
        return tuple(distribution.log_prob(actions)[0].cpu().tolist())

    @torch.no_grad()
    def values(self, observation: ObservationBatch, features: CandidateFeatures | None) -> tuple[float, ...]:
        _, values = self.distribution((observation,), (features,))
        return tuple(values[0].cpu().tolist())
