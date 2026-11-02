import math
import random

from ..contracts import ActionBatch
from .base import RoutingPolicy


class RulePolicy(RoutingPolicy):
    """Legal random or deterministic geographic routing; no learning is claimed."""

    def __init__(self, mode: str) -> None:
        if mode not in ("random", "geographic"):
            raise ValueError("Unknown rule policy: " + str(mode))
        self.mode = mode
        self.version = mode + "_v1"
        self.reset(0)

    def reset(self, seed):
        self.rng = random.Random(seed)

    def _distribution(self, local):
        legal = [i for i, valid in enumerate(local.available_actions) if valid]
        if self.mode == "random":
            return {action: 1.0 / len(legal) for action in legal}
        wait = len(local.available_actions) - 1
        candidates = [action for action in legal if action != wait]
        if not candidates:
            return {wait: 1.0}
        gx, gy = local.goal_offset
        # Only local relative coordinates and the declared destination beacon.
        action = min(candidates, key=lambda node: (
            math.hypot(local.node_features[node][0] - gx,
                       local.node_features[node][1] - gy), node
        ))
        return {action: 1.0}

    def sample(self, observation, features):
        distributions = [self._distribution(local) for local in observation.agents]
        actions = tuple(self.rng.choice(list(dist)) for dist in distributions)
        return ActionBatch(
            actions, tuple(math.log(dist[action]) for dist, action in zip(distributions, actions)),
            self.values(observation, features), self.version,
        )

    def evaluate_actions(self, observation, action_batch, features):
        actions = action_batch.actions
        if len(actions) != len(observation.agents):
            raise ValueError("Incorrect action batch length")
        result = []
        for local, action in zip(observation.agents, actions):
            probability = self._distribution(local).get(action, 0.0)
            result.append(math.log(probability) if probability else -math.inf)
        return tuple(result)

    def values(self, observation, features):
        return (0.0,) * len(observation.agents)
