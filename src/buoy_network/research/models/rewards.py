"""Potential differencing is independent of the learned prefix scorer."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PotentialRewardShaper:
    gamma: float
    alpha: float

    def __post_init__(self):
        if not 0 < self.gamma <= 1 or not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("Invalid shaping gamma/alpha")

    def apply(self, task_reward: float, phi_t: float, phi_tp1: float,
              env_terminated: bool) -> tuple[float, float]:
        if not all(math.isfinite(v) for v in (task_reward, phi_t, phi_tp1)):
            raise ValueError("Nonfinite reward or potential")
        next_phi = 0.0 if env_terminated else phi_tp1
        shaping = self.alpha * (self.gamma * next_phi - phi_t)
        return task_reward + shaping, shaping
