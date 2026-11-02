from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts import ActionBatch, CandidateFeatures, ObservationBatch


class RoutingPolicy(ABC):
    version: str

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Reset policy RNG/recurrent state at a true task boundary."""

    @abstractmethod
    def sample(self, observation: ObservationBatch,
               features: CandidateFeatures | None) -> ActionBatch:
        """Sample final actions; all distribution inputs must be replayable."""

    @abstractmethod
    def evaluate_actions(self, observation: ObservationBatch, action_batch: ActionBatch,
                         features: CandidateFeatures | None) -> tuple[float, ...]:
        """Recompute log probabilities with cached inputs and pre-sampling RNN states.

        This operation must not advance the live policy's recurrent state or RNG.
        """

    @abstractmethod
    def values(self, observation: ObservationBatch,
               features: CandidateFeatures | None) -> tuple[float, ...]:
        """Bootstrap value; must not advance live recurrent state or RNG."""
