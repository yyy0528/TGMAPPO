from abc import ABC, abstractmethod
from typing import Sequence

from ..contracts import ObservationBatch, PacketTrace, StateSnapshot, StepResult


class ResearchEnvironment(ABC):
    """One joint step is one slot; received packets wait until the next slot."""

    @abstractmethod
    def reset(self, seed: int, episode_id: str) -> ObservationBatch:
        """Start a finite task and return its decision-time observation."""

    @abstractmethod
    def observe(self) -> ObservationBatch:
        """Expose only the declared local information to each actor."""

    @abstractmethod
    def export_state(self) -> StateSnapshot:
        """Export an immutable full state for supervision and diagnostics."""

    @abstractmethod
    def step_joint(self, actions: Sequence[int]) -> StepResult:
        """Execute the supplied global indices without remapping actions."""

    @abstractmethod
    def packet_traces(self) -> tuple[PacketTrace, ...]:
        """Include completed traces and censored, still unresolved prefixes."""

    def available_actions(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(obs.available_actions for obs in self.observe().agents)

    def close(self) -> None:
        """Release optional external simulator resources."""
