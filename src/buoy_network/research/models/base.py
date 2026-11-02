from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts import CandidateFeatures, LocalHistoryBatch, PacketTrace


class WorldModel(ABC):
    version: str

    @abstractmethod
    def predict_candidates(self, local_history: LocalHistoryBatch
                           ) -> CandidateFeatures | None:
        """Predict [N,N+1,6] features under the current legal action masks.

        Implementations own graph/temporal encoding, eval/no_grad mode and frozen
        normalizers. The topology head must not depend on the candidate action.
        Each agent is predicted independently using its own history/masks only.
        """


class ProgressModel(ABC):
    version: str

    @abstractmethod
    def score_prefix(self, trace: PacketTrace) -> float:
        """Return a [0,1] outcome-supervised completion score, without future labels."""

    def packet_potential(self, trace: PacketTrace) -> float:
        """Reliability variants override this to apply a gate BEFORE differencing."""
        return 0.0 if trace.outcome is not None else self.score_prefix(trace)

    def team_potential(self, traces: tuple[PacketTrace, ...], initial_count: int) -> float:
        return sum(self.packet_potential(trace) for trace in traces) / max(initial_count, 1)
