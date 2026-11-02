from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import ExperimentConfig
from ..contracts import TrainingReport
from ..data.storage import DatasetStore
from ..registry import ComponentRegistry
from ..runtime.artifacts import RunArtifacts


@dataclass(frozen=True)
class TrainingContext:
    config: ExperimentConfig
    device: str
    artifacts: RunArtifacts
    registry: ComponentRegistry
    dataset: DatasetStore | None = None


class Trainer(ABC):
    """Offline WM/PM and on-policy PPO trainers share lifecycle, not replay data.

    fit() owns optimizer/checkpoint state. PPO implementations use EpisodeRunner
    and RolloutBatch only; offline implementations use context.dataset, fitting
    normalizers on the training split and freezing them for validation/test.
    """

    @abstractmethod
    def fit(self, context: TrainingContext) -> TrainingReport:
        """Train, persist a compatible checkpoint, and return measured metrics."""
