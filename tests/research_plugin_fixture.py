"""TEST ONLY: exercise model/trainer wiring; no learned weights or optimization."""

from pathlib import Path

from src.buoy_network.research.bootstrap import build_runner
from src.buoy_network.research.contracts import CandidateFeatures, TrainingReport
from src.buoy_network.research.models.base import ProgressModel, WorldModel
from src.buoy_network.research.runtime.checkpoints import CheckpointMetadata
from src.buoy_network.research.training.base import Trainer


class FixtureWorldModel(WorldModel):
    version = "fixture_wm_v1"

    def predict_candidates(self, local_history):
        return CandidateFeatures(tuple(tuple(
            (0.5 if valid else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for valid in agent.observations[-1].available_actions
        ) for agent in local_history.agents), self.version)


class FixtureProgressModel(ProgressModel):
    version = "fixture_pm_v1"

    def score_prefix(self, trace):
        return 0.5


class FixtureTrainer(Trainer):
    def __init__(self, component):
        self.component = component

    def fit(self, context):
        metrics = {"test_fixture_only": 1}
        if context.dataset is not None:
            metrics["train_rows"] = sum(1 for _ in context.dataset.iter_rows("train"))
            metrics["validation_rows"] = sum(1 for _ in context.dataset.iter_rows("validation"))
        else:
            runner = build_runner(context.config, context.device, context.registry)
            try:
                records = runner.run_episode("fixture_policy", context.config.runtime.seed)
                metrics["environment_steps"] = len(records)
            finally:
                runner.close()
        version = {"world_model": FixtureWorldModel.version,
                   "progress_model": FixtureProgressModel.version,
                   "policy": "fixture_policy_v1"}[self.component]
        path = context.artifacts.root / "checkpoints" / (self.component + ".json")
        CheckpointMetadata.for_config(context.config, self.component, version).save(path)
        return TrainingReport(self.component, version, metrics, str(path))


def register_components(registry):
    def world_model(config, device):
        metadata = CheckpointMetadata.load(config.world_model.checkpoint)
        metadata.validate(config, "world_model")
        if metadata.model_version != FixtureWorldModel.version:
            raise ValueError("Wrong fixture WM version")
        return FixtureWorldModel()

    def progress_model(config, device):
        metadata = CheckpointMetadata.load(config.progress.checkpoint)
        metadata.validate(config, "progress_model")
        if metadata.model_version != FixtureProgressModel.version:
            raise ValueError("Wrong fixture PM version")
        return FixtureProgressModel()

    registry.register("world_model", "fixture_wm", world_model)
    registry.register("progress_model", "fixture_pm", progress_model)
    for name, component in (("fixture_wm", "world_model"), ("fixture_pm", "progress_model"),
                            ("fixture_policy", "policy")):
        registry.register("trainer", name, lambda cfg, device, component=component: FixtureTrainer(component))
