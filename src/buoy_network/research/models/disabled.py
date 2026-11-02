from .base import ProgressModel, WorldModel


class DisabledWorldModel(WorldModel):
    version = "disabled"

    def predict_candidates(self, local_history):
        return None


class DisabledProgressModel(ProgressModel):
    version = "disabled"

    def score_prefix(self, trace):
        return 0.0
