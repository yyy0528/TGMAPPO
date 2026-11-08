"""Frozen WM bridge for the ORIGINAL TG runner (Python 3.6/Torch 1.8)."""

import hashlib
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .world_model import GraphWorldModel, encode_inputs


ARCHITECTURE = "legacy_tg_graph_gru_wm_v4_motion_context"
FEATURES = ("ack_probability", "sender_joint_queue_fraction_delta", "receiver_joint_queue_fraction_delta")


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_frozen_wm(path, n):
    try:
        # weights_only is unavailable in the installed Torch 1.8.
        if int(torch.__version__.split('.')[0]) >= 2:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        else:
            payload = torch.load(path, map_location="cpu")
        if (payload["architecture"] != ARCHITECTURE or payload["node_count"] != n
                or payload["normalization_fit_split"] != "train"
                or type(payload["history_length"]) is not int or not 1 <= payload["history_length"] <= 32):
            raise ValueError("Incompatible WM architecture/nodes/history/normalization")
        with torch.random.fork_rng(devices=[]):
            model = GraphWorldModel(payload["hidden"])
        expected = model.state_dict()
        state = payload["state_dict"]
        if set(state) != set(expected):
            raise ValueError("WM tensor names differ")
        for name, value in state.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
                raise ValueError("Invalid WM tensor: " + name)
        if any((state[key] <= 0).any() for key in ("node_std", "queue_scale", "motion_scale")):
            raise ValueError("Invalid WM normalization scale")
        if not 0 <= state["ack_prior"].item() <= 1:
            raise ValueError("Invalid WM ACK prior")
        model.load_state_dict(state)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, payload["history_length"]
    except Exception as exc:
        raise ValueError("Cannot load frozen WM {}: {}".format(path, exc)) from None


class FrozenWMRuntime:
    def __init__(self, path, n, feature_source="learned", device="cpu"):
        if feature_source not in ("learned", "history"):
            raise ValueError("Unknown candidate feature source")
        self.feature_source = feature_source
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda") or (self.device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("WM inference requires an available cpu/cuda device")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.model, self.history_length = load_frozen_wm(path, n)
        self.model.to(self.device)
        self.n = n
        self.path = str(Path(path).resolve())
        self.file_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.initial_digest = tensor_digest(self.model.state_dict())
        self.predictions = 0
        self.reset()

    def reset(self):
        self.histories = [deque(maxlen=self.history_length) for _ in range(self.n)]
        self.previous_actions = [None] * self.n
        self.previous_ack = [None] * self.n
        self.previous_attempt = [False] * self.n
        self.next_step = 0

    def record_actions(self, actions):
        self.previous_actions = [int(value) for value in actions[0, :, 0]]
        self.previous_ack = [None] * self.n
        self.previous_attempt = [False] * self.n

    def record_attempt(self, sender, receiver, ack):
        if self.previous_actions[sender] != receiver:
            raise ValueError("WM history action differs from executed action")
        self.previous_ack[sender] = bool(ack)
        self.previous_attempt[sender] = True

    @torch.no_grad()
    def predict(self, obs, nodes, distances, available, step):
        if step != self.next_step:
            raise ValueError("WM history must advance once per decision")
        if (np.shape(obs) != (1, self.n, self.n+1)
                or np.shape(nodes) != (1, self.n, self.n, 4)
                or np.shape(distances) != (1, self.n, self.n, self.n)
                or np.shape(available) != (1, self.n, self.n+1)):
            raise ValueError("WM online observation shape mismatch")
        if (not all(np.isfinite(array).all() for array in (obs, nodes, distances, available))
                or not np.isin(available, [0, 1]).all() or not available.any(-1).all()):
            raise ValueError("WM requires finite inputs and valid binary action masks")
        result = np.zeros((1, self.n, self.n+1, 3), dtype=np.float32)
        records, locations = [], []
        for node in range(self.n):
            self.histories[node].append({"step": step,
                "own_obs": obs[0, node].tolist(), "node_features": nodes[0, node].tolist(),
                "distances": distances[0, node].tolist(),
                "own_previous_action": self.previous_actions[node],
                "own_previous_ack": self.previous_ack[node],
                "own_previous_attempt": self.previous_attempt[node]})
            # WAIT-only nodes have no choice and need no untrained extrapolation.
            if not available[0, node, :self.n].any():
                continue
            for action in np.flatnonzero(available[0, node]):
                if self.feature_source == "history":
                    history = list(self.histories[node])
                    feedback = [float(frame["own_previous_ack"]) for frame in history
                                if frame["own_previous_action"] == action and frame["own_previous_attempt"]]
                    ack = (sum(feedback)+1)/(len(feedback)+2) if feedback else obs[0, node, action]
                    def trend(target):
                        values = [frame["node_features"][target][2] for frame in history]
                        return (values[-1]-values[0])/(len(values)-1) if len(values) > 1 else 0.
                    result[0, node, action] = (0. if action == self.n else ack, trend(node),
                                               0. if action == self.n else trend(action))
                    continue
                records.append(encode_inputs({"agent_id": node, "inputs": {
                    "history": list(self.histories[node]), "candidate_action": int(action)}}, self.history_length))
                locations.append((node, int(action)))
        if records:
            data = {key: torch.stack([row[key] for row in records]).to(self.device) for key in records[0]}
            prediction = self.model(data)
            values = torch.cat((prediction["ack_logit"].sigmoid().unsqueeze(-1), prediction["queue"]), -1)
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite frozen WM prediction")
            values = values.cpu().numpy()
            for index, (node, action) in enumerate(locations):
                result[0, node, action] = values[index]
                if action == self.n:
                    result[0, node, action, 0] = 0.0
                    result[0, node, action, 2] = 0.0
            result[..., 1:] = np.clip(result[..., 1:], -1, 1)
        result[..., 1:] = np.clip(result[..., 1:], -1, 1)
        self.next_step += 1
        self.predictions += 1
        return result

    def assert_frozen(self):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("WM must remain frozen/eval")
        if any(t.device != self.device for t in list(self.model.parameters())+list(self.model.buffers())):
            raise ValueError("Frozen WM inference device changed")
        if tensor_digest(self.model.state_dict()) != self.initial_digest:
            raise ValueError("Frozen WM weights/normalization changed")

    def metadata(self):
        self.assert_frozen()
        return {"checkpoint": self.path, "checkpoint_sha256": self.file_sha256,
                "state_sha256": self.initial_digest, "history_length": self.history_length,
                "inference_device": str(self.device),
                "features": list(FEATURES), "predictions": self.predictions,
                "feature_source": self.feature_source, "learned_forward_used": self.feature_source == "learned",
                "frozen": True, "critic_uses_wm": False,
                "not_used": ["motion", "delivery", "drop", "uncertainty"]}
