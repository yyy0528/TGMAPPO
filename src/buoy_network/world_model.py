"""Minimal supervised WM for original TG exports; PyTorch, no Gym/PyG.

Independent graph/history encoder. Current candidate action enters ONLY the
business head. Privileged simulator truth and packet outcomes never enter it.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .dataset import iter_wm_samples


def encode_sample(sample, history_length=4):
    history = sample["inputs"]["history"][-history_length:]
    node = sample["agent_id"]
    n = len(history[-1]["node_features"])
    action = sample["inputs"]["candidate_action"]
    nodes = torch.zeros(history_length, n, 4)
    distances = torch.zeros(history_length, n, n)
    context = torch.zeros(history_length, 8)
    for index, frame in enumerate(history):
        nodes[index] = torch.tensor(frame["node_features"])
        distances[index] = torch.tensor(frame["distances"])
        rates = torch.tensor(frame["own_obs"][:-1])
        previous = frame["own_previous_action"]
        delta = ([0.0, 0.0] if previous is None or previous == n else
                 ((nodes[index, previous, :2]-nodes[index, node, :2])/4).tolist())
        context[index] = torch.tensor([rates.mean().item(), rates.std(unbiased=False).item(),
                                       *delta, float(previous == n), frame["step"] / 80.0,
                                       float(frame.get("own_previous_ack") or 0),
                                       float(frame.get("own_previous_attempt", False))])
    current = nodes[len(history)-1]
    target_node = node if action == n else action
    candidate = torch.tensor([
        (current[target_node, 0] - current[node, 0]).item() / 4,
        (current[target_node, 1] - current[node, 1]).item() / 4,
        current[target_node, 2].item(),
        0.0 if action == n else history[-1]["own_obs"][action], float(action == n)])
    next_nodes = torch.tensor(sample["targets"]["next_node_features"])
    queue = torch.stack((next_nodes[node, 2] - current[node, 2],
                         next_nodes[target_node, 2] - current[target_node, 2]))
    return {"nodes": nodes, "distances": distances, "context": context,
            "length": torch.tensor(len(history)), "agent": torch.tensor(node),
            "target": torch.tensor(target_node), "candidate": candidate,
            "ack": torch.tensor(float(sample["targets"]["ack"] or 0)),
            "ack_mask": torch.tensor(sample["targets"]["ack_loss_mask"]),
            "queue": queue, "queue_mask": torch.tensor([True, action != n]),
            "motion": next_nodes[node, :2] - current[node, :2]}


def load_split(root, split, history_length=4, limit=None):
    records = []
    for sample in iter_wm_samples(root, split, history_length):
        records.append(encode_sample(sample, history_length))
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError("No samples in " + split)
    return {key: torch.stack([row[key] for row in records]) for key in records[0]}


def batch(data, indices, device="cpu"):
    return {key: value[indices].to(device) for key, value in data.items()}


class GraphWorldModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.input = nn.Linear(4, hidden)
        self.messages = nn.ModuleList([nn.Linear(2*hidden, hidden) for _ in range(2)])
        self.history = nn.GRU(2*hidden+8, hidden, batch_first=True)
        self.topology = nn.Linear(hidden, 2)
        self.business = nn.Sequential(nn.Linear(2*hidden+5, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.register_buffer("node_mean", torch.zeros(4))
        self.register_buffer("node_std", torch.ones(4))
        self.register_buffer("queue_scale", torch.ones(2))
        self.register_buffer("motion_scale", torch.ones(2))
        self.register_buffer("ack_prior", torch.tensor(0.5))

    @torch.no_grad()
    def fit_normalization(self, training):
        valid = torch.arange(training["nodes"].shape[1])[None, :] < training["length"][:, None]
        nodes = training["nodes"][valid].reshape(-1, 4)
        self.node_mean.copy_(nodes.mean(0))
        self.node_std.copy_(nodes.std(0, unbiased=False).clamp_min(0.01))
        for column in range(2):
            selected = training["queue"][:, column][training["queue_mask"][:, column]]
            self.queue_scale[column] = selected.square().mean().sqrt().clamp_min(0.01)
        self.motion_scale.copy_(training["motion"].square().mean(0).sqrt().clamp_min(0.01))
        self.ack_prior.copy_(training["ack"][training["ack_mask"]].mean())
        if not torch.isfinite(self.ack_prior):
            raise ValueError("Training split contains no attempted transmissions")

    def forward(self, data):
        nodes, distances = data["nodes"], data["distances"]
        b, t, n, _ = nodes.shape
        normalized = (nodes - self.node_mean) / self.node_std
        encoded = torch.relu(self.input(normalized.reshape(b*t, n, 4)))
        edges = ((distances <= 1.5) & (distances >= 0)).reshape(b*t, n, n).float()
        weights = edges / edges.sum(-1, keepdim=True).clamp_min(1)
        for layer in self.messages:
            encoded = torch.relu(layer(torch.cat((encoded, torch.bmm(weights, encoded)), -1)))
        encoded = encoded.reshape(b, t, n, -1)
        rows = torch.arange(b, device=nodes.device)
        ego = encoded[rows[:, None], torch.arange(t, device=nodes.device)[None, :], data["agent"][:, None]]
        temporal = torch.cat((ego, encoded.mean(2), data["context"]), -1)
        packed = nn.utils.rnn.pack_padded_sequence(temporal, data["length"].cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.history(packed)
        hidden = hidden[-1]
        target = encoded[rows, data["length"]-1, data["target"]]
        target = target * (1 - data["candidate"][:, -1:])
        business = self.business(torch.cat((hidden, target, data["candidate"]), -1))
        return {"ack_logit": business[:, 0], "queue": business[:, 1:] * self.queue_scale,
                "motion": self.topology(hidden) * self.motion_scale}

    def loss(self, data):
        prediction = self(data)
        mask = data["ack_mask"]
        ack = (F.binary_cross_entropy_with_logits(prediction["ack_logit"][mask], data["ack"][mask])
               if mask.any() else prediction["ack_logit"].sum()*0)
        queue = F.smooth_l1_loss((prediction["queue"]/self.queue_scale)[data["queue_mask"]],
                                 (data["queue"]/self.queue_scale)[data["queue_mask"]])
        motion = F.smooth_l1_loss(prediction["motion"]/self.motion_scale, data["motion"]/self.motion_scale)
        return ack + queue + motion


@torch.no_grad()
def evaluate(model, data, device="cpu", batch_size=256):
    model.eval()
    predictions = {key: [] for key in ("ack_logit", "queue", "motion")}
    for indices in torch.arange(len(data["length"])).split(batch_size):
        result = model(batch(data, indices, device))
        for key in predictions:
            predictions[key].append(result[key].cpu())
    pred = {key: torch.cat(value) for key, value in predictions.items()}
    mask = data["ack_mask"]
    probability = pred["ack_logit"].sigmoid()[mask]
    labels = data["ack"][mask]
    prior = model.ack_prior.cpu().expand_as(labels)
    link = data["candidate"][:, 3][mask].clamp(0, 1)
    return {"samples": len(data["length"]), "attempts": int(mask.sum()),
            "ack_brier": float((probability-labels).square().mean()),
            "ack_nll": float(F.binary_cross_entropy(probability.clamp(1e-6, 1-1e-6), labels)),
            "ack_prior_brier": float((prior-labels).square().mean()),
            "ack_history_link_brier": float((link-labels).square().mean()),
            "queue_mse": float((pred["queue"]-data["queue"])[data["queue_mask"]].square().mean()),
            "queue_persistence_mse": float(data["queue"][data["queue_mask"]].square().mean()),
            "motion_mse": float((pred["motion"]-data["motion"]).square().mean()),
            "motion_persistence_mse": float(data["motion"].square().mean())}
