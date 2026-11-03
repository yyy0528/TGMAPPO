"""Original-TG packet-prefix PM with right-censored discrete delivery times.

No TTL failures are invented. Censored traces contribute observed survival only.
The score is delivery probability within a known finite window, not eventual success.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .dataset import read_json, read_rows, run_roots, require

ARCHITECTURE = "legacy_tg_survival_pm_v1"
FEATURES = ("sink_distance", "own_queue", "neighbor_queue", "best_neighbor_distance",
            "best_link", "mean_link", "degree", "elapsed_window", "remaining_window",
            "hops", "revisit_fraction", "waiting", "neighbor_change", "sink_queue")


def prefix_features(actor, node, destination, route, step, horizon, previous=None):
    nodes = np.asarray(actor["node_features"], dtype=np.float32)
    distances = np.asarray(actor["distances"], dtype=np.float32)
    n = len(nodes)
    require(nodes.shape == (n, 4) and distances.shape == (n, n), "PM graph shape")
    require(0 <= step < horizon and route and route[-1] == node, "PM prefix/time mismatch")
    neighbors = set(i for i in range(n) if i != node and 0 < distances[node, i] <= 1.5)
    separation = np.linalg.norm(nodes[:, :2]-nodes[destination, :2], axis=1) / math.sqrt(32)
    ordered = sorted(neighbors)
    rates = np.asarray(actor["obs"][node], dtype=np.float32)[:n]
    last_move = step if previous is None or len(route) != previous["route_length"] else previous["last_move"]
    old_neighbors = neighbors if previous is None else previous["neighbors"]
    change = len(neighbors.symmetric_difference(old_neighbors))/max(len(neighbors.union(old_neighbors)), 1)
    values = [separation[node], nodes[node, 2], float(nodes[ordered, 2].mean()) if ordered else 0,
              float(separation[ordered].min()) if ordered else separation[node],
              float(rates[ordered].max()) if ordered else 0, float(rates[ordered].mean()) if ordered else 0,
              len(neighbors)/max(n-1, 1), step/horizon, (horizon-step)/horizon,
              (len(route)-1)/horizon, (len(route)-len(set(route)))/len(route), (step-last_move)/horizon,
              change, nodes[destination, 2]]
    require(np.isfinite(values).all(), "Nonfinite PM features")
    return np.asarray(values, dtype=np.float32), {"neighbors": neighbors, "route_length": len(route), "last_move": last_move}


def load_packets(root, split):
    packets = []
    for run in run_roots(root, split):
        manifest = read_json(run/"manifest.json")
        require(manifest["complete"] and manifest["environment"] == "legacy_fixed_v1", "PM source incompatible")
        horizon = manifest["config"]["episode_length"]
        states = {row["snapshot_id"]: row["actor_inputs"] for row in read_rows(run/"snapshots.jsonl")}
        for trace in read_rows(run/"packet_traces.jsonl"):
            require(trace["outcome"] in (1, None), "PM v1 has no actual failure events")
            require(trace["censored"] == (trace["outcome"] is None), "PM censor flag mismatch")
            end = trace["last_observed_step"]
            features, previous, route = [], None, [trace["source"]]
            for prefix in trace["prefix"][:-1]:
                step, node = prefix["step"], prefix["node"]
                if route[-1] != node:
                    route.append(node)
                value, previous = prefix_features(states[prefix["snapshot_ref"]], node, trace["destination"],
                                                  route, step, horizon, previous)
                features.append(value)
            require(len(features) == end and end > 0 and end <= horizon, "Incomplete PM trace")
            packets.append({"features": torch.tensor(np.asarray(features)), "event": trace["outcome"] == 1,
                            "length": end, "horizon": horizon, "packet_key": trace["packet_key"],
                            "behavior": manifest["behavior"], "scenario_seed": manifest["scenario_seed"]})
    require(bool(packets), "Empty PM split: "+split)
    require(len({p["horizon"] for p in packets}) == 1, "Mixed PM horizons")
    return packets


def pack(packets):
    horizon = packets[0]["horizon"]
    values = torch.zeros(len(packets), horizon, len(FEATURES))
    lengths = torch.tensor([p["length"] for p in packets])
    for index, packet in enumerate(packets):
        values[index, :packet["length"]] = packet["features"]
    return {"features": values, "lengths": lengths, "events": torch.tensor([p["event"] for p in packets]),
            "horizon": horizon}


def survival_terms(logits, lengths, events):
    """Mean prefix likelihood per packet (packets, not frames, are the unit)."""
    b, t, k = logits.shape
    time = torch.arange(t, device=logits.device)[None, :]
    duration = lengths[:, None]-time
    valid = duration > 0
    offsets = torch.arange(k, device=logits.device)[None, None, :]
    observed = offsets < duration[:, :, None]
    losses = (F.softplus(logits)*observed).sum(-1)
    event_logit = logits.gather(-1, (duration-1).clamp(0, k-1).unsqueeze(-1)).squeeze(-1)
    losses = losses-event_logit*events[:, None].float()*valid
    return (losses*valid).sum(-1)/lengths.float()


class ProgressModel(nn.Module):
    def __init__(self, horizon=80, hidden=64):
        super().__init__()
        self.horizon, self.hidden = horizon, hidden
        self.project = nn.Sequential(nn.Linear(len(FEATURES), 32), nn.Tanh())
        self.history = nn.GRU(32, hidden, batch_first=True)
        self.hazard = nn.Linear(hidden, horizon)
        self.register_buffer("feature_mean", torch.zeros(len(FEATURES)))
        self.register_buffer("feature_std", torch.ones(len(FEATURES)))
        self.register_buffer("prior_logits", torch.zeros(horizon))

    @torch.no_grad()
    def fit(self, data):
        n, t, _ = data["features"].shape
        valid = torch.arange(t)[None, :] < data["lengths"][:, None]
        weights = valid.float()/data["lengths"][:, None].float()/n
        mean = (data["features"]*weights[:, :, None]).sum((0, 1))
        variance = (((data["features"]-mean)**2)*weights[:, :, None]).sum((0, 1))
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(variance.sqrt().clamp_min(0.01))
        duration = data["lengths"][:, None]-torch.arange(t)[None, :]
        offsets = torch.arange(self.horizon)[None, None, :]
        risk = ((offsets < duration[:, :, None])*weights[:, :, None]).sum((0, 1))
        events = ((offsets == duration[:, :, None]-1)*data["events"][:, None, None]*weights[:, :, None]).sum((0, 1))
        prior = (events/risk.clamp_min(1e-8)).clamp(1e-4, 1-1e-4)
        self.prior_logits.copy_(torch.log(prior/(1-prior)))
        nn.init.zeros_(self.hazard.weight)
        self.hazard.bias.copy_(self.prior_logits)

    def forward(self, features, lengths):
        projected = self.project((features-self.feature_mean)/self.feature_std)
        packed = nn.utils.rnn.pack_padded_sequence(projected, lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden, _ = self.history(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True, total_length=self.horizon)
        return self.hazard(hidden)

    def step(self, features, hidden):
        projected = self.project((features-self.feature_mean)/self.feature_std)
        values, following = self.history(projected.unsqueeze(1), hidden)
        return self.hazard(values[:, 0]), following


@torch.no_grad()
def evaluate(model, data, batch_size=32):
    model.eval()
    results = []
    for start in range(0, len(data["lengths"]), batch_size):
        lengths = data["lengths"][start:start+batch_size]
        features = data["features"][start:start+batch_size]
        events = data["events"][start:start+batch_size]
        logits = model(features, lengths)
        baseline = model.prior_logits.expand_as(logits)
        terms = survival_terms(logits, lengths, events)
        prior_terms = survival_terms(baseline, lengths, events)
        time = torch.arange(model.horizon)[None, :]
        remaining = model.horizon-time
        observed = lengths[:, None]-time
        valid = observed > 0
        offsets = torch.arange(model.horizon)[None, None, :]
        def brier(values):
            probability = -torch.expm1(-(F.softplus(values)*(offsets < remaining[:, :, None])).sum(-1))
            # Endpoint-window outcomes are known only if delivery occurred or window is fully observed.
            known = valid & (events[:, None] | (observed >= remaining))
            error = (probability-events[:, None].float())**2
            return (error*known).sum(-1)/known.sum(-1).clamp_min(1).float(), known.any(-1)
        actual_brier, known = brier(logits)
        prior_brier, _ = brier(baseline)
        results.append(torch.stack((terms, prior_terms, actual_brier, prior_brier, known.float()), -1))
    rows = torch.cat(results)
    identifiable = rows[:, 4] > 0
    require(bool(identifiable.any()), "No identifiable completion windows for PM Brier")
    metrics = torch.cat((rows[:, :2].mean(0), rows[identifiable, 2:4].mean(0))).tolist()
    return dict(zip(("survival_nll", "prior_survival_nll", "window_brier", "prior_window_brier"), metrics),
                packets=len(data["lengths"]), delivered=int(data["events"].sum()),
                censored=int((~data["events"]).sum()), identifiable_window_packets=int(identifiable.sum()))
