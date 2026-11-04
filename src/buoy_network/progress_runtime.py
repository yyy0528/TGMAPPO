"""Frozen censored-time PM and potential shaping in the original TG runner."""
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from .progress_model import ARCHITECTURE, FEATURES, ProgressModel, prefix_features
from .wm_runtime import tensor_digest


def shaping_reward(environment_reward, phi_t, phi_tp1, gamma, alpha, terminated=False, truncated=False):
    if (not all(math.isfinite(x) for x in (environment_reward, phi_t, phi_tp1, gamma, alpha))
            or not 0 < gamma <= 1 or alpha < 0 or (terminated and truncated)):
        raise ValueError("Invalid PM shaping inputs/boundary")
    following = 0.0 if terminated else phi_tp1
    progress = alpha*(gamma*following-phi_t)
    return float(environment_reward+progress), float(progress), float(following)


def completion_potential(remaining_probability_mass, delivered, initial_packets):
    """Expected completed fraction, retaining real deliveries before the terminal boundary."""
    if (type(delivered) is not int or type(initial_packets) is not int or
            not 0 <= delivered <= initial_packets or not math.isfinite(remaining_probability_mass)
            or not 0 <= remaining_probability_mass <= initial_packets-delivered+1e-6):
        raise ValueError("Invalid PM completion mass")
    return (remaining_probability_mass+delivered)/max(initial_packets,1)


class FrozenPMRuntime:
    def __init__(self, path, n, horizon, gamma, alpha, potential_mode="remaining"):
        if potential_mode not in ("remaining", "completion_credit"):
            raise ValueError("Unknown PM potential mode")
        shaping_reward(0, 0, 0, gamma, alpha)
        try:
            payload = (torch.load(path, map_location="cpu", weights_only=True)
                       if int(torch.__version__.split('.')[0]) >= 2 else torch.load(path, map_location="cpu"))
            if (payload["architecture"] != ARCHITECTURE or payload["node_count"] != n
                    or payload["horizon"] != horizon or payload["features"] != list(FEATURES)
                    or payload["normalization_fit_split"] != "train" or not payload["validation_gate_passed"]):
                raise ValueError("PM schema/horizon/nodes/validation gate mismatch")
            metrics = payload["validation"]
            if not (0 <= metrics["survival_nll"] < metrics["prior_survival_nll"]
                    and 0 <= metrics["window_brier"] < metrics["prior_window_brier"]):
                raise ValueError("PM validation metrics fail the declared gate")
            with torch.random.fork_rng(devices=[]):
                self.model = ProgressModel(horizon, payload["hidden"])
            state, expected = payload["state_dict"], self.model.state_dict()
            if set(state) != set(expected):
                raise ValueError("PM parameter names differ")
            for key, value in state.items():
                if value.shape != expected[key].shape or value.dtype != expected[key].dtype or not torch.isfinite(value).all():
                    raise ValueError("Invalid PM tensor: "+key)
            if not (state["feature_std"] > 0).all():
                raise ValueError("Invalid PM normalization")
            self.model.load_state_dict(state)
        except Exception as exc:
            raise ValueError("Cannot load frozen PM {}: {}".format(path, exc)) from None
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.path = str(Path(path).resolve())
        self.file_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.initial_digest = tensor_digest(self.model.state_dict())
        self.n, self.horizon, self.gamma, self.alpha = n, horizon, gamma, alpha
        self.potential_mode = potential_mode
        self.settings = (n, horizon, gamma, alpha, potential_mode)
        self.summaries, self.episode_records = [], []
        self.started = False

    def assert_frozen(self):
        if ((self.n, self.horizon, self.gamma, self.alpha, self.potential_mode) != self.settings or self.model.training
                or any(p.requires_grad for p in self.model.parameters())
                or tensor_digest(self.model.state_dict()) != self.initial_digest):
            raise ValueError("PM weights/normalization/alpha/gamma must remain fixed")

    @torch.no_grad()
    def potential(self, env, step):
        alive = {p.id: p for p in env.packets if not p.if_sink}
        if set(alive) != self.packet_ids-self.finished:
            raise ValueError("PM packet conservation/delivery callback mismatch")
        if step == self.horizon:
            return 0.0
        if not alive:
            return (completion_potential(0.,len(self.finished),self.initial_packets)
                    if self.potential_mode == "completion_credit" else 0.0)
        obs, _, nodes, distances = env.observe_graph()
        actor = {"obs": obs, "node_features": nodes[0], "distances": distances[0]}
        keys, rows, hidden = [], [], []
        for key, packet in sorted(alive.items()):
            value, state = prefix_features(actor, packet.agent_id, packet.destination, packet.route,
                                          step, self.horizon, self.prefix_state.get(key))
            self.prefix_state[key] = state
            keys.append(key)
            rows.append(value)
            hidden.append(self.hidden.get(key, torch.zeros(self.model.hidden)))
        logits, following = self.model.step(torch.tensor(np.asarray(rows)), torch.stack(hidden).unsqueeze(0))
        probability = -torch.expm1(-torch.nn.functional.softplus(logits[:, :self.horizon-step]).sum(-1))
        if not torch.isfinite(probability).all():
            raise ValueError("Nonfinite PM score")
        for index, key in enumerate(keys):
            self.hidden[key] = following[0, index].clone()
        mass = float(probability.sum())
        if self.potential_mode == "completion_credit":
            return completion_potential(mass,len(self.finished),self.initial_packets)
        return mass/max(self.initial_packets, 1)

    def start_episode(self, env, episode):
        self.assert_frozen()
        if self.started:
            raise ValueError("PM previous task has not terminated")
        self.episode, self.step = episode, 0
        self.hidden, self.prefix_state, self.finished = {}, {}, set()
        self.packet_ids = {p.id for p in env.packets if not p.if_sink}
        self.initial_packets = len(self.packet_ids)
        self.episode_records = []
        self.phi = self.initial_phi = self.potential(env, 0)
        self.started = True

    def delivered(self, packet_id):
        if not self.started or packet_id not in self.packet_ids or packet_id in self.finished:
            raise ValueError("Duplicate/unknown PM delivery")
        self.finished.add(packet_id)
        self.hidden.pop(packet_id, None)
        self.prefix_state.pop(packet_id, None)

    def advance(self, env, step, environment_reward, terminated):
        self.assert_frozen()
        if not self.started or step != self.step+1 or step > self.horizon or terminated != (step == self.horizon):
            raise ValueError("PM task/time boundary mismatch")
        # Validate conservation even at terminal. Current legacy rollout equals the whole finite task.
        following = self.potential(env, step)
        train, progress, following = shaping_reward(environment_reward, self.phi, following,
                                                     self.gamma, self.alpha, terminated=terminated)
        row = {"episode": self.episode, "step": self.step, "reward_env": float(environment_reward),
               "phi_t": self.phi, "phi_tp1": following, "reward_progress": progress,
               "reward_train": train, "terminated": bool(terminated), "rollout_truncated": False}
        self.episode_records.append(row)
        self.phi, self.step = following, step
        if terminated:
            discounted = sum(self.gamma**r["step"]*r["reward_progress"] for r in self.episode_records)
            error = abs(discounted+self.alpha*self.initial_phi)
            if error > 1e-7:
                raise ValueError("PM discounted boundary identity failed")
            self.summaries.append({"episode": self.episode, "initial_phi": self.initial_phi,
                                   "discounted_progress": discounted, "boundary_error": error,
                                   "terminal_phi": following, "delivered": len(self.finished),
                                   "unresolved_censored": self.initial_packets-len(self.finished)})
            self.started = False
        return row

    def metadata(self):
        self.assert_frozen()
        return {"checkpoint": self.path, "checkpoint_sha256": self.file_sha256,
                "state_sha256": self.initial_digest, "architecture": ARCHITECTURE,
                "features": list(FEATURES), "alpha": self.alpha, "gamma": self.gamma, "horizon": self.horizon,
                "potential_mode": self.potential_mode,
                "frozen": True, "target": "delivery within remaining finite window; not eventual failure",
                "reliability_gate": "constant_one; no uncertainty claim", "episodes": self.summaries}
