"""Versioned team reward from observed packet events; no model or future labels."""
import math


def reward_contract(args):
    version = getattr(args, "tg_reward_version", "legacy_v1")
    if version == "legacy_v1":
        return {"version": version}
    if version != "task_v2":
        raise ValueError("Unknown TG reward version")
    weights = {key: float(getattr(args, "tg_"+key, 1.))
               for key in ("delivery_weight", "delay_weight", "unfinished_weight")}
    if (not all(math.isfinite(v) for v in weights.values())
            or weights["delivery_weight"] <= 0 or weights["delay_weight"] <= 0
            or weights["unfinished_weight"] < 0):
        raise ValueError("Task reward requires finite delivery/delay > 0 and unfinished >= 0")
    return dict(weights, version=version, normalization="initial_packets_and_task_horizon_v1")


class TaskRewardLedger:
    """One event ledger per finite task. Unfinished cost never creates drop labels.

    Time is charged for packets alive BEFORE each slot, including a delivery slot.
    Weights are fixed at construction; normalization uses initial P and full H.
    """
    def __init__(self, contract, packet_ids, horizon, episode=0):
        ids = list(packet_ids)
        if len(ids) != len(set(ids)) or type(horizon) is not int or horizon <= 0:
            raise ValueError("Require unique packet IDs and a positive integer task horizon")
        self.contract = dict(contract)
        self.initial = len(ids)
        self.unresolved = set(ids)
        self.pending = set()
        self.horizon, self.episode, self.step = horizon, episode, 0

    def delivered(self, packet_id):
        if packet_id not in self.unresolved or packet_id in self.pending:
            raise ValueError("Unknown or duplicate packet delivery")
        self.pending.add(packet_id)

    def advance(self, step, legacy_reward, terminated, rollout_truncated=False):
        if (step != self.step+1 or step > self.horizon
                or bool(terminated) != (step == self.horizon)
                or (terminated and rollout_truncated) or not math.isfinite(float(legacy_reward))):
            raise ValueError("Invalid task step/boundary/reward")
        before, delivered = len(self.unresolved), len(self.pending)
        after = before-delivered
        scale = max(self.initial, 1)
        row = {"episode": self.episode, "step": step-1, "next_step": step, "reward_version": self.contract["version"],
               "initial_packets": self.initial, "unresolved_before": before,
               "delivered_this_step": delivered, "unresolved_after": after,
               "reward_legacy": float(legacy_reward), "reward_delivery": 0.,
               "reward_delay": 0., "reward_window_end": 0.,
               "terminated": bool(terminated), "rollout_truncated": bool(rollout_truncated)}
        if self.contract["version"] == "task_v2":
            row["reward_delivery"] = self.contract["delivery_weight"]*delivered/scale
            row["reward_delay"] = -self.contract["delay_weight"]*before/(scale*self.horizon)
            row["reward_window_end"] = -self.contract["unfinished_weight"]*after/scale if terminated else 0.
            row["reward_base"] = row["reward_delivery"]+row["reward_delay"]+row["reward_window_end"]
        elif self.contract["version"] == "legacy_v1":
            row["reward_base"] = float(legacy_reward)
        else:
            raise ValueError("Unknown reward contract")
        self.unresolved.difference_update(self.pending)
        self.pending.clear()
        self.step = step
        return row


REWARD_FIELDS = ("episode", "step", "next_step", "reward_version", "initial_packets", "unresolved_before",
                "delivered_this_step", "unresolved_after", "reward_legacy", "reward_delivery",
                "reward_delay", "reward_window_end", "reward_base", "reward_progress", "reward_train",
                "terminated", "rollout_truncated")
