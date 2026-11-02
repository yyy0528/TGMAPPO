"""Opt-in records of the actual legacy TG simulator (Python 3.6 compatible).

No policy update, RNG draw, synthetic transition, or task-rule change lives here.
Outcome=None means censored/unresolved, never an invented negative PM label.
"""

import json
import math
import os
from pathlib import Path


SCHEMA = "legacy_tg_transition_v1"


def routing_actions(env, behavior, rng):
    """Data-only legal baselines; a private RNG never advances simulator RNG."""
    if behavior not in ("random", "geographic"):
        raise ValueError("Unknown data routing behavior: " + behavior)
    mask = env.available_actions()[0]
    actions, logs = [], []
    for node, row in enumerate(mask):
        legal = [index for index, valid in enumerate(row) if valid]
        if behavior == "random":
            selected = rng.choice(legal)
            logp = -math.log(len(legal))
        else:
            forward = [index for index in legal if index != env.buoy_number]
            selected = min(forward, key=lambda index: (
                math.hypot(env.buoys[index].position[0] - env.sink[0],
                           env.buoys[index].position[1] - env.sink[1]), index)) if forward else env.buoy_number
            logp = 0.0
        actions.append([selected])
        logs.append([logp])
    return [actions], [logs]


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def snapshot(env, episode_id, step, added_value):
    """Detached actor inputs, with privileged simulator truth kept separately."""
    obs, ids, nodes, distances = env.observe_graph()
    return plain({
        "snapshot_id": "{}:{}".format(episode_id, step), "step": step,
        "actor_inputs": {"obs": obs, "agent_id": ids,
                         "node_features": nodes[0], "distances": distances[0],
                         "available_actions": env.available_actions()[0]},
        "diagnostic_only": {
            "nodes": [{"id": node.id, "position": list(node.position),
                       "packet_counter": node.packet, "sink": node.if_sink,
                       "lost_rate_truth": node.lost_rate} for node in env.buoys],
            "live_packet_objects": [{"id": p.id, "node": p.agent_id,
                                     "delivered": p.if_sink, "stay_time": p.stay_time,
                                     "legacy_life_static": p.life} for p in env.packets],
            "history_reward_table": added_value,
        },
    })


class TransitionRecorder:
    def __init__(self, root, config, max_steps=20000, max_bytes=268435456, max_packets=10000):
        if min(max_steps, max_bytes, max_packets) <= 0:
            raise ValueError("Data budgets must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.config = dict(vars(config))
        self.max_steps, self.max_bytes, self.max_packets = max_steps, max_bytes, max_packets
        self.steps = self.bytes = self.packet_count = 0
        self.episodes = []
        self.current = None
        self.run_id = Path(config.output_root).name
        self.streams = {name: (self.root / (name + ".jsonl")).open("x", encoding="utf-8", newline="\n")
                        for name in ("snapshots", "transitions", "packet_traces")}
        self._manifest(False)

    def _manifest(self, complete):
        data = {"schema_version": SCHEMA, "environment": "legacy_fixed_v1",
                "run_id": self.run_id, "config": self.config,
                "split": self.config.get("wm_data_split", "train"),
                "policy_seed": self.config.get("seed", 0),
                "behavior": self.config.get("data_behavior", "tg-mappo"),
                "behavior_values": ("critic_prediction" if self.config.get("data_behavior", "tg-mappo") == "tg-mappo"
                                    else "unused_initial_critic_prediction"),
                "scenario_seed": (2568 if self.config.get("scenario_seed") is None
                                  else self.config["scenario_seed"]),
                "scenario_seed_override": self.config.get("scenario_seed") is not None,
                "complete": complete, "steps": self.steps,
                "packets": self.packet_count, "jsonl_bytes": self.bytes,
                "budgets": {"max_steps": self.max_steps, "max_bytes": self.max_bytes,
                            "max_packets": self.max_packets}, "episodes": self.episodes,
                "semantics": {
                    "graph": "global legacy actor inputs; graph copied once per joint snapshot",
                    "reward": "legacy dense/history components; not a sparse task reward",
                    "life": "static legacy field, NOT remaining TTL",
                    "outcome": "1=actual delivery; null=unresolved/censored; no implemented TTL failures",
                    "horizon": "PPO done at legacy horizon; unresolved packets are not labelled failed",
                    "diagnostic_only": "never pass privileged truth/future outcomes as model inputs",
                    "history": "added_value and RNG stream carry across episodes in this run",
                }}
        temp = self.root / "manifest.tmp"
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(str(temp), str(self.root / "manifest.json"))

    def _write(self, name, record):
        line = json.dumps(plain(record), ensure_ascii=False, allow_nan=False) + "\n"
        size = len(line.encode("utf-8"))
        if self.bytes + size > self.max_bytes:
            raise ValueError("WM export byte budget exceeded; dataset remains incomplete")
        self.streams[name].write(line)
        self.bytes += size

    def start_episode(self, env, episode, added_value):
        if self.current is not None:
            raise ValueError("Previous episode is not finished")
        episode_id = "{}:episode_{:06d}".format(self.run_id, episode)
        if self.packet_count + len(env.packets) > self.max_packets:
            raise ValueError("WM export packet budget exceeded")
        self.packet_count += len(env.packets)
        self.current = snapshot(env, episode_id, 0, added_value)
        self.episode_id = episode_id
        self.episode_index = episode
        self.traces = {}
        for packet in env.packets:
            self.traces[packet.id] = {
                "packet_key": [episode_id, packet.id], "source": packet.route[0],
                "destination": packet.destination, "generated_step": 0,
                "route": list(packet.route), "node": packet.agent_id,
                "outcome": None, "delivered_step": None,
                "prefix": [{"step": 0, "snapshot_ref": self.current["snapshot_id"],
                            "node": packet.agent_id, "event_ref": None}],
            }
        self._write("snapshots", self.current)

    def start_step(self, env, actions, log_probs, values, services):
        if self.steps >= self.max_steps:
            raise ValueError("WM export step budget exceeded before environment execution")
        self.step = env.current_step
        self.events = []
        self.forwarded = set()
        for node, action in enumerate(actions[0]):
            selected = int(action[0])
            packet_id = services.get(node)
            self.events.append({
                "sender": node, "action": selected, "packet_id": packet_id,
                "packet_key": None if packet_id is None else [self.episode_id, packet_id],
                "attempted": False, "ack": None, "delivered_this_step": False,
                "dropped_this_step": False,
                "reason": "wait" if selected == env.buoy_number else "no_service_packet",
                "queue_at_attempt": None,
                "reward_components": {"step_penalty": 0.0, "history": 0.0,
                                      "distance_progress": 0.0, "link_quality": 0.0,
                                      "failed_transmission": 0.0},
                "reward_legacy": 0.0,
            })
        kind = self.config.get("data_behavior", "tg-mappo")
        version = ("{}:ppo_update_{}".format(self.run_id, self.episode_index)
                   if kind == "tg-mappo" else "{}:{}".format(self.run_id, kind))
        self.behavior = plain({"policy_version": version, "source": kind,
                               "wm_version": "disabled", "pm_version": "disabled",
                               "actions": actions[0], "old_log_probs": log_probs[0],
                               "old_values": values[0]})

    def attempt(self, env, sender, receiver, packet_id, success, reward, history, penalty, before):
        event = self.events[sender]
        if event["action"] != receiver or event["packet_id"] != packet_id:
            raise ValueError("Executed action/packet does not match decision-time service")
        distance_progress = 0.0
        if success:
            source = env.buoys[sender].position
            target = env.buoys[receiver].position
            ds = math.hypot(source[0] - env.sink[0], source[1] - env.sink[1])
            dr = math.hypot(target[0] - env.sink[0], target[1] - env.sink[1])
            distance_progress = (ds - dr) / (2 * ds)
        failed = 0.0 if success else -0.5
        # Recover the executed link term from the actual scalar reward; no RNG draw.
        link = float(reward - history - penalty - distance_progress - failed) if success else 0.0
        event.update(attempted=True, ack=bool(success), reason="ack_success" if success else "ack_failure",
                     reward_legacy=float(reward), queue_at_attempt={
                         "before": list(before),
                         "after": [env.buoys[sender].packet, env.buoys[receiver].packet]},
                     reward_components={"step_penalty": float(penalty), "history": float(history),
                                        "distance_progress": distance_progress, "link_quality": link,
                                        "failed_transmission": failed})

    def transmitted(self, packet):
        trace = self.traces[packet.id]
        sender = trace["node"]
        event = self.events[sender]
        if event["packet_id"] != packet.id or event["ack"] is not True:
            raise ValueError("Packet moved without its recorded successful attempt")
        if packet.id in self.forwarded:
            raise ValueError("Packet forwarded twice in one joint slot")
        self.forwarded.add(packet.id)
        trace.update(node=packet.agent_id, route=list(packet.route))
        if packet.if_sink:
            trace.update(outcome=1, delivered_step=self.step + 1)
            event["delivered_this_step"] = True

    def end_step(self, env, added_value, reward, terminal):
        expected = sum(event["reward_legacy"] for event in self.events)
        if not math.isclose(float(reward), expected, rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError("Team reward differs from recorded executed rewards")
        for event in self.events:
            if event["ack"] is True and event["packet_id"] not in self.forwarded:
                raise ValueError("ACK success did not produce an actual packet transfer")
        following = snapshot(env, self.episode_id, self.step + 1, added_value)
        self._write("snapshots", following)
        record = {"schema_version": SCHEMA, "episode_id": self.episode_id, "step": self.step,
                  "state_t_ref": self.current["snapshot_id"],
                  "state_tp1_ref": following["snapshot_id"], "behavior": self.behavior,
                  "events": self.events, "reward_legacy": float(reward),
                  "reward_progress": 0.0, "reward_train": float(reward),
                  "env_terminated": bool(terminal), "rollout_truncated": False,
                  "end_reason": "legacy_episode_horizon" if terminal else None}
        self._write("transitions", record)
        for packet_id, trace in self.traces.items():
            if trace["delivered_step"] is not None and trace["delivered_step"] < self.step + 1:
                continue
            own_event = next((index for index, event in enumerate(self.events)
                              if event["packet_id"] == packet_id), None)
            trace["prefix"].append({"step": self.step + 1,
                "snapshot_ref": following["snapshot_id"], "node": trace["node"],
                "event_ref": None if own_event is None else [self.step, own_event]})
        self.steps += 1
        self.current = following
        if terminal:
            delivered = 0
            for trace in self.traces.values():
                trace["censored"] = trace["outcome"] is None
                trace["end_reason"] = "delivered" if trace["outcome"] == 1 else "collection_horizon_unresolved"
                trace["last_observed_step"] = trace["prefix"][-1]["step"]
                delivered += trace["outcome"] == 1
                self._write("packet_traces", trace)
            self.episodes.append({"episode_id": self.episode_id, "steps": self.step + 1,
                                  "generated": len(self.traces), "delivered": delivered,
                                  "unresolved": len(self.traces) - delivered})
            self.current = None
            self._manifest(False)

    def close(self):
        for stream in self.streams.values():
            stream.close()
        complete = self.current is None and len(self.episodes) == self.config["num_episodes"]
        self._manifest(complete)
