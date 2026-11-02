"""Standard-library audit and WM sample reader for legacy TG exported data."""

import copy
import json
import math
from collections import deque
from pathlib import Path

from .transition_data import SCHEMA


def read_json(path):
    def bad_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=bad_constant)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
                # Also rejects nonfinite values in ordinary JSON parser output.
                json.dumps(value, allow_nan=False)
                yield value
            except (ValueError, TypeError) as exc:
                raise ValueError("{} line {}: {}".format(path, number, exc))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_run(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    require(manifest["schema_version"] == SCHEMA and manifest["complete"], "Incomplete/incompatible dataset")
    disk_bytes = sum((root / (name + ".jsonl")).stat().st_size
                     for name in ("snapshots", "transitions", "packet_traces"))
    require(disk_bytes == manifest["jsonl_bytes"] <= manifest["budgets"]["max_bytes"], "Disk byte budget/count mismatch")
    require(manifest["steps"] <= manifest["budgets"]["max_steps"]
            and manifest["packets"] <= manifest["budgets"]["max_packets"], "Step/packet budget exceeded")
    n = manifest["config"]["grid_size"] ** 2
    snapshots = {}
    for state in read_rows(root / "snapshots.jsonl"):
        key = state["snapshot_id"]
        require(key not in snapshots, "Duplicate snapshot")
        actor = state["actor_inputs"]
        require(len(actor["obs"]) == n and all(len(row) == n + 1 for row in actor["obs"]), "Observation shape")
        require(len(actor["node_features"]) == n and all(len(row) == 4 for row in actor["node_features"]), "Node shape")
        require(len(actor["distances"]) == n and all(len(row) == n for row in actor["distances"]), "Distance shape")
        require(len(actor["available_actions"]) == n and all(len(row) == n + 1 for row in actor["available_actions"]), "Mask shape")
        require(all(row[-1] == 1 and all(value in (0, 1) for value in row)
                    for row in actor["available_actions"]), "WAIT/mask invalid")
        snapshots[key] = state
    episodes = {row["episode_id"]: row for row in manifest["episodes"]}
    require(len(episodes) == len(manifest["episodes"]), "Duplicate episode")
    counters = {key: 0 for key in episodes}
    moves, deliveries = {}, {}
    attempted = successful = waits = count = 0
    transitions = {}
    for row in read_rows(root / "transitions.jsonl"):
        episode, step = row["episode_id"], row["step"]
        require(episode in episodes and step == counters[episode], "Wrong transition order")
        require(row["schema_version"] == SCHEMA, "Transition schema")
        before, after = snapshots[row["state_t_ref"]], snapshots[row["state_tp1_ref"]]
        require(row["state_t_ref"] == "{}:{}".format(episode, step)
                and row["state_tp1_ref"] == "{}:{}".format(episode, step + 1), "Wrong snapshot reference")
        require(before["step"] == step and after["step"] == step + 1, "Wrong t/t+1")
        behavior = row["behavior"]
        require(all(len(behavior[key]) == n and all(len(v) == 1 for v in behavior[key])
                    for key in ("actions", "old_log_probs", "old_values")), "Behavior shape")
        require(len(row["events"]) == n, "Missing node events")
        total = 0.0
        for node, event in enumerate(row["events"]):
            selected = behavior["actions"][node][0]
            require(type(selected) is int and 0 <= selected <= n, "Action index")
            require(event["sender"] == node and selected == event["action"], "Executed action mismatch")
            require(before["actor_inputs"]["available_actions"][node][selected] == 1, "Illegal action")
            require(math.isclose(sum(event["reward_components"].values()), event["reward_legacy"], abs_tol=1e-5), "Reward components")
            total += event["reward_legacy"]
            if event["attempted"]:
                attempted += 1
                require(selected != n and type(event["ack"]) is bool and event["packet_key"] is not None, "Invalid attempt")
                require(event["packet_key"] == [episode, event["packet_id"]], "Packet key mismatch")
                queue = event["queue_at_attempt"]
                require(len(queue["before"]) == len(queue["after"]) == 2 and min(queue["before"]) >= 0,
                        "Queue event shape")
                expected = ([queue["before"][0] - 1, queue["before"][1] + 1]
                            if event["ack"] else queue["before"])
                require(queue["after"] == expected, "Attempt queue change inconsistent with ACK")
                if event["ack"]:
                    successful += 1
                    packet_key = (episode, event["packet_id"])
                    packet_moves = moves.setdefault(packet_key, [])
                    require(not packet_moves or packet_moves[-1][0] != step, "Multiple hops per packet/slot")
                    packet_moves.append((step, node, selected))
                    if event["delivered_this_step"]:
                        require(packet_key not in deliveries, "Duplicate delivery")
                        deliveries[packet_key] = step + 1
            else:
                require(event["ack"] is None and event["queue_at_attempt"] is None, "Unattempted ACK label")
            if selected == n:
                waits += 1
                require(not event["attempted"], "WAIT cannot attempt transmission")
        require(math.isclose(total, row["reward_legacy"], abs_tol=1e-5), "Team reward mismatch")
        require(row["reward_progress"] == 0 and row["reward_train"] == row["reward_legacy"], "Unexpected shaping")
        terminal = step + 1 == episodes[episode]["steps"]
        require(row["env_terminated"] == terminal and row["rollout_truncated"] is False, "Termination mismatch")
        require(row["end_reason"] == ("legacy_episode_horizon" if terminal else None), "End reason")
        counters[episode] += 1
        count += 1
        transitions[(episode, step)] = row
    require(count == manifest["steps"] and all(counters[k] == episodes[k]["steps"] for k in episodes), "Missing transitions")
    require(len(snapshots) == count + len(episodes), "Snapshots not deduplicated/complete")
    packet_keys = set()
    delivered_count = unresolved_count = 0
    per_episode = {key: [0, 0] for key in episodes}
    for trace in read_rows(root / "packet_traces.jsonl"):
        packet_key = tuple(trace["packet_key"])
        require(packet_key not in packet_keys and packet_key[0] in episodes, "Duplicate/unknown packet")
        packet_keys.add(packet_key)
        episode = packet_key[0]
        per_episode[episode][0] += 1
        route = [trace["source"]]
        for step, sender, receiver in moves.get(packet_key, []):
            require(route[-1] == sender, "Packet route is not continuous")
            route.append(receiver)
        require(route == trace["route"] and trace["node"] == route[-1], "Archived route mismatch")
        if trace["outcome"] == 1:
            delivered_count += 1
            per_episode[episode][1] += 1
            require(packet_key in deliveries and trace["delivered_step"] == deliveries[packet_key], "Delivery time mismatch")
            require(route[-1] == trace["destination"] and not trace["censored"]
                    and trace["end_reason"] == "delivered", "Bad delivery outcome")
            end = trace["delivered_step"]
        else:
            unresolved_count += 1
            require(trace["outcome"] is None and trace["censored"] and trace["delivered_step"] is None
                    and packet_key not in deliveries, "Unresolved packet labelled failed/delivered")
            end = episodes[episode]["steps"]
        require(trace["last_observed_step"] == end, "Wrong last observation")
        require([prefix["step"] for prefix in trace["prefix"]] == list(range(end + 1)), "Incomplete prefix")
        node = trace["source"]
        for prefix in trace["prefix"]:
            prefix_step = prefix["step"]
            for move_step, sender, receiver in moves.get(packet_key, []):
                if move_step + 1 == prefix_step:
                    node = receiver
            require(prefix["node"] == node and prefix["snapshot_ref"] == "{}:{}".format(episode, prefix_step), "Prefix mismatch")
            if prefix["event_ref"] is not None:
                event_step, event_index = prefix["event_ref"]
                require(event_step + 1 == prefix_step
                        and transitions[(episode, event_step)]["events"][event_index]["packet_key"] == list(packet_key),
                        "Prefix event mismatch")
    require(len(packet_keys) == manifest["packets"] and set(moves).issubset(packet_keys), "Packet conservation")
    for key, (generated, delivered) in per_episode.items():
        require(generated == episodes[key]["generated"] and delivered == episodes[key]["delivered"]
                and generated - delivered == episodes[key]["unresolved"], "Episode packet counts")
    return {"passed": True, "root": str(root), "split": manifest["split"],
            "scenario_seed": manifest["scenario_seed"], "episodes": len(episodes), "steps": count,
            "snapshots": len(snapshots), "packets": len(packet_keys), "delivered": delivered_count,
            "unresolved": unresolved_count, "attempts": attempted, "ack_successes": successful,
            "waits": waits, "bytes": manifest["jsonl_bytes"]}


def run_roots(root, split=None):
    root = Path(root).resolve()
    if (root / "dataset_manifest.json").exists():
        manifest = read_json(root / "dataset_manifest.json")
        require(manifest.get("schema_version") == "legacy_tg_collection_v1", "Incompatible collection schema")
        require(manifest["complete"], "Collection is incomplete")
        seen = {}
        seen_paths = set()
        for run in manifest["runs"]:
            require(run["scenario_seed"] not in seen or seen[run["scenario_seed"]] == run["split"],
                    "Scenario seed reused across dataset splits")
            seen[run["scenario_seed"]] = run["split"]
            candidate = (root / run["data_path"]).resolve()
            require(root in candidate.parents, "Dataset path escapes collection")
            require(candidate not in seen_paths, "Dataset source run reused")
            seen_paths.add(candidate)
            actual = read_json(candidate / "manifest.json")
            require(actual["split"] == run["split"] and actual["scenario_seed"] == run["scenario_seed"], "Split/seed metadata mismatch")
            require(actual["scenario_seed_override"], "Collection needs explicit scenario seeds")
            if "behavior" in run:
                require(actual["behavior"] == run["behavior"], "Behavior metadata mismatch")
            if split is None or run["split"] == split:
                yield candidate
    else:
        manifest = read_json(root / "manifest.json")
        if split is None or manifest["split"] == split:
            yield root


def iter_wm_samples(root, split="train", history_length=4):
    """Only own past actions and decision-time actor inputs enter WM inputs.

    Executed current joint actions, next state and outcomes are labels/metadata,
    never placed in the history. No normalization is fitted on validation/test.
    """
    require(type(history_length) is int and history_length > 0, "history_length must be positive")
    for run_root in run_roots(root, split):
        manifest = read_json(run_root / "manifest.json")
        require(manifest["complete"], "Dataset incomplete")
        states = {row["snapshot_id"]: row for row in read_rows(run_root / "snapshots.jsonl")}
        current_episode = None
        for row in read_rows(run_root / "transitions.jsonl"):
            before, after = states[row["state_t_ref"]], states[row["state_tp1_ref"]]
            if current_episode != row["episode_id"]:
                current_episode = row["episode_id"]
                history = [deque(maxlen=history_length) for _ in row["events"]]
                previous = [None] * len(row["events"])
            for node, event in enumerate(row["events"]):
                actor = before["actor_inputs"]
                history[node].append({"step": row["step"], "own_obs": actor["obs"][node],
                                      "node_features": actor["node_features"], "distances": actor["distances"],
                                      "available_actions": actor["available_actions"][node],
                                      "own_previous_action": previous[node]})
                previous[node] = event["action"]
                if event["packet_id"] is None:
                    continue
                yield {"episode_id": current_episode, "step": row["step"], "agent_id": node,
                       "packet_key": event["packet_key"],
                       "inputs": {"history": copy.deepcopy(list(history[node])), "candidate_action": event["action"]},
                       "targets": {"next_own_obs": after["actor_inputs"]["obs"][node],
                                   "next_node_features": after["actor_inputs"]["node_features"],
                                   "ack": event["ack"], "ack_loss_mask": event["attempted"],
                                   "queue_at_attempt": event["queue_at_attempt"],
                                   "delivered_this_step": event["delivered_this_step"],
                                   "dropped_this_step": event["dropped_this_step"],
                                   "reward_legacy_sender": event["reward_legacy"]}}
