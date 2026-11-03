"""Independent reward audit against actual physical delivery timestamps."""
import csv
import json
import math
from collections import Counter
from pathlib import Path


def audit_reward_csv(log_dir, spec):
    root = Path(log_dir)
    physical = [json.loads(line) for line in (root/"TG_physical_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    with (root/"TG_reward_components.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with_pm = (root/"TG_PM_rewards.csv").exists()
    if with_pm:
        with (root/"TG_PM_rewards.csv").open(encoding="utf-8", newline="") as stream:
            pm = list(csv.DictReader(stream))
        if len(pm) != len(rows):
            raise ValueError("Missing PM reward rows")
    cursor, totals = 0, []
    def close(a, b):
        if not math.isfinite(float(a)) or not math.isclose(float(a), float(b), abs_tol=1e-9, rel_tol=1e-9):
            raise ValueError("Reward channel mismatch: {} vs {}".format(a, b))
    for ep, metrics in enumerate(physical):
        count, horizon = metrics["generated_packets"], metrics["horizon"]
        deliveries = Counter(metrics["success_delays"])
        alive, total = count, 0.
        for step in range(1, horizon+1):
            row = rows[cursor]
            if (int(row["episode"]) != ep or int(row["step"]) != step-1 or int(row["next_step"]) != step
                    or row["reward_version"] != spec["version"]
                    or int(row["initial_packets"]) != count or int(row["unresolved_before"]) != alive
                    or int(row["delivered_this_step"]) != deliveries[step]
                    or int(row["unresolved_after"]) != alive-deliveries[step]
                    or row["terminated"] != str(step == horizon) or row["rollout_truncated"] != "False"):
                raise ValueError("Reward event counts/boundaries differ from physical delivery times")
            scale = max(count, 1)
            expected = (spec["delivery_weight"]*deliveries[step]/scale,
                        -spec["delay_weight"]*alive/(scale*horizon),
                        -spec["unfinished_weight"]*(alive-deliveries[step])/scale if step == horizon else 0.)
            for field, value in zip(("reward_delivery", "reward_delay", "reward_window_end"), expected):
                close(row[field], value)
            close(row["reward_base"], sum(expected))
            close(row["reward_train"], float(row["reward_base"])+float(row["reward_progress"]))
            if with_pm:
                p = pm[cursor]
                if int(p["episode"]) != ep or int(p["step"]) != step-1:
                    raise ValueError("PM reward time mismatch")
                for x,y in (("reward_base", "reward_env"), ("reward_progress", "reward_progress"), ("reward_train", "reward_train")):
                    close(row[x], p[y])
            else:
                close(row["reward_progress"], 0.)
            total += float(row["reward_base"])
            alive -= deliveries[step]
            cursor += 1
        if alive != metrics["unresolved_packets"] or metrics["true_drop_rate"] is not None:
            raise ValueError("Unresolved packets incorrectly reclassified")
        # Closed form: each delivered packet waits its actual delay; others wait H.
        expected_total = (spec["delivery_weight"]*metrics["delivered_packets"]/max(count,1)
                          -spec["delay_weight"]*(sum(metrics["success_delays"])+alive*horizon)/(max(count,1)*horizon)
                          -spec["unfinished_weight"]*alive/max(count,1))
        close(total, expected_total)
        totals.append(total)
    if cursor != len(rows):
        raise ValueError("Extra reward rows")
    return {"episodes": len(physical), "slots": cursor, "base_returns": totals,
            "physical_event_counts_and_reward_channels_verified": True}
