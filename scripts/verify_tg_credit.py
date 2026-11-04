"""Verify exact seed-33 replays and summarize observational credit evidence."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def tensor_hash(state):
    h = hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def describe(rows):
    if not rows:
        return {"count":0}
    return {"count":len(rows),"positive_advantage":sum(r["advantage_normalized"] > 0 for r in rows),
        "positive_advantage_and_probability_increase":sum(r["advantage_normalized"] > 0 and r["p_selected_change"] > 1e-6 for r in rows),
        **{key+"_mean":statistics.mean(r[key] for r in rows)
           for key in ("advantage_raw","advantage_normalized","p_selected_change","reward","return","value")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--report",required=True)
    p.add_argument("--failed-root",required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    tests = read(root/"test_report.json")
    require(tests["success"] and sum(s["tests"] for s in tests["stages"]) == 7
            and not any(s[k] for s in tests["stages"] for k in ("failures","errors","skipped")),"New regression tests failed")
    reports,rows_by_epoch,artifacts,policies = {},{},{},{}
    for epochs in (1,4):
        directory = root/("epoch"+str(epochs))
        report = read(directory/"report.json")
        require(report["success"] and all(report[k] for k in
            ("exact_actor_critic","exact_training_physics","exact_reward_csv","exact_training_updates")),"Replay failed")
        require(report["training_seed"] == 33 and report["ppo_epoch"] == epochs
                and report["environment_steps"] == 1600,"Wrong replay budget")
        require(sha(ROOT/"scripts/diagnose_tg_credit.py") == report["script_sha256"]
                and sha(ROOT/"src/buoy_network/credit_diagnostics.py") == report["helper_sha256"],"Untested reporter changed")
        require(sha(directory/"training_credit.jsonl") == report["training_rows_sha256"],"Credit rows changed")
        manifest = read(Path(report["suite"])/"manifest.json")
        job = next(j for j in manifest["jobs"] if j["key"] == "A_seed_33")
        reference,current = Path(job["output"]),directory/"replay/moves_5"
        for name in ("TG_physical_metrics.jsonl","TG_reward_components.csv"):
            require((reference/"logs"/name).read_bytes() == (current/"logs"/name).read_bytes(),"Physical/reward replay changed")
        a,b = [torch.load(str(t/"checkpoints/tg_policy_latest.pt"),map_location="cpu",weights_only=True) for t in (reference,current)]
        policies[epochs] = a["actor"]
        require(all(tensor_hash(a[k]) == tensor_hash(b[k]) for k in ("actor","critic")),"Saved replay weights changed")
        require(read(reference/"logs/TG_policy_report.json")["training_updates"] == read(current/"logs/TG_policy_report.json")["training_updates"],"PPO records changed")
        rows = [json.loads(line) for line in (directory/"training_credit.jsonl").read_text().splitlines()]
        require(len(rows) == report["row_count"],"Missing active rows")
        for row in rows:
            require(math.isclose(row["advantage_raw"],row["return"]-row["value"],abs_tol=2e-6,rel_tol=1e-5),"Wrong raw advantage")
            require(row["observation"][-1] == row["sender"] and row["node_self"][-1] == 0
                    and row["node_sink"][-1] == 1,"Sender/Sink features inconsistent")
        by_episode = [[r for r in rows if r["episode"] == e] for e in range(20)]
        for episode in by_episode:
            raw = [r["advantage_raw"] for r in episode]
            mean,std = statistics.mean(raw),statistics.pstdev(raw)
            require(all(math.isclose(r["advantage_normalized"],(r["advantage_raw"]-mean)/(std+1e-5),abs_tol=5e-5,rel_tol=1e-5)
                        for r in episode),"Standardized advantage differs from sampled active set")
        with (reference/"logs/TG_reward_components.csv").open(newline="",encoding="utf-8") as stream:
            rewards = list(csv.DictReader(stream))
        mc = {}
        for episode in range(20):
            sequence = [r for r in rewards if int(r["episode"]) == episode]
            require([int(r["step"]) for r in sequence] == list(range(80)) and sequence[-1]["terminated"] == "True", "Incomplete task window")
            total = 0.
            for reward in reversed(sequence):
                total = float(reward["reward_train"])+.99*total
                mc[(episode,int(reward["step"]))] = total
        calibration = {}
        for name,start,end in (("all_active",0,80),("steps0_19",0,20),("steps20_59",20,60),("steps60_79",60,80)):
            selected = [r for r in rows if start <= r["step"] < end]
            calibration[name] = {"count":len(selected),
                "value_minus_mc_mean":statistics.mean(r["value"]-mc[(r["episode"],r["step"])] for r in selected),
                "value_mc_rmse":math.sqrt(statistics.mean((r["value"]-mc[(r["episode"],r["step"])])**2 for r in selected)),
                "gae_target_minus_mc_mean":statistics.mean(r["return"]-mc[(r["episode"],r["step"])] for r in selected),
                "centered_positive_but_raw_negative":sum(r["advantage_normalized"] > 0 and r["advantage_raw"] < 0 for r in selected)}
        reports[str(epochs)] = {"row_count":len(rows),"exact_replay":True,"on_policy_mc_calibration":calibration,
            "all":describe(rows),"sink_attempts":describe([r for r in rows if r["action"] == 24]),
            "sink_deliveries":describe([r for r in rows if r["action"] == 24 and r["ack"] is True]),
            "sink_legal_not_chosen":describe([r for r in rows if r["sink_legal"] and r["action"] != 24]),
            "successful_revisits":describe([r for r in rows if r["ack"] is True and r["revisit"]]),
            "revisits_during_other_delivery":describe([r for r in rows if r["ack"] is True and r["revisit"] and r["other_deliveries"] > 0]),
            "by_rollout":report["rollouts"],
            "by_action":{str(a):describe([r for r in rows if r["action"] == a]) for a in (17,24,25)}}
        rows_by_epoch[epochs] = rows
        artifacts[str(epochs)] = {"path":str(directory),"report_sha256":sha(directory/"report.json"),
                                  "rows_sha256":sha(directory/"training_credit.jsonl")}
    first = [[r for r in rows_by_epoch[e] if r["episode"] == 0] for e in (1,4)]
    remove = ("p_selected_change","p_sink_change")
    require([{k:v for k,v in r.items() if k not in remove} for r in first[0]] ==
            [{k:v for k,v in r.items() if k not in remove} for r in first[1]],"First rollout samples/credit were not identical")
    first_summary = {str(e):{"all":describe(first[i]),
        "successful_revisits":describe([r for r in first[i] if r["ack"] is True and r["revisit"]]),
        "positive_revisits":describe([r for r in first[i] if r["ack"] is True and r["revisit"] and r["advantage_normalized"] > 0])}
        for i,e in enumerate((1,4))}
    cross_root = root/"epoch4"
    cross = [json.loads(line) for line in (cross_root/"cross_probabilities.jsonl").read_text().splitlines()]
    require(read(cross_root/"report.json")["frozen_same_input"]["cross_rows_sha256"] == sha(cross_root/"cross_probabilities.jsonl"),"Cross rows changed")
    actual_trace = [json.loads(line) for line in (cross_root/"frozen_evaluation/slots.jsonl").read_text().splitlines()]
    for row in cross:
        frame,i = actual_trace[row["slot"]],row["sender"]
        require(frame["actions"][i] == row["action"] and frame["probabilities"][i] == row["probabilities"],"Cross probe changed executed probabilities")
        require(row["control_action"] == max(range(26),key=lambda a:row["control_probabilities"][a]),"Wrong counterfactual argmax")
    sink_cases = [r for r in cross if r["sink_legal"]]
    diagnostic = read(cross_root/"frozen_evaluation/report.json")
    require(diagnostic["pilot_metrics_exact_match"] and diagnostic["weights_unchanged"] and diagnostic["optimizer_updates"] == 0,"Cross replay was not frozen")
    revisits = [r for r in rows_by_epoch[4] if r["ack"] is True and r["revisit"] and r["advantage_normalized"] > .2 and r["p_selected_change"] > .001]
    decomposition = []
    for sender in (18,23):
        r = next(r for r in sink_cases if r["sender"] == sender and r["action"] == 17)
        item = {"sender":sender}
        for e,field in ((1,"control_probabilities"),(4,"probabilities")):
            bias = policies[e]["act.action_out.linear.bias"]
            bias_gap = float(bias[17]-bias[24])
            total_gap = math.log(r[field][17]/r[field][24])
            item[str(e)] = {"p17":r[field][17],"p_sink":r[field][24],"head_bias_gap_17_minus_sink":bias_gap,
                            "logit_gap_17_minus_sink":total_gap,"state_feature_term_gap":total_gap-bias_gap}
        decomposition.append(item)
    receipt = {"success":True,"scope":"read-only seed33 routing/credit diagnosis; no policy fix or performance acceptance",
        "function_tests":tests,"exact_training_replays":2,"training_replay_steps":3200,
        "previous_failed_capture":read(Path(args.failed_root)/"failure_report.json"),
        "capture_minimal_probe":read(Path(args.failed_root)/"capture_probe.json"),
        "frozen_cross_replay_steps":160,"artifacts":artifacts,"reports":reports,"first_rollout_identical_before_update":True,
        "first_rollout_comparison":first_summary,"same_input_sink_cases":sink_cases,
        "same_input_logit_decomposition":decomposition,
        "current_sink_misses":sum(r["action"] != 24 for r in sink_cases),"control_sink_misses":sum(r["control_action"] != 24 for r in sink_cases),
        "representative_positive_revisits":revisits[:5],"observed_closed_cycles":diagnostic["analysis"]["top_closed_cycles"],
        "gae_terminal_residual_attenuation_79":(.99*.95)**79,
        "limitations":["team advantage is not causal per-action contribution", "probability change includes all shared-parameter updates and entropy regularization",
            "same-input control prediction does not predict an alternative complete rollout", "positive standardized advantage can occur with negative raw advantage",
            "GAE residual attenuation is not total terminal-penalty attenuation; an accurate critic can propagate future costs",
            "sampled Monte Carlo returns are noisy on-policy outcomes, not exact conditional state values; active decisions are correlated", "no new full regression suite or formal experiment"],
        "validator_sha256":sha(Path(__file__))}
    with Path(args.report).open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print("PASS: 7 new tests; 2 exact training replays; first-rollout credit identical; frozen same-input comparison exact")
    print("Sink misses: {} current, {} control on {} identical states".format(receipt["current_sink_misses"],receipt["control_sink_misses"],len(sink_cases)))


if __name__ == "__main__":
    main()
