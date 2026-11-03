"""Audit and describe baseline PPO epoch budgets at fixed environment interactions."""
import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src.buoy_network.ablation_suite import read,digest,summarize,require
from src.buoy_network.wm_runtime import tensor_digest
from src.buoy_network.task_reward_audit import audit_reward_csv
from verify_tg_ablation import equivalent


def state_hashes(root):
    payload = torch.load(str(Path(root)/"checkpoints/tg_policy_latest.pt"),map_location="cpu",weights_only=True)
    return {key:tensor_digest(payload[key]) for key in ("actor","critic")}


def summary_stat(values):
    values = [x for x in values if x is not None]
    return {"n_defined":len(values),"mean":statistics.mean(values) if values else None,
            "sample_sd":statistics.stdev(values) if len(values) > 1 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch1-suite",required=True)
    parser.add_argument("--epoch4-suite",required=True)
    parser.add_argument("--old-budget-root",required=True)
    parser.add_argument("--test-reports",nargs="+",required=True)
    parser.add_argument("--report",required=True)
    args = parser.parse_args()
    manifests,hashes,rows = {},{},[]
    for epochs,path in ((1,args.epoch1_suite),(4,args.epoch4_suite)):
        root = Path(path).resolve()
        m = read(root/"manifest.json")
        cfg = m["config"]
        require(m["state"] == "completed" and len(m["jobs"]) == 6 and cfg["groups"] == ["A"]
                and cfg["purpose"] == "diagnostic" and cfg["ppo_epoch"] == epochs
                and cfg["num_mini_batch"] == 2 and cfg["train_episodes"] == 20
                and cfg["eval_episodes"] == 2 and cfg["eval_seeds"] == [7001]
                and cfg["learning_rate_schedule"] == "linear","Wrong baseline diagnostic protocol")
        for name,checksum in m["source"]["files_sha256"].items():
            require(digest(root/"source_snapshot"/name) == checksum and digest(ROOT/name) == checksum,"Untested source change: "+name)
        calculated = summarize(m)
        require(equivalent(calculated,read(root/"summary.json")),"Summary differs from raw jobs")
        original_sha = digest(root/"summary.json")
        attempts = {j["key"]:len(j["attempts"]) for j in m["jobs"]}
        result = subprocess.run([cfg["python"],str(ROOT/"scripts/run_tg_ablation.py"),"--resume",str(root),"--mode","run"],
                                cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60)
        require(result.returncode == 0 and original_sha == digest(root/"summary.json")
                and attempts == {j["key"]:len(j["attempts"]) for j in read(root/"manifest.json")["jobs"]},"Resume repeated or changed jobs")
        manifests[epochs],hashes[epochs] = m,original_sha
        commands = []
        for seed in cfg["train_seeds"]:
            train = next(j for j in m["jobs"] if j["kind"] == "training" and j["train_seed"] == seed)
            evaluation = next(j for j in m["jobs"] if j["kind"] == "evaluation" and j["train_seed"] == seed)
            tr,ev = Path(train["output"]),Path(evaluation["output"])
            report,evaluated = read(tr/"logs/TG_policy_report.json"),read(ev/"logs/TG_policy_report.json")
            actual = read(tr/"logs/TG_MAPPO_environment.json")["config"]
            records = report["training_updates"]
            require(len(records) == report["profile"]["ppo_updates"] == 20 and report["profile"]["collect_calls"] == 1600,
                    "Wrong collected data/rollout count")
            require(report["training_implementation"] == "tg_ppo_v3_lr_switch" and report["wm"] is None and report["pm"] is None,
                    "Wrong learner/auxiliary scope")
            for e,r in enumerate(records):
                expected = (e+1)*epochs*2
                require(r["actor_optimizer_steps"] == r["critic_head_optimizer_steps"] == expected and r["critic_head_in_optimizer"],
                        "Measured optimizer budget mismatch")
                for field,initial in (("actor_lr",actual["lr"]),("critic_lr",actual["critic_lr"])):
                    require(math.isclose(r[field],initial-initial*(e/20.),rel_tol=1e-14,abs_tol=1e-16),"Learning-rate plan changed")
                require(all(math.isfinite(r[k]) for k in ("value_loss","policy_loss","dist_entropy","actor_grad_norm","critic_grad_norm","ratio","approx_kl","clip_fraction")),
                        "Nonfinite training diagnostic")
                require(r["approx_kl"] >= 0 and 0 <= r["clip_fraction"] <= 1,"Invalid policy-drift metric")
            require(evaluated["checkpoint_sha256"] == report["checkpoint_sha256"] and evaluated["profile"]["ppo_updates"] == 0
                    and evaluated["actor_unchanged"] and evaluated["critic_unchanged"],"Frozen evaluation mismatch")
            audit_reward_csv(tr/"logs",report["reward_contract"])
            audit_reward_csv(ev/"logs",report["reward_contract"])
            output = root/"diagnostics"/("A_seed_"+str(seed))
            command = [cfg["python"],str(root/"source_snapshot/scripts/diagnose_tg_policy.py"),"--suite",str(root),
                       "--worker","A",str(seed),"--stage","trained","--decode","greedy","--output-run",str(output)]
            reused = (output/"report.json").exists()
            if not (output/"report.json").exists():
                output.parent.mkdir(parents=True,exist_ok=True)
                with (output.parent/(output.name+".log")).open("xb") as log:
                    completed = subprocess.run(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,timeout=180)
                require(completed.returncode == 0,"Action replay failed: "+str(output))
            diagnostic = read(output/"report.json")
            require(diagnostic["weights_unchanged"] and diagnostic["optimizer_updates"] == 0
                    and diagnostic["pilot_metrics_exact_match"] and diagnostic["policy_sha256"] == report["checkpoint_sha256"]
                    and diagnostic["stage"] == "trained" and diagnostic["decode"] == "greedy"
                    and diagnostic["training_seed"] == seed,"Diagnostic replay changed behavior/model")
            require((output/"moves_5/logs/TG_physical_metrics.jsonl").read_bytes() == (ev/"logs/TG_physical_metrics.jsonl").read_bytes(),
                    "Instrumented physical replay differs")
            commands.append({"reproduction_command":command,"reused_complete_replay":reused,"exit_code":0})
            (root/"diagnostic_commands.json").write_text(json.dumps(commands,indent=2),encoding="utf-8")
            a = diagnostic["analysis"]
            item = {"ppo_epoch":epochs,"train_seed":seed,"generated":a["generated"],"delivered":a["delivered"],
                    "delivery_rate":a["delivered"]/a["generated"],"initial_actor_sha256":diagnostic["initial_actor_sha256"],
                    "active_wait_fraction":a["active_wait_fraction"],"sink_legal":a["sink_legal_count"],"sink_chosen":a["sink_chosen_count"],
                    "sink_choice_fraction":a["sink_chosen_count"]/a["sink_legal_count"] if a["sink_legal_count"] else None,
                    "normalized_entropy":a["mean_normalized_entropy"],"top_probability_gap":a["mean_top_probability_gap"],
                    "loop_transfers":a["revisit_transfers"],"loop_fraction":a["revisit_fraction_of_successes"],
                    "physical_reachable_fraction":a["physical_sink_reachable_fraction"],
                    "training_output":str(tr),"evaluation_output":str(ev),"diagnostic_output":str(output),
                    "actor_optimizer_steps":records[-1]["actor_optimizer_steps"],"critic_optimizer_steps":records[-1]["critic_head_optimizer_steps"],
                    "training_metrics":{},"training_updates":records,"diagnostic":diagnostic}
            for field in ("value_loss","policy_loss","dist_entropy","actor_grad_norm","critic_grad_norm","ratio","approx_kl","clip_fraction"):
                values = [r[field] for r in records]
                item["training_metrics"][field] = {"mean":statistics.mean(values),"max_rollout_mean":max(values),"last5_mean":statistics.mean(values[-5:])}
            rows.append(item)
    require(manifests[1]["source"]["files_sha256"] == manifests[4]["source"]["files_sha256"],"Budget conditions used different source")
    c1,c4 = manifests[1]["config"],manifests[4]["config"]
    require({k:v for k,v in c1.items() if k != "ppo_epoch"} == {k:v for k,v in c4.items() if k != "ppo_epoch"},"More than PPO epoch changed")
    parity,pairs = [],[]
    for seed in c1["train_seeds"]:
        a,b = [next(r for r in rows if r["ppo_epoch"] == e and r["train_seed"] == seed) for e in (1,4)]
        require(a["initial_actor_sha256"] == b["initial_actor_sha256"],"Paired seeds initialized different Actors")
        first = [json.loads((Path(r["training_output"])/"logs/TG_physical_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]) for r in (a,b)]
        require(first[0] == first[1],"First sampled rollout differed before any budget-dependent update")
        old_train = Path(args.old_budget_root)/"training/runs"/("A_seed_{}_e20".format(seed))/"moves_5"
        old_eval = Path(args.old_budget_root)/"evaluation/runs"/("A_seed_{}_e20_scene7001".format(seed))/"moves_5"
        require(state_hashes(a["training_output"]) == state_hashes(old_train),"Instrumentation changed 1-epoch policy weights")
        for current,previous in ((Path(a["training_output"]),old_train),(Path(a["evaluation_output"]),old_eval)):
            require((current/"logs/TG_physical_metrics.jsonl").read_bytes() == (previous/"logs/TG_physical_metrics.jsonl").read_bytes(),
                    "1-epoch control no longer matches old physical data")
        parity.append({"seed":seed,"actor_critic_exact":True,"training_evaluation_physics_exact":True})
        pairs.append({"seed":seed,"epoch1_delivered":a["delivered"],"epoch4_delivered":b["delivered"],
                      "delivery_rate_change":b["delivery_rate"]-a["delivery_rate"]})
    tests,count = [],0
    for path in args.test_reports:
        report = read(path)
        require(report["success"],"Failed regression report")
        for s in report["stages"]:
            require(s["tests"] > 0 and not any(s.get(k,0) for k in ("failures","errors","skipped")),"Failed/skipped regression")
            count += s["tests"]
        tests.append({"path":str(Path(path).resolve()),"sha256":digest(path),"report":report})
    require(count == 123,"Expected 123 tests")
    aggregate = {str(e):{field:summary_stat([r[field] for r in rows if r["ppo_epoch"] == e])
                for field in ("delivery_rate","normalized_entropy","active_wait_fraction","sink_choice_fraction","loop_fraction","loop_transfers")}
                for e in (1,4)}
    receipt = {"success":True,"scope":"baseline update-budget development diagnostic; no WM/PM or convergence claim",
        "controlled_change":"ppo_epoch 1 vs 4; minibatches 2, episodes 20, H80, task_v2, linear LR, trained seeds11/22/33, greedy scene7001 fixed",
        "test_count":count,"tests":tests,"train_eval_jobs":12,"read_only_replays":6,"resume_new_jobs":0,
        "environment_training_steps":9600,"environment_evaluation_steps":960,"extra_diagnostic_replay_steps":960,
        "suite_summary_sha256":hashes,"suites":{str(e):{"path":str(Path(p).resolve()),"manifest_sha256":digest(Path(p)/"manifest.json")} for e,p in ((1,args.epoch1_suite),(4,args.epoch4_suite))},
        "old_epoch1_parity":parity,"paired_results":pairs,"aggregate":aggregate,"by_training_seed":rows,
        "metric_scope":"minibatch pre-step approximate KL and clip fraction, averaged per rollout; not final exact full-rollout KL",
        "limitations":["three trained seeds and one previously used development scene", "more epochs also consume more shuffling RNG; later trajectories naturally differ", "same environment interactions, different optimizer compute budgets", "CPU jobs concurrent; no speed benchmark", "bounded runs do not establish convergence or significance"],
        "validator_sha256":digest(Path(__file__)),"new_test_sha256":digest(ROOT/"tests/test_legacy_ppo_budget.py")}
    with Path(args.report).open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print("PASS: 123 tests, 12 budget jobs, 6 exact frozen action replays, old 1-epoch weights/physics unchanged, 0 resumed jobs")
    print("Receipt: "+str(Path(args.report).resolve()))


if __name__ == "__main__":
    main()
