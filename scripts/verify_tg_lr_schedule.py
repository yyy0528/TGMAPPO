"""Audit LR semantics, two matched six-group pilots and fixed-budget probes."""
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


def check_training(path,kind,episodes):
    root = Path(path)
    report = read(root/"logs/TG_policy_report.json")
    cfg = read(root/"logs/TG_MAPPO_environment.json")["config"]
    require(report["training_implementation"] == "tg_ppo_v3_lr_switch"
            and report["learning_rate_schedule"]["kind"] == kind
            and cfg["use_linear_lr_decay"] == (kind == "linear"),"Wrong training/schedule version")
    require(report["profile"]["ppo_updates"] == episodes and report["profile"]["collect_calls"] == episodes*80
            and cfg["ppo_epoch"] == 1 and cfg["num_mini_batch"] == 2,"Wrong training budget")
    records = report["training_updates"]
    require(len(records) == episodes,"Missing optimizer diagnostics")
    for e,row in enumerate(records):
        for field,initial in (("actor_lr",cfg["lr"]),("critic_lr",cfg["critic_lr"])):
            expected = initial-initial*(e/float(episodes)) if kind == "linear" else initial
            require(math.isclose(row[field],expected,rel_tol=1e-14,abs_tol=1e-16),"Actual optimizer LR mismatch")
        require(row["critic_head_in_optimizer"] and row["critic_head_optimizer_steps"] == 2*(e+1),"Critic-head update regression")
        require(all(math.isfinite(row[k]) for k in ("value_loss","policy_loss","dist_entropy","actor_grad_norm","critic_grad_norm","ratio")),"Nonfinite PPO statistic")
    require(digest(report["checkpoint"]) == report["checkpoint_sha256"],"Policy artifact changed")
    audit_reward_csv(root/"logs",report["reward_contract"])
    return report


def check_evaluation(path,kind):
    root = Path(path)
    report = read(root/"logs/TG_policy_report.json")
    cfg = read(root/"logs/TG_MAPPO_environment.json")["config"]
    require(report["source_learning_rate_schedule"]["kind"] == kind and report["source_training_implementation"] == "tg_ppo_v3_lr_switch",
            "Wrong source training plan")
    require(report["optimizer_updates"] == report["profile"]["ppo_updates"] == 0
            and report["actor_unchanged"] and report["critic_unchanged"],"Evaluation trained or changed parameters")
    require(report["profile"]["collect_calls"] == 160 and cfg["num_episodes"] == 2
            and cfg["episode_length"] == 80,"Wrong evaluation budget")
    require(report["evaluation_optimizer_rates"] == {"actor":[cfg["lr"]],"critic":[cfg["critic_lr"]]},"Frozen evaluation changed optimizer LR")
    audit_reward_csv(root/"logs",report["reward_contract"])
    return report


def weights(path):
    payload = torch.load(str(Path(path)/"checkpoints/tg_policy_latest.pt"),map_location="cpu",weights_only=True)
    return {key:tensor_digest(payload[key]) for key in ("actor","critic")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-suite",required=True)
    parser.add_argument("--linear-suite",required=True)
    parser.add_argument("--old-suite",required=True)
    parser.add_argument("--before-probe",required=True)
    parser.add_argument("--fixed-probe",required=True)
    parser.add_argument("--linear-probe",required=True)
    parser.add_argument("--budget-root",required=True)
    parser.add_argument("--old-budget-root",required=True)
    parser.add_argument("--test-reports",nargs="+",required=True)
    parser.add_argument("--report",required=True)
    args = parser.parse_args()
    manifests,summaries = {},{}
    for kind,path in (("fixed",args.fixed_suite),("linear",args.linear_suite)):
        root = Path(path).resolve()
        m = read(root/"manifest.json")
        require(m["state"] == "completed" and len(m["jobs"]) == 36,"Incomplete six-group suite")
        for name,checksum in m["source"]["files_sha256"].items():
            require(digest(root/"source_snapshot"/name) == checksum and digest(ROOT/name) == checksum,"Untested source change: "+name)
        summary = summarize(m)
        require(equivalent(summary,read(root/"summary.json")),"Summary differs from raw results")
        attempts = {j["key"]:len(j["attempts"]) for j in m["jobs"]}
        summary_sha = digest(root/"summary.json")
        resumed = subprocess.run([m["config"]["python"],str(ROOT/"scripts/run_tg_ablation.py"),"--resume",str(root),"--mode","run"],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60,cwd=str(ROOT))
        require(resumed.returncode == 0 and attempts == {j["key"]:len(j["attempts"]) for j in read(root/"manifest.json")["jobs"]}
                and digest(root/"summary.json") == summary_sha,"Resume reran completed jobs or changed results")
        for job in m["jobs"]:
            if job["kind"] == "training":
                check_training(job["output"],kind,2)
            else:
                check_evaluation(job["output"],kind)
        manifests[kind],summaries[kind] = m,summary
    require(manifests["fixed"]["source"]["files_sha256"] == manifests["linear"]["source"]["files_sha256"],"The paired suites used different source")
    old = read(Path(args.old_suite)/"manifest.json")
    parity = []
    for job in manifests["linear"]["jobs"]:
        previous = next(j for j in old["jobs"] if j["key"] == job["key"])
        physical_equal = (Path(job["output"])/"logs/TG_physical_metrics.jsonl").read_bytes() == (Path(previous["output"])/"logs/TG_physical_metrics.jsonl").read_bytes()
        require(physical_equal,"Explicit linear plan changed old physical result: "+job["key"])
        item = {"key":job["key"],"old_physical_exact":True}
        if job["kind"] == "training":
            require(weights(job["output"]) == weights(previous["output"]),"Explicit linear plan changed policy weights")
            item["actor_critic_exact"] = True
        parity.append(item)
    # With E=2, the only LR difference starts at the final PPO update; sampled
    # physical training trajectories should therefore match between the plans.
    paired = []
    for a in manifests["fixed"]["jobs"]:
        if a["kind"] != "training":
            continue
        b = next(j for j in manifests["linear"]["jobs"] if j["key"] == a["key"])
        require((Path(a["output"])/"logs/TG_physical_metrics.jsonl").read_bytes() == (Path(b["output"])/"logs/TG_physical_metrics.jsonl").read_bytes(),
                "Unexpected rollout difference before final LR-dependent update")
        paired.append({"key":a["key"],"training_physics_exact":True,"final_weights_differ":weights(a["output"]) != weights(b["output"])})
    before,fixed,linear = [read(p) for p in (args.before_probe,args.fixed_probe,args.linear_probe)]
    require(digest(Path(args.old_suite)/"source_snapshot/src/buoy_network/runner.py") == before["source_runner_sha256"],
            "Before-fix source is not anchored to the immutable old archive")
    require(not before["linear_requested"] and before["optimizer_rates"][1]["actor"] == .00025,"Original switch bug not reproduced")
    require(not fixed["linear_requested"] and all(r["actor"] == r["critic"] == .0005 for r in fixed["optimizer_rates"]),"Fixed switch probe failed")
    require(linear["linear_requested"] and before["optimizer_rates"] == linear["optimizer_rates"]
            and all(before[k] == linear[k] for k in ("actor_sha256","critic_sha256")),"Enabled plan failed old-behavior parity")
    budget = []
    for seed in (11,22,33):
        train = Path(args.budget_root)/"training/runs"/("A_seed_{}_e20_fixed".format(seed))/"moves_5"
        evaluation = Path(args.budget_root)/"evaluation/runs"/("A_seed_{}_e20_fixed_scene7001".format(seed))/"moves_5"
        trained,evaluated = check_training(train,"fixed",20),check_evaluation(evaluation,"fixed")
        require(evaluated["checkpoint_sha256"] == trained["checkpoint_sha256"],"Budget probe loaded wrong model")
        cfg = read(train/"logs/TG_MAPPO_environment.json")["config"]
        ecfg = read(evaluation/"logs/TG_MAPPO_environment.json")["config"]
        require(cfg["seed"] == cfg["scenario_seed"] == seed and not cfg["use_wm_actor"] and not cfg["use_pm_reward"]
                and cfg["num_episodes"] == 20 and cfg["episode_length"] == 80 and cfg["num_moving_buoys"] == 5 and cfg["random_channel"]
                and ecfg["seed"] == ecfg["scenario_seed"] == 7001,"Budget probe protocol mismatch")
        rows = [json.loads(l) for l in (evaluation/"logs/TG_physical_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        previous = Path(args.old_budget_root)/"evaluation/runs"/("A_seed_{}_e20_scene7001".format(seed))/"moves_5"
        old_rows = [json.loads(l) for l in (previous/"logs/TG_physical_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        require(len(rows) == len(old_rows) == 2,"Missing budget evaluation episodes")
        require(sum(r["generated_packets"] for r in rows) == sum(r["generated_packets"] for r in old_rows) == 48,"Budget probe packet count differs")
        budget.append({"seed":seed,"fixed_delivered":sum(r["delivered_packets"] for r in rows),
            "linear_delivered":sum(r["delivered_packets"] for r in old_rows),"generated":48,"fixed_physical":rows,
            "old_evaluation":str(previous),"old_evaluation_sha256":digest(previous/"logs/TG_physical_metrics.jsonl"),
            "training":str(train),"evaluation":str(evaluation),"critic_head_optimizer_steps":trained["training_updates"][-1]["critic_head_optimizer_steps"]})
    tests,count = [],0
    for name in args.test_reports:
        report = read(name)
        require(report["success"],"Failed regression report")
        for stage in report["stages"]:
            require(stage["tests"] > 0 and not any(stage.get(k,0) for k in ("failures","errors","skipped")),"Failed/skipped tests")
            count += stage["tests"]
        tests.append({"path":str(Path(name).resolve()),"sha256":digest(name),"report":report})
    require(count == 115,"Expected 115 tests")
    rates = [r["fixed_delivered"]/48. for r in budget]
    receipt = {"success":True,"scope":"LR switch semantics and development tests; no convergence/innovation claim",
        "tests":tests,"test_count":count,"pilot_jobs":72,"resume_new_jobs":0,"source_files":len(manifests["fixed"]["source"]["files_sha256"]),
        "suites":{k:{"path":p,"manifest_sha256":digest(Path(p)/"manifest.json"),"summary":summaries[k]} for k,p in (("fixed",args.fixed_suite),("linear",args.linear_suite))},
        "old_linear_parity":parity,"paired_rollouts":paired,"before_probe":before,"fixed_probe":fixed,"linear_probe":linear,
        "baseline_20_episode_comparison":budget,"fixed_20_mean":statistics.mean(rates),"fixed_20_sample_sd":statistics.stdev(rates),
        "validator_sha256":digest(Path(__file__)),"new_test_sha256":digest(ROOT/"tests/test_legacy_lr_schedule.py")}
    with Path(args.report).open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print("PASS: 115 tests, 72 pilot jobs, 18 old-policy tensor matches, 36 old physical matches, 3 fixed 20-episode probes, 0 resumed jobs")
    print("Receipt: "+str(Path(args.report).resolve()))


if __name__ == "__main__":
    main()
