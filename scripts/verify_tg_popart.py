"""Verify critic optimizer repair, paired pilot, frozen compatibility and budget probes."""
import argparse
import ast
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.buoy_network.ablation_suite import read, digest, summarize, require
from src.buoy_network.task_reward_audit import audit_reward_csv
from verify_tg_ablation import equivalent


def check_training(path, episodes):
    root = Path(path)
    report = read(root/"logs/TG_policy_report.json")
    require(report["training_implementation"] == "tg_ppo_popart_v2", "Wrong critic implementation")
    require(digest(report["checkpoint"]) == report["checkpoint_sha256"],"Policy checkpoint changed")
    require(report["profile"]["ppo_updates"] == episodes and report["profile"]["collect_calls"] == 80*episodes,
            "Unexpected training budget")
    config = read(root/"logs/TG_MAPPO_environment.json")["config"]
    records = report["training_updates"]
    require(len(records) == episodes, "Missing optimizer diagnostics")
    for index,row in enumerate(records,1):
        require(row["popart"] and row["critic_head_in_optimizer"]
                and row["critic_head_optimizer_steps"] == index*config["ppo_epoch"]*config["num_mini_batch"],
                "Critic head is not receiving Adam steps")
        require(all(math.isfinite(row[key]) for key in ("value_loss","policy_loss","dist_entropy","actor_grad_norm","critic_grad_norm","ratio")),
                "Nonfinite PPO diagnostic")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",required=True)
    parser.add_argument("--old-suite",required=True)
    parser.add_argument("--before-probe",required=True)
    parser.add_argument("--after-probe",required=True)
    parser.add_argument("--compat-run",required=True)
    parser.add_argument("--diagnostic",required=True)
    parser.add_argument("--budget-root",required=True)
    parser.add_argument("--test-reports",nargs="+",required=True)
    parser.add_argument("--report",required=True)
    args = parser.parse_args()
    root = Path(args.suite).resolve()
    manifest,old = read(root/"manifest.json"),read(Path(args.old_suite)/"manifest.json")
    require(manifest["state"] == "completed" and len(manifest["jobs"]) == 36,"Pilot incomplete")
    formatting = {}
    for name,checksum in manifest["source"]["files_sha256"].items():
        snapshot,current = root/"source_snapshot"/name, ROOT/name
        require(digest(snapshot) == checksum,"Snapshot changed: "+name)
        if digest(current) != checksum:
            require(name.replace("\\","/") == "src/onpolicy/algorithms/graph_mappo.py"
                    and ast.dump(ast.parse(snapshot.read_text(encoding="utf-8"))) == ast.dump(ast.parse(current.read_text(encoding="utf-8"))),
                    "Untested working-code change: "+name)
            formatting[name] = {"snapshot_sha256":checksum,"working_sha256":digest(current),"ast_identical":True}
    summary = summarize(manifest)
    require(equivalent(summary,read(root/"summary.json")),"Summary differs from raw results")
    before_attempts = {j["key"]:len(j["attempts"]) for j in manifest["jobs"]}
    summary_sha = digest(root/"summary.json")
    result = subprocess.run([manifest["config"]["python"],str(ROOT/"scripts/run_tg_ablation.py"),"--resume",str(root),"--mode","run"],
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60,cwd=str(ROOT))
    require(result.returncode == 0 and before_attempts == {j["key"]:len(j["attempts"]) for j in read(root/"manifest.json")["jobs"]}
            and digest(root/"summary.json") == summary_sha,"Completed suite resume changed artifacts or reran jobs")
    heads = {}
    for job in manifest["jobs"]:
        report = read(Path(job["output"])/"logs/TG_policy_report.json")
        if job["kind"] == "training":
            report = check_training(job["output"],2)
            heads[job["key"]] = [r["critic_head_optimizer_steps"] for r in report["training_updates"]]
        else:
            require(report["source_training_implementation"] == "tg_ppo_popart_v2", "Evaluated wrong training implementation")
        if report["pm"]:
            require(all(e["boundary_error"] < 1e-7 for e in report["pm"]["episodes"]), "PM boundary identity failed")
    a,b = read(args.before_probe),read(args.after_probe)
    require(not a["head_parameters_in_optimizer"] and not a["head_changed_by_optimizer"]
            and a["denormalized_output_max_change"] > 1.,"Original bug not reproduced")
    require(b["parameter_identity_preserved"] and b["head_parameters_in_optimizer"] and b["head_changed_by_optimizer"]
            and b["denormalized_output_max_change"] < 1e-4
            and not any(b["normalization_parameters_require_grad"].values()),"PopArt repair probe failed")
    compat = Path(args.compat_run)
    reference = next(j for j in old["jobs"] if j["group"] == "A" and j["train_seed"] == 22 and j["kind"] == "evaluation")
    old_metrics = Path(reference["output"])/"logs/TG_physical_metrics.jsonl"
    require((compat/"logs/TG_physical_metrics.jsonl").read_bytes() == old_metrics.read_bytes(),"Old policy frozen physics changed")
    compatible = read(compat/"logs/TG_policy_report.json")
    require(compatible["source_training_implementation"] == "legacy_popart_v1" and compatible["actor_unchanged"]
            and compatible["critic_unchanged"] and compatible["optimizer_updates"] == 0,"Old checkpoint compatibility failed")
    diagnostic = read(args.diagnostic)
    require(diagnostic["pilot_metrics_exact_match"] and diagnostic["weights_unchanged"] and diagnostic["optimizer_updates"] == 0,
            "Action diagnosis differs from pilot")
    budget = []
    for seed in (11,22,33):
        train = Path(args.budget_root)/"training/runs"/("A_seed_{}_e20".format(seed))/"moves_5"
        evaluation = Path(args.budget_root)/"evaluation/runs"/("A_seed_{}_e20_scene7001".format(seed))/"moves_5"
        trained = check_training(train,20)
        train_config = read(train/"logs/TG_MAPPO_environment.json")["config"]
        require(train_config["seed"] == train_config["scenario_seed"] == seed
                and train_config["num_episodes"] == 20 and train_config["episode_length"] == 80
                and train_config["ppo_epoch"] == 1 and train_config["num_mini_batch"] == 2
                and not train_config["use_wm_actor"] and not train_config["use_pm_reward"],"Budget probe protocol differs")
        evaluated = read(evaluation/"logs/TG_policy_report.json")
        cfg = read(evaluation/"logs/TG_MAPPO_environment.json")["config"]
        require(cfg["seed"] == cfg["scenario_seed"] == 7001 and cfg["tg_evaluate"]
                and evaluated["checkpoint_sha256"] == trained["checkpoint_sha256"] and evaluated["optimizer_updates"] == 0
                and evaluated["profile"]["collect_calls"] == 160
                and evaluated["actor_unchanged"] and evaluated["critic_unchanged"],"Budget probe evaluation mismatch")
        audit_reward_csv(train/"logs",trained["reward_contract"])
        audit_reward_csv(evaluation/"logs",trained["reward_contract"])
        metrics = [json.loads(line) for line in (evaluation/"logs/TG_physical_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        require(len(metrics) == 2,"Missing budget probe evaluation")
        budget.append({"seed":seed,"training":str(train),"evaluation":str(evaluation),"train_steps":1600,
                       "critic_head_optimizer_steps":trained["training_updates"][-1]["critic_head_optimizer_steps"],
                       "generated":sum(r["generated_packets"] for r in metrics),"delivered":sum(r["delivered_packets"] for r in metrics),
                       "physical_episodes":metrics})
    tests,count = [],0
    for name in args.test_reports:
        report = read(name)
        require(report["success"],"Failed regression")
        for stage in report["stages"]:
            require(stage["tests"] > 0 and not any(stage.get(k,0) for k in ("failures","errors","skipped")),"Regression failure/skip")
            count += stage["tests"]
        tests.append({"path":str(Path(name).resolve()),"sha256":digest(name),"report":report})
    require(count == 107,"Expected 107 tests")
    receipt = {"success":True,"scope":"critic optimizer repair and bounded development experiments; no convergence claim",
        "suite":str(root),"manifest_sha256":digest(root/"manifest.json"),"summary_sha256":summary_sha,
        "tests":tests,"test_count":count,"pilot_jobs":36,"resume_new_jobs":0,"formatting_only_changes":formatting,
        "before_probe":a,"after_probe":b,"critic_optimizer_step_records":heads,"old_policy_frozen_compatibility":compatible,
        "diagnostic":diagnostic,"summary":summary,"baseline_20_episode_probes":budget,
        "budget_probe_scope":"baseline only; original per-total-episode learning-rate schedule retained; not WM/PM ablation",
        "validator_sha256":digest(Path(__file__)),"new_test_source_sha256":digest(ROOT/"tests/test_legacy_popart.py")}
    with Path(args.report).open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print("PASS: 107 tests, 36 pilot jobs, 3 baseline budget probes, actual critic-head Adam steps, frozen old-policy compatibility, 0 resumed jobs")
    print("Receipt: "+str(Path(args.report).resolve()))


if __name__ == "__main__":
    main()
