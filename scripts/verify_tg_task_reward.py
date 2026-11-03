"""Audit task-v2 pilot, physical events, legacy parity and explicit reward controls."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src.buoy_network.ablation_suite import read, digest, summarize, require
from src.buoy_network.wm_runtime import tensor_digest
from src.buoy_network.task_reward_audit import audit_reward_csv
from verify_tg_ablation import equivalent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",required=True)
    parser.add_argument("--old-suite",required=True)
    parser.add_argument("--probe",required=True)
    parser.add_argument("--old-probe",required=True)
    parser.add_argument("--parity-run",required=True)
    parser.add_argument("--diagnostics",required=True)
    parser.add_argument("--test-reports",nargs="+",required=True)
    parser.add_argument("--report",required=True)
    args = parser.parse_args()
    root, oldroot = Path(args.suite).resolve(), Path(args.old_suite).resolve()
    manifest, oldmanifest = read(root/"manifest.json"), read(oldroot/"manifest.json")
    config, spec = manifest["config"], manifest["config"]["reward_contract"]
    require(manifest["state"] == "completed" and len(manifest["jobs"]) == 36, "Incomplete six-group pilot")
    source = manifest["source"]["files_sha256"]
    for name,checksum in source.items():
        require(digest(root/"source_snapshot"/name) == checksum, "Changed source snapshot: "+name)
        require(digest(ROOT/name) == checksum, "Tested source differs from working source: "+name)
    summary = summarize(manifest)  # Includes independent per-slot reward audit for ALL 36 jobs.
    require(equivalent(summary,read(root/"summary.json")), "Summary differs from raw jobs")
    audits = {}
    for job in manifest["jobs"]:
        audits[job["key"]] = audit_reward_csv(Path(job["output"])/"logs",spec)
        report = read(Path(job["output"])/"logs/TG_policy_report.json")
        require(report["profile"]["ppo_updates"] == (2 if job["kind"] == "training" else 0), "Wrong PPO update count")
        for name in ("wm","pm"):
            if report[name] is not None:
                require(report[name]["frozen"], "Auxiliary model changed")
        if report["pm"]:
            require(all(e["boundary_error"] < 1e-7 for e in report["pm"]["episodes"]), "PBRS boundary identity failed")
    # Resume completed artifacts without scheduling new training.
    before = {j["key"]:len(j["attempts"]) for j in manifest["jobs"]}
    summary_sha = digest(root/"summary.json")
    resumed = subprocess.run([config["python"],str(ROOT/"scripts/run_tg_ablation.py"),"--resume",str(root),"--mode","run"],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60,cwd=str(ROOT))
    require(resumed.returncode == 0, "Resume validation failed: "+resumed.stdout)
    require(before == {j["key"]:len(j["attempts"]) for j in read(root/"manifest.json")["jobs"]}
            and digest(root/"summary.json") == summary_sha,"Resume reran jobs or changed summary")
    oldjob = next(j for j in oldmanifest["jobs"] if j["group"] == "A" and j["train_seed"] == 11 and j["kind"] == "training")
    old, current = Path(oldjob["output"]), Path(args.parity_run).resolve()
    load = lambda p: torch.load(str(p/"checkpoints/tg_policy_latest.pt"),map_location="cpu",weights_only=True)
    weights = [load(p) for p in (current,old)]
    parity = {key:tensor_digest(weights[0][key]) == tensor_digest(weights[1][key]) for key in ("actor","critic")}
    for name in ("TG_MAPPO_metrics.csv","TG_physical_metrics.jsonl"):
        parity[name] = (current/"logs"/name).read_bytes() == (old/"logs"/name).read_bytes()
    require(all(parity.values()),"Legacy default changed weights or physical results")
    controls, oldcontrols = read(args.probe)["reports"], read(args.old_probe)["reports"]
    for control in controls:
        require(control["reward_contract"] == spec and control["weights_unchanged"] and control["optimizer_updates"] == 0,"Invalid reward probe")
        previous = next(r for r in oldcontrols if r["control"] == control["control"])
        require(control["physical_episodes"] == previous["physical_episodes"], "Reward wrapper changed control physics")
    wait = next(r for r in controls if r["control"] == "wait")
    geo = next(r for r in controls if r["control"] == "geographic")
    require(all(abs(r+2) < 1e-10 for r in wait["episode_environment_returns"]), "WAIT should cost -2 per full window")
    require(all(r > 0 for r in geo["episode_environment_returns"])
            and sum(r["delivered_packets"] for r in geo["physical_episodes"]) == 48,"Geographic control reward ranking failed")
    require(float(geo["first_delivery_reward_components"]["reward_delivery"]) > 0,"Missing delivery benefit")
    diagnostic = []
    for seed in (11,22,33):
        path = Path(args.diagnostics)/("A"+str(seed))/"report.json"
        report = read(path)
        require(report["weights_unchanged"] and report["optimizer_updates"] == 0
                and report["pilot_metrics_exact_match"],"Diagnostic replay differs")
        diagnostic.append(report)
    tests, count = [], 0
    for name in args.test_reports:
        report = read(name)
        require(report["success"],"Failed regression report")
        for row in report["stages"]:
            require(row["tests"] > 0 and not any(row.get(k,0) for k in ("failures","errors","skipped")),"Regression failure")
            count += row["tests"]
        tests.append({"path":str(Path(name).resolve()),"sha256":digest(name),"report":report})
    negative = []
    for flags,reason in ((["--tg_reward_version","task_v2","--tg_delay_weight","0"],"finite delivery/delay"),
                         (["--tg_reward_version","task_v2","--tg_unfinished_weight","nan"],"finite delivery/delay"),
                         (["--tg_delay_weight","2"],"weights require"),
                         (["--tg_reward_version","task_v2","--export_wm_data","true"],"export requires legacy_v1")):
        command = [config["python"],str(ROOT/"run_algorithm.py"),"--algorithm","tg-mappo",
                   "--num_episodes","1","--episode_length","4","--output_root","results/tg_task_reward_cli_negative"]+flags
        result = subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=45,cwd=str(ROOT))
        require(result.returncode != 0 and reason in result.stdout,"CLI failed to reject invalid reward options: "+result.stdout)
        negative.append({"command":command,"exit_code":result.returncode,"expected_reason":reason})
    receipt = {"success":True,"scope":"task reward contract and short pilot; NOT a solved zero-delivery policy",
        "suite":str(root),"manifest_sha256":digest(root/"manifest.json"),"summary_sha256":summary_sha,
        "reward_contract":spec,"jobs":36,"training_steps":2880,"evaluation_steps":2880,
        "snapshot_files":len(source),"resume_new_jobs":0,"baseline_parity":parity,"parity_run":str(current),"old_baseline":str(old),
        "file_backed_regression_count":count,"test_reports":tests,"cli_rejections":negative,
        "reward_audits":audits,"controls":controls,"diagnostics":diagnostic,"summary":summary,
        "validator_sha256":digest(Path(__file__)),"extra_test_source_sha256":digest(ROOT/"tests/test_legacy_task_reward.py")}
    with Path(args.report).open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print("PASS: 36 jobs, per-slot physical/reward audit, unchanged legacy baseline, 0 resumed jobs, {} file-backed tests, 4 CLI rejections".format(count))
    print("Receipt: "+str(Path(args.report).resolve()))


if __name__ == "__main__":
    main()
