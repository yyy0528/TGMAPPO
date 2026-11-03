"""Audit the actual pilot, baseline parity, resume behavior, and regression receipts."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.buoy_network.ablation_suite import read, digest, summarize, require
from src.buoy_network.wm_runtime import tensor_digest


def equivalent(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(equivalent(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equivalent(a,b) for a,b in zip(left,right))
    if (isinstance(left, (int,float)) and isinstance(right, (int,float))
            and (isinstance(left,float) or isinstance(right,float))):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12)
    return left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--test-reports", nargs="+", required=True)
    parser.add_argument("--parity-run", required=True)
    parser.add_argument("--old-baseline", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    root = Path(args.suite).resolve()
    manifest = read(root/"manifest.json")
    require(manifest["state"] == "completed" and manifest["config"]["purpose"] == "pilot", "Pilot incomplete")
    require(len(manifest["jobs"]) == 36 and all(j["state"] == "completed" for j in manifest["jobs"]), "Expected 36 completed pilot jobs")
    for name, checksum in manifest["source"]["files_sha256"].items():
        require(digest(root/"source_snapshot"/name) == checksum, "Snapshot differs: "+name)
    expected = summarize(manifest)
    require(equivalent(expected, read(root/"summary.json")), "Summary differs from raw artifacts")
    before = {job["key"]: len(job["attempts"]) for job in manifest["jobs"]}
    original_summary_sha = digest(root/"summary.json")
    resumed = subprocess.run([manifest["config"]["python"], str(ROOT/"scripts/run_tg_ablation.py"),
        "--resume", str(root), "--mode", "run"], cwd=str(ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, universal_newlines=True, timeout=60)
    require(resumed.returncode == 0, "Resume failed: "+resumed.stdout)
    after = read(root/"manifest.json")
    require(before == {job["key"]: len(job["attempts"]) for job in after["jobs"]}, "Completed resume repeated training")
    require(digest(root/"summary.json") == original_summary_sha, "Pinned-interpreter resume changed summary")
    for seed in manifest["config"]["train_seeds"]:
        rows = {row["group"]:row for row in expected["by_training_seed"] if row["train_seed"] == seed}
        a = rows["A"]["resources"]["actor_trainable"]
        require(rows["B"]["resources"]["actor_trainable"]-a == 2209, "WM scorer capacity differs")
        require(rows["E"]["resources"]["actor_trainable"]-a == 2257, "Expansion capacity differs")
        require(rows["F"]["resources"]["actor_trainable"] == rows["B"]["resources"]["actor_trainable"]
                and not rows["F"]["resources"]["wm_active"], "History control used learned WM")
    current, old = Path(args.parity_run), Path(args.old_baseline)
    require((current/"logs/TG_MAPPO_metrics.csv").read_bytes() == (old/"logs/TG_MAPPO_metrics.csv").read_bytes(), "Legacy metric regression")
    weights = [torch.load(path/"checkpoints/tg_policy_latest.pt", map_location="cpu", weights_only=True) for path in (current,old)]
    parity = {key:tensor_digest(weights[0][key]) == tensor_digest(weights[1][key]) for key in ("actor", "critic")}
    require(all(parity.values()), "Metrics instrumentation changed baseline weights")
    reports, count = [], 0
    for path in args.test_reports:
        report = read(path)
        require(report["success"], "Test report failed")
        for row in report["stages"]:
            require(row["tests"] > 0 and not any(row.get(k,0) for k in ("failures", "errors", "skipped")), "Regression failure")
            count += row["tests"]
        reports.append(report)
    for module, n in (("tests.test_tg_progress_model",5), ("tests.test_tg_world_model",6)):
        completed = subprocess.run([sys.executable,"-m","unittest",module,"-v"], cwd=str(ROOT), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True, timeout=60)
        require(completed.returncode == 0 and "Ran {} tests".format(n) in completed.stdout, "Core tests failed: "+completed.stdout)
        reports.append({"module":module,"tests":n,"exit_code":completed.returncode,"log":completed.stdout})
        count += n
    tools = ("scripts/verify_tg_ablation.py", "scripts/plot_tg_ablation.py", "scripts/test_tg_ablation.py",
             "tests/test_legacy_ablation.py", "tests/test_ablation_suite.py")
    receipt = {"success":True,"scope":"three-seed short pilot, not formal performance acceptance", "test_count":count,
               "tests":reports, "suite":str(root), "manifest_sha256":digest(root/"manifest.json"),
               "summary_sha256":digest(root/"summary.json"), "resume_new_jobs":0,
               "snapshot_files":len(manifest["source"]["files_sha256"]), "baseline_parity":parity,
               "baseline_parity_run":str(current), "old_baseline":str(old), "summary":expected,
               "reporting_tools_sha256":{name:digest(ROOT/name) for name in tools}}
    with Path(args.report).open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print("PASS: {} regression tests, 36 real jobs, baseline weights unchanged, resume scheduled 0 new jobs".format(count))
    print("Receipt: "+str(Path(args.report).resolve()))


if __name__ == "__main__":
    main()
