"""Audit the step-5 2x80 CPU runs and write a compact, reproducible receipt.

Run with Python 3.10 (WM unit tests use modern torch.load).
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("src/buoy_network/agent.py", "src/buoy_network/main.py", "src/buoy_network/runner.py",
           "src/buoy_network/world_model.py", "src/buoy_network/wm_runtime.py", "src/buoy_network/policy_artifacts.py",
           "src/legacy_cli.py", "src/onpolicy/algorithms/graph_actor_critic.py",
           "src/onpolicy/algorithms/graph_MAPPOPolicy.py", "src/onpolicy/algorithms/graph_mappo.py",
           "src/onpolicy/utils/graph_buffer.py", "tests/test_legacy_wm_actor.py",
           "scripts/test_tg_wm_actor.py", "scripts/verify_tg_wm_actor_runs.py")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_run(path):
    root = Path(path)
    with (root/"logs"/"TG_MAPPO_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = [{key: float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
    require(len(rows) == 2 and all(math.isfinite(v) for row in rows for v in row.values()), "Expected two finite metric rows")
    environment = load(root/"logs"/"TG_MAPPO_environment.json")
    config = environment["config"]
    require(config["episode_length"] == 80 and config["effective_num_env_steps"] == 160, "Wrong step-5 budget")
    selected = {key: config[key] for key in ("seed", "scenario_seed", "grid_size", "num_agents", "comm_range",
                "num_moving_buoys", "random_channel", "episode_length", "num_episodes", "ppo_epoch",
                "num_mini_batch", "use_wm_actor", "tg_evaluate")}
    receipt = {"run": str(root.relative_to(ROOT)) if root.is_absolute() else str(root),
               "config": selected, "metrics": rows, "policy_report": load(root/"logs"/"TG_policy_report.json")}
    report = receipt["policy_report"]
    if report["mode"] == "training":
        artifact = Path(report["checkpoint"])
        require(hashlib.sha256(artifact.read_bytes()).hexdigest() == report["checkpoint_sha256"], "Policy file changed")
    if report["wm"] is not None:
        artifact = Path(report["wm"]["checkpoint"])
        require(hashlib.sha256(artifact.read_bytes()).hexdigest() == report["wm"]["checkpoint_sha256"], "WM file changed")
        require(report["wm"]["frozen"] and report["wm"]["predictions"] == 160, "Wrong WM freeze/count")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("baseline", "wm", "baseline_eval", "wm_eval", "wm_eval_repeat")
    for name in names:
        parser.add_argument("--"+name.replace('_', '-'), required=True)
    parser.add_argument("--regression-report", required=True)
    parser.add_argument("--integration-report", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    runs = {name: inspect_run(getattr(args, name)) for name in names}
    baseline, wm = runs["baseline"], runs["wm"]
    require(not baseline["config"]["use_wm_actor"] and wm["config"]["use_wm_actor"], "WM switch mismatch")
    expected = [{"episode": 0.0, "delivery_rate": 0.375, "average_hops": 54.083333333333336, "average_delay": 29.285714285714285},
                {"episode": 1.0, "delivery_rate": 0.25, "average_hops": 61.916666666666664, "average_delay": 31.75}]
    require(baseline["metrics"] == expected, "WM-off historical baseline regressed")
    updates = wm["policy_report"]["wm_ppo_updates"]
    require(len(updates) == 2, "Wrong number of WM PPO updates")
    for row in updates:
        require(math.isfinite(row["ratio_max_error"]) and row["ratio_max_error"] <= 1e-4
                and all(row[key] for key in ("actor_changed", "critic_changed", "fusion_changed", "wm_frozen")),
                "WM freeze/update/replay check failed")
    for name, parent in (("baseline_eval", baseline), ("wm_eval", wm), ("wm_eval_repeat", wm)):
        run = runs[name]
        report = run["policy_report"]
        require(run["config"]["seed"] == run["config"]["scenario_seed"] == 303, "Evaluation seed mismatch")
        require(report["mode"] == "frozen_greedy_evaluation" and report["optimizer_updates"] == 0
                and report["actor_unchanged"] and report["critic_unchanged"], "Evaluation updated weights")
        require(report["checkpoint_sha256"] == parent["policy_report"]["checkpoint_sha256"], "Evaluation loaded wrong policy")
        if parent["config"]["use_wm_actor"]:
            require(report["wm"]["state_sha256"] == parent["policy_report"]["wm"]["state_sha256"], "WM state changed")
    require(runs["wm_eval"]["metrics"] == runs["wm_eval_repeat"]["metrics"], "Same-seed evaluation differs")
    tests = {"legacy": load(args.regression_report), "wm_actor": load(args.integration_report)}
    require(all(report["success"] for report in tests.values()), "Regression failure")
    test_count = 0
    for report in tests.values():
        require({stage["name"] for stage in report["stages"]} == {"functions", "modules", "integration"},
                "Missing test stage")
        for stage in report["stages"]:
            require(stage["tests"] > 0 and not any(stage.get(key, 0) for key in ("failures", "errors", "skipped")),
                    "Failed or skipped tests")
            test_count += stage["tests"]
    completed = subprocess.run([sys.executable, "-m", "unittest", "tests.test_tg_world_model", "-v"],
                               cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, timeout=60)
    require(completed.returncode == 0 and "Ran 6 tests" in completed.stdout, "WM unit tests failed: "+completed.stdout)
    tests["wm_pretrain"] = {"python": sys.executable, "tests": 6, "exit_code": completed.returncode, "log": completed.stdout}
    receipt = {"success": True, "test_count": test_count+6, "scope": "step5 integration, not performance acceptance", "tests": tests,
               "runs": runs, "source_files_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
               "limitations": ["CPU only", "one independent training seed; 2x80 smoke only", "no performance gain demonstrated",
                               "PM disabled", "motion/delivery/drop/uncertainty not used", "no optimizer/RNG resume"]}
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("PASS: {} function/module/integration tests; baseline unchanged; frozen evaluations reproducible".format(test_count+6))
    print("Receipt: "+str(destination.resolve()))


if __name__ == "__main__":
    main()
